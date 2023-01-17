!pip install pyro-ppl --quiet
import argparse
import copy
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import MNIST
from torchvision.transforms import Compose, functional
from torchvision.utils import make_grid
from tqdm.auto import trange, tqdm

import pyro
import pyro.distributions as dist
from pyro.infer import SVI, Predictive, Trace_ELBO
class CVAEMNIST(Dataset):
    def __init__(self, root, train=True, transform=None, download=False):
        self.original = MNIST(root, train=train, download=download)
        self.transform = transform

    def __len__(self):
        return len(self.original)

    def __getitem__(self, item):
        image, digit = self.original[item]
        sample = {'original': image, 'digit': digit}
        if self.transform:
            sample = self.transform(sample)

        return sample


class ToTensor:
    def __call__(self, sample):
        sample['original'] = functional.to_tensor(sample['original'])
        sample['digit'] = torch.as_tensor(np.asarray(sample['digit']),
                                          dtype=torch.int64)
        return sample


class MaskImages:
    """This torchvision image transformation prepares the MNIST digits to be
    used in the tutorial. Depending on the number of quadrants to be used as
    inputs (1, 2, or 3), the transformation masks the remaining (3, 2, 1)
    quadrant(s) setting their pixels with -1. Additionally, the transformation
    adds the target output in the sample dict as the complementary of the input
    """
    def __init__(self, num_quadrant_inputs, mask_with=-1):
        if num_quadrant_inputs <= 0 or num_quadrant_inputs >= 4:
            raise ValueError('Number of quadrants as inputs must be 1, 2 or 3')
        self.num = num_quadrant_inputs
        self.mask_with = mask_with

    def __call__(self, sample):
        tensor = sample['original'].squeeze()
        out = tensor.detach().clone()
        h, w = tensor.shape

        # removes the bottom left quadrant from the target output
        out[h // 2:, :w // 2] = self.mask_with
        # if num of quadrants to be used as input is 2,
        # also removes the top left quadrant from the target output
        if self.num == 2:
            out[:, :w // 2] = self.mask_with
        # if num of quadrants to be used as input is 3,
        # also removes the top right quadrant from the target output
        if self.num == 3:
            out[:h // 2, :] = self.mask_with

        # now, sets the input as complementary
        inp = tensor.clone()
        inp[out != -1] = self.mask_with

        sample['input'] = inp
        sample['output'] = out
        return sample


def get_data(num_quadrant_inputs, batch_size):
    transforms = Compose([
        ToTensor(),
        MaskImages(num_quadrant_inputs=num_quadrant_inputs)
    ])
    datasets, dataloaders, dataset_sizes = {}, {}, {}
    for mode in ['train', 'val']:
        datasets[mode] = CVAEMNIST(
            '../data',
            download=True,
            transform=transforms,
            train=mode == 'train'
        )
        dataloaders[mode] = DataLoader(
            datasets[mode],
            batch_size=batch_size,
            shuffle=mode == 'train',
            num_workers=0
        )
        dataset_sizes[mode] = len(datasets[mode])

    return datasets, dataloaders, dataset_sizes
class BaselineNet(nn.Module):
    def __init__(self, hidden_1, hidden_2):
        super().__init__()
        self.fc1 = nn.Linear(784, hidden_1)
        self.fc2 = nn.Linear(hidden_1, hidden_2)
        self.fc3 = nn.Linear(hidden_2, 784)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.view(-1, 784)
        hidden = self.relu(self.fc1(x))
        hidden = self.relu(self.fc2(hidden))
        y = torch.sigmoid(self.fc3(hidden))
        return y
class MaskedBCELoss(nn.Module):
    def __init__(self, masked_with=-1):
        super().__init__()
        self.masked_with = masked_with

    def forward(self, input, target):
        target = target.view(input.shape)
        loss = F.binary_cross_entropy(input, target, reduction='none')
        loss[target == self.masked_with] = 0
        return loss.sum()
def baseline_train(device, dataloaders, dataset_sizes, learning_rate, num_epochs,
                   early_stop_patience, model_path):

    # Train baseline
    baseline_net = BaselineNet(500, 500)
    baseline_net.to(device)
    optimizer = torch.optim.Adam(baseline_net.parameters(), lr=learning_rate)
    criterion = MaskedBCELoss()
    best_loss = np.inf
    early_stop_count = 0

    bar = trange(num_epochs)
    for epoch in bar:
        for phase in ['train', 'val']:
            if phase == 'train':
                baseline_net.train()
            else:
                baseline_net.eval()

            running_loss = 0.0
            bar.set_description('NN Epoch {} {}'.format(epoch, phase).ljust(20))

            for i, batch in enumerate(dataloaders[phase]):
                inputs = batch['input'].to(device)
                outputs = batch['output'].to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    preds = baseline_net(inputs)
                    loss = criterion(preds, outputs)
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item()
                    

            epoch_loss = running_loss / dataset_sizes[phase]
            bar.set_postfix(loss='{:.2f}'.format(epoch_loss),
                            early_stop_count=early_stop_count)
            # deep copy the model
            if phase == 'val':
                if epoch_loss < best_loss:
                    best_loss = epoch_loss
                    best_model_wts = copy.deepcopy(baseline_net.state_dict())
                    early_stop_count = 0
                else:
                    early_stop_count += 1

        if early_stop_count >= early_stop_patience:
            break

    baseline_net.load_state_dict(best_model_wts)
    baseline_net.eval()

    # Save model weights
    Path(model_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(baseline_net.state_dict(), model_path)

    return baseline_net
class Encoder(nn.Module):
    def __init__(self, z_dim, hidden_1, hidden_2):
        super().__init__()
        self.fc1 = nn.Linear(784, hidden_1)
        self.fc2 = nn.Linear(hidden_1, hidden_2)
        self.fc31 = nn.Linear(hidden_2, z_dim)
        self.fc32 = nn.Linear(hidden_2, z_dim)
        self.relu = nn.ReLU()

    def forward(self, x, y):
        # put x and y together in the same image for simplification
        xc = x.clone()
        xc[x == -1] = y[x == -1]
        xc = xc.view(-1, 784)
        # then compute the hidden units
        hidden = self.relu(self.fc1(xc))
        hidden = self.relu(self.fc2(hidden))
        # then return a mean vector and a (positive) square root covariance
        # each of size batch_size x z_dim
        z_loc = self.fc31(hidden)
        z_scale = torch.exp(self.fc32(hidden))
        return z_loc, z_scale


class Decoder(nn.Module):
    def __init__(self, z_dim, hidden_1, hidden_2):
        super().__init__()
        self.fc1 = nn.Linear(z_dim, hidden_1)
        self.fc2 = nn.Linear(hidden_1, hidden_2)
        self.fc3 = nn.Linear(hidden_2, 784)
        self.relu = nn.ReLU()

    def forward(self, z):
        y = self.relu(self.fc1(z))
        y = self.relu(self.fc2(y))
        y = torch.sigmoid(self.fc3(y))
        return y


class CVAE(nn.Module):
    def __init__(self, z_dim, hidden_1, hidden_2, pre_trained_baseline_net):
        super().__init__()
        # The CVAE is composed of multiple MLPs, such as recognition network
        # qφ(z|x, y), (conditional) prior network pθ(z|x), and generation
        # network pθ(y|x, z). Also, CVAE is built on top of the NN: not only
        # the direct input x, but also the initial guess y_hat made by the NN
        # are fed into the prior network.
        self.baseline_net = pre_trained_baseline_net
        self.prior_net = Encoder(z_dim, hidden_1, hidden_2)
        self.generation_net = Decoder(z_dim, hidden_1, hidden_2)
        self.recognition_net = Encoder(z_dim, hidden_1, hidden_2)

    def model(self, xs, ys=None):
        # register this pytorch module and all of its sub-modules with pyro
        pyro.module("generation_net", self)
        batch_size = xs.shape[0]
        with pyro.plate("data"):

            # Prior network uses the baseline predictions as initial guess.
            # This is the generative process with recurrent connection
            with torch.no_grad():
                # this ensures the training process does not change the
                # baseline network
                y_hat = self.baseline_net(xs).view(xs.shape)

            # sample the handwriting style from the prior distribution, which is
            # modulated by the input xs.
            prior_loc, prior_scale = self.prior_net(xs, y_hat)
            zs = pyro.sample('z', dist.Normal(prior_loc, prior_scale).to_event(1))

            # the output y is generated from the distribution pθ(y|x, z)
            loc = self.generation_net(zs)

            if ys is not None:
                # In training, we will only sample in the masked image
                mask_loc = loc[(xs == -1).view(-1, 784)].view(batch_size, -1)
                mask_ys = ys[xs == -1].view(batch_size, -1)
                pyro.sample('y', dist.Bernoulli(mask_loc).to_event(1), obs=mask_ys)
            else:
                # In testing, no need to sample: the output is already a
                # probability in [0, 1] range, which better represent pixel
                # values considering grayscale. If we sample, we will force
                # each pixel to be  either 0 or 1, killing the grayscale
                pyro.deterministic('y', loc.detach())

            # return the loc so we can visualize it later
            return loc

    def guide(self, xs, ys=None):
        with pyro.plate("data"):
            if ys is None:
                # at inference time, ys is not provided. In that case,
                # the model uses the prior network
                y_hat = self.baseline_net(xs).view(xs.shape)
                loc, scale = self.prior_net(xs, y_hat)
            else:
                # at training time, uses the variational distribution
                # q(z|x,y) = normal(loc(x,y),scale(x,y))
                loc, scale = self.recognition_net(xs, ys)

            pyro.sample("z", dist.Normal(loc, scale).to_event(1))

    def save(self, model_path):
        torch.save({
            'prior': self.prior_net.state_dict(),
            'generation': self.generation_net.state_dict(),
            'recognition': self.recognition_net.state_dict()
        }, model_path)

    def load(self, model_path, map_location=None):
        net_weights = torch.load(model_path, map_location=map_location)
        self.prior_net.load_state_dict(net_weights['prior'])
        self.generation_net.load_state_dict(net_weights['generation'])
        self.recognition_net.load_state_dict(net_weights['recognition'])
        self.prior_net.eval()
        self.generation_net.eval()
        self.recognition_net.eval()
def cvae_train(device, dataloaders, dataset_sizes, learning_rate, num_epochs,
               early_stop_patience, model_path, pre_trained_baseline_net):

    # clear param store
    pyro.clear_param_store()

    cvae_net = CVAE(200, 500, 500, pre_trained_baseline_net)
    cvae_net.to(device)
    optimizer = pyro.optim.Adam({"lr": learning_rate})
    svi = SVI(cvae_net.model, cvae_net.guide, optimizer, loss=Trace_ELBO())

    best_loss = np.inf
    early_stop_count = 0
    Path(model_path).parent.mkdir(parents=True, exist_ok=True)

    bar = trange(num_epochs)
    for epoch in bar:
        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            running_loss = 0.0
            bar.set_description('CVAE Epoch {} {}'.format(epoch, phase).ljust(20))

            for i, batch in enumerate(dataloaders[phase]):
                inputs = batch['input'].to(device)
                outputs = batch['output'].to(device)

                if phase == 'train':
                    loss = svi.step(inputs, outputs)
                else:
                    loss = svi.evaluate_loss(inputs, outputs)

                # statistics
                running_loss += loss

            epoch_loss = running_loss / dataset_sizes[phase]
            bar.set_postfix(loss='{:.2f}'.format(epoch_loss),
                            early_stop_count=early_stop_count)
            
            # deep copy the model
            if phase == 'val':
                if epoch_loss < best_loss:
                    best_loss = epoch_loss
                    cvae_net.save(model_path)
                    early_stop_count = 0
                else:
                    early_stop_count += 1

        if early_stop_count >= early_stop_patience:
            break

    # Save model weights
    cvae_net.load(model_path)
    return cvae_net
def imshow(inp, image_path=None):
    inp = inp.cpu().numpy().transpose((1, 2, 0))
    space = np.ones((inp.shape[0], 50, inp.shape[2]))
    inp = np.concatenate([space, inp], axis=1)

    ax = plt.axes(frameon=False, xticks=[], yticks=[])
    ax.text(0, 23, 'Inputs:')
    ax.text(0, 23 + 28 + 3, 'Truth:')
    ax.text(0, 23 + (28 + 3) * 2, 'NN:')
    ax.text(0, 23 + (28 + 3) * 3, 'CVAE:')
    ax.imshow(inp)

    if image_path is not None:
        Path(image_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(image_path, bbox_inches='tight', pad_inches=0.1)
    else:
        plt.show()

    plt.clf()


def visualize(device, num_quadrant_inputs, pre_trained_baseline,
              pre_trained_cvae, num_images, num_samples, image_path=None):

    # Load sample random data
    datasets, _, dataset_sizes = get_data(
        num_quadrant_inputs=num_quadrant_inputs,
        batch_size=num_images
    )
    dataloader = DataLoader(datasets['val'], batch_size=num_images, shuffle=True)

    batch = next(iter(dataloader))
    inputs = batch['input'].to(device)
    outputs = batch['output'].to(device)
    originals = batch['original'].to(device)

    # Make predictions
    with torch.no_grad():
        baseline_preds = pre_trained_baseline(inputs).view(outputs.shape)

    predictive = Predictive(pre_trained_cvae.model,
                            guide=pre_trained_cvae.guide,
                            num_samples=num_samples)
    cvae_preds = predictive(inputs)['y'].view(num_samples, num_images, 28, 28)

    # Predictions are only made in the pixels not masked. This completes
    # the input quadrant with the prediction for the missing quadrants, for
    # visualization purpose
    baseline_preds[outputs == -1] = inputs[outputs == -1]
    for i in range(cvae_preds.shape[0]):
        cvae_preds[i][outputs == -1] = inputs[outputs == -1]

    # adjust tensor sizes
    inputs = inputs.unsqueeze(1)
    inputs[inputs == -1] = 1
    baseline_preds = baseline_preds.unsqueeze(1)
    cvae_preds = cvae_preds.view(-1, 28, 28).unsqueeze(1)

    # make grids
    inputs_tensor = make_grid(inputs, nrow=num_images, padding=0)
    originals_tensor = make_grid(originals, nrow=num_images, padding=0)
    separator_tensor = torch.ones((3, 5, originals_tensor.shape[-1])).to(device)
    baseline_tensor = make_grid(baseline_preds, nrow=num_images, padding=0)
    cvae_tensor = make_grid(cvae_preds, nrow=num_images, padding=0)

    # add vertical and horizontal lines
    for tensor in [originals_tensor, baseline_tensor, cvae_tensor]:
        for i in range(num_images - 1):
            tensor[:, :, (i + 1) * 28] = 0.3

    for i in range(num_samples - 1):
        cvae_tensor[:, (i + 1) * 28, :] = 0.3

    # concatenate all tensors
    grid_tensor = torch.cat([inputs_tensor, separator_tensor, originals_tensor,
                             separator_tensor, baseline_tensor,
                             separator_tensor, cvae_tensor], dim=1)
    # plot tensors
    imshow(grid_tensor, image_path=image_path)


def generate_table(device, num_quadrant_inputs, pre_trained_baseline,
                   pre_trained_cvae, num_particles, col_name):

    # Load sample random data
    datasets, dataloaders, dataset_sizes = get_data(
        num_quadrant_inputs=num_quadrant_inputs,
        batch_size=32
    )

    # Load sample data
    criterion = MaskedBCELoss()
    loss_fn = Trace_ELBO(num_particles=num_particles).differentiable_loss

    baseline_cll = 0.0
    cvae_mc_cll = 0.0
    num_preds = 0

    df = pd.DataFrame(index=['NN (baseline)', 'CVAE (Monte Carlo)'],
                      columns=[col_name])

    # Iterate over data.
    bar = tqdm(dataloaders['val'], desc='Generating predictions'.ljust(20))
    for batch in bar:
        inputs = batch['input'].to(device)
        outputs = batch['output'].to(device)
        num_preds += 1

        # Compute negative log likelihood for the baseline NN
        with torch.no_grad():
            preds = pre_trained_baseline(inputs)
        baseline_cll += criterion(preds, outputs).item() / inputs.size(0)

        # Compute the negative conditional log likelihood for the CVAE
        cvae_mc_cll += loss_fn(pre_trained_cvae.model,
                               pre_trained_cvae.guide,
                               inputs, outputs).detach().item() / inputs.size(0)

    df.iloc[0, 0] = baseline_cll / num_preds
    df.iloc[1, 0] = cvae_mc_cll / num_preds
    return df
all_num_quadrant_inp = [1, 2, 3] # num of quadrants to use as inputs
num_epochs = 100                 # number of training epochs
early_stop_patience = 3          # early stop patience
learning_rate = 1.0e-3           # learning rate
cuda = True                      # whether to use cuda
num_images = 10                  # number of images to visualize
num_samples = 10                 # number of samples to visualize per image
num_particles = 10           # n of particles to estimate logpθ(y|x,z) in ELBO
device = torch.device("cuda:0" if torch.cuda.is_available() and cuda
                      else "cpu")
device
results = []
columns = []

for num_quadrant_inputs in all_num_quadrant_inp:
    # Dataset
    datasets, dataloaders, dataset_sizes = get_data(
        num_quadrant_inputs=num_quadrant_inputs,
        batch_size=128
    )
    
    # adds an s in case of plural quadrants
    maybes = 's' if num_quadrant_inputs > 1 else ''

    print('Training with {} quadrant{} as input...'
          .format(num_quadrant_inputs, maybes))

    # Train baseline
    baseline_net = baseline_train(
        device=device,
        dataloaders=dataloaders,
        dataset_sizes=dataset_sizes,
        learning_rate=learning_rate,
        num_epochs=num_epochs,
        early_stop_patience=early_stop_patience,
        model_path='baseline_net_q{}.pth'.format(num_quadrant_inputs)
    )

    # Train CVAE
    cvae_net = cvae_train(
        device=device,
        dataloaders=dataloaders,
        dataset_sizes=dataset_sizes,
        learning_rate=learning_rate,
        num_epochs=num_epochs,
        early_stop_patience=early_stop_patience,
        model_path='cvae_net_q{}.pth'.format(num_quadrant_inputs),
        pre_trained_baseline_net=baseline_net
    )

    # Visualize conditional predictions
    visualize(
        device=device,
        num_quadrant_inputs=num_quadrant_inputs,
        pre_trained_baseline=baseline_net,
        pre_trained_cvae=cvae_net,
        num_images=num_images,
        num_samples=num_samples,
        image_path='cvae_plot_q{}.png'.format(num_quadrant_inputs)
    )

    # Retrieve conditional log likelihood
    df = generate_table(
        device=device,
        num_quadrant_inputs=num_quadrant_inputs,
        pre_trained_baseline=baseline_net,
        pre_trained_cvae=cvae_net,
        num_particles=num_particles,
        col_name='{} quadrant{}'.format(num_quadrant_inputs, maybes)
    )
    results.append(df)
    columns.append('{} quadrant{}'.format(num_quadrant_inputs, maybes))

results = pd.concat(results, axis=1, ignore_index=True)
results.columns = columns
results.loc['Performance gap', :] = results.iloc[0, :] - results.iloc[1, :]
results.to_csv('results.csv')
