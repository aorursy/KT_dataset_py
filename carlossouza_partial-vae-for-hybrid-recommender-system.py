!pip install git+https://github.com/ucals/hamiltorch.git
from datetime import date
import hamiltorch
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm
val_ratio = 0.1
batch_size = 100
num_epochs = 400
model_file = f'model-{date.today():}.pth'
learning_rate = 1e-3
class UserDataset(Dataset):
    def __init__(self, transform=None):
        self.transform = transform
        self.dfu = pd.read_csv(
            '../input/movielens-1m/ml-1m/users.dat', 
            delimiter='::', header=None)
        self.dfu.columns = ['UserID', 'Gender', 'Age', 'Occupation', 'Zip-code']
        self.dfr = pd.read_csv(
            '../input/movielens-1m/ml-1m/ratings.dat', 
            delimiter='::', header=None)
        self.dfr.columns = ['UserID', 'MovieID', 'Rating', 'Timestamp']

    def __len__(self):
        return len(self.dfu)

    def __getitem__(self, item):
        user = self.dfu.iloc[item]
        ratings = self.dfr[self.dfr['UserID'] == user['UserID']].\
            drop(columns=['UserID', 'Timestamp'])

        sample = {
            'user_data': user,
            'ratings': ratings,
            'num_ratings': len(ratings)
        }
        if self.transform:
            sample = self.transform(sample)

        return sample
# helper transform 
class ToTensor:
    def __call__(self, sample):
        return {
            'user_id': sample['user_data']['UserID'],
            'ratings': torch.from_numpy(sample['ratings'].values),
            'num_ratings': sample['num_ratings']
        }
# helper function to collate batches with irregular sizes
def collate_fn(batch):
    user_ids = []
    ratings = []
    num_ratings = []
    for sample in batch:
        user_ids.append(sample['user_id'])
        ratings.append(sample['ratings'])
        num_ratings.append(sample['num_ratings'])

    return {
        'user_id': torch.tensor(user_ids),
        'ratings': torch.cat(ratings, dim=0),
        'num_ratings': torch.tensor(num_ratings)
    }
class Model(nn.Module):
    def __init__(self, num_movies=3953, num_latent=10):
        super(Model, self).__init__()
        self.num_movies = num_movies
        self.num_latent = num_latent
        self.emb = nn.Embedding(num_movies, 8)
        self.fc1 = nn.Linear(8, 100)
        self.fc2 = nn.Linear(100, num_latent * 2)
        self.fc3 = nn.Linear(num_latent, 100)
        self.fc4 = nn.Linear(100, num_movies)
        self.bn1 = nn.BatchNorm1d(100)
        self.bn2 = nn.BatchNorm1d(num_latent * 2)
        self.bn3 = nn.BatchNorm1d(100)
        self.bn4 = nn.BatchNorm1d(num_movies)
        self.relu = nn.ReLU()

    def encode(self, ratings, num_ratings):
        emb = self.emb(ratings[:, 0].long())
        x = emb * ratings[:, 1].view(-1, 1)

        stack = []
        i0 = 0
        for i in range(num_ratings.size(0)):
            i1 = i0 + num_ratings[i]
            slc = x[i0:i1]
            assert slc.size(0) == num_ratings[i]
            stack.append(slc.sum(dim=0))
            i0 = i1

        x = torch.stack(stack, dim=0)
        assert x.size() == (num_ratings.size(0), 8)

        x = self.bn1(self.relu(self.fc1(x)))
        x = self.bn2(self.relu(self.fc2(x)))

        mu = x[:, :self.num_latent]
        log_var = x[:, self.num_latent:]
        return mu, log_var

    def reparameterize(self, mu, log_var):
        std = torch.exp(log_var/2)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        x = self.bn3(self.relu(self.fc3(z)))
        x = self.bn4(self.relu(self.fc4(x)))
        return x

    def forward(self, ratings, num_ratings, use_hmc=False, num_samples=10,
                num_steps=20, device=None):
        mu, log_var = self.encode(ratings, num_ratings)

        if not use_hmc:
            z = self.reparameterize(mu, log_var)
            x_hat = self.decode(z)
            return x_hat, mu, log_var

        else:
            # This Hamiltonian Monte Carlo implementation is too slow
            # TODO: figure out a way to accelerate this sampling
            if device is None:
                device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
                
            x_hat = []
            std = torch.exp(log_var/2)
            for i in range(mu.size(0)):
                params_init = torch.zeros(self.num_latent).to(device)
                params_hmc = hamiltorch.sample(
                    log_prob_func=log_prob_func(mu[i], std[i]),
                    params_init=params_init,
                    num_samples=num_samples + 1,
                    step_size=0.3,
                    num_steps_per_sample=num_steps,
                    silent=True
                )
                zs = torch.stack(params_hmc[1:], dim=0)
                x_hats = self.decode(zs).mean(dim=0)
                x_hat.append(x_hats)

            x_hat = torch.stack(x_hat, dim=0)
            assert x_hat.size() == (mu.size(0), self.num_movies)
            return x_hat, mu, log_var


def log_prob_func(mean, stddev):
    def lpf(params):
        return torch.distributions.Normal(mean, stddev).log_prob(params).sum()

    return lpf
# helper functions
def mask(x_hat, x, num_ratings):
    # filter ratings, to calculate loss only in the observations
    stack = []
    i0 = 0
    for i in range(x_hat.size(0)):
        i1 = i0 + num_ratings[i]
        movie_ids = x[i0:i1, 0].long()
        assert movie_ids.size(0) == num_ratings[i]
        i0 = i1

        preds = x_hat[i, movie_ids]
        stack.append(preds)

    x_pred = torch.cat(stack, dim=0)
    x_true = x[:, 1]
    return x_pred, x_true


def custom_loss(x_hat, x, num_ratings, mu, log_var):
    x_pred, x_true = mask(x_hat, x, num_ratings)

    # now, calculate the loss
    criterion = torch.nn.MSELoss()
    mse = criterion(x_pred, x_true)
    kld = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    return mse + kld, mse
# load data and initialize
dataset = UserDataset(transform=ToTensor())

val_size = int(len(dataset) * val_ratio)
train_size = len(dataset) - val_size
trainset, valset = random_split(dataset, [train_size, val_size])
dataloaders = {
    'train': DataLoader(trainset, batch_size=batch_size, shuffle=True,
                        collate_fn=collate_fn),
    'val': DataLoader(valset, batch_size=batch_size, shuffle=False,
                      collate_fn=collate_fn)
}

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = Model().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
best_mse = np.inf
# training loop
for epoch in range(num_epochs):
    # Each epoch has a training and validation phase
    for phase in ['train', 'val']:
        if phase == 'train':
            model.train()  # Set model to training mode
        else:
            model.eval()   # Set model to evaluate mode

        running_loss = 0.0
        running_mse = 0.0
        running_preds = 0
        bar = tqdm(dataloaders[phase])
        bar.set_description(f'Epoch {epoch} {phase}'.ljust(20))

        for batch in bar:
            x = batch['ratings'].float().to(device)
            num_ratings = batch['num_ratings'].to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward
            # track history if only in train
            with torch.set_grad_enabled(phase == 'train'):
                x_hat, mu, log_var = model(x, num_ratings)
                loss, mse = custom_loss(x_hat, x, num_ratings, mu, log_var)

                # backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    optimizer.step()

            running_loss += loss.item() * x.size(0)
            running_mse += mse.item() * x.size(0)
            running_preds += x.size(0)

            bar.set_postfix(loss=f'{running_loss / running_preds:0.6f}',
                            mse=f'{running_mse / running_preds:0.6f}')

        epoch_mse = running_mse / running_preds
        if epoch_mse < best_mse:
            best_mse = epoch_mse
            torch.save(model.state_dict(), model_file)
