%matplotlib inline

import torch

import torchvision

import numpy as np

from PIL import Image

import pandas as pd

from tqdm import tqdm

from sklearn.preprocessing import LabelBinarizer

from sklearn.model_selection import ShuffleSplit

import torchvision.models.inception

import matplotlib.pyplot as plt
# hyperparameter



# training

num_epochs = 20

batch_size = 32

num_workers = 6

lr = 0.001



# data sources

sample_submission = '../input/aptos2019-blindness-detection/sample_submission.csv'

root = '../input/aptos2019-blindness-detection/test_images/'

training_file = '../input/aptos2019-blindness-detection/train.csv'

trainroot = '../input/aptos2019-blindness-detection/train_images/'

pretrained = '../input/torchvision-inception-v3-imagenet-pretrained/inception_v3_google-1a9a5a14.pth'

test_size = 0.2



# data preprocessing from imagenet

mean = [0.485, 0.456, 0.406]

std = [0.229, 0.224, 0.225]



# device checker, use GPU if available

device = 'cuda' if torch.cuda.is_available() else 'cpu'

print("Using device", device)



# fixing random seed (for reproducibility)

seed = 555

torch.manual_seed(seed)

torch.backends.cudnn.deterministic = True

torch.backends.cudnn.benchmark = False

np.random.seed(seed)
# Loading the pretrained inception v3

net = torchvision.models.inception_v3()

ckpt = torch.load(pretrained, map_location='cpu')

net.load_state_dict(ckpt)



# as we only have 5 output classes (and want to use pretrained models)

# we need to replace the final layers by new layers which have only 5

# ourput channels. Inception v3 uses AuxLogits, a learning helper, during

# training, so we need to adjust this layer, too.

net.fc = torch.nn.Linear(in_features=2048, out_features=5)

net.AuxLogits = torchvision.models.inception.InceptionAux(in_channels=768, num_classes=5)

_ = net.to(device)
# Adam and Binary Cross Entropy are pretty standard for multi-class classification

optim = torch.optim.Adam(lr=lr, params=net.parameters())

crit = torch.nn.BCEWithLogitsLoss()
# simple preprocessing with resizing and cropping to 299x299

# followed by normalization (formally correct actually standardization)

# given mean and std above

transform = torchvision.transforms.Compose([

    torchvision.transforms.Resize(299),

    torchvision.transforms.CenterCrop(299),

    torchvision.transforms.ToTensor(),

    torchvision.transforms.Normalize(mean=mean, std=std)

])
# simple dataset class which takes the csv filename, the root dir of the images, and the

# transformation above and returns the transformed image and binarized label as tensors

class SimpleDataset():

    def __init__(self, data, root, transform):

        self.files = list(root + data['id_code'] + '.png')

        

        # LabelBinarizer takes numerical labels and returns a one-hot label

        binarizer = LabelBinarizer()

        self.targets = binarizer.fit_transform(data['diagnosis'].values)

        self.transform = transform



    def __len__(self):

        return len(self.files)



    def __getitem__(self, idx):

        img = Image.open(self.files[idx])

        x = self.transform(img)

        y = torch.tensor(self.targets[idx,:]).float()

        return x, y
data = pd.read_csv(training_file)

ssplit = ShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)



train_index, test_index = next(ssplit.split(data['id_code']))



dataset = SimpleDataset(data.iloc[train_index], trainroot, transform)

validationset = SimpleDataset(data.iloc[test_index], trainroot, transform)
train_loss = []

validation_loss = []

for ep in tqdm(range(num_epochs), position=0):

    

    # Training

    net.train()

    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    total_loss = 0

    for x, y in loader:

        x = x.to(device)

        y = y.to(device)

        pred = net(x)

        # as we use auxLogits (default True for inception) we get 2 outputs and need to calculat

        # the loss of both outputs

        loss = crit(pred[0], y) + crit(pred[1], y)

        total_loss += loss

        optim.zero_grad()

        loss.backward()

        optim.step()



    total_loss /= len(dataset) # average loss per image

    total_loss /= 2 # adjustment for summing aux loss and normal loss

    train_loss.append(total_loss)

    

    # Validation

    net.eval()

    loader = torch.utils.data.DataLoader(validationset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    total_loss = 0

    with torch.no_grad():

        for x, y in loader:

            x = x.to(device)

            y = y.to(device)

            pred = net(x)

            loss = crit(pred, y)

            total_loss += loss

        

        # this gives us the average loss per image

        total_loss /= len(validationset)

        validation_loss.append(total_loss)

        tqdm.write('Loss after epoch {:d}: train {:.4f}, test {:.4f}'.format(ep, train_loss[-1], validation_loss[-1]))
plt.plot(train_loss, label='train loss')

plt.plot(validation_loss, label='validation loss')

plt.xlabel('epochs')

plt.ylabel('loss/image [logits]')

plt.title('Training and validation loss')

plt.legend()
# Evaluation

submit = pd.read_csv(sample_submission)

net.eval()



with torch.no_grad():

    for name in tqdm(submit['id_code']):

        img = Image.open(root+name+'.png')

        x = transform(img).to(device).unsqueeze(0)

        y = net(x).cpu().numpy()

        diag = int(np.argmax(y[:5]))

        submit.loc[submit['id_code']==name, 'diagnosis'] = diag
submit.to_csv('submission.csv', index=False)
submit.head()