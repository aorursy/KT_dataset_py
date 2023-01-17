import numpy as np

import gzip

import os

import glob2

import copy

import math

import torch

import torch.nn as nn

import torch.nn.functional as F

import torchvision.models as vision_models



from torch.utils.data import Dataset

from torch.utils.data import SubsetRandomSampler, DataLoader
def read_spectrogram(spectrogram_file, chroma=True):

    with gzip.GzipFile(spectrogram_file, 'r') as f:

        spectrograms = np.load(f)

    # spectrograms contains a fused mel spectrogram and chromagram

    # Decompose as follows

    return spectrograms.T





class PaddingTransform(object):

    def __init__(self, max_length, padding_value=0):

        self.max_length = max_length

        self.padding_value = padding_value



    def __call__(self, s):

        if len(s) == self.max_length:

            return s



        if len(s) > self.max_length:

            return s[:self.max_length]



        if len(s) < self.max_length:

            s1 = copy.deepcopy(s)

            pad = np.zeros((self.max_length - s.shape[0], s.shape[1]), dtype=np.float32)

            s1 = np.vstack((s1, pad))

            return s1





class SpectrogramDatasetTrain(Dataset):

    def __init__(self, path, max_length=-1):

        p = os.path.join(path, 'train')

        self.index = os.path.join(path, 'train_labels.txt')

        self.files, self.labels = self.get_files_labels(self.index)

        self.feats = [read_spectrogram(os.path.join(p, f + '.fused.full.npy.gz')) 

                      for f in self.files]

        self.feat_dim = self.feats[0].shape[1]

        self.lengths = [len(i) for i in self.feats]

        self.max_length = max(self.lengths) if max_length <= 0 else max_length

        self.zero_pad_and_stack = PaddingTransform(self.max_length)



    def get_files_labels(self, txt):

        with open(txt, 'r') as fd:

            lines = [l.rstrip().split(',') for l in fd.readlines()[1:]]

        files, labels = [], []

        for l in lines:

            valence, energy, dance = float(l[1]), float(l[2]), float(l[3].strip())

            files.append(l[0])

            labels.append([valence, energy, dance])

        return files, labels



    def __getitem__(self, item):

        l = min(self.lengths[item], self.max_length)

        return self.zero_pad_and_stack(self.feats[item]), self.labels[item]



    def __len__(self):

        return len(self.labels)

    



class SpectrogramDatasetTest(Dataset):

    def __init__(self, max_length=-1):

        self.files = glob2.glob('../input/data/data/multitask_dataset/test/*.gz')

        self.feats = [read_spectrogram(f) for f in self.files]

        self.feat_dim = self.feats[0].shape[1]

        self.lengths = [len(i) for i in self.feats]

        self.max_length = max(self.lengths) if max_length <= 0 else max_length

        self.zero_pad_and_stack = PaddingTransform(self.max_length)



    def __getitem__(self, item):

        l = min(self.lengths[item], self.max_length)

        return self.zero_pad_and_stack(self.feats[item]), self.files[item], l



    def __len__(self):

        return len(self.labels)
def torch_train_val_split(

        dataset, batch_train, batch_eval,

        val_size=.2, shuffle=True, seed=42):

    # Creating data indices for training and validation splits:

    dataset_size = len(dataset)

    indices = list(range(dataset_size))

    val_split = int(np.floor(val_size * dataset_size))

    if shuffle:

        np.random.seed(seed)

        np.random.shuffle(indices)

    train_indices = indices[val_split:]

    val_indices = indices[:val_split]



    # Creating PT data samplers and loaders:

    train_sampler = SubsetRandomSampler(train_indices)

    val_sampler = SubsetRandomSampler(val_indices)



    train_loader = DataLoader(dataset,

                              batch_size=batch_train,

                              sampler=train_sampler)

    val_loader = DataLoader(dataset,

                            batch_size=batch_eval,

                            sampler=val_sampler)

    return train_loader, val_loader
class EarlyStopping(object):

    def __init__(self, patience, mode='min', base=None):

        self.best = base

        self.patience = patience

        self.patience_left = patience

        self.mode = mode

    

    def stop(self, value):

        if self.has_improved(value):

            self.patience_left = self.patience  # reset patience

        else:

            self.patience_left -= 1  # decrease patience

        print("patience left:{}, best({})".format(self.patience_left, self.best))



        # if no more patience left, then stop training

        return self.patience_left <= 0

    

    def has_improved(self, value):

         # init best value

        if self.best is None or math.isnan(self.best):

            self.best = value

            return True



        if (

                self.mode == "min" and value < self.best

                or

                self.mode == "max" and value > self.best

        ):  # the performance of the model has been improved :)

            self.best = value

            return True

        else:

            # no improvement :(

            return False 
def spearman(x, y):

    vx = x - torch.mean(x)

    vy = y - torch.mean(y)

    return torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2)))





def batch_metrics(model, criterion, batch):

        features, labels = batch

        val_gt = labels[0].float().to(device)

        en_gt = labels[1].float().to(device)

        dance_gt = labels[2].float().to(device)

        features = features.float().to(device)

        valence, energy, danceability = model(features)

        loss1 = criterion(valence, val_gt)

        loss2 = criterion(energy, en_gt)

        loss3 = criterion(danceability, dance_gt)

        loss = loss1 + loss2 + loss3

        s1 = spearman(valence, val_gt)

        s2 = spearman(energy, en_gt)

        s3 = spearman(danceability, dance_gt)

        s = (s1 + s2 + s3) / 3.0

        return loss, s





def train_epoch(train_loader, model, criterion, optimizer, device='cuda'):

    model.train()

    running_loss = 0.0

    running_corr = 0.0

    for num_batch, batch in enumerate(train_loader):

        optimizer.zero_grad()

        loss, corr = batch_metrics(model, criterion, batch)

        loss.backward()

        optimizer.step()

        running_corr += corr.item()

        running_loss += loss.item()

    train_loss = running_loss / num_batch

    train_corr = running_corr / num_batch

    return train_loss, train_corr



def val_epoch(val_loader, model, criterion):

    model.eval()

    running_loss = 0.0

    running_corr = 0.0

    with torch.no_grad():

        y_pred = torch.empty(0, dtype=torch.int64)

        y_true = torch.empty(0, dtype=torch.int64)

        for num_batch, batch in enumerate(val_loader):

            loss, corr = batch_metrics(model, criterion, batch)

            running_corr += corr.item()

            running_loss += loss.item()

    valid_loss = running_loss / num_batch

    valid_corr = running_corr / num_batch

    return valid_loss, valid_corr
class PretrainedImagenet(nn.Module):

    def __init__(self, original_model=None):

        super(PretrainedImagenet, self).__init__()

        if original_model is None:

            original_model = vision_models.resnet50(pretrained=True)

        self.features = nn.Sequential(*list(original_model.children())[:-3])

        for i, p in enumerate(self.features.parameters()):

            p.requires_grad = False

        self.pool = nn.AvgPool2d(kernel_size=7, stride=2, padding=0)

        self.val = nn.Linear(77824, 1)

        self.en = nn.Linear(77824, 1)

        self.danc = nn.Linear(77824, 1)



    def forward(self, x):

        out = self.features(x.unsqueeze(1).repeat(1, 3, 1, 1))

        out = self.pool(out)

        out = out.view(x.size(0),  -1)

        val = torch.sigmoid(self.val(out))

        en = torch.sigmoid(self.en(out))

        danc = torch.sigmoid(self.danc(out))

        return val.view(-1), en.view(-1), danc.view(-1)
specs = SpectrogramDatasetTrain('../input/data/data/multitask_dataset/')

train_loader, val_loader = torch_train_val_split(specs, 32 ,32, val_size=.33)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

print('Running on {}'.format(device))



cnn = PretrainedImagenet().to(device)  # PretrainedImagenet(resnet18(pretrained=True)).to(device)

print(cnn)

trainable_params = [p for p in cnn.parameters() if p.requires_grad]

print(trainable_params)
epochs = 60



criterion = torch.nn.MSELoss()

optimizer = torch.optim.Adam(cnn.parameters(), lr=0.001)

early_stopping = EarlyStopping(patience=1)

for epoch in range(epochs):

    # Training

    train_loss, train_corr = train_epoch(train_loader, cnn, criterion, optimizer)

    # Validation

    valid_loss, valid_corr = val_epoch(val_loader, cnn, criterion)

    print('Epoch {}: train loss = {}, valid loss = {}, train corr = {}, valid corr = {}'

          .format(epoch, train_loss, valid_loss, train_corr, valid_corr))

    if early_stopping.stop(valid_loss):

        print('early stopping...')

        break
testdata =  SpectrogramDatasetTest()



cnn.eval()

pred = []

with torch.no_grad():

    for num_batch, batch in enumerate(testdata):

        features, fname, lengths = batch

        fname = fname.split('/')[-1]

        features = torch.as_tensor(features).float().to(device).unsqueeze(0)

        valence, energy, danceability = cnn(features)

        pred.append((fname, valence.item(), energy.item(), danceability.item()))
# Now create a submission from pred.