import numpy as np 

import pandas as pd



from PIL import Image



import torch

import torch.nn as nn

import torch.utils.data as D

import torch.nn.functional as F



import torchvision

from torchvision import transforms as T



import tqdm



import warnings

warnings.filterwarnings('ignore')
path_data = '../input/'

device = 'cuda'

batch_size = 16
class ImagesDS(D.Dataset):

    def __init__(self, csv_file, img_dir, mode='train', site=1, channels=[1,2,3,4,5,6]):

        

        df = pd.read_csv(csv_file)

        self.records = df.to_records(index=False)

        self.channels = channels

        self.site = site

        self.mode = mode

        self.img_dir = img_dir

        self.len = df.shape[0]

        

    @staticmethod

    def _load_img_as_tensor(file_name):

        with Image.open(file_name) as img:

            return T.ToTensor()(img)



    def _get_img_path(self, index, channel):

        experiment, well, plate = self.records[index].experiment, self.records[index].well, self.records[index].plate

        return '/'.join([self.img_dir,self.mode,experiment,f'Plate{plate}',f'{well}_s{self.site}_w{channel}.png'])

        

    def __getitem__(self, index):

        paths = [self._get_img_path(index, ch) for ch in self.channels]

        img = torch.cat([self._load_img_as_tensor(img_path) for img_path in paths])

        

        if self.mode == 'train':

            return img, self.records[index].sirna

        else:

            return img, self.records[index].id_code



    def __len__(self):

        """

        Total number of samples in the dataset

        """

        return self.len
ds = ImagesDS(path_data+'/train.csv', path_data)

ds_test = ImagesDS(path_data+'/test.csv', path_data, mode='test')
class DensNet(nn.Module):

    def __init__(self, num_classes=1000, num_channels=6):

        super().__init__()

        preloaded = torchvision.models.densenet121(pretrained=True)

        self.features = preloaded.features

        self.features.conv0 = nn.Conv2d(num_channels, 64, 7, 2, 3)

        self.classifier = nn.Linear(1024, num_classes, bias=True)

        del preloaded

        

    def forward(self, x):

        features = self.features(x)

        out = F.relu(features, inplace=True)

        out = F.adaptive_avg_pool2d(out, (1, 1)).view(features.size(0), -1)

        out = self.classifier(out)

        return out
classes = 1108

model = DensNet(num_classes=classes)

model.to(device);
loader = D.DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=2)

tloader = D.DataLoader(ds_test, batch_size=batch_size, shuffle=False, num_workers=2)
criterion = nn.BCEWithLogitsLoss()

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
def accuracy(output, target, topk=(1,)):

    """Computes the accuracy over the k top predictions for the specified values of k"""

    with torch.no_grad():

        maxk = max(topk)

        batch_size = target.size(0)



        _, pred = output.topk(maxk, 1, True, True)

        pred = pred.t()

        correct = pred.eq(target.view(1, -1).expand_as(pred))



        res = []

        for k in topk:

            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)

            res.append(correct_k.mul_(100.0 / batch_size).item())

        return np.array(res)
epochs = 10

tlen = len(loader)

for epoch in range(epochs):

    tloss = 0

    acc = np.zeros(1)

    for x, y in loader: 

        x = x.to(device)

        optimizer.zero_grad()

        output = model(x)

        target = torch.zeros_like(output, device=device)

        target[np.arange(x.size(0)), y] = 1

        loss = criterion(output, target)

        loss.backward()

        optimizer.step()

        tloss += loss.item() 

        acc += accuracy(output.cpu(), y)

        del loss, output, y, x, target

    print('Epoch {} -> Train Loss: {:.4f}, ACC: {:.2f}%'.format(epoch+1, tloss/tlen, acc[0]/tlen))
@torch.no_grad()

def prediction(model, loader):

    preds = np.empty(0)

    for x, _ in loader: 

        x = x.to(device)

        output = model(x)

        idx = output.max(dim=-1)[1].cpu().numpy()

        preds = np.append(preds, idx, axis=0)

    return preds



preds = prediction(model, tloader)
submission = pd.read_csv(path_data + '/test.csv')

submission['sirna'] = preds.astype(int)

submission.to_csv('submission.csv', index=False, columns=['id_code','sirna'])