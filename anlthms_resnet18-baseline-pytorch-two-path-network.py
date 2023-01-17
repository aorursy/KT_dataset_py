import os

import numpy as np

import pandas as pd

from PIL import Image

from tqdm import tqdm



import torch

import torch.nn as nn

import torch.utils.data as D

from torchvision import models, transforms as T



import warnings

warnings.filterwarnings('ignore')
path_data = '../input'

device = 'cuda'

batch_size = 16

# XXX set this to a higher value

max_epochs = 3

img_size = 384

torch.manual_seed(0)

np.random.seed(0)
class ImagesDS(D.Dataset):

    def __init__(self, df, img_dir, mode='train', site=1, channels=[1,2,3,4,5,6]):

        self.records = df.to_records(index=False)

        self.channels = channels

        self.site = site

        self.mode = mode

        self.img_dir = img_dir

        self.len = df.shape[0]

        train_controls = pd.read_csv(path_data+'/train_controls.csv')

        test_controls = pd.read_csv(path_data+'/test_controls.csv')

        self.controls = pd.concat([train_controls, test_controls])



    @staticmethod

    def _load_img_as_tensor(file_name):

        with Image.open(file_name) as img:

            return T.ToTensor()(img)



    def _get_img_path(self, experiment, well, plate, channel):

        if self.mode == 'train':

            # pick one of the sites randomly

            site = np.random.randint(1, 3)

        else:

            site = self.site

        return '/'.join([self.img_dir, self.mode, experiment,

                        f'Plate{plate}', f'{well}_s{site}_w{channel}.png'])



    def __getitem__(self, index):

        rec = self.records[index]

        experiment, well, plate = rec.experiment, rec.well, rec.plate

        paths = [self._get_img_path(experiment, well, plate, ch) for ch in self.channels]



        df = self.controls

        negs = df[(df.experiment == experiment) & (df.plate == plate) & (df.sirna == 1138)]

        well = negs.iloc[np.random.randint(0, len(negs))].well

        paths.extend([self._get_img_path(experiment, well, plate, ch) for ch in self.channels])



        img = torch.cat([self._load_img_as_tensor(img_path) for img_path in paths])

        tr_img = torch.empty((12, img_size, img_size), dtype=torch.float32)



        if self.mode == 'train':

            # randomly crop

            row, col = np.random.randint(0, 512 - img_size + 1, 2)

            tr_img[:6] = img[:6, row:row + img_size, col:col + img_size]

            # randomly crop the negative control image

            row, col = np.random.randint(0, 512 - img_size + 1, 2)

            tr_img[6:] = img[6:, row:row + img_size, col:col + img_size]

            return tr_img, int(self.records[index].sirna)



        # center crop

        row =  col = (512 - img_size) // 2

        tr_img[:] = img[:, row:row + img_size, col:col + img_size]

        return tr_img, rec.id_code



    def __len__(self):

        return self.len
df = pd.read_csv(path_data+'/train.csv')

in_eval = df.experiment.isin(['HEPG2-07', 'HUVEC-16', 'RPE-07'])

df_train = df[~in_eval]

df_val = df[in_eval]



df_test = pd.read_csv(path_data+'/test.csv')
ds = ImagesDS(df_train, path_data, mode='train')

ds_val = ImagesDS(df_val, path_data, mode='train')

ds_test = ImagesDS(df_test, path_data, mode='test')
num_classes = 1108

num_workers = 4

model = models.resnet18(pretrained=True)

# add a new layer to combine outputs from two paths.

model.head = torch.nn.Linear(model.fc.out_features, num_classes) 



"""

                  ________

                  |      |

      image ----> |resnet|  \

                  ________   \

                              \

                            ________           _______

                            | minus |  ---->  | head  | ---->

                            _________          _______

                               /

                              /

                             /

                  ________

                  |      |

 -ve control ---> |resnet|

                  ________

"""



# let's make our model work with 6 channels

trained_kernel = model.conv1.weight

new_conv = nn.Conv2d(6, 64, kernel_size=7, stride=2, padding=3, bias=False)

with torch.no_grad():

    new_conv.weight[:] = torch.stack([torch.mean(trained_kernel, 1)]*6, dim=1)

model.conv1 = new_conv

model = model.to(device)

train_loader = D.DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)

eval_loader = D.DataLoader(ds_val, batch_size=batch_size, shuffle=True, num_workers=num_workers)

tloader = D.DataLoader(ds_test, batch_size=batch_size, shuffle=False, num_workers=num_workers)
criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(model.parameters(), lr=0.0003)



def train(train_loader, model, criterion, optimizer, epoch):

    model.train()

    if epoch == 0:

        # update only the last two FC layers

        for name, child in model.named_children():

            if (name != 'head') and (name != 'fc'):

                for param in child.parameters():

                    param.requires_grad = False

    elif epoch == 3:

        # enable update on all layers

        for name, child in model.named_children():

            for param in child.parameters():

                param.requires_grad = True



    loss_sum = 0

    for input, target in tqdm(train_loader):

        input1, input2 = input[:, :6].to(device), input[:, 6:].to(device)

        target = target.to(device)



        output = model.head(model(input1) - model(input2))

        loss = criterion(output, target)

        loss_sum += loss.data.cpu().numpy()



        optimizer.zero_grad()

        loss.backward()

        optimizer.step()



    return (loss_sum / len(train_loader))



def evaluate(eval_loader, model, criterion):

    model.eval()

    correct = 0

    with torch.no_grad():

        for input, target in tqdm(eval_loader):

            input1, input2 = input[:, :6].to(device), input[:, 6:].to(device)

            target = target.to(device)



            output = model.head(model(input1) - model(input2))

            preds = output.argmax(axis=1)

            correct += (target == preds).sum()



    return correct.cpu().numpy() * 100 / len(eval_loader.dataset)


model_file = 'model.pth'

if os.path.exists(model_file):

    print('loading model from checkpoint...')

    checkpoint = torch.load(model_file)

    model.load_state_dict(checkpoint['state_dict'])
for epoch in range(max_epochs):

    loss = train(train_loader, model, criterion, optimizer, epoch)

    acc = evaluate(eval_loader, model, criterion)

    print('epoch %d loss %.2f acc %.2f%%' % (epoch, loss, acc))

    

# save checkpoint

torch.save({'state_dict': model.state_dict()}, model_file)
model.eval()

with torch.no_grad():

    preds = np.empty(0)

    for input, _ in tqdm(tloader):

        input1, input2 = input[:, :6].to(device), input[:, 6:].to(device)

        output = model.head(model(input1) - model(input2))

        idx = output.max(dim=-1)[1].cpu().numpy()

        preds = np.append(preds, idx, axis=0)
submission = pd.read_csv(path_data + '/test.csv')

submission['sirna'] = preds.astype(int)

submission.to_csv('submission.csv', index=False, columns=['id_code','sirna'])