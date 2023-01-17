import pandas as pd

import numpy as np

import math



import torch

import torch.nn.functional as func

import torch.nn as nn

from torch.utils.data import DataLoader, Dataset



import torchvision

from torchvision import transforms



import tqdm

import matplotlib.pyplot as plt

from PIL import Image

%matplotlib inline



import warnings

from IPython.display import clear_output



warnings.filterwarnings("ignore")
torch.__version__
class CapsuleLayer(nn.Module):

    def __init__(self, num_caps, num_routes, in_channels, out_channels, k_size=None, 

                 stride=None, num_rounds=3,

                 use_padding=False):

        super(CapsuleLayer, self).__init__()



        self.num_routes = num_routes

        self.num_rounds = num_rounds

        self.num_caps = num_caps



        if num_routes != -1:

            self.W = nn.Parameter(torch.randn(num_caps, num_routes, in_channels, out_channels))

        else:

            self.capsules = nn.ModuleList(

                [nn.Conv2d(in_channels, out_channels, kernel_size=k_size, stride=stride,

                           padding=(k_size-1)//2 if use_padding else 0)

                 for _ in range(num_caps)]

            )



    @staticmethod

    def squash(x, dim=-1):

        s_norm = (x**2).sum(dim=dim, keepdim=True)

        scaled = s_norm / (1 + s_norm)

        return scaled * x / torch.sqrt(s_norm)



    def forward(self, x):



        if self.num_routes != -1:

            priors = x[None, :, :, None, :] @ self.W[:, None, :, :, :]

            logits = torch.zeros(*priors.size(), device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),

                                 requires_grad = True)

            for i in range(self.num_rounds):

                probs = func.softmax(logits, dim=2)

                outps = self.squash((probs * priors).sum(dim=2, keepdim=True))



                if i != self.num_rounds - 1:

                    del_logits = (priors * outps).sum(dim=-1, keepdim=True)

                    logits = logits + del_logits

        else:

            batch_size = x.size(0)

            outps = [cap(x).view(batch_size, -1, 1) for cap in self.capsules]

            outps = torch.cat(outps, dim=-1)

            outps = self.squash(outps)

        return outps



def conv_size(shape, k = 9, s = 1, p = False):

    H, W = shape

    if p:

        pad = (k-1)//2

    else:

        pad = 0



    Ho = math.floor(((H + 2*pad - (k - 1) - 1)/s) + 1)

    Wo = math.floor(((W + 2*pad - (k - 1) - 1)/s) + 1)



    return Ho, Wo



class CapsuleNetwork(nn.Module):

    def __init__(self, img_size, ic_channels, num_pcaps, num_classes, num_coc, num_doc, mode='mono', use_padding=False):

        super(CapsuleNetwork, self).__init__()



        self.initial_conv = nn.Conv2d(in_channels=1 if mode=='mono' else 3, out_channels=ic_channels, kernel_size=9, stride=1)

        Ho, Wo = conv_size(img_size, k=9, s=1, p=False)



        self.p_caps = CapsuleLayer(num_caps=num_pcaps, num_routes=-1, in_channels=ic_channels, out_channels=num_coc,

                                   k_size=9, stride=2)

        Ho, Wo = conv_size((Ho, Wo), k=9, s=2, p=use_padding)



        self.d_caps = CapsuleLayer(num_caps=num_classes, num_routes=num_coc*Ho*Wo, in_channels=num_pcaps, out_channels=num_doc)



        self.decoder = Decoder(num_doc)



    def forward(self, x):

        x = func.relu(self.initial_conv(x))

        x = self.p_caps(x)

        x = self.d_caps(x).squeeze().transpose(0,1)



        classes = (x ** 2).sum(dim=-1) ** 0.5

        classes = func.softmax(classes, dim=-1)



        _, max_index = classes.max(dim=1)

        y = torch.eye(10, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),

                             requires_grad = True).index_select(dim=0, index=max_index.data)

        reconst = self.decoder((x * y[:, :, None]).view(x.size(0), -1))



        return classes, reconst



class Decoder(nn.Module):



    def __init__(self, in_size):

        super(Decoder, self).__init__()



        self.fc1 = nn.Linear(in_size * 10, 512)

        self.relu = nn.ReLU()

        self.fc2 = nn.Linear(512, 784) # 1/8

        self.c1 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, padding=1) #1/4

        self.c2 = nn.Conv2d(in_channels=16, out_channels=8, kernel_size=3, padding=1) #1/2

        self.c3 = nn.Conv2d(in_channels=8, out_channels=1, kernel_size=3, padding=1) #1



    def forward(self, x):

        x = self.relu(self.fc1(x))

        x = self.relu(self.fc2(x))

        x = x.view(-1, 16, 7, 7)

        x = self.relu(self.c1(x))

        x = func.interpolate(x, scale_factor=2)

        x = self.relu(self.c2(x))

        x = func.interpolate(x, scale_factor=2)

        x = torch.sigmoid(self.c3(x))

        return x

    

class CapsuleLoss(nn.Module):

    def __init__(self):

        super(CapsuleLoss, self).__init__()

        self.reconst = nn.MSELoss(size_average=False)



    def forward(self, img, label, classes, reconst):

        label = torch.eye(10, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),

                  requires_grad=True).index_select(dim=0, index=label.data)

        left = func.relu(0.9-classes) ** 2

        right = func.relu(classes - 0.1) ** 2



        margin = label * left + 0.5 * (1-label) * right

        margin = margin.sum()



        recon = self.reconst(img, reconst)

        return (margin + 0.0005 * recon) / img.size(0)



    

def train_capnet(name, model, train_loader, valid_loader, epochs):

    optim = torch.optim.Adam(model.parameters(), lr=0.001)

    dev = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    loss = CapsuleLoss()

    

    LR = torch.optim.lr_scheduler.ReduceLROnPlateau(

        optimizer=optim,

        mode='max', 

        factor=0.1, 

        patience=3, 

        verbose=True, 

        threshold_mode='rel', 

        cooldown=2, 

        min_lr=0, 

        eps=1e-08

    )

    

    model.to(dev)

    loss.to(dev)

    best_model = None

    best_acc = 0.0

    terror = []

    tmean = []

    acc = []

    

    for e in range(epochs):

        lavg = 0



        for i, data in enumerate(train_loader):

            

            img, label = data

            img = (img - img.mean())/(img.std())

            img = img.to(dev)

            label = label.to(dev)

            optim.zero_grad()

            y, reconst = model(img)

            l = loss(img, label, y, reconst)

            lavg += l.item()

            if i > 0:

                lavg /= 2

            l.backward()

            terror.append(l.item())

            optim.step()

        tmean.append(sum(terror)/len(terror))



        correct = 0

        total = 0

        with torch.no_grad():

            for i, data in enumerate(valid_loader):

                img, labels = data

                img = (img - img.mean())/(img.std())

                img = img.to(dev)

                labels = labels.to(dev)

                y, _ = model(img)

                _, pred = torch.max(y.data, 1)

                total += labels.size(0)

                correct += (pred == labels).sum().item()

        accuracy = correct / total

        if accuracy > best_acc:

            best_model = model

            best_acc = accuracy

        acc.append(accuracy) 

        LR.step(accuracy)

        if e % 5 == 0:

            print(f'model: {name}, epoch: {e}, {correct}, {total}, accuracy: {100 * (correct/total)}%')

        

    return best_model
class MNIST(Dataset):

    

    def __init__(self, path, transforms = None):

        df = pd.read_csv(path)

        

        if len(df.columns) == 784:

            self.X = df.values.reshape((-1, 28, 28)).astype(np.float)[:, :, :]

            self.y = None

        else:

            self.X = df.iloc[:, 1:].values.reshape((-1, 28, 28)).astype(np.float)[:, :, :]

            self.y = torch.from_numpy(df.iloc[:,0].values)

        

        self.transforms = transforms

        

    def __len__(self):

        return len(self.X)

    

    def __getitem__(self, idx):

        X = self.X[idx]

        X = Image.fromarray(X)

        if self.y is not None:

            return self.transforms(X) if self.transforms is not None else X, self.y[idx]

        else:

            return self.transforms(X) if self.transforms is not None else X
batch_size = 512

n_caps = 8

ih, iw = 28, 28

n_class = 10

in_channels = 256

prime_channels = 32

dim_channels = 16
trans = transforms.Compose([

    transforms.RandomAffine(

        degrees=5, 

        translate=(0.05, 0.05), 

        scale=None, 

        shear=5, 

        resample=False, 

        fillcolor=0

    ),

     transforms.ToTensor()

])

data = MNIST(path = '../input/Kannada-MNIST/train.csv', transforms=trans)

print(len(data))

valid = MNIST(path = '../input/Kannada-MNIST/Dig-MNIST.csv', transforms=trans)

test = MNIST(path = '../input/Kannada-MNIST/test.csv', transforms=transforms.ToTensor())



data = torch.utils.data.ConcatDataset([data, valid])
print(len(data))

plt.imshow(data[35][0].squeeze(), 'gray')
cap_model = [('capsule', CapsuleNetwork(

    img_size = (ih, iw), 

    ic_channels = in_channels, 

    num_pcaps = n_caps, 

    num_classes = n_class, 

    num_coc = prime_channels, 

    num_doc = dim_channels, 

    mode='mono', 

    use_padding=False

))] # add more models to ensemble
train_load = DataLoader(dataset=data, batch_size=batch_size, shuffle=True, num_workers=4)

valid_load = DataLoader(dataset=valid, batch_size=batch_size, shuffle=True, num_workers=4)

test_load = DataLoader(dataset=test, batch_size=batch_size, shuffle=False, num_workers=4)
%%time

best_models = []

print('training using:', torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'))

for name, model in cap_model:    

    best = train_capnet(name = name, model=model, train_loader=train_load, valid_loader=valid_load, epochs=30)

    best_models.append(best)
def predicition(model, data_loader):

    model.eval()

    test_pred = torch.LongTensor()

    dev = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    model.to(dev)

    for i, d in enumerate(data_loader):

        d, _ = d

        d = d.to(dev)

        d = (d-d.mean())/d.std()

        with torch.no_grad():

            output = model(d)

        

        pred = output.cpu().data.max(1, keepdim=True)[1]

        test_pred = torch.cat((test_pred, pred), dim=0)

        

    return test_pred.numpy()



def predicition_cap(model, data_loader):

    model.eval()

    test_pred = torch.LongTensor()

    dev = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    model.to(dev)

    for i, d in enumerate(data_loader):

        d, _ = d

        d = d.to(dev)

        d = (d-d.mean())/d.std()

        with torch.no_grad():

            output, _ = model(d)

        

        pred = output.cpu().data.max(1, keepdim=True)[1]

        test_pred = torch.cat((test_pred, pred), dim=0)

        

    return test_pred.numpy()



def most_common(lst):

    return max(set(lst), key=lst.count)



def vote_pred(models):

    preds = []

    for model in models:

        pred = predicition_cap(model, test_load)

        preds.append(pred)

    preds = np.array(preds)

    return preds

    

preds = vote_pred(best_models)
pred = preds.squeeze()

print(pred.shape)

if len(pred.shape) > 1:

    pred = pred.transpose([1,0])

    pred = np.array([most_common(list(p)) for p in pred.squeeze()])

pred = pred.reshape([-1, 1])
submission = pd.read_csv('../input/Kannada-MNIST/sample_submission.csv')

submission['label'] = pred.astype(int)

submission.to_csv('submission.csv', index=False)
submission.head(10)