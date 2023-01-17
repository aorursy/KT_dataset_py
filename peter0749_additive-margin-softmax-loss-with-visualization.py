%matplotlib inline

import matplotlib.pyplot as plt

import numpy as np

import pandas as pd

import torch

import torch.nn as nn

import torch.nn.functional as F

import torch.optim as optim

from torch.autograd import Variable

from torch.nn import Parameter

from torchvision import datasets, transforms

from sklearn.model_selection import train_test_split

torch.backends.cudnn.bencmark = True

ndim = 28

fields = ndim**2

train_csv = pd.read_csv('../input/train.csv')

test_csv = pd.read_csv('../input/test.csv')



use_cuda = torch.cuda.is_available()

device = torch.device("cuda" if use_cuda else "cpu")
train_csv.head()
test_csv.head()
print(len(train_csv))

print(len(test_csv))
def norm(x):

    return (x-127.5)/128.0
d = np.asarray(train_csv.iloc[:,0].values, dtype=np.int32).reshape( len(train_csv) )

X_train, X_val, Y_train, Y_val = train_test_split(

    norm(np.asarray(train_csv.iloc[:,1:].values, dtype=np.float32).reshape( len(train_csv), 1 , ndim, ndim )), 

    d,

    test_size = 0.3, 

    shuffle = True,

    stratify = d

)

del d
X_train = torch.from_numpy(X_train)

Y_train = torch.from_numpy(Y_train).type(torch.LongTensor)

X_val = torch.from_numpy(X_val)

Y_val = torch.from_numpy(Y_val).type(torch.LongTensor)
train_dataset = torch.utils.data.TensorDataset(X_train, Y_train)

val_dataset   = torch.utils.data.TensorDataset(X_val, Y_val)

train_dataloader = torch.utils.data.DataLoader(

        train_dataset,

        batch_size=128, shuffle=True, num_workers=1)

val_dataloader = torch.utils.data.DataLoader(

        val_dataset,

        batch_size=128, shuffle=False, num_workers=1)
class AngleLinear(nn.Module):

    def __init__(self, in_features, out_features, m = 0.35, s = 40):

        super(AngleLinear, self).__init__()

        self.in_features = in_features

        self.out_features = out_features

        self.weight = Parameter(torch.Tensor(in_features,out_features))

        self.weight.data.uniform_(-1, 1).renorm_(2,1,1e-5).mul_(1e5)

        self.m = m

        self.s = s



    def forward(self, input):

        x = input   # x is a normalized feature vector, size=(B,F)    F is feature len

        w = self.weight # size=(F,Classnum) F=in_features Classnum=out_features



        ww = w.renorm(2,1,1e-5).mul(1e5)



        cos_theta = x.mm(ww) # size=(B,Classnum)

        cos_theta = cos_theta.clamp(-1,1)

        phi_theta = cos_theta - self.m



        output = (cos_theta*self.s,phi_theta*self.s)

        return output # size=(B,Classnum,2)



class AngleLoss(nn.Module):

    def __init__(self, test=False, mode='mean'):

        super(AngleLoss, self).__init__()

        self.it = 0

        self.LambdaMin = 1e-3

        self.LambdaMax = 1500.0

        self.lamb = 1500.0

        self.test = test

        self.mode = mode



    def forward(self, input, target):

        self.it += 1

        cos_theta,phi_theta = input

        target = target.view(-1,1) #size=(B,1)



        index = cos_theta.data * 0.0 #size=(B,Classnum)

        index.scatter_(1,target.data.view(-1,1),1)

        index = index.byte()

        index = Variable(index)



        output = cos_theta * 1.0 #size=(B,Classnum)

        if self.test:

            output[index] = phi_theta[index]

        else:

            self.lamb = max(self.LambdaMin,self.LambdaMax/(1+0.2*self.it ))

            output[index] -= cos_theta[index]*(1.0+0)/(1+self.lamb)

            output[index] += phi_theta[index]*(1.0+0)/(1+self.lamb)



        logpt = F.log_softmax(output, dim=-1)

        logpt = logpt.gather(1,target)

        logpt = logpt.view(-1) #.clamp(-2e9, 2e9)



        loss = -logpt

        if self.mode=='mean':

            loss = loss.mean()

        else:

            loss = loss.sum()



        return loss



# from PyTorch example

class Net(nn.Module):

    def __init__(self, feature=False):

        super(Net, self).__init__()

        # LeNet-5 like net:

        self.convs = nn.Sequential(

            nn.Conv2d(1, 6, kernel_size=(5, 5), padding=2),

            nn.ReLU(),

            nn.MaxPool2d(kernel_size=(2, 2), stride=2),

            nn.Dropout(0.2),

            nn.Conv2d(6, 16, kernel_size=(5, 5)),

            nn.ReLU(),

            nn.MaxPool2d(kernel_size=(2, 2), stride=2),

            nn.Dropout(0.25),

            nn.Conv2d(16, 120, kernel_size=(5, 5)),

            nn.ReLU(),

            nn.Dropout(0.3)

        )

        self.fc1 = nn.Linear(120, 300)

        self.fc2 = AngleLinear(300, 10)

        self.feature = feature



    def forward(self, x):

        x = self.convs(x)

        x = x.view(x.size(0), -1)

        x = self.fc1(x)

        x = x / x.pow(2).sum(1).pow(0.5).view(-1,1) ## normalized embedding feature

        if self.feature:

            return x

        x = self.fc2(x)

        return x
def train(model, device, train_loader, optimizer, epoch, loss_func):

    model.train()

    loss_tot = 0.0

    cnt = 0

    for batch_idx, (data, target) in enumerate(train_loader):

        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()

        output = model(data)

        loss = loss_func(output, target)

        loss.backward()

        optimizer.step()

        cnt += len(data)

        loss_tot += loss.item()*len(data)

    loss_tot /= cnt

    print('Train Epoch: {} \tLoss: {:.4f} \tλ: {:.4f}'.format(

        epoch, loss_tot, loss_func.lamb))

    return loss_tot



def test(model, device, test_loader):

    loss_func = AngleLoss(test=True,mode='sum')

    model.eval()

    test_loss = 0

    correct = 0

    with torch.no_grad():

        for data, target in test_loader:

            data, target = data.to(device), target.to(device)

            output = model(data)

            test_loss += loss_func(output, target).item() # sum up batch loss

            pred = output[0].argmax(dim=1, keepdim=True) # get the index of the max probability

            correct += pred.eq(target.view_as(pred)).sum().item()



    test_loss /= len(test_loader.dataset)



    print('Validation set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(

        test_loss, correct, len(test_loader.dataset),

        100. * correct / len(test_loader.dataset)))

    return test_loss



def inference(model, device, test_loader):

    model.eval()

    results = []

    with torch.no_grad():

        for data in test_loader:

            data = data[0].to(device)

            output = model(data)

            results += list(output[0].argmax(dim=1, keepdim=False).detach().cpu().numpy())

    return results



def get_feature(model, device, test_loader):

    model.eval()

    results = []

    with torch.no_grad():

        for data in test_loader:

            data = data[0].to(device)

            output = model(data)

            results += list(output.detach().cpu().numpy())

    return results
model = Net().to(device)

optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
epochs = 50

train_loss_func = AngleLoss()

train_loss = []

test_loss = []

best_loss = np.inf

best_state = None

for epoch in range(1, epochs + 1):

    train_loss += [train(model, device, train_dataloader, optimizer, epoch, train_loss_func)]

    test_loss += [test(model, device, val_dataloader)]

    if test_loss[-1] > 0 and test_loss[-1] < best_loss:

        best_loss = test_loss[-1]

        best_state = model.state_dict()
plt.plot(train_loss)

plt.plot(test_loss)

plt.title('A-softmax loss')

plt.ylabel('loss')

plt.xlabel('epoch')

plt.legend(['train','valid'], loc='upper right')

plt.show()
if not best_state is None:

    model.load_state_dict(best_state)
X_test = torch.from_numpy(norm(np.asarray(test_csv.values, dtype=np.float32).reshape( len(test_csv), 1 , ndim, ndim )))

test_dataset   = torch.utils.data.TensorDataset(X_test)

test_dataloader = torch.utils.data.DataLoader(

        test_dataset,

        batch_size=128, shuffle=False, num_workers=1)

results = inference(model, device, test_dataloader)

sub = pd.DataFrame()

sub['ImageId'] = list(range(1, 1+len(test_csv)))

sub['Label'] = results

sub.to_csv('./results.csv', index=False)

sub.head()
from sklearn.decomposition import PCA

pca = PCA(n_components=2)

del model

torch.cuda.empty_cache()

model = Net(feature=True).to(device)

if not best_state is None:

    model.load_state_dict(best_state)

features = np.asarray(get_feature(model, device, val_dataloader))

features_2d = pca.fit_transform(features)

labels = Y_val.cpu().numpy()
def plot_embedding(x_emb,y,n,title=''):

    cmap = plt.cm.gist_ncar_r

    for l in range(n):

        points = x_emb[y==l,:]

        plt.scatter(points[:,0], points[:,1], label=l, c=cmap((l+1)/(n+1)))

    plt.legend(loc='lower right', frameon=True, prop={'size': 10})

    plt.title(title)

    plt.show()
plot_embedding(features_2d, labels, 10, 'Embedding Visualization')