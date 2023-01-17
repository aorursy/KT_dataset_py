# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

        

import torch

import torch.nn as nn

import torch.nn.functional as F

from torch import nn, optim

from torch.autograd import Variable

from sklearn.decomposition import PCA



import pandas as pd

import numpy as np

from sklearn import preprocessing



# Any results you write to the current directory are saved as output.
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

device
DATA_PATH = '/kaggle/input/wineuci/Wine.csv'

pd.read_csv(DATA_PATH, sep=',', header=None, names=['Wine', 'Alcohol','Malic.acid','Ash','Acl',

                                                    'Mg', 'Phenols', 'Flavanoids','Nonflavanoid.phenols',

                                                    'Proanth','Color.int','Hue', 'OD','Proline']).head()
def load_data(path):

    # read in from csv

    df = pd.read_csv(DATA_PATH, sep=',', header=None, names=['Wine', 'Alcohol','Malic.acid','Ash','Acl',

                                                    'Mg', 'Phenols', 'Flavanoids','Nonflavanoid.phenols',

                                                    'Proanth','Color.int','Hue', 'OD','Proline'])

    # replace nan with -99

    df = df.fillna(-99)

    df_base = df.iloc[:, 1:]

    # get wine Label

    df_wine = df.iloc[:,0].values

    x = df_base.values.reshape(-1, df_base.shape[1]).astype('float32')

    # stadardize values

    standardizer = preprocessing.StandardScaler()

    x = standardizer.fit_transform(x)    

    return x, standardizer, df_wine
def numpyToTensor(x):

    x_train = torch.from_numpy(x).to(device)

    return x_train
x_pca, standardizer, df_wine = load_data(DATA_PATH)
x_pca.shape
pca = PCA(n_components=3)

principalComponents = pca.fit_transform(x_pca)

principalDf = pd.DataFrame(data = principalComponents

             , columns = ['principal component 1', 'principal component 2', 'principal component 3'])
finalDf = pd.concat([principalDf, pd.DataFrame(df_wine, columns = ['wine'])], axis = 1)

finalDf.head()
from torch.utils.data import Dataset, DataLoader

class DataBuilder(Dataset):

    def __init__(self, path):

        self.x, self.standardizer, self.wine = load_data(DATA_PATH)

        self.x = numpyToTensor(self.x)

        self.len=self.x.shape[0]

    def __getitem__(self,index):      

        return self.x[index]

    def __len__(self):

        return self.len
data_set=DataBuilder(DATA_PATH)

trainloader=DataLoader(dataset=data_set,batch_size=1024)
type(trainloader.dataset.x)
data_set.x
class Autoencoder(nn.Module):

    def __init__(self,D_in,H=50,H2=12,latent_dim=3):

        

        #Encoder

        super(Autoencoder,self).__init__()

        self.linear1=nn.Linear(D_in,H)

        self.lin_bn1 = nn.BatchNorm1d(num_features=H)

        self.linear2=nn.Linear(H,H2)

        self.lin_bn2 = nn.BatchNorm1d(num_features=H2)

        self.linear3=nn.Linear(H2,H2)

        self.lin_bn3 = nn.BatchNorm1d(num_features=H2)

        

#         # Latent vectors mu and sigma

        self.fc1 = nn.Linear(H2, latent_dim)

        self.bn1 = nn.BatchNorm1d(num_features=latent_dim)

        self.fc21 = nn.Linear(latent_dim, latent_dim)

        self.fc22 = nn.Linear(latent_dim, latent_dim)



#         # Sampling vector

        self.fc3 = nn.Linear(latent_dim, latent_dim)

        self.fc_bn3 = nn.BatchNorm1d(latent_dim)

        self.fc4 = nn.Linear(latent_dim, H2)

        self.fc_bn4 = nn.BatchNorm1d(H2)

        

#         # Decoder

        self.linear4=nn.Linear(H2,H2)

        self.lin_bn4 = nn.BatchNorm1d(num_features=H2)

        self.linear5=nn.Linear(H2,H)

        self.lin_bn5 = nn.BatchNorm1d(num_features=H)

        self.linear6=nn.Linear(H,D_in)

        self.lin_bn6 = nn.BatchNorm1d(num_features=D_in)

        

        self.relu = nn.ReLU()

        

    def encode(self, x):

        lin1 = self.relu(self.lin_bn1(self.linear1(x)))

        lin2 = self.relu(self.lin_bn2(self.linear2(lin1)))

        lin3 = self.relu(self.lin_bn3(self.linear3(lin2)))



        fc1 = F.relu(self.bn1(self.fc1(lin3)))



        r1 = self.fc21(fc1)

        r2 = self.fc22(fc1)

        

        return r1, r2

    

    def reparameterize(self, mu, logvar):

        if self.training:

            std = logvar.mul(0.5).exp_()

            eps = Variable(std.data.new(std.size()).normal_())

            return eps.mul(std).add_(mu)

        else:

            return mu

        

    def decode(self, z):

        fc3 = self.relu(self.fc_bn3(self.fc3(z)))

        fc4 = self.relu(self.fc_bn4(self.fc4(fc3)))



        lin4 = self.relu(self.lin_bn4(self.linear4(fc4)))

        lin5 = self.relu(self.lin_bn5(self.linear5(lin4)))

        return self.lin_bn6(self.linear6(lin5))





        

    def forward(self, x):

        mu, logvar = self.encode(x)

        z = self.reparameterize(mu, logvar)

        # self.decode(z) ist später recon_batch, mu ist mu und logvar ist logvar

        return self.decode(z), mu, logvar
class customLoss(nn.Module):

    def __init__(self):

        super(customLoss, self).__init__()

        self.mse_loss = nn.MSELoss(reduction="sum")

    

    # x_recon ist der im forward im Model erstellte recon_batch, x ist der originale x Batch, mu ist mu und logvar ist logvar 

    def forward(self, x_recon, x, mu, logvar):

        loss_MSE = self.mse_loss(x_recon, x)

        loss_KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())



        return loss_MSE + loss_KLD
# takes in a module and applies the specified weight initialization

def weights_init_uniform_rule(m):

    classname = m.__class__.__name__

    # for every Linear layer in a model..

    if classname.find('Linear') != -1:

        # get the number of the inputs

        n = m.in_features

        y = 1.0/np.sqrt(n)

        m.weight.data.uniform_(-y, y)

        m.bias.data.fill_(0)
D_in = data_set.x.shape[1]

H = 50

H2 = 12

model = Autoencoder(D_in, H, H2).to(device)

model.apply(weights_init_uniform_rule)

optimizer = optim.Adam(model.parameters(), lr=1e-3)
loss_mse = customLoss()
epochs = 1500

log_interval = 50

val_losses = []

train_losses = []
def train(epoch):

    model.train()

    train_loss = 0

    for batch_idx, data in enumerate(trainloader):

        data = data.to(device)

        optimizer.zero_grad()

        recon_batch, mu, logvar = model(data)

        loss = loss_mse(recon_batch, data, mu, logvar)

        loss.backward()

        train_loss += loss.item()

        optimizer.step()

#        if batch_idx % log_interval == 0:

#            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(

#                epoch, batch_idx * len(data), len(trainloader.dataset),

#                       100. * batch_idx / len(trainloader),

#                       loss.item() / len(data)))

    if epoch % 200 == 0:        

        print('====> Epoch: {} Average loss: {:.4f}'.format(

            epoch, train_loss / len(trainloader.dataset)))

        train_losses.append(train_loss / len(trainloader.dataset))
for epoch in range(1, epochs + 1):

    train(epoch)
standardizer = trainloader.dataset.standardizer
model.eval()

test_loss = 0

# no_grad() bedeutet wir nehmen die vorher berechneten Gewichte und erneuern sie nicht

with torch.no_grad():

    for i, data in enumerate(trainloader):

        data = data.to(device)

        recon_batch, mu, logvar = model(data)
standardizer.inverse_transform(recon_batch[65].cpu().numpy())
standardizer.inverse_transform(data[65].cpu().numpy())
mu_output = []

logvar_output = []



with torch.no_grad():

    for i, (data) in enumerate(trainloader):

            data = data.to(device)

            optimizer.zero_grad()

            recon_batch, mu, logvar = model(data)



            

            mu_tensor = mu   

            mu_output.append(mu_tensor)

            mu_result = torch.cat(mu_output, dim=0)



            logvar_tensor = logvar   

            logvar_output.append(logvar_tensor)

            logvar_result = torch.cat(logvar_output, dim=0)
mu_result.shape
mu_result[1:5,:]
from mpl_toolkits import mplot3d



%matplotlib inline

import numpy as np

import matplotlib.pyplot as plt




ax = plt.axes(projection='3d')





# Data for three-dimensional scattered points

winetype = finalDf.iloc[:,3].values

zdata = finalDf.iloc[:,0].values

xdata = finalDf.iloc[:,1].values

ydata = finalDf.iloc[:,2].values

ax.scatter3D(xdata, ydata, zdata, c=winetype);



ax = plt.axes(projection='3d')





# Data for three-dimensional scattered points

winetype = data_set.wine

zdata = mu_result[:,0].cpu().numpy()

xdata = mu_result[:,1].cpu().numpy()

ydata = mu_result[:,2].cpu().numpy()

ax.scatter3D(xdata, ydata, zdata, c=winetype);