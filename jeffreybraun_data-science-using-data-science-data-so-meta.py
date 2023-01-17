import os
import unicodedata
import warnings
warnings.simplefilter(action='ignore')
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns 
import sklearn as sl

df = pd.read_csv("../input/medium-articles-dataset/medium_data.csv")

def normalize_text(text : str) -> str:
    """ Normalize the unicode string
        :param text: text data
        :retrns clean_text: clean text
    """
    if text != np.nan:
        clean_text = unicodedata.normalize("NFKD",text)
    else:
        clean_text = text
    return clean_text

def create_wc(text : str) -> int:
    """ Count words in a text
        :param text: String to check the len
        :retirns wc: Word count
    """
    wc = 0
    norm_text = text.lower()
    wc = len(norm_text.split(" "))
    return wc

df['clean_title'] = df.title.apply(lambda x: normalize_text(x) if x!= np.nan else x)
df['clean_subtitle'] = df.subtitle.apply(lambda x: normalize_text(x) if x!= np.nan and type(x) == str else x)
df['title_wc'] = df.title.apply(lambda x: create_wc(x) if x!= np.nan else 0)
df['subtitle_wc'] = df.subtitle.apply(lambda x: create_wc(x) if x!= np.nan and type(x) == str else 0)
df.head()
df.info()
print("Value Counts for each publication:")
print(df.publication.value_counts())

sns.set(style="ticks", color_codes=True)
plt.style.use('fivethirtyeight')

g = sns.catplot(x="claps", y="publication", kind="box", data=df, order=df.publication.value_counts().iloc[:25].index)
g.set(xlim=(-25, 2500))
g.fig.set_size_inches(28, 10)
g.ax.set_xticks([0,250,500,750,1000,1250,1500,1750,2000,2250,2500], minor=True)
plt.title("Claps vs. Publication - Ordered by Most Common Publications")
plt.show()

df_ordered = df.groupby("publication").median().sort_values(by = 'claps', ascending=False)

g = sns.catplot(x="claps", y="publication", kind="box", data=df, order=df_ordered.iloc[:25].index)
g.set(xlim=(-25, 2500))
g.fig.set_size_inches(28, 10)
g.ax.set_xticks([0,250,500,750,1000,1250,1500,1750,2000,2250,2500], minor=True)
plt.title("Claps vs. Publication - Ordered by Median Claps")
plt.show()
plt.figure(figsize=(30,10))
g = sns.scatterplot(x="claps", y="reading_time", data=df, legend='full')
g.set(xlim=(-25, 1050))
plt.title("Claps vs. Reading Time")
plt.show()
print("Value Counts for each reading time:")
print(df.reading_time.value_counts())
print('\n')

plt.style.use('fivethirtyeight')

def bin_time(row):
    read_time = row['reading_time']
    if read_time <= 3:
        return 'Very Short (0-3 minutes)'
    elif read_time > 3 and read_time <= 7:
        return 'Short (4-7 minutes)'
    elif read_time > 7 and read_time <= 12:
        return 'Substantial (8-12 minutes)'
    elif read_time > 12:
        return 'Long (13+ minutes)'

df['reading_time_bin'] = df.apply(lambda row: bin_time(row), axis=1)
print("Value Counts for each reading time bin:")
print(df.reading_time_bin.value_counts())


df_ordered = df.groupby("reading_time_bin").median().sort_values(by = 'claps', ascending=False)

g = sns.catplot(x="claps", y="reading_time_bin", kind="box", data=df, order=df_ordered.iloc[:25].index)
g.set(xlim=(-25, 1050))
g.fig.set_size_inches(28, 10)
g.ax.set_xticks([0,100,200,300,400,500,600,700,800,900,1000], minor=True)
plt.title("Claps vs. Reading Time - Ordered by Median Claps")
plt.show()

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image

print(torch.__version__)
torch.cuda.is_available()
from tqdm.notebook import tqdm

df_img = df.dropna(subset=['image']).reset_index()

image_data = np.zeros([df_img.shape[0], 1, 15, 50], dtype=float)
image_labels = np.zeros([df_img.shape[0]], dtype=float)

for i in tqdm(range(df_img.shape[0])):
    img_file = df_img.loc[i, 'image']
    if img_file[-1] == '.':
        img_file = img_file.strip('.')
    img = Image.open('/kaggle/input/medium-articles-dataset/images/' + img_file).convert('LA').resize((50,15),resample=Image.BILINEAR)
    img_data = np.asarray(img)
    image_data[i,0,:,:] = img_data[:,:,0]
    num_clap = df_img.loc[i, 'claps']
    if num_clap < 50:
        image_labels[i] = 0
    elif num_clap >= 50 and num_clap < 100:
        image_labels[i] = 1
    elif num_clap >= 100 and num_clap < 400:
        image_labels[i] = 2
    else:
        image_labels[i] = 3
    
train_cut = int(np.floor(image_data.shape[0] * 5/7))
val_cut = int(np.floor(image_data.shape[0] * 1/7))
        
image_data = np.float32(image_data)
image_labels = np.float32(image_labels)
image_data = image_data/255.
train_dat = image_data[0:train_cut]
train_labels = image_labels[0:train_cut]
val_dat = image_data[train_cut:train_cut + val_cut]
val_labels = image_labels[train_cut:train_cut + val_cut]
test_dat = image_data[train_cut + val_cut:]
test_labels = image_labels[train_cut + val_cut:]
class MNIST_Net(nn.Module):
    """ 
    Original Model
    """
    def __init__(self,p=0.5,minimizer='Adam',step_size=0.001):
        super(MNIST_Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5)
        self.conv2_drop = nn.Dropout2d(p)
        self.fc1 = nn.Linear(640, 64)
        self.fc2 = nn.Linear(64, 4)
        if minimizer == 'Adam':
            self.optimizer = torch.optim.Adam(self.parameters(), lr = step_size)
        else:
            optimizer = torch.optim.SGD(self.parameters(), lr = step_size, momentum=0.9)
         
        self.criterion=nn.CrossEntropyLoss()
            
    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 640)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return x
    
    def get_acc_and_loss(self, data, targ):
        output = self.forward(data)
        loss = self.criterion(output, targ)
        pred = torch.max(output,1)[1]
        correct = torch.eq(pred,targ).sum()
        
        return loss,correct
        
    def run_grad(self,data,targ):
    
        loss, correct=self.get_acc_and_loss(data,targ)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss, correct
def run_epoch(net,epoch,train,batch_size, num=None, ttype="train"):
    
    error_rates_batch = []
    epoch_indices = []
    net.train()
    if ttype=='train':
        t1=time.time()
        n=train[0].shape[0]
        if (num is not None):
            n=np.minimum(n,num)
        ii=np.array(np.arange(0,n,1))
        tr=train[0][ii]
        y=train[1][ii]
        train_loss=0; train_correct=0
        with tqdm(total=len(y)) as progress_bar:
            for j in np.arange(0,len(y),batch_size):
                data=torch.torch.from_numpy(tr[j:j+batch_size]).to(device)
                targ=torch.torch.from_numpy(y[j:j+batch_size]).type(torch.long).to(device)
                loss, correct = net.run_grad(data,targ) 
                train_loss += loss.item()
                train_correct += correct.item()

                error_rates_batch.append(1 - correct.item() / batch_size)
                epoch_indices.append(epoch + j/len(y))
                
                progress_bar.set_postfix(loss=loss.item())
                progress_bar.update(data.size(0))
        train_loss /= len(y)
        print('\nTraining set epoch {}: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(epoch,
            train_loss, train_correct, len(y),
            100. * train_correct / len(y)))
        
    return error_rates_batch, epoch_indices
        
    

def net_test(net,val,batch_size,ttype='val'):

    error_rates_batch = []
    batch_indices = []
    net.eval()
    with torch.no_grad():
                test_loss = 0
                test_correct = 0
                vald=val[0]
                yval=val[1]
                for j in np.arange(0,len(yval),batch_size):
                    data=torch.torch.from_numpy(vald[j:j+batch_size]).to(device)
                    targ = torch.torch.from_numpy(yval[j:j+batch_size]).type(torch.long).to(device)
                    loss,correct=net.get_acc_and_loss(data,targ)

                    test_loss += loss.item()
                    test_correct += correct.item()

                    error_rates_batch.append(1 - correct.item()/batch_size)
                    batch_indices.append(j/len(yval))

                test_loss /= len(yval)
                SSS='Validation'
                if (ttype=='test'):
                    SSS='Test'
                print('\n{} set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(SSS,
                    test_loss, test_correct, len(yval),
                    100. * test_correct / len(yval)))
    return error_rates_batch, batch_indices

import time

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print(device)

def create_model(net_id, model_name):
    batch_size=64
    step_size=.001
    num_epochs=20
    numtrain=50000
    minimizer="Adam"
    dropout_p=0.5
    dim=28
    nchannels=1

    # use GPU when possible
    train = (train_dat, train_labels)
    val = (val_dat, val_labels)
    test = (test_dat, test_labels)
    if net_id == 0:
        net = MNIST_Net(p = dropout_p, minimizer=minimizer,step_size=step_size)
    net.to(device)
    #define optimizer

    ers_train = []
    eis_train = []
    ers_val = []
    eis_val = []
    
    for i in range(num_epochs):
        [error_rates, epoch_indices] = run_epoch(net,i,train,batch_size, num=numtrain, ttype="train")
        ers_train.extend(error_rates)
        eis_train.extend(epoch_indices)
        [error_rates, batch_indices] = net_test(net,val,batch_size)
        ers_val.extend(error_rates)
        eis_val.extend([x+i for x in batch_indices])

    plt.plot(eis_train, ers_train)
    plt.title("Training Set")
    plt.xlabel("Epoch")
    plt.ylabel("Error Rate")
    plt.show()

    plt.plot(eis_val, ers_val)
    plt.title("Validation Set")
    plt.xlabel("Epoch")
    plt.ylabel("Eror_Rate")
    plt.show()

    net_test(net,test,batch_size,ttype='test')

    #torch.save(net.state_dict(), datadir+'models/'+model_name)

    return net
my_net = create_model(0, "original_model")