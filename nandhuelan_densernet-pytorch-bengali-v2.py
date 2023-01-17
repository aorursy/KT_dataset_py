# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

import glob

import dask.dataframe as dd

import gc

import matplotlib.pyplot as plt

import plotly.express as px

from IPython.core.display import display, HTML

import cv2

import torch.nn as nn

from torch.utils.data import Dataset,DataLoader

import torch,torchvision

from torchvision import transforms,models

from torchvision.models import DenseNet

from tqdm import tqdm_notebook as tqdm

from collections import OrderedDict

import seaborn as sns

import sklearn.metrics

import warnings



## This library is for augmentations .

from albumentations import (

    PadIfNeeded,

    HorizontalFlip,

    VerticalFlip,    

    CenterCrop,    

    Crop,

    Compose,

    Transpose,

    RandomRotate90,

    ElasticTransform,

    GridDistortion, 

    OpticalDistortion,

    RandomSizedCrop,

    OneOf,

    CLAHE,

    RandomBrightnessContrast,    

    RandomGamma,

    ShiftScaleRotate    

)



warnings.filterwarnings('ignore')

%matplotlib inline



# Any results you write to the current directory are saved as output.
PATH='../input/bengaliai-cv19/'

feathers='../input/bengaliaicv19feather/'



class_map = pd.read_csv(PATH+"class_map.csv")

sample_submission = pd.read_csv(PATH+"sample_submission.csv")

test = pd.read_csv(PATH+"test.csv")

train = pd.read_csv(PATH+"train.csv")
train_parquet_files=glob.glob(PATH+'train_image_data*.parquet')

test_parquet_files=glob.glob(PATH+'test_image_data*.parquet')
HEIGHT=137

WIDTH=236
%%time

df = dd.read_parquet(train_parquet_files[0])



def load_as_npa(df):

    imageid=df.iloc[:, 0]

    

    return imageid.compute,df[list(df.columns[1:])].to_dask_array(lengths=True).reshape(-1,HEIGHT, WIDTH)



image_ids0, images0 = load_as_npa(df)



gc.collect()
%%time

df = pd.read_parquet(train_parquet_files[0])



def load_as_npa(df):

    return df.iloc[:, 0], df.iloc[:, 1:].values.reshape(-1, HEIGHT, WIDTH)



image_ids0, images0 = load_as_npa(df)
# preview the images first

plt.figure(figsize=(16,10))

x, y = 5,5



for i in range(10):  

    plt.subplot(y, x, i+1)

    plt.imshow(images0[i])
nrows = train.shape[0]

root = train["grapheme_root"].nunique()

vowel = train["vowel_diacritic"].nunique()

consonant = train["consonant_diacritic"].nunique()



max_root = train["grapheme_root"].value_counts().index[0]

max_vowel = train["vowel_diacritic"].value_counts().index[0]

max_consonant=train["consonant_diacritic"].value_counts().index[0]



display(HTML(f"""<br>Number of rows in the dataset: {nrows:,}</br>

             <br>Number of unique grapheme root in the dataset: {root:,}</br>

             <br>Number of unique vowels in the dataset: {vowel:,}</br>

             <br>Number of unique consonants in the dataset: {consonant:,}</br>

             <br>Most occuring grapheme root id {max_root}</br>

             <br>Most occuring vowel_diacritic id {max_vowel}</br>

             <br>Most occuring consonant_diacritic id {max_consonant}</br>

             """))
plots = train['grapheme_root'].value_counts().reset_index()

plots.columns=['Grapheme roots','Counts']

fig = px.scatter(plots, x="Grapheme roots", y="Counts",size='Counts', hover_data=['Grapheme roots'])



fig.update_traces(marker=dict(line=dict(width=2,

                                        color='MediumPurple')),

                  selector=dict(mode='markers'))



fig.show()

plots = train['vowel_diacritic'].value_counts().reset_index()

plots.columns=['Vowels','Counts']

fig = px.scatter(plots, x="Vowels", y="Counts",size='Counts', hover_data=['Vowels'])



fig.update_traces(marker=dict(line=dict(width=2,

                                        color='DarkSlateGrey')),

                  selector=dict(mode='markers'))

fig.show()
plots = train['consonant_diacritic'].value_counts().reset_index()

plots.columns=['Consonants','Counts']

fig = px.scatter(plots, x="Consonants", y="Counts",size='Counts', hover_data=['Consonants'])



fig.update_traces(marker=dict(line=dict(width=2,

                                        color='#bcbd22')),

                  selector=dict(mode='markers'))

fig.show()
del train_parquet_files,test_parquet_files



images0 = pd.read_feather(feathers+'train_image_data_0.feather')

images1 = pd.read_feather(feathers+'train_image_data_1.feather')

images2 = pd.read_feather(feathers+'train_image_data_2.feather')

images3 = pd.read_feather(feathers+'train_image_data_3.feather')



test_0=pd.read_feather(feathers+'test_image_data_0.feather')

test_1=pd.read_feather(feathers+'test_image_data_1.feather')

test_2=pd.read_feather(feathers+'test_image_data_2.feather')

test_3=pd.read_feather(feathers+'test_image_data_3.feather')
#Credits: https://www.kaggle.com/phoenix9032/pytorch-efficientnet-starter-code/data



SIZE = 128



def bbox(img):

    rows = np.any(img, axis=1)

    cols = np.any(img, axis=0)

    rmin, rmax = np.where(rows)[0][[0, -1]]

    cmin, cmax = np.where(cols)[0][[0, -1]]

    return rmin, rmax, cmin, cmax



def crop_resize(img0, size=SIZE, pad=16):

    

    #crop a box around pixels large than the threshold 

    #some images contain line at the sides

    ymin,ymax,xmin,xmax = bbox(img0[5:-5,5:-5] > 80)

    

    #cropping may cut too much, so we need to add it back

    xmin = xmin - 13 if (xmin > 13) else 0

    ymin = ymin - 10 if (ymin > 10) else 0

    xmax = xmax + 13 if (xmax < WIDTH - 13) else WIDTH

    ymax = ymax + 10 if (ymax < HEIGHT - 10) else HEIGHT

    img = img0[ymin:ymax,xmin:xmax]

    

    #remove lo intensity pixels as noise

    img[img < 28] = 0

    lx, ly = xmax-xmin,ymax-ymin

    l = max(lx,ly) + pad

    

    #make sure that the aspect ratio is kept in rescaling

    img = np.pad(img, [((l-ly)//2,), ((l-lx)//2,)], mode='constant')

    return cv2.resize(img,(size,size))



def Resize(df,size=128):

    resized = {} 

    df = df.set_index('image_id')

    

    for i in tqdm(range(df.shape[0])): 

        image0 = 255 - df.loc[df.index[i]].values.reshape(137,236).astype(np.uint8)

        

        #normalize each image by its max val

        img = (image0*(255.0/image0.max())).astype(np.uint8)

        image = crop_resize(img)

        resized[df.index[i]] = image.reshape(-1)

    resized = pd.DataFrame(resized).T.reset_index()

    resized.columns = resized.columns.astype(str)

    resized.rename(columns={'index':'image_id'},inplace=True)

    return resized
images0=Resize(images0)

images1=Resize(images1)

images2=Resize(images2)

images3=Resize(images3)
data_full = pd.concat([images0,images1,images2,images3],ignore_index=True)



del images0,images1,images2,images3

gc.collect()
nrow, ncol = 5, 5



fig, axes = plt.subplots(nrow, ncol, figsize=(15, 7))

axes = axes.flatten()

for i, ax in enumerate(axes):

    img0 = data_full.iloc[i, 1:].values.reshape(SIZE, SIZE).astype(np.uint8)

    ax.imshow(img0)



plt.tight_layout()

## Add Augmentations as suited from Albumentations library

train_aug = Compose([ 

    HorizontalFlip(p=0.1),              

    ShiftScaleRotate(p=1),

    RandomGamma(p=0.8)])
class GraphemeDataset(Dataset):

    def __init__(self,df,label=None,_type='train',transform =True,aug=train_aug):

        self.df = df

        self.label = label

        self.aug = aug

        self.transform = transform

        self.type=_type

        

    def __len__(self):

        return len(self.df)

    

    def __getitem__(self,idx):

        

        if self.type=='train':

            label1 = self.label.vowel_diacritic.values[idx]

            label2 = self.label.grapheme_root.values[idx]

            label3 = self.label.consonant_diacritic.values[idx]

            image = self.df.iloc[idx][1:].values.reshape(SIZE,SIZE).astype(np.float)

            

            augment = self.aug(image =image)

            image = augment['image']



            return image,label1,label2,label3

        else:

            image = self.df.iloc[idx][1:].values.reshape(SIZE,SIZE).astype(np.float)

            return image
train_dataset = GraphemeDataset(data_full ,train[['grapheme_root', 'vowel_diacritic', 'consonant_diacritic']],transform = True) 
##Visulization function for checking Original and augmented image



#Credits: https://www.kaggle.com/phoenix9032/pytorch-efficientnet-starter-code/data





def visualize(original_image,aug_image):

    fontsize = 18

    

    f, ax = plt.subplots(1, 2, figsize=(8, 8))



    ax[0].imshow(original_image, cmap='gray')

    ax[0].set_title('Original image', fontsize=fontsize)

    ax[1].imshow(aug_image,cmap='gray')

    ax[1].set_title('Augmented image', fontsize=fontsize)
## One image taken from raw dataframe another from dataset 



orig_image = data_full.iloc[0, 1:].values.reshape(128,128).astype(np.float)

aug_image = train_dataset[0][0]

## Check it 

visualize (orig_image,aug_image)



del aug_image,orig_image,train_dataset

gc.collect()
class Dense_Block(nn.Module):



    def __init__(self, in_channels):

        super(Dense_Block, self).__init__()



        self.relu = nn.ReLU(inplace = True)

        self.bn = nn.BatchNorm2d(num_features = in_channels)



        self.conv1 = nn.Conv2d(in_channels = in_channels, out_channels = 32, kernel_size = 3, stride = 1, padding = 1)

        self.conv2 = nn.Conv2d(in_channels = 32, out_channels = 32, kernel_size = 3, stride = 1, padding = 1)

        self.conv3 = nn.Conv2d(in_channels = 64, out_channels = 32, kernel_size = 3, stride = 1, padding = 1)

        self.conv4 = nn.Conv2d(in_channels = 96, out_channels = 32, kernel_size = 3, stride = 1, padding = 1)

        self.conv5 = nn.Conv2d(in_channels = 128, out_channels = 32, kernel_size = 3, stride = 1, padding = 1)



    

    def forward(self, x):



        bn = self.bn(x)

        conv1 = self.relu(self.conv1(bn))



        conv2 = self.relu(self.conv2(conv1))

        c2_dense = self.relu(torch.cat([conv1, conv2], 1))



        conv3 = self.relu(self.conv3(c2_dense))

        c3_dense = self.relu(torch.cat([conv1, conv2, conv3], 1))



        conv4 = self.relu(self.conv4(c3_dense))

        c4_dense = self.relu(torch.cat([conv1, conv2, conv3, conv4], 1))



        conv5 = self.relu(self.conv5(c4_dense))

        c5_dense = self.relu(torch.cat([conv1, conv2, conv3, conv4, conv5], 1))



        return c5_dense





class Transition_Layer(nn.Module):

    def __init__(self, in_channels, out_channels):

        super(Transition_Layer, self).__init__()



        self.relu = nn.ReLU(inplace = True)

        self.bn = nn.BatchNorm2d(num_features = out_channels)

        self.conv = nn.Conv2d(in_channels = in_channels, out_channels = out_channels, kernel_size = 1, bias = False)

        self.avg_pool = nn.AvgPool2d(kernel_size = 2, stride = 2, padding = 0)



    def forward(self, x):



        bn = self.bn(self.relu(self.conv(x)))

        out = self.avg_pool(bn)



        return out





class DenseNet(nn.Module):

    def __init__(self):

        super(DenseNet, self).__init__()



        self.lowconv = nn.Conv2d(in_channels = 1, out_channels = 64, kernel_size = 7, padding = 3, bias = False)

        self.relu = nn.ReLU()



        # Make Dense Blocks

        self.denseblock1 = self._make_dense_block(Dense_Block, 64)

        self.denseblock2 = self._make_dense_block(Dense_Block, 128)

        self.denseblock3 = self._make_dense_block(Dense_Block, 128)



        # Make transition Layers

        self.transitionLayer1 = self._make_transition_layer(Transition_Layer, in_channels = 160, out_channels = 128)

        self.transitionLayer2 = self._make_transition_layer(Transition_Layer, in_channels = 160, out_channels = 128)

        self.transitionLayer3 = self._make_transition_layer(Transition_Layer, in_channels = 160, out_channels = 64)



        # Classifier

        self.bn = nn.BatchNorm2d(num_features = 64)

        

        self.pre_classifier = nn.Linear(16384, 256)

        

         # vowel_diacritic

        self.fc1 = nn.Linear(256,11)

        

        # grapheme_root

        self.fc2 = nn.Linear(256,168)

        

        # consonant_diacritic

        self.fc3 = nn.Linear(256,7)

        

    def _make_dense_block(self, block, in_channels):

        layers = []

        layers.append(block(in_channels))

        return nn.Sequential(*layers)



    def _make_transition_layer(self, layer, in_channels, out_channels):

        modules = []

        modules.append(layer(in_channels, out_channels))

        return nn.Sequential(*modules)



    def forward(self, x):

        out = self.relu(self.lowconv(x))



        out = self.denseblock1(out)

        out = self.transitionLayer1(out)



        out = self.denseblock2(out)

        out = self.transitionLayer2(out)



        out = self.denseblock3(out)

        out = self.transitionLayer3(out)

    

        out = self.bn(out)

        out = out.view(out.size(0),-1)



        out = self.pre_classifier(out)

        

        x1 = self.fc1(out)

        x2 = self.fc2(out)

        x3 = self.fc3(out)



        return x1,x2,x3
def macro_recall_multi(pred_graphemes, true_graphemes,pred_vowels,true_vowels,pred_consonants,true_consonants, n_grapheme=168, n_vowel=11, n_consonant=7):

    

    pred_label_graphemes = torch.argmax(pred_graphemes, dim=1).cpu().numpy()



    true_label_graphemes = true_graphemes.cpu().numpy()

    

    pred_label_vowels = torch.argmax(pred_vowels, dim=1).cpu().numpy()



    true_label_vowels = true_vowels.cpu().numpy()

    

    pred_label_consonants = torch.argmax(pred_consonants, dim=1).cpu().numpy()



    true_label_consonants = true_consonants.cpu().numpy()    



    recall_grapheme = sklearn.metrics.recall_score(pred_label_graphemes, true_label_graphemes, average='macro')

    recall_vowel = sklearn.metrics.recall_score(pred_label_vowels, true_label_vowels, average='macro')

    recall_consonant = sklearn.metrics.recall_score(pred_label_consonants, true_label_consonants, average='macro')

    scores = [recall_grapheme, recall_vowel, recall_consonant]

    final_score = np.average(scores, weights=[2, 1, 1])



    return final_score
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = DenseNet().to(device)



optimizer = torch.optim.Adam(model.parameters(), lr=4e-4)



criterion = nn.CrossEntropyLoss()



batch_size=64
epochs = 80

model.train()

losses = []

accs = []

recall=[]



for epoch in range(epochs):

    reduced_index =train.groupby(['grapheme_root', 'vowel_diacritic', 'consonant_diacritic']).apply(lambda x: x.sample(5)).image_id.values

    

    reduced_train = train.loc[train.image_id.isin(reduced_index)]

    reduced_data = data_full.loc[data_full.image_id.isin(reduced_index)]

    

    train_image = GraphemeDataset(reduced_data,reduced_train,transform = True)

    train_loader = torch.utils.data.DataLoader(train_image,batch_size=batch_size,shuffle=True)

    

    print('epochs {}/{} '.format(epoch+1,epochs))

    running_loss = 0.0

    running_acc = 0.0

    running_recall=0.0

    

    for idx, (inputs,labels1,labels2,labels3) in tqdm(enumerate(train_loader),total=len(train_loader)):

        

        inputs = inputs.to(device)

        labels1 = labels1.to(device)

        labels2 = labels2.to(device)

        labels3 = labels3.to(device)

        

        optimizer.zero_grad()

        

        outputs1,outputs2,outputs3 = model(inputs.unsqueeze(1).float())

        

        loss1 = criterion(outputs1,labels1)

        loss2 = criterion(outputs2,labels2)

        loss3 = criterion(outputs3,labels3)

        running_loss += loss1+loss2+loss3

        

        running_acc += (outputs1.argmax(1)==labels1).float().mean()

        running_acc += (outputs2.argmax(1)==labels2).float().mean()

        running_acc += (outputs3.argmax(1)==labels3).float().mean()

        

        running_recall+= macro_recall_multi(outputs2,labels2,outputs1,labels1,outputs3,labels3)

        

        (loss1+loss2+loss3).backward()

        optimizer.step()

    

    recall.append(running_recall/len(train_loader))

    losses.append(running_loss/len(train_loader))

    accs.append(running_acc/(len(train_loader)*3))

    

    print('recall: {:.4f}'.format(running_recall/len(train_loader)))

    print('acc : {:.2f}%'.format(running_acc/(len(train_loader)*3)))

    print('loss : {:.4f}'.format(running_loss/len(train_loader)))

    

torch.save(model.state_dict(), 'densenet_epochs_saved_weights.pth')
fig,ax = plt.subplots(1,3,figsize=(15,5))

ax[0].plot(losses)

ax[0].set_title('loss')

ax[1].plot(accs)

ax[1].set_title('acc')

ax[2].plot(recall)

ax[2].set_title('Recall')
torch.cuda.empty_cache()

gc.collect()



model = DenseNet().to(device)

model.load_state_dict(torch.load('densenet_epochs_saved_weights.pth'))
model.eval()

test_data = ['test_image_data_0.parquet','test_image_data_1.parquet','test_image_data_2.parquet','test_image_data_3.parquet']

predictions = []

batch_size=1



for fname in test_data:

    data = pd.read_parquet(f'/kaggle/input/bengaliai-cv19/{fname}')

    data = Resize(data)

    test_image = GraphemeDataset(data,_type='test')

    test_loader = torch.utils.data.DataLoader(test_image,batch_size=1,shuffle=False)

    

    with torch.no_grad():

        for idx, (inputs) in tqdm(enumerate(test_loader),total=len(test_loader)):

            inputs.to(device)

            

            outputs1,outputs2,outputs3 = model(inputs.unsqueeze(1).float().cuda())

            predictions.append(outputs3.argmax(1).cpu().detach().numpy())

            predictions.append(outputs2.argmax(1).cpu().detach().numpy())

            predictions.append(outputs1.argmax(1).cpu().detach().numpy())
sample_submission.target = np.hstack(predictions)

sample_submission.to_csv('submission.csv',index=False)
sample_submission['alphabet_part']=sample_submission['row_id'].apply(lambda x: x.split('_')[-2])
sample = sample_submission.groupby(['alphabet_part','target']).count().reset_index()



plt.figure(figsize=(20,7))

ax = sns.barplot(x="row_id", y="target", hue="alphabet_part", data=sample)