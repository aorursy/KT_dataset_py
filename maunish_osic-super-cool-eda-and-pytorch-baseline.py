import os

import sys

import glob

from os import listdir

import glob

import tqdm

from typing import Dict

import cv2

import pydicom as dicom





import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

plt.style.use('fivethirtyeight')



#plotly

# !pip install chart_studio

import plotly.express as px

# import chart_studio.plotly as py

import plotly.graph_objs as go

from plotly.offline import iplot

import plotly.figure_factory as ff

import cufflinks

cufflinks.go_offline()

cufflinks.set_config_file(world_readable=True,theme='pearl')



#pydicom

import pydicom



#supress warnings

import warnings

warnings.filterwarnings('ignore')





# from bokeh.layouts import row, column

# from bokeh.models import ColumnDataSource, CustomJS, Label,Range1d,Slider,Span

# from bokeh.plotting import figure, output_notebook, show



#used for changing color of text in print statement

from colorama import Fore, Back, Style

y_ = Fore.YELLOW

r_ = Fore.RED

g_ = Fore.GREEN

b_ = Fore.BLUE

m_ = Fore.MAGENTA

sr_ = Style.RESET_ALL



# output_notebook()
folder_path = '../input/osic-pulmonary-fibrosis-progression'

train_csv = folder_path + '/train.csv'

test_csv = folder_path+ '/test.csv'

sample_csv = folder_path + '/sample_submission.csv'



train_data = pd.read_csv(train_csv)

test_data = pd.read_csv(test_csv)

sample = pd.read_csv(sample_csv)



print(f"{y_}Number of rows in train data: {r_}{train_data.shape[0]}\n{y_}Number of columns in train data: {r_}{train_data.shape[1]}")

print(f"{g_}Number of rows in test data: {r_}{test_data.shape[0]}\n{g_}Number of columns in test data: {r_}{test_data.shape[1]}")

print(f"{b_}Number of rows in submission data: {r_}{sample.shape[0]}\n{b_}Number of columns in submission data:{r_}{sample.shape[1]}")



train_data.head().style.applymap(lambda x: 'background-color:lightgreen')
def distribution(feature, color):

    plt.figure(dpi=100)

    sns.distplot(train_data[feature],color=color)

    print("{}Max value of {} is: {} {:.2f} \n{}Min value of {} is: {} {:.2f}\n{}Mean of {} is: {}{:.2f}\n{}Standard Deviation of {} is:{}{:.2f}"\

      .format(y_,feature,r_,train_data[feature].max(),g_,feature,r_,train_data[feature].min(),b_,feature,r_,train_data[feature].mean(),m_,feature,r_,train_data[feature].std()))
distribution("FVC","blue")
distribution("Age","brown")
distribution("Percent","blue")
distribution("Weeks","yellow")
plt.figure(dpi=100)

sns.countplot(data=train_data,x='SmokingStatus',hue='Sex');
def distribution2(feature):

    plt.figure(figsize=(15,7))

    plt.subplot(121)

    for i in train_data.Sex.unique():

        sns.distplot(train_data[train_data['Sex']==i][feature],label=i)

    plt.title(f"Distribution of {feature} based on Sex")

    plt.legend()



    plt.subplot(122)

    for i in train_data.SmokingStatus.unique():

        sns.distplot(train_data[train_data['SmokingStatus']==i][feature],label=i)

    plt.title(f"Distribution of {feature}  based on Smoking Status")

    plt.legend()

distribution2("FVC")
distribution2("Percent")
distribution2("Age")
distribution2("Weeks")
def vs(feature1,feature2,color=None):

    fig = px.scatter(train_data,x=feature1,y=feature2,color=color)

    fig.show()
vs("FVC","Percent",'SmokingStatus')
vs("FVC","Age",'SmokingStatus')
vs("FVC","Weeks","SmokingStatus")
rn = np.random.randint(0,train_data.Patient.nunique()-20,1)[0]

patients_ids = train_data.Patient.unique()[rn:rn+20]

fig =go.Figure()



for patient in patients_ids:

    df = train_data[train_data["Patient"] == patient]

    fig.add_trace(go.Scatter(x=df.Weeks,y=df.FVC,

                            mode='lines',

                            name=str(patient)))

fig.show()
print(f"{y_}Number of unique patient is {r_}{train_data.Patient.nunique()}")



df = train_data.Patient.value_counts()

fig = px.bar(x=[f"Patient {i}" for i in range(len(df.index))],y=df.values)

fig.show()
def box(feature1,feature2,color=None):

    fig = px.box(train_data,x=feature2,y=feature1,color=color)

    fig.show()
box("FVC","Sex","SmokingStatus")
box("Percent","Sex","SmokingStatus")
box("Age","Sex","SmokingStatus")
plt.figure(dpi=100)

sns.heatmap(train_data.corr(),annot=True);
train_image_path = folder_path + '/train/'

test_image_path = folder_path + '/test/'



train_images = os.listdir(train_image_path)

test_images = os.listdir(test_image_path)



image = train_image_path+train_images[0]+"/1.dcm"



def show_image(image):

    print(f"{y_} Image {r_}{image}")

    image = dicom.dcmread(image)

    image = image.pixel_array    

    plt.figure(figsize=(7,7))

    plt.imshow(image,cmap='gray')

    plt.axis('off')

    plt.show()



show_image(image)



    
def show_grid(cmap='gray'):

    rn = np.random.randint(0,len(train_images),1)[0]

    path= train_image_path+train_images[rn]

    images = [dicom.read_file(path+"/"+img) for img in os.listdir(path)]

    images.sort(key=lambda x: float(x.ImagePositionPatient[2]))

    plt.figure(figsize=(10,10))

    for i,image in enumerate(images[:100]):

        plt.subplot(10,10,i+1)

        plt.imshow(image.pixel_array,cmap=cmap)

        plt.axis('off')

    plt.show()



show_grid()
show_grid(cmap='jet')
show_grid(cmap='RdYlBu')
import matplotlib.animation as animation

from IPython.display import HTML



def show_animation():

    rn = np.random.randint(0,len(train_images),1)[0]

    fig = plt.figure()

    path= train_image_path+train_images[0]

    images = [dicom.read_file(path+"/"+img) for img in os.listdir(path)]

    images.sort(key=lambda x: float(x.ImagePositionPatient[2]))

    ims = list()

    for image in images:

        image = plt.imshow(image.pixel_array,cmap='gray',animated=True)

        plt.axis('off')

        ims.append([image])

    ani = animation.ArtistAnimation(fig,ims,interval=100,blit=False,repeat_delay=1000)

    return ani



ani = show_animation()
HTML(ani.to_jshtml())
# load the DICOM files

def reconstruct():

    files = []

    path= train_image_path+train_images[0]



    for fname in os.listdir(path):

    #     print("loading: {}".format(fname))

        files.append(pydicom.dcmread(path+"/"+fname))



    print("file count: {}".format(len(files)))



    # skip files with no SliceLocation (eg scout views)

    slices = []

    skipcount = 0

    for f in files:

        if hasattr(f, 'SliceLocation'):

            slices.append(f)

        else:

            skipcount = skipcount + 1



    print("skipped, no SliceLocation: {}".format(skipcount))



    # ensure they are in the correct order

    slices = sorted(slices, key=lambda s: s.SliceLocation)



    # pixel aspects, assuming all slices are the same

    ps = slices[0].PixelSpacing

    ss = slices[0].SliceThickness

    ax_aspect = ps[1]/ps[0]

    sag_aspect = ps[1]/ss

    cor_aspect = ss/ps[0]



    # create 3D array

    img_shape = list(slices[0].pixel_array.shape)

    img_shape.append(len(slices))

    img3d = np.zeros(img_shape)



    # fill 3D array with the images from the files

    for i, s in enumerate(slices):

        img2d = s.pixel_array

        img3d[:, :, i] = img2d



    # plot 3 orthogonal slices

    plt.figure(figsize=(15,7))

    a1 = plt.subplot(1,3,1)

    plt.imshow(img3d[:, :, img_shape[2]//2])

    a1.set_aspect(ax_aspect)

    plt.axis('off')





    a2 = plt.subplot(1, 3, 2)

    plt.imshow(img3d[:, img_shape[1]//2, :])

    a2.set_aspect(sag_aspect)

    plt.axis('off')





    a3 = plt.subplot(1, 3, 3)

    plt.imshow(img3d[img_shape[0]//2, :, :].T)

    a3.set_aspect(cor_aspect)

    plt.axis('off')

    plt.show()
reconstruct()
from sklearn.preprocessing import StandardScaler,LabelEncoder,OneHotEncoder,PowerTransformer

from sklearn.model_selection import train_test_split, cross_val_score,cross_validate, KFold,GroupKFold

from sklearn.metrics import make_scorer

from sklearn.compose import ColumnTransformer

from sklearn.pipeline import Pipeline
#getting base week for patient

def get_baseline_week(data):

    df = data.copy()

    df['Weeks'] = df['Weeks'].astype(int)

    df['min_week'] = df.groupby('Patient')['Weeks'].transform('min')

    df['baseline_week'] = df['Weeks'] - df['min_week']

    return df



#getting FVC for base week and setting it as base_FVC of patient

def get_base_FVC(data):

    df = data.copy()

    base = df.loc[df.Weeks == df.min_week][['Patient','FVC']].copy()

    base.columns = ['Patient','base_FVC']

    

    base['nb']=1

    base['nb'] = base.groupby('Patient')['nb'].transform('cumsum')

    

    base = base[base.nb==1]

    base.drop('nb',axis =1,inplace=True)

    df = df.merge(base,on="Patient",how='left')

    df.drop(['min_week'], axis = 1)

    return df 
train_data.drop_duplicates(keep=False,inplace=True,subset=['Patient','Weeks'])

train_data = get_baseline_week(train_data)

train_data = get_base_FVC(train_data)



sample = pd.read_csv(sample_csv)

sample.drop("FVC",axis=1,inplace=True)

sample[["Patient","Weeks"]] = sample["Patient_Week"].str.split("_",expand=True) 

sample = sample.merge(test_data.drop("Weeks",axis=1),on="Patient",how="left")



#we have to predict for all weeks 

sample["min_Weeks"] = np.nan

sample = get_baseline_week(sample)

sample = get_base_FVC(sample)



train_columns = ['baseline_week','base_FVC','Percent','Age','Sex','SmokingStatus']

train_label = ['FVC']

sub_columns = ['Patient_Week','FVC','Confidence']



train = train_data[train_columns]

test = sample[train_columns]

#Preprocessing

transformer = ColumnTransformer([('s',StandardScaler(),[0,1,2,3]),('o',OneHotEncoder(),[4,5])])

target = train_data[train_label].values

train = transformer.fit_transform(train)

test = transformer.transform(test)
train_data.head()
distribution("baseline_week",'green');
import torch

import torch.nn as nn

import torch.nn.functional as F

import torch.optim as optim

from torch.utils.data import TensorDataset, DataLoader
class Model(nn.Module):

    def __init__(self,n):

        super(Model,self).__init__()

        self.layer1 = nn.Linear(n,200)

        self.layer2 = nn.Linear(200,100)

        

        self.out1 = nn.Linear(100,3)

        self.relu3 = nn.ReLU()

        self.out2 = nn.Linear(100,3)

            

    def forward(self,xb):

        x1 =  F.leaky_relu(self.layer1(xb))

        x1 =  F.leaky_relu(self.layer2(x1))

        

        o1 = self.out1(x1)

        o2 = F.relu(self.out2(x1))

        return o1 + torch.cumsum(o2,dim=1)

        
def run():

    

    def score(outputs,target):

        confidence = outputs[:,2] - outputs[:,0]

        clip = torch.clamp(confidence,min=70)

        target=torch.reshape(target,outputs[:,1].shape)

        delta = torch.abs(outputs[:,1] - target)

        delta = torch.clamp(delta,max=1000)

        sqrt_2 = torch.sqrt(torch.tensor([2.])).to(device)

        metrics = (delta*sqrt_2/clip) + torch.log(clip*sqrt_2)

        return torch.mean(metrics)

    

    def qloss(outputs,target):

        qs = [0.25,0.5,0.75]

        qs = torch.tensor(qs,dtype=torch.float).to(device)

        e =  target - outputs

        e.to(device)

        v = torch.max(qs*e,(qs-1)*e)

        v = torch.sum(v,dim=1)

        return torch.mean(v)

    

    def loss_fn(outputs,target,l):

        return l * qloss(outputs,target) + (1- l) * score(outputs,target)

        

    def train_loop(train_loader,model,loss_fn,device,optimizer,lr_scheduler=None):

        model.train()

        losses = list()

        metrics = list()

        for i, (inputs, labels) in enumerate(train_loader):

            inputs = inputs.to(device)

            labels = labels.to(device)

            

            optimizer.zero_grad()

            with torch.set_grad_enabled(True):           

                outputs = model(inputs)                 

                metric = score(outputs,labels)



                loss = loss_fn(outputs,labels,0.8)

                metrics.append(metric.cpu().detach().numpy())

                losses.append(loss.cpu().detach().numpy())



                loss.backward()



                optimizer.step()

                if lr_scheduler != None:

                    lr_scheduler.step()

            

        return losses,metrics

    

    def valid_loop(valid_loader,model,loss_fn,device):

        model.eval()

        losses = list()

        metrics = list()

        for i, (inputs, labels) in enumerate(valid_loader):

            inputs = inputs.to(device)

            labels = labels.to(device)

            

            outputs = model(inputs)                 

            metric = score(outputs,labels)

            

            loss = loss_fn(outputs,labels,0.8)

            metrics.append(metric.cpu().detach().numpy())

            losses.append(loss.cpu().detach().numpy())

            

        return losses,metrics    



    NFOLDS =5

    kfold = KFold(NFOLDS,shuffle=True,random_state=42)

    

    #kfold

    for k , (train_idx,valid_idx) in enumerate(kfold.split(train)):

        batch_size = 64

        epochs = 50

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        print(f"{device} is used")

        x_train,x_valid,y_train,y_valid = train[train_idx,:],train[valid_idx,:],target[train_idx],target[valid_idx]

        n = x_train.shape[1]

        model = Model(n)

        model.to(device)

        lr = 0.1

        optimizer = optim.Adam(model.parameters(),lr=lr)

        lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)



        train_tensor = torch.tensor(x_train,dtype=torch.float)

        y_train_tensor = torch.tensor(y_train,dtype=torch.float)



        train_ds = TensorDataset(train_tensor,y_train_tensor)

        train_dl = DataLoader(train_ds,

                             batch_size = batch_size,

                             num_workers=4,

                             shuffle=True

                             )



        valid_tensor = torch.tensor(x_valid,dtype=torch.float)

        y_valid_tensor = torch.tensor(y_valid,dtype=torch.float)



        valid_ds = TensorDataset(valid_tensor,y_valid_tensor)

        valid_dl = DataLoader(valid_ds,

                             batch_size = batch_size,

                             num_workers=4,

                             shuffle=False

                             )

        

        print(f"Fold {k}")

        for i in range(epochs):

            losses,metrics = train_loop(train_dl,model,loss_fn,device,optimizer,lr_scheduler)

            valid_losses,valid_metrics = valid_loop(valid_dl,model,loss_fn,device)

            if (i+1)%5==0:

                print(f"epoch:{i} Training | loss:{np.mean(losses)} score: {np.mean(metrics)}| \n Validation | loss:{np.mean(valid_losses)} score:{np.mean(valid_metrics)}|")

        torch.save(model.state_dict(),f'model{k}.bin')

    
run()
def inference():

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    nfold = 5

    all_prediction = np.zeros((test.shape[0],3))

    

    for i in range(nfold):

        n = train.shape[1]

        

        model = Model(n)

        model.load_state_dict(torch.load(f"model{i}.bin"))

        predictions = list()

        model.to(device)

        test_tensor = torch.tensor(test,dtype=torch.float)

        test_dl = DataLoader(test_tensor,

                        batch_size=64,

                        num_workers=2,

                        shuffle=False)

    

        with torch.no_grad():

            for i, inputs in enumerate(test_dl):

                inputs = inputs.to(device, dtype=torch.float)

                outputs= model(inputs) 

                predictions.extend(outputs.cpu().detach().numpy())



        all_prediction += np.array(predictions)/nfold

        

    return all_prediction  
prediction = inference()

sample["Confidence"] = np.abs(prediction[:,2] - prediction[:,0])

sample["FVC"] = prediction[:,1]

sub = sample[sub_columns]
sub.to_csv("submission.csv",index=False)
plt.figure(figsize=(15,7))

plt.subplot(121)

sns.distplot(sub.Confidence)

plt.subplot(122)

sns.distplot(sub.FVC);
print(sub.shape)

sub.head()
sub.to_csv("submission.csv",index=False)