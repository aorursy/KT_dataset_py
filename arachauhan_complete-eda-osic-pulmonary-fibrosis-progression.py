import os

from os import listdir

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline



#plotly

!pip install chart_studio

import plotly.express as px

import chart_studio.plotly as py

import plotly.graph_objs as go

from plotly.offline import iplot

import cufflinks

cufflinks.go_offline()

cufflinks.set_config_file(world_readable=True, theme='pearl')



import seaborn as sns

sns.set(style="whitegrid")





#pydicom

import pydicom



# Beautiful plot scheme

plt.style.use('fivethirtyeight')

plt.show()
# List files available

list(os.listdir("../input/osic-pulmonary-fibrosis-progression"))
IMAGE_PATH = "../input/osic-pulmonary-fibrosis-progressiont/"



train_data = pd.read_csv('../input/osic-pulmonary-fibrosis-progression/train.csv')

test_data = pd.read_csv('../input/osic-pulmonary-fibrosis-progression/test.csv')



print('Training data shape: ', train_data.shape)

train_data.head(5)
train_data['SmokingStatus'].value_counts()
train_data.groupby(['SmokingStatus','Sex']).count()
# Lets Explore the  data

print('Train mets data Set !!')

print(train_data.info())

print('Test meta dataSet !!')

print(test_data.info())
# Total number of ecords in the Meta dataset(train+test)

print("Total Patient in Train data set: ",train_data['Patient'].count())

print("Total Patient in Test  data set: ",test_data['Patient'].count())
print("The total patient ids are : ") 

print(train_data['Patient'].count())

print("Total Uniquw unique patients are :") 

print(train_data['Patient'].value_counts().shape[0] )
columns = train_data.keys()

columns = list(columns)

print(columns)
train_data['SmokingStatus'].value_counts()
iplot(train_data.SmokingStatus.iplot(asFigure=True, kind='histogram', title='Smoking Distribution Data', dimensions=(1000,400)))
train_data['Weeks'].value_counts()
grpdata=train_data.groupby(['Weeks']).count()["Patient"]

grpdata
pd.options.plotting.backend = "plotly"

train_data.groupby(['Weeks']).count()["Patient"].plot()


grpval=train_data.groupby(['FVC']).count()["Patient"]

grpval
z=train_data.groupby(['SmokingStatus','Weeks'])['FVC'].count().to_frame().reset_index()

z.style.background_gradient(cmap='Reds') 
train_data.groupby(['FVC']).count()["Patient"].plot()
train_data.groupby(['FVC']).count()["Patient"].plot()



import plotly.express as px

fig = px.line(train_data, x='FVC', y='Percent', color='Sex')

fig.show()
plt.figure(figsize=(16, 6))

sns.kdeplot(train_data.loc[train_data['SmokingStatus'] == 'Ex-smoker', 'FVC'], label = 'Ex-smoker',shade=True)

sns.kdeplot(train_data.loc[train_data['SmokingStatus'] == 'Never smoked', 'FVC'], label = 'Never smoked',shade=True)

sns.kdeplot(train_data.loc[train_data['SmokingStatus'] == 'Currently smokes', 'FVC'], label = 'Currently smokes',shade=True)



# Labeling of plot

plt.xlabel('FVC'); plt.ylabel('Density'); plt.title('Distribution of Gender');
train_data['Percent'].value_counts()
train_data['Percent'].iplot(kind='hist',bins=35,color='green',xTitle='Percent distribution',yTitle='No Of Patients')
plt.figure(figsize=(16, 6))

sns.kdeplot(train_data.loc[train_data['Sex'] == 'Male', 'Percent'], label = 'Male',shade=True)

sns.kdeplot(train_data.loc[train_data['Sex'] == 'Female', 'Percent'], label = 'Female',shade=True)



# Labeling of plot

plt.xlabel('Perecent of FVC '); plt.ylabel('Density'); plt.title('Distribution of Gender for FVC');
train_data['Age'].iplot(kind='hist',bins=30,color='red',xTitle='Age distribution',yTitle='Count')
train_data['SmokingStatus'].value_counts()
plt.figure(figsize=(16, 6))

sns.kdeplot(train_data.loc[train_data['SmokingStatus'] == 'Ex-smoker', 'Age'], label = 'Ex-smoker',shade=True)

sns.kdeplot(train_data.loc[train_data['SmokingStatus'] == 'Never smoked', 'Age'], label = 'Never smoked',shade=True)

sns.kdeplot(train_data.loc[train_data['SmokingStatus'] == 'Currently smokes', 'Age'], label = 'Currently smokes',shade=True)



# Labeling of plot

plt.xlabel('Age (years)'); plt.ylabel('Density'); plt.title('Distribution of Ages');
plt.figure(figsize=(16, 6))

sns.kdeplot(train_data.loc[train_data['Sex'] == 'Female', 'Age'], label = 'Female',shade=True)

sns.kdeplot(train_data.loc[train_data['Sex'] == 'Male', 'Age'], label = 'Male',shade=True)



plt.xlabel('Age (years)'); plt.ylabel('Density'); plt.title('Distribution of Ages');
train_data['Sex'].value_counts()
iplot(train_data.Sex.iplot(asFigure=True, kind='histogram', title='Sex Distribution Data', dimensions=(1000,400)))
plt.figure(figsize=(16, 6))

a = sns.countplot(data=train_data, x='SmokingStatus', hue='Sex')



for p in a.patches:

    a.annotate(format(p.get_height(), ','), 

           (p.get_x() + p.get_width() / 2., 

            p.get_height()), ha = 'center', va = 'center', 

           xytext = (0, 4), textcoords = 'offset points')



plt.title('Gender split by SmokingStatus', fontsize=16)

sns.despine(left=True, bottom=True);
print('Train .dcm number of images:', len(list(os.listdir('../input/osic-pulmonary-fibrosis-progression/train'))), '\n' +

      'Test .dcm number of images:', len(list(os.listdir('../input/osic-pulmonary-fibrosis-progression/test'))), '\n' +

      '--------------------------------', '\n' +

      'There is the same number of images as in train/ test .csv datasets')



filename = "/kaggle/input/osic-pulmonary-fibrosis-progression/train/ID00123637202217151272140/137.dcm"

ds = pydicom.dcmread(filename)

plt.imshow(ds.pixel_array, cmap=plt.cm.bone) 
# directory for a patient

imdir = "/kaggle/input/osic-pulmonary-fibrosis-progression/train/ID00007637202177411956430"

print("total images for patient ID00007637202177411956430: ", len(os.listdir(imdir)))
print("images for patient ID00007637202177411956430 :")

mylist = os.listdir(imdir)

mylist.sort()

print(mylist)
# view first (columns*rows) images in order

w=10

h=10

fig=plt.figure(figsize=(12, 12))

columns = 4

rows = 5

imglist = os.listdir(imdir)

for i in range(1, columns*rows +1):

    filename = imdir + "/" + str(i) + ".dcm"

    ds = pydicom.dcmread(filename)

    fig.add_subplot(rows, columns, i)

    plt.imshow(ds.pixel_array, cmap=plt.cm.bone)

plt.show()
import glob

train_image_path = '../input/osic-pulmonary-fibrosis-progression/train'

train_image_files = glob.glob(os.path.join(train_image_path, '*', '*.dcm'))



train_image_data = pydicom.read_file(train_image_files[0])

train_image_data