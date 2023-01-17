import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

import os

from matplotlib import style



%matplotlib inline

style.use('fivethirtyeight')
dir_train_dicom = '../input/osic-pulmonary-fibrosis-progression/train'

dir_test_dicom = '../input/osic-pulmonary-fibrosis-progression/test'



dir_train_csv = '../input/osic-pulmonary-fibrosis-progression/train.csv'

dir_test_csv = '../input/osic-pulmonary-fibrosis-progression/test.csv'



dir_submission_csv = '../input/osic-pulmonary-fibrosis-progression/sample_submission.csv'
train = pd.read_csv(dir_train_csv)

test = pd.read_csv(dir_test_csv)

sample_submission = pd.read_csv(dir_submission_csv)
train
test
sample_submission
train.isnull().sum()
train.Patient.nunique()
train.describe()
sns.distplot(train.FVC)
sns.distplot(train.Age, color='g')
sns.distplot(train.Weeks, color='r')
f, axes = plt.subplots(4, figsize=(15, 15))

sns.boxplot(train.FVC, ax=axes[0])

sns.boxplot(train.Age, color='g', ax=axes[1])

sns.boxplot(train.Percent, color='y', ax=axes[2])

sns.boxplot(train.Weeks, color='r', ax=axes[3])
sns.pairplot(train, hue='Sex')
sns.countplot(x=train.Sex)
Porcentaje_hombres = train.Sex[train.Sex=='Male'].count()/len(train.Sex)

Porcentaje_mujeres = train.Sex[train.Sex=='Female'].count()/len(train.Sex)



print(f'Porcentaje train HOMBRES: {Porcentaje_hombres}')

print(f'Porcentaje train MUJERES: {Porcentaje_mujeres}')
Porcentaje_hombres = test.Sex[train.Sex=='Male'].count()/len(test.Sex)

Porcentaje_mujeres = test.Sex[train.Sex=='Female'].count()/len(test.Sex)



print(f'Porcentaje test HOMBRES: {Porcentaje_hombres}')

print(f'Porcentaje test MUJERES: {Porcentaje_mujeres}')
sns.countplot(train.SmokingStatus)
Porcentaje_exfumadores = train.SmokingStatus[train.SmokingStatus=='Ex-smoker'].count()/len(train.SmokingStatus)

Porcentaje_nuncafumaron = train.SmokingStatus[train.SmokingStatus=='Never smoked'].count()/len(train.SmokingStatus)

Porcentaje_fuman = train.SmokingStatus[train.SmokingStatus=='Currently smokes'].count()/len(train.SmokingStatus)



print(f'Porcentaje train EX-FUMADORES: {Porcentaje_exfumadores}')

print(f'Porcentaje train NUNCA FUMARON: {Porcentaje_nuncafumaron}')

print(f'Porcentaje train FUMAN: {Porcentaje_fuman}')
Porcentaje_exfumadores = test.SmokingStatus[test.SmokingStatus=='Ex-smoker'].count()/len(test.SmokingStatus)

Porcentaje_nuncafumaron = test.SmokingStatus[test.SmokingStatus=='Never smoked'].count()/len(test.SmokingStatus)

Porcentaje_fuman = test.SmokingStatus[test.SmokingStatus=='Currently smokes'].count()/len(test.SmokingStatus)



print(f'Porcentaje test EX-FUMADORES: {Porcentaje_exfumadores}')

print(f'Porcentaje test NUNCA FUMARON: {Porcentaje_nuncafumaron}')

print(f'Porcentaje test FUMAN: {Porcentaje_fuman}')
sns.pairplot(train, hue='SmokingStatus')
sns.heatmap(train.corr(), annot=True, fmt='.2f', cmap='YlGnBu')
train.Sex[train.Sex=='Female'].groupby(train.Age).value_counts()
sns.distplot(train.Age[train.Sex=='Female'].values)
train.Age[train.Sex=='Female'].describe()
train.Sex[train.Sex=='Male'].groupby(train.Age).value_counts()
sns.distplot(train.Age[train.Sex=='Male'].values)
train.Age[train.Sex=='Male'].describe()
sns.set(rc={'figure.figsize':(15,10)})



sns.distplot(train[train.SmokingStatus == 'Ex-smoker'].FVC)

sns.distplot(train[train.SmokingStatus =='Never smoked'].FVC)

sns.distplot(train[train.SmokingStatus =='Currently smokes'].FVC)



plt.legend(['Ex-smoker','Never smoked','Currently smokes'])
sns.distplot(train[train.Sex =='Female'].FVC)

sns.distplot(train[train.Sex =='Male'].FVC)



plt.legend(['Female','Male'])
print(train.Weeks.max(), train.Weeks.min())

print(np.linspace(train.Weeks.max(), train.Weeks.min(), 6))
sns.distplot(train[train.Weeks<=23].FVC)

sns.distplot(train[(train.Weeks>23)&(train.Weeks<=50)].FVC)

sns.distplot(train[(train.Weeks>50)&(train.Weeks<=78)].FVC)

sns.distplot(train[(train.Weeks>78)&(train.Weeks<=105)].FVC)

sns.distplot(train[(train.Weeks>105)&(train.Weeks<=133)].FVC)



plt.legend(['<23 weeks','23 y 50 weeks','50 y 78 weeks', '78 y 105 weeks', '105 y 133 weeks'])
train.Weeks_cortadas = pd.cut(train["Weeks"], 8)   

grupo = train.groupby(train.Weeks_cortadas)



ax = sns.barplot(x=train.Weeks_cortadas.unique(), y= grupo.FVC.mean())

plt.xticks(rotation=45)



for rect in ax.patches:

    height = rect.get_height()

    ax.text(rect.get_x() + rect.get_width()/2., 1*height,'%d' % int(height),ha='center', va='bottom')
print(train.Age.max(), train.Age.min())

print(np.linspace(train.Age.max(), train.Age.min(), 4))
sns.distplot(train[train.Age<=62].FVC)

sns.distplot(train[(train.Age>62)&(train.Age<=75)].FVC)

sns.distplot(train[(train.Age>75)&(train.Age<=88)].FVC)



plt.legend(['<62 age','62 y 75 age','75 y 88 age'])
train.Age_cortadas = pd.cut(train["Age"], 6)   

grupo = train.groupby(train.Age_cortadas)



ax = sns.barplot(x=train.Age_cortadas.unique(), y= grupo.FVC.mean())

plt.xticks(rotation=45)



for rect in ax.patches:

    height = rect.get_height()

    ax.text(rect.get_x() + rect.get_width()/2., 1*height,'%d' % int(height),ha='center', va='bottom')
train.groupby(train.Patient).count().Weeks.value_counts()
f, axes = plt.subplots(1,3, figsize=(20, 10))



grupo = train.groupby(train.Sex)



ax0 = sns.barplot(x=train.Sex.unique(), y= grupo.FVC.min(), ax=axes[0])

ax1 = sns.barplot(x=train.Sex.unique(), y= grupo.FVC.mean(), ax=axes[1])

ax2 = sns.barplot(x=train.Sex.unique(), y= grupo.FVC.max(), ax=axes[2])



for rect in ax0.patches:

    height = rect.get_height()

    ax0.text(rect.get_x() + rect.get_width()/2., 1*height,'%d' % int(height),ha='center', va='bottom')

    

for rect in ax1.patches:

    height = rect.get_height()

    ax1.text(rect.get_x() + rect.get_width()/2., 1*height,'%d' % int(height),ha='center', va='bottom')



for rect in ax2.patches:

    height = rect.get_height()

    ax2.text(rect.get_x() + rect.get_width()/2., 1*height,'%d' % int(height),ha='center', va='bottom')
ax = sns.barplot(x=train.SmokingStatus, y=train.FVC)



for rect in ax.patches:

    height = rect.get_height()

    ax.text(rect.get_x() + rect.get_width()/2., 1.05*height,'%d' % int(height),ha='center', va='bottom')
train.Age[train.Sex=='Male'].max(), train.Age[train.Sex=='Male'].min()
train.Age[train.Sex=='Female'].max(), train.Age[train.Sex=='Female'].min()
train[(train.Sex=='Male')&(train.FVC>5960)]
train[(train.Sex=='Female')&(train.FVC>4890)]
Paciente_0 = train.Patient[0]

Paciente = train[train.Patient==Paciente_0]

sns.lineplot(x=Paciente.Weeks, y=Paciente.FVC), 

plt.title(Paciente_0)
Paciente_10 = train.Patient[10]

Paciente = train[train.Patient==Paciente_10]

sns.lineplot(x=Paciente.Weeks, y=Paciente.FVC), 

plt.title(Paciente_10)
Paciente_20 = train.Patient[20]

Paciente = train[train.Patient==Paciente_20]

sns.lineplot(x=Paciente.Weeks, y=Paciente.FVC), 

plt.title(Paciente_20)
!pip install git+https://github.com/fastai/fastai2 

!pip install git+https://github.com/fastai/fastcore
import pydicom

from fastai2.basics import *

from fastai2.callback.all import *

from fastai2.vision.all import *

from fastai2.medical.imaging import *
train_dcm = get_dicom_files(dir_train_dicom)

train_dcm
dcm_random = train_dcm[6]

dimg = dcmread(dcm_random)

print(f'El paciente random es: {dcm_random}\n') 

print(dimg)
dimension = (int(dimg.Rows), int(dimg.Columns), len(dimg.PixelData))

dimension
dimg.show(figsize=(8,8))
pix = dimg.pixels.flatten()

sns.distplot(pix, kde=False)
dimg.show(max_px=-500, min_px=-1200, figsize=(8,8))
from skimage.measure import label,regionprops

from skimage.segmentation import clear_border
img = clear_border(dimg)

plt.figure(figsize=(10,10))

plt.imshow(img, cmap=plt.cm.bone)

plt.axis('off')