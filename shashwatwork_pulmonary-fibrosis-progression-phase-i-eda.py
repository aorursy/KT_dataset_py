import os

!pip install dexplot -q

import dexplot as dxp

from os import listdir

print(list(os.listdir("../input/osic-pulmonary-fibrosis-progression")))

import pandas as pd

import numpy as np

import glob

import tqdm

from typing import Dict

import matplotlib.pyplot as plt

%matplotlib inline



#plotly

import plotly.express as px

import plotly.graph_objs as go

from plotly.offline import iplot

import cufflinks

cufflinks.go_offline()

cufflinks.set_config_file(world_readable=True, theme='pearl')



#color



import seaborn as sns

#sns.set(style="whitegrid")



#pydicom

import pydicom



# Suppress warnings 

import warnings

warnings.filterwarnings('ignore')



# Settings for pretty nice plots

#plt.style.use('fivethirtyeight')

#plt.show()
IMAGE_PATH = "../input/osic-pulmonary-fibrosis-progressiont/"



patient_train_df = pd.read_csv('../input/osic-pulmonary-fibrosis-progression/train.csv')

patient_test_df = pd.read_csv('../input/osic-pulmonary-fibrosis-progression/test.csv')



print('Train data shape:',patient_train_df.shape)

print('Test data shape:',patient_test_df.shape)



print('^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^')

print('Train data:',patient_train_df.info())

print('Test data:',patient_test_df.info())

patient_train_df.head(5)
train_df = patient_train_df[['Patient', 'Age', 'Sex', 'SmokingStatus']].drop_duplicates()
dxp.count('Sex',data=train_df,figsize=(10,6),title='Count of Patient(Sex Wise)')
labels = train_df['Sex'].value_counts()[:10].index

values = train_df['Sex'].value_counts()[:10].values

colors=['#FA8072',

 '#98adbf']



fig = go.Figure(data=[go.Pie(labels=labels, values=values, textinfo='label+percent',

                             insidetextorientation='radial',marker=dict(colors=colors))])

fig.update_layout(title = 'Distribution of Sex for unique patients')

fig.show()
dxp.count('SmokingStatus',data=train_df,figsize=(10,6),title='Count of Patient(Smoker Perspective)',size=0.9)
labels = train_df['SmokingStatus'].value_counts()[:10].index

values = train_df['SmokingStatus'].value_counts()[:10].values

colors=['#FA8072',

 '#98adbf']



fig = go.Figure(data=[go.Pie(labels=labels, values=values, textinfo='label+percent',

                             insidetextorientation='radial',marker=dict(colors=colors))])

fig.update_layout(title = 'Distribution of Smoking Status')

fig.show()
dxp.hist(val='Age', data=train_df,title="Histogram for Age Column",figsize=(10,6))
dxp.hist(val='Age', data=train_df, orientation='v', split='Sex', bins=15,figsize=(10,6),title="Histogram for Age Column(Gender Wise)")
dxp.kde(x='Age', data=train_df, split='SmokingStatus', split_order=['Ex-smoker','Never smoked','Currently smokes'],xlabel='count',figsize=(10,6),title='Histogram for Age')
dxp.bar(x='Sex', y='Age', data=train_df, aggfunc='median', split='SmokingStatus',figsize=(10,6),size=0.9,stacked=True,title='Age Distribution(Smoker Perspective)')
dxp.count(val='Age', data=train_df, split='SmokingStatus',normalize=True,figsize=(10,6),size=0.9,stacked=True,title="Smokers(Patient's Age Perspective)")
pd.crosstab(index=train_df['SmokingStatus'], columns=train_df['Age'])
patient_train_df['Weeks'].iplot(kind='area',fill=True,opacity=1,xTitle='Weeks',yTitle='Days',title='Area Chart of Weeks Column')
import scipy



data = patient_train_df.FVC.tolist()

plt.figure(figsize=(18,6))

# Creating the main histogram

_, bins, _ = plt.hist(data, 15, density=1, alpha=0.5)



# Creating the best fitting line with mean and standard deviation

mu, sigma = scipy.stats.norm.fit(data)

best_fit_line = scipy.stats.norm.pdf(bins, mu, sigma)

plt.plot(bins, best_fit_line, color = 'green', linewidth = 3, label = 'fitting curve')

plt.title(f'FVC Distribution [ mean = {"{:.2f}".format(mu)}, standard_dev = {"{:.2f}".format(sigma)} ]', fontsize = 18)

plt.xlabel('FVC')

plt.show()



patient_train_df['FVC'].iplot(kind='hist',bins=25,color='grey',xTitle='Percent distribution',yTitle='Count')
fig = px.histogram(patient_train_df, x="FVC", y="Age", color="Sex", hover_data=patient_train_df.columns)

fig.update_layout(title='Histogram for FVC w.r.t Patient Gender')

fig.show()
dxp.box(x='Sex', y='FVC', data=patient_train_df,orientation='v',figsize=(10,6),title='Boxplot for FVC')
dxp.box(x='FVC', y='SmokingStatus', data=patient_train_df, 

        split='Sex', split_order=['Male', 'Female'],figsize=(10,6),title='Boxplot for FVC (Gender wise)')
df = patient_train_df.pivot_table(index='SmokingStatus', columns='Age', 

                        values='FVC', aggfunc='mean')

df = df.fillna(0,axis=1)

dxp.bar(data=df, orientation='v',figsize=(10,6),title='FVC Interpretation with Smoking Status')
dxp.scatter(x='Weeks', y='FVC', data=patient_train_df,regression=True,split='Sex',figsize=(10,6),title='Week Distribution(Gender Perspective)')
dxp.scatter(x='Weeks', y='FVC', data=patient_train_df,regression=True,split='SmokingStatus',figsize=(10,6),title='Week Distribution(Somker Perspective)')
dxp.scatter(x='Weeks', y='Percent', data=patient_train_df,regression=True,split='Sex',figsize=(10,6),title='Week Distribution(Sex Perspective) with Percentage')
grp = patient_train_df[['Patient','Percent']].groupby(['Patient'])['Percent'].mean() 

grp = pd.DataFrame(grp).reset_index()

grp = grp[:20]
dxp.bar(x='Patient', y='Percent', data=grp,figsize=(20,6),size=0.9,title='Patients with Mean FCV Percentage ',cmap='jet')
df = patient_train_df.pivot_table(index='Patient', columns='SmokingStatus', 

                        values='Percent', aggfunc='mean')

df = df.fillna(0,axis=1)

df=df[:20]
dxp.bar(data=df, orientation='v',figsize=(20,6),title='Top 10 Patients with High FVC Percentage')
img = "../input/osic-pulmonary-fibrosis-progression/train/ID00010637202177584971671/10.dcm"

ds = pydicom.dcmread(img)

plt.figure(figsize = (10,6))

plt.imshow(ds.pixel_array, cmap='gray')
image_dir = '../input/osic-pulmonary-fibrosis-progression/train/ID00027637202179689871102'



fig=plt.figure(figsize=(10,10))

columns = 5

rows = 6

image_list = os.listdir(image_dir)

for i in range(1, columns*rows +1):

    filename = image_dir + "/" + str(i) + ".dcm"

    ds = pydicom.dcmread(filename)

    fig.add_subplot(rows, columns, i)

    plt.imshow(ds.pixel_array, cmap='gray')