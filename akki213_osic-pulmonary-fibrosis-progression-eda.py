import os
from os import listdir
import pandas as pd
import numpy as np
import glob
import tqdm
from typing import Dict
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

#color
from colorama import Fore, Back, Style

import seaborn as sns
sns.set(style="whitegrid")

#pydicom
import pydicom

# Suppress warnings 
import warnings
warnings.filterwarnings('ignore')

#Sklearn
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Settings for pretty nice plots
plt.style.use('fivethirtyeight')
plt.show()
IMAGE_PATH='../input/osic-pulmonary-fibrosis-progression'
train=pd.read_csv('../input/osic-pulmonary-fibrosis-progression/train.csv')
test=pd.read_csv('../input/osic-pulmonary-fibrosis-progression/test.csv')

train.head()
train.shape
train['Patient'].nunique()
train.describe()
test.head()
print(Fore.BLUE+'Info about training set:',Style.RESET_ALL)
print(train.info())
print(Fore.YELLOW+'Info about testing set:',Style.RESET_ALL)
print(test.info())
print(Fore.BLUE+'Total number of patient entries are',Style.RESET_ALL,f"{train.shape[0]},",Fore.YELLOW+'whereas total number of unique patients are',Style.RESET_ALL,f"{train['Patient'].nunique()}.")
print(Fore.BLUE+'Total number of patient entries in training set are',Style.RESET_ALL,f"{train.shape[0]},",Fore.YELLOW+'whereas total number of patient entries in testing set are',Style.RESET_ALL,f"{test.shape[0]}.")
s_train=set(train['Patient'])
s_test=set(test['Patient'])
s_train.intersection(s_test)
train.isnull().sum()
test.isnull().sum()
train['Sex'].value_counts()
test['Sex'].value_counts()
train['Age'].describe()
train['SmokingStatus'].value_counts()
#Creating a dataset consisting of only one record per patient
train_dir = '../input/osic-pulmonary-fibrosis-progression/train/'
test_dir = '../input/osic-pulmonary-fibrosis-progression/test/'

patient_ids = os.listdir(train_dir)
patient_ids = sorted(patient_ids)

#Creating new rows
no_of_instances = []
age = []
sex = []
smoking_status = []
mean_FVC=[]
rec_checkup_weekno=[]
first_checkup_weekno=[]
min_FVC=[]
max_FVC=[]
for patient_id in patient_ids:
    patient_info = train[train['Patient'] == patient_id].reset_index()
    no_of_instances.append(len(os.listdir(train_dir + patient_id)))
    age.append(patient_info['Age'][0])
    sex.append(patient_info['Sex'][0])
    mean_FVC.append(round(patient_info['FVC'].mean()))
    min_FVC.append(patient_info['FVC'].min())
    max_FVC.append(patient_info['FVC'].max())
    rec_checkup_weekno.append(patient_info['Weeks'].max())
    first_checkup_weekno.append(patient_info['Weeks'].min())
    smoking_status.append(patient_info['SmokingStatus'][0])

#Creating the dataframe for the patient info    
patient_df = pd.DataFrame(list(zip(patient_ids, no_of_instances, age, sex,mean_FVC,min_FVC,max_FVC,rec_checkup_weekno,first_checkup_weekno, smoking_status)), 
                                 columns =['Patient', 'no_of_instances', 'Age', 'Sex','Mean FVC','Min FVC','Max FVC','Recent Checkup Week','First Checkup Week','SmokingStatus'])
print(patient_df.info())
patient_df.head()
import scipy

data = patient_df.Age.tolist()
plt.figure(figsize=(18,6))
_, bins, _ = plt.hist(data, 45, density=1, alpha=0.5)
mu, sigma = scipy.stats.norm.fit(data)
best_fit_line = scipy.stats.norm.pdf(bins, mu, sigma)
plt.plot(bins, best_fit_line, color = 'b', linewidth = 3, label = 'fitting curve')
plt.title(f'Age Distribution [ mean = {"{:.2f}".format(mu)}, standard_dev = {"{:.2f}".format(sigma)} ]', fontsize = 18)
plt.xlabel('Age -->')
plt.show()
patient_df['Age'].iplot(kind='hist',bins=45,color='blue',xTitle='Age Distribution',yTitle='Count')
fig = px.scatter(patient_df, x="Age", y="Mean FVC", color='SmokingStatus')
fig.show()
fig = px.scatter(patient_df, x="Age", color='Sex',hover_data=['Mean FVC','Recent Checkup Week','SmokingStatus'])
fig.show()
data = train.FVC.tolist()
plt.figure(figsize=(18,6))
_, bins, _ = plt.hist(data, 45, density=1, alpha=0.5)
mu, sigma = scipy.stats.norm.fit(data)
best_fit_line = scipy.stats.norm.pdf(bins, mu, sigma)
plt.plot(bins, best_fit_line, color = 'b', linewidth = 3, label = 'fitting curve')
plt.title(f'FVC Distribution [ mean = {"{:.2f}".format(mu)}, standard_dev = {"{:.2f}".format(sigma)} ]', fontsize = 18)
plt.xlabel('FVC -->')
plt.show()



train['FVC'].iplot(kind='hist',
                      xTitle='Lung Capacity(ml)', 
                      linecolor='black', 
                      opacity=0.8,
                      color='blue',
                      bargap=0.5,
                      gridcolor='white',
                      title='Distribution of FVC (On whole dataset)')
patient_df.sort_values(by='no_of_instances',ascending=False)
train_ID00078637202199415319443=train[train['Patient']=='ID00078637202199415319443']
train_ID00078637202199415319443
fig = px.line(train_ID00078637202199415319443, x="Weeks", y="FVC", title='FVC really increasing?')
fig.show()
train_all=patient_df.sort_values(by='no_of_instances',ascending=False).head(10)
train_all
l=list(train_all['Patient'].values)
train[train['Patient']==(i for i in l)]
df= pd.DataFrame(columns=['Patient','Weeks','FVC','Percent','Age','Sex','SmokingStatus'])
df
for i in l:
    t=train[train['Patient']==i]
    frames=[df,t]
    df=pd.concat(frames)
df
fig = px.line(df, x="Weeks", y="FVC", color='Patient')
fig.show()
patient_df['SmokingStatus'].value_counts()
patient_df['SmokingStatus'].value_counts().iplot(kind='bar',
                                              yTitle='Counts', 
                                              linecolor='black', 
                                              opacity=0.7,
                                              color='red',
                                              theme='pearl',
                                              bargap=0.5,
                                              gridcolor='white',
                                              title='Distribution of the SmokingStatus of the patients')
patient_df[['SmokingStatus','Sex']].value_counts().iplot(kind='bar',
                                              yTitle='Counts', 
                                              linecolor='black', 
                                              opacity=0.7,
                                              color='blue',
                                              theme='pearl',
                                              bargap=0.5,
                                              gridcolor='white',
                                              title='Distribution of the SmokingStatus of the patients along with their gender')
data = train.Percent.tolist()
plt.figure(figsize=(18,6))
_, bins, _ = plt.hist(data, 45, density=1, alpha=0.5)
mu, sigma = scipy.stats.norm.fit(data)
best_fit_line = scipy.stats.norm.pdf(bins, mu, sigma)
plt.plot(bins, best_fit_line, color = 'b', linewidth = 3, label = 'fitting curve')
plt.title(f'Percent Distribution [ mean = {"{:.2f}".format(mu)}, standard_dev = {"{:.2f}".format(sigma)} ]', fontsize = 18)
plt.xlabel('Percent -->')
plt.show()

train['Percent'].iplot(kind='hist',bins=30,color='blue',xTitle='Percent distribution',yTitle='Count')
dfFPA=train[['FVC','Percent','Age']].corr()
dfFPA
fig,ax =plt.subplots(figsize=(12,7))
title='FVC vs Percent vs Age'
plt.title(title,fontsize=18)



sns.heatmap(dfFPA,annot=True)
plt.show()
df = train[['FVC','Percent']]
X = df.FVC.values.reshape(-1, 1)

model = LinearRegression()
model.fit(X, df.Percent)

x_range = np.linspace(X.min(), X.max(), 100)
y_range = model.predict(x_range.reshape(-1, 1))

fig = px.scatter(train, x='FVC', y='Percent', color='Age', opacity=0.65)
fig.add_traces(go.Scatter(x=x_range, y=y_range, name='Regression Fit'))
fig.show()
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

df = train[['FVC','Percent']]
X = df.FVC.values.reshape(-1, 1)
X_train, X_test, y_train, y_test = train_test_split(X, df.Percent, random_state=0)

model1 = LinearRegression()
model1.fit(X_train, y_train)

x_range = np.linspace(X.min(), X.max(), 100)
y_range = model1.predict(x_range.reshape(-1, 1))


fig = go.Figure([
    go.Scatter(x=X_train.squeeze(), y=y_train, name='train', mode='markers'),
    go.Scatter(x=X_test.squeeze(), y=y_test, name='test', mode='markers'),
    go.Scatter(x=x_range, y=y_range, name='prediction')
])
fig.update_layout(xaxis_title="FVC",
    yaxis_title="Percent", 
    title="Generalized Regression fit",
)
fig.show()
dfFPW=train[['FVC','Percent','Weeks']].corr()

fig,ax =plt.subplots(figsize=(12,7))
title='FVC vs Percent vs Weeks'
plt.title(title,fontsize=18)



sns.heatmap(dfFPW,cmap='RdYlGn',annot=True)
plt.show()
df = train
fig = px.violin(df, y='Percent', x='SmokingStatus', box=True, color='Sex', points="all",
          hover_data=train.columns)
fig.show()

fig = px.bar(train, x='Age', y='Percent',
             color='FVC',
             height=400)
fig.show()
