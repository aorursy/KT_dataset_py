#Importing essential libraries



import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline
#Reading CSV file

df = pd.read_csv('/kaggle/input/jobs-on-naukricom/home/sdf/marketing_sample_for_naukri_com-jobs__20190701_20190830__30k_data.csv')
df.head()
#Overall information about the dataset

df.info()
#This shows the unique value,total count, top values and top value frequency per column

df.describe()
plt.figure(figsize=(12,8))

sns.heatmap(df.isnull(), yticklabels=False, cbar=False, cmap='viridis')
import cufflinks as cf

cf.go_offline()
df['Industry'].value_counts()[0:9].iplot(kind='bar')
df['Location'].value_counts()[0:9].iplot(kind='bar')
softwarejobs = df[df['Industry'] == 'IT-Software, Software Services']

swjobs_locationCount = softwarejobs['Location'].value_counts()

swjobs_locationCount[0:9].iplot(kind='bar',colors='red')
df['Job Salary'].value_counts()[0:9].iplot(kind='bar')
df['Job Experience Required'].value_counts()[0:9].iplot(kind='bar')
df['Key Skills'].value_counts()[0:9].iplot(kind='bar')
df['Role Category'].value_counts()[0:9].iplot(kind='bar')
df['Role'].value_counts()[0:9].iplot(kind='bar')