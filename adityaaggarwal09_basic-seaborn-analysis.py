# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline

from plotly import __version__

import cufflinks as cf



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data=pd.read_csv('/kaggle/input/jobs-on-naukricom/home/sdf/marketing_sample_for_naukri_com-jobs__20190701_20190830__30k_data.csv')
data.head()
data.info()
plt.figure(figsize=(12,4))

sns.heatmap(data.isnull(),cmap='Blues')
missing_percent= (data.isnull().sum()/len(data))[(data.isnull().sum()/len(data))>0].sort_values(ascending=False)

missing_data = pd.DataFrame({'Missing Percentage':missing_percent*100})

missing_data
mis_data= (data.isnull().sum() / len(data)) * 100

mis_data= mis_data.drop(mis_data[mis_data == 0].index).sort_values(ascending=False)

plt.figure(figsize=(16,5))

sns.barplot(x=mis_data.index,y=mis_data)
data.describe()
data['Job Title'].value_counts()
sns.barplot(x=data['Job Title'].value_counts()[0:9],y=data['Job Title'].value_counts()[0:9].index)
data['Job Salary'].value_counts()
sns.barplot(x=data['Job Salary'].value_counts()[0:9],y=data['Job Salary'].value_counts()[0:9].index)
data['Key Skills']
sns.barplot(x=data['Key Skills'].value_counts()[0:10],y=data['Key Skills'].value_counts()[0:10].index, palette="ch:.25")
data['Role Category'].value_counts()[0:10]
sns.barplot(x=data['Role Category'].value_counts()[0:10],y=data['Role Category'].value_counts()[0:10].index, palette="ch:.25")
data['Location'].value_counts()
sns.barplot(x=data['Location'].value_counts()[0:10],y=data['Location'].value_counts()[0:10].index)
data['Industry']
sns.barplot(x=data['Industry'].value_counts()[0:10],y=data['Industry'].value_counts()[0:10].index)
data['Job Experience Required'].value_counts()
sns.barplot(x=data['Job Experience Required'].value_counts()[0:10],y=data['Job Experience Required'].value_counts()[0:10].index)
data['Functional Area']
sns.barplot(x=data['Functional Area'].value_counts()[0:10],y=data['Functional Area'].value_counts()[0:10].index)
data['Role']
sns.barplot(x=data['Role'].value_counts()[0:10],y=data['Role'].value_counts()[0:10].index)