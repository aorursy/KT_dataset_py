# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import plotly.express as px

%matplotlib inline
data = pd.read_csv(r'/kaggle/input/coursera-course-dataset/coursea_data.csv')

data.head()
print("Number of datapoints in the dataset : ",data.shape[0])

print("Number of features in the dataset : ",data.shape[1])

print("Features : ",data.columns.values)
data.info()
data.describe()
print("Number of duplicate values in this dataset are : ",data.duplicated().sum())
data = data.reset_index(drop=True)

data.head()
fig = plt.figure(figsize=(20,18))



plt.subplot(2,2,1)

sns.countplot('course_difficulty',data=data)

plt.grid()



plt.subplot(2,2,2)

sns.countplot(x='course_Certificate_type',data=data)

plt.grid()



plt.show()
fig =px.histogram(data,x='course_Certificate_type',y='course_students_enrolled',color='course_difficulty')

fig.show()
fig =px.histogram(data,x='course_difficulty',y='course_students_enrolled',color='course_Certificate_type')

fig.show()
plt.figure(figsize=(20,15))



plt.subplot(2,2,1)

sns.countplot(x='course_Certificate_type',hue='course_difficulty',data=data,palette=['yellow','darkblue','purple','green'])

plt.grid()

plt.legend(loc='upper right')





plt.subplot(2,2,2)

sns.countplot(x='course_difficulty',hue='course_Certificate_type',data=data,palette=['red','teal','black'])

plt.grid()

plt.legend(loc='upper right')



plt.show()
organisations = data.groupby("course_title")['course_difficulty'].max().reset_index()

org = organisations.sort_values(by='course_difficulty').reset_index(drop=True)

print(org)

fig=px.pie(org,names ='course_difficulty',color_discrete_sequence=px.colors.sequential.RdBu)

fig.update_traces(marker=dict(line=dict(color='black',width=1.5)))

fig.show()
organisations = data.groupby("course_title")['course_Certificate_type'].max().reset_index()

org = organisations.sort_values(by='course_Certificate_type').reset_index(drop=True)

print(org)

fig=px.pie(org,names ='course_Certificate_type',color_discrete_sequence=px.colors.sequential.GnBu)

fig.update_traces(marker=dict(line=dict(color='black',width=1.5)))

fig.show()
plt.figure(figsize=(15,8))

sns.countplot('course_rating',data=data)

plt.show()
fig=px.pie(data,names ='course_rating',color_discrete_sequence=px.colors.sequential.YlOrBr)

fig.update_traces(marker=dict(line=dict(color='black',width=1.5)))

fig.show()
px.violin(data,y='course_rating',points='all',box=True)
px.violin(data,y='course_rating',points='all',box=True,color='course_rating')