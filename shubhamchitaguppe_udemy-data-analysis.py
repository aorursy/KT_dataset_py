# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
udemy=pd.read_csv("/kaggle/input/udemy-courses/udemy_courses.csv")
plt.show()
sns.set_style('darkgrid')

udemy.head()
udemy.info()
udemy.describe()
udemy.isnull().sum()
udemy.columns
udemy['published_timestamp']= pd.to_datetime(udemy['published_timestamp'], format = '%Y-%m-%dT%H:%M:%SZ')
udemy['date'] = pd.to_datetime(udemy['published_timestamp'].dt.date, format = '%Y-%m-%d')
udemy.head()
udemy.drop(['course_id','course_title','url'],inplace=True,axis=1)
udemy.columns
udemy['is_paid'].value_counts()
udemy.drop(udemy[udemy['num_subscribers'] > 100000].index, inplace=True)
udemy.head()

plt.figure(figsize=(15,8))
ax=sns.barplot(x='subject',y='price',data=udemy)
ax.set_title('Subject v/s Price',fontsize=22)
ax.set_xlabel('Subject',fontsize=22)
ax.set_ylabel('Prize',fontsize=22)
plt.figure(figsize=(15,8))
ax=sns.barplot(x='subject',y='price',data=udemy,hue='level')
ax.set_title('Subject v/s Price on level basis',fontsize=22)
ax.set_xlabel('Subject',fontsize=22)
ax.set_ylabel('Prize',fontsize=22)
plt.legend()
plt.figure(figsize=(15,8))
ax=sns.countplot(x='level',data=udemy,hue='is_paid')
ax.set_title('Course Levels ',fontsize=22)
ax.set_xlabel('Level',fontsize=22)
ax.set_ylabel('Total count',fontsize=22)
plt.legend()
plt.figure(figsize=(15,8))
sns.scatterplot(x='num_subscribers',y='content_duration',data=udemy)
plt.figure(figsize=(16,10))
sns.heatmap(udemy.corr())
udemy['log_price'] = np.log(udemy['price']+1)
udemy['log_num_subscribers'] = np.log(udemy['num_subscribers'])
udemy['log_num_reviews'] = np.log(udemy['num_reviews'] + 1)
udemy['log_num_lectures'] = np.log(udemy['num_lectures'])
udemy['log_content_duration'] = np.log(udemy['content_duration'])
plt.figure(figsize=(16,10))
ax=sns.swarmplot(x='subject',y='log_num_subscribers',data=udemy)
plt.figure(figsize=(16,10))
ax=sns.boxplot(x='level',y='log_num_subscribers',data=udemy,hue='subject')
plt.figure(figsize=(16,10))
ax=sns.violinplot(x='subject',y='log_num_reviews',data=udemy,hue='is_paid')
plt.figure(figsize=(16,10))
ax=sns.violinplot(x='subject',y='log_num_reviews',data=udemy,hue='level')