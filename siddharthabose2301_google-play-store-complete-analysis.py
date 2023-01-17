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
#importing the libraries

import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
data = pd.read_csv("/kaggle/input/google-play-store-apps/googleplaystore.csv")
data.head()
data.info()
data['Reviews'] = data['Reviews'].apply(lambda x: int(float(x[:-1])*(10**6)) if x[-1]=='M' else int(float(x)))
data.loc[10472,'Installs']='0'
data[data['Installs']=='Free']
data['Installs'].value_counts()
data['Installs'] = data['Installs'].apply(lambda x: int((''.join(x.split(','))).strip('+')))
data.drop(10472,axis = 0,inplace = True)
data.tail()
data_cat = pd.DataFrame(data['Category'].value_counts())
fig = plt.figure(figsize = (15,8))
fig = sns.barplot(data_cat['Category'],data_cat.index)
plt.title('Number of Apps in each Category',size = 30)
plt.show()
data_cat_wrt_reviews = data[['Category','Rating']].groupby('Category').mean().sort_values('Rating',ascending = False)
data_cat_wrt_reviews.head(3)
fig = plt.figure(figsize = (15,8))
fig = plt.barh(data_cat_wrt_reviews.index,data_cat_wrt_reviews['Rating'])
plt.title('Category vs Mean Ratings',size = 30)
plt.xlabel('Ratings',size = 15)
plt.xticks(np.linspace(0,5,6))
plt.show()
data_cat_wrt_downloads = data[['Category','Installs']].groupby('Category').sum().sort_values('Installs',ascending = False)
data_cat_wrt_downloads.head(3)
fig = plt.figure(figsize = (15,8))
fig = plt.barh(data_cat_wrt_downloads.index,data_cat_wrt_downloads['Installs'])
plt.title('Category vs Total Downloads',size = 30)
plt.xlabel('# of Downloads',size = 15)
plt.show()
