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
#reading the data
nfDATA = pd.read_csv("/kaggle/input/netflix-shows/netflix_titles.csv")
#printing the data
nfDATA.head()
#visualizing the data
nfDATA.hist(figsize=(10,7), color = 'green')
plt.show()
#information about the data
nfDATA.info()
#number of rows and columns in the data
nfDATA.shape
#name of the columns in the dataset
nfDATA.columns
#Description of the Data
nfDATA.describe().T
#cchecking is there is any null value or not
nfDATA.isnull().values.any()
#printing the null sum of null values in the dataset
nfDATA.isnull().sum()
#lets drop the columns having maximum null values 
nfDATA.drop(["director","cast"], axis=1, inplace=True)

#now lets print the data
nfDATA.head()
#now lets visulaize the data set
nfDATA.hist(figsize=(10,7), color = 'green')
plt.show()
nfDATA.country.value_counts()
#Replacing null values with Unites States
nfDATA.country.replace(np.nan,"United States", inplace = True)
#now lets check the null values again
nfDATA.isnull().sum()
#details of date_added
nfDATA.date_added.value_counts()
#replaced nan values with 'Not Date' and splited date_date
df = nfDATA[['date_added']].replace(np.nan,'Not Date')
df["release_month"] = df['date_added'].apply(lambda x: x.lstrip().split(" ")[0])
df.head()

#counts of realease months
df.release_month.value_counts()
#replaced NOT with 0
df.release_month.replace("Not",0,inplace=True)
df.release_month.value_counts()
#In a new dataFrame df we will delete date_added column
df.drop("date_added",axis=1,inplace=True)
df
#Lets check the null values
nfDATA.isnull().sum()
new_data=pd.concat([nfDATA,df],axis=1)
new_data.head()
new_data.isnull().sum()
new_data.drop("date_added", axis = 1, inplace = True)
new_data.head()
new_data.isnull().sum()
#counting of Rating
new_data.rating.value_counts()
new_data.rating.replace(np.nan,"TV-MA",inplace=True)
new_data.isnull().sum()  #after replacing printing the null values in new dataset
#printing the dataset
new_data.head()
#visualizing the new_data 
sns.countplot(x="type",data=new_data)
new_data.type.value_counts()
plt.figure(figsize=(10,7))
sns.countplot(x=new_data.country,order=new_data.country.value_counts().index[0:10]);
plt.xticks(rotation=45)
plt.show()
plt.figure(figsize=(30,10))
sns.countplot(x=new_data.release_year,order=new_data.release_year.value_counts().index[0:30]);
plt.xticks(rotation=45)
plt.show()
plt.figure(figsize=(10,7))
sns.countplot(x=new_data.rating,order=new_data.rating.value_counts().index[0:20]);
plt.xticks(rotation=45)
plt.show()


plt.figure(figsize=(10,7))
sns.countplot(x=new_data.release_month,order=new_data.release_month.value_counts().index[0:12]);
plt.xticks(rotation=45)
plt.show()


plt.figure(figsize=(20,10))
sns.countplot(x=new_data.country,hue= new_data.type,order = new_data['country'].value_counts().index[0:17])
plt.xticks(rotation=45)
plt.show()


labels = new_data.country.value_counts()[0:6].index
colors = ['grey','blue','red','yellow','green','brown']
explode = [0,0,0,0,0,0]
sizes = new_data.country.value_counts()[0:6].values

# visual
plt.figure(figsize = (7,7))
plt.pie(sizes, explode=explode, labels=labels, colors=colors,autopct='%1.1f%%')
plt.show()
plt.figure(figsize=(10,7))
sns.countplot(x=new_data.rating,hue=new_data.type,order=new_data.rating.value_counts().index[0:20]);
plt.xticks(rotation=45)
plt.show()