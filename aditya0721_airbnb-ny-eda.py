# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt 
# read the file from the current directory

data=pd.read_csv("/kaggle/input/new-york-city-airbnb-open-data/AB_NYC_2019.csv")
#check if the file was successfully downloaded and check for its first 5 elements 

data.head()
data.info()
data.shape

# Gives us the data entries with element in the tupple() as the number of rows and 2nd element as the number of columns 
# data['name'].str.lower().head()

#lets see how many missing values are there in our data 

data.isna().sum()
data.describe()
Q3=data.price.quantile(0.75)

Q1=data.price.quantile(0.25)

print(Q1)

print(Q3)

IQR=Q3-Q1

IQR
Outlier_price=(data['price']<(Q1-1.5*IQR))|(data['price']>(Q3+(1.5*IQR)))

print(Outlier_price.value_counts())

data[Outlier_price].price



#Now lets check out the unique categorical values in the column room_type and convert it to numerical.
data['room_type'].unique()
#Lets drop some columns which as of now are of less use and will be the last thing we want to know in our data analysis 



# data.drop(['host_id','host_name','name'], inplace =True)



data.head()



# Data will now look a bit better.
import seaborn as sns 
fig=plt.figure(figsize=(20,10))

sns.boxplot(x='neighbourhood_group',y='price',data=data)
sns.heatmap(data.corr(),cmap='coolwarm')

plt.title('Correlation')
sns.catplot(y='neighbourhood_group',

           kind='count',

           height=7,

           aspect=1.5,

           order=data.neighbourhood_group.value_counts().index,

           data=data)
sns.catplot(x='room_type',

            kind='count',

           height=7,

           aspect=2,

           data=data,

           color='grey',

           )

plt.xticks(size=30)

plt.yticks(size=30)

plt.xlabel('room_type',fontsize=40)

plt.ylabel('count',fontsize=40)

plt.title('Distribution as per Room Type',fontsize=40)
data['room_type'].unique()

from sklearn  import preprocessing

le=preprocessing.LabelEncoder()

data['room_type']=le.fit_transform(data['room_type'])

data['room_type'].value_counts()
data.head()
#We can also see that there are multiple host_name,id, host_id, & name in our data. It will be of less advantage to us in our further knowledge. So we will remove this feature from our 

# data
data.drop(['host_name','id','host_id','name'],axis=1,inplace=True)
data.groupby('neighbourhood_group').sum()