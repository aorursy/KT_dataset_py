# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
#Loading the dataset in a dataframe and printing the head of the values

dataset=pd.read_csv("../input/train_sample.csv")
print(dataset.head())
#Dropping the IP column for now as dont feel the need in the actual dataset

data=dataset.drop("ip",axis=1)
print(data.head())
#Converting the click_time and attributed_time to date time 

data["click_time"]=pd.to_datetime(data["click_time"])
data["attributed_time"]=pd.to_datetime(data["attributed_time"])
print(data.head(200))
#Checking and plotting the table form of is_attributed variable in order to check the distribution of data

data["is_attributed"].value_counts()
sns.countplot(x='is_attributed',data=data,palette='hls')
plt.show()
# Checking the count of all the values on basis of "OS,APP,DEVICE,CHANNEL"

data.groupby("os").count()
data.groupby("app").count()
data.groupby("device").count()
data.groupby("channel").count()
#Plotting the Device vs Is_Attributed and checking at the Distribution on basis of Device

%matplotlib inline
pd.crosstab(data.device,data.is_attributed).plot(kind='bar')
plt.title('Device with Is_Attribted')
plt.xlabel('Device')
plt.ylabel('Is_Attributed')
#Encoding the variables according to the one hot encoding.
encoded=pd.get_dummies(data,columns=["app","device","os","channel"],prefix=["app_encoded","device_encoded","os_encoded","channel_encoded"])
print(encoded.head())
print(encoded.columns.values)
#Adding a new variable which calculates the time difference between click_time and attributed_time
newencoded=encoded.fillna(0)
#newencoded['attributed_time'].astype('datetime64[ns]')
#newencoded["attributed_time"]=pd.to_datetime(newencoded["attributed_time"])
print(newencoded.dtypes)
#newencoded['time_difference']=newencoded['click_time']-newencoded['attributed_time']
#final_frame=encoded.drop(['click_time','attributed_time'],axis=1)
print(newencoded.head())

logreg = LogisticRegression()
rfe = RFE(logreg, 20)
rfe = rfe.fit(encoded[X], encoded[y] )
print(rfe.support_)
print(rfe.ranking_)