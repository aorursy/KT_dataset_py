# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

data_path="../input/melb_data.csv"


data=pd.read_csv(data_path)
data.head()
index_data=data['Unnamed: 0']
continuous_data=data[['Rooms','Price','Distance','Postcode','Bedroom2','Bathroom','Car','BuildingArea','Landsize','Lattitude','Longtitude','Propertycount']]

categorical_data=data[['Suburb','Address','Type', 'Method','SellerG', 'CouncilArea','Regionname']]
    
time_data=data[['Date','YearBuilt']]

#print(continuous_data,categorical_data,time_data)
for i in range(3):
    if(i==0):
        print(continuous_data.head())
    elif(i==1):
        print(categorical_data.head())
    else:
        print(time_data.head())




for i in continuous_data.columns:
    print(i)
#come back to this loop after solutions
for i in continuous_data.columns :
    plt.figure()
    continuous_data.plot(x="Lattitude", y="Longtitude", kind="scatter", alpha=0.4,figsize=(12,10),c="Price",colormap="gist_rainbow",s=continuous_data[i],label=i,colorbar=True)
    plt.legend()
    plt.show()

plt.figure()
data.plot(x="Lattitude", y="Longtitude", kind="scatter", alpha=0.4,figsize=(12,10),c="Price",colormap="gist_rainbow",s=data.Distance*6,label="Distance",colorbar=True)
plt.legend()
                                                                                                                                                                                               
            
continuous_data.size()
continuous_data.isnull().sum()
continuous_data[continuous_data.Postcode.isnull()|continuous_data.Distance.isnull()]
data.columns
data.Postcode.head()



continuous_data.describe()
continuous_data.isnull().sum()

#do it later afyter cleaning data
plt.figure()
sns.pairplot(continuous_data[0:10][:1000])#,hue="Postcode",palette="pastel")
plt.show()
#Plotting categorical variables







