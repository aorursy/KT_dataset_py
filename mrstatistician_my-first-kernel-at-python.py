# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns 



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
data  = pd.read_csv("/kaggle/input/new-york-city-airbnb-open-data/AB_NYC_2019.csv")
data.head()#show data first 5 observation



data.info() # about data information

data.isnull().sum()#count missing value
data["name"].fillna('missing name',inplace = True)

data["host_name"].fillna('missing name',inplace = True)

data["reviews_per_month"].fillna(np.std(data.reviews_per_month),inplace = True)

#here we do filling missing values basic method 

f,ax = plt.subplots(figsize = (12,12))

sns.heatmap(data.corr(),annot = True,linewidth = .60,

           fmt = ".1f",ax = ax )

plt.show()

#here visualization on the variable correlation
data.plot(kind = "scatter", x = "minimum_nights",y = "price",alpha = .6,color= "purple")

plt.show() # two variable between not correlation
data.price.plot(kind = "line",label = "price",color = "red",linewidth = 1,alpha = .5,

          grid = True,linestyle = ":")

data.minimum_nights.plot(kind = "line",label = "minimym nights",color = "purple",linewidth = 1,

                        alpha = .9)

plt.legend(loc = "upper left")

plt.ylabel("frequency")

plt.xlabel("data")

plt.title("nyc airbnb price and minimum nights")

plt.show()



data.number_of_reviews .plot(kind = "hist", bins = 50,figsize = (15,15),color = "orange",alpha = .67)

plt.xlabel("number_of_reviews")

plt.ylabel("frequency")

plt.title("price with hist")

plt.show()
x = np.logical_and(data["price"]>100,data["calculated_host_listings_count"]>5)

data[x].tail()
data.plot(kind = "hist",x = "reviews_per_month" , y ='availability_365')

plt.show()
for index , value in data[["availability_365"]][0:12].iterrows():

    print(index,":",value)
data.info()
pricem = sum(data.price) /len(data.price)
data["price_mean"] = ["high" if i > pricem else "low " for i in data.price]
data[data.neighbourhood_group == "Manhattan"].head()
visu = data.loc[:,["price","price_mean"]]
melted = pd.melt(frame = data.head(),id_vars = 'host_name', value_vars  = ["neighbourhood_group","price_mean","availability_365"])

melted
piv = melted.pivot(index = 'host_name' ,columns = 'variable' , values = "value" ) # re melt = pivot 

piv
df1 = data.head()

df2 = data.tail()
df = pd.concat([df1,df2],axis = 0 , ignore_index=True)
df.dtypes.value_counts()
df["last_review"] = pd.Timestamp("20180715")