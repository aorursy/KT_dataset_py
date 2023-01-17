# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
data=pd.read_csv("../input/googleplaystore.csv")

data.head()

data.info()
#rename columns

data.rename(columns={'Content Rating':"content_rating",'Size':'app_size','Last Updated':"last_updated",'Current Ver':"current_ver",'Android Ver':"android_ver"},inplace=True)

data.columns=[each.lower() for each in data.columns]

data.columns

#cleaning Nan

data.dropna(how ='any', inplace = True)

data.info()
data.app_size.unique()

#converting app_size M and k to numeric

data.app_size=data.app_size.str.replace('k','e+3')

data.app_size=data.app_size.str.replace('M','e+6')

#checking if each value is convertable

def is_convertable(v):

    try:

        float(v)

        return True

    except ValueError:

        return False

    

temp=data.app_size.apply(lambda x: is_convertable(x))

temp.head()

data.app_size[~temp].value_counts()
data.app_size=data.app_size.replace('Varies with device',np.nan)

data.app_size=data.app_size.astype("float")

data.app_size.head()

data.installs=data.installs.str.replace(",","")

data.installs=data.installs.str.replace("+","")

data.installs=data.installs.astype("float")

data.installs.head()
data.reviews=data.reviews.astype("float")
data.price=data.price.str.replace("$","")

data.price=data.price.astype("float")

data.info()

#now all numeric datas are numeric types
#correlation

f, ax = plt.subplots(figsize=(13,13))

sns.heatmap(data.corr(), annot= True, linewidths= 5, fmt= ".1f", ax=ax)
data.plot(kind="scatter",x="installs",y="reviews",color="blue")

data.plot(kind="scatter",x="rating",y="price",color="red")

#plotting mean rating values of each category

category_mean=data.groupby("category").rating.mean()

print(category_mean)

plt.subplot(4,1,1)

plt.scatter(category_mean.index[:9],category_mean[:9],color="red")

plt.ylabel("rating")

plt.subplot(4,1,2)

plt.scatter(category_mean.index[9:18],category_mean[9:18],color="red")

plt.ylabel("rating")

plt.subplot(4,1,3)

plt.scatter(category_mean.index[18:27],category_mean[18:27],color="red")

plt.ylabel("rating")

plt.subplot(4,1,4)

plt.scatter(category_mean.index[27:],category_mean[27:],color="red")

plt.ylabel("rating")
#dividing apps into rating points above 3.5 and below 3.5

filter1=data.rating>3.5

filter3=data.installs<10000

good_apps=data[filter1&filter3]

filter2=data.rating<=3.5

bad_apps=data[filter2&filter3]

good_apps.head()

bad_apps.head()
#plotting good and bad apps's installs value

plt.subplot(2,1,1)

plt.scatter(good_apps.installs,good_apps.rating,color="red")

plt.xlabel("installs")

plt.ylabel("ratings")

plt.subplot(2,1,2)

plt.scatter(bad_apps.installs,bad_apps.rating,color="red")

plt.xlabel("installs")

plt.ylabel("ratings")
#some extra stuff that i try

#concatenating

data1=data.head()

data2=data.tail()



conc_data_row=pd.concat([data1,data2], axis=0, ignore_index=True)

conc_data_row

#melted

data1=data["rating"].head()

data2=data["reviews"].head()

data3=data["installs"].head()

df=pd.concat([data1,data2,data3], axis=1)

melted=pd.melt(frame=df,id_vars="rating",value_vars=["reviews","installs"])

melted
pvt=melted.pivot(index="rating",columns="variable",values="value")

pvt