# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
data = pd.read_csv('../input/googleplaystore.csv')
data1 = pd.read_csv('../input/googleplaystore_user_reviews.csv')
data1.head()
data1.info()
data1.describe()
data1.columns
data1.plot(kind = "scatter",x = 'Sentiment_Polarity',y = 'Sentiment_Subjectivity',alpha = 0.5,figsize = (20,20))

plt.xlabel('Sentiment_Polarity')

plt.ylabel('Sentiment_Subjectivity')

plt.title('Sentiment_Polarity - Sentiment_Subjectivity')

plt.legend()

plt.show()
data1.corr()
data1.Sentiment_Polarity.plot(kind = "hist",alpha = 0.5,color = "red",figsize = (10,10))

plt.show()

data1.Sentiment_Subjectivity.plot(kind = "hist",alpha = 0.5,color = "blue",figsize = (10,10))

plt.show()
a = np.logical_and(data1['Sentiment_Polarity']> 0.5, data1['Sentiment_Subjectivity']< 0.1)

data1[a]
data.head()
data.info()
data.columns
data.describe()
data.Rating.plot(kind = "hist",color = "red",alpha = 0.5,bins = 30)

plt.show()
c = data['Genres'] == "Auto & Vehicles"

data[c]
f = np.logical_and(data['Installs'] == "50,000+",data['Size'] == "20M")

data[f]
data[np.logical_and(data['Rating']>4.5,data['Installs'] == "100,000+")]
data[data['Category'] =="MEDICAL"]
a = data['Rating']>4.7

data[a]
data[np.logical_and(data["Rating"]>2.9,data["Rating"]<4.0)]
data[data['Rating']<2.0]
for index,value in data[['Rating']][0:10].iterrows():

    print(index, ":", value)
def abc(x,y):

    return data.loc[x:y]

    
abc(0,0)
data.shape
data1.shape
data.dtypes
data.head()
data1.head()
data.boxplot(column = 'Rating',by = 'Type',figsize = (20,20))

plt.show()
data_new = data.head(10)
melted = pd.melt(frame = data_new,id_vars = 'App', value_vars = ['Rating','Installs'])
melted
melted.pivot(index = 'App' , columns = 'variable', values = 'value')
data_new = data.head()

data2 = data.tail()

data_altalta_concat = pd.concat([data_new,data2],axis = 0,ignore_index = True)

data_altalta_concat
dataa = data['Rating'].head()

datab = data['Installs'].head()

data_yanyana_concat = pd.concat([dataa,datab],axis = 1)

data_yanyana_concat
print(data['Rating'].value_counts(dropna= False))
data.dtypes
data['Rating'].dropna(inplace = True)
assert data['Rating'].notnull().all()
data.loc[0:10,['Rating','rate_level']]
data.plot(kind = "hist",y = 'Rating',figsize = (20,20), range = (0,5),bins = 30,normed = True)

plt.show()
data1.plot(kind = "scatter",x = 'Sentiment_Polarity',y = 'Sentiment_Subjectivity',figsize = (20,20))

plt.show()
data.plot(kind= "hist",y ='Rating',normed = True,figsize = (20,20),cumulative = True,bins = 30,range = (0,5))

plt.show()
list1 = ["ayse","mehmet","veli","ali"]

list2 = [21,43,65,12]

list3 = ["isim","yas"]

list4 = [list1,list2]

zipped = list(zip(list3,list4))

dicti = dict(zipped)

df = pd.DataFrame(dicti)

df

df['maas'] = [1500,2150,3000,0]

df
liste = ["14-03-2000","21-04-1999","25-04-2000","30-12-1999"]

datetimes = pd.to_datetime(liste)
df['datetime'] = datetimes

df = df.set_index("datetime")

df
df.loc["30-12-1999":"14-03-2000"]
df.resample("A").mean()
df.resample("M").first().interpolate("linear")
df.resample("M").mean().interpolate("linear")
df.describe()
data["index"] = [i+1 for i in data.index]
data3=data.set_index("index")
data3.loc[1:10]
data3.loc[1,["Rating"]]
data3[["Rating","Price"]]
data3.loc[1:3,"Rating":"Price"]
data3.loc[3:1:-1,"Rating":"Price"]
data3.loc[10835: ,"Price": ]
data3.index.name = " "

data3
k =data["Size"] == "14M"

l =data["Rating"]>4.7

data[k&l]
data["Size"][data["Rating"]>4.9]
data.Size[data.Rating>4.9]
def alk(n):

    return n-1



data.loc[:5,"Rating"].apply(alk)
data.loc[:5,["Rating"]].apply(lambda w : w-1)
data
data["square"] = [i**2 for i in data["Rating"]]

data
data_new = data.head()
data_new.set_index(["Type","Genres"])
list1 = ["ayse","mehmet","veli","ali"]

list2 = [21,43,43,21]

list3 = ["isim","yas"]

list4 = [list1,list2]

zipped = list(zip(list3,list4))

dicti = dict(zipped)

df = pd.DataFrame(dicti)

df

df['maas'] = [1500,2150,3000,0]

df['boy'] = [170,150,163,193]

df
df.pivot(index = "isim",columns = "yas",values = "maas")
df1 = df.set_index(["yas","isim"])
df1
df1.unstack(level = 0)
df1.unstack(level = 1)
df1.swaplevel(0,1)
df2 = pd.melt(df,id_vars = "isim",value_vars = ["yas","maas"])
df2
df.groupby("yas").mean()
df.groupby("yas").boy.max()
df.groupby("yas")[["boy","maas"]].min()