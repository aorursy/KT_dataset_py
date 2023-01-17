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
data = pd.read_csv('../input/pokemon_alopez247.csv')

data.head()
threshold = sum(data.Speed)/len(data.Speed)

print("threshold", threshold)

data["speed_level"] = ["high" if i > threshold else "low" for i in data.Speed]

data.loc[:10,["speed_level","Speed"]]
data.tail()
data.columns
data.info()
data.describe()
data.boxplot(column='Defense', by='Attack')

#visual exploratory data analysis

data.boxplot(column='Defense', by='isLegendary')

#tidy data

data_new = data.head()

data_new
melted = pd.melt(frame=data_new, id_vars = 'Name', value_vars=['Attack', 'speed_level','Color'])

melted
#pivoting(reversing melt)

melted.pivot(index='Name', columns='variable', values='value')
#concatenating data

data1=data.head()

data2=data.tail()

#vertical

concat_data = pd.concat([data1,data2], axis=0, ignore_index=True)

concat_data
data1=data['Attack'].head()

data2=data['Defense'].head()

conc_data_col=pd.concat([data1,data2],axis=1)

conc_data_col
data.dtypes
data['Type_1'] = data['Type_1'].astype('category')

data['Speed'] = data['Speed'].astype('float')

data.dtypes
#missing data

data['Type_2'].value_counts(dropna=False)
data.info()
data1=data

data1['Type_2'].dropna(inplace=True)
assert data['Type_2'].notnull().all() #dropping non values
data['Type_2'].fillna('empty',inplace=True)
assert data['Type_2'].notnull().all #return nothing because we don't have non values
assert data.columns[1] == 'Name'
assert data.Speed.dtypes == np.float
data.Speed.dtypes
#Reviewing Pandas

country=['Spain','France','Italy','Germany','England','Turkey','Russia']

population=['100','200','50','350','500','700','400']

list_label=['country','population']

list_col=[country,population]



zipped=list(zip(list_label,list_col))

data_dict=dict(zipped)

df = pd.DataFrame(data_dict)

df
df["capital"]=["madrid","paris","rome","berlin","london","ankara","moscow"]

df
#broadcasting

df["income"]=0

df
data1=data.loc[:,['Attack','Defense','Speed']]

data1.plot()
data1.plot(subplots = True)

plt.show() 
##scatter plot

data1.plot(kind = "scatter", x = "Attack", y = "Defense")

plt.show()
#histogram plot

#normed=True -> normalized between 0 and 1

data1.plot(kind="hist", x="Attack", y="Defense", bins=50, range=(0,250), normed = True)

plt.show()

fig,axes=plt.subplots(nrows=2,ncols=1)

data1.plot(kind="hist",y="Defense",bins=50,range=(0,250),normed=True,ax=axes[0])

data1.plot(kind="hist",y="Defense",bins=50,range=(0,250),normed=True,ax=axes[1],cumulative=True)

plt.savefig("graph.png")

plt



##cumulative = CDF cumulative distribution function in statistic
data.describe()
data.head()
#TIME SERIES

time_list=["1992-03-08","1992-04-12"]

print(type(time_list[1]))



datetime_object = pd.to_datetime(time_list)

print(type(datetime_object))
data2 = data.head()

date_list=["1992-01-10","1992-02-10","1992-03-10","1993-03-15","1993-03-16"]

datetime_object = pd.to_datetime(date_list)

data2["date"] = datetime_object

data2=data2.set_index("date")

data2
#loc datanın içinde belli bir indexteki verileri çeker

print(data2.loc["1993-03-16"])

print(data2.loc["1992-03-10":"1993-03-16"])
#Resampling pandas time series
#M=months A=years

data2.resample("A").mean()
data2.resample("M").mean()
data2.resample("M").first().interpolate("linear")

#linear interpolation

#lineer şekilde boşlukların arasını dolduruyor
data2.resample("M").mean().interpolate("linear")
data.head()
data.index
data = pd.read_csv('../input/pokemon_alopez247.csv', index_col="Number")

data.head()
data["HP"][1]
data.loc[1,["HP"]]

#first row, HP column
#selecting column

data[["HP","Attack"]].head()
#SLICING DATA FRAMES

print(type(data["HP"])) #series

print(type(data[["HP"]])) #data frame
data.loc[1:10, "HP":"Defense"]
data.loc[10:1:-1, "HP":"Defense"]
data.loc[1:10, "Speed":] #from speed to end
data.loc[1:5, :"Speed"] #from start to speed
#filtering

filter1 = data.HP > 200

data[filter1]
filter_a = data.HP > 150

filter_b = data.Attack > 50

data[filter_a & filter_b]

data[data.Speed<15]
data.HP[data.Speed <15] # HP of data that it's speed less than 15 
#TRANSFORMING DATA

def div(n):

    return n/2

data.HP.apply(div).head()
data.Attack.apply(div).head()
data.HP.apply(lambda n: n/2).head() #short way
data["total_power"] = data.Attack + data.Defense

data.head()

data.loc[1:5, "total_power":]
print(data.index.name)
data.index.name="index_name"
data.head()
data.index_name="Number"
data.index_name
#HIEARARCHICAL INDEXING

data1=data.set_index(["Type_1","Type_2"])

data1.head(10)
dic = {"threatment" : ["A","A","B","B"], "gender": ["F","M","F","M"],

      "response": [10,45,5,9], "age" : [15,4,72,65]}

df=pd.DataFrame(dic)

df
#pivoting(re-shaping)

df.pivot(index="threatment",columns="gender",values="response")
#STACKING AND UNSTAKING DF

df1 = df.set_index(["threatment","gender"])

df1
df1.unstack(level=0)
df1.unstack(level=1)
df2 = df1.swaplevel(0,1)

df2
#Melting df

df
pd.melt(df,id_vars="threatment", value_vars=["age","response"])
df
#GROUP BY

df.groupby("threatment").mean()
df.groupby("threatment").age.max()
df.groupby("threatment")[["age","response"]].mean()
df.info()
 