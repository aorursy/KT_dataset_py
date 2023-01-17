# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns  # visualization tool

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
data = pd.read_csv("../input/2017.csv")
data.info()
data.corr()
sns.heatmap(data.corr(), annot=True, linewidths=.5, fmt= '.1f')
plt.show()
data.head(10)
data.columns
data.plot(kind = 'line', x="Happiness.Rank", y="Happiness.Score", color = 'g',label = 'Happiness.Score',linewidth=2,alpha = 1,grid = True,linestyle = '-')

plt.legend(loc='upper right')     # legend = puts label into plot
plt.xlabel('x axis')              # label = name of label
plt.ylabel('y axis')
plt.title('Line Plot')            # title = title of plot
plt.show()
data.plot(kind='scatter', x='Economy..GDP.per.Capita.', y='Happiness.Score',alpha = 0.5,color = 'red')
plt.xlabel('Economy..GDP.per.Capita.')              # label = name of label
plt.ylabel('Happiness.Score')
plt.title('Economy..GDP.per.Capita. vs Happiness.Score Scatter Plot')   
series = data['Country']        # data['Defense'] = series
print(type(series))
data_frame = data[['Country']]  # data[['Defense']] = data frame
print(type(data_frame))
x=data[data["Economy..GDP.per.Capita."]>1.5]
y=data[(data["Economy..GDP.per.Capita."]>1.5) & (data["Happiness.Score"]>5.0)]
y1=data[np.logical_and(data["Economy..GDP.per.Capita."]>1.5, data["Happiness.Score"]>5.0 )]
y
y1
for index,value in data[['Country']][0:6].iterrows():
    print(index," : ",value)
score = sum(data["Happiness.Score"])/len(data["Happiness.Score"])

score
data["Happiness_level"] = ["high" if i > score else "low" for i in data["Happiness.Score"]]
data.loc[0:100,["Country","Happiness.Score","Happiness_level"]]
data.head()
data.tail()
data.info()
print(data["Happiness_level"].value_counts(dropna =False))
data.describe()
data.boxplot(column="Happiness.Score", by = "Happiness_level")
data1=data.head(10)
data1
data3 = data1[['Happiness.Score','Family']]
data3
data2=pd.melt(frame=data1, id_vars = 'Country',value_vars= ['Happiness.Score','Family'] )
data2
data2.pivot(index = 'Country', columns = 'variable',values='value')
data1=data.head()
data2=data.tail()
data1
data2
conc_data_row = pd.concat([data1,data2],axis =0,ignore_index =False)
conc_data_row
data1 = data['Happiness.Rank'].head()
data2 = data['Happiness.Score'].head()
conc_data_col = pd.concat([data1,data2],axis =1) # axis = 0 : adds dataframes in row
conc_data_col

data3=data[['Happiness.Rank','Happiness.Score']].head()
data3
data.dtypes
data.info()
assert  data['Country'].notnull().all()
data1=data[["Happiness.Score","Freedom","Family"]]
data1.plot()
plt.show()
data1.plot(subplots = True)
plt.show()
data1.plot(kind="scatter", x="Family", y="Happiness.Score")
plt.show()
data1.plot(kind="scatter", x="Freedom", y="Happiness.Score")
plt.show()
data1["Happy"]=[int(i) for i in data1["Happiness.Score"]]
data1

data1["Happy"].value_counts(dropna=False)
data1.plot(kind = "hist",y = "Happy",bins = 20,range= (0,10), normed=False)

data2=data.head()
data2
date_list = ["1992-01-10","1992-02-10","1992-03-10","1993-03-15","1993-03-16"]
datetime_object = pd.to_datetime(date_list)
data2["date"]=datetime_object
data2
data2 = data2.set_index("date")
data2
print(data2.loc["1993-03-16"])
print(data2.loc["1992-03-10":"1993-03-16"])
data2.resample("A").mean()
data2.resample("M").mean()
data2.resample("M").first().interpolate("linear")
data2.resample("M").mean().interpolate("linear")
data.head()
data1=data.head(10)
data1
data.Family[1]
data["Family"][1]
data1.loc[1,["Family"]]
data1[["Country","Happiness.Rank"]]
data1.loc[1:5,"Country":"Family"]
data1.loc[5:1:-1,"Country":"Family"]
data1
x=data1.Family>1.5
data1[x]
data1[data1.Family>1.5]
data1.Family>1.5
data1[data1["Family"]>1.5]
x=data1["Family"]>1.5
y=data1["Happiness.Score"]>7.5
data1[x&y]
data1["Happiness.Score"][data1["Family"]>1.5]
def div(n):
    return n/2
data1.Family=data1.Family.apply(div)
data1
data1.Family=data1.Family.apply(lambda n : n/2)
data1
data1["berk"]=data1["Happiness.Score"]+data1.Family
data1
print(data1.index.name)
data1.index.name="index_name"
print(data1.index.name)
data1
data1.index=range(100,110,1)
data1
data=pd.read_csv("../input/2017.csv")
data.head()
data1=data.head(10)
data1
data2=data1.set_index(["Happiness.Rank"])
data2
data3.index=data1["Happiness.Rank"]
data3
data4=data1.set_index(["Happiness.Rank","Family"])
data4
dic = {"treatment":["A","A","B","B"],"gender":["F","M","F","M"],"response":[10,45,5,9],"age":[15,4,72,65]}
df = pd.DataFrame(dic)
df
df.index=range(1,5)
df
df1=df.head()
df1
df.pivot(index="treatment",columns = "gender",values="response")
df1 = df.set_index(["treatment","gender"])
df1
df1.unstack(level=0)
df1.unstack(level=1)
df2 = df1.swaplevel(0,1)
df2
pd.melt(df,id_vars="treatment",value_vars=["age","response"])
df
df.groupby("treatment").mean() 
df.groupby("treatment").age.max() 
df.groupby("treatment")[["age","response"]].min() 
