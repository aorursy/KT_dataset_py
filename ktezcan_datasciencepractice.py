import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns  # visualization tool



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))
data = pd.read_csv("../input/heart.csv")
data.info()
data.corr()
#correlation map

f,ax = plt.subplots(figsize=(18, 18))

sns.heatmap(data.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)

plt.show()
# Line Plot

# color = color, label = label, linewidth = width of line, alpha = opacity, grid = grid, linestyle = sytle of line

data.age.plot(kind = 'line', color = 'g',label = 'thalach',linewidth=1,alpha = 0.9,grid = True,linestyle = ':')

data.chol.plot(color = 'r',label = 'ca',linewidth=1, alpha = 0.9,grid = True,linestyle = '-.')

plt.legend(loc='upper right')     # legend = puts label into plot

plt.xlabel('x axis')              # label = name of label

plt.ylabel('y axis')

plt.title('Line Plot')            # title = title of plot

plt.show()
# Scatter Plot 

# x = attack, y = defense

data.plot(kind='scatter', x='age', y='chol',alpha = 0.5,color = 'red')

plt.xlabel('age')              # label = name of label

plt.ylabel('chol')

plt.title('age chol Scatter Plot')            # title = title of plot
# Histogram

# bins = number of bar in figure

data.chol.plot(kind = 'hist',bins = 100,figsize = (12,12))

plt.show()
# clf() = cleans it up again you can start a fresh

data.chol.plot(kind = 'hist',bins = 50)

plt.clf()

# We cannot see plot due to clf()

data.columns
data.shape
print(data['exang'].value_counts(dropna =False))
data.describe()
data.boxplot(column='chol',by = 'sex')
data_new = data.head()    

data_new
melted = pd.melt(frame=data_new,id_vars = 'oldpeak', value_vars= ['age','chol'])

melted
melted.pivot(index = 'oldpeak', columns = 'variable',values='value')
data1 = data.head()

data2= data.tail()

conc_data_row = pd.concat([data1,data2],axis =0,ignore_index =True) # axis = 0 : adds dataframes in row

conc_data_row
data1 = data['age'].head()

data2= data['chol'].head()

conc_data_col = pd.concat([data1,data2],axis =1) # axis = 0 : adds dataframes in row

conc_data_col
data.dtypes
data.info()
data1 = data.loc[:,["oldpeak","cp","ca"]]

data1.plot()
data1.plot(subplots = True)

plt.show()
data.plot(kind = "hist",y = "thalach",bins = 50,range= (0,250),normed = True)
# histogram subplot with non cumulative and cumulative

fig, axes = plt.subplots(nrows=2,ncols=1)

data.plot(kind = "hist",y = "thalach",bins = 50,range= (0,250),normed = True,ax = axes[0])

data.plot(kind = "hist",y = "thalach",bins = 50,range= (0,250),normed = True,ax = axes[1],cumulative = True)

plt.savefig('graph.png')

plt
import warnings

warnings.filterwarnings("ignore")

# In order to practice lets take head of pokemon data and add it a time list

data2 = data.head()

date_list = ["1992-01-10","1992-02-10","1992-03-10","1993-03-15","1993-03-16"]

datetime_object = pd.to_datetime(date_list)

data2["date"] = datetime_object

# lets make date as index

data2= data2.set_index("date")

data2 
print(data2.loc["1993-03-16"])

print(data2.loc["1992-03-10":"1993-03-16"])
# We will use data2 that we create at previous part

data2.resample("A").mean()

# Lets resample with month

data2.resample("M").mean()

# As you can see there are a lot of nan because data2 does not include all months
# In real life (data is real. Not created from us like data2) we can solve this problem with interpolate

# We can interpolete from first value

data2.resample("M").first().interpolate("linear")

data = pd.read_csv('../input/heart.csv')

data= data.set_index("#")

data.head()
# indexing using square brackets

data["chol"][1]
# using column attribute and row label

data.chol[1]
# using loc accessor

data.loc[1,["chol"]]
# Selecting only some columns

data[["age","chol"]]
# Difference between selecting columns: series and dataframes

print(type(data["chol"]))     # series

print(type(data[["chol"]]))   # data frames
# Slicing and indexing series

data.loc[1:10,"age":"chol"]   # 10 and "Defense" are inclusive
# Reverse slicing 

data.loc[10:1:-1,"age":"chol"] 
# From something to end

data.loc[1:10,"ca":] 
# Creating boolean series

boolean = data.age > 65

data[boolean]
# Combining filters

first_filter = data.age > 65

second_filter = data.chol > 300

data[first_filter & second_filter]
# Filtering column based others

data.age[data.chol<150] #cholestrol 150 den küçük yaş listesi.
# Plain python functions

def div(n):

    return n/2

data.chol.apply(div)
# Or we can use lambda function

data.chol.apply(lambda n : n/2)
# Defining column using other columns

data["denemelikdeger"] = data.cp + data.oldpeak

data.head()
# our index name is this:

print(data.index.name)

# lets change it

data.index.name = "index_name"

data.head()
# Overwrite index

# if we want to modify index we need to change all of them.

data.head()

# first copy of our data to data3 then change index 

data3 = data.copy()

# lets make index start from 100. It is not remarkable change but it is just example

data3.index = range(100,403,1)

data3.head()
# Setting index : type 1 is outer type 2 is inner index

data = pd.read_csv("../input/heart.csv")

data4 = data.set_index(["sex","restecg"]) 

data4.head(100)
dic = {"treatment":["A","A","B","B"],"gender":["F","M","F","M"],"response":[10,45,5,9],"age":[15,4,72,65]}

df = pd.DataFrame(dic)

df
# pivoting

df.pivot(index="treatment",columns = "gender",values="response")
df1 = df.set_index(["treatment","gender"])

df1

# lets unstack it
# level determines indexes

df1.unstack(level=0)
df1.unstack(level=1)
# change inner and outer level index position

df2 = df1.swaplevel(0,1)

df2
# df.pivot(index="treatment",columns = "gender",values="response")

pd.melt(df,id_vars="treatment",value_vars=["age","response"])
# We will use df

df
# according to treatment take means of other features

df.groupby("treatment").mean()   # mean is aggregation / reduction method

# there are other methods like sum, std,max or min
# we can only choose one of the feature

df.groupby("treatment").age.max()
# Or we can choose multiple features

df.groupby("treatment")[["age","response"]].min() 
df.info()

# as you can see gender is object

# However if we use groupby, we can convert it categorical data. 

# Because categorical data uses less memory, speed up operations like groupby

#df["gender"] = df["gender"].astype("category")

#df["treatment"] = df["treatment"].astype("category")

#df.info()