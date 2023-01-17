# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns #visualization tool



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
data = pd.read_csv('/kaggle/input/fenerbahce-squad-database/fenerbahce_footballers.csv')
data.info()
data.shape
data.corr()
#correlation map



f,ax = plt.subplots(figsize=(10,10))

sns.heatmap(data.corr(),annot = True , linewidth = .5, fmt = '.2f',ax = ax)

plt.show()
data.head(7)
data.tail(5)
data.columns
#line plot

data.age.plot(kind = "line" , color = "red" , alpha = 0.5 , grid = True , linestyle = ":",label = "age")

plt.legend(loc = "upper right")

plt.xlabel("x axis")

plt.ylabel("y axis")

plt.title("Footballer's Ages Line Plot")

#scatter plot

data.plot(kind = "scatter" , x = "age" , y = "contract_to" , color = "red")

plt.xlabel("Age")

plt.ylabel("Contract Expired")

plt.title("Age-Contract Scatter Plot")

plt.show()
#histogram 

data.age.plot(kind = "hist",bins = 20,figsize = (15,15) )

plt.show()
#clf() = cleans it up again you can start fresh

data.age.plot(kind = "hist",bins = 20)

plt.clf()
#create dictionary and look its keys and values

footballers = {"Zanka" : "CB" , "Jailson" : "DMF"}

print(footballers.keys())

print(footballers.values())
footballers["Jailson"] = "CB" #update existing value

footballers["Altay Bayindir"] = "GK" #insert new record

print(footballers)

print("Zanka" in footballers) #check exist or not

footballers.clear() #clear all dict

print(footballers)
#filtering data

filter = data['age'] > 28

data[filter]

#filtering with logical and

data[np.logical_and(data['age']<20,data['contract_to']>2020)]
for index,value in data[['player']][0:12].iterrows():

    print(index ," : ",value)
#list compherension



list_compherension = [i*2 for i in data['age'][0:5]]

list_compherension
#list of frequency of players' positions

print(data['position'].value_counts(dropna=False))
data1 = data.copy()



average_age = sum(data1.age)/len(data1.age)



data1['age_level'] = ["low" if value < average_age else "high" for value in data1.age ]



data1.age_level
#box Plot

data1.boxplot(column = "age" , by = "age_level")

plt.show()
#melting

data2 = data.head()

melted = pd.melt(frame = data2,id_vars = 'player',value_vars = ["position" , "nationality"])

melted

#pivoting data

#reverse of melting

melted.pivot(index = "player" , columns = "variable" , values = "value")
#concatenating data

data_head = data.head()

data_tail = data.tail()

conc_data_row = pd.concat([data_head,data_tail],axis=0,ignore_index=True)

conc_data_row
data.dtypes
#assert statment

assert 1 == 1 #returns nothing because it is True
assert data["position"].notnull().all() #returns nothing because we don't have NaN  values.
#HIERARCHICAL INDEXING



data_new = data.copy()





data_new= data_new.set_index(["age",'nationality'])

data_new