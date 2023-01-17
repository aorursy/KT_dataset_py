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

data=pd.read_csv('../input/Pokemon.csv')

data.info()
data.describe()
data.tail()
data.columns
data.plot(kind = "scatter",x="Attack",y="Defense",color = 'r',label = 'Attack',linewidth=1, alpha = 0.5,grid = True,linestyle = '-.')

plt.show()
data.corr()
data.Attack.plot(kind="hist",color="red",alpha=0.5)

plt.show()
df_attack_logical = np.logical_and(data["Attack"] >40,data["Attack"]< 60)

data2=data[df_attack_logical]#"df_attack_logical" has boolean values (true or false) for each sample (pokemon) in data set.

                             # It's boolean values changes according to our (between 40-60) filter.

print(data2.head()["Name"])  # We need only names so I printed first five names coloumn (feature)
for index in data2.head()[['Attack']].iterrows():

    print(index)
data.columns = [each.lower() for each in data.columns] 

data.columns = [each.split()[0]+"_"+each.split()[1] if(len(each.split())>1) else each for each in data.columns]

data.head() # Listing first five samples in data set



data.rename(columns = {'#':'number'}, inplace = True) #If we choose "inplace= true" it will change name permanently, else it will change for just one running of code

data.head()
data.boxplot(column="attack",by="generation")

plt.show()
data_first_five=data.head()

data_first_five

melted=pd.melt(frame=data_first_five,id_vars="name", value_vars=["attack","defense"])

melted
melted.pivot(index = 'name', columns = 'variable',values='value')
data1 = data.head()

data2= data.tail()

conc_data_row = pd.concat([data1,data2],axis =0,ignore_index =True) # axis = 0 : adds dataframes in row, ignore_index =True means give  new index numbers for data2



conc_data_row
data.dtypes
data['attack'] = data['attack'].astype('object');

data.dtypes
data.info()
data["type_2"].value_counts() # It shows us the frequancies of type_2 values but there is no null values
data["type_2"].value_counts(dropna =False) # Now we can see null values as we calculate there is 386 null values
data["type_2"].dropna(inplace = True)

data["type_2"].value_counts(dropna=False)
data["type_2"].notnull().all()