import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns



data = pd.read_csv("../input/heart.csv") 



import os

print(os.listdir("../input"))

data.age.value_counts()[:10]    # quantity of age  
sns.barplot(x=data.age.value_counts()[:10].index , y=data.age.value_counts()[:10]  )

plt.xlabel("Age")

plt.ylabel("Age Counter")

plt.title("Age Analysis ")

plt.show()

minAge = min(data.age)

maxAge = max(data.age)

averageAge = data.age.mean()



print("Min Age: ",minAge)

print("Max Age: ",maxAge)

print("Average Age: ",averageAge)





young_Ages = data[(data.age>29) & (data.age<40)]

middle_Ages = data[(data.age>=40) & (data.age<55)]

elderly_Ages = data[(data.age>55)]



print('Number Of Young Ages :',len(young_Ages))

print('Number Of Middle Ages :',len(middle_Ages))

print('Number Of Elderly Ages :',len(elderly_Ages))
sns.barplot(x = ["Young Age","Middle Age","Elderly Age"],y = [len(young_Ages),len(middle_Ages),len(elderly_Ages)])

plt.xlabel("Age")

plt.ylabel("Age Counts")

plt.title("Age States In Dataset")

plt.show()

data['AgeRange']=0

youngAge_index=data[(data.age>=29)&(data.age<40)].index

middleAge_index=data[(data.age>=40)&(data.age<55)].index

elderlyAge_index=data[(data.age>55)].index



for index in elderlyAge_index:

    data.loc[index,'AgeRange']=2

    

for index in middleAge_index:

    data.loc[index,'AgeRange']=1



for index in youngAge_index:

    data.loc[index,'AgeRange']=0



sns.swarmplot(x="AgeRange", y="age",hue='sex',

              palette=["r", "c", "y"], data=data)

plt.show()
sns.set_color_codes("pastel")

sns.barplot(y="AgeRange" , x="sex" , data=data,

            label="Total" , color="b"

            )

plt.show()

sns.countplot(elderly_Ages.sex)

plt.title("Elderly Sex Operations")

plt.show()

elderly_Ages.groupby(elderly_Ages['sex'])['thalach'].agg('sum')
sns.barplot(x=elderly_Ages.groupby(elderly_Ages['sex'])['thalach'].agg('sum').index,y=elderly_Ages.groupby(elderly_Ages['sex'])['thalach'].agg('sum').values)

plt.title("Gender Group Thalach Show Sum Time")

plt.show()
