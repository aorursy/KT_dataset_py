import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns
data = pd.read_csv("../input/mercedesbenz-greener-manufacturing/train.csv")
data.head(5)
data = data[['X0', 'X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X8']]
data
dic = {}

for i in data.columns:

    print(i , ":\n", data[i].value_counts())

    dic.update({i : len(data[i].unique())})

    
data.info()
data.isnull().sum()
dic
df = data.copy()
X3_dummy = pd.get_dummies(df['X3'], drop_first = True)
X3_dummy
df = pd.concat([df,X3_dummy], axis = 1)
df
df = data.copy()
df.head()
dic
#we take X0 for this encoding
df = df[['X0']]
df.value_counts()

#from this taking top ten counts for creating dummies... anothers are eliminated
X0_lst = df['X0'].value_counts().index[0:15]

def hot_encoding(df, var, lst):

    for i in lst:

        df[var + "_" + i] = np.where(df[var] == i, 1 , 0 )
hot_encoding(df, 'X0', X0_lst)
df
#mean encoding is done only based on the y value of the dataset
data = pd.read_csv("../input/titanic/train.csv")
data.head()
data.info()
df = data.copy()
#from this we take cabin column for this encoding

df['Cabin'] = df['Cabin'].astype(str).str[0]
df['Cabin'].value_counts()
mean_for_cab = df.groupby(['Cabin'])['Survived'].mean().to_dict()
mean_for_cab
df['Cabin'] = df['Cabin'].map(mean_for_cab)
#this time we take Embarked column



df['Embarked']
df['Embarked'].value_counts()
prob = df.groupby(['Embarked'])['Survived'].mean()
prob
neg_prob = 1 - prob
neg_prob
probability_ratio = prob/neg_prob
probability_ratio
dic = dict(probability_ratio)
df['Embarked'] = df['Embarked'].map(dic)
prob_data = df['Embarked']
plt.figure(figsize = (10,4))

sns.countplot(prob_data)
df = data.copy()
df['Embarked'].value_counts()
#Replace that with the frequency

dic = df['Embarked'].value_counts().to_dict()
df['Embarked'] = df['Embarked'].map(dic)
df['Embarked'].plot(kind = "hist")
df = data.copy()
df['Cabin']
df['Cabin'] = df['Cabin'].astype(str).str[0]
lst = df['Cabin'].value_counts().index
dic = {j:i for i,j in enumerate(lst , 0)}
dic
#now map the dic to the variable
df['Cabin'] = df['Cabin'].map(dic)
print(df['Cabin'].value_counts())

df['Cabin'].plot(kind = "hist")
#it is based on the y variables
df = data.copy()
df['Embarked']
mean_val = df.groupby(['Embarked'])['Survived'].mean()
mean_val
lst = mean_val.index
lst
#mapping based the mean 

dic = {j:i for i,j in enumerate(lst,0)}
df['Embarked'] = df['Embarked'].map(dic)
df['Embarked'].value_counts()