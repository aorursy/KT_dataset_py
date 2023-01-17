# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)





import matplotlib.pyplot as plt

import seaborn as sns



from sklearn.model_selection import train_test_split #Spliting

from sklearn.neighbors import KNeighborsClassifier #KNN

from sklearn.linear_model import LogisticRegression #Logistic Regression



import warnings as wrn

wrn.filterwarnings('ignore')



import plotly as py

from plotly.offline import init_notebook_mode, iplot

init_notebook_mode(connected=True)

import plotly.graph_objs as go

import plotly.express as px





# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data = pd.read_csv('/kaggle/input/pokemon-challenge/pokemon.csv')
data.head()
data.info()
data.isnull().sum()
fig,ax = plt.subplots(figsize=(12,8))

sns.countplot(data.Legendary,ax=ax)

plt.show()
fig,ax = plt.subplots(figsize=(12,8))

sns.countplot(data.Legendary,hue=data["Type 1"],ax=ax)

plt.show()
fig,ax = plt.subplots(figsize=(10,10))

sns.heatmap(data.corr(),annot=True,linewidth=1.5)

plt.show()

data.drop(["Name","#"],axis=1,inplace=True)

data.head()
data.Legendary = data.Legendary.astype(int)
data.head()
type1 = list(data["Type 1"].unique())

type2 = list(data["Type 2"].unique())

type2.remove(np.nan)





print(type1,"\n")

print(type2)

type1 = list(zip(type1,[i for i in range(1,len(type1)+1)]))

type2 = list(zip(type2,[i for i  in range(1,len(type2)+1)]))



print(type1,"\n")

print(type2)
data2 = data.copy()

type1_list = [i for i in range(0,len(data))]

for type_,value in type1:

    

    index = data[data["Type 1"] == type_].index.values

    for ind in index:

        type1_list[ind] = value



        

type2_list = [0 if type(each)==type(np.nan) else each for each in data["Type 2"]]



for type_,value in type2:

    

    index = data[data["Type 2"] == type_].index.values

    for ind in index:

        type2_list[ind] = value
print(type1_list[0:10])



print(type2_list[0:10])
data["Type 1"] = type1_list

data["Type 2"] = type2_list

data.head()
data.dtypes
data = (data-np.min(data)) / (np.max(data)-np.min(data))

data.head()
x = data.drop("Legendary",axis=1)

y = data["Legendary"]



x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=1)
knn = KNeighborsClassifier(n_neighbors=10)



knn.fit(x_train,y_train)



print(knn.score(x_test,y_test))
data2.head() # It is a copy of dataset

x2 = data2.drop(["Type 1","Type 2","Legendary"],axis=1) # I am going to drop them

y2 = data2.Legendary

x_train2,x_test2,y_train2,y_test2 = train_test_split(x2,y2,test_size=0.2,random_state=1)



knn2 = KNeighborsClassifier(n_neighbors=10)

knn2.fit(x_train2,y_train2)

print(knn2.score(x_test2,y_test2))

lr = LogisticRegression() #I've created my model using SKLearn



lr.fit(x_train,y_train) # I've trained my model with my arrays



print(lr.score(x_test,y_test)) #I've tested my model 