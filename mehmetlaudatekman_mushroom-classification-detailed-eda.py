# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



"""

Data Manipulating

"""

import numpy as np 

import pandas as pd 





"""

Visualization

"""

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
data = pd.read_csv('/kaggle/input/mushroom-classification/mushrooms.csv')
data.head()
data.info()
data.isnull().sum()
def converter(df,features):

    """

    A function that converts features into category

    """

    for ftr in features:

        df[ftr] = df[ftr].astype("category")

    

    return df
data = converter(data,data)
data.head()
data.info()
x = data.drop("class",axis=1)

y = data["class"]
def get_dummies(df,features):

    

    for ftr in features:

        

        df = pd.get_dummies(df,ftr)

    

    return df
x = get_dummies(x,x)
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.25,random_state=1)
print("Len of x_train is ",len(x_train))

print("Len of x_test is ",len(x_test))

print("Len of y_train is ",len(y_train))

print("Len of y_test is ",len(y_test))
from sklearn.svm import SVC



svc = SVC(random_state=12)

svc.fit(x_train,y_train)



print(svc.score(x_test,y_test))
from sklearn.tree import DecisionTreeClassifier

DTC = DecisionTreeClassifier(random_state=12)

DTC.fit(x_train,y_train)



print(DTC.score(x_test,y_test))
from sklearn.naive_bayes import GaussianNB



NBC = GaussianNB()

NBC.fit(x_train,y_train)

print(NBC.score(x_test,y_test))
from sklearn.ensemble import RandomForestClassifier



RFC = RandomForestClassifier(n_estimators=50,random_state=12)

RFC.fit(x_train,y_train)



print(RFC.score(x_test,y_test))