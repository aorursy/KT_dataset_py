# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
estimate=pd.read_csv("/kaggle/input/malnutrition-across-the-globe/malnutrition-estimates.csv")



average=pd.read_csv("/kaggle/input/malnutrition-across-the-globe/country-wise-average.csv")

average.head()
average.describe()
x=average.iloc[:,1:]

x
from sklearn.impute import KNNImputer

knn=KNNImputer()

X=knn.fit_transform(x)

X=pd.DataFrame(X)
X.columns=["Income Classification","Severe Wasting","Wasting", "Overweight", "Stunting","Underweight","U5 Population ('000s)"]

X.describe()
country=average.iloc[:,0:1]

X["country"]=country

X
import matplotlib.pyplot as plt

import seaborn as sns

plt.figure(figsize=(15,10))

sns.relplot(data=X,x="Income Classification",y="Overweight")
X.groupby(by="Income Classification").mean()[["Overweight"]]
sns.relplot(data=X,x="Income Classification",y="Underweight")
X.groupby(by="Income Classification").mean()[["Underweight"]]
X[X["Income Classification"]<=0]["country"]
X[X["Income Classification"]==3]["country"]
X[X["country"]=="OMAN"]
y=X["Income Classification"]

xx=X.iloc[:,1:7]

xx
from sklearn.tree import DecisionTreeClassifier

dtree=DecisionTreeClassifier(max_depth=3)

dtree.fit(xx,y)

dtree.score(xx,y)
omanprediction=dtree.predict([[1.68,7.783333,3.55,16.066667,11.916667,332.156]])

print(float(omanprediction))
from sklearn import tree

plt.figure(figsize=(20,15))

tree.plot_tree(dtree,filled=True,feature_names=xx.columns)

plt.show()
from eli5.sklearn import explain_decision_tree

explain_decision_tree(dtree,feature_names=list(xx.columns))
X[["Stunting","Underweight","U5 Population ('000s)"]].corr()
X[["Stunting","Overweight","U5 Population ('000s)"]].corr()
X[["Income Classification","Overweight",]].corr()
X[X["country"].str.contains("turk",case=False)]
X.sort_values(by="Severe Wasting",ascending=True)[0:20]
X[X["Income Classification"]==2].sort_values(by="Severe Wasting",ascending=True)[0:20]
X[X["Income Classification"]==2].sort_values(by="Underweight",ascending=True)[0:26]