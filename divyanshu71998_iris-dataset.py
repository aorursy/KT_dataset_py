import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn import preprocessing

import warnings

warnings.simplefilter("ignore")

%matplotlib inline
data=pd.read_csv("../input/iris/Iris.csv")
data.head()
data.columns
data.shape
data.isin([0]).sum()
species=data['Species'].unique()

species
sns.set_style("whitegrid")

g=sns.FacetGrid(data,hue='Species',size=8)

g.map(plt.scatter,'SepalLengthCm', 'SepalWidthCm') 

g.add_legend()

plt.show()

sns.set_style("whitegrid")

g=sns.FacetGrid(data,hue='Species',size=8)

g.map(plt.scatter, 'PetalLengthCm', 'PetalWidthCm') 

g.add_legend()

plt.show()
sns.lmplot(x='SepalLengthCm', y='SepalWidthCm',col="Species",data=data)

sns.lmplot(x='PetalLengthCm', y='PetalWidthCm',col="Species",data=data)
label_encoder = preprocessing.LabelEncoder() 

data['Species']= label_encoder.fit_transform(data['Species']) 

data['Species'].unique()



input_data=data.iloc[:,[0,1,2,3]].values

target=data.iloc[:,-1].values

from sklearn.model_selection import StratifiedKFold

def skf(model,a,b):

    f=StratifiedKFold(n_splits=30)

    scores=[]

    for train_index,test_index in f.split(a,b):

        x_train ,x_test,y_train,y_test=a[train_index],a[test_index],b[ train_index],b[test_index]

        model.fit(x_train,y_train)

        scores.append(model.score(x_test,y_test))

    return scores
from sklearn.svm import SVC

from sklearn import tree

from sklearn.linear_model import LogisticRegression
np.array(skf(SVC(),input_data,target)).mean()

    
np.array(skf(tree.DecisionTreeClassifier(),input_data,target)).mean()
np.array(skf(LogisticRegression(max_iter=150),input_data,target)).mean()
from sklearn.ensemble import RandomForestClassifier

np.array(skf(RandomForestClassifier(n_estimators=100),input_data,target)).mean()