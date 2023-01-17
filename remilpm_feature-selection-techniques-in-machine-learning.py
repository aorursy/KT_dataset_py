#Feature Selection techniques in machine learning by Univariate Selection

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.feature_selection import SelectKBest

from sklearn.feature_selection import chi2

Data_Set1= pd.read_csv("../input/Titanic.csv")

Data_Set1.tail()

Data_Set2=Data_Set1[['PassengerId','Pclass','Parch','SibSp','Age','Survived']]

cond1=Data_Set2['Age']>0

Data_Set2=Data_Set2[cond1]

x=Data_Set2[['PassengerId','Pclass','Parch','SibSp','Age']]

x.head()
y=Data_Set2[['Survived']]

y.head()
x.shape,y.shape
#apply SelectKBest class to extract top 5 best features

bestfeatures = SelectKBest(score_func=chi2, k=5)

fit = bestfeatures.fit(x,y)

dfscores = pd.DataFrame(fit.scores_)

dfcolumns = pd.DataFrame(x.columns)

#concat two dataframes for better visualization 

featureScores = pd.concat([dfcolumns,dfscores],axis=1)

featureScores.columns = ['Specs','Score']  #naming the dataframe columns

print(featureScores.nlargest(5,'Score'))  #print 5 best features