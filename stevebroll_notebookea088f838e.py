import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sb #plotting for heatmap

import sklearn.linear_model as sklm ##Includes Logistic Regression, which will be tested for predictive capability

from sklearn.preprocessing import Imputer 

from sklearn.pipeline import Pipeline

from sklearn.decomposition import PCA

train = pd.read_csv('../input/train.csv')

#train['Sex'] = pd.get_dummies(train['Sex'])

#train['Embarked'] = pd.get_dummies(train['Embarked'])
sb.heatmap(abs(train.drop('Survived',1).corr()), cmap = 'bwr') 
test = pd.read_csv('../input/test.csv')[['Pclass','Sex','Age','SibSp','Parch','Fare','Embarked']]

#test['Sex'] = pd.get_dummies(train['Sex'])

#test['Embarked'] = pd.get_dummies(train['Embarked'])
trainX = (train.drop('Survived',1))

trainY = train['Survived']
pca = PCA() #empty model space

imp = Imputer(missing_values='NaN',strategy="mean",axis=0) #only include first 10 components

logreg = sklm.LogisticRegression()#empty model space

pipeline = Pipeline([('imputer', imp), ('pca', pca),('logistic', logreg)])
fit = pipeline.fit(trainX,trainY)

predict = pipeline.predict(test)
predict