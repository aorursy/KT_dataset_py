import sklearn

import numpy as np

import pandas as pd

from matplotlib import pyplot as pltq

from sklearn import preprocessing as pp



def featurize(data):

    trn = data.copy()



    trn['Age'][np.isnan(trn['Age'])] = trn['Age'].median()



    sle = pp.LabelEncoder()

    sle.fit(['male', 'female'])

    trn['Sex_m'] = sle.transform(trn['Sex'])



    trn['Embarked_m'] = trn['Embarked']

    trn['Embarked_m'] = trn['Embarked_m'].fillna(trn['Embarked_m'].mode().unique().any())

    ele = pp.LabelEncoder()

    ele.fit(trn['Embarked_m'].unique())

    trn['Embarked_m'] = ele.transform(trn['Embarked_m'])



    cnt, bins = np.histogram(trn.loc[:, ['Age']], bins=4)

    trn['Age_m'] = np.digitize(trn['Age'], bins)



    trn['HasParents'] = trn['Parch'].map(lambda x: 0 if x == 0 else 1)

    trn['HasSiblings'] = trn['SibSp'].map(lambda x: 0 if x == 0 else 1)



    trn_m = trn[['Pclass','Sex_m','Age_m','Embarked_m','HasParents','HasSiblings']]

    return trn_m









trn2 = trn1.copy(deep=True)

trn2['Survived'] = trn['Survived']

trn2  = trn2.drop(labels=['HasParents','HasSiblings'],axis=1)





pd.options.display.width=200



trn = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')



trn1 = featurize(trn)

test1 = featurize(test)
from sklearn.linear_model.logistic import LogisticRegression

from sklearn.model_selection import cross_val_score





lr = LogisticRegression()

lr.fit(trn1,trn['Survived'])



cross_val_score(lr,trn1,trn['Survived'],cv=5)
from sklearn.ensemble import AdaBoostClassifier

trn1.corr()
from pgmpy.estimators import HillClimbSearch as HCS, BicScore



h = HCS(trn2,scoring_method=BicScore(trn2))

h.estimate(max_indegree=2)

h.estimate(max_indegree=2).edges()