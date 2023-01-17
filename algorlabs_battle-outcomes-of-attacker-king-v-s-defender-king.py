#import all needed libs

#from os import path

import pandas as pd

import numpy as np

import seaborn as sns

%matplotlib inline
#extract the dataset from local

#my_path = path.join('..','DataSets','game-of-thrones','battles.csv')

# from Kaggle

battles_df = pd.read_csv('../input/battles.csv')
# Replace string with numbers

battles_df.attacker_outcome.replace('win',1,inplace=True)

battles_df.attacker_outcome.replace('loss',0,inplace=True)
#Here outcome is kept 0 which is loss as default to fill NaN

battles_df.attacker_outcome.fillna(0,inplace=True)
#attacker king's army size and it's outcome

sns.barplot(x='attacker_king',y='attacker_size',hue='attacker_outcome',data=battles_df)
# a subset of battles_df for predictions of battle outcome between attacker king and defender king

atk_dfn_outcome = battles_df[['attacker_king','defender_king','attacker_outcome']]

#fill NaN values here with string

atk_dfn_outcome.attacker_king.fillna("None",inplace=True)

atk_dfn_outcome.defender_king.fillna("None",inplace=True)
#import sklean Decision Tree model

from sklearn.tree import DecisionTreeClassifier

Tree = DecisionTreeClassifier()

# import features preprocessing 

from sklearn import preprocessing

le = preprocessing.LabelEncoder()



atk_dfn_outcome.attacker_king = le.fit_transform(atk_dfn_outcome.attacker_king)

atk_dfn_outcome.defender_king = le.fit_transform(atk_dfn_outcome.defender_king)



x = atk_dfn_outcome[['attacker_king','defender_king']]

y= atk_dfn_outcome['attacker_outcome']
#Fit Tree with learning data

Tree.fit(x,y)
#Predict outcome of battles by entering king's corresponging number

Tree.predict([2,3])
from sklearn.cross_validation import cross_val_score

scores = cross_val_score(Tree,x,y,cv=10,scoring='accuracy')

scores.mean()*100