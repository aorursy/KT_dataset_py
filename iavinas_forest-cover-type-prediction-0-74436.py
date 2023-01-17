import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

sns.set(style="darkgrid")

%matplotlib inline

from sklearn.model_selection import GridSearchCV , train_test_split , cross_val_score , StratifiedKFold

from sklearn.metrics import roc_curve, auc

import os

import warnings

warnings.filterwarnings('ignore')



print(os.listdir("../input/"))
train = pd.read_csv('../input/train.csv' , index_col='Id')

test = pd.read_csv('../input/test.csv'  , index_col='Id')

labels = train.Cover_Type

train.drop('Cover_Type' , axis = 1 , inplace =True)

train.head(3)
### Cover type
# names =  {

#     1 : 'Spruce', 

#     2 : 'Lodgepole',

#     3 : 'Ponderosa',

#     4 : 'Cottonwood',

#     5 : 'Aspen',

#     6 : 'Douglas',

#     7 : 'Krummholz' 

# }

# train.Cover_Type = train.Cover_Type.map(names)
ax = sns.countplot(x = labels)
train.columns
train.isna().sum()
test.isna().sum()
from sklearn.preprocessing import RobustScaler

scaler = RobustScaler()

train = scaler.fit_transform(train)

test = scaler.transform(test)
from sklearn.ensemble import GradientBoostingClassifier

clf = GradientBoostingClassifier(learning_rate=0.65 , n_estimators= 250 ,max_depth = 9  )

cross_val_score(clf , train , labels , cv = 3)
clf.fit(train, labels)
pre = clf.predict(test)
ansdf = pd.read_csv('../input/sampleSubmission.csv')

ansdf['Cover_Type'] = pre

ansdf.to_csv('submit.csv', index = False)
ansdf.head()