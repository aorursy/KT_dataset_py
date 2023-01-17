# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))

import json



# Any results you write to the current directory are saved as output.
train = pd.read_csv("../input/train/train.csv")

test  = pd.read_csv("../input/test/test.csv")
sentiment_files = os.listdir("../input/train_sentiment")
documentSentiments = {}

for petid in train.PetID:

    if (petid + '.json') in sentiment_files:

        documentSentiments[petid] = json.load(open("../input/train_sentiment/" + petid + '.json'))['documentSentiment']

    else:

        documentSentiments[petid] = {'magnitude':None,'score':None}
documentSentiments = pd.DataFrame(pd.Series(documentSentiments))
documentSentiments.columns = ["DocumentSentiments"]

documentSentiments.index.name = 'PetID'
train = pd.merge(train, documentSentiments, on='PetID', how='inner')

train['sentiment_score'] = train['DocumentSentiments'].map(lambda x:x['score'])

train['sentiment_magnitude'] = train['DocumentSentiments'].map(lambda x:x['magnitude'])
train['sentiment_score'].fillna(train['sentiment_score'].median(),inplace=True)

train['sentiment_magnitude'].fillna(train['sentiment_magnitude'].median(),inplace=True)
from sklearn.metrics import cohen_kappa_score, accuracy_score



def classification_report_custom(y_true,y_pred):

    print("Accuracy : " + str(accuracy_score(y_true,y_pred)))

    print("Kappa : " + str(cohen_kappa_score(y_true,y_pred,weights='quadratic')))
target = train['AdoptionSpeed']



from sklearn.model_selection import train_test_split



x_train, x_valid, y_train, y_valid = train_test_split(train.drop("AdoptionSpeed",axis=1),

                                                    target,

                                                    test_size=0.1,

                                                    random_state=0,

                                                    stratify=target)
predictions = np.random.randint(0,4,size=target.shape)
classification_report_custom(target,predictions)
train_features = x_train.select_dtypes([int,float])

valid_features = x_valid.select_dtypes([int,float])
categorical_features = ['Type','Age','Breed1','Breed2','Gender','Color1',

                        'Color2','MaturitySize','FurLength','Vaccinated',

                        'Dewormed','Sterilized','Health','State']



train_features[categorical_features].astype('category',inplace=True)

valid_features[categorical_features].astype('category',inplace=True);
from sklearn.ensemble import AdaBoostRegressor

cf2 = AdaBoostRegressor(

    n_estimators=1500,

    learning_rate=.1,

    random_state=2)

cf2.fit(train_features,y_train)

from sklearn.ensemble import GradientBoostingRegressor

cf3 = GradientBoostingRegressor(

                                 learning_rate = 0.1,

                                 n_estimators = 1500,

                                 warm_start = True,

                                 random_state= 2 )

cf3.fit(train_features,y_train)

from xgboost import XGBClassifier



cf4 = XGBClassifier(booster='gbtree',max_depth=3)
cf4.fit(train_features,y_train)
valid_preds = np.round((cf2.predict(valid_features)+cf3.predict(valid_features))/2)
valid_preds2 = cf4.predict(valid_features)
classification_report_custom(y_valid,np.floor(valid_preds))
classification_report_custom(y_valid,valid_preds2)
testdocumentSentiments = {}

testsentiment_files = os.listdir("../input/test_sentiment")

for petid in test.PetID:

    if (petid + '.json') in testsentiment_files:

        testdocumentSentiments[petid] = json.load(open("../input/test_sentiment/" + petid + '.json'))['documentSentiment']

    else:

        testdocumentSentiments[petid] = {'magnitude':None,'score':None}
testdocumentSentiments = pd.DataFrame(pd.Series(testdocumentSentiments))

testdocumentSentiments.columns = ["DocumentSentiments"]

testdocumentSentiments.index.name = 'PetID'

test = pd.merge(test, testdocumentSentiments, on='PetID', how='inner')

test['sentiment_score'] = test['DocumentSentiments'].map(lambda x:x['score'])

test['sentiment_magnitude'] = test['DocumentSentiments'].map(lambda x:x['magnitude'])

test['sentiment_score'].fillna(test['sentiment_score'].median(),inplace=True)

test['sentiment_magnitude'].fillna(test['sentiment_magnitude'].median(),inplace=True)
categorical_features = ['Type','Age','Breed1','Breed2','Gender','Color1',

                        'Color2','MaturitySize','FurLength','Vaccinated',

                        'Dewormed','Sterilized','Health','State']



test[categorical_features].astype('category',inplace=True)



test = test.select_dtypes([int,float])

test_predictions = cf4.predict(test)
test = pd.read_csv("../input/test/test.csv")

submission = pd.DataFrame(

    {'PetID': test['PetID'].values, 

     'AdoptionSpeed': test_predictions.astype(np.int32)})

submission.head()

submission.to_csv('submission.csv', index=False)