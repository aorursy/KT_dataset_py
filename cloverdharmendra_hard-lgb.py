# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
import numpy as np

#step-1-Import the important lib.
import pandas as pd
import json
#Step-2-Load The traning dataset
training_set1 = pd.read_csv("../input/train/train.csv",  encoding='utf-8')

train = training_set1[["Age", "Breed1", "Breed2","Color1", "Color2","MaturitySize","Quantity","Health","Fee","State", "PhotoAmt"]]


train_id = training_set1['PetID']
doc_sent_mag = []
doc_sent_score = []
nf_count = 0
for pet in train_id:
    try:
        with open('../input/train_sentiment/' + pet + '.json', 'r') as f:
            sentiment = json.load(f)
        doc_sent_mag.append(sentiment['documentSentiment']['magnitude'])
        doc_sent_score.append(sentiment['documentSentiment']['score'])
    except FileNotFoundError:
        nf_count += 1
        doc_sent_mag.append(-1)
        doc_sent_score.append(-1)

train.loc[:, 'doc_sent_mag'] = doc_sent_mag
train.loc[:, 'doc_sent_score'] = doc_sent_score
X = train.iloc[:,:].values
y = training_set1.iloc[:,-1].values



from lightgbm import LGBMClassifier
lgbm = LGBMClassifier(boosting_type= 'gbdt', objective='multiclass', random_state=2, n_estimators=200 , max_depth=100, learning_rate=0.055)

model = lgbm.fit(X, y)
test_set = pd.read_csv("../input/test/test.csv",  encoding='utf-8')

test = test_set[["Age", "Breed1", "Breed2","Color1", "Color2","MaturitySize","Quantity","Health","Fee","State", "PhotoAmt"]]
#training_set["State"] = training_set["State"].astype(int)
test_id = test_set["PetID"]
doc_sent_mag = []
doc_sent_score = []
nf_count = 0
for pet in test_id:
    try:
        with open('../input/test_sentiment/' + pet + '.json', 'r') as f:
            sentiment = json.load(f)
        doc_sent_mag.append(sentiment['documentSentiment']['magnitude'])
        doc_sent_score.append(sentiment['documentSentiment']['score'])
    except FileNotFoundError:
        nf_count += 1
        doc_sent_mag.append(-1)
        doc_sent_score.append(-1)

test.loc[:, 'doc_sent_mag'] = doc_sent_mag
test.loc[:, 'doc_sent_score'] = doc_sent_score
model.booster_.feature_importance(importance_type='gain')


test_actual = test.iloc[:,:].values

#predict actual result
test_actual = model.predict(test_actual)
sub=test_set.loc[:,['PetID']].copy()
sub.loc[:,'AdoptionSpeed']=test_actual
sub.to_csv('submission.csv', index=False)


sub

