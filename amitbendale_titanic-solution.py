import numpy as np

import pandas as pd

train_data = pd.read_csv("../input/train.csv",dtype=str)

test_data = pd.read_csv("../input/test.csv",dtype=str)

submit_data = pd.read_csv("../input/gender_submission.csv",dtype=str)
train_data.head()
test_data.head()
submit_data.head()
features = ['Pclass','Sex','SibSp','Parch','Fare']

train_data = train_data[features+['Survived']]

train_data['Sex']=train_data['Sex'].apply(lambda x: 0 if x=='male' else 1)

train_data = train_data.fillna(0)

test_data = test_data[['PassengerId']+features]

test_data['Sex']=test_data['Sex'].apply(lambda x: 0 if x=='male' else 1)

test_data = test_data.fillna(0)
from sklearn.linear_model import LogisticRegression

lr = LogisticRegression()

lr.fit(train_data[features],train_data['Survived'])

lr.score(train_data[features],train_data['Survived'])
lr.coef_
outputDf=pd.DataFrame()

outputDf['PassengerId']=test_data.PassengerId

outputDf['Survived'] = lr.predict(test_data[features])

outputDf.head()

outputDf.to_csv("submission.csv",index=False)