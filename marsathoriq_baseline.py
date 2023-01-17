import pandas as pd

import numpy as np

from sklearn.preprocessing import LabelEncoder

from sklearn.linear_model import LogisticRegression
train = pd.read_csv('../input/seleksidukungaib/train.csv')

test = pd.read_csv('../input/seleksidukungaib/test.csv')
train.head()
train.shape, test.shape
train.fillna(0,inplace=True)
data = pd.concat([train,test],ignore_index=True)
date = ['date_collected','date']

bin = ['premium','super','pinEnabled']

col = date + bin

le = LabelEncoder()

for i in col: 

    data[i] = le.fit_transform(list(data[i].values))
train = data[~data.isChurned.isnull()]

test = data[data.isChurned.isnull()]
X_train = train.drop(['isChurned','idx'],axis=1)

y_train = train['isChurned']
model = LogisticRegression()

model.fit(X_train,y_train)

pred = model.predict(test.drop(['isChurned','idx'],axis=1))
submission = pd.DataFrame({'idx':test['idx'],'isChurned':pred.astype(int)})

submission.to_csv('submission.csv',index=False)
submission.head()