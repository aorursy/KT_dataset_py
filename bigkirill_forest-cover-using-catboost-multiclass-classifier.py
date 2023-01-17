import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.model_selection import train_test_split
from catboost import Pool, CatBoostClassifier
train_data=pd.read_csv('../input/train.csv')
labels=train_data['Cover_Type']
labels=labels-1
train_data.head()
plt.scatter(x=train_data['Horizontal_Distance_To_Fire_Points'], y=train_data['Cover_Type'],)
plt.show()
drop=['Cover_Type','Id']
train=train_data.drop(drop,axis=1)
scaler = MinMaxScaler()
scaler.fit(train)
train=scaler.transform(train)
train_X, test_X, train_y, test_y = train_test_split(train, labels, test_size=0.2, random_state=42)
train_pool = Pool(data=train_X, label=train_y)
test_pool = Pool(data=test_X, label=test_y.values) 
model = CatBoostClassifier(
    iterations=1000,
    learning_rate=0.1,
    random_strength=0.1,
    depth=8,
    loss_function='MultiClass',
    eval_metric='Accuracy',
    leaf_estimation_method='Newton'
)
model.fit(train_pool,plot=True,eval_set=test_pool)
test_data=pd.read_csv('../input/test.csv')
res=pd.read_csv('../input/sample_submission.csv')
test_data=test_data.drop(['Id'],axis=1)
test_data=scaler.transform(test_data)
k=model.predict(test_data)
k=k+1
res['Cover_Type']=k
res['Cover_Type'].value_counts()
res['Cover_Type']=res['Cover_Type'].astype(int)
res.to_csv('result.csv', index=False)