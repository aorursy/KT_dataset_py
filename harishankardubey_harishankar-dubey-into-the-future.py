import numpy as np

import pandas as pd
train,test = pd.read_csv('../input/dataset/train.csv'),pd.read_csv('../input/dataset/test.csv')
train
test
train_10 = train.head(10)

train_10
test_10 = test.head(10)

test_10
train_details,test_details = test.describe(),test.describe()
train_details
train.info()
test.info()
correlation = train.corr()
import seaborn as sns
sns.heatmap(correlation, annot=True)
sns.pairplot(train, kind='reg')
sns.boxenplot(x='feature_1', data=train)
sns.boxenplot(y='feature_2', data=train)
sns.jointplot(x=train['feature_1'],y=train['feature_2'],kind='reg')
sns.heatmap(train.isnull())
from sklearn.linear_model import LinearRegression
X,Y = train[['feature_1']],train[['feature_2']]
X
Y
model = LinearRegression()

model.fit(X,Y)
print(model.intercept_)

print(model.coef_)
sns.heatmap(test.isnull())
test_x = test['feature_1']
test_x
test_X = pd.DataFrame(test_x)

test_Y=model.predict(test_X)
test_Y
test_Y.shape
test_Y.reshape(375,)
test_X.shape
result = pd.DataFrame(test)
result
result['feature_2']=test_Y
result
sns.pairplot(data=result)
sns.jointplot(x=result['feature_1'],y=result['feature_2'])
result.to_csv('/kaggle/working/result.csv')