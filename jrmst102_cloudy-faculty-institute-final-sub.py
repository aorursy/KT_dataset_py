import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
test.shape
train.shape
train.head()
test.head()
train.columns.values
test.columns.values
train.describe()
test.describe()
#checking missing values
train.isnull().sum()
test.isnull().sum()
train.info()
test.info()
train['Outcome'].hist(bins = 20)
plt.show()
train.Outcome.unique()
train.Outcome.value_counts()
X = train.drop(['Outcome'], axis = 1)
y = train.Outcome
X.shape
y.shape
y.head()
X.head()
train.shape
from sklearn.ensemble import GradientBoostingClassifier
clf = GradientBoostingClassifier(n_estimators = 105)
clf.fit(X,y)
predicted = clf.predict(test)
print(predicted)
test.shape
predicted.shape
output = pd.DataFrame(predicted,columns = ['Outcome'])
test = pd.read_csv('../input/test.csv')
output['Id'] = test['Id']
output[['Id','Outcome']].to_csv('submission_cloudy10.csv', index = False)
output.head()
