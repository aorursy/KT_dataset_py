import pandas as pd

import numpy as np

data = pd.read_csv('../input/database/train.csv')

data.drop('Id', axis=1, inplace=True)

data.head()
X = data.drop(['Attrition'], axis=1)

y = data['Attrition']

Cat_col = [col for col in X.columns if X[col].dtype=='object']

Cat_col
from sklearn.preprocessing import OneHotEncoder

ohe = OneHotEncoder(handle_unknown='ignore', sparse=False)

ohe_X = pd.DataFrame(ohe.fit_transform(X[Cat_col]))

ohe_X.index = X.index

X_num = X.drop(Cat_col, axis=1)

X_training = pd.concat([X_num, ohe_X], axis=1)

X_training.head()
from sklearn.ensemble import ExtraTreesClassifier

import matplotlib.pyplot as plt

model = ExtraTreesClassifier()

model.fit(X_training,y)
print(model.feature_importances_)
feat_importances = pd.Series(model.feature_importances_, index=X_training.columns)



plt.figure(figsize=(10,10))

feat_importances.nlargest(30).plot(kind='barh')

plt.show()
top30 = list(feat_importances.nlargest(30).index)
X_top30 = X_training[top30]

X_top30.head()
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X_top30, y, test_size=0.3, random_state=420)
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(n_estimators=90, criterion='entropy', random_state=420)

model.fit(X_train, y_train)

predict_train = model.predict_proba(X_train)[:,1]

predict_test  = model.predict_proba(X_test)[:,1]
model.score(X_test, y_test)
test_ = pd.read_csv('test.csv')

test = test_.drop('Id', axis=1)

test.head()
ohe_test = pd.DataFrame(ohe.fit_transform(test[Cat_col]))

ohe_test.index = test.index

test_num = test.drop(Cat_col, axis=1)

test_data = pd.concat([test_num, ohe_test], axis=1)



test_data_top30 = test_data[top30]
prediction = model.predict_proba(test_data_top30)[:,1]
import matplotlib.pyplot as plt

from sklearn.metrics import roc_curve, auc

fpr, tpr, threshold = roc_curve(y_test, predict_test)

auc = auc(fpr, tpr)



plt.figure(figsize=(10,10))

plt.plot(fpr, tpr, linestyle='-', label='auc = %0.3f'%auc)



plt.xlabel('False Positive Rate -->')

plt.ylabel('True Positive Rate -->')

plt.legend()

plt.show()
fig, (ax1, ax2) = plt.subplots(2,1,figsize=(15,15))



ax1.hist(model.predict_proba(X_top30),bins=100)

ax2.hist(model.predict_proba(test_data_top30), bins=100)

plt.show()
output = pd.Series(prediction)

output_final = pd.concat([test_['Id'], output], axis=1)

output_final.columns=['Id', 'Attrition']

output_final.set_index('Id',inplace=True)

output_final.describe()