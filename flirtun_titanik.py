# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
df = pd.read_csv('/kaggle/input/titanic/gender_submission.csv')
train = pd.read_csv('/kaggle/input/titanic/train.csv')
test = pd.read_csv('/kaggle/input/titanic/test.csv')


train
train.info()
train = train.drop('Cabin', axis = 1)
train['Age'] = train['Age'].fillna(train['Age'].median())
test.info()
test = test.drop('Cabin', axis = 1)
test['Age'] = test['Age'].fillna(test['Age'].median())
test['Fare'] = test['Fare'].fillna(test['Fare'].median())
df.info()

test  = test.merge(df, on='PassengerId', how='left')
data = train.append(test)
data = data.reset_index().drop('index',axis=1)
data = data.dropna()
data.info()
data.hist(figsize=(12,10), bins =100)
data['Fare'].max()
q3_a = np.percentile(data['Fare'], 75, interpolation = 'midpoint') 
q1_a = np.percentile(data['Fare'], 25, interpolation = 'midpoint') 
iqr_a = q3_a - q1_a 
iqr_a
test = data.query('Fare <= @iqr_a')


data

from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import OrdinalEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
data_ohe = pd.get_dummies(data, drop_first=True)

#target = data_ohe['Exited']
#features = data_ohe.drop('Exited', axis=1)

data_train, data_test = train_test_split(data_ohe, test_size=0.4, random_state=12345)
#train = train.drop('PassengerId', axis = 1)
#test = test.drop('PassengerId', axis = 1)
#data_ohe_train = pd.get_dummies(train, drop_first=True)
#data_ohe_test = pd.get_dummies(test, drop_first=True)

features_test = data_test.drop(['Survived'], axis=1)
target_test =  data_test['Survived']

features_train = data_train.drop(['Survived'], axis=1)
target_train = data_train['Survived']


features_test
features_test.shape

features_train.shape
numeric = ['Age', 'Fare']

scaler = StandardScaler()
scaler.fit(features_train[numeric])
features_train[numeric] = scaler.transform(features_train[numeric])

features_test[numeric] = scaler.transform(features_test[numeric])

from sklearn.utils import shuffle
target_train.value_counts()

target_test.value_counts()
def upsample(features, target, repeat):
    features_zeros = features[target == 0]
    features_ones = features[target == 1]
    target_zeros = target[target == 0]
    target_ones = target[target == 1]

    features_upsampled = pd.concat([features_zeros] + [features_ones] * repeat)
    target_upsampled = pd.concat([target_zeros] + [target_ones] * repeat)
    
    features_upsampled, target_upsampled = shuffle(
        features_upsampled, target_upsampled, random_state=12345)
    
    return features_upsampled, target_upsampled

features_upsampled, target_upsampled = upsample(features_train, target_train, 2)
for depth in range(1, 23):
    model_d = DecisionTreeClassifier(random_state=12345, max_depth=depth, class_weight='balanced') 
    model_d.fit(features_upsampled, target_upsampled)
    predictions_test = model_d.predict(features_test)
    probabilities_test = model_d.predict_proba(features_test)
    probabilities_one_test = probabilities_test[:, 1]

    auc_roc = roc_auc_score(target_test, probabilities_one_test)
    print("max_depth =", depth, 'auc_roc =', auc_roc, "f1_score =", end='') 
    print(f1_score(target_test, predictions_test))
for estimators in range(1, 101):    
    for depth in range(1, 66):
        model_f = RandomForestClassifier(random_state=12345, max_depth=depth, n_estimators=estimators, class_weight='balanced')
        model_f.fit(features_upsampled, target_upsampled)
        predictions_test = model_f.predict(features_test)
        probabilities_test = model_f.predict_proba(features_test)
        probabilities_one_test = probabilities_test[:, 1]
        auc_roc = roc_auc_score(target_test, probabilities_one_test)
        print("max_depth =", depth, "estimators =", estimators, 'auc_roc =', auc_roc, "f1_score =", end='')
        print(f1_score(target_test, predictions_test))
from sklearn.metrics import precision_score, recall_score
model_f = DecisionTreeClassifier(random_state=12345, max_depth=4, class_weight='balanced') 
model_f.fit(features_upsampled, target_upsampled)
probabilities_test = model_f.predict_proba(features_test)
probabilities_one_test = probabilities_test[:, 1]

for threshold in np.arange(0, 0.8, 0.02):
    predicted_test = probabilities_one_test > threshold
    precision = precision_score(target_test, predicted_test)
    recall = recall_score(target_test, predicted_test)
    f1 = f1_score(target_test, predicted_test)
    auc_roc = roc_auc_score(target_test, predicted_test)
    

    print("Порог = {:.2f} | Точность = {:.3f}, Полнота = {:.3f}, F1 = {:.3f}, AUX = {:.3f}".format(
        threshold, precision, recall, f1, auc_roc))
model_d = DecisionTreeClassifier(random_state=12345, max_depth=4, class_weight='balanced') 
model_d.fit(features_upsampled, target_upsampled)
predictions_test = model_d.predict(features_test)
probabilities_test = model_d.predict_proba(features_test)
probabilities_one_test = probabilities_test[:, 1]

auc_roc = roc_auc_score(target_test, probabilities_one_test)
print('auc_roc =', auc_roc, "f1_score =", end='') 
print(f1_score(target_test, predictions_test))