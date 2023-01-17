import numpy as np

import pandas as pd



import matplotlib.pyplot as plt

import seaborn as sns



import sklearn

from sklearn.preprocessing import LabelEncoder, StandardScaler

from sklearn.model_selection import StratifiedShuffleSplit, ShuffleSplit, GridSearchCV



from sklearn.metrics import accuracy_score, log_loss

from sklearn.neighbors import KNeighborsClassifier

from sklearn.svm import SVC

from sklearn.ensemble import RandomForestClassifier



%matplotlib inline
train = pd.read_csv('../input/leaf-classification/train.csv', delimiter=',')

test = pd.read_csv('../input/leaf-classification/test.csv', delimiter=',')
train.head()
train.info() # 990 samples, 192 features
train['species'].nunique() # 99 unique species
train['species'].value_counts() # each species has 10 samples in training set
test.head()
test.info() # Target: classify 594 test samples into 99 species
le = LabelEncoder().fit(train['species'])
# encode species in training set

train['label'] = le.transform(train['species'])
# drop id & species columns. seperate labels in training set

labels = train['label']

train_df = train.drop(columns=['id','species','label'])
from sklearn.preprocessing import StandardScaler



scaler = StandardScaler().fit(train_df)

train_scale = pd.DataFrame(scaler.transform(train_df))
# create train & validation set. In this case, we dont want just simple random sampling but stratification because of large number of classes (99)

# stratification will make sure there's an equal number of samples per class in training set

sss = StratifiedShuffleSplit(n_splits=10, test_size=0.2, random_state=42)

for train_index, val_index in sss.split(train_scale,  labels):

    x_train, x_val = train_scale.iloc[train_index], train_scale.iloc[val_index]

    y_train, y_val = labels.iloc[train_index], labels.iloc[val_index]
print(x_train.shape,y_train.shape)

print(x_val.shape, y_val.shape)
y_train.value_counts() # each class has 8 samples in training set
test_id = test['id']

test_features = test.drop('id', axis=1)

test_features_scale = scaler.transform(test_features)
cv_sets = ShuffleSplit(n_splits=10,test_size=0.20,random_state=42)

classifiers = [RandomForestClassifier(), SVC(), KNeighborsClassifier()]

params = [{'n_estimators' : [3,10,30], 'max_features':[2,4,6,8]},

          {'kernel':('linear','poly','sigmoid','rbf'),'C':[0.01,0.05,0.025,0.07,0.09,1.0], 'gamma':['scale'], 'probability':[True]},

          {'n_neighbors': [3,5,7,9]}]
best_estimators = []

for classifier, param in zip(classifiers, params):

    grid = GridSearchCV(classifier,param,cv=cv_sets)

    grid = grid.fit(x_train,y_train)

    best_estimators.append(grid.best_estimator_)
best_estimators 
for estimator in best_estimators:

    estimator.fit(x_train, y_train)

    name = estimator.__class__.__name__

    

    print("="*30)

    print(name)

    

    print('****Results****')

    print('**Training set**')

    train_predictions = estimator.predict(x_train)

    acc = accuracy_score(y_train, train_predictions)

    print("Accuracy: {:.4%}".format(acc))

    train_predictions = estimator.predict_proba(x_train)

    ll = log_loss(y_train, train_predictions)

    print("Log Loss: {}".format(ll))

    

    print('**Validation set**')

    train_predictions = estimator.predict(x_val)

    acc = accuracy_score(y_val, train_predictions)

    print("Accuracy: {:.4%}".format(acc))

    train_predictions = estimator.predict_proba(x_val)

    ll = log_loss(y_val, train_predictions)

    print("Log Loss: {}".format(ll))

    

print("="*30)
pred = best_estimators[2].predict_proba(test_features_scale) # KNeighbors classifer model
submission = pd.DataFrame(pred, index = test_id, columns = le.classes_ )
submission.to_csv('submission.csv')