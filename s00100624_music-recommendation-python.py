"""Import all necessary packages"""
import pandas as pd
import os
import re
import pickle
from catboost import CatBoostClassifier
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, roc_auc_score
"""read data from source"""
data = {}
for file in os.listdir("../input"):
    data[re.sub(".csv", "", file)] = pd.read_csv("../input/"+file)
"""Extract all necessary data, merge them and see result"""
# join data into single table
test = data['test'].merge(data['members'], how='left', on='msno')
test = test.merge(data['songs'], how='left', on='song_id')

train = data['train'].merge(data['members'], how='left', on='msno')
train = train.merge(data['songs'], how='left', on='song_id')

del data

#get a glimpse of how the dataset looks like
train.head()
"""check if there is inconsistant data type in data"""
for x in train.columns:
    print(x + ": " + train[x].dtype.kind)
"""Convert datatype based on observation"""
#it was discovered that 'city', 'registered_via' and 'language' should be categorical variables, convert their data type
for column in ['city', 'registered_via', 'language']:
    train[column] = train[column].astype('str')
    test[column] = test[column].astype('str')

#it was discovered that 'registration_init_time' and 'expiration_date' should be datetime object, convert their data type
for column in ['registration_init_time', 'expiration_date']:
    train[column] = pd.to_datetime(train[column], format='%Y%m%d')
    train['Year_' + column] = train[column].dt.year.astype(str)
    train['Month_' + column] = train[column].dt.month.astype(str)
    train['Day_' + column] = train[column].dt.day.astype(str)
    train.drop([column], axis=1, inplace=True)
    
    test[column] = pd.to_datetime(test[column], format='%Y%m%d')
    test['Year_' + column] = test[column].dt.year.astype(str)
    test['Month_' + column] = test[column].dt.month.astype(str)
    test['Day_' + column] = test[column].dt.day.astype(str)
    test.drop([column], axis=1, inplace=True)
"""check if there is inconsistant data type in data again"""
for x in train.columns:
    print(x + ": " + train[x].dtype.kind)
"""fill NAs for categorical features"""
# fill na for all categorical features
for x in [x for x in train.columns if train[x].dtype.kind == 'O']:
    train[x].fillna('NA', inplace=True)
    
for x in [x for x in test.columns if test[x].dtype.kind == 'O']:
    test[x].fillna('NA', inplace =True)

# check which numerical feature(s) still consist of null value and fill na for each of them using using appropriate method
for x in [x for x in train.columns if train[x].dtype.kind in ['f','i']]:
    print(x + ": ", train[x].isnull().sum())
for x in [x for x in test.columns if test[x].dtype.kind in ['f','i']]:
    print(x + ": ", test[x].isnull().sum())

train['song_length'] = train['song_length'].fillna(train['song_length'].median())
test['song_length'] = test['song_length'].fillna(test['song_length'].median())
"""convert all categorical features to 'category data type'"""
for x in [x for x in train.columns if train[x].dtype.kind == 'O']:
    train[x] = train[x].astype('category')

for x in [x for x in test.columns if test[x].dtype.kind == 'O']:
    test[x] = test[x].astype('category')
"""split training data into train, cv and test set"""
X = train.drop(['target'],axis=1)
y = train.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, stratify=y, random_state=123)

del X
del y
del train

X_train, X_cv, y_train, y_cv = train_test_split(X_train, y_train, test_size = 0.25, stratify=y_train, random_state=123)
"""prepare and train the model"""
cat_features_index = [i for i, x in enumerate(X_train.columns) if X_train[x].dtype.kind == 'O']
model = CatBoostClassifier(iterations = 50, learning_rate = 0.3, eval_metric='AUC', max_ctr_complexity=2, boosting_type = 'Plain', bootstrap_type= 'Bernoulli', use_best_model=True, random_seed=123)
model.fit(X_train, y_train, cat_features=cat_features_index, eval_set=(X_cv, y_cv))
"""Check result"""
y_pred = model.predict_proba(X_test)
print(roc_auc_score(y_test, y_pred[:,1]))
"""show the ROC curve"""
fpr, tpr, threshold = roc_curve(y_test, y_pred[:,1])
plt.scatter(x=fpr, y=tpr)
plt.show()