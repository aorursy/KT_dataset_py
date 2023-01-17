# load basic neccesities

import os



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import seaborn as sns

import matplotlib.pyplot as plt

sns.set()



pd.options.display.max_columns = 200
from sklearn.metrics import confusion_matrix, accuracy_score

from sklearn.preprocessing import StandardScaler

from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import train_test_split
train = pd.read_csv("../input/learn-together/train.csv")

test = pd.read_csv("../input/learn-together/test.csv")
print(train.shape)

print(test.shape)
train.head(5)
train.describe()
test.describe()
classes = set(train.Cover_Type)

n_classes = len(classes)

print(classes, ' - ', n_classes)
sns.countplot(train.Cover_Type)
# check if there are NaN cells

train.count()
test.head(5)
test.count()
df_train = train.drop('Id', axis=1).copy()

train_cols = df_train.columns.tolist()

train_cols = train_cols[-1:] + train_cols[:-1]

df_train = df_train[train_cols]

corr = df_train.corr()
# Generate a mask for the upper triangle

mask = np.zeros_like(corr, dtype=np.bool)

mask[np.triu_indices_from(mask)] = True



# Set up the matplotlib figure

f, ax = plt.subplots(figsize=(15, 15))



# Generate a custom diverging colormap

cmap = sns.diverging_palette(220, 10, as_cmap=True)



# Draw the heatmap with the mask and correct aspect ratio

sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,

            square=True, linewidths=.5, cbar_kws={"shrink": .5})
df_train = train.drop('Id', axis=1).copy()

train_cols = df_train.columns.tolist()

train_cols = train_cols[-1:] + train_cols[:-1]

df_train[train_cols[15:]].sum()/df_train.shape[0]*100
test_cols = test.columns.tolist()

test[test_cols[14:]].sum()/test.shape[0]*100
# lets start with Light GBM

import lightgbm as lgb

from lightgbm import LGBMClassifier
# almost basic parameters. some adjustments to prevent overfitting

LGB_PARAMS_1 = {

    'objective': 'multiclass',

    "num_class" : n_classes,

    'metric': 'multiclass',

    'verbosity': -1,

    'boosting_type': 'gbdt',

    'learning_rate': 0.1,

    'num_leaves': 64,

#    'num_leaves': 128,

    'max_depth': -1,

#    'subsample_freq': 10,

    'subsample_freq': 1,

    'subsample': 0.5,

    'bagging_seed': 1970,

    'reg_alpha': 0.3,

#    'reg_alpha': 0.1,

    'reg_lambda': 0.3,

    'colsample_bytree': 0.90

}
# basic parameters

LGB_PARAMS = {

    'objective': 'multiclass',

    "num_class" : n_classes,

    'metric': 'multiclass',

    'reg_alpha': 0.9,

    'reg_lambda': 0.9,

    'verbosity': -1

}
X_data = train.drop(['Id','Cover_Type','Soil_Type7','Soil_Type15'], axis = 1).copy()

y_data = train.Cover_Type

cols = X_data.columns

print(X_data.shape)

print(y_data.shape)
# split train set into train and validation

X_train, X_val, y_train, y_val = train_test_split(X_data, y_data, test_size=0.2, random_state=128)
# number of samples in Validation dataset

N_valid = y_val.shape[0]
LB_model = LGBMClassifier(**LGB_PARAMS, n_estimators=1500, n_jobs = -1)
LB_model.fit(X_train, y_train, 

        eval_set=[(X_train, y_train), (X_val, y_val)], eval_metric='multiclass',

        verbose=10, early_stopping_rounds=50)
LG_pred = LB_model.predict(X_val)

print(accuracy_score(LG_pred, y_val))
confusion_matrix(y_val, LG_pred)
df_importance = pd.DataFrame({'feature': cols, 'importance': LB_model.feature_importances_})
# Lets plot it

plt.figure(figsize=(6, 10))

sns.barplot(x="importance", y="feature", data=df_importance.sort_values('importance', ascending=False));

new_features = df_importance.loc[df_importance.importance>100].sort_values('importance', ascending=False)

new_columns = new_features.feature
KNN_clf = KNeighborsClassifier(n_neighbors = 5, weights ='distance', metric = 'minkowski', p = 2)

KNN_clf.fit(X_train, y_train)

yk_pred = KNN_clf.predict(X_val)

cm = confusion_matrix(y_val, yk_pred)

print(accuracy_score(yk_pred, y_val))
cm
from sklearn.ensemble import RandomForestClassifier



# just default settings

#RF_clf = RandomForestClassifier(n_estimators=1000, max_depth=10,

RF_clf = RandomForestClassifier(n_estimators=1000, 

                              random_state=1970)

RF_clf.fit(X_train, y_train) 
RF_pred = RF_clf.predict(X_val)

accuracy_score(y_val, RF_pred)
confusion_matrix(y_val, RF_pred)
df_importance = pd.DataFrame({'feature': cols, 'importance': RF_clf.feature_importances_})
# Lets plot it

plt.figure(figsize=(6, 10))

sns.barplot(x="importance", y="feature", data=df_importance.sort_values('importance', ascending=False));
submission_id = test.Id

test.drop(['Id','Soil_Type7','Soil_Type15'], axis = 1, inplace = True)

prediction = LB_model.predict(test)

submission = pd.DataFrame({'Id': submission_id, 'Cover_Type': prediction})

submission.to_csv('submission_LGBM.csv', index = False)
#lets check Ensemble prediction on validation set

RF_val = RF_clf.predict_proba(X_val)

KNN_val = KNN_clf.predict_proba(X_val)

LG_val = LB_model.predict_proba(X_val)



val_probs = (RF_val + LG_val + KNN_val)/3

val_preds = np.argmax(val_probs, axis=1)+1



print(accuracy_score(y_val, val_preds))

# lets check the distribution of predicted classes

# as we know all classes shall be evenly split. at least they shall be close to it

sns.countplot(val_preds)
# final Test set prediction

RF_probs = RF_clf.predict_proba(test)

KNN_probs = KNN_clf.predict_proba(test)

LG_probs = LB_model.predict_proba(test)
# averaging the prediction

final_probs = (RF_probs + LG_probs + KNN_probs)/3

final_preds = np.argmax(final_probs, axis=1)+1
# lets see the distribution of predicted classes

# hint - as we know class 1 shall be around 37%, and as I've checked it class 2 shall be around 50% actually

sns.countplot(final_preds)
submission = pd.DataFrame({'Id': submission_id, 'Cover_Type': final_preds})

submission.to_csv('submission_ensemble.csv', index = False)
