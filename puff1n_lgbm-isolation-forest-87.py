import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
base_dir = '/kaggle/input/health-insurance-cross-sell-prediction/'
import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV

from sklearn.metrics import roc_curve, roc_auc_score

from lightgbm import LGBMClassifier

from sklearn.ensemble import IsolationForest
base_dir + 'train.csv'
df = pd.read_csv(base_dir + 'train.csv', index_col='id')

df_test = pd.read_csv(base_dir + 'test.csv', index_col='id')

submission = pd.read_csv(base_dir + 'sample_submission.csv')
def to_categorical(frame, train=True): 

    frame['Gender'] = pd.Categorical(frame['Gender'])

    frame['Driving_License'] = pd.Categorical(frame['Driving_License'])

    frame['Previously_Insured'] = pd.Categorical(frame['Previously_Insured'])

    frame['Vehicle_Damage'] = pd.Categorical(frame['Vehicle_Damage'])

    frame['Vehicle_Age'] = pd.Categorical(frame['Vehicle_Age'])

    frame['Region_Code'] = pd.Categorical(frame['Region_Code'])

    if train:

        frame['Response'] = pd.Categorical(frame['Response'])

    

    return frame

# df['Gender'] = pd.Categorical(df['Gender'])

# df['Driving_License'] = pd.Categorical(df['Driving_License'])

# df['Previously_Insured'] = pd.Categorical(df['Previously_Insured'])

# df['Vehicle_Damage'] = pd.Categorical(df['Vehicle_Damage'])

# df['Response'] = pd.Categorical(df['Response'])

# df['Vehicle_Age'] = pd.Categorical(df['Vehicle_Age'])

# df['Region_Code'] = pd.Categorical(df['Region_Code'])



df_train = to_categorical(df)



df_train = pd.concat([df[['Age', 'Annual_Premium', 'Policy_Sales_Channel', 'Vintage', 'Response']],

           pd.get_dummies(df[['Gender', 'Driving_License', 

                   'Previously_Insured', 'Vehicle_Damage', 

                   'Vehicle_Age']])], axis=1)



df_train
iso = IsolationForest(warm_start=True, n_jobs=-1)

iso.fit_predict(df_train.drop('Response', axis=1))

iso_pred = iso.predict(df_train.drop('Response', axis=1))

#plt.scatter(df_train['Annual_Premium'], df_train['Policy_Sales_Channel'], c=iso_pred)
iso_pred = np.where(iso_pred==1, 0, iso_pred)

iso_pred = np.where(iso_pred==-1, 1, iso_pred)

iso_pred = np.array(iso_pred, dtype=bool)

print(np.unique(iso_pred, return_counts=True))
df_train_reduced = df_train.loc[iso_pred]

X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(df_train_reduced.drop('Response', axis=1), 

                                                            df_train_reduced['Response'])
# lgb_params = {

#     'num_leaves' : [10, 50, 100],

#     'n_estimators' : [100, 300, 350],

#     'reg_lambda' : [.01, .5], 

#     'reg_alpha' : [0.01, .5], 

#     'subsample' : [.25, .5],

#     'learning_rate' : np.linspace(0.1, .7, 5), 

#     'importance_type' : ['split'], 

#     'colsample_bytree' : [.2, .5, .9, 1]

# }
lgb_params = {'colsample_bytree': 0.5,

              'importance_type': 'split',

              'learning_rate': 0.1,

              'n_estimators': 100,

              'num_leaves': 10,

              'reg_alpha': 0.01,

              'reg_lambda': 0.5,

              'subsample': 0.25, 

              'objective' : 'binary'

         }

lgb = LGBMClassifier(**lgb_params)

lgb.fit(X_train_r, y_train_r)
fpr, tpr, _ = roc_curve(y_test_r, lgb.predict_proba(X_test_r)[:, 1])

plt.plot(fpr, tpr)

print(roc_auc_score(y_test_r, lgb.predict_proba(X_test_r)[:, 1]))
df_test = to_categorical(df_test, train=False)

df_test = pd.concat([df_test[['Age', 'Annual_Premium', 'Policy_Sales_Channel', 'Vintage']],

                      pd.get_dummies(df_test[['Gender', 'Driving_License', 'Previously_Insured', 'Vehicle_Damage', 'Vehicle_Age']])], axis=1)
df_test.head()
y_pred = lgb.predict_proba(df_test)[:, 1]

submission['Response'] = y_pred

submission.head()
submission.to_csv('submission.csv')