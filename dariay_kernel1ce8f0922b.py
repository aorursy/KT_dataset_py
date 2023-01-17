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
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import pandas_profiling

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier

from sklearn import tree

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC, LinearSVC

from xgboost import XGBClassifier

from sklearn.metrics import roc_curve, roc_auc_score, classification_report, accuracy_score, confusion_matrix, mean_squared_error, r2_score

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score

from sklearn.preprocessing import MinMaxScaler
training_df = pd.read_csv('/kaggle/input/devrepublik02/training_set.csv')

valid_df = pd.read_csv('/kaggle/input/devrepublik02/validation_set.csv')
training_df['deposit'] = training_df['deposit'].apply(lambda x: 1 if x == 'yes' else 0)

training_df.head()
def column_transform(column, df, df2):

    temp_df = df.groupby([column])['deposit'].mean()

    temp_df.sort_values(ascending=True, inplace=True)

    temp_df = pd.DataFrame(temp_df)

    temp_df.reset_index(inplace=True)

    temp_df.reset_index(inplace=True)

    temp_df['index'] = temp_df['index']+1

    plt.bar(x=temp_df[column], height=temp_df['deposit'])

    plt.ylabel('Conversion rate')

    plt.xticks(rotation=45)

    plt.show()

    df = df.merge(temp_df.drop('deposit', axis=1), how='left', on=column)

    df.rename(columns={'index': column+'_encoded'}, inplace=True)

    df2 = df2.merge(temp_df.drop('deposit', axis=1), how='left', on=column)

    df2.rename(columns={'index': column+'_encoded'}, inplace=True)

    return df, df2
columns = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'poutcome']

for i in columns:

    training_df, valid_df = column_transform(column=i, df=training_df, df2=valid_df)
#pandas_profiling.ProfileReport(training_df)
training_df['ds_type'] = 'train'

valid_df['ds_type'] = 'valid'
model_df = pd.concat([training_df, valid_df], ignore_index=False)

model_df.tail(1001)
#pandas_profiling.ProfileReport(model_df)
model_df['age_scaled'] = (model_df['age']-min(model_df['age']))/(max(model_df['age'])-min(model_df['age']))

model_df['balance_scaled'] = (model_df['balance']-min(model_df['balance']))/(max(model_df['balance'])-min(model_df['balance']))

model_df['day_scaled'] = (model_df['day']-min(model_df['day']))/(max(model_df['day'])-min(model_df['day']))

model_df['pdays_scaled'] = (model_df['pdays']-min(model_df['pdays']))/(max(model_df['pdays'])-min(model_df['pdays']))

model_df['previous_scaled'] = (model_df['previous']-min(model_df['previous']))/(max(model_df['previous'])-min(model_df['previous']))

model_df['campaign_scaled'] = (model_df['campaign']-min(model_df['campaign']))/(max(model_df['campaign'])-min(model_df['campaign']))

model_df.drop(['campaign', 'job', 'marital', 'education', 'default', 'duration', 'housing', 'loan', 'contact', 

               'month', 'poutcome', 'age', 'balance', 'day', 'pdays', 'previous'],

              axis=1, inplace=True)

model_df.head()
model_df.describe()
training_df_transf = model_df[model_df['ds_type'] == 'train'] 

valid_df_transf = model_df[model_df['ds_type'] == 'valid']
valid_df_transf.head()
training_df_transf.drop('ds_type', axis=1, inplace=True)

valid_df_transf.drop(['deposit', 'ds_type'], axis=1, inplace=True)
X = training_df_transf.drop('deposit', axis=1)

y = training_df_transf['deposit']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train.shape
dt_model = RandomForestClassifier(criterion='gini', n_estimators=60)

dt_model.fit(X_train, y_train)

print(dt_model.score(X_train, y_train), dt_model.score(X_test, y_test))

feat_imp_dt = dict(zip(X_train.columns, dt_model.feature_importances_))

feat_imp_dt = dict(sorted(feat_imp_dt.items(), key=lambda x: x[1]))

plt.figure(figsize=(15,6))

plt.bar(x=feat_imp_dt.keys(), height=feat_imp_dt.values())

plt.title('The most important features with Decision Tree Classifier')

plt.xticks(rotation=45)

plt.show()
import warnings

warnings.filterwarnings('ignore')

models = [tree.DecisionTreeClassifier(random_state=8),

          RandomForestClassifier(criterion='gini', n_estimators=60, random_state=8),

          AdaBoostClassifier(random_state=8),

          LogisticRegression(random_state=8), 

          SVC(random_state=8),

          LinearSVC(random_state=8),

          XGBClassifier(random_state=8),

         GradientBoostingClassifier(random_state=8)]

for model in models:

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)

    roc_auc_metrics = roc_auc_score(y_test, y_pred)

    cross_valid_score = cross_val_score(estimator=model, X=X_train, y=y_train, cv=10)

    print("Accuracy score: %.2f%%" % (accuracy*100), 

          "ROC AUC score: %.2f%%" % (roc_auc_metrics*100),

          "K-fold Cross Validation mean score: %.2f%%" % (cross_valid_score.mean()*100))
final_model = GradientBoostingClassifier(random_state=42)

final_model.fit(X_train, y_train)

final_model.score(X_train, y_train), final_model.score(X_test, y_test)
parameters = {'loss': ['deviance', 'exponential'],

             'learning_rate': [0.1, 0.2, 0.3]}

rf_gscv = GridSearchCV(final_model, param_grid=parameters, verbose=True, scoring='roc_auc')

rf_gscv.fit(X, y)

rf_gscv.best_params_
parameters = {'max_features': [4, 5, 8, 15],

             'n_estimators': [50, 100, 200],

             'min_samples_leaf': [0.5, 1]}

final_model = GradientBoostingClassifier(loss='deviance', learning_rate=0.2, random_state=42)

rf_gscv = GridSearchCV(final_model, param_grid=parameters, verbose=True, scoring='roc_auc')

rf_gscv.fit(X, y)

rf_gscv.best_params_
final_model = XGBClassifier(

    booster='gbtree', 

    learning_rate=0.2,

    max_depth=5,

    max_features=5,

    random_state=42)

final_model.fit(X_train, y_train)

y_pred = final_model.predict(X_test)

print(roc_auc_score(y_test, y_pred))

final_model.score(X_train, y_train), final_model.score(X_test, y_test)
subm = pd.DataFrame()

subm['deposit'] = final_model.predict_proba(valid_df_transf)[:,1]

subm.reset_index(drop=False, inplace=True)

subm.to_csv('my_submission.csv', index=False)