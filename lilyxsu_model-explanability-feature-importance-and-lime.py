import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import sklearn

from sklearn.model_selection import train_test_split

import category_encoders as ce

from sklearn.pipeline import make_pipeline

from sklearn.preprocessing import StandardScaler

from sklearn.metrics import r2_score, mean_squared_error

from sklearn.experimental import enable_hist_gradient_boosting 

from sklearn.ensemble import HistGradientBoostingRegressor

from sklearn.model_selection import GroupShuffleSplit

from sklearn.ensemble import GradientBoostingRegressor 

from mlxtend.regressor import StackingCVRegressor

from sklearn.ensemble import RandomForestRegressor

from sklearn.linear_model import Lasso

from sklearn.linear_model import Ridge

import time

from sklearn.pipeline import Pipeline

from sklearn.feature_selection import SelectKBest

from sklearn.ensemble import GradientBoostingRegressor

from sklearn.preprocessing import OrdinalEncoder

from sklearn.compose import ColumnTransformer

from sklearn.model_selection import GridSearchCV

from sklearn.ensemble import StackingRegressor



df = pd.read_csv('../input/osic-pulmonary-fibrosis-progression/train.csv')

df.head(3)
enc = OrdinalEncoder()

enc.fit(df[['Sex', 'SmokingStatus']])

enc.categories_
np_encoded = OrdinalEncoder().fit_transform(df[['Sex', 'SmokingStatus']])

df_encoded = pd.DataFrame({'Sex': np_encoded[:, 0], 'SmokingStatus': np_encoded[:, 1]})

df_encoded.nunique().sort_values(ascending = False)
df_encoded.sample(2)
scaler = StandardScaler()

print(scaler.fit(df[['Weeks','Percent', 'Age']]))

print(scaler.mean_)

print(scaler.transform(df[['Weeks','Percent', 'Age']]))
np_scaler = StandardScaler().fit_transform(df[['Weeks','Percent', 'Age']])

df_scaler = pd.DataFrame({'Weeks': np_scaler[:, 0], 'Percent': np_scaler[:, 1], 'Age': np_scaler[:, 2]})

df_scaler.head(2)
transformed_df = df_scaler[['Weeks','Percent', 'Age']]

transformed_df = pd.concat([df[['Patient','FVC']], transformed_df, df_encoded], axis = 1)

transformed_df.head(2)
df = transformed_df



train_inds, val_inds = next(GroupShuffleSplit(test_size=.4, n_splits=2, random_state = 42).split(df, groups=df['Patient']))

train = df.iloc[train_inds]

val = df.iloc[val_inds]



col = 'Patient'



cardinality = len(pd.Index(df[col]).value_counts())

print("Number of " + df[col].name + "s in original DataFrame df: " + str(cardinality) + '\n')   

cardinality = len(pd.Index(train[col]).value_counts())

print("Number of " + train[col].name + "s in train: " + str(cardinality) + '\n')

cardinality = len(pd.Index(val[col]).value_counts())

print("Number of " + val[col].name + "s in val: " + str(cardinality))



target = 'FVC'

features = train.drop(columns=[target, 'Patient']).columns.tolist()



X_train = train[features]

y_train = train[target]

X_val = val[features]

y_val = val[target]
params = {'n_estimators': 500,

          'max_depth': 4,

          'min_samples_split': 5,

          'learning_rate': 0.01,

          'loss': 'ls'}



reg = GradientBoostingRegressor(**params)

reg.fit(X_train, y_train)



mse = mean_squared_error(y_val, reg.predict(X_val))

print("The mean squared error (MSE) on test set: {:.4f}".format(mse))
test_score = np.zeros((params['n_estimators'],), dtype=np.float64)

for i, y_pred in enumerate(reg.staged_predict(X_val)):

    test_score[i] = reg.loss_(y_val, y_pred)



fig = plt.figure(figsize=(6, 6))

plt.subplot(1, 1, 1)

plt.title('Deviance')

plt.plot(np.arange(params['n_estimators']) + 1, reg.train_score_, 'b-',

         label='Training Set Deviance')

plt.plot(np.arange(params['n_estimators']) + 1, test_score, 'r-',

         label='Test Set Deviance')

plt.legend(loc='upper right')

plt.xlabel('Boosting Iterations')

plt.ylabel('Deviance')

fig.tight_layout()

plt.show()
from sklearn.inspection import permutation_importance



feature_importance = reg.feature_importances_

sorted_idx = np.argsort(feature_importance)

pos = np.arange(sorted_idx.shape[0]) + .5

fig = plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)

plt.barh(pos, feature_importance[sorted_idx], align='center')

plt.yticks(pos, np.array(list(X_train.columns))[sorted_idx])

plt.title('Feature Importance (MDI)')



result = permutation_importance(reg, X_val, y_val, n_repeats=10,

                                random_state=42, n_jobs=2)

sorted_idx = result.importances_mean.argsort()

plt.subplot(1, 2, 2)

plt.boxplot(result.importances[sorted_idx].T,

            vert=False, labels=np.array(list(X_train.columns))[sorted_idx])

plt.title("Permutation Importance (test set)")

fig.tight_layout()

plt.show()
from xgboost import XGBRegressor



params = {'n_estimators': 500,

          'max_depth': 4,

          'min_samples_split': 5,

          'learning_rate': 0.01,

          'loss': 'ls'}



model = XGBRegressor(**params)



model.fit(

    X_train.values,

    y_train

)

y_val_pred = model.predict(X_val.values)

print(model.feature_importances_)
import lime

from lime import lime_tabular



explainer = lime_tabular.LimeTabularExplainer(training_data=X_train.values , feature_names=features, class_names="FVC", mode="regression")

explained = explainer.explain_instance(X_val.values[0], model.predict, num_features=4)

explained.as_pyplot_figure()
explained.show_in_notebook()
explained = explainer.explain_instance(X_val.values[623], model.predict, num_features=4)

explained.show_in_notebook()
model.fit(

    X_train,

    y_train

)

y_pred = model.predict(test)

print(model.feature_importances_)
sub = pd.read_csv('../input/osic-pulmonary-fibrosis-progression/sample_submission.csv')

sub['Weeks'] = sub['Patient_Week'].apply(lambda x: int(x.split('_')[-1]))

sub['Patient'] = sub['Patient_Week'].apply(lambda x: x.split('_')[0])

test = pd.read_csv('../input/osic-pulmonary-fibrosis-progression/test.csv')

test = pd.merge(sub, test, on = ['Patient'], how='outer')

test.drop(columns = ['Patient_Week','FVC_x','Confidence','Weeks_y'], inplace=True)

test.rename(columns = {'Weeks_x':'Weeks', 'FVC_y':'FVC'}, inplace = True)

test.head(2)
test.shape
np_encoded = OrdinalEncoder().fit_transform(test[['Sex', 'SmokingStatus']])

df_encoded = pd.DataFrame({'Sex': np_encoded[:, 0], 'SmokingStatus': np_encoded[:, 1]})



np_scaler = StandardScaler().fit_transform(test[['Weeks','Percent', 'Age']])

df_scaler = pd.DataFrame({'Weeks': np_scaler[:, 0], 'Percent': np_scaler[:, 1], 'Age': np_scaler[:, 2]})

df_scaler.head(2)



transformed_df = df_scaler[['Weeks','Percent', 'Age']]

test = pd.concat([test[['Patient','FVC']], transformed_df, df_encoded], axis=1)

test.drop(columns = ['Patient', 'FVC'], inplace = True)

test.shape
test.sample(4)

# print(test.shape)
sub.shape
test.head(2)
s.head(2)
s = pd.read_csv('../input/osic-pulmonary-fibrosis-progression/sample_submission.csv')

y_pred = model.predict(test)

s['FVC'] = y_pred



s.head(2)
s['Confidence'] = s['FVC'].std()

s['FVC'] = s['FVC'].astype(int)

s['Confidence'] = s['Confidence'].astype(int)

s.to_csv("submission.csv", index=False)

s.head(2)
s.head(2)
standard_deviation = s['Confidence'][0]



σ = max(standard_deviation, 70)

σ
actual_FVC_value = 1500 # pretended actual number

# pretended actual number 1500 - the first observation 1127 = 373

# the absolute value of 373 is smaller than 1000

Δ = min(abs(actual_FVC_value - s['FVC'][0]), 1000)

Δ
import math



metric = (math.sqrt(2) * Δ / σ) - np.log(math.sqrt(2) * σ)

metric