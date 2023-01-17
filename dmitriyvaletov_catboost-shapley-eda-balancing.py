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
import random
import numpy as np
import pandas as pd
train_df = pd.read_csv("/kaggle/input/killer-shrimp-invasion/train.csv")
test_df = pd.read_csv("/kaggle/input/killer-shrimp-invasion/test.csv")

columns = list(train_df.columns)
features = columns[1:-1]
categoricals = ['Substrate']
target = columns[-1]

print(train_df.Presence.value_counts())
# filling none with medians and modes (when categoricals)

numerical_features = [f for f in features if f not in categoricals]
train_df[numerical_features] = train_df[numerical_features].fillna(train_df[numerical_features].median())
test_df[numerical_features] = test_df[numerical_features].fillna(test_df[numerical_features].median())

train_df[categoricals] = train_df[categoricals].fillna(train_df[categoricals].mode().iloc[0])
test_df[categoricals] = test_df[categoricals].fillna(test_df[categoricals].mode().iloc[0])
# data standartization

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
df = pd.concat([train_df[numerical_features], test_df[numerical_features]], ignore_index=True)
scaler.fit(df[numerical_features])
x_test = test_df.copy()[features]
x_test[numerical_features] = scaler.transform(x_test[numerical_features])
from catboost import CatBoostClassifier
from imblearn.over_sampling import RandomOverSampler # this will duplicate the minor class rows necessary times to get balanced
from imblearn.under_sampling import RandomUnderSampler # this will sample from major class rows the number of minor class rows
from sklearn.metrics import accuracy_score, roc_auc_score, log_loss, mean_squared_error, mean_absolute_error, f1_score
rus = RandomUnderSampler() 
_, _ = rus.fit_resample(train_df[features], train_df.Presence)
train_idx = rus.sample_indices_


x_train = train_df.loc[train_idx, features]
y_train = train_df.loc[train_idx, target]
x_train[numerical_features] = scaler.transform(x_train[numerical_features])

model = CatBoostClassifier(iterations=50, verbose=False)
model.fit(x_train, y_train)
y_pred = model.predict(x_train)
score = roc_auc_score(y_train, y_pred)
print('score: {}'.format(score))
import shap
shap.initjs()

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(x_train)
s_i = 56
shap.force_plot(explainer.expected_value, shap_values[s_i,:], x_train.iloc[s_i,:])
s_i = 1
shap.force_plot(explainer.expected_value, shap_values[s_i,:], x_train.iloc[s_i,:])
shap.dependence_plot("Temperature_today", shap_values, x_train)
shap.summary_plot(shap_values, x_train)
shap.summary_plot(shap_values, x_train, plot_type="bar")
y_test = model.predict(x_test)

result = pd.read_csv("/kaggle/input/killer-shrimp-invasion/temperature_submission.csv")
result.Presence = y_test
result.to_csv("submission.csv", index=False)