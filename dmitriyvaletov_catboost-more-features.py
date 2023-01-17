# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
RANDOM_STATE = 42
na_filling = "imputer"
scaling = False
overSample = True
import matplotlib.pyplot as plt
import seaborn as sns

import random
import numpy as np
import pandas as pd


from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler

from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, log_loss, mean_squared_error, mean_absolute_error, f1_score
train_df = pd.read_csv("/kaggle/input/killer-shrimp-invasion/train.csv")
test_df = pd.read_csv("/kaggle/input/killer-shrimp-invasion/test.csv")

columns = list(train_df.columns)

features = [
    'Salinity_today', 
    'Temperature_today', 
    'Substrate', 
    'Depth', 
    'Exposure', 

    'Temperature_today_exp', 
    'Depth_log', 
    'Exposure_log', 
    'Salin_div_depth',
    'Temp_div_depth',
    
]

categoricals = [
    'Substrate',
]

numerical_features = [f for f in features if f not in categoricals]

target = 'Presence'


# init
for f in features:
    if f not in train_df:
        train_df[f] = 0.0
        test_df[f]  = 0.0
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer


if na_filling == "imputer":
    imputer = IterativeImputer(max_iter = 10, random_state = RANDOM_STATE)
    imputer.fit(train_df[features])
    train_df[features] = pd.DataFrame(imputer.transform(train_df[features]), columns = features)
    test_df[features] = pd.DataFrame(imputer.transform(test_df[features]), columns = features)

else:
    train_df[numerical_features] = train_df[numerical_features].fillna(train_df[numerical_features].median())
    test_df[numerical_features] = test_df[numerical_features].fillna(test_df[numerical_features].median())

    train_df[categoricals] = train_df[categoricals].fillna(train_df[categoricals].mode().iloc[0])
    test_df[categoricals] = test_df[categoricals].fillna(test_df[categoricals].mode().iloc[0])
train_df['Exposure_log'] = np.log(train_df['Exposure'])
test_df['Exposure_log']  = np.log(test_df['Exposure'])

train_df['Depth_log'] = np.log(np.abs(train_df['Depth']))
test_df['Depth_log'] = np.log(np.abs(test_df['Depth']))

train_df['Temperature_today_exp'] = np.exp(train_df['Temperature_today'])
test_df['Temperature_today_exp'] = np.exp(test_df['Temperature_today'])

train_df['Temp_div_depth'] = train_df['Temperature_today'] / train_df['Depth']
test_df['Temp_div_depth'] = test_df['Temperature_today']   / test_df['Depth']

train_df['Salin_div_depth'] = train_df['Salinity_today'] / train_df['Depth']
test_df['Salin_div_depth'] = test_df['Salinity_today']   / test_df['Depth']
features = set(features)
for df in [train_df, test_df]:
    for i in range(2, 7):
        new_feature = f'Temperature_today^{i}'
        df[new_feature] = df['Temperature_today'] ** i
        features.add(new_feature)
        
features = list(features)
train_df.head(5)
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
df = pd.concat([train_df[numerical_features], test_df[numerical_features]], ignore_index=True)
scaler.fit(df[numerical_features])
models = []

for turn_i in range(10):
    ros = RandomOverSampler()
    _, _ = ros.fit_resample(train_df[features], train_df.Presence)
    train_idx = ros.sample_indices_
    _, _ = ros.fit_resample(train_df[features], train_df.Presence)
    val_idx = ros.sample_indices_
    
    x_train, x_val = train_df.loc[train_idx, features], train_df.loc[val_idx, features]
    y_train, y_val = train_df.loc[train_idx, target], train_df.loc[val_idx, target]
    
    model = CatBoostClassifier(iterations=50, verbose=False)
    model.fit(x_train, y_train)
    val_pred = model.predict(x_val)
    score = roc_auc_score(y_val, val_pred)
    models.append([score, turn_i])
    model.save_model(f"model_{turn_i}.cbm")
    print('Turn: {} score: {}'.format(turn_i, score))
models.sort(key=lambda x: x[0])
best_model_i = models[-1][1]
best_model = model.load_model(f"model_{best_model_i}.cbm")
import shap
shap.initjs()

explainer = shap.TreeExplainer(best_model)
shap_values = explainer.shap_values(train_df[features])
s_i = 100
shap.force_plot(explainer.expected_value, shap_values[s_i,:], train_df[features].iloc[s_i,:])
s_i = 0
shap.force_plot(explainer.expected_value, shap_values[s_i,:], train_df[features].iloc[s_i,:])
shap.dependence_plot("Temperature_today", shap_values, train_df[features])
shap.summary_plot(shap_values, train_df[features])
shap.summary_plot(shap_values, train_df[features], plot_type="bar")
y_probas = best_model.predict_proba(test_df[features])
result = pd.read_csv("/kaggle/input/killer-shrimp-invasion/temperature_submission.csv")
result.Presence = y_probas[:, 1].round(3)
result.to_csv("submission.csv", index=False)