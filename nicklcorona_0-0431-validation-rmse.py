# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import catboost

from catboost import CatBoostRegressor, Pool

from sklearn.model_selection import train_test_split

import seaborn as sns

import shap

from scipy.constants import golden

import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error

from sklearn.ensemble import RandomForestRegressor





# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

shap.initjs()

np.random.seed(42)
df = pd.read_csv('/kaggle/input/graduate-admissions/Admission_Predict_Ver1.1.csv')
print(df.head()); print(df.columns)
df.head()
# left skewed distribution. raise your standard university!
sns.distplot(df['Chance of Admit '], hist=False)
y = df['Chance of Admit ']

df = df.drop('Chance of Admit ', axis=1)
df.describe(); df.isnull().sum()
df.dtypes
m = CatBoostRegressor(iterations=1000, silent=True)
valid_idx = 400
X_train, y_train, X_valid, y_valid = df[:valid_idx], y[:valid_idx], df[valid_idx:], y[valid_idx:]
train_pool, validate_pool = Pool(X_train, y_train), Pool(X_valid, y_valid)
m.fit(train_pool, eval_set=validate_pool, plot=True, logging_level='Silent')
# create series feature importance

imp = pd.Series(m.feature_importances_, index=df.columns).sort_values()
# apparently university rating is pretty unimportant. You'd expect the best universities to have better admission rates.
plt.figure(figsize=(10, 10/golden), dpi=144)

sns.barplot(imp, imp.index)
plot = pd.concat([X_train, y], axis=1)

sns.lineplot('University Rating', 'Chance of Admit ', data=plot)
sns.heatmap(X_train.corr())
m = CatBoostRegressor(iterations=1000, silent=True, ignored_features=['University Rating'])

m.fit(train_pool, eval_set=validate_pool, plot=True)
def feature_importance(model):

    """Return feature importance of catboost model and plot it."""

    imp = pd.Series(model.feature_importances_, index=df.columns).sort_values()

    plt.figure(figsize=(10, 10 / golden), dpi=144)

    sns.barplot(imp, imp.index)

    return imp
imp = feature_importance(m)
explainer = shap.TreeExplainer(m)

shap_values = explainer.shap_values(X_train)
shap.force_plot(explainer.expected_value, shap_values[2,:], X_train.iloc[2,:])
plt.axvline(8, 0, 0.68, color='red')

sns.distplot(plot['CGPA'], hist=False)
sns.regplot('CGPA', 'Chance of Admit ', data=plot,lowess=True)
sns.regplot('Serial No.', 'Chance of Admit ', data=plot, lowess=True)
m = CatBoostRegressor(iterations=1000, silent=True, ignored_features=['University Rating', 'Serial No.'])

m.fit(train_pool, eval_set=validate_pool, plot=True)
imp = feature_importance(m)
m = CatBoostRegressor(iterations=1000, silent=True)

m.fit(train_pool, eval_set=validate_pool, plot=True)
shap.dependence_plot("Serial No.", shap_values, X_train, interaction_index=None)
shap.dependence_plot("CGPA", shap_values, X_train, interaction_index=None)
shap.summary_plot(shap_values, X_train)
m = CatBoostRegressor(iterations=1000, silent=True, ignored_features=['University Rating'])

m.fit(train_pool, eval_set=validate_pool, plot=True)
explainer.shap_interaction_values(X_train)
y_pred = m.predict(X_valid)

round(mean_squared_error(y_valid, y_pred, squared=False), 4)
shap.__version__