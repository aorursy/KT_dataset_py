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
df_train = pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/train.csv")
df_test = pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/test.csv")
df_train.info()
df_test.info()
df_train.describe()
df_test.describe()
from sklearn.preprocessing import LabelEncoder

def fill_all_columns(df, verbose=False):
    for column in df.columns:
        if df[column].isnull().values.any():
            if df[column].dtype == 'float64':
                df.fillna(value={column: df[column].median()}, inplace=True)
            elif df[column].dtype == 'int64' or df[column].dtype == 'object':
                df[column].fillna(method='bfill', inplace=True)
                df[column].fillna(method='ffill', inplace=True)
                
        if df[column].dtype == 'object':
            encoder_ = LabelEncoder().fit(df[column])
            df[column] = encoder_.transform(df[column])
                
    if verbose:
        df.info()
def plot_regression_results(ax, y_true, y_pred, title, scores):
    """Scatter plot of the predicted vs true targets."""
    ax.plot([y_true.min(), y_true.max()],
            [y_true.min(), y_true.max()],
            '--r', linewidth=2)
    ax.scatter(y_true, y_pred, alpha=0.2)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
    ax.spines['left'].set_position(('outward', 10))
    ax.spines['bottom'].set_position(('outward', 10))
    ax.set_xlim([y_true.min(), y_true.max()])
    ax.set_ylim([y_true.min(), y_true.max()])
    ax.set_xlabel('Measured')
    ax.set_ylabel('Predicted')
    extra = plt.Rectangle((0, 0), 0, 0, fc="w", fill=False,
                          edgecolor='none', linewidth=0)
    ax.legend([extra], [scores], loc='upper left')
    #title = title + '\n Evaluation in {:.2f} seconds'.format(elapsed_time)
    ax.set_title(title)
fill_all_columns(df_train, True)
fill_all_columns(df_test, True)
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import GridSearchCV
from scipy import stats

classifier = GridSearchCV(estimator=RandomForestRegressor(n_jobs=8, random_state=12), 
                          param_grid=dict(criterion=['mse', 'mae'], bootstrap=[True,False], n_estimators=range(7,11), max_depth=range(8,11)), 
                          scoring='neg_mean_squared_error', 
                          n_jobs=-1,
                          cv=10)
#Remove Outliers
df_train = df_train[(np.abs(stats.zscore(df_train)) < 3).all(axis=1)]
X = df_train.loc[:, df_train.columns != 'SalePrice']
y = df_train['SalePrice']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

classifier.fit(X_train, y_train)
print()
#print(classifier.cv_results_)
print(classifier.best_estimator_)
print(-classifier.cv_results_['mean_test_score'])
print(classifier.cv_results_['std_test_score'])
print(classifier.best_params_)
predicoes = classifier.predict(X_test)

%matplotlib inline
import matplotlib.pyplot as plt
fig, axs = plt.subplots(1, 1, figsize=(9, 7))

plot_regression_results(axs, y_test, predicoes,'RandomForestRegressor',-classifier.score(X_test, y_test))
print("Id,SalePrice")
y_pred_final = classifier.predict(df_test)
for i, j in zip(df_test['Id'], y_pred_final):
    print(str(i) + "," + str(j))
submission=pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv')
submission['SalePrice']=y_pred_final

submission= submission.drop(submission.loc[:, submission.columns != 'SalePrice'], axis=1)
submission.to_csv('submission.csv',index=False)