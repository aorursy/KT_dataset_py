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
import pandas as pd



df = pd.read_csv('/kaggle/input/health-care-data-set-on-heart-attack-possibility/heart.csv')



df.head()
df.isnull().sum()
df.info()
import seaborn as sns

import matplotlib.pyplot as plt

plt.figure(figsize=(10,10), dpi= 100)

sns.heatmap(df.corr(), cmap='RdYlGn', center=0)



# Decorations

plt.title('Correlation Graph', fontsize=22)

plt.xticks(fontsize=12)

plt.yticks(fontsize=12)

plt.show()
# Mean Encoding some categorical columns which don't have good correlation with target

cumsum = df.groupby('sex')['target'].cumsum() - df['target']

cumcnt = df.groupby('sex').cumcount()

df['sex'] = cumsum/cumcnt

cumsum = df.groupby('fbs')['target'].cumsum() - df['target']

cumcnt = df.groupby('fbs').cumcount()

df['fbs'] = cumsum/cumcnt

cumsum = df.groupby('thal')['target'].cumsum() - df['target']

cumcnt = df.groupby('thal').cumcount()

df['thal'] = cumsum/cumcnt

cumsum = df.groupby('ca')['target'].cumsum() - df['target']

cumcnt = df.groupby('ca').cumcount()

df['ca'] = cumsum/cumcnt

cumsum = df.groupby('oldpeak')['target'].cumsum() - df['target']

cumcnt = df.groupby('oldpeak').cumcount()

df['oldpeak'] = cumsum/cumcnt
df=df.dropna()

from sklearn.model_selection import train_test_split

y = df['target']

X=df.drop('target', 1)



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .2, random_state=10)
import xgboost as xgb

from xgboost import XGBClassifier

xgb = XGBClassifier(learning_rate=0.001, n_estimators=1000, max_depth= 10,

                        min_child_weight=0, gamma=0, subsample=0.52, colsample_bytree=0.6,

                        objective='binary:logistic', nthread=4, scale_pos_weight=1, 

                    seed=27, reg_alpha=5, reg_lambda=2, booster='gbtree',

            n_jobs=-1, max_delta_step=0, colsample_bylevel=0.6, colsample_bynode=0.6)

model = xgb.fit(X_train, y_train)

y_pred = model.predict(X_test)
from sklearn import metrics

print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
col_sorted_by_importance=xgb.feature_importances_.argsort()

feat_imp=pd.DataFrame({

    'cols':X.columns[col_sorted_by_importance],

    'imps':xgb.feature_importances_[col_sorted_by_importance]

})



!pip install plotly-express

import plotly_express as px

px.bar(feat_imp, x='cols', y='imps')