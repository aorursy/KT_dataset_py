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
from sklearn.model_selection import cross_val_score, cross_validate, train_test_split

from sklearn.metrics import make_scorer, r2_score, mean_absolute_error

from sklearn.linear_model import LinearRegression, Ridge

from sklearn.feature_selection import SelectFromModel

from sklearn.ensemble import RandomForestRegressor

from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt

import seaborn as sns
df_test = pd.read_csv('/kaggle/input/codenation-enem2/test.csv')

df_train = pd.read_csv('/kaggle/input/codenation-enem2/train.csv')
use_list = ['NU_NOTA_COMP1', 'NU_NOTA_COMP2', 'NU_NOTA_COMP3', 'NU_NOTA_COMP4', 'NU_NOTA_COMP5', 'NU_NOTA_CN', 'NU_NOTA_CH', 

            'NU_NOTA_LC', 'NU_NOTA_REDACAO', 'NU_NOTA_MT']



df_train = df_train[use_list] #selection all columns

df_test = df_test[use_list[:-1]] #selection all other columns except 'NU_NOTA_MT' column for testing
df_train.head()
df_test.head()
print(df_train.isna().sum() / df_train.shape[0] * 100)
print(df_test.isna().sum() / df_test.shape[0] * 100)
df_train_filled = df_train.fillna(0, axis=0)

df_test_filled = df_test.fillna(0, axis=0)
correlacao_notas = df_train_filled.corr()



plt.figure(figsize=(10, 6))

sns.heatmap(correlacao_notas, annot=True, cmap="BrBG", vmin=-1, vmax=1)

plt.xticks(rotation=45)

plt.show()
X = df_train_filled.drop(columns=['NU_NOTA_COMP5', 'NU_NOTA_MT'])

y = df_train_filled['NU_NOTA_MT']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
lr = LinearRegression(normalize=True)

lr.fit(X_train, y_train)
lr.score(X_test, y_test)
mae = make_scorer(mean_absolute_error)

r2 = make_scorer(r2_score)



cvs = cross_validate(estimator=LinearRegression(normalize=True), X=X, y=y, cv=10, verbose=10, 

                      scoring={'mae': mae, 'r2':r2})
print("The mean of the result is %.3f" % (cvs['test_r2'].mean()))

print("The standard desviation error is %.3f" % (cvs['test_r2'].std()))
print("The mean of the result is %.3f" % (cvs['test_mae'].mean()))

print("The standard desviation error is %.3f" % (cvs['test_mae'].std()))