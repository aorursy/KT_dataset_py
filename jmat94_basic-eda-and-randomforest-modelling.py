# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
df = pd.read_csv('../input/weatherAUS.csv')
df.shape
df.head(10)
df.info()
def check_null(df):
    null_values = df.isnull().sum()
    return null_values / len(df) * 100
# print("The null % is :", check_null(df.info()))
check_null(df)
def fill_nan(df, col):
    df[col] = df[col].fillna(df[col].median())
    return df
fill_nan(df, 'MaxTemp')
# df['MaxTemp'].isnull().sum()
# sns.distplot(df[df['MaxTemp'] > 15]['MaxTemp'], bins=10, kde = False);
# sns.lineplot('MaxTemp', data=df)
# df['MaxTemp'].value_counts().sort_index().plot.line()
df = df.drop(columns=['Location','Date','Evaporation','Sunshine', 'Cloud9am','Cloud3pm',
                           'WindGustDir','WindGustSpeed','WindDir9am','WindDir3pm','WindSpeed9am',
                           'WindSpeed3pm'], axis=1)
# df.head()
# int_values = {'No': 0,
#              'Yes': 1}
df['RainTomorrow'] = df['RainTomorrow'].replace({'No': 0,'Yes': 1})
# df['RainTomorrow'].fillna(0, inplace=True)
# df['RainTomorrow'].head()
# df.RainTomorrow.isnull().sum()
y = df['RainTomorrow']
y = y.fillna(0)
df.drop(['RainTomorrow'], axis=1, inplace=True)
# df.head()
X = df
X = X.replace({'No': 0, 'Yes': 1})
# X.info()
min_max = preprocessing.MinMaxScaler()
X_scaled = pd.DataFrame(min_max.fit_transform(X), columns = X.columns)
X_scaled.head()
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size = 0.2, random_state=49)
X_train.shape, X_test.shape, y_train.shape, y_test.shape
model = RandomForestClassifier(n_estimators = 20, n_jobs = -1)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print("Accuracy Score: {}".format(metrics.accuracy_score(y_pred, y_test)))
metrics.confusion_matrix(y_pred, y_test)


