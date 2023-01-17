# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from datetime import datetime

import seaborn as sns

from sklearn.metrics import confusion_matrix



sns.set_style('darkgrid')



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df_codes = pd.read_csv('/kaggle/input/offense_codes.csv', encoding='ISO-8859-1')

df_codes.head()
df = pd.read_csv('/kaggle/input/crime.csv', encoding='ISO-8859-1')

df.head()
print(df.shape)

df.isnull().sum()
df.drop(['DISTRICT', 'SHOOTING', 'UCR_PART', 'STREET', 'Lat', 'Long'], axis=1, inplace=True)
sorted(df['REPORTING_AREA'].unique())[:10]
## replace empty reporting areas with '-1'

df['REPORTING_AREA'] = df['REPORTING_AREA'].str.replace(' ', '-1')

sorted(df['REPORTING_AREA'].unique())

df['REPORTING_AREA'] = df['REPORTING_AREA'].astype(int)
# code day of week to ints

df['OCCURRED_ON_DATE'] = pd.to_datetime(df['OCCURRED_ON_DATE'])

df['DAY_OF_WEEK'] = df['OCCURRED_ON_DATE'].dt.dayofweek
df['OFFENSE_CODE_GROUP'].value_counts().plot(kind='bar', figsize=(20,5), title='Offense Code Group Counts')
df_new = df.copy(deep=True)

df_new['MV'] = np.where(df_new['OFFENSE_CODE_GROUP'] == 'Motor Vehicle Accident Response', 1, 0)
df_new.head()
df_mv = df_new[['MV', 'REPORTING_AREA', 'YEAR', 'MONTH', 'DAY_OF_WEEK', 'HOUR']]

df_mv.head()
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import mean_squared_error, r2_score



# shuffle the data if you want

df_mv = df_mv.sample(frac=1).reset_index(drop=True)



X = df_mv[df_mv.columns[1:]]

y = df_mv['MV']



X_train, X_test, y_train, y_test = train_test_split(X,y)
reg = LogisticRegression(random_state=0, solver='lbfgs').fit(X_train, y_train)

y_pred = reg.predict(X_test)



print('Mean squared error: {:.2f}%'.format(mean_squared_error(y_test, y_pred)))

print('Variance score: {:.2f}%'.format(r2_score(y_test, y_pred)))

print('Coefficients: {}'.format(reg.coef_))

print('Intercept: {}'.format(reg.intercept_))
## To get more output about coefficients of logistic regression use statsmodels to perform same logistic regression



import statsmodels.discrete.discrete_model as sm

from statsmodels.tools.tools import add_constant



# statsmodels doesn't include a constant by default

# sklearn.linear_model DOES include a constant by default

X_ols = add_constant(X)



sm.Logit(y,X_ols).fit().summary()
df_knn = df[df['OFFENSE_CODE_GROUP'].isin(list(df['OFFENSE_CODE_GROUP'].value_counts()[:3].index))].copy(deep=True)

print(df_knn.shape)

df_knn.head()
# encode the OFFENSE_CODE_GROUP

from sklearn.preprocessing import LabelEncoder

from sklearn.neighbors import KNeighborsClassifier



lb_make = LabelEncoder()

df_knn['office_code_lbl'] = lb_make.fit_transform(df_knn['OFFENSE_CODE_GROUP'])

df_knn.head()
df_knn = df_knn[['office_code_lbl', 'REPORTING_AREA', 'YEAR', 'MONTH', 'DAY_OF_WEEK', 'HOUR']]



X = df_knn[['REPORTING_AREA', 'YEAR', 'MONTH', 'DAY_OF_WEEK', 'HOUR']]

y = df_knn['office_code_lbl']



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
df_knn.dtypes




neighbors_list = np.arange(1,5)

scores = []

for n_neighbors in neighbors_list:

    knn = KNeighborsClassifier(n_neighbors=n_neighbors)

    knn.fit(X_train, y_train)

    y_pred = knn.predict(X_test)

    cm = confusion_matrix(y_test, y_pred)

    scores.append(((cm[0][0] + cm[1][1]) / sum(sum(cm)))* 100)

    

max_score = max([x for x in scores])

max_idx = np.argmax(scores) + 1



print('Best % correct: {:.2f}%, for n_neighbors={}'.format(max_score, max_idx))
# 0 = Monday

df.groupby('DAY_OF_WEEK')['YEAR'].count().plot(kind='bar')
df.groupby('MONTH')['DAY_OF_WEEK'].count().plot(kind='bar')
df.groupby('YEAR')['DAY_OF_WEEK'].count().plot(kind='bar')