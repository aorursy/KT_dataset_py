import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

import scipy as sc

%matplotlib inline



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
df = pd.read_csv('..//input/hotel-booking-demand/hotel_bookings.csv')

print(df.shape)

df.head()
df.describe()
# set up aesthetic design

plt.style.use('seaborn')

sns.set_style('whitegrid')



# create NA plot for train data

plt.figure(figsize = (15,3)) # positioning for 1st plot

df.isnull().mean().sort_values(ascending = False).plot.bar(color = 'blue')

plt.axhline(y=0.1, color='r', linestyle='-')

plt.title('Missing values average per columns in data', fontsize = 20)

plt.show()
df['country'] = df['country'].fillna('missing')

df['company'] = df['company'].fillna('missing')

df['agent'] = df['agent'].fillna(df['agent'].mean())

df['children'] = df['children'].fillna(0)
plt.figure(figsize=(15,15))

mask = np.zeros_like(df.corr())

mask[np.triu_indices_from(mask)] = 1

sns.heatmap(df.corr(), mask = mask, annot = True)

plt.show()
categories_columns = list(df.columns[df.dtypes == 'object'])

#categories_columns.append('agent')

#categories_columns.append('is_canceled')





numeric_columns = list(set(df.columns) - set(categories_columns))

print('categories columns are ',categories_columns)

print(' ')

print('numeric columns are ', numeric_columns)
from sklearn.preprocessing import LabelEncoder



le = LabelEncoder()

df_cat = df.loc[:,categories_columns]

df_cat = df_cat.drop(['company'], axis = 1)

df_cat.head()


for col in categories_columns:

    print(col)

    try:

        df_cat[col] = le.fit_transform(df_cat[col])

    except:

        print('fail at ',col)

        pass
df = df.loc[:,numeric_columns]

df = pd.concat([df, df_cat], axis = 1)



# remove due to high correlation to cancel

df = df.drop(['reservation_status','reservation_status_date'], axis = 1)

df.head()
import lightgbm as lgb

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score



X = df.drop(['is_canceled'], axis=1)

Y = df['is_canceled']



X_tr, X_test, y_tr, y_test = train_test_split(X,Y, test_size=0.2)



X_train, X_valid, y_train, y_valid = train_test_split(X_tr, y_tr, test_size=0.2)



lgb_train = lgb.Dataset(X_train, y_train)

lgb_eval = lgb.Dataset(X_valid, y_valid)
# specify parameters

params = {

    'boosting_type': 'gbdt',

    'objective': 'binary',

    'metric': ['auc','binary_logloss'],

    'feature_fraction': 1,

    'bagging_fraction':1,

    'verbose': -1

}



# training

print('start training')

gbm = lgb.train(params, lgb_train,

               num_boost_round = 20,

               valid_sets = lgb_eval,

               early_stopping_rounds = 5)
lgb.plot_importance(gbm)
y_pred = gbm.predict(X_test)

y_pred[y_pred <= 0.5] = 0

y_pred[y_pred>0.5] = 1

y_pred[:10]

accuracy_score(y_test, y_pred)
from sklearn.svm import SVC



svc = SVC(kernel = 'linear')

model = svc.fit(X_train, y_train)
model.score(X_train,y_train)
model.predict(X_test)



accuracy_score(y_test, y_pred)
from sklearn.metrics import confusion_matrix



print(confusion_matrix(y_test, y_pred))
from sklearn.ensemble import RandomForestClassifier



rf = RandomForestClassifier()



model_rf = rf.fit(X_train, y_train)

y_pred = model_rf.predict(X_test)



print('accuracy = ',accuracy_score(y_test, y_pred))
features = X_train.columns

importances = model_rf.feature_importances_



indices = np.argsort(importances)



plt.figure()

plt.barh(range(len(indices)), importances[indices], color = 'b',alpha=0.5)

plt.yticks(range(len(indices)), [features[i] for i in indices])

plt.xlabel('Relative Importance')

plt.title('Feature Importances')

plt.show()