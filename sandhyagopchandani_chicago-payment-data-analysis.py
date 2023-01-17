!pip install keras_metrics
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import datetime as dt

import matplotlib.pylab as plt

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from sklearn.model_selection import train_test_split

import os

print(os.listdir("../input"))

from tabulate import tabulate

import seaborn as sns

# Any results you write to the current directory are saved as output.
payments = pd.read_csv("../input/payments.csv")
sum_missing = payments.isnull().sum()

percent_missing = payments.isnull().sum() * 100 / len(payments)



missing_value_df = pd.DataFrame({'column_name': payments.columns,

                                 'total_missing':sum_missing,

                                 'percent_missing': percent_missing})
missing_value_df
dep_payment = payments[payments['DEPARTMENT NAME'].notnull()]

dept_df = dep_payment.groupby('DEPARTMENT NAME').size().reset_index(name='counts').sort_values(by=['counts'], ascending=False)
#fig = plt.figure(figsize=(10,4))

ax = dept_df.plot.bar(x='DEPARTMENT NAME', y='counts', rot=0, figsize=(10,7))

ax.set_xticklabels(dept_df['DEPARTMENT NAME'], rotation=90)

plt.show()
payments[payments['CONTRACT NUMBER'] == 'DV'].shape[0] / len(payments)
payments['check_date'] = pd.to_datetime(payments['CHECK DATE'])

checkDate_df = payments.groupby(payments['check_date'].map(lambda x: x.year)).size().reset_index(name='counts').sort_values(by=['check_date'], ascending=True)
ax = checkDate_df.plot.bar(x='check_date', y='counts',figsize=(10,5))

ax.set_xticklabels(checkDate_df['check_date'], rotation=90)

plt.show()
payments[payments['VENDOR NAME'].isnull()]
per_vendor_pay_count_df = payments.groupby('VENDOR NAME').size().reset_index(name='counts').sort_values(by=['counts'], ascending=False)
len(per_vendor_pay_count_df[per_vendor_pay_count_df['counts'] < 2]) / len(per_vendor_pay_count_df)
len(per_vendor_pay_count_df[per_vendor_pay_count_df['counts'] >= 100]) / len(per_vendor_pay_count_df)

len(per_vendor_pay_count_df[per_vendor_pay_count_df['counts'] >= 500]) / len(per_vendor_pay_count_df)
per_vendor_pay_count_500 = per_vendor_pay_count_df[per_vendor_pay_count_df['counts'] >= 500]
fig = plt.figure(figsize=(10,4))

ax = per_vendor_pay_count_500.plot.bar(x='VENDOR NAME', y='counts', rot=0, figsize=(10,7))

ax.set_xticklabels(per_vendor_pay_count_500['VENDOR NAME'], rotation=90)

plt.show()
top_100_vendors = per_vendor_pay_count_df[per_vendor_pay_count_df['counts'] >= 100]

payment_subset = pd.merge(payments, top_100_vendors, on='VENDOR NAME')
print('vendor percent', len(top_100_vendors) / len(per_vendor_pay_count_df))

print('amount percent', payment_subset['AMOUNT'].sum() / payments['AMOUNT'].sum())
per_vendor_pay_count_1 = per_vendor_pay_count_df[per_vendor_pay_count_df['counts'] < 2]

payment_1 = pd.merge(payments, per_vendor_pay_count_1, on='VENDOR NAME')
print('vendor percent', len(per_vendor_pay_count_1) / len(per_vendor_pay_count_df))

print('amount percent', payment_1['AMOUNT'].sum() / payments['AMOUNT'].sum() * 100)
#per_vendor_pay_count_between = per_vendor_pay_count_df[2 <= per_vendor_pay_count_df['counts'] <= 99 ]



per_vendor_pay_count_between = per_vendor_pay_count_df[(per_vendor_pay_count_df['counts'] >= 2) & (per_vendor_pay_count_df['counts'] <= 99)]

payment_between = pd.merge(payments, per_vendor_pay_count_between, on='VENDOR NAME')
print('vendor percent', len(per_vendor_pay_count_between) / len(per_vendor_pay_count_df))

print('amount percent', payment_between['AMOUNT'].sum() / payments['AMOUNT'].sum())

print(tabulate([['=1', '72%', '1.27%'], ['>=100','0.15%', '38%'], ['2-99','28%','62%']], 

               headers=['Range', 'Vendor %', 'Amount %']))
len(payment_subset)/len(payments)
len(payment_subset[payment_subset['DEPARTMENT NAME'].isnull()])/len(payment_subset)
len(payment_subset[payment_subset['CONTRACT NUMBER'] == 'DV'])/len(payment_subset)
# for predicting Department Name 

# train = payments[payments['DEPARTMENT NAME'].notnull()]

# test = payments[payments['DEPARTMENT NAME'].isnull()]

# test = test[test['VENDOR NAME'].notnull()]

# #train_df = train.sample(50000, replace=True).reset_index()

# #test_df = test.sample(50000, replace=True).reset_index()

# train_df = train_df[['AMOUNT','VENDOR NAME', 'CASHED', 'DEPARTMENT NAME']]

# test_df =  test_df[['AMOUNT','VENDOR NAME', 'CASHED', 'DEPARTMENT NAME']]
per_vendor_pay_count_df = payments.groupby('VENDOR NAME').size().reset_index(name='counts').sort_values(by=['counts'], ascending=False)

per_vendor_pay_count_df['high_vendor'] = [1 if x >= 1500 else 0 for x in per_vendor_pay_count_df['counts']]

per_vendor_pay_count_df['medium_vendor'] = [1 if 700 <= x < 1500 else 0 for x in per_vendor_pay_count_df['counts']] 

per_vendor_pay_count_df['low_vendor'] = [1 if x < 700 else 0 for x in per_vendor_pay_count_df['counts']] 

per_vendor_pay_count_df.columns = ['VENDOR NAME', 'vendor_count', 'high_vendor','medium_vendor','low_vendor']
payments = pd.merge(payments, per_vendor_pay_count_df, on='VENDOR NAME')
payments.head(10)
payment_data = payments[['AMOUNT','DEPARTMENT NAME', 'vendor_count', 'high_vendor', 'medium_vendor', 

                         'low_vendor','CASHED']]
payment_data.CASHED = [1 if i == True else 0 for i in payment_data.CASHED]
payment_data["AMOUNT"] = (payment_data["AMOUNT"] - payment_data["AMOUNT"].min()) / (payment_data["AMOUNT"].max() - payment_data["AMOUNT"].min())

payment_data["vendor_count"] = (payment_data["vendor_count"] - payment_data["vendor_count"].min()) / (payment_data["vendor_count"].max()- payment_data["vendor_count"].min())
payment_data_encoded = pd.concat([payment_data,pd.get_dummies(payment_data['DEPARTMENT NAME'],prefix='DEPT', drop_first=True, dummy_na=True)],axis=1)

payment_data_encoded.drop(['DEPARTMENT NAME'],axis=1, inplace=True)
payment_data_encoded.head(5)
payment_data_encoded.groupby('CASHED').size()
735174/len(payment_data_encoded)
Y = payment_data_encoded.pop('CASHED').to_frame().reset_index(drop=True)

X = payment_data_encoded
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=42)
#fig = plt.figure(figsize=(10,4))

# dept_df = y_train.groupby('DEPARTMENT NAME').size().reset_index(name='counts').sort_values(by=['counts'], ascending=False)

# ax = dept_df.plot.bar(x='DEPARTMENT NAME', y='counts', rot=0, figsize=(10,7))

# ax.set_xticklabels(dept_df['DEPARTMENT NAME'], rotation=90)

# plt.show()
#from sklearn.preprocessing import OneHotEncoder

#enc = OneHotEncoder(handle_unknown = 'ignore')

#enc.fit(y_train)

#y_train_encoded = enc.transform(y_train).toarray()

#y_val_encoded = enc.transform(y_val).toarray()
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import confusion_matrix

clf=RandomForestClassifier(random_state=0, n_estimators=100, max_depth=3, class_weight="balanced")

clf.fit(X_train,y_train)
clf_predict = clf.predict(X_test) 

accuracy = clf.score(X_test, y_test)

print('accuracy',accuracy)

cm = confusion_matrix(y_test, clf_predict)

print(cm)
feature_imp = pd.Series(clf.feature_importances_,index=list(X_train.columns)).sort_values(ascending=False)[:10]

feature_imp
# Creating a bar plot

sns.barplot(x=feature_imp, y=feature_imp.index)

# Add labels to your graph

plt.xlabel('Feature Importance Score')

plt.ylabel('Features')

plt.title("Visualizing Important Features")

plt.legend()

plt.show()
from keras.models import Sequential

from keras.layers import Dense

from keras.wrappers.scikit_learn import KerasClassifier

from keras.utils import np_utils

import keras_metrics

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import KFold

from sklearn.preprocessing import LabelEncoder

from sklearn.pipeline import Pipeline

np.random.seed(7)
# For Department Name Prediction

#encode class values as integers

#encoder = LabelEncoder()

#encoder.fit(y_train)

#encoded_train_Y = encoder.transform(y_train)

#encoded_val_Y = encoder.transform(y_test)

# convert integers to dummy variables (i.e. one hot encoded)

#dummy_train_y = np_utils.to_categorical(encoded_train_Y)

#dummy_val_y = np_utils.to_categorical(encoded_val_Y)
def baseline_model():

    # create model

    model = Sequential()

    model.add(Dense(20, input_dim=56, activation='relu'))

    model.add(Dense(1, activation='sigmoid'))

    # Compile model

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=[keras_metrics.precision(), keras_metrics.recall()])

    return model
#estimator = KerasClassifier(build_fn=baseline_model, epochs=10, batch_size=32, verbose=1)

#results = cross_val_score(estimator, X_train, y_train )
#kfold = KFold(n_splits=2, shuffle=True, random_state=seed)
model = baseline_model()

model.fit(X_train, y_train, epochs=5, validation_data=(X_test, y_test), verbose=1)