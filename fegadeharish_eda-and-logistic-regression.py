from mpl_toolkits.mplot3d import Axes3D

from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt # plotting

import numpy as np # linear algebra

import os # accessing directory structure

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression, RidgeClassifier

from sklearn.metrics import accuracy_score,classification_report,confusion_matrix,mean_squared_error

import warnings

warnings.filterwarnings('ignore')

%matplotlib inline
print(os.listdir('../input'))
df1 = pd.read_csv('../input/UCI_Credit_Card.csv', delimiter=',')

df1.dataframeName = 'UCI_Credit_Card.csv'

nRow, nCol = df1.shape

print(f'There are {nRow} rows and {nCol} columns')
defaulters = df1.copy()

print(defaulters.shape)

defaulters.head()
defaulters.describe().T
defaulters.rename(columns={'default.payment.next.month':'def_pay'}, inplace=True)

defaulters.rename(columns={'PAY_0':'PAY_1'}, inplace=True)
# checking the datatype of each feature

defaulters.info()
defaulters.isna().sum()    # check for missing values for surity
def_cnt = (defaulters.def_pay.value_counts(normalize=True)*100)

def_cnt.plot.bar(figsize=(6,6))

plt.xticks(fontsize=12, rotation=0)

plt.yticks(fontsize=12)

plt.title("Probability Of Defaulting Payment Next Month", fontsize=15)

for x,y in zip([0,1],def_cnt):

    plt.text(x,y,y,fontsize=12)

plt.show()
plt.subplots(figsize=(20,5))

plt.subplot(121)

sns.distplot(defaulters.LIMIT_BAL)



plt.subplot(122)

sns.distplot(defaulters.AGE)



plt.show()
bins = [20,30,40,50,60,70,80]

names = ['21-30','31-40','41-50','51-60','61-70','71-80']

defaulters['AGE_BIN'] = pd.cut(x=defaulters.AGE, bins=bins, labels=names, right=True)



age_cnt = defaulters.AGE_BIN.value_counts()

age_0 = (defaulters.AGE_BIN[defaulters['def_pay'] == 0].value_counts())

age_1 = (defaulters.AGE_BIN[defaulters['def_pay'] == 1].value_counts())



plt.subplots(figsize=(8,5))

# sns.barplot(data=defaulters, x='AGE_BIN', y='LIMIT_BAL', hue='def_pay', ci=0)

plt.bar(age_0.index, age_0.values, label='0')

plt.bar(age_1.index, age_1.values, label='1')

for x,y in zip(names,age_0):

    plt.text(x,y,y,fontsize=12)

for x,y in zip(names,age_1):

    plt.text(x,y,y,fontsize=12)

plt.xticks(fontsize=12)

plt.yticks(fontsize=12)

plt.title("Number of clients in each age group", fontsize=15)

plt.legend(loc='upper right', fontsize=15)

plt.show()

plt.subplots(figsize=(20,10))



ind = sorted(defaulters.PAY_1.unique())

pay_0 = (defaulters.PAY_1[defaulters['def_pay'] == 0].value_counts(normalize=True))

pay_1 = (defaulters.PAY_1[defaulters['def_pay'] == 1].value_counts(normalize=True))

total = pay_0.values+pay_1.values

pay_0_prop = np.true_divide(pay_0, total)*100

pay_1_prop = np.true_divide(pay_1, total)*100

plt.subplot(231)

plt.bar(ind, pay_1_prop, bottom=pay_0_prop, label='1')

plt.bar(ind, pay_0_prop, label='0')

plt.title("Repayment Status M-0", fontsize=15)



ind = sorted(defaulters.PAY_2.unique())

pay_0 = (defaulters.PAY_2[defaulters['def_pay'] == 0].value_counts(normalize=True))

pay_1 = (defaulters.PAY_2[defaulters['def_pay'] == 1].value_counts(normalize=True))

for i in pay_0.index:

    if i not in pay_1.index:

        pay_1[i]=0

total = pay_0.values+pay_1.values

pay_0_prop = np.true_divide(pay_0, total)*100

pay_1_prop = np.true_divide(pay_1, total)*100

plt.subplot(232)

plt.bar(ind, pay_1_prop, bottom=pay_0_prop, label='1')

plt.bar(ind, pay_0_prop, label='0')

plt.title("Repayment Status M-1", fontsize=15)



ind = sorted(defaulters.PAY_3.unique())

pay_0 = (defaulters.PAY_3[defaulters['def_pay'] == 0].value_counts(normalize=True))

pay_1 = (defaulters.PAY_3[defaulters['def_pay'] == 1].value_counts(normalize=True))

for i in pay_0.index:

    if i not in pay_1.index:

        pay_1[i]=0

total = pay_0.values+pay_1.values

pay_0_prop = np.true_divide(pay_0, total)*100

pay_1_prop = np.true_divide(pay_1, total)*100

plt.subplot(233)

plt.bar(ind, pay_1_prop, bottom=pay_0_prop, label='1')

plt.bar(ind, pay_0_prop, label='0')

plt.title("Repayment Status M-2", fontsize=15)



ind = sorted(defaulters.PAY_4.unique())

pay_0 = (defaulters.PAY_4[defaulters['def_pay'] == 0].value_counts(normalize=True))

pay_1 = (defaulters.PAY_4[defaulters['def_pay'] == 1].value_counts(normalize=True))

for i in pay_0.index:

    if i not in pay_1.index:

        pay_1[i]=0

total = pay_0.values+pay_1.values

pay_0_prop = np.true_divide(pay_0, total)*100

pay_1_prop = np.true_divide(pay_1, total)*100

plt.subplot(234)

plt.bar(ind, pay_1_prop, bottom=pay_0_prop, label='1')

plt.bar(ind, pay_0_prop, label='0')

plt.title("Repayment Status M-3", fontsize=15)



ind = sorted(defaulters.PAY_5.unique())

pay_0 = (defaulters.PAY_5[defaulters['def_pay'] == 0].value_counts(normalize=True))

pay_1 = (defaulters.PAY_5[defaulters['def_pay'] == 1].value_counts(normalize=True))

for i in pay_0.index:

    if i not in pay_1.index:

        pay_1[i]=0

for i in pay_1.index:

    if i not in pay_0.index:

        pay_0[i]=0

total = pay_0.values+pay_1.values

pay_0_prop = np.true_divide(pay_0, total)*100

pay_1_prop = np.true_divide(pay_1, total)*100

plt.subplot(235)

plt.bar(ind, pay_1_prop, bottom=pay_0_prop, label='1')

plt.bar(ind, pay_0_prop, label='0')

plt.title("Repayment Status M-4", fontsize=15)



ind = sorted(defaulters.PAY_6.unique())

pay_0 = (defaulters.PAY_6[defaulters['def_pay'] == 0].value_counts(normalize=True))

pay_1 = (defaulters.PAY_6[defaulters['def_pay'] == 1].value_counts(normalize=True))

for i in pay_0.index:

    if i not in pay_1.index:

        pay_1[i]=0

for i in pay_1.index:

    if i not in pay_0.index:

        pay_0[i]=0

total = pay_0.values+pay_1.values

pay_0_prop = np.true_divide(pay_0, total)*100

pay_1_prop = np.true_divide(pay_1, total)*100

plt.subplot(236)

plt.bar(ind, pay_1_prop, bottom=pay_0_prop, label='1')

plt.bar(ind, pay_0_prop, label='0')

plt.title("Repayment Status M-5", fontsize=15)



plt.xticks(ind, fontsize=12)

plt.yticks(fontsize=12)

plt.legend(loc="upper right", fontsize=15)

plt.suptitle("Repayment Status for last 6 months with proportion of defaulting payment next month", fontsize=20)



plt.show()
g = sns.FacetGrid(defaulters, row='def_pay', col='MARRIAGE')

g = g.map(plt.hist, 'AGE')

plt.show()
g = sns.FacetGrid(defaulters, row='def_pay', col='SEX')

g = g.map(plt.hist, 'AGE')
plt.subplots(figsize=(20,10))



plt.subplot(231)

plt.scatter(x=defaulters.PAY_AMT1, y=defaulters.BILL_AMT1, c='r', s=1)



plt.subplot(232)

plt.scatter(x=defaulters.PAY_AMT2, y=defaulters.BILL_AMT2, c='b', s=1)



plt.subplot(233)

plt.scatter(x=defaulters.PAY_AMT3, y=defaulters.BILL_AMT3, c='g', s=1)



plt.subplot(234)

plt.scatter(x=defaulters.PAY_AMT4, y=defaulters.BILL_AMT4, c='c', s=1)

plt.ylabel("Bill Amount in past 6 months", fontsize=25)



plt.subplot(235)

plt.scatter(x=defaulters.PAY_AMT5, y=defaulters.BILL_AMT5, c='y', s=1)

plt.xlabel("Payment in past 6 months", fontsize=25)



plt.subplot(236)

plt.scatter(x=defaulters.PAY_AMT6, y=defaulters.BILL_AMT6, c='m', s=1)



plt.show()
y1 = defaulters.AGE[defaulters["def_pay"] == 0]

y2 = defaulters.AGE[defaulters["def_pay"] == 1]

x1 = defaulters.LIMIT_BAL[defaulters["def_pay"] == 0]

x2 = defaulters.LIMIT_BAL[defaulters["def_pay"] == 1]



fig,ax = plt.subplots(figsize=(20,10))

plt.scatter(x1,y1, color="r", marker="*", label='0')

plt.scatter(x2,y2, color="b", marker=".", label='1')

plt.xlabel("LIMITING BALANCE", fontsize=20)

plt.ylabel("AGE", fontsize=20)

plt.xticks(fontsize=15)

plt.yticks(fontsize=15)

plt.legend(loc='upper right', fontsize=20)

plt.show()

plt.subplots(figsize=(30,20))

sns.heatmap(defaulters.corr(), annot=True)

plt.show()
#saleprice correlation matrix

k = 10 #number of variables for heatmap

corrmat = defaulters.corr()

cols = corrmat.nlargest(k, 'def_pay')['def_pay'].index

cm = np.corrcoef(defaulters[cols].values.T)

sns.set(font_scale=1.25)

plt.subplots(figsize=(10,10))

hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)

plt.show()
defaulters.info()
df_X = defaulters.drop(['def_pay','AGE_BIN'], axis=1)

df_y = defaulters.def_pay



X_train, X_test, y_train, y_test = train_test_split(df_X, df_y, test_size=0.2, random_state=10)



model1 = LogisticRegression()

model1.fit(X_train, y_train)



y_pred = model1.predict(X_test)



print(classification_report(y_pred, y_test))

print(confusion_matrix(y_pred, y_test))

print('\nAccuracy Score for model1: ', accuracy_score(y_pred,y_test))
# change the datatype of categorical features from integer to category

defaulters.SEX = defaulters.SEX.astype("category")

defaulters.EDUCATION = defaulters.EDUCATION.astype("category")

defaulters.MARRIAGE = defaulters.MARRIAGE.astype("category")

defaulters.PAY_1 = defaulters.PAY_1.astype("category")

defaulters.PAY_2 = defaulters.PAY_2.astype("category")

defaulters.PAY_3 = defaulters.PAY_3.astype("category")

defaulters.PAY_4 = defaulters.PAY_4.astype("category")

defaulters.PAY_5 = defaulters.PAY_5.astype("category")

defaulters.PAY_6 = defaulters.PAY_6.astype("category")

defaulters.def_type = defaulters.def_pay.astype("category")
df_X = defaulters.drop(['def_pay','AGE_BIN'], axis=1)

df_y = defaulters.def_pay



X_train, X_test, y_train, y_test = train_test_split(df_X, df_y, test_size=0.2, random_state=10)



model2 = LogisticRegression()

model2.fit(X_train, y_train)



y_pred = model2.predict(X_test)



print(classification_report(y_pred, y_test))

print(confusion_matrix(y_pred, y_test))

print('\nAccuracy Score for model2: ', accuracy_score(y_pred,y_test))
df_X = defaulters.drop(['def_pay','AGE_BIN','PAY_2','PAY_3','PAY_4','PAY_5','PAY_6'], axis=1)

df_y = defaulters.def_pay



X_train, X_test, y_train, y_test = train_test_split(df_X, df_y, test_size=0.2, random_state=10)



model3 = LogisticRegression()

model3.fit(X_train, y_train)



y_pred = model3.predict(X_test)



print(classification_report(y_pred, y_test))

print(confusion_matrix(y_pred, y_test))

print('\nAccuracy Score for model3: ', accuracy_score(y_pred,y_test))
df_X = defaulters[['SEX','MARRIAGE','AGE','BILL_AMT1','EDUCATION','PAY_1']]

df_y = defaulters.def_pay



X_train, X_test, y_train, y_test = train_test_split(df_X, df_y, test_size=0.1, random_state=20)



model4 = LogisticRegression()

model4.fit(X_train, y_train)



y_pred = model4.predict(X_test)

y_train_pred = model4.predict(X_train)



print(classification_report(y_pred, y_test))

print(confusion_matrix(y_pred, y_test))

print('\nTest Accuracy Score for model4: ', accuracy_score(y_pred,y_test))

print('\nTrain Accuracy Score for model4: ', accuracy_score(y_train_pred,y_train))
df_X = defaulters[['SEX','MARRIAGE','AGE','BILL_AMT1','EDUCATION','PAY_1']]

df_y = defaulters.def_pay



X_train, X_test, y_train, y_test = train_test_split(df_X, df_y, test_size=0.3, random_state=20)



model5 = RidgeClassifier()

model5.fit(X_train, y_train)



y_pred = model5.predict(X_test)

y_train_pred = model5.predict(X_train)



print(classification_report(y_pred, y_test))

print(confusion_matrix(y_pred, y_test))

print('\nTest Accuracy Score for model5: ', accuracy_score(y_pred,y_test))

print('\nTrain Accuracy Score for model5: ', accuracy_score(y_train_pred,y_train))