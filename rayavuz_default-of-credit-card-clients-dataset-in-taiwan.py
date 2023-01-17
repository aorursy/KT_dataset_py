# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 

import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from sklearn.model_selection import KFold

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import f1_score,roc_auc_score,accuracy_score,roc_curve

import itertools

from sklearn.model_selection import GridSearchCV



import warnings

warnings.filterwarnings("ignore")
data = pd.read_csv('../input/default-of-credit-card-clients-dataset/UCI_Credit_Card.csv')
data.columns
data.shape
data.head(10)
data.info()
data.isnull().sum().sort_values(ascending=False)
data.EDUCATION.value_counts()
data.MARRIAGE.value_counts()
data = data.rename(columns={'default.payment.next.month': 'def_pay', 

                            'PAY_0': 'PAY_1'})

fill = (data.EDUCATION == 5) | (data.EDUCATION == 6) | (data.EDUCATION == 0)

data.loc[fill, 'EDUCATION'] = 4

data.loc[data.MARRIAGE == 0, 'MARRIAGE'] = 3
data.columns
data.EDUCATION.value_counts()
data.MARRIAGE.value_counts()
data.EDUCATION.describe()
data.SEX.describe()
data.MARRIAGE.describe()
data.AGE.describe()
data.PAY_1.describe()
data.PAY_2.describe()
data.PAY_3.describe()
data.PAY_4.describe()
data.PAY_5.describe()
data.PAY_6.describe()
data.BILL_AMT1.describe()
data.BILL_AMT2.describe()
data.BILL_AMT3.describe()
data.BILL_AMT4.describe()
data.BILL_AMT5.describe()
data.BILL_AMT6.describe()
data.PAY_AMT1.describe()
data.PAY_AMT2.describe()
data.PAY_AMT3.describe()
data.PAY_AMT4.describe()
data.PAY_AMT5.describe()
data.PAY_AMT6.describe()
data.LIMIT_BAL.describe()
def formgroup(Col1, Col2):

    res = data.groupby([Col1, Col2]).size().unstack()

    return res
data['SE_MA'] = data.SEX * data.MARRIAGE

formgroup('SE_MA', 'def_pay')

data['SE_MA_2'] = 0

data.loc[((data.SEX == 1) & (data.MARRIAGE == 1)) , 'SE_MA_2'] = 1 #married man

data.loc[((data.SEX == 1) & (data.MARRIAGE == 2)) , 'SE_MA_2'] = 2 #single man

data.loc[((data.SEX == 1) & (data.MARRIAGE == 3)) , 'SE_MA_2'] = 3 #divorced man

data.loc[((data.SEX == 2) & (data.MARRIAGE == 1)) , 'SE_MA_2'] = 4 #married woman

data.loc[((data.SEX == 2) & (data.MARRIAGE == 2)) , 'SE_MA_2'] = 5 #single woman

data.loc[((data.SEX == 2) & (data.MARRIAGE == 3)) , 'SE_MA_2'] = 6 #divorced woman

formgroup('SE_MA_2', 'def_pay')

del data['SE_MA']

data = data.rename(columns={'SE_MA_2': 'SE_MA'})
data['AgeBin'] = 0 #creates a column of 0

data.loc[((data['AGE'] > 20) & (data['AGE'] < 30)) , 'AgeBin'] = 1

data.loc[((data['AGE'] >= 30) & (data['AGE'] < 40)) , 'AgeBin'] = 2

data.loc[((data['AGE'] >= 40) & (data['AGE'] < 50)) , 'AgeBin'] = 3

data.loc[((data['AGE'] >= 50) & (data['AGE'] < 60)) , 'AgeBin'] = 4

data.loc[((data['AGE'] >= 60) & (data['AGE'] < 70)) , 'AgeBin'] = 5

data.loc[((data['AGE'] >= 70) & (data['AGE'] < 81)) , 'AgeBin'] = 6

plt.figure()

plt.title('20den baslayarak 10 ar 10 ar yaş aralığı ve sayıları')

data.AgeBin.hist()

plt.show()
agedefpay=formgroup('AgeBin', 'def_pay')

print(agedefpay)

agesex=formgroup('AgeBin', 'SEX')

print(agesex)
data['SE_AG'] = 0

data.loc[((data.SEX == 1) & (data.AgeBin == 1)) , 'SE_AG'] = 1 #erkek 20'li

data.loc[((data.SEX == 1) & (data.AgeBin == 2)) , 'SE_AG'] = 2 #erkek 30'lu

data.loc[((data.SEX == 1) & (data.AgeBin == 3)) , 'SE_AG'] = 3 #erkek 40'lı

data.loc[((data.SEX == 1) & (data.AgeBin == 4)) , 'SE_AG'] = 4 #erkek 50'li

data.loc[((data.SEX == 1) & (data.AgeBin == 5)) , 'SE_AG'] = 5 #erkek 60+

data.loc[((data.SEX == 2) & (data.AgeBin == 1)) , 'SE_AG'] = 6 #kadın 20'li

data.loc[((data.SEX == 2) & (data.AgeBin == 2)) , 'SE_AG'] = 7 #kadın 30'lu

data.loc[((data.SEX == 2) & (data.AgeBin == 3)) , 'SE_AG'] = 8 #kadın 40'lı

data.loc[((data.SEX == 2) & (data.AgeBin == 4)) , 'SE_AG'] = 9 #kadın 50'li

data.loc[((data.SEX == 2) & (data.AgeBin == 5)) , 'SE_AG'] = 10 #kadın 60+

formgroup('SE_AG', 'def_pay')
data['active_6'] = 1

data['active_5'] = 1

data['active_4'] = 1

data['active_3'] = 1

data['active_2'] = 1

data['active_1'] = 1

data.loc[((data.PAY_6 == 0) & (data.BILL_AMT6 == 0) & (data.PAY_AMT6 == 0)) , 'active_6'] = 0

data.loc[((data.PAY_5 == 0) & (data.BILL_AMT5 == 0) & (data.PAY_AMT5 == 0)) , 'active_5'] = 0

data.loc[((data.PAY_4 == 0) & (data.BILL_AMT4 == 0) & (data.PAY_AMT4 == 0)) , 'active_4'] = 0

data.loc[((data.PAY_3 == 0) & (data.BILL_AMT3 == 0) & (data.PAY_AMT3 == 0)) , 'active_3'] = 0

data.loc[((data.PAY_2 == 0) & (data.BILL_AMT2 == 0) & (data.PAY_AMT2 == 0)) , 'active_2'] = 0

data.loc[((data.PAY_1 == 0) & (data.BILL_AMT1 == 0) & (data.PAY_AMT1 == 0)) , 'active_1'] = 0
pd.Series([data[data.active_6 == 1].def_pay.count(),

          data[data.active_5 == 1].def_pay.count(),

          data[data.active_4 == 1].def_pay.count(),

          data[data.active_3 == 1].def_pay.count(),

          data[data.active_2 == 1].def_pay.count(),

          data[data.active_1 == 1].def_pay.count()], [6,5,4,3,2,1])
data['average_5'] = ((data['BILL_AMT5'] - (data['BILL_AMT6'] - data['PAY_AMT5']))) / data['LIMIT_BAL']

data['average_4'] = (((data['BILL_AMT5'] - (data['BILL_AMT6'] - data['PAY_AMT5'])) +

                 (data['BILL_AMT4'] - (data['BILL_AMT5'] - data['PAY_AMT4']))) / 2) / data['LIMIT_BAL']

data['average_3'] = (((data['BILL_AMT5'] - (data['BILL_AMT6'] - data['PAY_AMT5'])) +

                 (data['BILL_AMT4'] - (data['BILL_AMT5'] - data['PAY_AMT4'])) +

                 (data['BILL_AMT3'] - (data['BILL_AMT4'] - data['PAY_AMT3']))) / 3) / data['LIMIT_BAL']

data['average_2'] = (((data['BILL_AMT5'] - (data['BILL_AMT6'] - data['PAY_AMT5'])) +

                 (data['BILL_AMT4'] - (data['BILL_AMT5'] - data['PAY_AMT4'])) +

                 (data['BILL_AMT3'] - (data['BILL_AMT4'] - data['PAY_AMT3'])) +

                 (data['BILL_AMT2'] - (data['BILL_AMT3'] - data['PAY_AMT2']))) / 4) / data['LIMIT_BAL']

data['average_1'] = (((data['BILL_AMT5'] - (data['BILL_AMT6'] - data['PAY_AMT5'])) +

                 (data['BILL_AMT4'] - (data['BILL_AMT5'] - data['PAY_AMT4'])) +

                 (data['BILL_AMT3'] - (data['BILL_AMT4'] - data['PAY_AMT3'])) +

                 (data['BILL_AMT2'] - (data['BILL_AMT3'] - data['PAY_AMT2'])) +

                 (data['BILL_AMT1'] - (data['BILL_AMT2'] - data['PAY_AMT1']))) / 5) / data['LIMIT_BAL']

average=data[['LIMIT_BAL', 'average_5', 'BILL_AMT5', 'average_4', 'BILL_AMT4','average_3', 'BILL_AMT3',

    'average_2', 'BILL_AMT2', 'average_1', 'BILL_AMT1', 'def_pay']].sample(20)

print(average)
data['InvoiceLimit_6'] = (data.LIMIT_BAL - data.BILL_AMT6) / data.LIMIT_BAL

data['InvoiceLimit_5'] = (data.LIMIT_BAL - data.BILL_AMT5) / data.LIMIT_BAL

data['InvoiceLimit_4'] = (data.LIMIT_BAL - data.BILL_AMT4) / data.LIMIT_BAL

data['InvoiceLimit_3'] = (data.LIMIT_BAL - data.BILL_AMT3) / data.LIMIT_BAL

data['InvoiceLimit_2'] = (data.LIMIT_BAL - data.BILL_AMT2) / data.LIMIT_BAL

data['InvoiceLimit_1'] = (data.LIMIT_BAL - data.BILL_AMT1) / data.LIMIT_BAL

InvoiceLimit=data[['InvoiceLimit_6', 'InvoiceLimit_5', 'InvoiceLimit_4', 'InvoiceLimit_3', 'InvoiceLimit_2',

   'InvoiceLimit_1', 'def_pay']].sample(20)

print(InvoiceLimit)
data['PAY_1_-1'] = (data.PAY_1 == -1)

data['PAY_1_-2'] = (data.PAY_1 == -2)

data['PAY_1_0'] = (data.PAY_1 == 0)

data['PAY_1_1'] = (data.PAY_1 == 1)

data['PAY_1_2'] = (data.PAY_1 == 2)

data['PAY_1_3'] = (data.PAY_1 == 3)

data['PAY_1_4'] = (data.PAY_1 == 4)

data['PAY_1_5'] = (data.PAY_1 == 5)

data['PAY_1_6'] = (data.PAY_1 == 6)

data['PAY_1_7'] = (data.PAY_1 == 7)

data['PAY_1_8'] = (data.PAY_1 == 8)



data['PAY_2_-1'] = (data.PAY_1 == -1)

data['PAY_2_-2'] = (data.PAY_1 == -2)

data['PAY_2_0'] = (data.PAY_1 == 0)

data['PAY_2_1'] = (data.PAY_1 == 1)

data['PAY_2_2'] = (data.PAY_1 == 2)

data['PAY_2_3'] = (data.PAY_1 == 3)

data['PAY_2_4'] = (data.PAY_1 == 4)

data['PAY_2_5'] = (data.PAY_1 == 5)

data['PAY_2_6'] = (data.PAY_1 == 6)

data['PAY_2_7'] = (data.PAY_1 == 7)

data['PAY_2_8'] = (data.PAY_1 == 8)



data['PAY_3_-1'] = (data.PAY_1 == -1)

data['PAY_3_-2'] = (data.PAY_1 == -2)

data['PAY_3_0'] = (data.PAY_1 == 0)

data['PAY_3_1'] = (data.PAY_1 == 1)

data['PAY_3_2'] = (data.PAY_1 == 2)

data['PAY_3_3'] = (data.PAY_1 == 3)

data['PAY_3_4'] = (data.PAY_1 == 4)

data['PAY_3_5'] = (data.PAY_1 == 5)

data['PAY_3_6'] = (data.PAY_1 == 6)

data['PAY_3_7'] = (data.PAY_1 == 7)

data['PAY_3_8'] = (data.PAY_1 == 8)



data['PAY_4_-1'] = (data.PAY_1 == -1)

data['PAY_4_-2'] = (data.PAY_1 == -2)

data['PAY_4_0'] = (data.PAY_1 == 0)

data['PAY_4_1'] = (data.PAY_1 == 1)

data['PAY_4_2'] = (data.PAY_1 == 2)

data['PAY_4_3'] = (data.PAY_1 == 3)

data['PAY_4_4'] = (data.PAY_1 == 4)

data['PAY_4_5'] = (data.PAY_1 == 5)

data['PAY_4_6'] = (data.PAY_1 == 6)

data['PAY_4_7'] = (data.PAY_1 == 7)

data['PAY_4_8'] = (data.PAY_1 == 8)



data['PAY_5_-1'] = (data.PAY_1 == -1)

data['PAY_5_-2'] = (data.PAY_1 == -2)

data['PAY_5_0'] = (data.PAY_1 == 0)

data['PAY_5_1'] = (data.PAY_1 == 1)

data['PAY_5_2'] = (data.PAY_1 == 2)

data['PAY_5_3'] = (data.PAY_1 == 3)

data['PAY_5_4'] = (data.PAY_1 == 4)

data['PAY_5_5'] = (data.PAY_1 == 5)

data['PAY_5_6'] = (data.PAY_1 == 6)

data['PAY_5_7'] = (data.PAY_1 == 7)

data['PAY_5_8'] = (data.PAY_1 == 8)



data['PAY_6_-1'] = (data.PAY_1 == -1)

data['PAY_6_-2'] = (data.PAY_1 == -2)

data['PAY_6_0'] = (data.PAY_1 == 0)

data['PAY_6_1'] = (data.PAY_1 == 1)

data['PAY_6_2'] = (data.PAY_1 == 2)

data['PAY_6_3'] = (data.PAY_1 == 3)

data['PAY_6_4'] = (data.PAY_1 == 4)

data['PAY_6_5'] = (data.PAY_1 == 5)

data['PAY_6_6'] = (data.PAY_1 == 6)

data['PAY_6_7'] = (data.PAY_1 == 7)

data['PAY_6_8'] = (data.PAY_1 == 8)

data['PAY_6_8'] = (data.PAY_1 == 8)
pd.crosstab(data.SEX,data.def_pay,normalize=False).plot(kind="bar",rot=0,figsize=(20,6))

plt.title('Default Payment by Sex')

plt.xlabel('Sex (1 = Male, 2 = Female)' )

plt.legend(["No Payment", "Paying"])

plt.ylabel('Frequency')

plt.grid()

plt.show()
pd.crosstab(data.MARRIAGE,data.def_pay,normalize=False).plot(kind="bar",rot=0,figsize=(20,6))

plt.title('Default Payment by Marriage')

plt.xlabel('Marriage(1=married, 2=single ,3=others)' )

plt.legend(["No Payment", "Paying"])

plt.ylabel('Frequency')

plt.grid()

plt.show()
pd.crosstab(data.EDUCATION,data.def_pay,normalize=False).plot(kind="bar",rot=0,figsize=(20,6))

plt.title('Default Payment by Education')

plt.xlabel('Education(1=Graduate School, 2=University ,3=High School ,4=Others)' )

plt.legend(["No Payment", "Paying"])

plt.ylabel('Frequency')

plt.grid()

plt.show()
pd.crosstab(data.AgeBin,data.def_pay,normalize=False).plot(kind="bar",rot=0,figsize=(20,6))

plt.title('Default Payment by AgeBin')

plt.xlabel('Age Bin\n (for 1) Age = [20,30) \n (for 2) Age = [30,40) \n' +

           ' (for 3) Age = [40,50) \n (for 4) Age = [50,60) \n (for 5) Age = [60,70) \n(for 6) Age = [70,81)')

plt.legend(["No Payment", "Paying"])

plt.ylabel('Frequency')

plt.grid()

plt.show()
pd.crosstab(data.AgeBin,data.def_pay,normalize=False).plot(kind="bar",rot=0,figsize=(20,6))

plt.title('Default Payment by AgeBin')

plt.xlabel('Age Bin\n (for 1) Age = [20,30) \n (for 2) Age = [30,40) \n' +

           ' (for 3) Age = [40,50) \n (for 4) Age = [50,60) \n (for 5) Age = [60,70) \n(for 6) Age = [70,81)')

plt.legend(["No Payment", "Paying"])

plt.ylabel('Frequency')

plt.grid()

plt.show()
pd.crosstab(data.SE_MA,data.def_pay,normalize=False).plot(kind="bar",rot=0,figsize=(20,6))

plt.title('Default Payment by SEX and MARRIAGE')

plt.xlabel('SEX & MARRIAGE\n 1 = Married Man, 2 = Single Man, 3 = Divorced Man, 4 = Married Woman, 5 = Single Woman, 6 = Divorced Woman')

plt.legend(["No Payment", "Paying"])

plt.ylabel('Frequency')

plt.grid()

plt.show()
pd.crosstab(data.SE_AG,data.def_pay,normalize=False).plot(kind="bar",rot=0,figsize=(20,6))

plt.title('Default Payment by SEX and AGE')

plt.xlabel('SEX & AGE\n 1 = 20s Man, 2 = 30s Man, ..., 5 = 60s Man, 6 = 20s Woman, 7 = 30s Woman, ..., 10 = 60s Woman')

plt.legend(["No Payment", "Paying"])

plt.ylabel('Frequency')

plt.grid()

plt.show()
features = ['LIMIT_BAL', 'EDUCATION','BILL_AMT1', 'BILL_AMT2',

            'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6', 'PAY_AMT1',

            'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6', 

            'SE_MA', 'AgeBin', 'SE_AG', 'average_5', 'average_4',

            'average_3', 'average_2', 'average_1', 'InvoiceLimit_5', 'InvoiceLimit_6',

            'InvoiceLimit_4', 'InvoiceLimit_3', 'InvoiceLimit_2','InvoiceLimit_1',

            'active_6','active_5','active_4','active_3','active_2','active_1','PAY_1_-1',

            'PAY_1_-2', 'PAY_1_0', 'PAY_1_1', 'PAY_1_2', 'PAY_1_3', 'PAY_1_4', 

            'PAY_1_5', 'PAY_1_6', 'PAY_1_7', 'PAY_1_8', 'PAY_2_-1', 'PAY_2_-2', 

            'PAY_2_0', 'PAY_2_1', 'PAY_2_2', 'PAY_2_3', 'PAY_2_4', 'PAY_2_5', 

            'PAY_2_6', 'PAY_2_7', 'PAY_2_8', 'PAY_3_-1', 'PAY_3_-2', 'PAY_3_0', 

            'PAY_3_1', 'PAY_3_2', 'PAY_3_3', 'PAY_3_4', 'PAY_3_5', 'PAY_3_6', 

            'PAY_3_7', 'PAY_3_8', 'PAY_4_-1', 'PAY_4_-2', 'PAY_4_0', 'PAY_4_1', 

            'PAY_4_2', 'PAY_4_3', 'PAY_4_4', 'PAY_4_5', 'PAY_4_6', 'PAY_4_7', 

            'PAY_4_8', 'PAY_5_-1', 'PAY_5_-2', 'PAY_5_0', 'PAY_5_2', 'PAY_5_3', 

            'PAY_5_4', 'PAY_5_5', 'PAY_5_6', 'PAY_5_7', 'PAY_5_8', 'PAY_6_-1', 

            'PAY_6_-2', 'PAY_6_0', 'PAY_6_2', 'PAY_6_3', 'PAY_6_4', 'PAY_6_5', 

            'PAY_6_6', 'PAY_6_7', 'PAY_6_8']

target = 'def_pay'

y = data['def_pay'].copy()

X = data[features].copy()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)



data_train = X_train.join(y_train)
rfclassifier = RandomForestClassifier(random_state=42,n_estimators=200,criterion='entropy',

                                       max_features='sqrt',max_depth=7,verbose=False)

rfclassifier.fit(X_train[features], y_train)

rfprediction = rfclassifier.predict(X_test[features])

print('Accuracy of Random Forest Classifier: ',accuracy_score(rfprediction,y_test))
cmrf = pd.crosstab(y_test.values, rfprediction, rownames=['Actual'], colnames=['Predicted'])

fig, (ax1) = plt.subplots(ncols=1, figsize=(5,5))

sns.heatmap(cmrf, fmt="d",

            xticklabels=['Not Default', 'Default'],

            yticklabels=['Not Default', 'Default'],

            annot=True,ax=ax1,

            linewidths=.2,linecolor="Green", cmap="Greens")

plt.title('Confusion Matrix for Random Forest', fontsize=14)

plt.show()



rocaucscorerf=roc_auc_score(y_test.values, rfprediction)

print('Roc Score: ',rocaucscorerf)
dtclassifier = DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=4,

                       max_features=None, max_leaf_nodes=20,

                       min_impurity_decrease=0.0, min_impurity_split=None,

                       min_samples_leaf=1, min_samples_split=5,

                       min_weight_fraction_leaf=0.0, presort=False,

                       random_state=42, splitter='best')

dtclassifier.fit(X_train, y_train)

dtprediction = dtclassifier.predict(X_test)

print('Accuracy of Decision Tree:', accuracy_score(dtprediction,y_test))
cmdt = pd.crosstab(y_test.values, dtprediction, rownames=['Actual'], colnames=['Predicted'])

fig, (ax2) = plt.subplots(ncols=1, figsize=(5,5))

sns.heatmap(cmdt, fmt = "d",

            xticklabels=['Not Default', 'Default'],

            yticklabels=['Not Default', 'Default'],

            annot=True,ax=ax2,

            linewidths=.2,linecolor="Red", cmap="Reds")

plt.title('Confusion Matrix for Decision Tree', fontsize=14)

plt.show()



rocaucscoredt=roc_auc_score(y_test.values, dtprediction)

print('Roc Score: ',rocaucscoredt)
knnclassifier=KNeighborsClassifier(n_neighbors=8,algorithm='auto',

                                    leaf_size=30,metric='minkowski')

knnclassifier.fit(X_train, y_train)

trainaccuracy=knnclassifier.score(X_train, y_train)

testaccuracy=knnclassifier.score(X_test, y_test)

predictionknn=knnclassifier.predict(X_test)

print('train accuracy: {}\ntest accuracy: {}\n'.format(trainaccuracy,testaccuracy))
cmknn = pd.crosstab(y_test.values, predictionknn, rownames=['Actual'], colnames=['Predicted'])

fig, (ax3) = plt.subplots(ncols=1, figsize=(5,5))

sns.heatmap(cmknn, fmt="d",

            xticklabels=['Not Default', 'Default'],

            yticklabels=['Not Default', 'Default'],

            annot=True,ax=ax3,

            linewidths=.2,linecolor="Blue", cmap="Blues")

plt.title('Confusion Matrix for KNN', fontsize=14)

plt.show()





rocaucscoreknn=roc_auc_score(y_test.values, predictionknn )

print('Roc Score: ',rocaucscoreknn)
adaboostclassifier = AdaBoostClassifier(base_estimator=None, 

                                         n_estimators=50, 

                                         learning_rate=1.5, 

                                         algorithm='SAMME', 

                                         random_state=42)



adaboostclassifier.fit(X_train[features], y_train.values)

adaboostprediction = adaboostclassifier.predict(X_test[features])

print('Accuracy of Ada Boost:', accuracy_score(adaboostprediction,y_test))
cmadaboost = pd.crosstab(y_test.values, adaboostprediction, 

                     rownames=['Actual'], colnames=['Predicted'])

fig, (ax4) = plt.subplots(ncols=1, figsize=(5,5))

sns.heatmap(cmadaboost, fmt="d",

            xticklabels=['Not Default', 'Default'],

            yticklabels=['Not Default', 'Default'],

            annot=True,ax=ax4,

            linewidths=.2,linecolor="Purple", cmap="Purples")

plt.title('Confusion Matrix for Adaboost', fontsize=14)

plt.show()



rocaucscoreadaboost=roc_auc_score(y_test.values, adaboostprediction)

print('Roc Score: ',rocaucscoreadaboost)
y_pred_proba_DT = dtclassifier.predict_proba(X_test)[::,1]

fpr1, tpr1, _ = roc_curve(y_test, y_pred_proba_DT)

auc1 = roc_auc_score(y_test, y_pred_proba_DT)



y_pred_proba_RF = rfclassifier.predict_proba(X_test)[::,1]

fpr2, tpr2, _ = roc_curve(y_test,  y_pred_proba_RF)

auc2 = roc_auc_score(y_test, y_pred_proba_RF)



y_pred_proba_KNN = knnclassifier.predict_proba(X_test)[::,1]

fpr3, tpr3, _ = roc_curve(y_test,  y_pred_proba_KNN)

auc3 = roc_auc_score(y_test, y_pred_proba_KNN)



y_pred_proba_ADABOOST = adaboostclassifier.predict_proba(X_test)[::,1]

fpr4, tpr4, _ = roc_curve(y_test,  y_pred_proba_ADABOOST)

auc4 = roc_auc_score(y_test, y_pred_proba_ADABOOST)



plt.figure(figsize=(10,7))

plt.title('ROC', size=15)

plt.plot([0, 1], [0, 1], 'k--')

plt.plot(fpr1,tpr1,label="Decision Tree, auc="+str(round(auc1,2)))

plt.plot(fpr2,tpr2,label="Random Forest, auc="+str(round(auc2,2)))

plt.plot(fpr3,tpr3,label="KNearest Neighbor, auc="+str(round(auc3,2)))

plt.plot(fpr4,tpr4,label="AdaBoost, auc="+str(round(auc4,2)))

plt.legend(loc='best', title='Models', facecolor='white')

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.box(False)

plt.grid()

plt.show()

clf_list = [DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=4,

                                   max_features=None, max_leaf_nodes=20,

                                   min_impurity_decrease=0.0, min_impurity_split=None,

                                   min_samples_leaf=1, min_samples_split=5,

                                   min_weight_fraction_leaf=0.0, presort=False,

                                   random_state=42, splitter='best'), 

            RandomForestClassifier(random_state=42,n_estimators=200,criterion='entropy',

                                    max_features='sqrt',max_depth=7,verbose=False),

            KNeighborsClassifier(n_neighbors=8,algorithm='auto',

                                    leaf_size=30,metric='minkowski'), 

            AdaBoostClassifier(base_estimator=None, 

                                    n_estimators=50, 

                                    learning_rate=1.5, 

                                    algorithm='SAMME', 

                                    random_state=42)

           ]

# use Kfold to evaluate the normal training set

kf = KFold(n_splits=5,random_state=42,shuffle=True)



mdl = []

fold = []

fscr = []

rocscr = []

accscr = []





for i,(train_index, test_index) in enumerate(kf.split(data_train)):

    training = data.iloc[train_index,:]

    valid = data.iloc[test_index,:]

    print(i)

    for clf in clf_list:

        model = clf.__class__.__name__

        feats = training[features] #defined above

        label = training['def_pay']

        valid_feats = valid[features]

        valid_label = valid['def_pay']

        clf.fit(feats,label) 

        pred = clf.predict(valid_feats)

        fscore = f1_score(y_true = valid_label, y_pred = pred)

        rocscore = roc_auc_score(valid_label, pred)

        accscore = accuracy_score(y_true = valid_label, y_pred = pred)

        fold.append(i+1)

        fscr.append(fscore)

        rocscr.append(rocscore)

        accscr.append(accscore)

        mdl.append(model)

        print(model)
performance = pd.DataFrame({'Model': mdl, 'Score':fscr,

                            'Roc_Auc_Score':rocscr,'Accuracy_Score':accscr,'Fold':fold})



dtcc = performance[performance['Model'] == 'DecisionTreeClassifier']

rfcc = performance[performance['Model'] == 'RandomForestClassifier']

abcc = performance[performance['Model'] == 'AdaBoostClassifier']

knnn = performance[performance['Model'] == 'KNeighborsClassifier']



plt.figure(figsize=(15,10))

plt.plot(dtcc.Fold,dtcc.Score,'red',label="DT F1",marker='o')

plt.plot(dtcc.Fold,dtcc.Roc_Auc_Score,'firebrick',label="DT Roc",marker='x')

plt.plot(dtcc.Fold,dtcc.Accuracy_Score,'rosybrown',label="DT Acc",marker='.')



plt.plot(rfcc.Fold,rfcc.Score,'olive',label="RF F1",marker='o')

plt.plot(rfcc.Fold,rfcc.Roc_Auc_Score,'yellowgreen',label="RF Roc",marker='x')

plt.plot(rfcc.Fold,rfcc.Accuracy_Score,'lightgreen',label="RF Acc",marker='.')



plt.plot(knnn.Fold,knnn.Score,'purple',label="KNN F1",marker='o')

plt.plot(knnn.Fold,knnn.Roc_Auc_Score,'violet',label="KNN Roc",marker='x')

plt.plot(knnn.Fold,knnn.Accuracy_Score,'fuchsia',label="KNN Acc",marker='.')



plt.plot(abcc.Fold,abcc.Score,'lightskyblue',label="AdaB F1",marker='o')

plt.plot(abcc.Fold,abcc.Roc_Auc_Score,'blue',label="AdaB Roc",marker='x')

plt.plot(abcc.Fold,abcc.Accuracy_Score,'navy',label="AdaB Acc",marker='.')



plt.title("Classifiers")

plt.grid()

plt.legend(loc='best')

plt.show()
plt.figure(figsize=(10,8))

plt.plot(dtcc.Fold,dtcc.Score,'r',label="DecisionTreeClassifier",marker='o')

plt.plot(rfcc.Fold,rfcc.Score,'b',label="RandomForestClassifier",marker='o')

plt.plot(knnn.Fold,knnn.Score,'c',label="KNeighborsClassifier",marker='o')

plt.plot(abcc.Fold,abcc.Score,'g',label="AdaBoostClassifier",marker='o')

plt.title("Classifiers F1 Score")

plt.grid()

plt.legend(loc='best')

plt.show()
plt.figure(figsize=(10,8))

plt.plot(dtcc.Fold,dtcc.Roc_Auc_Score,'r',label="DecisionTreeClassifier",marker='o')

plt.plot(rfcc.Fold,rfcc.Roc_Auc_Score,'b',label="RandomForestClassifier",marker='o')

plt.plot(knnn.Fold,knnn.Roc_Auc_Score,'c',label="KNeighborsClassifier",marker='o')

plt.plot(abcc.Fold,abcc.Roc_Auc_Score,'g',label="AdaBoostClassifier",marker='o')

plt.title("Classifiers Roc Score")

plt.grid()

plt.legend(loc='best')

plt.show()
plt.figure(figsize=(8,8))

plt.plot(dtcc.Fold,dtcc.Accuracy_Score,'r',label="DecisionTreeClassifier",marker='o')

plt.plot(rfcc.Fold,rfcc.Accuracy_Score,'b',label="RandomForestClassifier",marker='o')

plt.plot(knnn.Fold,knnn.Accuracy_Score,'c',label="KNeighborsClassifier",marker='o')

plt.plot(abcc.Fold,abcc.Accuracy_Score,'g',label="AdaBoostClassifier",marker='o')

plt.title("Classifiers Accuracy Score")

plt.grid()

plt.legend(loc='best')

plt.show()
new_features = ['LIMIT_BAL', 'BILL_AMT1', 'BILL_AMT2',

            'BILL_AMT3', 'BILL_AMT5', 'PAY_AMT1',

            'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6', 

            'average_4','average_3', 'average_2', 'average_1', 'InvoiceLimit_5',

            'InvoiceLimit_4', 'InvoiceLimit_3', 'InvoiceLimit_2','InvoiceLimit_1', 

            'PAY_1_1', 'PAY_1_2',

            'PAY_2_0', 'PAY_2_2', 'PAY_2_3', 

            'PAY_3_0', 

            'PAY_3_1', 'PAY_3_2', 'PAY_3_3', 'PAY_4_0', 'PAY_4_1', 

            'PAY_4_2', 'PAY_4_3', 'PAY_5_0', 'PAY_5_2', 'PAY_5_3',

            'PAY_6_2']



rfclassifier_new = RandomForestClassifier(random_state=42,n_estimators=200,criterion='entropy',

                                       max_features='sqrt',max_depth=7,verbose=False)

rfclassifier_new.fit(X_train[new_features], y_train)

rfprediction_new = rfclassifier_new.predict(X_test[new_features])

print('Accuracy of New Random Forest: ',accuracy_score(rfprediction_new,y_test))
cmrf_new = pd.crosstab(y_test.values, rfprediction_new, rownames=['Actual'], colnames=['Predicted'])

fig, (ax5) = plt.subplots(ncols=1, figsize=(5,5))

sns.heatmap(cmrf_new, fmt="d",

            xticklabels=['Not Default', 'Default'],

            yticklabels=['Not Default', 'Default'],

            annot=True,ax=ax5,

            linewidths=.2,linecolor="Green", cmap="Greens")

plt.title('Confusion Matrix in New Random Forest', fontsize=14)

plt.show()

rocaucscorerf=roc_auc_score(y_test.values, rfprediction)

print('Roc Score: ',rocaucscorerf)
tmp = pd.DataFrame({'Features': new_features, 'Importance of Features': rfclassifier_new.feature_importances_})

tmp = tmp.sort_values(by='Importance of Features',ascending=False)

plt.figure(figsize = (25,15))

plt.title('Importance of Features',fontsize=14)

s = sns.barplot(x='Features',y='Importance of Features',data=tmp)

s.set_xticklabels(s.get_xticklabels(),rotation=90)

plt.grid()

plt.show()