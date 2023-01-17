from os.path import join as pjoin

from collections import Counter



import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import datetime

PATH_TO_DATA = '../input/lab34-classification-table'
data = pd.read_csv(pjoin(PATH_TO_DATA, 'train.csv'))

data.shape





dataTest = pd.read_csv(pjoin(PATH_TO_DATA, 'test.csv'))





data.describe()
labels = data['y']




data = data.astype({

    'birth_date': 'datetime64[ns]',

    'contact_date': 'datetime64[ns]',

})

dataTest = dataTest.astype({

    'birth_date': 'datetime64[ns]',

    'contact_date': 'datetime64[ns]',

})



data['birth_year']=data['birth_date'].dt.year

data['birth_month']=data['birth_date'].dt.month

data['birth_day']=data['birth_date'].dt.day



data['contact_year']=data['contact_date'].dt.year

data['contact_month']=data['contact_date'].dt.month

data['contact_day']=data['contact_date'].dt.day











data = data.drop(columns=['contact_date', 'birth_date'])









dataTest['birth_year']=dataTest['birth_date'].dt.year

dataTest['birth_month']=dataTest['birth_date'].dt.month

dataTest['birth_day']=dataTest['birth_date'].dt.day



dataTest['contact_year']=dataTest['contact_date'].dt.year

dataTest['contact_month']=dataTest['contact_date'].dt.month

dataTest['contact_day']=dataTest['contact_date'].dt.day



dataTest = dataTest.drop(columns=['contact_date', 'birth_date'])





print(data)


# женатые



isMarried = data['marital'].str.lower()

#data['is_married'] = isMarried 



isMarriedTest = dataTest['marital'].str.lower()

#dataTest['is_married'] = isMarriedTest





pd.crosstab(isMarried, labels).plot(kind='bar')
#возраст





now = datetime.datetime.now()

print(now)

print(type(now))

print(now.timetuple()[0])

print(type(now.timetuple()[0]))









ages = now.timetuple()[0] - data['birth_year']

data['years_old'] = ages.astype(int)



agesTest = now.timetuple()[0] - dataTest['birth_year']

dataTest['years_old'] = agesTest



#cont_years_old = 



pd.crosstab(ages, labels).plot(kind='bar')
#в каком возрасте обратился





now = datetime.datetime.now()

print(now)

print(type(now))





ages = now.timetuple()[0] - data['contact_year']

data['years_old_contact'] = ages.astype(int)



agesTest = now.timetuple()[0] - dataTest['contact_year']

dataTest['years_old_contact'] = agesTest



#cont_years_old = 



pd.crosstab(ages, labels).plot(kind='bar')



categorical_columns = [c for c in data.columns if data[c].dtype.name == 'object']

numerical_columns   = [c for c in data.columns if data[c].dtype.name != 'object']



categorical_columnsTest = [c for c in dataTest.columns if data[c].dtype.name == 'object']

numerical_columnsTest   = [c for c in dataTest.columns if data[c].dtype.name != 'object']







print(categorical_columns)

print(numerical_columns)
dataTest = dataTest.fillna(dataTest.median(axis=0), axis=0)

dataTest_describe = dataTest.describe(include=[object])

for c in categorical_columnsTest:

    dataTest[c] = dataTest[c].fillna(dataTest_describe[c]['top'])

    

    

dataTest.describe(include=[object])
for c in categorical_columns:

    print(data[c].unique())

for c in categorical_columns:

    print(dataTest[c].unique())
data.describe()
data[categorical_columns].describe()
data.head()

pd.crosstab(data['education'], labels).plot(kind='bar')
pd.crosstab(data['job'], labels).plot(kind='bar')
pd.crosstab(data['default'], labels).plot(kind='bar')
pd.crosstab(data['housing'], labels).plot(kind='bar')
pd.crosstab(data['loan'], labels).plot(kind='bar')
pd.crosstab(data['contact'], labels).plot(kind='bar')
pd.crosstab(data['pdays'], labels).plot(kind='bar')
pd.crosstab(data['poutcome'], labels).plot(kind='bar')
data.count(axis=0)



data_describe = data.describe(include=[object])

binary_columns    = [c for c in categorical_columns if data_describe[c]['unique'] == 2]

nonbinary_columns = [c for c in categorical_columns if data_describe[c]['unique'] > 2]









dataTest_describe = dataTest.describe(include=[object])

binary_columnsTest    = [c for c in categorical_columnsTest if dataTest_describe[c]['unique'] == 2]

nonbinary_columnsTest = [c for c in categorical_columnsTest if dataTest_describe[c]['unique'] > 2]

print(binary_columns, nonbinary_columns)

print(binary_columnsTest, nonbinary_columnsTest)



# векторизация бинарных признаков



data.at[data['contact'] == 'cellular', 'contact'] = 0

data.at[data['contact'] == 'telephone', 'contact'] = 1





data.at[data['default'] == 'yes', 'default'] = 0

data.at[data['default'] == 'no', 'default'] = 1

data.at[data['default'] == 'unknown', 'default'] = 2

nonbinary_columns.remove('default')

numerical_columns.append('default')



dataTest.at[dataTest['contact'] == 'cellular', 'contact'] = 0

dataTest.at[dataTest['contact'] == 'telephone', 'contact'] = 1



dataTest.at[dataTest['default'] == 'yes', 'default'] = 0

dataTest.at[dataTest['default'] == 'no', 'default'] = 1

dataTest.at[dataTest['default'] == 'unknown', 'default'] = 2



binary_columnsTest.remove('default')

numerical_columnsTest.append('default')



print(binary_columns, nonbinary_columns, numerical_columns)

print(binary_columnsTest, nonbinary_columnsTest, numerical_columnsTest)





data['contact'].describe()

data_describe = data.describe(include=[object])



dataTest_describe = dataTest.describe(include=[object])

data





data_nonbinary = pd.get_dummies(data[nonbinary_columns])



dataTest_nonbinary = pd.get_dummies(dataTest[nonbinary_columnsTest])



print(data_nonbinary.columns)

data_nonbinary
data_numerical = data[numerical_columns]

dataTest_numerical = dataTest[numerical_columnsTest]







print(numerical_columns)

print(binary_columns)

print(data_nonbinary.columns)



print(data_numerical)

print(data[binary_columns])

print(data_nonbinary)





data = pd.concat((data_numerical.astype(int), data[binary_columns].astype(int), data_nonbinary.astype(int)), axis=1)

data = pd.DataFrame(data, dtype=int)



print(data)



#print(data[numerical_columnsTest])

#print(data[binary_columnsTest])

#print(data_nonbinary)











dataTest = pd.concat((dataTest_numerical.astype(int), dataTest[binary_columns].astype(int), dataTest_nonbinary.astype(int)), axis=1)



dataTest = pd.DataFrame(dataTest, dtype=int)

print(data.shape)

print(data.columns)

print(dataTest.shape)



from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, precision_score, recall_score, precision_recall_curve

from catboost import CatBoostClassifier

#W = data.columns(columns=['is_married'])

cols = list(data)

W = data.drop(columns=['y'])

cols.remove('y')

l = data.drop(columns=cols)





WTest = dataTest

print(W.columns)

print(WTest.columns)

WTrain, WTrainTest, lTrain, lTrainTest = train_test_split(W, l, test_size=0.3, random_state=1)

# WTrain = W

# lTrain = l





def getAcuracy(lTrue, predProba, threshold=0.5):

    pred = np.zeros_like(predProba)

    pred[predProba > threshold] = 1

    acuracy = accuracy_score(lTrue, pred)

    return acuracy


gb = CatBoostClassifier(

    cat_features= WTrain,

    eval_metric='AUC',

    random_seed=1,

    nan_mode='Forbidden',

    task_type='CPU',

    verbose=True,

    n_estimators=150,

    max_depth=6,

)

gb.fit(WTrain, lTrain)
# Quality on train

predProbaTrain = gb.predict_proba(WTrain)[:, 1]

acc = getAcuracy(lTrain, predProbaTrain)

print("Acuracy = ", acc)

# Quality on test

predProbaTest = gb.predict_proba(WTrainTest)[:, 1]

acc = getAcuracy(lTrainTest, predProbaTest)

print("Acuracy = ", acc)
pr, rec, thr = precision_recall_curve(lTrain, predProbaTrain)

f1 = 2 * (pr * rec) / (pr + rec)

best_thr = thr[f1.argmax() - 1]

best_thr, f1.max()

print(W.columns)

print(WTest.columns)
predTest = gb.predict_proba(WTest)[:, 1]



answers = np.zeros(len(predTest), dtype=bool)





for i in range(len(predTest)):

    if predTest[i] >= best_thr :

        answers[i] = True

    

        





res = pd.DataFrame({'y': predTest, 'id': WTest.index})

res.to_csv('result.csv', index=False)

print(WTest.shape)

print(predTest.shape)


