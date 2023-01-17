import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import sklearn as skl

import math

from sklearn.model_selection import train_test_split

%matplotlib inline



np.random.seed(0)
test = pd.read_csv('../input/bits-f464-l1/test.csv', sep=',')

train = pd.read_csv('../input/bits-f464-l1/train.csv', sep=',')

train = train.drop(["id"], axis = 1)

train.head(10)
null_columns = train.isnull().sum().sort_values(ascending=False)

null_columns = null_columns[null_columns > 0]

null_columns

for column in null_columns.index:

        train[column].fillna(train[column].mean(),inplace=True)

train.info()

gnull_columns = train.isnull().sum().sort_values(ascending=False)

gnull_columns = gnull_columns[gnull_columns > 0]

gnull_columns
null_columns = test.isnull().sum().sort_values(ascending=False)

null_columns = null_columns[null_columns > 0]

null_columns

for column in null_columns.index:

        test[column].fillna(test[column].mean(),inplace=True)

test.info()

gnull_columns = test.isnull().sum().sort_values(ascending=False)

gnull_columns = gnull_columns[gnull_columns > 0]

gnull_columns
#train = pd.get_dummies(data = train, columns=['type'])

#test = pd.get_dummies(data = test, columns=['type'])

# corr_matrix = train.corr().abs()

# pd.options.display.max_rows = None



# to_drop = []

# temp = corr_matrix["label"]

# for i in temp.index:

#     if(pd.isna(temp[i])):

#         to_drop.append(i)

# to_drop

# #train = train.drop(train[to_drop], axis=1)

# #test = test.drop(test[to_drop],axis=1)
train0 = train.loc[train['a0']==1]

train0 = train0.drop(columns = ['a1','a2','a3','a4','a5','a6'])



label0 = train0['label']

train0 = train0.drop(columns = ['label'])



# corr_matrix = train0.corr().abs()

# pd.options.display.max_rows = None



# to_drop = []

# temp = corr_matrix["label"]

# for i in temp.index:

#     if(pd.isna(temp[i])):

#         to_drop.append(i)

# to_drop

# train0 = train0.drop(train0[to_drop], axis=1)



test0 = test.loc[test['a0']==1]

test0 = test0.drop(columns = ['a1','a2','a3','a4','a5','a6'])

id0 = test0['id']

# test0 = test0.drop(test0[to_drop], axis=1)



corr = train0.corr().abs()

columns = np.full((corr.shape[0],), True, dtype=bool)

for i in range(corr.shape[0]):

    for j in range(i+1, corr.shape[0]):

        if (corr.iloc[i,j] >= 0.99):

            if columns[j]:

                columns[j] = False

selected_columns = train0.columns[columns]

train0 = train0[selected_columns]



test0=test0[selected_columns]
from sklearn import preprocessing

scaler = preprocessing.StandardScaler()

scaled_data = scaler.fit_transform(train0)

train0 = pd.DataFrame(scaled_data,columns=train0.columns)

train0.head()



scaled_data = scaler.fit_transform(test0)

test0 = pd.DataFrame(scaled_data,columns=test0.columns)

test0.head()
x_train0_val,x_test0_val,y_train0_val,y_test0_val = train_test_split(train0,label0,test_size=0.3,random_state=27)
from sklearn.ensemble import RandomForestRegressor

regressor = RandomForestRegressor(n_estimators=350, max_depth=20, n_jobs=-1, random_state=60)

regressor.fit(x_train0_val, y_train0_val)



from sklearn.metrics import mean_squared_error

np.sqrt(mean_squared_error(y_test0_val,[round(s) for s in regressor.predict(x_test0_val)]))
from sklearn.ensemble import RandomForestRegressor

regressor = RandomForestRegressor(n_estimators=350, max_depth=20, n_jobs=-1, random_state=60)

regressor.fit(train0, label0)

y_pred0 = regressor.predict(test0)



ids = np.array(id0)

dict = {'id': ids, 'label': y_pred0}

submission_dataFrame = pd.DataFrame(dict)

submission_dataFrame.to_csv('mfinal0.csv', header=True, index=False)
train1 = train.loc[train['a1']==1]

train1 = train1.drop(columns = ['a0','a2','a3','a4','a5','a6'])

label1 = train1['label']

train1 = train1.drop(columns = ['label'])



corr_matrix = train1.corr().abs()

pd.options.display.max_rows = None



# to_drop = []

# temp = corr_matrix["label"]

# for i in temp.index:

#     if(pd.isna(temp[i])):

#         to_drop.append(i)

# to_drop

# train1 = train1.drop(train1[to_drop], axis=1)



test1 = test.loc[test['a1']==1]

test1 = test1.drop(columns = ['a0','a2','a3','a4','a5','a6'])

id1 = test1['id']

# test1 = test1.drop(test1[to_drop], axis=1)



corr = train1.corr().abs()

columns = np.full((corr.shape[0],), True, dtype=bool)

for i in range(corr.shape[0]):

    for j in range(i+1, corr.shape[0]):

        if (corr.iloc[i,j] >= 0.99):

            if columns[j]:

                columns[j] = False

selected_columns = train1.columns[columns]

train1 = train1[selected_columns]



test1=test1[selected_columns]
from sklearn import preprocessing

scaler = preprocessing.StandardScaler()

scaled_data = scaler.fit_transform(train1)

train1 = pd.DataFrame(scaled_data,columns=train1.columns)

train1.head()



scaled_data = scaler.fit_transform(test1)

test1 = pd.DataFrame(scaled_data,columns=test1.columns)

test1.head()
x_train1_val,x_test1_val,y_train1_val,y_test1_val = train_test_split(train1,label1,test_size=0.3,random_state=27)
from sklearn.ensemble import RandomForestRegressor

regressor = RandomForestRegressor(n_estimators=350, max_depth=20, n_jobs=-1, random_state=60)

regressor.fit(x_train1_val, y_train1_val)



from sklearn.metrics import mean_squared_error

np.sqrt(mean_squared_error(y_test1_val,[round(s) for s in regressor.predict(x_test1_val)]))
from sklearn.ensemble import RandomForestRegressor

regressor = RandomForestRegressor(n_estimators=350, max_depth=20, n_jobs=-1, random_state=60)

regressor.fit(train1, label1)

y_pred1 = regressor.predict(test1)



ids = np.array(id1)

dict = {'id': ids, 'label': y_pred1}

submission_dataFrame = pd.DataFrame(dict)

submission_dataFrame.to_csv('mfinal1.csv', header=True, index=False)
train2 = train.loc[train['a2']==1]

train2 = train2.drop(columns = ['a0','a1','a3','a4','a5','a6'])

label2 = train2['label']

train2 = train2.drop(columns = ['label'])



corr_matrix = train2.corr().abs()

pd.options.display.max_rows = None



# to_drop = []

# temp = corr_matrix["label"]

# for i in temp.index:

#     if(pd.isna(temp[i])):

#         to_drop.append(i)

# to_drop

# train2 = train2.drop(train2[to_drop], axis=1)



test2 = test.loc[test['a2']==1]

test2 = test2.drop(columns = ['a0','a1','a3','a4','a5','a6'])

id2 = test2['id']

# test2 = test2.drop(test2[to_drop], axis=1)



corr = train2.corr().abs()

columns = np.full((corr.shape[0],), True, dtype=bool)

for i in range(corr.shape[0]):

    for j in range(i+1, corr.shape[0]):

        if (corr.iloc[i,j] >= 0.99):

            if columns[j]:

                columns[j] = False

selected_columns = train2.columns[columns]

train2 = train2[selected_columns]



test2=test2[selected_columns]
from sklearn import preprocessing

scaler = preprocessing.StandardScaler()

scaled_data = scaler.fit_transform(train2)

train2 = pd.DataFrame(scaled_data,columns=train2.columns)

train2.head()



scaled_data = scaler.fit_transform(test2)

test2 = pd.DataFrame(scaled_data,columns=test2.columns)

test2.head()
x_train2_val,x_test2_val,y_train2_val,y_test2_val = train_test_split(train2,label2,test_size=0.3,random_state=27)
from sklearn.ensemble import RandomForestRegressor

regressor = RandomForestRegressor(n_estimators=350, max_depth=20, n_jobs=-1, random_state=60)

regressor.fit(x_train2_val, y_train2_val)



from sklearn.metrics import mean_squared_error

np.sqrt(mean_squared_error(y_test2_val,[round(s) for s in regressor.predict(x_test2_val)]))
from sklearn.ensemble import RandomForestRegressor

regressor = RandomForestRegressor(n_estimators=350, max_depth=20, n_jobs=-1, random_state=60)

regressor.fit(train2, label2)

y_pred2 = regressor.predict(test2)



ids = np.array(id2)

dict = {'id': ids, 'label': y_pred2}

submission_dataFrame = pd.DataFrame(dict)

submission_dataFrame.to_csv('mfinal2.csv', header=True, index=False)
train3 = train.loc[train['a3']==1]

train3 = train3.drop(columns = ['a0','a1','a2','a4','a5','a6'])

label3 = train3['label']

train3 = train3.drop(columns = ['label'])



corr_matrix = train3.corr().abs()

pd.options.display.max_rows = None



# to_drop = []

# temp = corr_matrix["label"]

# for i in temp.index:

#     if(pd.isna(temp[i])):

#         to_drop.append(i)

# to_drop

# train3 = train3.drop(train3[to_drop], axis=1)



test3 = test.loc[test['a3']==1]

test3 = test3.drop(columns = ['a0','a1','a2','a4','a5','a6'])

id3 = test3['id']

# test3 = test3.drop(test3[to_drop], axis=1)



corr = train3.corr().abs()

columns = np.full((corr.shape[0],), True, dtype=bool)

for i in range(corr.shape[0]):

    for j in range(i+1, corr.shape[0]):

        if (corr.iloc[i,j] >= 0.99):

            if columns[j]:

                columns[j] = False

selected_columns = train3.columns[columns]

train3 = train3[selected_columns]



test3=test3[selected_columns]
from sklearn import preprocessing

scaler = preprocessing.StandardScaler()

scaled_data = scaler.fit_transform(train3)

train3 = pd.DataFrame(scaled_data,columns=train3.columns)

train3.head()



scaled_data = scaler.fit_transform(test3)

test3 = pd.DataFrame(scaled_data,columns=test3.columns)

test3.head()
x_train3_val,x_test3_val,y_train3_val,y_test3_val = train_test_split(train3,label3,test_size=0.3,random_state=27)
from sklearn.ensemble import RandomForestRegressor

regressor = RandomForestRegressor(n_estimators=350, max_depth=20, n_jobs=-1, random_state=60)

regressor.fit(x_train3_val, y_train3_val)



from sklearn.metrics import mean_squared_error

np.sqrt(mean_squared_error(y_test3_val,[round(s) for s in regressor.predict(x_test3_val)]))
from sklearn.ensemble import RandomForestRegressor

regressor = RandomForestRegressor(n_estimators=350, max_depth=20, n_jobs=-1, random_state=60)

regressor.fit(train3, label3)

y_pred3 = regressor.predict(test3)



ids = np.array(id3)

dict = {'id': ids, 'label': y_pred3}

submission_dataFrame = pd.DataFrame(dict)

submission_dataFrame.to_csv('mfinal3.csv', header=True, index=False)
train4 = train.loc[train['a4']==1]

train4 = train4.drop(columns = ['a0','a1','a2','a3','a5','a6'])

label4 = train4['label']

train4 = train4.drop(columns = ['label'])



corr_matrix = train4.corr().abs()

pd.options.display.max_rows = None



# to_drop = []

# temp = corr_matrix["label"]

# for i in temp.index:

#     if(pd.isna(temp[i])):

#         to_drop.append(i)

# to_drop

# train4 = train4.drop(train4[to_drop], axis=1)



test4 = test.loc[test['a4']==1]

test4 = test4.drop(columns = ['a0','a1','a2','a3','a5','a6'])

id4 = test4['id']

# test4 = test4.drop(test4[to_drop], axis=1)



corr = train4.corr().abs()

columns = np.full((corr.shape[0],), True, dtype=bool)

for i in range(corr.shape[0]):

    for j in range(i+1, corr.shape[0]):

        if (corr.iloc[i,j] >= 0.99):

            if columns[j]:

                columns[j] = False

selected_columns = train4.columns[columns]

train4 = train4[selected_columns]



test4 = test4[selected_columns]
from sklearn import preprocessing

scaler = preprocessing.StandardScaler()

scaled_data = scaler.fit_transform(train4)

train4 = pd.DataFrame(scaled_data,columns=train4.columns)

train4.head()



scaled_data = scaler.fit_transform(test4)

test4 = pd.DataFrame(scaled_data,columns=test4.columns)

test4.head()
x_train4_val,x_test4_val,y_train4_val,y_test4_val = train_test_split(train4,label4,test_size=0.3,random_state=27)
from sklearn.ensemble import RandomForestRegressor

regressor = RandomForestRegressor(n_estimators=350, max_depth=20, n_jobs=-1, random_state=60)

regressor.fit(x_train4_val, y_train4_val)



from sklearn.metrics import mean_squared_error

np.sqrt(mean_squared_error(y_test4_val,[round(s) for s in regressor.predict(x_test4_val)]))
from sklearn.ensemble import RandomForestRegressor

regressor = RandomForestRegressor(n_estimators=350, max_depth=20, n_jobs=-1, random_state=60)

regressor.fit(train4, label4)

y_pred4 = regressor.predict(test4)



ids = np.array(id4)

dict = {'id': ids, 'label': y_pred4}

submission_dataFrame = pd.DataFrame(dict)

submission_dataFrame.to_csv('mfinal4.csv', header=True, index=False)
train5 = train.loc[train['a5']==1]

train5 = train5.drop(columns = ['a0','a1','a2','a3','a4','a6'])

label5 = train5['label']

train5 = train5.drop(columns = ['label'])



corr_matrix = train5.corr().abs()

pd.options.display.max_rows = None



# to_drop = []

# temp = corr_matrix["label"]

# for i in temp.index:

#     if(pd.isna(temp[i])):

#         to_drop.append(i)

# to_drop

# train5 = train5.drop(train5[to_drop], axis=1)



test5 = test.loc[test['a5']==1]

test5 = test5.drop(columns = ['a0','a1','a2','a3','a4','a6'])

id5 = test5['id']

# test5 = test5.drop(test5[to_drop], axis=1)



corr = train5.corr().abs()

columns = np.full((corr.shape[0],), True, dtype=bool)

for i in range(corr.shape[0]):

    for j in range(i+1, corr.shape[0]):

        if (corr.iloc[i,j] >= 0.99):

            if columns[j]:

                columns[j] = False

selected_columns = train5.columns[columns]

train5 = train5[selected_columns]



test5 = test5[selected_columns]
from sklearn import preprocessing

scaler = preprocessing.StandardScaler()

scaled_data = scaler.fit_transform(train5)

train5 = pd.DataFrame(scaled_data,columns=train5.columns)

train5.head()



scaled_data = scaler.fit_transform(test5)

test5 = pd.DataFrame(scaled_data,columns=test5.columns)

test5.head()
x_train5_val,x_test5_val,y_train5_val,y_test5_val = train_test_split(train5,label5,test_size=0.3,random_state=27)
from sklearn.ensemble import RandomForestRegressor

regressor = RandomForestRegressor(n_estimators=350, max_depth=20, n_jobs=-1, random_state=60)

regressor.fit(x_train5_val, y_train5_val)



from sklearn.metrics import mean_squared_error

np.sqrt(mean_squared_error(y_test5_val,[round(s) for s in regressor.predict(x_test5_val)]))
from sklearn.ensemble import RandomForestRegressor

regressor = RandomForestRegressor(n_estimators=350, max_depth=20, n_jobs=-1, random_state=60)

regressor.fit(train5, label5)

y_pred5 = regressor.predict(test5)



ids = np.array(id5)

dict = {'id': ids, 'label': y_pred5}

submission_dataFrame = pd.DataFrame(dict)

submission_dataFrame.to_csv('mfinal5.csv', header=True, index=False)
train6 = train.loc[train['a6']==1]

train6 = train6.drop(columns = ['a0','a1','a2','a3','a4','a5'])

label6 = train6['label']

train6 = train6.drop(columns = ['label'])



corr_matrix = train6.corr().abs()

pd.options.display.max_rows = None



# to_drop = []

# temp = corr_matrix["label"]

# for i in temp.index:

#     if(pd.isna(temp[i])):

#         to_drop.append(i)

# to_drop

# train6 = train6.drop(train6[to_drop], axis=1)



test6 = test.loc[test['a6']==1]

test6 = test6.drop(columns = ['a0','a1','a2','a3','a4','a5'])

id6 = test6['id']

# test6 = test6.drop(test6[to_drop], axis=1)



corr = train6.corr().abs()

columns = np.full((corr.shape[0],), True, dtype=bool)

for i in range(corr.shape[0]):

    for j in range(i+1, corr.shape[0]):

        if (corr.iloc[i,j] >= 0.99):

            if columns[j]:

                columns[j] = False

selected_columns = train6.columns[columns]

train6 = train6[selected_columns]



test6 = test6[selected_columns]
from sklearn import preprocessing

scaler = preprocessing.StandardScaler()

scaled_data = scaler.fit_transform(train6)

train6 = pd.DataFrame(scaled_data,columns=train6.columns)

train6.head()



scaled_data = scaler.fit_transform(test6)

test6 = pd.DataFrame(scaled_data,columns=test6.columns)

test6.head()
x_train6_val,x_test6_val,y_train6_val,y_test6_val = train_test_split(train6,label6,test_size=0.3,random_state=27)
from sklearn.ensemble import RandomForestRegressor

regressor = RandomForestRegressor(n_estimators=350, max_depth=20, n_jobs=-1, random_state=60)

regressor.fit(x_train6_val, y_train6_val)



from sklearn.metrics import mean_squared_error

np.sqrt(mean_squared_error(y_test6_val,[round(s) for s in regressor.predict(x_test6_val)]))
from sklearn.ensemble import RandomForestRegressor

regressor = RandomForestRegressor(n_estimators=350, max_depth=20, n_jobs=-1, random_state=60)

regressor.fit(train6, label6)

y_pred6 = regressor.predict(test6)



ids = np.array(id6)

dict = {'id': ids, 'label': y_pred6}

submission_dataFrame = pd.DataFrame(dict)

submission_dataFrame.to_csv('mfinal6.csv', header=True, index=False)
import csv

all_filenames =['mfinal0.csv','mfinal1.csv','mfinal2.csv','mfinal3.csv','mfinal4.csv','mfinal5.csv','mfinal6.csv']

combined_csv = pd.concat([pd.read_csv(f) for f in all_filenames ])

combined_csv.to_csv( "output.csv", index=False)



df = pd.read_csv('output.csv', sep=',')



df = df.sort_values(by ='id' )

df.to_csv( "final.csv", index=False)