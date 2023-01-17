import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns



import sklearn as skl

import math



%matplotlib inline



np.random.seed(0)


train = pd.read_csv('../input/bits-f464-l1/train.csv', sep=',')



train = train.drop(["id"], axis = 1)

test = pd.read_csv('../input/bits-f464-l1/test.csv', sep=',')







train.head()
train.info()
test.info()
train0=train[train['a0']==1]

train0 = train0.drop(["a1","a2","a3","a4","a5","a6"],axis=1)

test0=test[test['a0']==1]

test0 = test0.drop(["a1","a2","a3","a4","a5","a6"],axis=1)

id_test0 = test0['id']





train1=train[train['a1']==1]

train1 = train1.drop(["a0","a2","a3","a4","a5","a6"],axis=1)

test1=test[test['a1']==1]

test1 = test1.drop(["a0","a2","a3","a4","a5","a6"],axis=1)

id_test1 = test1['id']





train2=train[train['a2']==1]

train2 = train2.drop(["a1","a0","a3","a4","a5","a6"],axis=1)

test2=test[test['a2']==1]

test2 = test2.drop(["a1","a0","a3","a4","a5","a6"],axis=1)

id_test2 = test2['id']





train3=train[train['a3']==1]

train3 = train3.drop(["a1","a2","a0","a4","a5","a6"],axis=1)

test3=test[test['a3']==1]

test3 = test3.drop(["a1","a2","a0","a4","a5","a6"],axis=1)

id_test3 = test3['id']



                 

train4=train[train['a4']==1]

train4 = train4.drop(["a1","a2","a3","a0","a5","a6"],axis=1)

test4=test[test['a4']==1]

test4 = test4.drop(["a1","a2","a3","a0","a5","a6"],axis=1)

id_test4 = test4['id']



                 

train5=train[train['a5']==1]

train5 = train5.drop(["a1","a2","a3","a4","a0","a6"],axis=1)

test5=test[test['a5']==1]

test5 = test5.drop(["a1","a2","a3","a4","a0","a6"],axis=1)

id_test5 = test5['id']



                 

train6=train[train['a6']==1]

train6 = train6.drop(["a1","a2","a3","a4","a5","a0"],axis=1)

test6=test[test['a6']==1]

test6 = test6.drop(["a1","a2","a3","a4","a5","a0"],axis=1)

id_test6 = test6['id']





test = test.drop(["id"], axis = 1)
rating = train0['label']

train0 = train0.drop(columns = ['label'])





corr = train0.corr()

columns = np.full((corr.shape[0],), True, dtype=bool)

for i in range(corr.shape[0]):

    for j in range(i+1, corr.shape[0]):

        if (corr.iloc[i,j] >= 0.99 or corr.iloc[i,j] <= -0.99):

            if columns[j]:

                columns[j] = False

selected_columns = train0.columns[columns]

train0 = train0[selected_columns]



test0=test0[selected_columns]

from sklearn.model_selection import train_test_split



x_train_val,x_test_val,y_train_val,y_test_val = train_test_split(train0,rating,test_size=0.3,random_state=27)
from sklearn.ensemble import RandomForestRegressor

regressor = RandomForestRegressor(n_estimators=350, max_depth=20, n_jobs=-1, random_state=60)

regressor.fit(x_train_val, y_train_val)



from sklearn.metrics import mean_squared_error

np.sqrt(mean_squared_error(y_test_val,[round(s) for s in regressor.predict(x_test_val)]))
from sklearn.ensemble import RandomForestRegressor

regressor = RandomForestRegressor(n_estimators=350, max_depth=20, n_jobs=-1, random_state=60)

regressor.fit(train0, rating)

y_pred = regressor.predict(test0)
ids = np.array(id_test0)

dict = {'id': ids, 'label': y_pred}

submission_dataFrame = pd.DataFrame(dict)

submission_dataFrame.to_csv('final0.csv', header=True, index=False)
rating = train1['label']

train1 = train1.drop(columns = ['label'])



corr = train1.corr()



columns = np.full((corr.shape[0],), True, dtype=bool)

for i in range(corr.shape[0]):

    for j in range(i+1, corr.shape[0]):

        if corr.iloc[i,j] >= 0.99:

            if columns[j]:

                columns[j] = False

selected_columns = train1.columns[columns]

train1 = train1[selected_columns]



test1=test1[selected_columns]
from sklearn.model_selection import train_test_split



x_train_val,x_test_val,y_train_val,y_test_val = train_test_split(train1,rating,test_size=0.3,random_state=27)
from sklearn.ensemble import RandomForestRegressor

regressor = RandomForestRegressor(n_estimators=350, max_depth=20, n_jobs=-1, random_state=60)

regressor.fit(x_train_val, y_train_val)



from sklearn.metrics import mean_squared_error

np.sqrt(mean_squared_error(y_test_val,[round(s) for s in regressor.predict(x_test_val)]))
from sklearn.ensemble import RandomForestRegressor

regressor = RandomForestRegressor(n_estimators=350, max_depth=20, n_jobs=-1, random_state=60)

regressor.fit(train1, rating)

y_pred = regressor.predict(test1)
ids = np.array(id_test1)

dict = {'id': ids, 'label': y_pred}

submission_dataFrame = pd.DataFrame(dict)

submission_dataFrame.to_csv('final1.csv', header=True, index=False)
rating = train2['label']

train2 = train2.drop(columns = ['label'])



corr = train2.corr()



columns = np.full((corr.shape[0],), True, dtype=bool)

for i in range(corr.shape[0]):

    for j in range(i+1, corr.shape[0]):

        if corr.iloc[i,j] >= 0.99:

            if columns[j]:

                columns[j] = False

selected_columns = train2.columns[columns]

train2 = train2[selected_columns]

test2=test2[selected_columns]
from sklearn.model_selection import train_test_split



x_train_val,x_test_val,y_train_val,y_test_val = train_test_split(train2,rating,test_size=0.3,random_state=27)
from sklearn.ensemble import RandomForestRegressor

regressor = RandomForestRegressor(n_estimators=350, max_depth=20, n_jobs=-1, random_state=60)

regressor.fit(x_train_val, y_train_val)



from sklearn.metrics import mean_squared_error

np.sqrt(mean_squared_error(y_test_val,[round(s) for s in regressor.predict(x_test_val)]))
from sklearn.ensemble import RandomForestRegressor

regressor = RandomForestRegressor(n_estimators=350, max_depth=20, n_jobs=-1, random_state=60)

regressor.fit(train2, rating)

y_pred = regressor.predict(test2)
ids = np.array(id_test2)

dict = {'id': ids, 'label': y_pred}

submission_dataFrame = pd.DataFrame(dict)

submission_dataFrame.to_csv('final2.csv', header=True, index=False)
rating = train3['label']

train3 = train3.drop(columns = ['label'])



corr = train3.corr()



columns = np.full((corr.shape[0],), True, dtype=bool)

for i in range(corr.shape[0]):

    for j in range(i+1, corr.shape[0]):

        if corr.iloc[i,j] >= 0.99:

            if columns[j]:

                columns[j] = False

selected_columns = train3.columns[columns]

train3 = train3[selected_columns]

test3=test3[selected_columns]
from sklearn.model_selection import train_test_split



x_train_val,x_test_val,y_train_val,y_test_val = train_test_split(train3,rating,test_size=0.3,random_state=27)
from sklearn.ensemble import RandomForestRegressor

regressor = RandomForestRegressor(n_estimators=350, max_depth=20, n_jobs=-1, random_state=60)

regressor.fit(x_train_val, y_train_val)



from sklearn.metrics import mean_squared_error

np.sqrt(mean_squared_error(y_test_val,[round(s) for s in regressor.predict(x_test_val)]))
from sklearn.ensemble import RandomForestRegressor

regressor = RandomForestRegressor(n_estimators=350, max_depth=20, n_jobs=-1, random_state=60)

regressor.fit(train3, rating)

y_pred = regressor.predict(test3)
ids = np.array(id_test3)

dict = {'id': ids, 'label': y_pred}

submission_dataFrame = pd.DataFrame(dict)

submission_dataFrame.to_csv('final3.csv', header=True, index=False)
rating = train4['label']

train4 = train4.drop(columns = ['label'])



corr = train4.corr()



columns = np.full((corr.shape[0],), True, dtype=bool)

for i in range(corr.shape[0]):

    for j in range(i+1, corr.shape[0]):

        if corr.iloc[i,j] >= 0.99:

            if columns[j]:

                columns[j] = False

selected_columns = train4.columns[columns]

train4 = train4[selected_columns]

test4=test4[selected_columns]
from sklearn.model_selection import train_test_split



x_train_val,x_test_val,y_train_val,y_test_val = train_test_split(train4,rating,test_size=0.3,random_state=27)
from sklearn.ensemble import RandomForestRegressor

regressor = RandomForestRegressor(n_estimators=350, max_depth=20, n_jobs=-1, random_state=60)

regressor.fit(x_train_val, y_train_val)



from sklearn.metrics import mean_squared_error

np.sqrt(mean_squared_error(y_test_val,[round(s) for s in regressor.predict(x_test_val)]))
from sklearn.ensemble import RandomForestRegressor

regressor = RandomForestRegressor(n_estimators=350, max_depth=20, n_jobs=-1, random_state=60)

regressor.fit(train4, rating)

y_pred = regressor.predict(test4)
ids = np.array(id_test4)

dict = {'id': ids, 'label': y_pred}

submission_dataFrame = pd.DataFrame(dict)

submission_dataFrame.to_csv('final4.csv', header=True, index=False)
rating = train5['label']

train5 = train5.drop(columns = ['label'])



corr = train5.corr()



columns = np.full((corr.shape[0],), True, dtype=bool)

for i in range(corr.shape[0]):

    for j in range(i+1, corr.shape[0]):

        if corr.iloc[i,j] >= 0.99:

            if columns[j]:

                columns[j] = False

selected_columns = train5.columns[columns]

train5 = train5[selected_columns]

test5=test5[selected_columns]
from sklearn.model_selection import train_test_split



x_train_val,x_test_val,y_train_val,y_test_val = train_test_split(train5,rating,test_size=0.3,random_state=27)
from sklearn.ensemble import RandomForestRegressor

regressor = RandomForestRegressor(n_estimators=350, max_depth=20, n_jobs=-1, random_state=60)

regressor.fit(x_train_val, y_train_val)



from sklearn.metrics import mean_squared_error

np.sqrt(mean_squared_error(y_test_val,[round(s) for s in regressor.predict(x_test_val)]))
from sklearn.ensemble import RandomForestRegressor

regressor = RandomForestRegressor(n_estimators=350, max_depth=20, n_jobs=-1, random_state=60)

regressor.fit(train5, rating)

y_pred = regressor.predict(test5)
ids = np.array(id_test5)

dict = {'id': ids, 'label': y_pred}

submission_dataFrame = pd.DataFrame(dict)

submission_dataFrame.to_csv('final5.csv', header=True, index=False)
rating = train6['label']

train6 = train6.drop(columns = ['label'])



corr = train6.corr()



columns = np.full((corr.shape[0],), True, dtype=bool)

for i in range(corr.shape[0]):

    for j in range(i+1, corr.shape[0]):

        if corr.iloc[i,j] >= 0.99:

            if columns[j]:

                columns[j] = False

selected_columns = train6.columns[columns]

train6 = train6[selected_columns]

test6=test6[selected_columns]
from sklearn.model_selection import train_test_split



x_train_val,x_test_val,y_train_val,y_test_val = train_test_split(train6,rating,test_size=0.3,random_state=27)
from sklearn.ensemble import RandomForestRegressor

regressor = RandomForestRegressor(n_estimators=350, max_depth=20, n_jobs=-1, random_state=60)

regressor.fit(x_train_val, y_train_val)



from sklearn.metrics import mean_squared_error

np.sqrt(mean_squared_error(y_test_val,[round(s) for s in regressor.predict(x_test_val)]))
from sklearn.ensemble import RandomForestRegressor

regressor = RandomForestRegressor(n_estimators=350, max_depth=20, n_jobs=-1, random_state=60)

regressor.fit(train6, rating)

y_pred = regressor.predict(test6)
ids = np.array(id_test6)

dict = {'id': ids, 'label': y_pred}

submission_dataFrame = pd.DataFrame(dict)

submission_dataFrame.to_csv('final6.csv', header=True, index=False)
import csv

all_filenames =['final0.csv','final1.csv','final2.csv','final3.csv','final4.csv','final5.csv','final6.csv']

combined_csv = pd.concat([pd.read_csv(f) for f in all_filenames ])

combined_csv.to_csv( "output.csv", index=False)
df = pd.read_csv('output.csv', sep=',')
df.info()
df = df.sort_values(by ='id' )

df.to_csv( "final.csv", index=False)
