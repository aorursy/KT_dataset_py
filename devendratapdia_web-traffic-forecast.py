# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input/web-traffic-time-series-forecasting'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



trainfile2 = os.path.join(r'/kaggle/input/web-traffic-time-series-forecasting', 'train_2.csv.zip')

keyfile2 = os.path.join(r'/kaggle/input/web-traffic-time-series-forecasting', 'key_2.csv.zip')

submissionfile2 = os.path.join(r'/kaggle/input/web-traffic-time-series-forecasting', 'sample_submission_2.csv.zip')



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import numpy as np

import pandas as pd

from matplotlib import pyplot as plt

import seaborn as sns



train_1 = pd.read_csv(trainfile2, compression='zip')

key_1 = pd.read_csv(keyfile2, compression='zip')

submission_2 = pd.read_csv(submissionfile2, compression='zip')





key_1.shape, train_1.shape





key_1.head()



train_1.head()



# key_1.loc[0, 'Page']

# key_1.loc[1, 'Page']

# key_1.loc[8703779:, 'Page'].values



key_1['date'] = key_1.Page.apply(lambda x: x[-10:])

key_1['Page'] = key_1.loc[:, 'Page'].apply(lambda x: x[:-11])





def chunks(x, size):

    for pos in range(0, len(x), size):

        yield x[pos:pos + size]





import hashlib

key_1['Page'] = key_1.Page.apply(lambda x: hashlib.md5(x.encode('utf-8')).hexdigest())





train_1.shape



train_1.head()





import hashlib

train_1['Page'] = train_1.Page.apply(lambda x: hashlib.md5(x.encode('utf-8')).hexdigest())



train_1.head()





# Replace NA values with 0

train_1.fillna(0, inplace=True)





train_1.head()





# # Problem Statement:

# 

# Stage 1:

# 

#     Training Set: 1 July 2015 - 31 Dec 2016 : 

#     

#     2015-07-01    2016-12-31

# 

#     Predict: 1 Jan 2017 - 1 Mar 2017 : 

#     

#     2017-01-01    2017-03-01

# 

# Stage 2:

# 

#     Test Set: 1 July 2015 - 1 Sept 2017 : 

#     

#     2015-07-01    2017-09-01

# 

#     Predict: 13 Sept 2017 and 13 Nov 2017: 

#     

#     2017-09-13    2017-11-13



# In[ ]:



train_page_1 = train_1.Page





train_page_1 = pd.DataFrame(train_page_1)



train_1.drop('Page', axis=1, inplace=True)



train_1.head()





# from sklearn.preprocessing import MinMaxScaler

# sc = MinMaxScaler()

# #vals = np.array(train_1.iloc[:, :]).reshape(-1, 1)

# train_n1 = sc.fit_transform(train_1)



# train_n1.shape



# train_n1[:, -61:].shape
# Using Dataframe train dataset

y_train_1 = train_1.iloc[:, -64:]

X_train_1 = train_1.iloc[:, :-64]



# y_train_1 = train_n1[:, -61:]

# X_train_1 = train_n1[:, :-61]





from keras.models import Sequential

from keras.layers import Dense, Dropout

from keras.wrappers.scikit_learn import KerasRegressor

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import KFold

from sklearn.preprocessing import StandardScaler

from sklearn.pipeline import Pipeline





from sklearn.model_selection import KFold

# 

# define 10-fold cross validation test harness



# Create the model using the NestedLSTM class - two layers are a good starting point

# Feel free to play around with the number of nodes & other model parameters

model = Sequential()

model.add(Dense(256, input_dim=X_train_1.shape[1], init='normal', activation='relu'))

#model.add(Dense(512, init='normal', activation='relu'))

model.add(Dense(256, init='normal', activation='relu'))

model.add(Dense(128, init='normal', activation='relu'))

model.add(Dense(64, init='normal', activation='relu'))

model.add(Dense(32, init='normal', activation='relu'))

model.add(Dense(64, init='normal'))

model.compile(loss='mean_squared_error', optimizer = 'adam', metrics = ['accuracy'])



# It's training time!

BATCH = 2000



print('Training time, it is...')

model.fit(X_train_1, y_train_1,

          batch_size=BATCH,

          epochs=50,

          validation_split=0.2,

          shuffle=True

         )
# Using Dataframe test dataset

X_test_1 = train_1.iloc[:, 64:]



# #y_train_1 = train_n1[:, -61:]

# X_test_1 = train_n1[:, 61:]



y_test_1 = model.predict(X_test_1)



X_test_1.shape, y_test_1.shape





dti = pd.DataFrame(pd.date_range(start='2017-09-11', periods=64, freq='D'), columns=['d'])



test1 = np.concatenate((X_test_1, y_test_1), axis=1)



test1.shape



test1.shape, train_1.shape



# y_pred1 = sc.inverse_transform(test1)

# y_pred1 = y_pred1.iloc[:, -61:]



# Using Dataframe

y_pred1 = pd.DataFrame(test1)

y_pred1 = y_pred1.iloc[:, -64:]





y_pred1.columns = dti.d.values



y_pred1.head()



#y_pred1 = y_pred1.iloc[:, :-1]



y_pred1 = y_pred1.clip(lower=0)



y_pred1.head()



train_1.head()



#pd.concat([train_1, y_pred1], axis=1)

ypred1_cols = y_pred1.columns
final_train1 = pd.concat([train_1, y_pred1], axis=1)



final_train1.columns = np.append(final_train1.columns[:-64].values , np.hstack(pd.DataFrame(final_train1.columns[-64:].values).apply(lambda x: x.dt.strftime('%Y-%m-%d')).values))



final_train1.head()



train_file_2 = pd.read_csv(trainfile2, compression='zip')



final_train1['Page'] = train_file_2.Page



final_train1.head()



#final_train1.to_csv('final_train2.csv')



#submission_2 = pd.read_csv('sample_submission_2.csv', index_col=0)

#final_train1 = pd.read_csv('final_train2.csv', index_col=0)



final_train1 = final_train1.melt(id_vars='Page', var_name='date', value_name='val')



#key_1 = pd.read_csv('key_2_new.csv', index_col=0)



key_1.head()



final_train1.shape



final_train1.head()





import hashlib

# final_train1.loc[80000000:88633493, 'Page'] = final_train1.loc[80000000:, 'Page'].apply(lambda x: hashlib.md5(x.encode('utf-8')).hexdigest())



def chunks(x, size):

    for pos in range(0, len(x), size):

        yield x[pos:pos + size]



final_train1.tail()



final_train1 = final_train1.iloc[-8993906:, :]



final_train1.shape



size = 1000000

import hashlib

#for pos in range(0, len(final_train1), size):

final_train1.loc[:, 'Page'] = final_train1.loc[:, 'Page'].apply(lambda x: hashlib.md5(x.encode('utf-8')).hexdigest())





final_train1.tail()



key_1.head()



final_train1.head()





# i = 1 

# for chunk in chunks(final_train1, 200000):

#     chunk.to_csv('final_train1_chunk_' + str(i) + '.csv')

#     i = i + 1



key_1.shape



final_train_res1 = final_train1



final1 = pd.merge(key_1, final_train_res1, on=['Page', 'date'], how='left')



final1.drop(['Page', 'date'], axis=1, inplace=True)



final1.head()



final1.isna().sum()



final1.fillna(0, inplace=True)



final1.shape, submission_2.shape





final1.columns = ['Id', 'Visits']

final1.to_csv('submission.csv', index=False)





final1.tail()





submission_2.shape, final1.shape



submission_2.tail()



final1.iloc[8993904:, :]