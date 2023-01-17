import pandas as pd

import numpy as np

from sklearn.preprocessing import OneHotEncoder
train = pd.read_csv('../input/train.csv', na_values=-1)

test = pd.read_csv('../input/test.csv', na_values=-1)
cat = [ i for i in train.columns if "cat" in i ]

train[cat] = train[cat].apply( lambda d: d.fillna(d.max()+1), axis=0 )

test[cat] = test[cat].apply( lambda d: d.fillna(d.max()+1), axis=0 )



train.drop(['ps_car_11_cat','id'] ,axis=1,inplace=True)

test.drop(['ps_car_11_cat','id'] ,axis=1,inplace=True)
encoder = OneHotEncoder(sparse=False)

train_arr = np.empty([train.shape[0],0])



for feature in train.columns :

    if "cat" in feature :

        print ("categorical column :\t", feature, " - ", train[feature].value_counts().shape[0])

        t = encoder.fit_transform(train[feature].values.reshape(-1,1))

        train_arr = np.append(train_arr, t, axis=1)

    else :

        train_arr = np.append(train_arr, train[feature].values.reshape(-1,1), axis=1)

        

        

print ("\n\nnew shape : ", train_arr.shape)

# the column [:,0] is for labels.
encoder = OneHotEncoder(sparse=False)

test_arr = np.empty([test.shape[0],0])





for feature in test.columns :

    if "cat" in feature :

        print ("categorical column :\t", feature, " - ", test[feature].value_counts().shape[0])

        t = encoder.fit_transform(test[feature].values.reshape(-1,1))

        test_arr = np.append(test_arr, t, axis=1)

    else :

        test_arr = np.append(test_arr, test[feature].values.reshape(-1,1), axis=1)

        

        

print ("\n\nnew shape : ", test_arr.shape)
# save to disk

#np.save('../output/train.npy',train_arr)

#np.save('../output/test.npy',test_arr)
# save samples to disk

#np.save('../nn/data/train_sample.npy',train_arr[100000,:])

#np.save('../nn/data/test_sample.npy', test_arr[100000,:])