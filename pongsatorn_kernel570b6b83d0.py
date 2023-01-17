# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd

import copy as cp

# data processing, CSV file I/O (e.g. pd.read_csv)

# import keras as k

pd.set_option('display.max_rows', 500)

pd.set_option('display.max_columns', 500)

pd.set_option('display.width', 1000)

from sklearn.model_selection import train_test_split





from keras.callbacks import Callback, TensorBoard , ModelCheckpoint , EarlyStopping

from keras.layers import Dense, Dropout

from keras.models import Sequential , load_model , Model

from keras import optimizers ,regularizers

from keras.backend import clear_session

from keras.layers import LeakyReLU

from sklearn.metrics import r2_score

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from sklearn import preprocessing



from sklearn.ensemble import RandomForestClassifier
le = preprocessing.LabelEncoder()



#     train_df['Alley'] = le.transform(list(train_df['Alley']))

#     train_df['Alley'] = train_df['Alley'] +1

#     print(train_df.head())



def data_preparation():

    train_df = pd.read_csv("../input/train.csv")

    test_df = pd.read_csv("../input/test.csv")

    

    

    cat_data_missing_val = train_df.isnull().sum()/train_df.isnull().count()

    con1 = cat_data_missing_val.values <0.15

    passed_missing_value = list(cat_data_missing_val[con1].index)

    

    corr_columns = [] 

    total_cat_col = []

    total_num_col = []



    for col in train_df[passed_missing_value].columns:

        if train_df[col].dtype == 'int64':

            total_num_col.append(col)

            if train_df['SalePrice'].corr(train_df[col]) > 0.5:

                corr_columns.append(col)

        else: total_cat_col.append(col)

            

    train_df = train_df[corr_columns+total_cat_col]

    

#     print(train_df['SalePrice'].describe())

        

#     train_df = train_df.drop(train_df[(train_df['GrLivArea']>4000) | (train_df['SalePrice']>350000)].index)

#     train_df = train_df.drop(train_df[(train_df['SalePrice']>350000)].index)



    x_train = train_df.drop(columns=['SalePrice'])

    labels = train_df['SalePrice']

    

    x_test = test_df.drop(columns= ['Id'])

    id_test = test_df['Id']



    col_keys = x_train.columns

    x_train_tran = cp.copy(x_train)

    none_str = 'NONE'



    # print(x_train_tran['Neighborhood'].dtype)



    for key in col_keys:

        # if string columns

        if(x_train[key].dtype == np.object):



            # fill na with mode values

#             x_train[key] = x_train[key].fillna(x_train[key].mode()[0])



            x_train[key] = x_train[key].fillna('NONE')

            x_test[key] = x_test[key].fillna('NONE')

           

            

            fitt = list(x_train[key])+ list(x_test[key])

            le.fit(fitt)

            

            x_train_tran[key] = le.transform(list(x_train[key]))

            x_train_tran[key] = x_train_tran[key] + 1

            

            

            x_test[key] = le.transform(list(x_test[key]))

            x_test[key] = x_test[key] + 1

    

        else:

            # fill number with mean

            x_train_tran[key] = x_train[key].fillna(round(x_train[key].mean()))

            x_test[key] = x_test[key].fillna(0)





#     X_train, X_test, y_train, y_test = train_test_split(x_train_tran,labels, test_size=0.3)

    

    return  x_train_tran, labels , x_test, id_test

    

def DNN():

    nodes = 400

    

    model = Sequential()



    model.add(Dense(nodes, input_dim=50, kernel_initializer='normal',activation='relu'))

#     model.add(LeakyReLU(alpha=0.1))

    model.add(Dense(int(nodes/2), kernel_initializer='normal',activation='relu'))

#     model.add(LeakyReLU(alpha=0.1))

    model.add(Dense(int(nodes/4), kernel_initializer='normal',activation='relu'))



#     model.add(LeakyReLU(alpha=0.1))

    model.add(Dense(int(nodes/8), kernel_initializer='normal',activation='relu'))

#     model.add(LeakyReLU(alpha=0.1))

    model.add(Dense(int(nodes/10), kernel_initializer='normal',activation='relu'))

#     model.add(LeakyReLU(alpha=0.1))

    model.add(Dense(int(nodes/20), kernel_initializer='normal',activation='relu'))



#     model.add(LeakyReLU(alpha=0.1))

    model.add(Dense(1, activation= 'linear'))





    adam = optimizers.Adam(lr=0.001, amsgrad=True,decay=0.00001) #0.0003 , 0.0007

    #         'rmsprop'

    #         model.compile(loss='mae', optimizer=adam, metrics=['mape'])

    model.compile(loss='mean_absolute_error', optimizer=adam)



    model.summary()

    

    return model



def RandomF():

    clf = RandomForestClassifier(n_estimators=100, max_depth=2, random_state=0)

    

    return clf

# Any results you write to the current directory are saved as output.
X_train, y_train, x_test, id_test = data_preparation()

DNN_model = DNN()

monitor = EarlyStopping(monitor='val_loss', patience=200, verbose=1, mode='auto', restore_best_weights=True)

history = DNN_model.fit(X_train, y_train, batch_size= 100, epochs= 3000, verbose=2, validation_split= 0.3, callbacks= [monitor])

# loss = DNN_model.evaluate(X_test, y_test, batch_size= 50)

# results = DNN_model.predict(x_test, batch_size= 60 )

train_df = pd.read_csv("../input/train.csv")



cat_data_missing_val = train_df.isnull().sum()/train_df.isnull().count()

con1 = cat_data_missing_val.values <0.15

passed_missing_value = list(cat_data_missing_val[con1].index)



corr_columns = [] 

total_cat_col = []

total_num_col = []



for col in train_df[passed_missing_value].columns:

    if train_df[col].dtype == 'int64':

        total_num_col.append(col)

        if train_df['SalePrice'].corr(train_df[col]) > 0.5:

            corr_columns.append(col)

    else: total_cat_col.append(col)

        

corr_columns.remove('SalePrice')



results = DNN_model.predict(x_test[corr_columns+total_cat_col], batch_size= 60 )

results
import matplotlib.pyplot as plt



plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])

plt.title('model loss')

plt.ylabel('loss')

plt.xlabel('epochs')

plt.legend(['train', 'test'], loc='upper left')

plt.show()
list_prices = []

for each in list(results):

    list_prices.append(each[0])



result_df = pd.DataFrame({'Id': id_test, 'SalePrice': list_prices})



result_df.to_csv('submission.csv', index=False)
result_df.head()
for s in train_df['MSZoning'].isnull():

    if(s is True):

        print('found true')