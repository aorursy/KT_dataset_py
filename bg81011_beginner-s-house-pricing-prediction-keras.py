#import things

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline



from subprocess import check_output



# Any results you write to the current directory are saved as output.
df = pd.read_csv('../input/train.csv')



print(df.dtypes.value_counts())

print(df.columns.tolist())

print('length of columns :', len(df.columns.tolist()))

print(df.count())
fig = plt.figure(figsize=(100,100))



data_corr = df.corr()

sns.heatmap(df.corr(), annot=True, cmap='summer_r')
fig = plt.figure(figsize=(10,10))



def reduce_dataframe(dataframe):

    corr = dataframe.corr()

    # filter to get the elements that are higher than 0.5  in repect of saleprice

    #['OverallQual', 'YearBuilt', 'YearRemodAdd', 'TotalBsmtSF', '1stFlrSF',

    #   'GrLivArea', 'FullBath', 'TotRmsAbvGrd', 'GarageCars', 'GarageArea',

    #   'SalePrice']

    filterd_corr = corr.loc[corr['SalePrice'] > 0.5, 'SalePrice']

    

    reduced_df = pd.DataFrame()

    

    for s in filterd_corr.index.tolist():

        reduced_df[s] = df[s]

        

    return reduced_df



reduced_df = reduce_dataframe(df)



print(reduced_df.columns)

sns.heatmap(reduced_df.corr(), annot=True, cmap='summer_r')
df_object = df.select_dtypes(['object'])



#add sale prices column

df_object['SalePrice'] = df['SalePrice']



print(df_object.columns)



sns.violinplot('HouseStyle', 'SalePrice', data=df_object)



plt.show()
fig = plt.figure(figsize=(20,20))



melt_df_object = pd.melt(df_object, 

                        id_vars=['SalePrice'], #variables to keep

                        var_name='all') #new column name of melted variable



melt_df_object.head



sns.violinplot(x='all', y='SalePrice', data=melt_df_object)
print(df_object.columns)

print(len(df_object.columns))
fig = plt.figure(figsize=(20, 300))



for i,c in enumerate(df_object.columns):

    if c != 'SalePrice':

        ax = fig.add_subplot(22, 2, 1+i)

        sns.violinplot(x=df_object[c], y=df_object['SalePrice'], ax=ax)

    
def replace_nan_with_mean_value(reduced_data):



    print('before replace  nan :', reduced_data.count())

    

    for i,n in enumerate(reduced_data.count()):            

        

        str = reduced_data.count().index[i]

        

        if reduced_data.count().index[i] != 'SalePrice':

            reduced_data.loc[(reduced_data[str].isnull()), str] = reduced_data[str].mean()

            reduced_data[str] = reduced_data[str]/reduced_data[str].max()



    print('after replace nan :', reduced_data.count())



    return reduced_data



reduced_data = reduce_dataframe(df)

reduced_data = replace_nan_with_mean_value(reduced_data)

from collections import Counter

reduced_data.isnull().any()

print(reduced_data.head)

from keras.models import Sequential

from keras.layers.core import Dense, Dropout, Activation

from keras.callbacks import ModelCheckpoint

from keras.optimizers import SGD



reduced_data = reduce_dataframe(df)

reduced_data = replace_nan_with_mean_value(reduced_data)



reduced_data = reduced_data[['SalePrice'] + reduced_data.columns[:-1].tolist()]



X_train = reduced_data[reduced_data.columns[1:]]

Y_train = reduced_data[reduced_data.columns[:1]]



X_train = X_train.as_matrix()

Y_train = Y_train.as_matrix()

print(X_train.shape)

print(Y_train.shape)



checkpoint = ModelCheckpoint(filepath='my_model.h5', verbose=1, save_best_only=True)

#sgd = SGD(lr=0.01, clipnorm=1.)



model = Sequential()

model.add(Dense(10, input_shape=(10,)))

model.add(Dense(20, activation='relu'))

model.add(Dense(30, activation='relu'))

model.add(Dense(1))

model.summary()



model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])



hist = model.fit(X_train, Y_train, epochs=300, batch_size=10, verbose=0, callbacks=[checkpoint], validation_split=0.2)
fig = plt.figure(figsize=(20,10))



ax1 = fig.add_subplot(1, 2, 1)

ax2 = fig.add_subplot(1, 2, 2)



ax1.plot(hist.history['loss'], color='r')

ax1.plot(hist.history['val_loss'], color='b')



ax2.plot(hist.history['acc'], color='r')

ax2.plot(hist.history['val_acc'], color='b')



from keras.models import load_model



#model = load_model('my_model.h5')



test_df = pd.read_csv('../input/test.csv')

id_col = test_df['Id'].values.tolist()



test_df = test_df[['OverallQual', 'YearBuilt', 'YearRemodAdd', 

 'TotalBsmtSF', '1stFlrSF', 'GrLivArea', 'FullBath', 'TotRmsAbvGrd', 'GarageCars', 'GarageArea']]



test_df = replace_nan_with_mean_value(test_df)



print(test_df.isnull().any())



X_test = test_df.as_matrix()

pred = model.predict(X_test)



print(pred)



submission = pd.DataFrame()

submission['Id'] = id_col

submission['SalePrice'] = pred



print(submission.isnull().any())



submission.count()

submission.to_csv('output.csv', index=False)
