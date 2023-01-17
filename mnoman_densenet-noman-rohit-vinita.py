# #colab

# # google-drive-ocamlfuseのインストール

# # https://github.com/astrada/google-drive-ocamlfuse

# !apt-get install -y -qq software-properties-common python-software-properties module-init-tools

# !add-apt-repository -y ppa:alessandro-strada/ppa 2>&1 > /dev/null

# !apt-get update -qq 2>&1 > /dev/null

# !apt-get -y install -qq google-drive-ocamlfuse fuse



# # Colab用のAuth token作成

# from google.colab import auth

# auth.authenticate_user()



# # Drive FUSE library用のcredential生成

# from oauth2client.client import GoogleCredentials

# creds = GoogleCredentials.get_application_default()

# import getpass

# !google-drive-ocamlfuse -headless -id={creds.client_id} -secret={creds.client_secret} < /dev/null 2>&1 | grep URL

# vcode = getpass.getpass()

# !echo {vcode} | google-drive-ocamlfuse -headless -id={creds.client_id} -secret={creds.client_secret}



# !mkdir -p drive

# !google-drive-ocamlfuse -o nonempty drive
# !pip install kaggle

# !echo '{"username":"mnoman","key":"3b91f4fccedd3705ab8235b65ab5f85f"}' > /root/.kaggle/kaggle.json

# !kaggle competitions download -c rossmann-store-sales
# !ls

# !unzip store.csv.zip

# !unzip train.csv.zip

# !unzip test.csv.zip
import pandas as pd

from pandas import DataFrame as df

import numpy as np
path= "../input/rossmann-store-sales/"
store_data=pd.read_csv(path+"store.csv")

train_data=pd.read_csv(path+"train.csv")

test_data=pd.read_csv(path+"test.csv")

store_data=store_data.fillna(-1)





print(store_data.keys(),"\n",train_data.keys())

months_dict={

     "-1":0,  "Jan":1,  "Feb":2,    "Mar":3,     "Apr":4,     "May":5,     "Jun":6,

     "Jul":7, "Aug":8,  "Sept":9,   "Oct":10,    "Nov":11,    "Dec":12

     }

characters_dict={

    "a":0, "b":1, "c":2, "d":3, "e":4, "f":5, "g":6, "h":7, "i":8, "j":9,"k":10,"l":11,"m":12,"n":13, 

    "o":14, "p":15, "q":16, "r":17, "s":18, "t":19, "u":20, "v":21, "w":22, "x":23, "y":24,"z":25,

}





def pre_process_store_data(store_data,months_dict,characters_dict):

    

    store_data_numpy = store_data.to_numpy()

    new_store_data_list=[]

    

    for idx,(Store, StoreType,Assortment, CompetitionDistance, \

        CompetitionOpenSinceMonth, CompetitionOpenSinceYear, Promo2, \

        Promo2SinceWeek , Promo2SinceYear, PromoInterval) in enumerate(store_data_numpy):

               

        #store type embedding

        StoreType = characters_dict[StoreType]



        #store type Assortment

        Assortment = characters_dict[Assortment]

        

        # store type PromoInterval

        sales_month_vector = np.zeros(len(months_dict.keys()))



        promo_months = str(PromoInterval).split(",")

        month_in_numbers = [months_dict[a] for a in promo_months]

        for i in month_in_numbers:

            sales_month_vector[i]=1;

        PromoInterval = sales_month_vector



        # make a new array with replaced embeddings

        temp = [Store, StoreType,Assortment, CompetitionDistance, \

        CompetitionOpenSinceMonth, CompetitionOpenSinceYear, Promo2, \

        Promo2SinceWeek , Promo2SinceYear] + list(PromoInterval)

        

        new_store_data_list.append(temp)

        # new_store_data_list=np.array(new_store_data_list)

    

    return new_store_data_list



def pre_process_train_data(train_data):

    train_data = train_data.to_numpy()

    state_holiday_dict={

        '0':0,

        'a':1,

        'b':2,

        'c':3

    }

    

    new_store_data_list=[]

    sales=[]

    for idx , (Store, DayOfWeek, Date, Sales, Customers, \

               _open, Promo,StateHoliday, SchoolHoliday) in enumerate(train_data):         

        

        #splitting date

        Date = Date.split("-")

        

        #processing StateHoliday

        StateHoliday = state_holiday_dict[str(StateHoliday)]

        

        if _open==0:

            # print(_open)

            continue

        #new_store_data_list.append([Store, DayOfWeek ,int(Date[0]),int(Date[1]),int(Date[2]),\

        #                            Sales, Customers, _open, Promo,StateHoliday, SchoolHoliday])

        new_store_data_list.append([Store, DayOfWeek ,int(Date[0]),int(Date[1]),int(Date[2]),\

                                    _open, Promo,StateHoliday, SchoolHoliday])

        

        sales.append(Sales)

    return new_store_data_list ,sales





def connect_store_data_with_train_data(preprocessed_train_data,preprocessed_store_data):

    

    preprocessed_dataset=[]

    for train_instance in preprocessed_train_data:



        store_id = train_instance[0]-1

        store_instance = preprocessed_store_data[store_id]



        train_instance = store_instance + train_instance[:]

        # train_instance =  train_instance[1:]



        preprocessed_dataset.append(train_instance)





    preprocessed_dataset=np.array(preprocessed_dataset,dtype=np.float32)

    

    return preprocessed_dataset



    


preprocessed_store_data = pre_process_store_data(store_data,months_dict,characters_dict)



preprocessed_train_data, preprocessed_train_labels = pre_process_train_data(train_data) 



preprocessed_dataset = connect_store_data_with_train_data(preprocessed_train_data,preprocessed_store_data)

import pandas as pd

from pandas import DataFrame as df

import numpy as np



store_data=pd.read_csv(path+"store.csv",parse_dates=[3])

train_data=pd.read_csv(path+"train.csv",parse_dates=[2])

test_data=pd.read_csv(path+"test.csv",parse_dates=[3])

print(store_data.keys(),"\n",train_data.keys())



train = pd.merge(train_data, store_data, on='Store')

test = pd.merge(test_data, store_data, on='Store')

print(train.keys(),"\n",test.keys())



preprocessed_dataset=[]

preprocessed_train_labels=[]





print(train.Store.isnull().sum(),train.DayOfWeek.isnull().sum(),train.Date.isnull().sum(),train.Open.isnull().sum(),\

      train.Promo.isnull().sum(),train.StateHoliday.isnull().sum(),train.SchoolHoliday.isnull().sum(),train.StoreType.isnull().sum(),\

      train.Assortment.isnull().sum(),train.CompetitionDistance.isnull().sum(),train.CompetitionOpenSinceMonth.isnull().sum(),train.CompetitionOpenSinceYear.isnull().sum(),\

      train.Promo2.isnull().sum(),train.Promo2SinceWeek.isnull().sum(),train.Promo2SinceYear.isnull().sum(),train.PromoInterval.isnull().sum())





train=train.fillna(0)

# test=test.fillna(0)

# test=test.fillna(0)







print(train.Store.isnull().sum(),train.DayOfWeek.isnull().sum(),train.Date.isnull().sum(),train.Open.isnull().sum(),\

      train.Promo.isnull().sum(),train.StateHoliday.isnull().sum(),train.SchoolHoliday.isnull().sum(),train.StoreType.isnull().sum(),\

      train.Assortment.isnull().sum(),train.CompetitionDistance.isnull().sum(),train.CompetitionOpenSinceMonth.isnull().sum(),train.CompetitionOpenSinceYear.isnull().sum(),\

      train.Promo2.isnull().sum(),train.Promo2SinceWeek.isnull().sum(),train.Promo2SinceYear.isnull().sum(),train.PromoInterval.isnull().sum())



print(test.Store.isnull().sum(),test.DayOfWeek.isnull().sum(),test.Date.isnull().sum(),test.Open.isnull().sum(),\

      test.Promo.isnull().sum(),test.StateHoliday.isnull().sum(),test.SchoolHoliday.isnull().sum(),test.StoreType.isnull().sum(),\

      test.Assortment.isnull().sum(),test.CompetitionDistance.isnull().sum(),test.CompetitionOpenSinceMonth.isnull().sum(),test.CompetitionOpenSinceYear.isnull().sum(),\

      test.Promo2.isnull().sum(),test.Promo2SinceWeek.isnull().sum(),test.Promo2SinceYear.isnull().sum(),test.PromoInterval.isnull().sum())



test["Open"]=test["Open"].fillna(0)

test=test.fillna(0)



print(test.Store.isnull().sum(),test.DayOfWeek.isnull().sum(),test.Date.isnull().sum(),test.Open.isnull().sum(),\

      test.Promo.isnull().sum(),test.StateHoliday.isnull().sum(),test.SchoolHoliday.isnull().sum(),test.StoreType.isnull().sum(),\

      test.Assortment.isnull().sum(),test.CompetitionDistance.isnull().sum(),test.CompetitionOpenSinceMonth.isnull().sum(),test.CompetitionOpenSinceYear.isnull().sum(),\

      test.Promo2.isnull().sum(),test.Promo2SinceWeek.isnull().sum(),test.Promo2SinceYear.isnull().sum(),test.PromoInterval.isnull().sum())

mappings = {'0':0, 'a':1, 'b':2, 'c':3, 'd':4}

train.StoreType.replace(mappings, inplace=True)

train.Assortment.replace(mappings, inplace=True)

train.StateHoliday.replace(mappings, inplace=True)



test.StoreType.replace(mappings, inplace=True)

test.Assortment.replace(mappings, inplace=True)

test.StateHoliday.replace(mappings, inplace=True)



train['Year'] = train.Date.dt.year

train['Month'] = train.Date.dt.month

train['Day'] = train.Date.dt.day

train['DayOfWeek'] = train.Date.dt.dayofweek

train['WeekOfYear'] = train.Date.dt.weekofyear





test['Year'] = test.Date.dt.year

test['Month'] = test.Date.dt.month

test['Day'] = test.Date.dt.day

test['DayOfWeek'] = test.Date.dt.dayofweek

test['WeekOfYear'] = test.Date.dt.weekofyear


train['CompetitionOpen'] = 12 * (train.Year - train.CompetitionOpenSinceYear) +         (train.Month - train.CompetitionOpenSinceMonth)

train['PromoOpen'] = 12 * (train.Year - train.Promo2SinceYear) +         (train.WeekOfYear - train.Promo2SinceWeek) / 4.0

train['CompetitionOpen'] = train.CompetitionOpen.apply(lambda x: x if x > 0 else 0)        

train['PromoOpen'] = train.PromoOpen.apply(lambda x: x if x > 0 else 0)





test['CompetitionOpen'] = 12 * (test.Year - test.CompetitionOpenSinceYear) +         (test.Month - test.CompetitionOpenSinceMonth)

test['PromoOpen'] = 12 * (test.Year - test.Promo2SinceYear) +         (test.WeekOfYear - test.Promo2SinceWeek) / 4.0

test['CompetitionOpen'] = test.CompetitionOpen.apply(lambda x: x if x > 0 else 0)        

test['PromoOpen'] = test.PromoOpen.apply(lambda x: x if x > 0 else 0)
month2str = {1:'Jan', 2:'Feb', 3:'Mar', 4:'Apr', 5:'May', 6:'Jun', 7:'Jul', 8:'Aug', 9:'Sept', 10:'Oct', 11:'Nov', 12:'Dec'}

train['monthStr'] = train.Month.map(month2str)

train.loc[train.PromoInterval == 0, 'PromoInterval'] = ''

train['IsPromoMonth'] = 0

for interval in train.PromoInterval.unique():

    if interval != '':

        for month in interval.split(','):

            train.loc[(train.monthStr == month) & (train.PromoInterval == interval), 'IsPromoMonth'] = 1





month2str = {1:'Jan', 2:'Feb', 3:'Mar', 4:'Apr', 5:'May', 6:'Jun', 7:'Jul', 8:'Aug', 9:'Sept', 10:'Oct', 11:'Nov', 12:'Dec'}

test['monthStr'] = test.Month.map(month2str)

test.loc[test.PromoInterval == 0, 'PromoInterval'] = ''

test['IsPromoMonth'] = 0

for interval in test.PromoInterval.unique():

    if interval != '':

        for month in interval.split(','):

            test.loc[(test.monthStr == month) & (test.PromoInterval == interval), 'IsPromoMonth'] = 1

train.keys(),test.keys()
train.drop(['Date','Customers','Open','PromoInterval','monthStr'],axis=1,inplace =True)

test.drop(['Date','Open','PromoInterval','monthStr'],axis=1,inplace =True)





# train = train[train.Sales != 0]



ho_xtrain = train.drop(['Sales'],axis=1 )

ho_ytrain = train.Sales



ho_xtest=test

ho_xtest=ho_xtest.sort_values(by=['Id'])

# ho_xtest = test.drop(['Sales'],axis=1 )

# ho_ytest = test.Sales
ho_xtest



ho_xtest.keys() , ho_xtrain.keys()


preprocessed_dataset=ho_xtrain.to_numpy()



#preprocessed_train_labels=np.log1p(ho_ytrain.to_numpy()+1)





preprocessed_train_labels=(ho_ytrain.to_numpy()+1)/1000



preprocessed_test_dataset=ho_xtest.to_numpy()

# preprocessed_tr_labels=np.log1p(ho_ytest.to_numpy())





# preprocessed_train_labels=ho_ytrain.to_numpy()


preprocessed_test_dataset=ho_xtest.to_numpy()

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split( preprocessed_dataset, preprocessed_train_labels, test_size=0.2)



X_train=np.expand_dims(X_train,axis=-1)

X_test=np.expand_dims(X_test,axis=-1)

y_train=np.array(y_train)

y_test=np.array(y_test)

y_train=y_train

y_test=y_test

print(X_train.shape)

y_train.max(),y_test.max()







def rmspe(y_true, y_pred):

    '''

    RMSPE calculus to use during training phase

    '''

    return K.sqrt(K.mean(K.square(((y_true)  - (y_pred) ) / (y_true)), axis=-1))



def rmse(y_true, y_pred):

    '''

    RMSE calculus to use during training phase

    '''

    return K.sqrt(K.mean(K.square(y_pred - y_true)))





def rmspe_val(y_true, y_pred):

    '''

    RMSPE calculus to validate evaluation metric about the model

    '''

    return np.sqrt(np.mean(np.square(((y_true) - (y_pred) ) / (y_true)), axis=0))[0]



from keras.utils.np_utils import to_categorical

from keras.models import Model, Sequential, model_from_json

from keras.optimizers import SGD, Adam, RMSprop

from keras.layers import Input, Dense, Dropout, Flatten, Lambda, Embedding,BatchNormalization,Input,Add,Concatenate

from keras.initializers import RandomNormal, Constant, he_normal

from keras.callbacks import ModelCheckpoint, EarlyStopping

from keras import regularizers

from keras.layers import Conv2D, MaxPooling2D, Conv1D, MaxPooling1D,Dense, Dropout, Flatten, Reshape, GlobalAveragePooling1D

import keras

from keras import backend as K

import tensorflow as tf



def model1():

    initializer = he_normal()



    dilation_rate=1

    bn=BatchNormalization





    inp=Input(shape=(X_train.shape[1],1))

    

    x1=bn()(Conv1D(50, kernel_size=5, dilation_rate=dilation_rate, activation='relu', padding="same",kernel_initializer=initializer)(inp))    

    x2=bn()(Conv1D(50, kernel_size=5, dilation_rate=dilation_rate, activation='relu', padding="same",kernel_initializer=initializer)(x1))

    x2= Concatenate()([x1,x2])

    x3=bn()(Conv1D(50, kernel_size=5, dilation_rate=dilation_rate, activation='relu', padding="same",kernel_initializer=initializer)(x2))

    x3=Concatenate()([x1,x2,x3])



    x=bn()(Conv1D(50, kernel_size=1, dilation_rate=dilation_rate, activation='relu', padding="same",kernel_initializer=initializer)(x3))



    x=MaxPooling1D(2)(x)

    

    x3=bn()(Conv1D(100, kernel_size=5, dilation_rate=dilation_rate, activation='relu', padding="same",kernel_initializer=initializer)(x))

    x4=bn()(Conv1D(100, kernel_size=5, dilation_rate=dilation_rate, activation='relu', padding="same",kernel_initializer=initializer)(x3))

    x4= Concatenate()([x3,x4])

    x5=bn()(Conv1D(100, kernel_size=5, dilation_rate=dilation_rate, activation='relu', padding="same",kernel_initializer=initializer)(x4))



    x=Concatenate()([x3,x4,x5])



    x=bn()(Conv1D(100, kernel_size=1, dilation_rate=dilation_rate, activation='relu', padding="same",kernel_initializer=initializer)(x))



    x=GlobalAveragePooling1D()(x)



    x=Dense(500, activation="linear")(x)

    y=Dense(1)(x)



    model= Model(inputs=inp, outputs= y)



    adam = Adam(lr=1e-3)

    model.compile(loss="mae", optimizer=adam, metrics=[rmspe,"mse","mae",rmse])

    # Compile model

    return model

    # model_m.compile(loss="mae", optimizer=adam, metrics=[rmspe,"mae","mse",rmse])













model_m=model1()



print('Build model...')

model_m.summary()


batch_size=80000

nb_epoch=400



print('Fit model...')

filepath="weights_rossmann.best.hdf5"

checkpoint = ModelCheckpoint(filepath, monitor='val_rmspe', verbose=1, save_best_only=True, mode='min')

callbacks_list = [checkpoint]



log = model_m.fit(X_train, y_train,

          validation_data=(X_test,y_test), batch_size=batch_size ,epochs=nb_epoch, shuffle=True, callbacks=callbacks_list)
model_m.load_weights(filepath)

preprocessed_test_dataset



ypred=model_m.predict(np.expand_dims(preprocessed_test_dataset[:,1:],axis=-1))



# results=np.concatenate([np.expand_dims(preprocessed_test_dataset[:,0],axis=-1),np.expm1(ypred)-1],axis=-1)



results=np.concatenate([np.expand_dims(preprocessed_test_dataset[:,0],axis=-1),ypred*1000],axis=-1)
import csv

with open('submission.csv', mode='w') as csv_file:

    fieldnames = ['Id', 'Sales']

    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)



    writer.writeheader()

    for i in results:

        print(i)

        writer.writerow({'Id':i[0], 'Sales': max(0,i[1])})



    #writer.writerow({'emp_name': 'John Smith', 'dept': 'Accounting', 'birth_month': 'November'})

    #writer.writerow({'emp_name': 'Erica Meyers', 'dept': 'IT', 'birth_month': 'March'})

# !pip install kaggle

# !kaggle competitions submit -c rossmann-store-sales -f submission.csv -m "submision"
# from google.colab import files

# files.download('out.csv') 

# files.download('weights_rossmann.best.hdf5') 

for a,b in zip(y_test,X_test):

    if a==0:

        print(a,b[0],b[1],b[2],b[3],b[4],b[5],b[6],b[7])





#Store, DayOfWeek ,int(Date[0]),int(Date[1]),int(Date[2]), Open, Promo,StateHoliday, SchoolHoliday
!ls
# !rsync -avz --progress ./model/model_both_a_13.pkl ../drive/Job/
from matplotlib import pyplot as plt

history=log



plt.figure(figsize=(25,6))

plt.subplot(131)

plt.plot(history.history['loss'][:])

plt.plot(history.history['val_loss'][:])

plt.title('model loss')

plt.ylabel('loss')

plt.xlabel('epoch')

plt.legend(['train', 'val'], loc='upper right')





plt.subplot(132)

plt.plot(history.history['rmse'][:])

plt.plot(history.history['val_rmse'][:])

plt.title('model rmse')

plt.ylabel('rmse')

plt.xlabel('epoch')

plt.legend(['train', 'val'], loc='upper right')





plt.subplot(133)

plt.plot(history.history['rmspe'][:])

plt.plot(history.history['val_rmspe'][:])

plt.title('model rmspe')

plt.ylabel('rmspe min('+str(min(history.history['val_rmspe']) ))

plt.xlabel('epoch')

plt.legend(['train', 'val'], loc='upper right')

plt.show()

!ls