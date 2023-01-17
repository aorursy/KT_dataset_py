import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

        
from tensorflow.keras.layers import Activation,Dense,Dropout,Conv1D,MaxPool1D,AveragePooling1D,GlobalAveragePooling1D,GlobalMaxPooling1D,Flatten,Input,Concatenate,Add,Activation,BatchNormalization,Average

from tensorflow.keras.models import Sequential,Model

from tensorflow.keras.optimizers import Adam

from tensorflow.keras.callbacks import EarlyStopping,ReduceLROnPlateau

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt
train_df=pd.read_csv('/kaggle/input/2dataset/train.csv')

test_df=pd.read_csv('/kaggle/input/2dataset/test.csv')

train_df




train_df = train_df.replace('', np.NaN)

test_df = test_df.replace('', np.NaN)







#train_df=train_df.fillna(train_df.mode().T[0])

train_df=train_df.fillna(train_df.mean())

#train_df=train_df.dropna()

#train_df=train_df.fillna(0)







#test_df=test_df.fillna(test_df.mode().T[0])

test_df=test_df.fillna(test_df.mean())

#test_df=test_df.fillna(0)





'''

#trainとtestを生成

mydf= pd.concat([train_df,test_df])

X_train = mydf[train_df.columns.values].dropna().drop(['number_of_reviews','last_review','reviews_per_month','id','NaN_Flag','price'], axis=1).to_numpy()

y_train = mydf[['number_of_reviews','last_review','reviews_per_month']].dropna().to_numpy()

X_test1 = train_df[train_df.isnull().any(axis=1)].drop(['number_of_reviews','last_review','reviews_per_month','id','NaN_Flag','price'], axis=1).to_numpy()

X_test2 = test_df[test_df.isnull().any(axis=1)].drop(['number_of_reviews','last_review','reviews_per_month','id','NaN_Flag'], axis=1).to_numpy()

'''
'''

in_ = Input((X_train.shape[1]))



x=in_

x = BatchNormalization()(x)

x = Activation("relu")(x)

x=Dense(256)(x)

x = BatchNormalization()(x)

x = Activation("relu")(x)

x=Dense(128)(x)

x = BatchNormalization()(x)

x = Activation("relu")(x)

x=Dense(256)(x)

x = BatchNormalization()(x)

x = Activation("relu")(x)

x=Dense(64)(x)

x = BatchNormalization()(x)

x = Activation("relu")(x)

x=Dense(256)(x)

x = BatchNormalization()(x)

x = Activation("relu")(x)

x=Dense(128)(x)

x = BatchNormalization()(x)

x = Activation("relu")(x)

out_=Dense(3)(x)



model = Model(inputs=in_, outputs=out_)

model.compile(loss='mse',optimizer=Adam(lr=0.001),metrics=['mae'])

'''
'''

# 2エポックval_lossが改善しなかったら学習率を0.5倍

reduce_lr = ReduceLROnPlateau(

                        monitor='val_loss',

                        factor=0.5,

                        patience=2,

                        min_lr=0.0001

                )

'''
'''

#学習

history=model.fit(X_train,y_train,

                  batch_size=64,

                  epochs=65,

                  validation_split=0.2,

                  callbacks=[reduce_lr]

                  )

'''
'''

plt.plot(history.history['mae'],label='mae')

plt.plot(history.history['val_mae'],label='val_mae')

plt.ylabel('mae')

plt.xlabel('epoch')

plt.legend(loc='best')

plt.show()

'''
'''

plt.plot(history.history['loss'],label='loss')

plt.plot(history.history['val_loss'],label='val_loss')

plt.ylabel('loss')

plt.xlabel('epoch')

plt.legend(loc='best')

plt.show()

'''
'''

#推論

p_test1=model.predict(X_test1)

p_test2=model.predict(X_test2)

pd.DataFrame(p_test2)

'''
'''

c=0

for i in range(train_df.shape[0]):

    if np.isnan(train_df['last_review'][i]):

        train_df['number_of_reviews'][i]=p_test1[c][0]

        train_df['last_review'][i]=p_test1[c][1]

        train_df['reviews_per_month'][i]=p_test1[c][2]

        c=c+1

print(c)



c=0

for i in range(test_df.shape[0]):

    if np.isnan(test_df['last_review'][i]):

        test_df['number_of_reviews'][i]=p_test2[c][0]

        test_df['last_review'][i]=p_test2[c][1]

        test_df['reviews_per_month'][i]=p_test2[c][2]

        c=c+1

print(c)

'''
mean1=train_df.latitude.mean(axis=0)

mean2=train_df.longitude.mean(axis=0)

mean3=train_df.last_review.mean(axis=0)

mean4=train_df.reviews_per_month.mean(axis=0)

mean5=train_df.calculated_host_listings_count.mean(axis=0)

mean6=train_df.availability_365.mean(axis=0)



std1=train_df.latitude.std(axis=0)

std2=train_df.longitude.std(axis=0)

std3=train_df.last_review.std(axis=0)

std4=train_df.reviews_per_month.std(axis=0)

std5=train_df.calculated_host_listings_count.std(axis=0)

std6=train_df.availability_365.std(axis=0)







train_df.latitude=(train_df.latitude-mean1)/std1

train_df.longitude=(train_df.longitude-mean2)/std2

train_df.last_review=(train_df.last_review-mean3)/std3

train_df.reviews_per_month=(train_df.reviews_per_month-mean4)/std4

train_df.calculated_host_listings_count=(train_df.calculated_host_listings_count-mean5)/std5

train_df.availability_365=(train_df.availability_365-mean6)/std6



test_df.latitude=(test_df.latitude-mean1)/std1

test_df.longitude=(test_df.longitude-mean2)/std2

test_df.last_review=(test_df.last_review-mean3)/std3

test_df.reviews_per_month=(test_df.reviews_per_month-mean4)/std4

test_df.calculated_host_listings_count=(test_df.calculated_host_listings_count-mean5)/std5

test_df.availability_365=(test_df.availability_365-mean6)/std6



#last_review,reviews_per_month,NaN_Flag,number_of_reviews



train_df=train_df.drop('last_review', axis=1)

train_df=train_df.drop('reviews_per_month', axis=1)

train_df=train_df.drop('number_of_reviews', axis=1)

test_df=test_df.drop('last_review', axis=1)

test_df=test_df.drop('reviews_per_month', axis=1)

test_df=test_df.drop('number_of_reviews', axis=1)





train_df=train_df.drop('NaN_Flag', axis=1)

test_df=test_df.drop('NaN_Flag', axis=1)

train_df=train_df.drop('id', axis=1)

test_df=test_df.drop('id', axis=1)



x_train = train_df[test_df.columns.values].to_numpy()



y_train = train_df['price'].to_numpy()



x_test = test_df[test_df.columns.values].to_numpy()
print(x_train.shape)

print(y_train.shape)

print(x_test.shape)



x_train=x_train.reshape(-1,x_train.shape[1],1)

y_train=y_train.reshape(-1)

x_test=x_test.reshape(-1,x_test.shape[1],1)



print(x_train.shape)

print(y_train.shape)

print(x_test.shape)



in_ = Input((x_train.shape[1],1))





#resnet

b=in_

b = BatchNormalization()(b)

b = Activation("relu")(b)



x=Conv1D(64,3,padding='same')(b)

x = BatchNormalization()(x)

x = Activation("relu")(x)

b=Conv1D(64,1)(b)

b=Add()([x,b])



x=Conv1D(64,3,padding='same')(b)

x = BatchNormalization()(x)

x = Activation("relu")(x)

b=Conv1D(64,1)(b)

b=Add()([x,b])



x=Conv1D(128,3,padding='same')(b)

x = BatchNormalization()(x)

x = Activation("relu")(x)

b=Conv1D(128,1)(b)

b=Add()([x,b])



x=Conv1D(128,3,padding='same')(b)

x = BatchNormalization()(x)

x = Activation("relu")(x)

b=Conv1D(128,1)(b)

b=Add()([x,b])



#densenet(3)

k=64

b2=in_

b2 = BatchNormalization()(b2)

b2 = Activation("relu")(b2)



x2=Conv1D(k,3,padding='same')(b2)

x2 = BatchNormalization()(x2)

x2 = Activation("relu")(x2)

b2=Concatenate()([x2,b2])



x2=Conv1D(k,3,padding='same')(b2)

x2 = BatchNormalization()(x2)

x2 = Activation("relu")(x2)

b2=Concatenate()([x2,b2])



x2=Conv1D(k,3,padding='same')(b2)

x2 = BatchNormalization()(x2)

x2 = Activation("relu")(x2)

b2=Concatenate()([x2,b2])



x2=Conv1D(k,3,padding='same')(b2)

x2 = BatchNormalization()(x2)

x2 = Activation("relu")(x2)

b2=Concatenate()([x2,b2])





#にゅーらるねっとわーく

x3=in_

x3=BatchNormalization()(x3)

x3=Activation('relu')(x3)

x3=Dense(64)(x3)

x3 = BatchNormalization()(x3)

x3 = Activation("relu")(x3)

x3=Dense(32)(x3)

x3 = BatchNormalization()(x3)

x3 = Activation("relu")(x3)

x3=Dense(64)(x3)

x3 = BatchNormalization()(x3)

x3 = Activation("relu")(x3)

x3=Dense(16)(x3)

x3 = BatchNormalization()(x3)

x3 = Activation("relu")(x3)

x3=Dense(64)(x3)

x3 = BatchNormalization()(x3)

x3 = Activation("relu")(x3)

x3=Dense(32)(x3)

x3 = BatchNormalization()(x3)

b3 = Activation("relu")(x3)











b=Concatenate()([b,b2,b3])

x=Flatten()(b)



x=Dense(64,activation='relu')(x)

x=Dense(32,activation='relu')(x)

x=Dense(64,activation='relu')(x)

#x=Dropout(0.2)(x)

x=Dense(16,activation='relu')(x)

#x=Dropout(0.2)(x)

out_=Dense(1,activation='relu')(x)



model = Model(inputs=in_, outputs=out_)

model.compile(loss='mse',optimizer=Adam(lr=0.0001),metrics=['mae'])



# 2エポックval_lossが改善しなかったら学習率を0.5倍

reduce_lr = ReduceLROnPlateau(

                        monitor='val_loss',

                        factor=0.5,

                        patience=2,

                        min_lr=0.00001

                )



history=model.fit(x_train,y_train,

                  batch_size=64,

                  epochs=20,

                  validation_split=0.2,

                  callbacks=[reduce_lr],

                  )

plt.plot(history.history['mae'],label='mae')

plt.plot(history.history['val_mae'],label='val_mae')

plt.ylabel('mae')

plt.xlabel('epoch')

plt.legend(loc='best')

plt.show()
plt.plot(history.history['loss'],label='loss')

plt.plot(history.history['val_loss'],label='val_loss')

plt.ylabel('loss')

plt.xlabel('epoch')

plt.legend(loc='best')

plt.show()
p_test=model.predict(x_test)

p_test
#print(p_test)

print(p_test.shape)
submit_df = pd.read_csv('/kaggle/input/2dataset/sampleSubmission.csv', index_col=0)

submit_df['price'] = p_test

submit_df
submit_df.to_csv('/kaggle/working/submission.csv')

for dirname, _, filenames in os.walk('./'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
#model.save('my_model.h5')
#model = keras.models.load_model('my_model.h5')
from keras.utils import plot_model

plot_model(model, # 構築したモデルを指定

    show_shapes=True, # グラフ中に出力のshapeを表示するかどうか

    show_layer_names=True # グラフ中にレイヤー名を表示するかどうか

    #,to_file='model.png'

    )
#todo バイトの報告書

#todo サブミット

#sudo rm -rf ~/*

#あれがリバティ、ユートピアのパロディ