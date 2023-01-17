import numpy as np

import pandas as pd

import tensorflow as tf

from tensorflow.keras.callbacks import TensorBoard

from time import time

import matplotlib

from math import floor, ceil
#Read train and test csv

train = pd.read_csv("../input/train/train.csv") 

test = pd.read_csv("../input/test/test.csv") 
train.plot.scatter(x='Age', y='AdoptionSpeed')

train.plot.scatter(x='Breed1', y='AdoptionSpeed')
print(train.columns.values.tolist())
x=(train[['Type', 'Age', 'Breed1', 'Breed2','Gender','Color1','Color2','Color3','MaturitySize','FurLength','Vaccinated','Dewormed','Sterilized','Health','Quantity','Fee']])

y=(train[['AdoptionSpeed']])

testx=(test[['Type', 'Age', 'Breed1', 'Breed2','Gender','Color1','Color2','Color3','MaturitySize','FurLength','Vaccinated','Dewormed','Sterilized','Health','Quantity','Fee']])
train_x,dev_x=x[200:],x[:200] 

train_y,dev_y=y[200:],y[:200]

#Data shape

print(train_x.shape,train_y.shape,dev_x.shape,dev_y.shape)

print(train['Age'].max())
from sklearn.tree import DecisionTreeRegressor

from sklearn.metrics import accuracy_score



tree_model=DecisionTreeRegressor(random_state=1)

tree_model.fit(train_x,train_y)




print("Making predictions for the following 5 pets:")

print(dev_x.head())

print("The predictions are")

pred=tree_model.predict(dev_x)

pred1=[ceil(item) for item in pred ]

print(pred1[:5])

flat_y = [item for sublist in dev_y.astype(float).values for item in sublist]

print(flat_y[:5])

print("Dev accuracy:",accuracy_score(pred1,dev_y))

pred1=tree_model.predict(dev_x)

tpred1=tree_model.predict(train_x)

from sklearn.ensemble import RandomForestRegressor





forest_model = RandomForestRegressor(random_state=1)

forest_model.fit(train_x, train_y)

melb_preds = forest_model.predict(dev_x)

pred2=[ceil(item) for item in melb_preds ]

print(pred2[:5])

print(flat_y[:5])

print(accuracy_score(flat_y, pred2))

pred2 = forest_model.predict(dev_x)

tpred2=forest_model.predict(train_x)
from xgboost import XGBRegressor



my_model = XGBRegressor()

# Add silent=True to avoid printing out updates with each cycle

my_model.fit(train_x, train_y, verbose=False)



# make predictions

preds = my_model.predict(dev_x)

preds=[ceil(item) for item in preds ]

print(preds[:5])

print(flat_y[:5])

print(accuracy_score(flat_y, preds))









my_model = XGBRegressor(n_estimators=1000, learning_rate=0.05)

my_model.fit(train_x, train_y, #early_stopping_rounds=2000, 

             eval_set=[(dev_x, dev_y)], verbose=False)







preds = my_model.predict(dev_x)

pred3=[ceil(item) for item in preds ]

print(pred3[:5])

print(flat_y[:5])

print(accuracy_score(flat_y, pred3))



pred3 = my_model.predict(dev_x)

tpred3=my_model.predict(train_x)
#train_x['Pred1']=tpred1

#train_x['Pred2']=tpred2

#train_x['Pred3']=tpred3

#dev_x['Pred1']=pred1

#dev_x['Pred2']=pred2

#dev_x['Pred3']=pred3

#print(train_x)

#Pandas to numpy

train_x=train_x.values

train_y=train_y.values



#dev_x=dev_x.values

#dev_y=dev_y.values

#Transform targets [0,3,...,4] to [[1,0,0,0,0],[0,0,0,1,0],...,[0,0,0,0,1]]

train_y = tf.keras.utils.to_categorical(train_y, 5)

dev_y = tf.keras.utils.to_categorical(dev_y, 5)
#Create the model

model = tf.keras.Sequential()

                                                     

model.add(tf.keras.layers.BatchNormalization(input_shape=(16,))) #BachNorm

model.add(tf.keras.layers.Activation("relu"))   #Relu Activation



model.add(tf.keras.layers.Dense(1024))

model.add(tf.keras.layers.BatchNormalization()) #BachNorm

model.add(tf.keras.layers.Activation("relu")) #Relu Activation



model.add(tf.keras.layers.Dense(512)) 

model.add(tf.keras.layers.BatchNormalization()) #BachNorm

model.add(tf.keras.layers.Activation("relu")) #Relu Activation



model.add(tf.keras.layers.Dense(256)) 

model.add(tf.keras.layers.Dropout(0.5)) #Dropout

 

model.add(tf.keras.layers.Dense(128,activation='relu')) #Dense Layer with relu activation

model.add(tf.keras.layers.Dense(5, activation='softmax')) #Dense Layer with softmax activation so it can predict one of the 5 Labels



model.compile(loss=tf.keras.losses.categorical_crossentropy,

              optimizer=tf.keras.optimizers.Adadelta(),

              metrics=['accuracy'])

print(model.summary())
#Train the model

tensorboard = TensorBoard(log_dir="logs/{}".format(time()))

bestEpoch=tf.keras.callbacks.ModelCheckpoint("logs/checkpoint", monitor='val_acc', verbose=0, save_best_only=True, save_weights_only=False, mode='auto', period=1)



model.fit(train_x, train_y,

          batch_size=1000,

          epochs=100,

          verbose=1,

          validation_data=(dev_x, dev_y),

          callbacks=[tensorboard,bestEpoch])
#Predict on the test set

model=tf.keras.models.load_model(

    "logs/checkpoint",

    custom_objects=None,

    compile=True

)

prediction=model.predict(testx)

yy=(test[['PetID']])



pred=model.predict(dev_x)

print(pred.argmax(axis=1)[:5])

print("Dev accuracy:",accuracy_score(pred.argmax(axis=1),flat_y))
#Save results

final=pd.DataFrame(np.array(prediction.argmax(axis=1)),columns=['AdoptionSpeed'])

final['PetID']=yy

final=final[['PetID','AdoptionSpeed']]

final.to_csv("submission.csv", index=False)