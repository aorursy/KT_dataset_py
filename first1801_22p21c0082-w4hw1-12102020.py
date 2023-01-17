# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory
import tensorflow as tf
import tensorflow.keras as keras
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
Mtest = pd.read_csv('/kaggle/input/thai-mnist-classification/submit.csv')
Mtest
Mtest["id"] = Mtest["id"].apply(lambda x:str(x)+str(".png"))
Mtest
Mtrain = pd.read_csv('/kaggle/input/thai-mnist-classification/mnist.train.map.csv')
Mtrain
Mtrain["category"].value_counts()
Mtrain["category"] = Mtrain["category"].astype(str)
Mtrain["category"].value_counts()
Mtrain.iloc[0].id
TRAIN_PATH = '/kaggle/input/thai-mnist-classification/train/'
TEST_PATH = '/kaggle/input/thai-mnist-classification/test/'
# datagen = keras.preprocessing.image.ImageDataGenerator( rescale=1./255,
#                              validation_split=0.2  ,
#                              rotation_range=40,
#                              shear_range=0.2)
datagen = keras.preprocessing.image.ImageDataGenerator( rescale=1./255, validation_split=0.2  ,
                                                           
                            
    fill_mode='nearest',
)
                             
                             
                            



train_it = datagen.flow_from_dataframe(Mtrain,directory=TRAIN_PATH,class_mode="categorical",
                                            x_col="id",
                                            y_col="category",
                                       subset="training",
                                      batch_size=64,
                                       color_mode='grayscale'
                                      )
Valid_set = datagen.flow_from_dataframe(Mtrain,directory=TRAIN_PATH,
                                            class_mode="categorical",
                                            subset="validation",
                                            x_col="id",
                                            y_col="category",
                                        color_mode='grayscale',
                                        batch_size=64
                                           )

from tensorflow.keras.applications import Xception ,InceptionResNetV2 ,MobileNetV2
from tensorflow.keras import layers,Model,Sequential
from tensorflow.keras.layers import Conv2D,MaxPool2D,Dropout,Dense,Flatten
from tensorflow.keras import optimizers
# X_I = layers.Input(shape=(256,256,1))
# x = layers.Conv2D(3, 3, activation='relu',padding="same")(X_I)


# x=layers.Flatten()(x)
# x = layers.Dense(128 , activation='relu')(x)
# x  =layers.BatchNormalization()(x)
# x = layers.Dense(10 , activation='softmax')(x)

# model = Model(X_I,x)
# model.summary()
# model.compile(loss='categorical_crossentropy',
#               optimizer=optimizers.Adam(lr=5e-4),
#               metrics=['accuracy']) 
!pip install -U git+https://github.com/qubvel/efficientnet
ires  = MobileNetV2(include_top=False, weights='imagenet'  )
for l in ires.layers[:]:
  l.trainable  = False

x_in = layers.Input(shape=(256, 256, 1))
x = layers.Conv2D(3, 3, activation='relu',padding="same")(x_in)
x = ires(x)
x=layers.Flatten()(x)
x = layers.Dense(128, activation='relu')(x)
x = layers.Dropout(0.5)(x)
x = layers.BatchNormalization()(x)
x = layers.Dense(10 , activation='softmax')(x)
model_1 = Model(x_in, x)
model_1.summary()
model_1.compile(loss='categorical_crossentropy',
              optimizer=optimizers.Adam(lr=3e-5),
              metrics=['accuracy'])
XCNet  = Xception(include_top=False, weights='imagenet' )
for l in XCNet.layers[:]:
    l.trainable  = False
x_in = layers.Input(shape=(256, 256, 1))
x = layers.Conv2D(3, 3, activation='relu',padding="same")(x_in)
x = XCNet(x)

x=layers.Flatten()(x)
x = layers.Dense(128 , activation='relu')(x)
x  =layers.BatchNormalization()(x)
x = layers.Dense(10 , activation='softmax')(x)
XCNetmodel = Model(x_in, x)
XCNetmodel.summary()
XCNetmodel.compile(loss='categorical_crossentropy',
              optimizer=optimizers.Adam(lr=1e-4),
              metrics=['accuracy'])
learning_rate_reduction = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', patience = 1, verbose=1,factor=0.3, min_lr=0.000001)
Earlystop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)
model_1.fit(train_it , verbose=1 ,epochs=30 , validation_data =Valid_set ,callbacks=[Earlystop],steps_per_epoch=train_it.n//train_it.batch_size)
XCNetmodel.fit(train_it , verbose=1 ,epochs=30 , validation_data =Valid_set ,callbacks=[Earlystop],steps_per_epoch=train_it.n//train_it.batch_size)
# import matplotlib.pyplot as plt
# plt.plot(history.history['accuracy'])
# plt.plot(history.history['val_accuracy'])
# plt.title('model accuracy')
# plt.ylabel('accuracy')
# plt.xlabel('epoch')
# plt.legend(['train', 'test'], loc='upper left')
# plt.show()
# # summarize history for loss
# plt.plot(history.history['loss'])
# plt.plot(history.history['val_loss'])
# plt.title('model loss')
# plt.ylabel('loss')
# plt.xlabel('epoch')
# plt.legend(['train', 'test'], loc='upper left')
# plt.show()
Real_train = pd.read_csv('/kaggle/input/thai-mnist-classification/train.rules.csv')
Real_train
def extractfeature(x_col,PATH,DF):
    ########################## DATAGEN ######################################
    test_datagen=keras.preprocessing.image.ImageDataGenerator(rescale=1./255.)
    test_generator=test_datagen.flow_from_dataframe(
    dataframe=DF,
    directory=PATH,
    x_col=x_col,
    y_col=None,
    batch_size=32,
    seed=42,
    shuffle=False,
    class_mode=None,
    )
    
    #####################################################################
    Pred_Train_rule = (model_1.predict_generator(generator=test_generator))
    Pred_train_reshape = np.reshape(np.argmax(Pred_Train_rule,axis=1),(-1,1))
    
    return Pred_train_reshape

# Real_train["Feature2_EX"] = extractfeature("feature2",TRAIN_PATH,Real_train)
# Real_train["Feature3_EX"] = extractfeature("feature3",TRAIN_PATH,Real_train)
# Real_train
# #fill 0
# DropReal_train = Real_train.dropna()
# DropReal_train["Feature1_EX"] = extractfeature("feature1",TRAIN_PATH,DropReal_train)
# DropReal_train
# result = pd.merge(DropReal_train, Real_train, how='right')
# result
# result =result.fillna(0)
# result
# ForTrain= result[["Feature1_EX","Feature2_EX","Feature3_EX","predict"]]
# ForTrain
# ForTrain[["Feature1_EX","Feature2_EX","Feature3_EX"]]
# from sklearn.model_selection import train_test_split
# data_train, data_test, labels_train, labels_test = train_test_split(ForTrain[["Feature1_EX","Feature2_EX","Feature3_EX"]], ForTrain["predict"], test_size=0.20, random_state=42)
# import xgboost as xgb
# model1 = xgb.XGBClassifier()
# model1.fit(data_train,labels_train)
# from sklearn.metrics import accuracy_score
# pred =model1.predict(data_test)
# print('Model 1 XGboost Report %r' % (accuracy_score(labels_test, pred)))
################ predict ################
Test_rule = pd.read_csv("/kaggle/input/thai-mnist-classification/test.rules.csv")
Test_rule
MyTest_rule = Test_rule[["id","feature1","feature2","feature3"]]
MyTest_rule = MyTest_rule.dropna()
MyTest_rule
MyTest_rule["Feature1_EX"] = extractfeature("feature1",TEST_PATH,MyTest_rule)
MyTest_rule
ForTest = Test_rule[["id","feature1","feature2","feature3"]]
ForTest["Feature2_EX"] = extractfeature("feature2",TEST_PATH,ForTest)
ForTest["Feature3_EX"] = extractfeature("feature3",TEST_PATH,ForTest)
ForTest
result_Test = pd.merge(ForTest, MyTest_rule, how='left')
result_Test = result_Test[["id","Feature1_EX","Feature2_EX","Feature3_EX"]]
result_Test
def _f(a,b,c):
    if(a==0):
        return b*c
    if(a == 1):
        return abs(b-c)
    if(a == 2):
        return (b+c)*abs(b-c)
    if(a==3):
        return abs((c*(c +1) - b*(b-1))//2)
    if(a==4):
        return 50+b-c
    if(a==5):
        return min(b,c)
    if(a==6):
        return max(b,c)
    if(a==7):
        return ((b*c)%9)*11
    if(a==8):
        return (((b**2)+1)*(b) +(c)*(c+1))%99
    if(a==9):
        return 50+b
    
    return b+c
result_Test["predict"] = [_f(row['Feature1_EX'],row['Feature2_EX'],row['Feature3_EX']) for index, row in result_Test.iterrows()]
result_Test
submit = pd.read_csv("/kaggle/input/thai-mnist-classification/submit.csv")
submit
submit = pd.merge(submit, result_Test, how='left',on='id')
submit = submit[["id","predict_y"]]
submit.rename(columns={'predict_y':'predict'}, inplace=True)
submit
submit.to_csv('/kaggle/working/MYSUBMIT3.csv',index=False)
XCNetmodel.save("XCNeWeight")
# Image.load_img(TRAIN_PATH+Real_train["feature2"][0])
# feature["feature1"].apply(lambda x: XCNetmodel.predict())
# input_layers1 = layers.Input(shape=(256, 256, 1))
# input_layers2 = layers.Input(shape=(256, 256, 1))
# input_layers3 = layers.Input(shape=(256, 256, 1))
# combinedInput = layers.concatenate([input_layers1,input_layers2,input_layers3])
# x = layers.Dense(99 , activation='softmax')(combinedInput)
# MultiModel = Model(combinedInput,x)

# combinedInput = layers.concatenate([XCNetmodel.input, XCNetmodel.input,XCNetmodel.input])
# combinedOutput = layers.concatenate([XCNetmodel.output, XCNetmodel.output,XCNetmodel.output])
# x = layers.Dense(99 , activation='softmax')(combinedOutput)
# NarutoModel = Model(combinedInput,x)
# NarutoModel.summary()
# keras.utils.plot_model(NarutoModel, show_shapes=True)