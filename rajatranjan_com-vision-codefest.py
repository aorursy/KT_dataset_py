# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
import os
print(os.listdir("../input/train_hnzkrpw"))

# Any results you write to the current directory are saved as output.
train=pd.read_csv('../input/train_hnzkrpw/train.csv')
btrain=pd.read_csv('../input/train_hnzkrpw/bbox_train.csv')
test=pd.read_csv('../input/test_Rj9YEaI.csv')
img_dir="../input/train_hnzkrpw/image_data/"
train.head()
train.tail()
btrain.head()
btrain.nunique()
from PIL import Image
train_images=np.array(train.iloc[:,0])
test_images=np.array(test.iloc[:,0])
trainimagearr=[]
for i in train_images:
    img=Image.open(img_dir+i).convert('L').resize((128,128))
    trainimagearr.append(np.array(img))
testimagearr=[]
for i in test_images:
    img=Image.open(img_dir+i).convert('L').resize((128,128))
    testimagearr.append(np.array(img))

train_img=np.array(trainimagearr)
print(train_img.shape)
test_img=np.array(testimagearr)
print(test_img.shape)
plt.imshow(train_img[0])
train_img=train_img.reshape(-1, 128, 128, 1)
print(train_img.shape)
test_img=test_img.reshape(-1, 128, 128, 1)
test_img.shape
train_img=train_img/train_img.max()
test_img=test_img/test_img.max()
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Convolution2D, MaxPooling2D,Activation
from keras.callbacks import EarlyStopping
from keras.optimizers import RMSprop, Adam, Adadelta

def baseline_model():
    nb_filters = 5
    nb_conv = 5
    image_size=128
    model = Sequential()
#     model.add(Dense(128, input_shape = (-1,128,128,1), kernel_initializer='normal', activation='relu'))
#     model.add(Dense(1, kernel_initializer='normal'))
    
#     ----------------------------
#     model = Sequential()
#     model.add(Convolution2D(64,(1,1), activation='relu', padding='same',input_shape = (128,128,1)))
#     #if you resize the image above, change the input shape
#     model.add(MaxPooling2D(pool_size=(2,2)))

#     model.add(Convolution2D(128,(3,3), activation='relu', padding='same'))
#     model.add(MaxPooling2D(pool_size=(2,2)))
#     model.add(Dropout(0.25))

#     model.add(Dense(128, activation='relu'))
#     model.add(Dense(512, activation='relu'))
#     model.add(Dropout(0.25))
#     model.add(Dense(1,  kernel_initializer='normal'))
# #     model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
    
#     -------------------------------------
    # Compile model
    model.add(Convolution2D(nb_filters, nb_conv, nb_conv,input_shape=(image_size, image_size,1) ) )
    model.add(Activation('relu'))
    
    
    model.add(Convolution2D(nb_filters, nb_conv, nb_conv))
    model.add(Activation('relu'))

    model.add(Convolution2D(nb_filters, nb_conv, nb_conv))
    model.add(Activation('relu'))
    model.add(Dropout(0.25))
    
    model.add(Convolution2D(nb_filters*2, nb_conv, nb_conv))
    model.add(Activation('relu'))
    
    
    model.add(Convolution2D(nb_filters*2, nb_conv, nb_conv))
    model.add(Activation('relu'))

    model.add(Convolution2D(nb_filters*2, nb_conv, nb_conv))
    model.add(Activation('relu'))
    model.add(Dropout(0.25))
    
    model.add(Flatten())

    model.add(Dense(256))
    model.add(Activation('relu'))
    model.add(Dropout(0.25))
    
    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(Dropout(0.25))

    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dropout(0.25))
    
    model.add(Dense(1))
    model.add(Activation('linear'))
    
    model.compile(loss='mean_squared_error', optimizer=Adadelta())
    model.summary()
    return model
seed = 7
np.random.seed(seed)
# evaluate model with standardized dataset
# estimator = KerasRegressor(build_fn=baseline_model, nb_epoch=100, batch_size=5, verbose=0)
# kfold = KFold(n_splits=10, random_state=seed)
# results = cross_val_score(estimator, train_img, train.HeadCount, cv=kfold, n_jobs=1)
# print("Results: %.2f (%.2f) MSE" % (results.mean(), results.std()))
reg=KerasRegressor(build_fn=baseline_model, epochs=400, batch_size=25,verbose=1)
reg.fit(train_img, train.HeadCount)

# result = np.sqrt(mean_squared_error(y_test,prediction))
# print("Testing RMSE: {}".format(result))
prediction=reg.predict(test_img, verbose=1)
prediction
sub=pd.DataFrame({'Name':test.Name, 'HeadCount':prediction})
sub.sample(20)
sub.to_csv('vision1.csv',index=False)