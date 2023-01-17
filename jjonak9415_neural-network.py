# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import matplotlib.pyplot as plt

%matplotlib inline



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
train = pd.read_csv('/kaggle/input/titanic/train.csv')

test = pd.read_csv('/kaggle/input/titanic/test.csv')
# カテゴリ変数を変換



train['Sex'][train['Sex'] == 'male'] = 0

train['Sex'][train['Sex'] == 'female'] = 1

test['Sex'][test['Sex'] == 'male'] = 0

test['Sex'][test['Sex'] == 'female'] = 1
# Ticketをわかりやすい値に変換



Ticket_id_train = []

for v in train['Ticket'].values:

    for index, v_unique in enumerate(train['Ticket'].unique()):

        if v == v_unique:

            Ticket_id_train.append(index)

            break

train['Ticket'] = Ticket_id_train        
Ticket_id_test = []

for v in test['Ticket'].values:

    for index, v_unique in enumerate(test['Ticket'].unique()):

        if v == v_unique:

            Ticket_id_test.append(index)

            break

test['Ticket'] = Ticket_id_test
import keras

from keras.models import Sequential

from keras.models import Model

from keras.layers import Dense,Flatten,Activation,Input,Dropout

from keras.optimizers import Adam

from keras.layers.advanced_activations import ReLU

from keras.losses import binary_crossentropy



from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler



import tensorflow as tf
# kerasの学習の再現



np.random.seed(7)

tf.random.set_seed(7)

train_x = train[['Pclass','Sex','Ticket']].values

train_x = train_x.astype('float')

train_y = train['Survived'].values



train_x,val_x,train_y,val_y = train_test_split(train_x,train_y,test_size=0.2)
#Ticketだけscaleが大きいので標準化する



scaler = StandardScaler()

train_x = scaler.fit_transform(train_x)

val_x = scaler.transform(val_x)
print(train_x.shape,val_x.shape,train_y.shape,val_y.shape)
def load_model():

    

    model = Sequential()

    model.add(Dense(12,input_dim = train_x.shape[1],activation='relu'))

    model.add(Dense(9,activation='relu'))

    model.add(Dense(1,activation='sigmoid'))

    

    model.compile(loss=binary_crossentropy,optimizer=Adam(lr=0.001),metrics=['accuracy'])

    

    return model
model = load_model()

model.summary()
from sklearn.model_selection import GridSearchCV

from keras.wrappers.scikit_learn import KerasClassifier
grid_model = KerasClassifier(build_fn=load_model,verbose=0)



batch_size = [32,64,128]

epochs = [50,100,150]

param_grid = dict(batch_size=batch_size,epochs=epochs)



grid = GridSearchCV(estimator=grid_model,param_grid=param_grid,cv=3,verbose=2)

# grid_result = grid.fit(train_x,train_y)
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
history = model.fit(train_x,train_y,batch_size=64,epochs=50,validation_data=(val_x,val_y))
plt.plot(history.history['accuracy'])

plt.plot(history.history['val_accuracy'])

plt.title('accuracy')

plt.ylabel('Accuracy')

plt.xlabel('Epoch')

plt.legend(['Train', 'Validation'], loc='upper left')

plt.show()



# plot training & validation loss value

plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])

plt.title('loss')

plt.ylabel('Loss')

plt.xlabel('Epoch')

plt.legend(['Train', 'Validation'], loc='upper left')

plt.show()
test_feature = test[['Pclass','Sex','Ticket']].values

test_feature = test_feature.astype('float')



test_feature = scaler.transform(test_feature)
test_feature
test['Survived'] = model.predict(test_feature)

test['Survived'] =test['Survived'].apply(lambda x: round(x,0)).astype('int')



solution = test[['PassengerId', 'Survived']]
# solution.to_csv("keras_baseline.csv", index=False)