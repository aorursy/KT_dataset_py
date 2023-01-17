# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
np.random.seed(1337)
import matplotlib.pyplot as plt
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from keras import layers
from keras.layers import Input, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D
from keras.layers import AveragePooling2D, MaxPooling2D, Dropout, GlobalMaxPooling2D, GlobalAveragePooling2D
from keras.models import Model
from keras.preprocessing import image
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras.applications.imagenet_utils import preprocess_input
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
from keras.utils import plot_model
from keras import regularizers
from keras import optimizers


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
dataset = pd.read_csv('../input/Iris.csv')
X= dataset.iloc[: ,1:5].values
labels = dataset.iloc[: , 5:6].values
print(X)
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
labelencoder = LabelEncoder()
labels[:,0] = labelencoder.fit_transform(labels[:,0])
onehotencoder = OneHotEncoder(categorical_features=[0])
labels = onehotencoder.fit_transform(labels).toarray()
print('Features shape:'+str(X.shape))
print('labels shape:'+str(labels.shape))
#Dividing into train and test set
from sklearn.cross_validation import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,labels,test_size=0.4,random_state=0)
print('Training Set:'+  str(X_train.shape))
print('Test Set:'+ str(X_test.shape))

from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
#Creating model
def model(input_shape):
    X_input = Input(input_shape)
    
    X =  Dense(8, activation=None, use_bias=True, kernel_initializer='glorot_uniform', 
          bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, 
          activity_regularizer=None, kernel_constraint=None, bias_constraint=None)(X_input)
    X= BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001)(X)
    X = Activation('relu')(X)
    X = Dropout(0.2)(X)
    '''
    X =  Dense(10, activation=None, use_bias=True, kernel_initializer='glorot_uniform', 
          bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, 
          activity_regularizer=None, kernel_constraint=None, bias_constraint=None)(X)
    X= BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001)(X)
    X = Activation('relu')(X)
    X = Dropout(0.4)(X)
    
    '''
        
    X =  Dense(16, activation=None, use_bias=True, kernel_initializer='glorot_uniform', 
          bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, 
          activity_regularizer=None, kernel_constraint=None, bias_constraint=None)(X)
    X= BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001)(X)
    X = Activation('relu')(X)
    X = Dropout(0.3)(X)
    
    
    
    
    X =  Dense(3, activation=None, use_bias=True, kernel_initializer='glorot_uniform', 
          bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, 
          activity_regularizer=None, kernel_constraint=None, bias_constraint=None)(X)
    
    X = Activation('softmax')(X)
    model = Model(inputs = X_input, outputs = X, name='IrisClassifier')
    
    return model
    

    

    
    
    
#Fetching Model
irisModel = model(X_train[1].shape)
#Compiling model
sgd = optimizers.SGD(lr=0.03, decay=1e-5, momentum=0.9, nesterov=True)
irisModel.compile(optimizer= sgd , loss = 'categorical_crossentropy' , metrics=["accuracy"])
#Fitting model
history = irisModel.fit(x=X_train , y=Y_train, epochs = 70 , batch_size=32)
print(history.history.keys())

plt.plot(history.history['acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train'], loc='upper left')
plt.show()
preds = irisModel.evaluate(x=X_test , y=Y_test)
### END CODE HERE ###
print()
print ("Loss = " + str(preds[0]))
print ("Test Accuracy = " + str(preds[1]))
print(irisModel.predict(X_test).shape)
print(irisModel.predict(X_test))


