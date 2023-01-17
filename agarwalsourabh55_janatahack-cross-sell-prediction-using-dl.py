# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train=pd.read_csv('../input/janatahack-crosssell-prediction/train.csv')
test=pd.read_csv('../input/janatahack-crosssell-prediction/test.csv')

train=pd.get_dummies(train,drop_first=True)
test=pd.get_dummies(test,drop_first=True)
train.shape

target=train['Response']
train=train.drop(['Response','id'],axis=1)


ids=test['id']
test=test.drop(['id'],axis=1)
'''
a={'Male':0,'Female':1}
train['Gender']=train['Gender'].map(a)
a={'> 2 Years':0, '1-2 Year':2, '< 1 Year':1}
train['Vehicle_Age']=train['Vehicle_Age'].map(a)
a={'Yes':1,'No':0}
train['Vehicle_Damage']=train['Vehicle_Damage'].map(a)



a={'Male':0,'Female':1}
test['Gender']=test['Gender'].map(a)
a={'> 2 Years':0, '1-2 Year':2, '< 1 Year':1}
test['Vehicle_Age']=test['Vehicle_Age'].map(a)
a={'Yes':1,'No':0}
test['Vehicle_Damage']=test['Vehicle_Damage'].map(a)

'''


#from imblearn.combine import SMOTETomek
#os=SMOTETomek(0.75)
#X_train_ns,y_train_ns=os.fit_sample(train,target)


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(train,target,test_size=0.4,random_state=3)


#col=['Age','Region_Code','Annual_Premium','Policy_Sales_Channel','Vintage']
##X_train_new=X_train[col]
#X_test_new=X_test[col]
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.transform(X_test)

# Feature Scaling
#In 90 5 of the case in deep learning we use feature sacling
# so that our data will be in same scale
#this will reduce the computation time 

from sklearn.preprocessing import Normalizer
sc = Normalizer()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#X_train
#for i in ['Gender','Driving_License','Previously_Insured','Vehicle_Age','Vehicle_Damage']:
#        X_train_new=np.append(np.array(X_train[i]),X_train_new)

#a=np.array([[1,2,3,4]])
#b=np.array([[6,7,8,9]])

#c=np.concatenate((a,b),axis=0)
X_train.shape
# Importing the Keras libraries and packages
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LeakyReLU,PReLU,ELU
from keras.layers import Dropout
# Initialising the ANN
classifier = Sequential()

# Adding the input layer and the first hidden layer

classifier.add(Dense(units = 6, kernel_initializer = 'he_uniform',input_dim = 11,activation='relu'))
# Adding the second hidden layer

classifier.add(Dense(units = 6, kernel_initializer = 'he_uniform',activation='relu'))
# Adding the output layer

classifier.add(Dense(units = 1, kernel_initializer = 'he_uniform', activation = 'sigmoid'))

# Compiling the ANN
classifier.compile(optimizer = 'sgd' , loss = 'binary_crossentropy', metrics = ['accuracy'])

# Fitting the ANN to the Training set
model_history=classifier.fit(X_train, y_train,validation_split=0.33, batch_size = 10, epochs = 10)
for layer in classifier.layers:
    weights = layer.get_weights()
    print(weights)


import matplotlib.pyplot as plt 
from keras.utils import plot_model
plot_model(classifier, to_file='/tmp/model.png', show_shapes=True,)
classifier.get_config()

print(model_history.history.keys())
# summarize history for accuracy
plt.plot(model_history.history['accuracy'])
plt.plot(model_history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(model_history.history['loss'])
plt.plot(model_history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
test=sc.transform(test)
y_pred = classifier.predict(test)

y_pred=y_pred[:,0]

from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV

from keras.models import Sequential
from keras.layers import Dense, Activation, Embedding, Flatten, LeakyReLU, BatchNormalization, Dropout
from keras.activations import relu, sigmoid



def create_model(layers, activation):
    model = Sequential()
    for i, nodes in enumerate(layers):
        if i==0:
            model.add(Dense(nodes,kernel_initializer = 'he_uniform',input_dim=X_train.shape[1]))
            model.add(Activation(activation))
            model.add(Dropout(0.3))
        else:
            model.add(Dense(nodes,kernel_initializer = 'he_uniform'))
            model.add(Activation(activation))
            model.add(Dropout(0.3))
            
    model.add(Dense(units = 1, kernel_initializer= 'glorot_uniform', activation = 'sigmoid')) # Note: no activation beyond this point
    
    model.compile(optimizer='adam', loss='binary_crossentropy',metrics=['accuracy'])
    return model
    
model = KerasClassifier(build_fn=create_model, verbose=1)

layers = [(10,15)] 
activations = ['sigmoid', 'relu']
param_grid = dict(layers=layers, activation=activations, batch_size = [12800, 25600], epochs=[10])
grid = GridSearchCV(estimator=model, param_grid=param_grid,cv=5)


#[grid_result.best_score_,grid_result.best_params_]
grid_result = grid.fit(X_train, y_train)

import matplotlib.pyplot as plt 
print([grid_result.best_score_,grid_result.best_params_])
# summarize history for accuracy

output = pd.DataFrame({'id': ids, 'Response': y_pred})
output.to_csv('my_submission5.csv', index=False)
print("Your submission was successfully saved!")
