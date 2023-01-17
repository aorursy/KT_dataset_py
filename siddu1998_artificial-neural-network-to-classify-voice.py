import numpy as np

import matplotlib.pyplot as plt

import pandas as pd
dataset=pd.read_csv("../input/voice.csv")

X=dataset.iloc[:,0:20]

y=dataset.iloc[:,-1].values
from sklearn.preprocessing import LabelEncoder

labelencoder = LabelEncoder()

y = labelencoder.fit_transform(y)



from sklearn.preprocessing import StandardScaler

X_sc = StandardScaler()

X= X_sc.fit_transform(X)
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)

import keras

from keras.models import Sequential

from keras.layers import Dense

from keras.callbacks import History

from keras.utils import plot_model

from keras.optimizers import SGD

classifier=Sequential()

history = History()



#number of input variables =20

#first layer 

#input_dim is only for the first layer

classifier.add(Dense(output_dim=11,init='uniform',activation='relu',input_dim=20))

#first Hidden layer

classifier.add(Dense(output_dim=11,init='uniform',activation='relu'))

#Second Hidden

classifier.add(Dense(output_dim=6,init='uniform',activation='relu'))

#output layer

classifier.add(Dense(output_dim=1,init='uniform',activation='sigmoid'))

#Running the artificial neural network

classifier.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

#fitting

classifier.fit(X_train,y_train,batch_size=10,epochs=10,validation_split=0.1,callbacks=[history],shuffle=2)



import sklearn.metrics as metrics

y_pred=classifier.predict(X_test)

y_pred = np.round(y_pred)



print('Accuracy we are able to achieve with our ANN is',metrics.accuracy_score(y_pred,y_test)*100,'%')



plt.plot(history.history['loss'], color = 'red',label='Variaton Loss over the epochs',)

plt.plot(history.history['acc'],color='cyan',label='Variation in Profit over the epochs')



plt.xlabel('Epochs')

plt.title('Loss/Accuracy VS Epoch')

plt.ylabel('Loss/Accuracy')

plt.legend(loc='best')

plt.show()


