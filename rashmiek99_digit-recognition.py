import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score



from keras.utils import to_categorical

from keras.models import Sequential

from keras.layers import Dense

from keras.optimizers import Adam



import warnings

warnings.filterwarnings('ignore')
train_df = pd.read_csv('/kaggle/input/digit-recognizer/train.csv')

test_df = pd.read_csv('/kaggle/input/digit-recognizer/test.csv')
train_df.head()
train_df.shape,test_df.shape
X = train_df.iloc[:,1:785].values

y = train_df.iloc[:,0].values



final_test = test_df.values



X = X/255

final_test = final_test/255



y = to_categorical(y)



X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3)
network = Sequential()

network.add(Dense(512,input_dim=X_train.shape[1],activation='relu'))

network.add(Dense(128, activation = 'relu'))

network.add(Dense(10,activation='softmax'))
#complie the network

#To get network ready to fit to the training data, you have to first compile it. This involves

#specifying the optimizer (a choice of strategies to apply to solve for the network parameters),

#the loss function to minimize (categorical cross-entropy in this case as is common for multi-class 

#classification problems), and a choice of metrics to track in the iterative process.



network.compile(optimizer=Adam(lr=1e-4),loss='categorical_crossentropy',metrics=['accuracy'])

#network.compile(loss="categorical_crossentropy",

#              optimizer=optimizers.Adadelta(lr=1.0, rho=0.95, epsilon=1e-08, decay=0.0),

#              metrics=['accuracy'])
#Fit the model with training data

#As optional keyword arguments, specify epochs=5 (the number of sweeps through the data to make) and 

#batch_size=128 (the number of data points to use in each sweep through the data). This is in principle

#the same as the iterations of stochastic gradient descent (with a batch size of 1) 

#made in the perceptron algorithm.



history = network.fit(X_train,y_train,epochs=20,batch_size=128,verbose=0)

scores = network.evaluate(X_test,y_test)

   

print("Loss=" + str(scores[0]))

print("Accuracy=" + str(scores[1]))
#predicted_classes = network.predict_classes(final_test)



y_pred = network.predict(final_test)

predicted_classes = np.argmax(y_pred,axis=1)



submissions=pd.DataFrame({"ImageId": list(range(1,len(predicted_classes)+1)),

                         "Label": predicted_classes})

submissions.to_csv("my_digit.csv", index=False, header=True)
from sklearn.decomposition import PCA

from sklearn.preprocessing import StandardScaler





train_img   = train_df.iloc[:,1:785].values

train_label = train_df.iloc[:,0].values



test_img    = test_df.values



scaler  = StandardScaler()



scaler.fit(train_img)



train_df = scaler.transform(train_img)

test_df = scaler.transform(test_img)



pca = PCA(0.95)



pca.fit(train_img)



n = pca.components_

print(len(n))



train_img = pca.transform(train_img)

test_img = pca.transform(test_img)

final_test = pca.transform(final_test)
log_Reg = LogisticRegression(solver='lbfgs')



log_Reg.fit(train_img,train_label)

#predicted_classes = network.predict_classes(final_test)



y_pred = log_Reg.predict(test_img)

#predicted_classes = np.argmax(y_pred,axis=1)



submissions=pd.DataFrame({"ImageId": list(range(1,len(y_pred)+1)),

                         "Label": y_pred})

submissions.to_csv("my_digit_pca.csv", index=False, header=True)