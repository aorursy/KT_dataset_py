import pandas as pd

import matplotlib as plt

from sklearn.metrics import accuracy_score

from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import train_test_split

from sklearn.metrics import classification_report

from sklearn.neural_network import MLPClassifier

from keras.models import Sequential

from keras.layers import Dense

%matplotlib inline
df=pd.read_csv("../input/diabetes/diabetes.csv") #Import data

df.head() #see the top 5 data
#Separate data between features and targets

y=df.iloc[:,-1] #Target

X=df.iloc[:,0:7] #Feature
#train and test data

(trainX,testX,trainY,testY)=train_test_split(X,y,random_state=0, test_size=0.25)
clf =DecisionTreeClassifier()

train=clf.fit(trainX,trainY)

y_pred=train.predict(testX)
accuracy_score(testY, y_pred)
mlp = MLPClassifier()

train2=mlp.fit(trainX,trainY)

y_pred1=train2.predict(testX)
accuracy_score(testY, y_pred1)
model=Sequential()



model.add(Dense(units=12, activation='relu', input_dim=7))

model.add(Dense(units=8, activation='relu'))

model.add(Dense(units=1, activation='sigmoid'))



model.compile(loss='binary_crossentropy',

              optimizer='adam',

              metrics=['accuracy'])



# x_train and y_train are Numpy arrays --just like in the Scikit-Learn API.

model.fit(trainX, trainY, epochs=150, batch_size=10,verbose=0)

y_pred2 = model.predict_classes(testX)

accuracy_score(testY, y_pred2)