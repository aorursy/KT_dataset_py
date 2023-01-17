import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import keras 



np.random.seed(2)
data=pd.read_csv("../input/creditcard.csv")
## DATA EXPORATION
data.head()
## data preprocessing
from sklearn.preprocessing import StandardScaler

data['normalizeAmount']=StandardScaler().fit_transform(data['Amount'].values.reshape(-1,1))

data=data.drop(['Amount'], axis=1)
data.head()
data= data.drop(['Time'], axis=1)
data.head()
## seperating the independet and dependent variable
X=data.iloc[:, data.columns !='Class']

y=data.iloc[:, data.columns == 'Class']
X.head()
y.head()
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test= train_test_split(X, y, test_size=0.3, random_state=0)
X_train.shape
X_test.shape
X_train=np.array(X_train)

X_test=np.array(X_test)

y_train=np.array(y_train)

y_test=np.array(y_test)
## Deep neural network
from keras.models import Sequential

from keras.layers import Dense

from keras.layers import Dropout 
model=Sequential([

    Dense(units=16, input_dim=29, activation='relu'),

    Dense(units=24, activation='relu'),

    Dropout(0.5),

    Dense(20, activation='relu'),

    Dense(24, activation='relu'),

    Dense(1, activation='sigmoid'),

])
model.summary()
### Training the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.fit(X_train, y_train, batch_size=15, epochs=5)
score=model.evaluate(X_test, y_test)
print(score)
from sklearn.metrics import confusion_matrix

y_pred=model.predict(X_test)

y_test=pd.DataFrame(y_test)
cm_matrix=confusion_matrix(y_test, y_pred.round())
print(cm_matrix)
y_pred=model.predict(X)

y_expected=pd.DataFrame(y)

cm_matrix=confusion_matrix(y_expected, y_pred.round())

##random forest
from sklearn.ensemble import RandomForestClassifier
random_forest=RandomForestClassifier(n_estimators=100)

random_forest.fit(X_train,y_train)
y_pred=random_forest.predict(X_test)
random_forest.score(X_test, y_test)