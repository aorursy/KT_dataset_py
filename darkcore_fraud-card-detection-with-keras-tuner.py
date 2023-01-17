import numpy as np

import pandas as pd 

import matplotlib.pyplot as plt

from warnings import filterwarnings

filterwarnings("ignore")



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
data = pd.read_csv("/kaggle/input/creditcardfraud/creditcard.csv")

data.head()
data.drop(["Time"],inplace=True,axis=1)
data.Class.value_counts()
databalanced = data.sort_values(by="Class")[-492*2:]

#databalanced = data.sort_values(by="Class") #for non-undersample
databalanced.Class.value_counts()
databalanced.reset_index(inplace=True)

databalanced.head()
X = databalanced.iloc[:,2:-1]

#X = databalanced.iloc[:,1:-1] #for non-undersample

y = databalanced.iloc[:,-1]
from sklearn.preprocessing import MinMaxScaler, StandardScaler



#mms = MinMaxScaler(feature_range=(0,1))

mms = StandardScaler()

x = mms.fit_transform(X)
from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score, confusion_matrix

from seaborn import heatmap



x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state = 6)
from xgboost import XGBClassifier

xgb = XGBClassifier()

xgb.fit(x_train,y_train)

ypred = xgb.predict(x_test)

heatmap(confusion_matrix(y_test,ypred),annot=True,cbar=False,fmt="1d")

plt.title("XGB Accuracy: {}".format(accuracy_score(y_test,ypred)),fontsize=20)

plt.show()
from tensorflow.keras.models import Sequential, model_from_json

from tensorflow.keras.layers import Dense

from tensorflow.keras.optimizers import Adam

from kerastuner.tuners import RandomSearch

from kerastuner.engine.hyperparameters import HyperParameters



def build_model(hp):

    model = Sequential()



    model.add(Dense(hp.Int("first_dense",min_value=8,max_value=1024,step=8),activation="relu",input_dim=(x_train.shape[1])))

    

    for i in range(hp.Int("layers",1,2)):

        model.add(Dense(hp.Int(f"{i}_dense",min_value=8,max_value=1024,step=8),activation="relu"))

        

    model.add(Dense(1,activation="tanh"))



    model.compile(loss="binary_crossentropy",optimizer=Adam(lr=0.00005),metrics=["accuracy"])

    

    return model



tuntun = RandomSearch(build_model,max_trials=10,executions_per_trial=2,objective="val_accuracy")



tuntun.search(x=x_train,y=y_train,verbose=2,epochs=20,batch_size=2,validation_data=(x_test,y_test))
print(tuntun.get_best_models()[0].summary())

ypred = tuntun.get_best_models()[0].predict_classes(x_test)

heatmap(confusion_matrix(y_test,ypred),annot=True,cbar=False,fmt="1d")

plt.title("ANN Accuracy: {}".format(accuracy_score(y_test,ypred)),fontsize=20)

plt.show()
#let's test it

arc = tuntun.get_best_models()[0].to_json()

model = model_from_json(arc)



model.compile(loss="binary_crossentropy",optimizer=Adam(lr=0.00005),metrics=["accuracy"])

model.fit(x_train,y_train,epochs=20,verbose=2,batch_size=2)
ypred = model.predict_classes(x_test)

heatmap(confusion_matrix(y_test,ypred),annot=True,cbar=False,fmt="1d")

plt.title("ANN Accuracy: {}".format(accuracy_score(y_test,ypred)),fontsize=20)

plt.show()