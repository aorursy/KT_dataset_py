import pandas as pd

td_path="/kaggle/input/titanic/train.csv"

raw_td=pd.read_csv(td_path)
#Dump unwanted columns

get_column=["Pclass","Sex","Age","SibSp","Parch","Fare","Cabin","Embarked"]

raw_x=raw_td[get_column]

#Also get the training set output

y=raw_td["Survived"]

#Deal with Age nans

raw_x["Age"]=raw_x["Age"].fillna(raw_x["Age"].mean())

#Encoding Sex Feature

dict={"male":0,"female":1}

raw_x["Sex"]=raw_x["Sex"].apply(lambda x:dict[x])

#Encoding Cabin Feature

raw_x["Cabin"]=pd.get_dummies(raw_x["Cabin"])

#Encoding Embarked Feature

raw_x["Embarked"]=raw_x["Embarked"].fillna("U")

raw_x["Embarked"]=pd.get_dummies(raw_x["Embarked"])

#Normalization with Scikit

import sklearn.preprocessing as sk

import sklearn.model_selection as sms

x=sk.normalize(raw_x,copy=False)

#split into training and evaluation set

tx,ex,ty,ey=sms.train_test_split(x,y,test_size=0.1)
import keras

from keras.models import Sequential

from keras.layers import Dense



titanic=Sequential()

titanic.add(Dense(80,activation='relu',input_dim=8))

titanic.add(Dense(40,activation='relu'))

titanic.add(Dense(1,activation='sigmoid'))
#The plotting function by Piotr Migdał on github

#Modified to plot accuracy and labels 

import matplotlib.pyplot as plt

from IPython.display import clear_output

class PlotLosses(keras.callbacks.Callback):

    def on_train_begin(self, logs={}):

        self.i = 0

        self.x = []

        self.losses = []

        self.val_losses = []

        

        self.fig = plt.figure()

        

        self.logs = []



    def on_epoch_end(self, epoch, logs={}):

        

        self.logs.append(logs)

        self.x.append(self.i)

        self.losses.append(logs.get('accuracy'))

        self.val_losses.append(logs.get('val_accuracy'))

        self.i += 1

        

        clear_output(wait=True)

        plt.plot(self.x, self.losses, label="accuracy")

        plt.plot(self.x, self.val_losses, label="val_accuracy")

        plt.legend()

        plt.title("Titanic_keras")

        plt.xlabel("Epoch")

        plt.ylabel("Accuracy")

        plt.show();

plot_losses = PlotLosses()
titanic.compile(optimizer="adam",loss="mean_squared_error",metrics=['accuracy'])

titanic.fit(tx,ty,epochs=100,batch_size=100,callbacks=[plot_losses],validation_data=(ex,ey))
#read test data 

td_path="/kaggle/input/titanic/test.csv"

tds=pd.read_csv(td_path)



#process as before

get_column=["Pclass","Sex","Age","SibSp","Parch","Fare","Cabin","Embarked"]

raw_x=tds[get_column]

#Deal with Age nans

raw_x["Age"]=raw_x["Age"].fillna(raw_x["Age"].mean())

#fill fare nas

raw_x["Fare"]=raw_x["Fare"].fillna(raw_x["Fare"].mean())

#Encoding Sex Feature

dict={"male":0,"female":1}

raw_x["Sex"]=raw_x["Sex"].apply(lambda x:dict[x])

#Encoding Cabin

raw_x["Cabin"]=pd.get_dummies(raw_x["Cabin"])

#Encoding Embarked Feature

raw_x["Embarked"]=raw_x["Embarked"].fillna("U")

raw_x["Embarked"]=pd.get_dummies(raw_x["Embarked"])

#Normalization with Scikit

x=sk.normalize(raw_x)
out=titanic.predict_classes(x)
#Reshape the output

out=out.reshape((1,418))

#Add the PassengerId column

csv=pd.DataFrame({'PassengerId':range(892,1310,1),'Survived':out[0]})

#print to file

filename = 'Titanic Predictions 2.csv'

csv.to_csv(filename,index=False)