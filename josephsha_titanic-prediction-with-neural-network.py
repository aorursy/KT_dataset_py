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



#Set the `python` built-in pseudo-random generator at a fixed value for reproducibility

import random

seed_value=0

random.seed(seed_value)

np.random.seed(seed_value)
titanic_data=pd.read_csv("/kaggle/input/titanic/train.csv")

titanic_data.head()
test_data=pd.read_csv("/kaggle/input/titanic/test.csv")

test_data.head()
#Bar chart on Victims and those who survived

import matplotlib.pyplot as plt



myplt=titanic_data["Survived"].value_counts().plot(kind='bar',color=['r','g'])

myplt.set(ylabel="# of passengers")

myplt.set_xticklabels(["Victims","Survived"])

#seaborn is a good visualization library

import seaborn as sns

fig, axs = plt.subplots(nrows=1, figsize=(10, 10))





sns.heatmap(titanic_data.drop(['PassengerId'], axis=1).corr(), annot=True,square=True,cmap='vlag')
#Ensure that the internet option is turned on in the settings tab (right side) in Kaggle Notebook. 

#This allows us to install external libraries. We need nameparser for splitting Title from Name field.

#First ensure that the pip version is the latest

!pip install --upgrade pip

!pip install nameparser

from nameparser import HumanName



titanic_data['NameTitle']=titanic_data['Name'].apply(lambda x: HumanName(x).title)

test_data['NameTitle']=test_data['Name'].apply(lambda x: HumanName(x).title)

titanic_data.groupby(['NameTitle']).agg('count')

#titanic_test.groupby(['NameTitle']).agg('count')
feature_list=["Pclass","Sex","Age","SibSp","Parch","Fare","NameTitle", "Survived"] 

feature_list_minus_y=["Pclass","Sex","Age","SibSp","Parch","Fare","NameTitle"] 



titanic_data=titanic_data[feature_list]

titanic_test=test_data #Keep a copy of the test data with all the fields

test_data=test_data[feature_list_minus_y]
#let's check the training data for missing values

import missingno as msno

msno.bar(titanic_data)
msno.bar(test_data)
#Define a null replace fuction for Age. 

def fillna_Values(df):

    df['Age'] = df.groupby(['Sex', 'NameTitle'])['Age'].apply(lambda x: x.fillna(x.mean()))

    df['Age'] = df.groupby(['Sex'])['Age'].apply(lambda x: x.fillna(x.mean()))   #we reapply for any nulls left behind

    df['Fare'] = df.groupby(['Pclass'])['Fare'].apply(lambda x: x.fillna(x.mean()))

    return df
#Call function to Replace nulls

pd.options.mode.chained_assignment = None  # default='warn'

titanic_data=fillna_Values(titanic_data)

test_data=fillna_Values(test_data)

#Now le'ts check again the nulls

msno.bar(titanic_data)
#Define a function for standardization

def standardizeData(X):

    from sklearn.preprocessing import StandardScaler

    

    #We want to redue the name titles

    X=X.replace(to_replace =["Capt.","Col.","Sir."],  

                            value ="Mr.") 

    X=X.replace(to_replace =["Lady.","Mlle.","Mme.","Ms.","the Countess. of"],  

                            value ="Mrs.") 

   

    train_numerical_features = ["Age","SibSp","Parch","Fare"]

        

    

    from sklearn.preprocessing import MinMaxScaler

    ss_scaler = MinMaxScaler(feature_range=(0,1))    

    X_ss = pd.DataFrame(data = X)

   

    X_ss[train_numerical_features] = ss_scaler.fit_transform(X_ss[train_numerical_features])

    

    

    X_ss=pd.get_dummies(X_ss, columns=["Pclass","Sex","NameTitle"])

    return X_ss
#Call the function to Standarize numerical values and handle categorical values

titanic_data=standardizeData(titanic_data)

test_data=standardizeData(test_data)
titanic_data.head()
from sklearn.model_selection import train_test_split



train_data, mytest_data = train_test_split(titanic_data, test_size=0.2)
#Prepare X and y

y_train=train_data["Survived"]

y_test=mytest_data["Survived"]



#drop Survived clolum in y_test data

X_train=train_data.drop("Survived",axis=1)

X_test=mytest_data.drop("Survived",axis=1)
from sklearn.ensemble import GradientBoostingClassifier

#Fit the model

model=GradientBoostingClassifier(n_estimators=100,max_depth=5,random_state=1)

model.fit(X_train,y_train)
#Predict mytest and print the accuracy

mypredictions=model.predict(X_test)



from sklearn.metrics import accuracy_score

print(accuracy_score(y_test, mypredictions))

from sklearn.metrics import f1_score

print('F1 score:',f1_score(y_test, mypredictions))
#Creating a deep neural network

#Also a dropout layer to prevent overfitting.

import tensorflow as tf



#Set the random generator at a fixed value for reproducibility

tf.random.set_seed(seed_value)



#By trial and error we fine tuned the architecture with  hidden layers and dropouts to regularize



nmodel=tf.keras.models.Sequential([tf.keras.layers.InputLayer(input_shape=(16,)),    

                                   tf.keras.layers.Dense(8,activation=tf.nn.relu),

                                   #tf.keras.layers.BatchNormalization(),

                                   tf.keras.layers.Dropout(0.4),  

                                   tf.keras.layers.Dense(8,activation=tf.nn.relu),

                                   tf.keras.layers.Dropout(0.4),  

                                   tf.keras.layers.Dense(8,activation=tf.nn.relu),

                                   tf.keras.layers.Dropout(0.4),  

                                   tf.keras.layers.Dense(1,activation='sigmoid')])

print(nmodel.summary())
#Compile model - use optimizer as Adam & loss function as mean_squared_error 





#nmodel.compile(optimizer='Adam',     #use default learning_rate i.e.0.001      

#             loss='mean_squared_error',metrics=['accuracy'])



nmodel.compile(optimizer='Adam',     #use default learning_rate i.e.0.001      

             loss='mean_squared_error',metrics=['accuracy'])





#Train model for 50 epochs

nmodel.fit(X_train,y_train,batch_size=32,epochs=50)
#Let's check and see how the model is working with test set

from sklearn.metrics import accuracy_score



y_pred=nmodel.predict(X_test)

y_pred = [np.round(x) for x in y_pred]

print('Accuracy:',accuracy_score(y_test, y_pred))



score=nmodel.evaluate(X_test,y_test,verbose=1)

#Train for entire train_data

#Prepare X and y

#Drop Survived column.

y_train=titanic_data["Survived"]



#Copy the full training data

X_train=titanic_data

X_train=X_train.drop("Survived",axis=1)
import matplotlib.pyplot as plt

#Train the model with the entire training data. Run for 250 epochs

training = nmodel.fit(X_train, y_train, epochs=250, batch_size=32, validation_split=0.2, verbose=0)





#Plot the accuracy of training & validation

plt.plot(training.history['accuracy'])

plt.plot(training.history['val_accuracy'])

plt.title('model accuracy')

plt.ylabel('Accuracy')

plt.xlabel('epoch')

plt.legend(['train', 'validation'], loc='upper left')

plt.show()
#Finally use the neural network model, predict 

final_df=titanic_test

final_df["Survived"]=nmodel.predict(test_data)

final_df["Survived"]=titanic_test["Survived"].apply(lambda x: round(x,0)).astype('int')

#Submit the result

output=pd.DataFrame({'PassengerId':final_df.PassengerId,'Survived':final_df.Survived})

output.to_csv('my_submission.csv',index=False)

print("Your submission was successfuly saved!")