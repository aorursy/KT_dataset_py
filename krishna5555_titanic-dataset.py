# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

import seaborn as sns

import missingno

from sklearn import preprocessing

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split

from sklearn.model_selection import cross_val_score

from sklearn import tree

from sklearn.ensemble import RandomForestClassifier

import tensorflow as tf	

import numpy as np	

from tensorflow import keras	

from keras.models import Sequential	

from keras.layers import Dense	

from keras.layers import Dropout	

from keras import regularizers

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df=pd.read_csv("/kaggle/input/train.csv")

print(df)
print(df.columns)
print(df.head())
print(len(df))
missingno.matrix(df)
print(df.dtypes)
df_bin=df

df_bin["Sex"]=np.where(df_bin["Sex"]=="female",1,0)

sns.distplot(df_bin.loc[df_bin["Survived"]==1]["Sex"],kde_kws={"label":"Survived"})

sns.distplot(df_bin.loc[df_bin["Survived"]==0]["Sex"],kde_kws={"label":"Did not survive"})
label=df['Survived']

df=df.drop('Survived',axis=1)

print(label.head())
print(len(df['PassengerId'].unique()))

print(len(df['Pclass'].unique()))

print(len(df['Sex'].unique()))

print(len(df['Age'].unique()))

print(len(df['SibSp'].unique()))

print(len(df['Parch'].unique()))

print(len(df['Ticket'].unique()))

print(len(df['Fare'].unique()))

print(len(df['Cabin'].unique()))

print(len(df['Embarked'].unique()))
print(df['Pclass'].value_counts())
sns.countplot(y='Pclass',data=df)
#Convert string data to numeric value using label encoder

le = preprocessing.LabelEncoder()

df=pd.read_csv("/kaggle/input/train.csv")

label=df['Survived']

df=df.drop('Survived',axis=1)

test_df=pd.read_csv("/kaggle/input/test.csv")

traintest_df=df.append(test_df,ignore_index=False)



traintest_df=traintest_df.drop('PassengerId',axis=1)

traintest_df=traintest_df.drop('Name',axis=1)

# traintest_df=traintest_df.drop('Ticket',axis=1)



traintest_df=traintest_df.fillna("-1")



le.fit(traintest_df['Sex'])

traintest_df['Sex']=le.transform(traintest_df['Sex'])



le.fit(traintest_df['Embarked'])

traintest_df['Embarked']=le.transform(traintest_df['Embarked'])



le.fit(traintest_df['Cabin'])

traintest_df['Cabin']=le.transform(traintest_df['Cabin'])



le.fit(traintest_df['Ticket'])

traintest_df['Ticket']=le.transform(traintest_df['Ticket'])



df=traintest_df[:len(df)]

test_df=traintest_df[len(df):]

print(df.head(),len(df))

print(test_df.head(),len(test_df))
#Logistic regression

clf=LogisticRegression()

clf.fit(df,label)

clf.score(df,label)

cross_val_score(clf, df, label, cv=50).mean()
#Decision tree

clf=tree.DecisionTreeClassifier(max_depth=20)

clf.fit(df,label)

print(clf.score(df,label))

cross_val_score(clf, df, label, cv=50).mean()
#Random forest

clf = RandomForestClassifier(n_estimators=50, max_depth=11,random_state=0)

clf.fit(df,label)

print(clf.score(df,label))

cross_val_score(clf, df, label, cv=5).mean()
#Ada Boost classifier

from sklearn.ensemble import AdaBoostClassifier

clf=AdaBoostClassifier(base_estimator=RandomForestClassifier(n_estimators=14, max_depth=12),n_estimators=14)

clf.fit(df,label)

print(clf.score(df,label))

cross_val_score(clf,df,label,cv=5).mean()
#Ada Boost classifier by splitting data to train and test

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(df,label,test_size=0.2)

clf=AdaBoostClassifier(base_estimator=RandomForestClassifier(n_estimators=14, max_depth=8),n_estimators=14)

clf.fit(x_train,y_train)

print(clf.score(x_train,y_train))

print(clf.score(x_test,y_test))
test_predictions=np.array(clf.predict(test_df))

print(test_predictions)

test_dataframe=pd.DataFrame()

testdf_sample=pd.read_csv("/kaggle/input/test.csv")

test_dataframe["PassengerId"]=testdf_sample["PassengerId"]

test_dataframe["Survived"]=test_predictions

print(test_dataframe.head())
#Export to csv file

test_dataframe.to_csv("classifier_submission.csv",index=False)
#Neural network

x_train,x_test,y_train,y_test=train_test_split(df,label,test_size=0.2)

model = Sequential()	

model.add(Dense(12, input_dim=9, activation='relu'))	

model.add(Dropout(0.2))	

model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])	

history=model.fit(x_train, y_train, epochs=1000, validation_data=(x_test,y_test),verbose=0)
import matplotlib.image  as mpimg	

import matplotlib.pyplot as plt	

#-----------------------------------------------------------	

# Retrieve a list of list results on training and test data	

# sets for each training epoch	

#-----------------------------------------------------------	

acc=history.history['acc']	

val_acc=history.history['val_acc']	

loss=history.history['loss']	

val_loss=history.history['val_loss']	

epochs=range(len(acc)) # Get number of epochs	

#------------------------------------------------	

# Plot training and validation accuracy per epoch	

#------------------------------------------------	

plt.plot(epochs, acc, 'r')	

plt.plot(epochs, val_acc, 'b')	

plt.title('Training and validation accuracy')	

plt.xlabel("Epochs")	

plt.ylabel("Accuracy")	

plt.legend(["Accuracy", "Validation Accuracy"])	

plt.figure()	

#------------------------------------------------	

# Plot training and validation loss per epoch	

#------------------------------------------------	

plt.plot(epochs, loss, 'r')	

plt.plot(epochs, val_loss, 'b')	

plt.title('Training and validation loss')	

plt.xlabel("Epochs")	

plt.ylabel("Loss")	

plt.legend(["Loss", "Validation Loss"])	

plt.figure()
test_predictions=model.predict_classes(test_df)

test_dataframe=pd.DataFrame()

testdf_sample=pd.read_csv("/kaggle/input/test.csv")

test_dataframe["PassengerId"]=testdf_sample["PassengerId"]

test_dataframe["Survived"]=test_predictions

print(test_dataframe.head())
#Export to csv file

test_dataframe.to_csv("classifier_submission_using_tensorflow.csv",index=False)