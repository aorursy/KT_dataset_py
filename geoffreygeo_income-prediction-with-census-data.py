# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.preprocessing import LabelEncoder # labelencoder 

from sklearn.model_selection import train_test_split #Data splitting

from sklearn.metrics import accuracy_score,confusion_matrix #Testing model





#Tensorflow 

import tensorflow as tf

from tensorflow import keras



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import seaborn as sns

sns.set(style="darkgrid")

import os

print(os.listdir("../input"))

import matplotlib.pyplot as plt

import scikitplot as skplt



#Machine learning algorithms

from sklearn.tree import DecisionTreeClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn import svm

from sklearn.naive_bayes import GaussianNB

from sklearn.linear_model import LinearRegression

# Any results you write to the current directory are saved as output.
df = pd.read_csv('../input/adult.csv')

display(tf.version)

display(df.head())

df.describe()
a4_dims = (14, 8.7)

fig, ax = plt.subplots(figsize=a4_dims,ncols=1)

sns.countplot(x="workclass",data=df)
print("percentage of Null values in workclass feature - {0:.2f}%".format((len((df[df['workclass']=='?']))/len(df['workclass']))*100))

print("Percentage of Null values in occupation feature - {0:.2f}%".format((len((df[df['occupation']=='?']))/len(df['occupation']))*100))

print("Percentage of Null values in Native-country -{0:.2f}%".format((len((df[df['native-country']=='?']))/len(df['native-country']))*100))
shape_with_nullvalues= df.shape

print("Shape before removing null values  {}".format(shape_with_nullvalues))



df = df[df['workclass'] != '?']

df = df[df['occupation'] != '?']

df = df[df['native-country'] != '?']



print('Shape after removing null values {}'.format(df.shape))

print("percentage of data removed - {0:.2f}%".format(((shape_with_nullvalues[0]-df.shape[0])/shape_with_nullvalues[0])*100))
a4_dims = (14, 8.7)

fig, ax = plt.subplots(figsize=a4_dims,ncols=2)

sns.countplot(x="workclass",hue='income',data=df,ax=ax[0])

sns.countplot(x="occupation",hue='income',data=df,ax=ax[1])
a4_dims = (14, 8)

fig, ax = plt.subplots(figsize=a4_dims,ncols=2)

sns.countplot(x="education",hue="income", data=df,ax=ax[0])

sns.countplot(x="race",hue="income", data=df,ax=ax[1])
category_col =['workclass', 'race', 'education','marital-status', 'occupation',

               'relationship', 'gender', 'native-country', 'income'] 

for c in category_col:

    print (c)

    print (df[c].value_counts())
import seaborn as sns

#age	fnlwgt	educational-num	capital-gain	capital-loss	hours-per-week

a4_dims = (14, 8)

fig, ax = plt.subplots(figsize=a4_dims,ncols=2)

sns.boxplot(x=df['age'],ax=ax[0])

sns.boxplot(x=df['fnlwgt'],ax=ax[1])
#converting income type into numerical data

encoder= LabelEncoder()

df['income'] = encoder.fit_transform(df['income'].astype('str'))

#Converting categorial data into numerical data using one-hot encoding 

df=df=pd.get_dummies(df)

print("Number of feautres after one-hot encoding {}".format(len(list(df))))
#Test Train split 

trainset,testset= train_test_split(df,test_size = 0.33)

trainlabel = trainset.pop('income')

testlabel = testset.pop('income')

print(trainset.shape)

print(testset.shape)
#Declearing Machine learnig model with different algorithms

DT = DecisionTreeClassifier()

NB = GaussianNB()

KNN=KNeighborsClassifier()

LR = LinearRegression()

predictions=dict()

DT.fit(trainset,trainlabel)

#Prediction 

prediction = DT.predict(testset)

accur = accuracy_score(testlabel,prediction) 

predictions['Decision Tree'] = accur

display("The accuracy score of Decision Tress is {}".format(accur))

display(confusion_matrix(testlabel,prediction))

skplt.metrics.plot_confusion_matrix(testlabel,prediction, normalize=True)

plt.show()

NB.fit(trainset,trainlabel)

#Prediction 

prediction = NB.predict(testset)

accur = accuracy_score(testlabel,prediction) 

predictions['Naive Bayes'] = accur

display("The accuracy of Naive Bayes is {}".format(accur))

display(confusion_matrix(testlabel,prediction))

skplt.metrics.plot_confusion_matrix(testlabel,prediction, normalize=True)

plt.show()
KNN.fit(trainset,trainlabel)

#Prediction 

prediction = KNN.predict(testset)

accur = accuracy_score(testlabel,prediction) 

predictions['KNN'] = accur

display("The accuracy of Naive Bayes is {}".format(accur))

display(confusion_matrix(testlabel,prediction))

skplt.metrics.plot_confusion_matrix(testlabel,prediction, normalize=True)

plt.show()
LR.fit(trainset,trainlabel)

#Prediction 

prediction = LR.predict(testset)

prediction[prediction > .5] = 1

prediction[prediction <=.5] = 0

accur = accuracy_score(testlabel,prediction) 

predictions['Linear Regression'] = accur

display("The accuracy of Naive Bayes is {}".format(accur))

display(confusion_matrix(testlabel,prediction))

skplt.metrics.plot_confusion_matrix(testlabel,prediction, normalize=True)

plt.show()
#final analysis

df=pd.DataFrame(list(predictions.items()),columns=['Algorithms','Percentage'])

display(df)

fig, (ax1) = plt.subplots(ncols=1, sharey=True,figsize=(15,5))

sns.pointplot(x="Algorithms", y="Percentage", data=df,ax=ax1);

plt.show()
df_train = trainset.values

df_test = testset.values

df_label = tf.one_hot(trainlabel,2)

print(df_train[0].shape)

print(df_label[0].shape)
def labelMaker(val):

    if val == 0:

        return [1, 0]

    elif val == 1:

        return [0, 1]

df_labels = trainlabel.values

df_labels = np.array([labelMaker(i) for i in trainlabel])

test_label = np.array([labelMaker(i) for i in testlabel])

print(df_labels[0].shape)

print(trainlabel[0].shape)

print(df_train[0].shape)
print(df_test.shape)

print(test_label.shape)
model = keras.Sequential([

    keras.layers.Flatten(input_shape=df_train[0].shape),

    keras.layers.Dense(52, activation=tf.nn.relu),

     keras.layers.Dense(22, activation=tf.nn.relu),

    keras.layers.Dense(2, activation=tf.nn.softmax)

])

model.compile(optimizer=tf.keras.optimizers.Adam(0.1), 

              loss='mean_squared_error',

              metrics=['accuracy'])
model.summary()
history = model.fit(df_train, df_labels, epochs=10,validation_data=(df_test, test_label))
test_loss, test_acc = model.evaluate(df_test, test_label)



print('Test accuracy:', test_acc)
import matplotlib.pyplot as plt



plt.xlabel('Epoch Number')

plt.ylabel('Loss Number')

plt.plot(history.history['loss'])

plt.show()