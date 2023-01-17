# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
data = pd.read_csv("../input/Iris.csv")
data.head(2)
data.describe()
data.drop('Id',axis=1, inplace= True)
data.info()
data['Species'].value_counts()
import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline


g = sns.pairplot(data, hue='Species')

plt.show()


g = sns.violinplot(y='Species', x='SepalLengthCm', data=data, inner='quartile')

plt.show()

g = sns.violinplot(y='Species', x='SepalWidthCm', data=data, inner='quartile')

plt.show()

g = sns.violinplot(y='Species', x='PetalLengthCm', data=data, inner='quartile')

plt.show()

g = sns.violinplot(y='Species', x='PetalWidthCm', data=data, inner='quartile')

plt.show()





sns.heatmap(data.corr(),cmap= 'cubehelix_r') #Always convert data to data.corr()
from sklearn.linear_model import LogisticRegression #Logistic Regression

from sklearn.model_selection import train_test_split #splitting test and train 

from sklearn.neighbors import KNeighborsClassifier # K nearest

from sklearn import svm # support vector machines

from sklearn import metrics # checking model accuracy

from sklearn.tree import DecisionTreeClassifier #Decision Tree
train, test = train_test_split(data, test_size = 0.3)# in this our main data is split into train and test

# the attribute test_size=0.3 splits the data into 70% and 30% ratio. train=70% and test=30%

print(train.shape)

print(test.shape)
train.head()
train_X = train.drop("Species",axis=1)

test_X = test.drop("Species",axis=1)

train_y = train["Species"]

test_y = test["Species"]
train_X.head(2)
test_X.head(2)
train_y.head(2)
test_y.head(2)
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

sc.fit(train_X)

train_X = sc.transform(train_X)

test_X = sc.transform(test_X)
pd.DataFrame(train_X).head(2) #So train_X is not a dataframe , we have to transform into dataframe. Good to learn that
model = svm.SVC()
model.fit(train_X,train_y)

prediction = model.predict(test_X)

print("The accuracy of SVM is ", metrics.accuracy_score(test_y,prediction))
model = LogisticRegression()
model.fit(train_X,train_y)

prediction = model.predict(test_X)

print("The accuracy of Logistic Regression is ", metrics.accuracy_score(test_y,prediction))

model = DecisionTreeClassifier()

model.fit(train_X,train_y)

prediction = model.predict(test_X)

print ( "The accuracy of Decision Tree Algorithm is  : ", metrics.accuracy_score(test_y,prediction))





model = KNeighborsClassifier(n_neighbors= 8)

model.fit(train_X,train_y)

prediction = model.predict(test_X)

print (" The accuracy of KNN is : " , metrics.accuracy_score(test_y,prediction))

a_index = list(range(1,20))

a = pd.Series()

for i in list(range(1,20)):

    model = KNeighborsClassifier(n_neighbors= i)

    model.fit(train_X,train_y)

    prediction = model.predict(test_X)

    a = a.append(pd.Series( metrics.accuracy_score(test_y,prediction)))

    

plt.plot(a_index,a)

plt.xticks(a_index)

data.head(2)

data.columns
Sepal = data[['SepalLengthCm', 'SepalWidthCm','Species']]

Petal = data[['PetalLengthCm', 'PetalWidthCm', 'Species']]
Sepal.head(2)
Petal.head(2)
sc = StandardScaler()

train_p,test_p = train_test_split(Petal,test_size = 0.3, random_state = 0)

train_x_p = train_p[['PetalLengthCm', 'PetalWidthCm']]

train_y_p = train_p[['Species']]

test_x_p = test_p[['PetalLengthCm', 'PetalWidthCm']]

test_y_p = test_p[['Species']]
train_s,test_s = train_test_split(Sepal,test_size = 0.3, random_state = 0)

train_x_s = train_s[['SepalLengthCm', 'SepalWidthCm']]

train_y_s = train_s[['Species']]

test_x_s = test_s[['SepalLengthCm', 'SepalWidthCm']]

test_y_s = test_s[['Species']]
sc.fit(train_x_p)

sc.fit(train_x_s)
train_x_p = sc.transform(train_x_p)

train_x_s = sc.transform(train_x_s)

test_x_p = sc.transform(test_x_p)

test_x_s = sc.transform(test_x_s)
model_p= svm.SVC()

model_s= svm.SVC()

model_p.fit(train_x_p,train_y_p)

prediction_p = model_p.predict(test_x_p)

model_s.fit(train_x_s,train_y_s)

prediction_s = model_s.predict(test_x_s)

print("The accuracy for Petals SVM is : ", metrics.accuracy_score(test_y_p,prediction_p))

print("The accuracy for Sepals SVM is : ", metrics.accuracy_score(test_y_s,prediction_s))

model_p= LogisticRegression()

model_s= LogisticRegression()

model_p.fit(train_x_p,train_y_p)

prediction_p = model_p.predict(test_x_p)

model_s.fit(train_x_s,train_y_s)

prediction_s = model_s.predict(test_x_s)

print("The accuracy for Petals Logistic Regression is : ", metrics.accuracy_score(test_y_p,prediction_p))

print("The accuracy for Sepals Logistic Regression is : ", metrics.accuracy_score(test_y_s,prediction_s))





model_p= DecisionTreeClassifier()

model_s= DecisionTreeClassifier()

model_p.fit(train_x_p,train_y_p)

prediction_p = model_p.predict(test_x_p)

model_s.fit(train_x_s,train_y_s)

prediction_s = model_s.predict(test_x_s)

print("The accuracy for Petals Logistic Regression is : ", metrics.accuracy_score(test_y_p,prediction_p))

print("The accuracy for Sepals Logistic Regression is : ", metrics.accuracy_score(test_y_s,prediction_s))





model_p= KNeighborsClassifier(n_neighbors=8)

model_s= KNeighborsClassifier(n_neighbors=8)

model_p.fit(train_x_p,train_y_p)

prediction_p = model_p.predict(test_x_p)

model_s.fit(train_x_s,train_y_s)

prediction_s = model_s.predict(test_x_s)

print("The accuracy for Petals KNN is : ", metrics.accuracy_score(test_y_p,prediction_p))

print("The accuracy for Sepals KNN is : ", metrics.accuracy_score(test_y_s,prediction_s))





pd.DataFrame(train_X).head(2)
from sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder()
train_y = encoder.fit_transform(train_y)

test_y = encoder.fit_transform(test_y)

train_y = pd.get_dummies(train_y).values

test_y = pd.get_dummies(test_y).values
# Model Creation

from keras.models import Sequential

from keras.layers import Dense

from keras.optimizers import SGD,Adam



model = Sequential()



model.add(Dense(10,input_shape=(4,),activation='relu'))

model.add(Dense(8,activation='relu'))

model.add(Dense(6,activation='relu'))

model.add(Dense(3,activation='softmax'))



model.compile(Adam(lr=0.04),'categorical_crossentropy',metrics=['accuracy'])



model.summary()
model.fit(train_X,train_y,epochs=30)

y_pred = model.predict(test_X)



y_test_class = np.argmax(test_y,axis=1)

y_pred_class = np.argmax(y_pred,axis=1)