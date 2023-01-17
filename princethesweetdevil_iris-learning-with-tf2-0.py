!ls /kaggle/input/iris-flower-dataset/
#Import Part

#All Import Statements Will Come Here

import pandas as pd

import matplotlib.pyplot as plt

import numpy as np

import seaborn as sns
data = pd.read_csv("/kaggle/input/iris-flower-dataset/IRIS.csv")

print(data)
#print(list(data))

names = ["Iris-setosa","Iris-virginica","Iris-versicolor"]

counts = []

for x in names:

    counts.append(data["species"][data["species"]==x].count())

plt.figure()

sns.barplot(names,counts)

plt.show()
plt.figure()

plt.pie(counts,labels=names,)

plt.legend()

plt.show()
print(data["sepal_length"])

print(data["sepal_width"])
plt.figure()

plt.scatter(data["sepal_length"][data["species"]=="Iris-setosa"],data["sepal_width"][data["species"]=="Iris-setosa"],label='Setosa')

plt.scatter(data["sepal_length"][data["species"]=="Iris-virginica"],data["sepal_width"][data["species"]=="Iris-virginica"],label='Virginica')

plt.scatter(data["sepal_length"][data["species"]=="Iris-versicolor"],data["sepal_width"][data["species"]=="Iris-versicolor"],label='Versicolor')

plt.legend()

plt.show()
plt.figure()

plt.scatter(data["petal_length"][data["species"]=="Iris-setosa"],data["petal_width"][data["species"]=="Iris-setosa"],label='Setosa')

plt.scatter(data["petal_length"][data["species"]=="Iris-virginica"],data["petal_width"][data["species"]=="Iris-virginica"],label='Virginica')

plt.scatter(data["petal_length"][data["species"]=="Iris-versicolor"],data["petal_width"][data["species"]=="Iris-versicolor"],label='Versicolor')

plt.legend()

plt.show()
from sklearn.preprocessing import OneHotEncoder



Data_Array = np.array(data.values)

X = Data_Array[:,:-1]

X = X.astype(np.float)

Y = Data_Array[:,-1]



encode = OneHotEncoder()

Y = encode.fit_transform(Y.reshape(-1,1)).toarray()

print(Y[0])

print(Y[51])

print(Y[101])
from sklearn.model_selection import train_test_split

XTrain,XTest,YTrain,YTest = train_test_split(X,Y,random_state=10)

print(XTrain.shape,YTrain.shape,XTest.shape,YTest.shape)
#Deep Learning Library Tensorflow 2.X

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense,Dropout
np.random.seed(8)

Model = Sequential([

    Dense(4,input_dim=4,activation='relu'),

    Dense(20,activation='relu'),

    Dense(3,activation='softmax')

])



Model.compile(loss="categorical_crossentropy",optimizer='sgd',metrics=['accuracy'])
History = Model.fit(XTrain,YTrain,epochs=1000,batch_size=5,use_multiprocessing=True,validation_data=(XTest,YTest),verbose=0)
import matplotlib.pyplot as pl

pl.figure()

pl.plot(History.history['loss'])

pl.show()

pl.plot(History.history['accuracy'])

pl.show()
Loss,Accuracy=Model.evaluate(XTest,YTest,verbose=0)

print("Final Loss Is ",Loss)

print("Final Accuracy Is ",Accuracy)

right,wrong = 0,0

for x in range(len(Y)):

    if Y[x].argmax(-1) == Model.predict(X[x].reshape(1,4)).argmax(-1)[0]:

        right+=1

    else:

        wrong+=1

print('Right Answers : ',right)

print('Wrong Answers : ',wrong)

print('Total Answers : ',right+wrong)       