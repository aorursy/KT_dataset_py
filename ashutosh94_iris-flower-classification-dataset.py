import numpy as np
import pandas as pd
from sklearn import preprocessing,cross_validation
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
df = pd.read_csv("../input/Iris.csv")
df.head(3)
df.head(5)
df.drop(['Id'], axis=1 ,inplace=True)
df.info()
df['Species'].value_counts()
figure = df[df.Species == "Iris-versicolor"].plot(kind='scatter',x ='SepalLengthCm',y = 'SepalWidthCm' , label = 'Versicolor' , color = 'Blue')
df[df.Species == 'Iris-setosa'].plot(kind = 'scatter',x='SepalLengthCm' , y= 'SepalWidthCm' , label = 'Setosa' , color = 'Red' , ax = figure)
df[df.Species == 'Iris-virginica'].plot(kind='scatter' , x='SepalLengthCm' , y= 'SepalWidthCm' , label = 'Virginica' , color = 'Brown' , ax = figure)
figure.set_xlabel('Sepal Length')
figure.set_ylabel('Sepal Width')
figure.set_title("Sepal length vs sepal width")
plt.show()
fig = df[df.Species == "Iris-versicolor"].plot(kind='scatter',x='PetalLengthCm',y='PetalWidthCm',label='versicolor',color='Blue')
df[df.Species == 'Iris-setosa'].plot(kind='scatter',x='PetalLengthCm',y='PetalWidthCm',label='setosa',color='Red',ax=fig)
df[df.Species == 'Iris-virginica'].plot(kind='scatter' , x='PetalLengthCm' , y = 'PetalWidthCm',label='virginica',color='Brown',ax=fig)
fig.set_xlabel('PetalLengthCm')
fig.set_ylabel('PetalWidthCm')
fig.set_title('Petal Length vs Petal Width')
plt.show()
df.head(5)
y = df.Species
X = df[['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm']]
X_train,X_test,y_train,y_test = cross_validation.train_test_split(X,y,test_size=0.25)
X_train.tail(5)
X_train.shape
df.tail(5)
logreg = LogisticRegression()
logreg.fit(X_train,y_train)
accuracy = logreg.score(X_test,y_test)
print(accuracy)
clf = KNeighborsClassifier(6)
clf.fit(X_train,y_train)
accuracy = clf.score(X_test,y_test)
print(accuracy)
ex = np.array([8.7 , 6.5 , 3.5 , 4.6])
ex = ex.reshape(1,-1)
prediction = logreg.predict(ex)
print (prediction)
classifier = SVC()
classifier.fit(X_train,y_train)
accuracy = classifier.score(X_test,y_test)
print(accuracy)