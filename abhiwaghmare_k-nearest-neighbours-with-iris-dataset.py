import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split

from sklearn.neighbors import KNeighborsClassifier

df = pd.read_csv('../input/iris/Iris.csv',index_col=0)
df.head()
df.describe
df.shape
x = df.drop(['Species'],axis=1)
y = df['Species']
df.isnull().sum()
fig = df[df.Species=='Iris-setosa'].plot(kind='scatter',x='SepalLengthCm',y='SepalWidthCm',color='orange', label='Setosa')

df[df.Species=='Iris-versicolor'].plot(kind='scatter',x='SepalLengthCm',y='SepalWidthCm',color='blue', label='versicolor',ax=fig)

df[df.Species=='Iris-virginica'].plot(kind='scatter',x='SepalLengthCm',y='SepalWidthCm',color='green', label='virginica', ax=fig)

fig.set_xlabel("Sepal Length")

fig.set_ylabel("Sepal Width")

fig.set_title("Sepal Length VS Width")

fig=plt.gcf()

fig.set_size_inches(10,6)

plt.show()
fig = df[df.Species=='Iris-setosa'].plot.scatter(x='PetalLengthCm',y='PetalWidthCm',color='orange', label='Setosa')

df[df.Species=='Iris-versicolor'].plot.scatter(x='PetalLengthCm',y='PetalWidthCm',color='blue', label='versicolor',ax=fig)

df[df.Species=='Iris-virginica'].plot.scatter(x='PetalLengthCm',y='PetalWidthCm',color='green', label='virginica', ax=fig)

fig.set_xlabel("Petal Length")

fig.set_ylabel("Petal Width")

fig.set_title(" Petal Length VS Width")

fig=plt.gcf()

fig.set_size_inches(10,6)

plt.show()
plt.figure(figsize=(15,10))

plt.subplot(2,2,1)

sns.violinplot(x='Species',y='PetalLengthCm',data=df)

plt.subplot(2,2,2)

sns.violinplot(x='Species',y='PetalWidthCm',data=df)

plt.subplot(2,2,3)

sns.violinplot(x='Species',y='SepalLengthCm',data=df)

plt.subplot(2,2,4)

sns.violinplot(x='Species',y='SepalWidthCm',data=df)
df.hist(edgecolor='black', linewidth=1.2)

fig=plt.gcf()

fig.set_size_inches(12,6)

plt.show()
sns.set(style="ticks", color_codes=True)

g = sns.pairplot(df,hue = "Species", size=3, markers=["o", "s", "D"])
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.4,random_state = 5)
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report

accu = []



for i in range(1,50):

    knn = KNeighborsClassifier(n_neighbors= i)

    knn.fit(x_train,y_train)

    pred_i = knn.predict(x_test)

    w = accuracy_score(y_test,pred_i)

    accu.append(w)

plt.figure(figsize=(10,6))

plt.plot(range(1,50),accu,marker='o',markersize=10,markerfacecolor ='red')
knn = KNeighborsClassifier(n_neighbors= 9)

knn.fit(x_train,y_train)

y_pred = knn.predict(x_test)
print(confusion_matrix(y_test,y_pred))

print(classification_report(y_test,y_pred))
knn = KNeighborsClassifier(n_neighbors= 3)

knn.fit(x_train,y_train)

y_pred = knn.predict(x_test)





print(confusion_matrix(y_test,y_pred))

print(classification_report(y_test,y_pred))
accuracy = accuracy_score(y_test,y_pred)*100

print('Accuracy of our model is equal ' + str(round(accuracy, 3)) + ' %.')