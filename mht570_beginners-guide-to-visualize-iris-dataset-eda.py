import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np 

import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

%matplotlib inline
df=pd.read_csv('../input/iris/Iris.csv')
df.head()
image=plt.imread('../input/irisimage/iris-machinelearning.png')
plt.figure(figsize=(15,10))
plt.imshow(image)
plt.axis('off')
df=df.drop('Id',axis=1)
df.describe()
from pandas.tools.plotting import parallel_coordinates
plt.figure(figsize=(12,8))
parallel_coordinates(df, 'Species', colormap=plt.get_cmap("Set3"))
plt.xlabel("Features of data set")
plt.ylabel("cm")
plt.show()

sns.pairplot(df,hue='Species')
plt.figure(figsize=(15,10))
plt.subplot(2,2,1)
sns.swarmplot(x='Species',y='PetalLengthCm',data=df)
plt.subplot(2,2,2)
sns.swarmplot(x='Species',y='PetalWidthCm',data=df)
plt.subplot(2,2,3)
sns.swarmplot(x='Species',y='SepalLengthCm',data=df)
plt.subplot(2,2,4)
sns.swarmplot(x='Species',y='SepalWidthCm',data=df)

plt.figure(figsize=(15,10))
plt.subplot(2,2,1)
sns.violinplot(x='Species',y='PetalLengthCm',data=df)
plt.subplot(2,2,2)
sns.violinplot(x='Species',y='PetalWidthCm',data=df)
plt.subplot(2,2,3)
sns.violinplot(x='Species',y='SepalLengthCm',data=df)
plt.subplot(2,2,4)
sns.violinplot(x='Species',y='SepalWidthCm',data=df)
plt.figure(figsize=(15,10))
plt.subplot(2,2,1)
sns.boxplot(x='Species',y='PetalLengthCm',data=df)
plt.subplot(2,2,2)
sns.boxplot(x='Species',y='PetalWidthCm',data=df)
plt.subplot(2,2,3)
sns.boxplot(x='Species',y='SepalLengthCm',data=df)
plt.subplot(2,2,4)
sns.boxplot(x='Species',y='SepalWidthCm',data=df)

x=df.iloc[:,:-1].values
y=df.iloc[:,-1].values

x=(x-x.mean(0))/x.std(0)
x_train, x_test, y_train, y_test = train_test_split( x, y, test_size=0.33, random_state=42)
from sklearn.preprocessing import StandardScaler,LabelEncoder

sc = StandardScaler()
sc.fit(x_train)
x_train = sc.transform(x_train)
x_test = sc.transform(x_test)

lb=LabelEncoder()
y_train=lb.fit_transform(y_train)
y_test=lb.fit_transform(y_test)

model = LogisticRegression()
model.fit(x_train,y_train)
prediction=model.predict(x_test)
print('Accuracy on iris dataset using Logistic Regression is',metrics.accuracy_score(prediction,y_test))
model = SVC() 
model.fit(x_train,y_train) 
prediction=model.predict(x_test)
print('The accuracy of the SVM is:',metrics.accuracy_score(prediction,y_test))
model=DecisionTreeClassifier()
model.fit(x_train,y_train)
prediction=model.predict(x_test)
print('The accuracy of the Decision Tree is',metrics.accuracy_score(prediction,y_test))
# finding the value of k when the  accuray of knn would be highest
x_range=[]
y_range=[]
for i in range(1,20):
    model=KNeighborsClassifier(n_neighbors=i) 
    model.fit(x_train,y_train)
    prediction=model.predict(x_test)
    y_range.append((metrics.accuracy_score(prediction,y_test)))
    x_range.append(i)
plt.figure(figsize=(8,5))    
plt.plot(x_range, y_range)
plt.xlabel('values of n_neighbors')
plt.ylabel('Accuracy')
plt.show()
model=KNeighborsClassifier(n_neighbors=5)
model.fit(x_train,y_train)
prediction=model.predict(x_test)
print('The accuracy of the KNN is',metrics.accuracy_score(prediction,y_test))