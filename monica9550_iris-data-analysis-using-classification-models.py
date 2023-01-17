

import numpy as np

import matplotlib.pyplot as plt

import pandas as pd

import seaborn as sns

sns.set(color_codes=True)

plt.style.use('fivethirtyeight')  

import warnings

warnings.filterwarnings('ignore')  #this will ignore the warnings.it wont display warnings in notebook

dataset=pd.read_csv('../input/Iris.csv')
dataset.head()
#Dropping Id Coloumn

dataset=dataset.drop('Id',axis=1)

dataset.head()
#dimension

print(dataset.shape)  

#checking if any null object is present or not

print(dataset.info())   

# Statistical summary

print(dataset.describe()) 
#class distribution

print(dataset.groupby('Species').size())
#Statistical Summary of the data

dataset.describe().plot(kind = "area",fontsize=27, figsize = (20,8), table = True,colormap="rainbow")

plt.xlabel('Statistics',)

plt.ylabel('Value')

plt.title("General Statistics of Iris Dataset")
## Box and Whisker plots

dataset.plot(kind='box', sharex=False, sharey=False)
# Boxplot on each feature split out by species

dataset.boxplot(by="Species", figsize=(12, 6))
# Histograms

dataset.hist(edgecolor='red', linewidth=1.4)
#STRIP PLOT

fig=plt.gcf()

fig.set_size_inches(10,7)

fig=sns.stripplot(x='Species',y='SepalLengthCm',data=dataset,jitter=True,edgecolor='gray',size=8,palette='winter',orient='v')
#Bar plot

sns.countplot('Species',data=dataset)

plt.show()
#pie plot

dataset['Species'].value_counts().plot.pie(explode=[0.1,0.1,0.1],autopct='%1.1f%%',shadow=True,figsize=(10,8))

plt.show()
#HEATMAP

fig=plt.gcf()

fig.set_size_inches(10,7)

fig=sns.heatmap(dataset.corr(),annot=True,cmap='cubehelix',linewidths=1,linecolor='k',square=True,mask=False, vmin=-1, vmax=1,cbar_kws={"orientation": "vertical"},cbar=True)
#SCATTERPPLOT(or pair plot)

sns.pairplot(dataset,hue='Species')
#VIOLIN PLOT

plt.figure(figsize=(15,10))

plt.subplot(2,2,1)

sns.violinplot(x='Species',y='SepalLengthCm',data=dataset)

plt.subplot(2,2,2)

sns.violinplot(x='Species',y='PetalWidthCm',data=dataset)

plt.subplot(2,2,3)

sns.violinplot(x='Species',y='PetalLengthCm',data=dataset)

plt.subplot(2,2,4)

sns.violinplot(x='Species',y='SepalWidthCm',data=dataset)


#Separating dataset into independent and dependent variables

X = dataset.iloc[:, :-1].values

Y= dataset.iloc[:, -1].values
# Splitting the dataset into training set and test set

from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test=train_test_split(X,Y,test_size=0.2,random_state=0)
# Feature Scaling

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

X_train = sc.fit_transform(X_train)

X_test = sc.transform(X_test)
#KNN MODEL



#FITTING CLASSIFIER TO THE TRAINING SET

from sklearn.neighbors import KNeighborsClassifier

classifier=KNeighborsClassifier(n_neighbors=5,metric='minkowski', p=2) 

classifier.fit(X_train,Y_train)

#predicting the test set result

Y_pred=classifier.predict(X_test)

#Making the confusion matrix

from sklearn.metrics import confusion_matrix    

cm=confusion_matrix(Y_test,Y_pred)               

# Accuracy score

from sklearn.metrics import accuracy_score

print('Accuracy is',accuracy_score(Y_pred,Y_test))



#Logistics Regression

#Fitting the logistic regression to the dataset

from sklearn.linear_model import LogisticRegression

classifier=LogisticRegression(random_state= 0)

classifier.fit(X_train,Y_train)

#predicting the test set result

Y_pred=classifier.predict(X_test)

#Making the confusion matrix

from sklearn.metrics import confusion_matrix     

cm=confusion_matrix(Y_test,Y_pred)              

# Accuracy score

from sklearn.metrics import accuracy_score

print('Accuracy is',accuracy_score(Y_pred,Y_test))
#SVM



# Fitting classifier to the Training set

# Create your classifier here

from sklearn.svm import SVC

classifier=SVC(kernel='linear',random_state=0)

classifier.fit(X_train,Y_train)

# Predicting the Test set results

Y_pred = classifier.predict(X_test)

# Making the Confusion Matrix

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(Y_test, Y_pred)

# Accuracy score

from sklearn.metrics import accuracy_score

print('Accuracy is',accuracy_score(Y_pred,Y_test))


#Kernel SVM

# Fitting classifier to the Training set

# Create your classifier here

from sklearn.svm import SVC

classifier=SVC(kernel='rbf',random_state=0)

classifier.fit(X_train,Y_train)

# Predicting the Test set results

Y_pred = classifier.predict(X_test)

# Making the Confusion Matrix

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(Y_test, Y_pred)

# Accuracy score

from sklearn.metrics import accuracy_score

print('Accuracy is',accuracy_score(Y_pred,Y_test))
#Naive Bayes

# Fitting classifier to the Training set

# Create your classifier here

from sklearn.naive_bayes import GaussianNB

classifier=GaussianNB()

classifier.fit(X_train,Y_train)

# Predicting the Test set results

Y_pred = classifier.predict(X_test)

# Making the Confusion Matrix

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(Y_test, Y_pred)

# Accuracy score

from sklearn.metrics import accuracy_score

print('Accuracy is',accuracy_score(Y_pred,Y_test))
#Decision Tree



# Fitting Decision Tree Classification to the Training set

from sklearn.tree import DecisionTreeClassifier

classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)

classifier.fit(X_train, Y_train)

# Predicting the Test set results

Y_pred = classifier.predict(X_test)

# Making the Confusion Matrix

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(Y_test, Y_pred)

# Accuracy score

from sklearn.metrics import accuracy_score

print('Accuracy is',accuracy_score(Y_pred,Y_test))


#Random forest

# Fitting classifier to the Training set

# Create your classifier here

from sklearn.ensemble import RandomForestClassifier

classifier=RandomForestClassifier(n_estimators=10,criterion='entropy', random_state=0)

classifier.fit(X_train,Y_train)

# Predicting the Test set results

Y_pred = classifier.predict(X_test)

# Making the Confusion Matrix

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(Y_test, Y_pred)

# Accuracy score

from sklearn.metrics import accuracy_score

print('Accuracy is',accuracy_score(Y_pred,Y_test))