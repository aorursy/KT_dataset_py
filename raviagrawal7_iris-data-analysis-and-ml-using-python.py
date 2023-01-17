#importing the libraries

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

sns.set(color_codes=True)
#Importing the dataset



data = '../input/Iris.csv'

dataset = pd.read_csv(data)
#View of top 5 data row

dataset.head(5)
#Removing the Id column as unnecessary part

dataset = dataset.drop('Id',axis=1)
#Review the dataset without Id column

dataset.head(5)
#information about the data

dataset.info()
#class distribution of data

dataset.groupby('Species').size()
#description of data

dataset.describe()
#boxplot

plt.figure(figsize=(20,8)) #setting figure size

dataset.plot(kind='box',sharex=False,sharey=False)
#boxplot by each category of species

dataset.boxplot(by='Species',figsize=(12,8))
#detailed boxplot of sepal length

sns.set(style='darkgrid',palette='deep')

plt.figure(figsize=(12,6))

plt.title("Compare the distribution of Sepal Length")

sns.boxplot(x='Species',y= 'SepalLengthCm',data=dataset)

plt.show()
#detailed boxplot of sepal Width

sns.set(style='darkgrid',palette='deep')

plt.figure(figsize=(12,6))

plt.title("Compare the distribution of Sepal Width")

sns.boxplot(x='Species',y= 'SepalWidthCm',data=dataset)

plt.show()
#detailed boxplot of Petal length

sns.set(style='darkgrid',palette='deep')

plt.figure(figsize=(12,6))

plt.title("Compare the distribution of Petal Length")

sns.boxplot(x='Species',y= 'PetalLengthCm',data=dataset)

plt.show()
#detailed boxplot of petal width

sns.set(style='darkgrid',palette='deep')

plt.figure(figsize=(12,6))  #setting figure size

plt.title("Compare the distribution of Petal Width")

sns.boxplot(x='Species',y= 'PetalWidthCm',data=dataset)

plt.show()
#histograms 

dataset.hist(edgecolor='black',linewidth=2)
#pairplot 

#scatter plot for features and histogram with custom marker

sns.pairplot(dataset,hue='Species',markers=['o','s','D'],diag_kind='hist')



#removing the spines from the plot

sns.despine()



#show plot

plt.show()
#pairplot 

#scatter plot for features and kde with custom marker

sns.pairplot(dataset,hue='Species',markers=['o','s','D'],diag_kind='kde')



#removing the spines from the plot

sns.despine()



#show plot

plt.show()
#pairplot 

#scatter plot for features and regression with custom marker

sns.pairplot(dataset,hue='Species',markers=['o','s','D'],kind='reg')



#removing the spines from the plot

sns.despine()



#show plot

plt.show()
#swarmplot

plt.figure(figsize=(12,8))  #setting figsize



#melt the dataset

data1 = pd.melt(dataset,'Species',var_name='measurement')



#categorical scatterplot

sns.swarmplot(x='measurement',y='value',hue='Species',palette='deep',data=data1)



#removing the spines from the plot

sns.despine()



#show plot

plt.show()
#Violine Plot of PetalLength

plt.figure(figsize=(12,8)) #setting figuresize



#plotting violineplot

sns.violinplot(x='Species',y='PetalLengthCm',palette='deep',data=dataset)



#removing spines

sns.despine()



#show plot

plt.show()
#Violine Plot of PetalWidth

plt.figure(figsize=(12,8)) #setting figuresize



#plotting violineplot

sns.violinplot(x='Species',y='PetalWidthCm',palette='deep',data=dataset)



#removing spines

sns.despine()



#show plot

plt.show()
#Violine Plot of SepalLength

plt.figure(figsize=(12,8)) #setting figuresize



#plotting violineplot

sns.violinplot(x='Species',y='SepalLengthCm',palette='deep',data=dataset)



#removing spines

sns.despine()



#show plot

plt.show()
#Violine Plot of SepalWidth

plt.figure(figsize=(12,8)) #setting figuresize



#plotting violineplot

sns.violinplot(x='Species',y='SepalWidthCm',palette='deep',data=dataset)



#removing spines

sns.despine()



#show plot

plt.show()
#Creating dataset of dependent(X)and Independent(y)variables

X = dataset.iloc[:,:-1].values

y= dataset.iloc[:,-1].values



#Splitting dataset into training and test set

from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test =train_test_split(X,y,test_size=0.2,random_state=0)

#Logistics Regression

#importing the library

from sklearn.linear_model import LogisticRegression

#creating local variable classifier

classifier = LogisticRegression()

#Training the model

classifier.fit(X_train,y_train)



#predicting the value of Y

y_pred = classifier.predict(X_test)



#importing metrics for evaluation

from sklearn.metrics import confusion_matrix

from sklearn.metrics import classification_report



#summary of the model predicion

print(classification_report(y_test,y_pred))

print('Confusion Matrix:\n',confusion_matrix(y_test,y_pred))



#accuracy score of the model

from sklearn.metrics import accuracy_score

print('accuracy score :',accuracy_score(y_pred,y_test))

#K-Nearest Neighbour

#importing the library

from sklearn.neighbors import KNeighborsClassifier

#creating local variable classifier

classifier = KNeighborsClassifier(n_neighbors=8)

#Training the model

classifier.fit(X_train,y_train)



#predicting the value of Y

y_pred = classifier.predict(X_test)



#importing metrics for evaluation

from sklearn.metrics import confusion_matrix

from sklearn.metrics import classification_report



#summary of the model predicion

print(classification_report(y_test,y_pred))

print('Confusion Matrix:\n',confusion_matrix(y_test,y_pred))



#accuracy score of the model

from sklearn.metrics import accuracy_score

print('accuracy score :',accuracy_score(y_pred,y_test))
#Support Vector Machine(SVM)

#importing the library

from sklearn.svm import SVC

#creating local variable classifier

classifier = SVC(kernel='linear',random_state=0)

#Training the model

classifier.fit(X_train,y_train)



#predicting the value of Y

y_pred = classifier.predict(X_test)



#importing metrics for evaluation

from sklearn.metrics import confusion_matrix

from sklearn.metrics import classification_report



#summary of the model predicion

print(classification_report(y_test,y_pred))

print('Confusion Matrix:\n',confusion_matrix(y_test,y_pred))



#accuracy score of the model

from sklearn.metrics import accuracy_score

print('accuracy score :',accuracy_score(y_pred,y_test))
#Kernel SVM

#importing the library

from sklearn.svm import SVC

#creating local variable classifier

classifier = SVC(kernel='rbf',gamma ='scale',random_state=0)

#Training the model

classifier.fit(X_train,y_train)



#predicting the value of Y

y_pred = classifier.predict(X_test)



#importing metrics for evaluation

from sklearn.metrics import confusion_matrix

from sklearn.metrics import classification_report



#summary of the model predicion

print(classification_report(y_test,y_pred))

print('Confusion Matrix:\n',confusion_matrix(y_test,y_pred))



#accuracy score of the model

from sklearn.metrics import accuracy_score

print('accuracy score :',accuracy_score(y_pred,y_test))
#Naive Bayes

#importing the library

from sklearn.naive_bayes import GaussianNB

#creating local variable classifier

classifier = GaussianNB()

#Training the model

classifier.fit(X_train,y_train)



#predicting the value of Y

y_pred = classifier.predict(X_test)



#importing metrics for evaluation

from sklearn.metrics import confusion_matrix

from sklearn.metrics import classification_report



#summary of the model predicion

print(classification_report(y_test,y_pred))

print('Confusion Matrix:\n',confusion_matrix(y_test,y_pred))



#accuracy score of the model

from sklearn.metrics import accuracy_score

print('accuracy score :',accuracy_score(y_pred,y_test))
#Decision Tree Classifier

#importing the library

from sklearn.tree import DecisionTreeClassifier

#creating local variable classifier

classifier = DecisionTreeClassifier()

#Training the model

classifier.fit(X_train,y_train)



#predicting the value of Y

y_pred = classifier.predict(X_test)



#importing metrics for evaluation

from sklearn.metrics import confusion_matrix

from sklearn.metrics import classification_report



#summary of the model predicion

print(classification_report(y_test,y_pred))

print('Confusion Matrix:\n',confusion_matrix(y_test,y_pred))



#accuracy score of the model

from sklearn.metrics import accuracy_score

print('accuracy score :',accuracy_score(y_pred,y_test))