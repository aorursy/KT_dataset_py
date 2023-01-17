#Import libraries

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt
#Reading the dataset

df=pd.read_csv('../input/diabetes2.csv')
#Check the first five values

df.head()
#Checking the null values..

df.isnull().sum()
#Checking the meta data

df.info()
#Doing some basic statistic

df.describe()
#Checking the unique values of dependent variable

df["Outcome"].unique()
#Check the dimensions

df.shape
#Ploting the headmap to check the correlation between all variables

import seaborn as sns

sns.heatmap(df)
#Ploting the histogram to check the distribution of each  columns 

p = df.hist(figsize = (20,20))
#Pairplot to visualize the correlation with one variable and all other variables respectively.

p=sns.pairplot(df, hue = 'Outcome')
#Checking the correlation using heatmap

plt.figure(figsize=(12,10))  # on this line I just set the size of figure to 12 by 10.

p=sns.heatmap(df.corr(), annot=True,cmap ='RdYlGn')  # seaborn has very simple solution for heatmap
#Shortcut to scale all independent variable in one go.... 

from sklearn.preprocessing import StandardScaler

sc_X = StandardScaler()

X =  pd.DataFrame(sc_X.fit_transform(df.drop(["Outcome"],axis = 1),),

        columns=['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin',

       'BMI', 'DiabetesPedigreeFunction', 'Age'])
X.head()


#Separating dataset into independent and dependent variables

X = df.iloc[:, 0:8].values

y = df.iloc[:, -1].values





#Splitting into training and testing dataset....

from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(X,y, test_size=0.3,random_state=0)
#Fitting Decisiontree into dataset

from sklearn.tree import DecisionTreeClassifier

#Creating a confusion matrix

from sklearn.metrics import confusion_matrix

#Check the accuracy

from sklearn.metrics import accuracy_score





dtree_c=DecisionTreeClassifier(criterion='entropy',random_state=0)

dtree_c.fit(X_train,y_train)

dtree_pred=dtree_c.predict(X_test)

dtree_cm=confusion_matrix(y_test,dtree_pred)

print("The accuracy of DecisionTreeClassifier is:",accuracy_score(dtree_pred,y_test))









print(dtree_cm)
#Fitting Randomforest into dataset

from sklearn.ensemble import RandomForestClassifier

rdf_c=RandomForestClassifier(n_estimators=10,criterion='entropy',random_state=0)

rdf_c.fit(X_train,y_train)

rdf_pred=rdf_c.predict(X_test)

rdf_cm=confusion_matrix(y_test,rdf_pred)

print("The accuracy of RandomForestClassifier is:",accuracy_score(rdf_pred,y_test))

print(rdf_cm)
#Fitting Logistic regression into dataset

from sklearn.linear_model import LogisticRegression

lr_c=LogisticRegression(random_state=0)

lr_c.fit(X_train,y_train)

lr_pred=lr_c.predict(X_test)

lr_cm=confusion_matrix(y_test,lr_pred)

print("The accuracy of  LogisticRegression is:",accuracy_score(y_test, lr_pred))

print(lr_cm)
#Fitting KNN into dataset

from sklearn.neighbors import KNeighborsClassifier



knn=KNeighborsClassifier(n_neighbors=2)

knn.fit(X_train,y_train)

knn_pred=knn.predict(X_test)

knn_cm=confusion_matrix(y_test,knn_pred)

print("The accuracy of KNeighborsClassifier is:",accuracy_score(knn_pred,y_test))





print(knn_cm)
#Fitting Naive bayes into dataset

from sklearn.naive_bayes import GaussianNB







gaussian=GaussianNB()

gaussian.fit(X_train,y_train)

bayes_pred=gaussian.predict(X_test)

bayes_cm=confusion_matrix(y_test,bayes_pred)

print("The accuracy of naives bayes is:",accuracy_score(bayes_pred,y_test))



print(bayes_cm)




#confusion matrix.....

plt.figure(figsize=(20,10))

plt.subplot(2,4,3)

plt.title("LogisticRegression_cm")

sns.heatmap(lr_cm,annot=True,cmap="prism",fmt="d",cbar=False)



plt.subplot(2,4,5)

plt.title("bayes_cm")

sns.heatmap(bayes_cm,annot=True,cmap="binary_r",fmt="d",cbar=False)

plt.subplot(2,4,2)

plt.title("RandomForest")

sns.heatmap(rdf_cm,annot=True,cmap="ocean_r",fmt="d",cbar=False)



plt.subplot(2,4,1)

plt.title("DecisionTree_cm")

sns.heatmap(dtree_cm,annot=True,cmap="twilight_shifted_r",fmt="d",cbar=False)

plt.subplot(2,4,4)

plt.title("kNN_cm")

sns.heatmap(knn_cm,annot=True,cmap="Wistia",fmt="d",cbar=False)

plt.show()
