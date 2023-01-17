#Usual suspects

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

plt.style.use('fivethirtyeight')



#To ignore Warnings

import warnings

warnings.filterwarnings("ignore")
#Get the data

iris = pd.read_csv("/kaggle/input/iris/Iris.csv")
#To display top rows

iris.head()
#Shape of iris dataset

print ("Number of Observations :",iris.shape[0])

print ("Number of Features/Columns  :",iris.shape[1])
#Infos 

iris.info()
#Basic descriptive stats

iris.describe()
#Checking for any missing values

iris.isna().sum()
#Remove unnecessary feature

iris =iris.drop('Id',axis=1)
#Lets see final dataset shape

iris.shape
# Visualizing relationship between features :. 

sns.pairplot(iris,hue='Species',palette='Dark2');
#Dataset balance checking

sns.countplot(iris.Species);
#Import required ML models

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

from sklearn.ensemble import RandomForestClassifier



#Standard scaler

from sklearn.preprocessing import StandardScaler
#Separate X and Y varaibles

X = iris.iloc[:, 0:4].values

y = iris.iloc[:, 4].values



#train test split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
#Feature scaling

sc = StandardScaler()

X_train = sc.fit_transform(X_train)

X_test = sc.transform(X_test) 
#Implementing Lda

lda = LDA(n_components=1)

X_train_lda = lda.fit_transform(X_train, y_train)

X_test_lda = lda.transform(X_test)
#Lets check our dataset shape

print("Training data shape",X_train_lda.shape)
#Explaned variance ratio

lda.explained_variance_ratio_ * 100
#Random forest classifer

classifier = RandomForestClassifier(max_depth=2, random_state=0)



classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)
from sklearn.metrics import confusion_matrix,classification_report

from sklearn.metrics import accuracy_score



cm = confusion_matrix(y_test, y_pred)

print(cm)
#Seaborn heatmap for confusion matrix

sns.heatmap(cm,cmap='viridis',annot=True);

print('Accuracy of test data' ,accuracy_score(y_test, y_pred))
#Classification report

print(f"classification report :")

print("\n")

print(classification_report(y_test, y_pred))
from sklearn.decomposition import PCA

pca = PCA(n_components=1).fit(X_train)



#Fitting our model

X_train_pca = pca.transform(X_train)

x_test_pca = pca.transform(X_test)
#Lets check variance

print(f"Pca explained varaince ratio (n_comp=1) :{pca.explained_variance_ratio_ *100}")
#Make a model with RF

clf2 = RandomForestClassifier(max_depth=2,random_state=42)

clf2.fit(X_train,y_train)

pca_pred = clf2.predict(X_test)
##Confusion Matrix

cm_pca = confusion_matrix(y_test,pca_pred)

print(cm_pca)
##Let's visualize our CM

sns.heatmap(cm_pca,cmap='viridis',annot=True);

print('Accuracy of test data (PCA)' ,accuracy_score(y_test, pca_pred)*100)