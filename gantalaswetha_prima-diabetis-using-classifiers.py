############## DIABETIES DATASET #################
#####importing libraries

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

from sklearn.linear_model import LinearRegression

from sklearn.linear_model import LogisticRegression



import statsmodels.formula.api as smf

from scipy.stats import shapiro,levene
########## importing the data ########################

data=pd.read_csv('../input/diabetes.csv')

#checking the head of the data

data.head()
#describing the data

data.describe()
#getting the information regarding the data

data.info()
#checking the shape of the data

data.shape
#checking for null values in the data and performing EDA

data.isnull().sum()
#getting the individua count of the outcome yes or no in the dataset

data['Outcome'].value_counts()
#dropping the outcome in the x and considering it in y as y is the target variable

x=data.drop('Outcome',axis=1)

x.head()

y=data['Outcome']

y.head()
#### plotting a HISTOGRAM on the data

data.hist(figsize=(10,8))

plt.show()
#### BOXPLOT for checking the outliers

data.plot(kind= 'box' , subplots=True,layout=(3,3), sharex=False, sharey=False, figsize=(10,8))
#### checking the correlation in matrix for variables using HEATMAP

import seaborn as sns

sns.heatmap(data.corr(), annot = True)
X=data.iloc[:,:-1]

X.head()

Y=data.iloc[:,-1]

Y.head()
#### splitting X and y into training and testing sets 

from sklearn.model_selection import train_test_split 

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=100)
# Scaling the data

from sklearn.preprocessing import StandardScaler

sc=StandardScaler()

X_train = sc.fit_transform(X_train)

X_test = sc.transform(X_test)
#logistic regression model

model=LogisticRegression()

model.fit(X_train,y_train)

ypred=model.predict(X_test)

ypred
# accuracy score

from sklearn import metrics

accuracy=metrics.accuracy_score(y_test,ypred)

print(accuracy)
#confusion matrix

cm=metrics.confusion_matrix(y_test,ypred)

print(cm)

plt.imshow(cm, cmap='binary')
#sensitivity and specificity check

tpr=cm[1,1]/cm[1,:].sum()

print(tpr*100)

tnr=cm[0,0]/cm[0,:].sum()

print(tnr*100)
#checking roc and auc curves

from sklearn.metrics import roc_curve,auc

fpr,tpr,_=roc_curve(y_test,ypred)

roc_auc=auc(fpr,tpr)

print(roc_auc)

plt.figure()

plt.plot(fpr,tpr)

plt.xlim([0.0,1.0])

plt.ylim([0.0,1.0])

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.title('Receiver Operating Characteristic')

plt.show()
#### importing the classifier and building the model

from sklearn import tree



model = tree.DecisionTreeClassifier()
model.fit(X_train, y_train)
y_predict = model.predict(X_test)



from sklearn.metrics import accuracy_score



accuracy_score(y_test, y_predict)