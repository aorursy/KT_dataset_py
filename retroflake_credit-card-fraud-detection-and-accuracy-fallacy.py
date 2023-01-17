#Importing the basic libraries

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import plotly.offline as py

from plotly import tools

py.init_notebook_mode(connected=True)

import plotly.graph_objs as go

from sklearn.model_selection import train_test_split

import warnings

warnings.filterwarnings('ignore')
#Reading the dataset 

df = pd.read_csv('../input/creditcard.csv')

df.shape
#Let's check if there are any null values 

df.isnull().sum()
#Describe the dataset to get rough idea about the data 

df.describe()
#Well let's see all the columns in the dataset 

df.columns
#A brief look at the initial rows of the dataset 

df.head()
#It's good to shuffle the datset 

df = df.sample(frac=1, random_state=42).reset_index(drop=True)
#Let's take a look again 

df.head()
#Non fradulant cases

df.Class.value_counts()[0]
#Fradulant cases 

df.Class.value_counts()[1]
print('Percentage of correct transactions: {}'.format((df.Class.value_counts()[0]/df.shape[0])*100))
print('Percentage of fradulent transactions: {}'.format((df.Class.value_counts()[1]/df.shape[0])*100))
#Let's visualize the distribution of the classes (0 means safe and 1 means fraudulent)

import seaborn as sns

colors = ['green', 'red']



sns.countplot('Class', data=df, palette=colors)

plt.title('Normal v/s Fraudulent')
#Now let's map how much a feature affects our class 

cor = df.corr()

fig = plt.figure(figsize = (12, 9))



#Plotting the heatmap

sns.heatmap(cor, vmax = 0.7)

plt.show()
cor.shape
#This is how much a each feature affects the our class 

cor.iloc[-1,:]
#We need to delet the least and greatest values 

#From above analysis I've selected the following features 

#Note that I've included the class variable also because I intend to create a new dataframe using the new features

new_features=['V1','V3','V4','V7','V10','V11','V12','V14','V16','V17','V18','Class']
#Let's plot a heatmap again and see the relationship

cor = df[new_features].corr()

fig = plt.figure(figsize = (12, 9))



#Plotting the heatmap

sns.heatmap(cor, vmax = 0.7)

plt.show()
#We see that the class rows and columns are darker and brighter 

#This means that all the variables in our new dataset have a significant affect 
#Now splitting the dataset into the dependent variable(y) and independent variales(x)

x=df[new_features].iloc[:,:-1].values

y=df[new_features].iloc[:,-1].values
#Withoud reducing the features 

#x=df.iloc[:,:-1].values

#y=df.iloc[:,-1].values

#Feel free to try using all the features :)
x.shape
y.shape
x[:5]
y[:5]
#Spliting the data into train and test sets 

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.25, random_state = 0)

#Let's see how many safe and fraudulent cases are there in training set 

safe_train=(y_train==0).sum()

fraud_train=(y_train==1).sum()

print("Safe: {} \nFraud: {}".format(safe_train,fraud_train))
#Let's see how many safe and fraudulent cases are there in test set 

safe_test=(y_test==0).sum()

fraud_test=(y_test==1).sum()

print("Safe: {} \nFraud: {}".format(safe_test,fraud_test))
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix
#Using Logistic Regression 

clf = LogisticRegression(random_state = 0)

clf.fit(x_train, y_train)
#Let's evaluate our model 

y_pred = clf.predict(x_test)

print("Training Accuracy: ",clf.score(x_train, y_train))

print("Testing Accuracy: ", clf.score(x_test, y_test))

cm = confusion_matrix(y_test, y_pred)

print(cm)

print(classification_report(y_test,y_pred)) 
from sklearn.metrics import roc_curve, auc
#Calculating the FPR and TPR

fpr, tpr, thresholds = roc_curve(y_test, y_pred)

roc_auc = auc(fpr, tpr)
#Plotting the curves 

label = 'Logistic Regressoin Classifier AUC:' + ' {0:.2f}'.format(roc_auc)

label2 = 'Random Model' 

plt.figure(figsize = (20, 12))

plt.plot([0,1], [0,1], 'r--', label=label2)

plt.plot(fpr, tpr, c = 'g', label = label, linewidth = 4)

plt.xlabel('False Positive Rate', fontsize = 16)

plt.ylabel('True Positive Rate', fontsize = 16)

plt.title('Receiver Operating Characteristic', fontsize = 16)

plt.legend(loc = 'lower right', fontsize = 16)
#We see that AUC is 0.78 which is not bad, not great but fair enough
from sklearn.neighbors import KNeighborsClassifier

clf = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)

clf.fit(x_train, y_train)
y_pred = clf.predict(x_test)

print("Training Accuracy: ",clf.score(x_train, y_train))

print("Testing Accuracy: ", clf.score(x_test, y_test))

cm = confusion_matrix(y_test, y_pred)

print(cm)

print(classification_report(y_test,y_pred)) 
fpr, tpr, thresholds = roc_curve(y_test, y_pred)

roc_auc = auc(fpr, tpr)
label = 'KNN Classifier AUC:' + ' {0:.2f}'.format(roc_auc)

label2 = 'Random Model' 

plt.figure(figsize = (20, 12))

plt.plot([0,1], [0,1], 'r--', label=label2)

plt.plot(fpr, tpr, c = 'g', label = label, linewidth = 4)

plt.xlabel('False Positive Rate', fontsize = 16)

plt.ylabel('True Positive Rate', fontsize = 16)

plt.title('Receiver Operating Characteristic', fontsize = 16)

plt.legend(loc = 'lower right', fontsize = 16)
#We see that AUC is 0.90 which is good, with different random state of data you may get different AUC.

#I once achieved 0.94

#Though I've kept a constant seed for random state wile shuffling the dataset, feel free to mess with that :D
(df.Class.value_counts()[1]/df.Class.value_counts()[0])
#Importing and fitting the Isolation Forest Algorithm 

from sklearn.ensemble import IsolationForest

clf=IsolationForest(contamination=(df.Class.value_counts()[1]/df.shape[0]), random_state=123,max_features=x.shape[1])

clf.fit(x)
#Predicting the class

y_pred = clf.predict(x)
#Since the algorithm classifies one class as 1 and other as -1

#Let's see how many classes it predicted as fraudulent 

(y_pred==-1).sum()
#Since our class variables are either 0 or 1, so we need to replace the predicted classes as 0 and 1

y_pred[y_pred == 1] = 0

y_pred[y_pred == -1] = 1
#Let's see how good our model performed 

print("Training Accuracy: ",accuracy_score(y, y_pred))

cm = confusion_matrix(y, y_pred)

print(cm)

print(classification_report(y,y_pred))
fpr, tpr, thresholds = roc_curve(y, y_pred)

roc_auc = auc(fpr, tpr)
label = 'Isolation Forest Classifier AUC:' + ' {0:.2f}'.format(roc_auc)

label2 = 'Random Model' 

plt.figure(figsize = (20, 12))

plt.plot([0,1], [0,1], 'r--', label=label2)

plt.plot(fpr, tpr, c = 'g', label = label, linewidth = 4)

plt.xlabel('False Positive Rate', fontsize = 16)

plt.ylabel('True Positive Rate', fontsize = 16)

plt.title('Receiver Operating Characteristic', fontsize = 16)

plt.legend(loc = 'lower right', fontsize = 16)
#We get AUC score of 0.75, that's bad. Seems like this algorithm is not working very well on our dataset 
#Let's try another unsupervised algorithm 

from sklearn.neighbors import LocalOutlierFactor

clf=LocalOutlierFactor(n_neighbors=5,contamination=(df.Class.value_counts()[1]/df.shape[0]))
y_pred = clf.fit_predict(x)

y_pred[y_pred == 1] = 0

y_pred[y_pred == -1] = 1
print("Training Accuracy: ",accuracy_score(y, y_pred))

cm = confusion_matrix(y, y_pred)

print(cm)

print(classification_report(y,y_pred))
fpr, tpr, thresholds = roc_curve(y, y_pred)

roc_auc = auc(fpr, tpr)
label = 'KNN Classifier AUC:' + ' {0:.2f}'.format(roc_auc)

label2 = 'Random Model' 

plt.figure(figsize = (20, 12))

plt.plot([0,1], [0,1], 'r--', label=label2)

plt.plot(fpr, tpr, c = 'g', label = label, linewidth = 4)

plt.xlabel('False Positive Rate', fontsize = 16)

plt.ylabel('True Positive Rate', fontsize = 16)

plt.title('Receiver Operating Characteristic', fontsize = 16)

plt.legend(loc = 'lower right', fontsize = 16)
#WHAT!!!!? That's a garbage model with an AUC of 0.5.

#Moreover it made 0 true negatives and that's a teriible, terrible model