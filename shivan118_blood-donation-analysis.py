# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

#Importing library for visualization

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline



#Importing all the required model for model comparision

from sklearn.ensemble import RandomForestClassifier



from sklearn.linear_model import LogisticRegression



from sklearn.tree import DecisionTreeClassifier



from sklearn.neural_network import MLPClassifier



from sklearn.svm import SVC



#Importing library for splitting model into train and test and for data transformation

from sklearn.model_selection import train_test_split



from sklearn.metrics import confusion_matrix,accuracy_score,roc_auc_score



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
#Filter the unwanted warning

import warnings

warnings.simplefilter("ignore")
path = '/kaggle/input/predicting-blood-analysis/blood-train.csv'

path1 = '/kaggle/input/predicting-blood-analysis/blood-test.csv'

blood = path + 'blood-train.csv'

blood1 = path1 + 'blood-test.csv'
train = pd.read_csv(path)

test = pd.read_csv(path1)
train.head()
#Printing the train and test size

print("Train Shape : ",train.shape)

print("Test Shape : ",test.shape)
#Printing first five rows of data

train.head()
#Counting the number of people who donated and not donated

train["Made Donation in March 2007"].value_counts()
#Storing dependent variable in Y

Y=train.iloc[:,-1]

Y.head()
#Printing last 5 rows

train.tail()
#Removing Unnamed: 0 columns

old_train=train

train=train.iloc[:,1:5]

test=test.iloc[:,1:5]
#Printing firsr  rows

train.head()
#Merging both train and test data

df=pd.merge(train,test)
df.head()
#Setting the independent variable and dependent variable

X=df.iloc[:,:]

X.head()
# Statistics of the data

train.describe()
#Boxplot for Months since Last Donation

plt.figure(figsize=(20,10)) 

sns.boxplot(y="Months since Last Donation",data=old_train)
#Correlation between all variables [Checking how different variable are related]

corrmat=X.corr()

f, ax = plt.subplots(figsize =(9, 8)) 

sns.heatmap(corrmat, ax = ax, cmap ="YlGnBu", linewidths = 0.1,fmt = ".2f",annot=True) 
#Printing all unique value for Month Since Last donation

train["Months since Last Donation"].unique()
#Creating new variable for calculating how many times a person have donated

X["Donating for"] = (X["Months since First Donation"] - X["Months since Last Donation"])
#Seeing first five rows of the DataFrame

X.head()
#Correlation between all variables

corrmat=X.corr()

f, ax = plt.subplots(figsize =(9, 8)) 

sns.heatmap(corrmat, ax = ax, cmap ="YlGnBu", linewidths = 0.1,fmt = ".2f",annot=True) 
#Dropping the unnecessary column

X.drop([ 'Total Volume Donated (c.c.)'], axis=1, inplace=True)
X.head()
#Shape of independent variable

X.shape
#Feature Scaling

from sklearn.preprocessing import StandardScaler

scale=StandardScaler()



#Fitting and transforming data

X=scale.fit_transform(X)
train=X[:576]
train.shape
test=X[576:]
Y=Y[:576]
Y.shape
#Splitting into train and test set

xtrain,xtest,ytrain,ytest=train_test_split(train,Y,test_size=0.2,random_state=0)
#Building the model

logreg = LogisticRegression(random_state=7)

#Fitting the model

logreg.fit(xtrain,ytrain)
#Predicting on the test data

pred=logreg.predict(xtest)
accuracy_score(pred,ytest)
#Printing the roc_auc_score

roc_auc_score(pred,ytest)
### SVC classifier

SVMC = SVC(probability=True)

#Fitting the model

SVMC.fit(train,Y)
#Predicting on the test data

pred=SVMC.predict(xtest)
accuracy_score(pred, ytest)
#Printing the confusion matrix

confusion_matrix(pred,ytest)
#Printing the roc auc score

roc_auc_score(pred,ytest)
#Buildin the model

RFC = RandomForestClassifier()

#Fitting the model

RFC.fit(xtrain,ytrain)
#Predicting the test data result

pred=RFC.predict(xtest)
#Printing the confusion matrix

confusion_matrix(pred,ytest)
accuracy_score(pred, ytest)
#Printingthe roc auc score

roc_auc_score(pred,ytest)
#Building the model

model=DecisionTreeClassifier(max_leaf_nodes=4,max_features=3,max_depth=15)
#Fitting the model

model.fit(xtrain,ytrain)
#Predicting the test data

pred=model.predict(xtest)
accuracy_score(pred, ytest)
#printing the confusion matrix

confusion_matrix(pred,ytest)
#Printing accuracy score

accuracy_score(pred,ytest)
#Printing roc auc score

roc_auc_score(pred,ytest)
#Building the Model

clf_neural = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(25,),random_state=1)

clf_neural.fit(train, Y)
#Predicting from the fitted model on test data

print('Predicting...\nIn Test Data')

predicted = clf_neural.predict(xtest)
#printing confusion matrix

confusion_matrix(predicted,ytest)
#Printing roc auc score

roc_auc_score(pred,ytest)
accuracy_score(pred, ytest)