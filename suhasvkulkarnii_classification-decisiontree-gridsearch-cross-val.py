# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df=pd.read_excel("/kaggle/input/bank-loan-modelling/Bank_Personal_Loan_Modelling.xlsx",'Data')
df.head(5)
df=df.drop(['ID'],axis=1)
df.head(5)
#Finding if their are any null values in the data set

df.isnull().sum()
#Finding if any duplicate values

df.duplicated().sum()
#Since all the features we have are of integer type we can describe the data

df.describe().transpose()
import seaborn as sns

import matplotlib.pyplot as plt
col=['Age', 'Experience', 'Income','CCAvg','Mortgage']

j=0

i=3

plt.figure(figsize=(14,12))

for k in col :

    plt.subplot(i,i,i*(j+1)//i)

    sns.distplot(df[k])

    j=j+1

plt.show()
# Replacing negative experience with median value of the Experience column

negexp=df[df['Experience']<0]
negexp['Experience'].value_counts()
negval=[-3,-2,-1]



for i in negval:

    df['Experience']=df['Experience'].replace(negval,np.median(df['Experience']))
df['Experience'].describe()
#Finding corelation between features

cor=df.corr()
plt.figure(figsize=(10,8))

plt.title("Correlation plot")

sns.heatmap(cor,annot=True)

plt.show()
plt.figure(figsize=(10,8))

plt.title("Scatter plot for Experience and Age")

sns.scatterplot(x='Age',y='Experience',hue='Personal Loan',data=df)

plt.show()
#Dropping Experience from the dataset

df=df.drop(['Experience'],axis=1)
df.columns
#Plotting scatterplot for multivariate variables

col=['Income','CCAvg','Mortgage']

plt.figure(figsize=(14,12))

j=3

k=0

for i in col :

    plt.subplot(1,j,j*(k+1)//j)

    sns.scatterplot(x='Age',y=i,hue='Personal Loan',data=df)

    k=k+1

plt.show()
#Plotting countplot for Categorical variables

col=['Securities Account', 'CD Account', 'Online',

       'CreditCard']

plt.figure(figsize=(14,12))

j=2

k=0

for i in col :

    plt.subplot(2,j,j*(k+1)//j)

    sns.countplot(x=i,hue='Personal Loan',data=df)

    k=k+1

    plt.grid(True)

plt.show()
df.columns
plt.figure(figsize=(8,6))

sns.boxplot(x='Family',y='Income',hue='Personal Loan',data=df)

plt.show()
plt.figure(figsize=(10,8))

sns.boxplot(x='Education',y='CCAvg',hue='Personal Loan',data=df)

plt.show()
df.columns
df=df.drop(['ZIP Code'],axis=1)
df.head(5)
#checking class balance for Personal loan

df['Personal Loan'].value_counts()

df1=df
df1['Personal Loan'].value_counts()
#Class label is having imbalance data so we will re-balance the class variable using upsample method

#splitting major and minor class data frames

df_majority=df[df['Personal Loan']==0]

df_minority=df[df['Personal Loan']==1]

print("Majority class shape {}".format(df_majority.shape))

print("Minority class shape {}".format(df_minority.shape))
from sklearn.utils import resample
#Upsampling

df_minority_upsample=resample(df_minority,n_samples=4520)
#Joining both dataframes

df=pd.concat([df_majority,df_minority_upsample])
df['Personal Loan'].value_counts()

#Seperating x and y variables

x=df.drop(['Personal Loan'],axis=1)

y=df['Personal Loan']
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3)
y_train.head(5)
from sklearn.tree import DecisionTreeClassifier

dt=DecisionTreeClassifier(criterion="gini")
#using entropy

dt_en=DecisionTreeClassifier(criterion="entropy")
dt.fit(x_train,y_train)
dt_en.fit(x_train,y_train)
#Predicting using gini criteria

y_pred_dt_gini=dt.predict(x_test)
#Predicting using entropy criteria

y_pred_dt_en=dt_en.predict(x_test)
#Checking accuracy of the model

from sklearn.metrics import accuracy_score ,confusion_matrix,classification_report
acc=accuracy_score(y_test,y_pred_dt_gini)

print(acc)
#Classification error

print(1-acc)
acc_en=accuracy_score(y_test,y_pred_dt_en)

print(acc_en)
#Model validaton for Gini

confusion_matrix(y_test,y_pred_dt_gini)

#Model validaton for Entropy

confusion_matrix(y_test,y_pred_dt_en)
print(classification_report(y_test,y_pred_dt_gini))
#Hypertuing Decision Tree

from sklearn.model_selection import GridSearchCV

dt_base=DecisionTreeClassifier()
parameters={'criterion': ['gini','entropy'],'max_depth' : np.arange(1,50),'min_samples_leaf': [1,2,5,10,13,15]}
grid=GridSearchCV(dt_base,parameters)
grid.fit(x_train,y_train)
best_dt=grid.best_params_

print(best_dt)
grid.best_score_
model=grid.best_estimator_
model.fit(x_train,y_train)
y_pred_best=model.predict(x_test)
#Cross validation using cross_val_score

from sklearn.model_selection import cross_val_score

a=cross_val_score(model, x, y, cv=10)

print(a)
accur = np.mean(a)

print(accur)
#Plotting tree

from sklearn import tree

plt.figure(figsize=(20,14))

tree.plot_tree(model)

plt.show()
#For imbalance dataset

x_imb=df1.drop(['Personal Loan'],axis=1)

y_imb=df1['Personal Loan']

x_train_imb,x_test_imb,y_train_imb,y_test_imb=train_test_split(x_imb,y_imb,test_size=0.3)
model.fit(x_train_imb,y_train_imb)
y_pred_imb=model.predict(x_test_imb)
accuracy_score(y_test_imb,y_pred_imb)
#Predict proabilities

probs = model.predict_proba(x_test)
probs = probs[:, 1]

print(probs)
from sklearn import metrics

auc = metrics.roc_auc_score(y_test, probs)

print(auc)

from sklearn import metrics

fpr, tpr, thresholds = metrics.roc_curve(y_test, probs)

print(fpr,tpr)
def plot_roc_curve(fpr, tpr):

    plt.plot(fpr, tpr, color='orange', label='ROC')

    plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')

    plt.xlabel('False Positive Rate')

    plt.ylabel('True Positive Rate')

    plt.title('Receiver Operating Characteristic (ROC) Curve')

    plt.legend()

    plt.show()
plot_roc_curve(fpr, tpr)