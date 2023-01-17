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
#Find Null values in the data set:

df.isnull().sum()
#Finding of the duplicate values:

df.duplicated().sum()
#Since all the features in the data set are numerical hence describing the data:

df.describe().transpose()
import seaborn as sns

import matplotlib.pyplot as plt
df.columns
col=['Age', 'Experience', 'Income', 'CCAvg','Mortgage']



i=3

j=0

plt.figure(figsize=(14,12))

for k in col :

    plt.subplot(i,i,i*(j+1)//i)

    sns.distplot(df[k])

    j=j+1

plt.show()
# Replacing negative experience values with the median value in the Experience column:

negexp=df[df['Experience']<0]
negexp['Experience'].value_counts()
negval=[-3, -2, -1]



for i in negval:

    df['Experience']=df['Experience'].replace(negval,np.median(df['Experience']))
df['Experience'].describe()
# Finding Corelation between the features:

cor=df.corr()
# Heatmap for Corelation:

plt.figure(figsize=(10,8))

plt.title("Corelation Plot")

sns.heatmap(cor,annot=True)

plt.show()
plt.figure(figsize=(10,8))

plt.title("Scatter plot for Experience & Age")

sns.scatterplot(x='Age',y='Experience', hue='Personal Loan', data=df)

plt.show()
df=df.drop(['Experience'],axis=1)
# Plotting Scatter plot for multivariate features:

col=['Income','CCAvg','Mortgage']

plt.figure(figsize=(14,12))

j=3

k=0

for i in col:

    plt.subplot(1,j,j*(k+1)//j)

    sns.scatterplot(x='Age',y=i,hue='Personal Loan', data=df)

    k=k+1

plt.show()
# Plotting Counts plot for Categorical features:

col=['Securities Account','CD Account','Online','CreditCard']

plt.figure(figsize=(14,12))

j=2

k=0

for i in col:

    plt.subplot(2,j,j*(k+1)//j)

    sns.countplot(x=i,hue='Personal Loan', data=df)

    k=k+1

    plt.grid(True)

plt.show()
df.columns
plt.figure(figsize=(9,7))

sns.boxplot(x='Family',y='Income',hue='Personal Loan', data=df)

plt.show()
plt.figure(figsize=(12,10))

sns.boxplot(x='Education',y='CCAvg',hue='Personal Loan', data=df)

plt.show()
df.columns
df=df.drop(['ZIP Code'],axis=1)
df1=df
df1['Personal Loan'].value_counts()
df.head(5)
# Checking class balance for Personal Loan:

df['Personal Loan'].value_counts()
# Class label has imbalanced data, so this feature needs to be re-balanced using upsample method:

# Splitting major & minor class data frames:

df_majority=df[df['Personal Loan']==0]

df_minority=df[df['Personal Loan']==1]
print("Majority calss shape {}".format(df_majority.shape))

print("Minority calss shape {}".format(df_minority.shape))
# Upsampling:

from sklearn.utils import resample

df_minority_upsample=resample(df_minority,n_samples=4520)
df=pd.concat([df_majority,df_minority_upsample])
df['Personal Loan'].value_counts()
# Model Building:

x=df.drop(['Personal Loan'],axis=1)

y=df['Personal Loan']
# Splitting of Data:

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3)
# Decision Tree Model Prediction

from sklearn.tree import DecisionTreeClassifier

dt=DecisionTreeClassifier()
dt.fit(x_train,y_train)
y_pred_base=dt.predict(x_test)
# Finding Accuracy:

from sklearn.metrics import accuracy_score

acc=accuracy_score(y_test,y_pred_base)

print(acc)
# Model validation:

from sklearn.metrics import confusion_matrix,classification_report

confusion_matrix(y_test,y_pred_base)
#Classification Report:

clf_report=classification_report(y_test,y_pred_base)

print(clf_report)
# Hyper Parameter Tuning:

from sklearn.model_selection import GridSearchCV

parameters={'criterion':['gini','entropy'],'max_depth':np.arange(1,50),'min_samples_leaf':[1,2,3,6,9,4]}

grid=GridSearchCV(dt,parameters)
model=grid.fit(x_train,y_train)
grid.best_score_
grid.best_params_
clf_best=grid.best_estimator_
clf_best.fit(x_train,y_train)
y_pred_best=clf_best.predict(x_test)
accuracy_score(y_test,y_pred_best)
# Cross Validation:

from sklearn.model_selection import cross_val_score
cross_val=cross_val_score(clf_best,x,y,cv=10)

print(cross_val)
np.mean(cross_val)
# Visualizg the Tree:

from sklearn import tree

plt.figure(figsize=(16,14))

tree.plot_tree(clf_best)

plt.show()
# For the imbalance data set:

x_imbal=df1.drop(['Personal Loan'],axis=1)

y_imbal=df1['Personal Loan']
x_train_imbal,x_test_imbal,y_train_imbal,y_test_imbal=train_test_split(x_imbal,y_imbal,test_size=0.3)
clf_best.fit(x_train_imbal,y_train_imbal)
y_pred_imbal=clf_best.predict(x_test_imbal)
accuracy_score(y_test_imbal,y_pred_imbal)