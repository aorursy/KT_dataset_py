# this diabetes model deployed into docker if you want end to end code mentioned below link



#https://github.com/satyamuralidhar/ML-Ops_ModelDeployment_k8s
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
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns
df = pd.read_csv('/kaggle/input/pima-indians-diabetes-database/diabetes.csv')
df.head()
df.plot()
df.isnull().sum()
plt.figure(figsize=(20,25),facecolor='white')

plotnumber = 1

for column in df:

    if plotnumber <= 9:

        ax = plt.subplot(3,3,plotnumber)

        sns.distplot(df[column])

        plt.xlabel(column,fontsize=20)

    plotnumber += 1

plt.show()
df.describe()
#replace null values with mean 

df['Pregnancies'] = df['Pregnancies'].replace(0,df['Pregnancies'].mean())

df['Glucose'] = df['Glucose'].replace(0,df['Glucose'].mean())

df['BloodPressure'] = df['BloodPressure'].replace(0,df['BloodPressure'].mean())

df['SkinThickness'] = df['SkinThickness'].replace(0,df['SkinThickness'].mean())

df['Insulin'] = df['Insulin'].replace(0,df['Insulin'].mean())

df['BMI'] = df['BMI'].replace(0,df['BMI'].mean())
df.describe()
plt.figure(figsize=(20,25),facecolor='white')

plotnumber = 1

for column  in df:

    if plotnumber <=9:

        plt.subplot(3,3,plotnumber)

        sns.distplot(df[column])

        plt.xlabel(column,fontsize=20)

    plotnumber += 1

plt.show()
X = df.drop(columns = 'Outcome')

y = df['Outcome']
pd.crosstab(df.Age,df.BloodPressure).plot(kind="bar",figsize=(20,6))

plt.title('BP Frequency with respective Ages')

plt.xlabel('Age')

plt.ylabel('BP')

#plt.savefig('BP with repective Ages.png')

plt.show()
# plt.figure(figsize = (10,15),facecolor='white')

# plotnumber = 1

# for column in df:

#     if plotnumber <=9:

#         ax = plt.subplot(3,3,plotnumber)

#         sns.boxplot(df[column])

#         plt.xlabel(column,fontsize=20)

#     plotnumber += 1

# plt.show()
#finding outliers

fig,ax = plt.subplots(figsize=(15,10),facecolor='white')

sns.boxplot(data = df , ax = ax ,width = 0.5 , fliersize = 3)
#we are removing 2% of data from pregnencies

q = df["Pregnancies"].quantile(0.98)

data_cleaned = df[df['Pregnancies']<q]

#we are removing 3% of data from BloodPressure

q = df['BloodPressure'].quantile(0.97)

data_cleaned = df[df['BloodPressure']<q]

#we are removing 3% of data from SkinThickness

q = df['SkinThickness'].quantile(0.97)

data_cleaned = df[df['SkinThickness']<q]

#we are removing 6% of data from Insulin

q = df['Insulin'].quantile(0.94)

data_cleaned = df[df['Insulin']<q]

#we are removing 3% of data from BMI

q = df['BMI'].quantile(0.97)

data_cleaned = df[df['BMI']<q]

#we are removing 1% of data from DiabetesPedigreeFunction

q = df['DiabetesPedigreeFunction'].quantile(0.99)

data_cleaned = df[df['DiabetesPedigreeFunction']<q]

#we are removing 2% of data from Age

q = df['Age'].quantile(0.98)

data_cleaned = df[df['Age']<q]

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)
from statsmodels.stats.outliers_influence import variance_inflation_factor

vif = pd.DataFrame()

vif['vif'] = [variance_inflation_factor(X_scaled,i)for i in range(X_scaled.shape[1])]

vif['Features'] = X.columns
vif
# creating a model

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import roc_curve , roc_auc_score , confusion_matrix , accuracy_score

from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(X_scaled,y,test_size=0.4,random_state=120)

rf = RandomForestClassifier()

rf.fit(X_train,y_train)

y_pred=rf.predict(X_test)

accuracy_score(y_pred,y_test)

#auc score

auc = roc_auc_score(y_test,y_pred)

auc
#confusion matrix

confusion = confusion_matrix(y_test,y_pred)

confusion
tp = confusion[0][0]

fp = confusion[0][1]

fn = confusion[1][0]

tn = confusion[1][1]
#correlation

sns.heatmap(X.corr())
#plotting confusion matrix

from sklearn.metrics import plot_confusion_matrix

disp = plot_confusion_matrix(rf,X_test,y_test,cmap=plt.cm.Blues,normalize=None)

#disp = plot_confusion_matrix(lg,X_test,y_test,cmap='viridis',normalize=None)

disp.confusion_matrix
# finding accuracy 

accuracy = (tp+tn)/(tp+tn+fp+fn)

accuracy
#plotting roc curve 

fpr , tpr , thresholds = roc_curve(y_test,y_pred)

plt.plot(fpr,tpr,color = 'darkblue',label = 'ROC')

plt.plot([0,1],[0,1],color='orange',linestyle='--',label="ROC Curve(area=%0.2f)"%auc)

plt.xlabel('False + ve rate')

plt.ylabel('True +ve rate')

plt.legend()

plt.show()
#model creation using pickle

import pickle 

model = open('model.pkl','wb')

pickle.dump(rf,model)

model.close()
rf.predict([[2,2,3,4,5,6,7,8]])