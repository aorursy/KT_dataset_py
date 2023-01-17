#Importing the necessary libraries.

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import warnings

%matplotlib inline

sns.set_style("darkgrid")

warnings.filterwarnings("ignore")

import os

print(os.listdir("../input"))
#Importing the CSV data in form of dataframe.

df = pd.read_csv('../input/heart.csv')

df.head()
#Checking the info of the dataframe.

df.info()
#Visualizing the data.

fig,ax = plt.subplots(1,2,figsize=(10,5))

ax[0].set_title('Presence of Heart disease')

df['target'].value_counts().plot.pie(explode=[0.0,0.1],autopct='%.1f%%',shadow=True, cmap='Reds',ax= ax[0])

ax[0].set_ylabel("0 ->No , 1 ->Yes")

plt.title('Number of men and women')

df['sex'].value_counts().plot.pie(explode=[0.0,0.1],autopct='%.1f%%',shadow=True, cmap='Reds',ax= ax[1])

ax[1].set_ylabel("0 ->Female , 1 ->Male");
plt.figure(figsize = (5,5))

sns.countplot(x = 'target',data = df,hue = 'sex',palette = 'husl')

plt.xlabel('Present or not')

plt.ylabel('Number of People')

plt.legend(['Female','Male']);
sns.distplot(df['age'],color='Purple',hist_kws={'alpha':0.5,'linewidth': 1}, kde_kws={'color': 'orange', 'lw': 3, 'label': 'KDE'});
plt.figure(figsize = (8,3))

bins = [20,30,40,50,60,70,80]

df['age_bins']=pd.cut(df['age'], bins=bins)

g1=sns.countplot(x='age_bins',data=df ,hue='target',palette='Greens',linewidth=1)

plt.legend(['Absent','Present'])

plt.xlabel('Age Group')

plt.ylabel('Number of Patients')

plt.title('Classification of number of people in different age groups on the basis of presence of disease');
sns.countplot(x='cp',data=df,hue='target',palette='Purples',linewidth=1)

plt.title('Chest Pain Measure vs Heart Disease presence')

plt.ylabel('Number of Patients')

plt.xlabel('Chest Pain Measure')

plt.legend(['Disease Absent','Disease Present']);
#Heatmap of the correlation.

plt.figure(figsize=(12,12))

sns.heatmap(df.corr(),square = True,annot=True,linewidths=1);
plt.figure(figsize = (15,5))

sns.barplot(x=df['age_bins'],y=df['trestbps'],hue = df['target'])

plt.xlabel('Age Group')

plt.ylabel('Resting Blood Pressure')

plt.title('Age Group to Resting Blood Pressure Analysis on the basis of presence of disease');
plt.figure(figsize = (7,5))

sns.swarmplot(x='age_bins',y= 'chol',data=df,hue='target',palette='husl')

plt.title('Analysis of Serum Cholestrol Measure for different Age Groups on the basis of presence of disease')

plt.xlabel('Age Group')

plt.ylabel('Serum Cholestrol Measure')

plt.legend(['Disease Absent','Disease Present']);
plt.figure(figsize = (7,5))

sns.violinplot(x=df['fbs'],y=df['age_bins'])

plt.xlabel('Fasting Blood Pressure')

plt.ylabel('Age Groups')

plt.title('Age Group to Fasting Blood Pressure Analysis');
plt.figure(figsize = (7,5))

sns.boxplot(x='target',y= 'slope',data=df,palette='husl')

plt.title('The slope of the peak exercise ST segment vs Presence of Disease')

plt.xlabel('Presence of Disease, 1- Yes : 0- No')

plt.ylabel('The slope of the peak exercise ST segment');
sns.lmplot(x="trestbps", y="chol",data=df,hue='target')

plt.title('Serum Cholestrol measure vs Resting Blood Pressure');
plt.figure(figsize=(10,6))

count= df['ca'].value_counts()

sns.barplot(x=count.index, y=count.values)

plt.ylabel("Number of ca values")

plt.xlabel("Different types of Ca values")

plt.title("Ca values count");
plt.figure(figsize=(8,6))

sns.scatterplot(x='chol',y='thalach',data=df,hue='target')

plt.title('Serum Cholestrol measure vs Maximum Heart rate achieved analysis on the basis of presence of disease');
plt.figure(figsize=(12,5))

sns.barplot(x=df['age_bins'],y=df['thalach'],data = df)

plt.title('Age vs Thalach analysis');
sns.jointplot(x = df['age'], y = df['oldpeak']);
fig,ax=plt.subplots(figsize=(15,5))

sns.heatmap(df.isnull(), annot=True, cmap = 'Purples')

plt.title('Null value check');
#Droping the unnecessary columns and assigning the target column to y.

df.drop('age_bins',axis = 1,inplace = True)

y = df['target']
#Changing the data to categorical.

df['sex']=df['sex'].astype('category')

df['cp']=df['cp'].astype('category')

df['fbs']=df['fbs'].astype('category')

df['restecg']=df['restecg'].astype('category')

df['exang']=df['exang'].astype('category')

df['slope']=df['slope'].astype('category')

df['ca']=df['ca'].astype('category')

df['thal']=df['thal'].astype('category')

df['target']=df['target'].astype('category')
#Getting dummies for our data for our model.

df=pd.get_dummies(df,drop_first=True)

df.head()
#Assigning the value of X.

X=df.drop('target_1',axis = 1)

X.info()
#We are setting the test data to 25% of the total data.

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.16, random_state=65)
#Making an instance of Logistic Regression.

from sklearn.linear_model import LogisticRegression

logmod = LogisticRegression()
#Fitting the training data to train our model.

logmod.fit(X_train,y_train)
#Now lets predict off our test data!

predictions = logmod.predict(X_test)
predictions
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score



log_pred=logmod.predict(X_test)

print('Confusion Matrix: ')

print(confusion_matrix(y_test,log_pred))

print('\n')

print('Classification report: ')

print(classification_report(y_test,log_pred))

print('\n')

print('Accuracy score: {}%'.format(round(accuracy_score(y_test,log_pred)*100,2)))