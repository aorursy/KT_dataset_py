# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn import metrics

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
#Fetching the data

df = pd.read_csv("../input/HR_comma_sep.csv")

df.head()
#So we have no missing values that is AWESOME

df.isnull().sum()
#Now we will check the Data types

df.dtypes
df.shape
#The result shows that we have 15000 employees data

#Now we will find out how many employees belong to which department

plt.figure(figsize=(15,6))

sns.countplot(df['sales'])
#As e=we now salary is being categorised into 3 groups 1-Low, 2-Medium, 3-High

#Now we will see employees of which Salary group had left their jobs

plt.figure(figsize=(15,6))

sns.countplot(df[df.left == 1].salary)

#As Expected Employees with LOW salaries have left their jobs the most
#Now we will see that Employees of which department have left their jobs the most

plt.figure(figsize=(15,6))

sns.countplot(df[df.left == 1].sales)
#tne = Total Number of Employees per Department

tne = df['sales'].value_counts()

tne
#tnel = Total Number of Employees leaving per department

tnel = df[df.left == 1].sales.value_counts()

tnel
#Now finding the LEAVING RATIO of each department

lr = (tnel/tne) * 100

lr.sort_values(axis=0)

#Surprisingly for me the HR department has the highest leaving Ratio
#Visualising for better understanding

plt.figure(figsize=(15,8))

plt.scatter(lr.index,lr.values)

plt.plot(lr.index,lr.values)
#Finding the relations between the features

plt.figure(figsize=(15,10))

sns.heatmap(df.corr(),annot=True,cmap="coolwarm")
#Now trying to find why Employees by Comparing them to the Employees who does not left

#Comparing the Number of Projects

fig, axs = plt.subplots(ncols=2,figsize=(12,6))

sns.countplot(df[df.left == 1].number_project,ax =axs[0])

sns.countplot(df[df.left == 0].number_project,ax =axs[1])

axs[0].set_title('Employees who LEFT')

axs[1].set_title('Employees who did not LEFT')
#Comparing the time spend in the Company

fig, axs = plt.subplots(ncols=2,figsize=(12,6))

sns.countplot(df[df.left == 1].time_spend_company,ax =axs[0])

sns.countplot(df[df.left == 0].time_spend_company,ax =axs[1])

axs[0].set_title('Employees who LEFT')

axs[1].set_title('Employees who did not LEFT')
#Comparing Promotions of last 5 Years

fig, axs = plt.subplots(ncols=2,figsize=(12,6))

sns.countplot(df[df.left == 1].promotion_last_5years,ax =axs[0])

sns.countplot(df[df.left == 0].promotion_last_5years,ax=axs[1])

axs[0].set_title('Employees who LEFT')

axs[1].set_title('Employees who did not LEFT')
#Comparing Work Accidents

fig, axs = plt.subplots(ncols=2,figsize=(12,6))

sns.countplot(df[df.left == 1].Work_accident,ax =axs[0])

sns.countplot(df[df.left == 0].Work_accident,ax=axs[1])

axs[0].set_title('Employees who LEFT')

axs[1].set_title('Employees who did not LEFT')
#Now we are finding the Satisfaction level of both type of Employees 

#sll = Satisfaction level of the Employees who left

#sldl = Satisfaction level of the Employees who did not left

sll = df[df.left == 1].satisfaction_level.median() * 100

sldl = df[df.left == 0].satisfaction_level.median() * 100

print('The Satisfaction level of the Employees who left were {} %'.format(sll))

print('The Satisfaction level of the Employees who did not left were {} %'.format(sldl))
sll = df[df.left == 1].last_evaluation.median() * 100

sldl = df[df.left == 0].last_evaluation.median() * 100

print('The Last Eva of the Employees who left were {} %'.format(sll))

print('The Satisfaction level of the Employees who did not left were {} %'.format(sldl))
dfm = df[df.sales == 'management']
dfhr = df[df.sales == 'hr']
#As we Know from our previous Analysis that the Management department has the least leaving Ratio

#And HR Department has the highest leaving Ratio

#SO we can find Valuable INSIGHTS by comparing both the departments

fig, axs = plt.subplots(ncols=2,figsize=(12,6))

sns.countplot(dfm.salary,ax=axs[0])

sns.countplot(dfhr.salary,ax=axs[1])

axs[0].set_title('Management Department')

axs[1].set_title('HR Department')

# So according to the graph Management department have more Employees having High salaries in coparision to

# Hr Department. As we all Know Satisfaction level is directly proptional to SALARY xD
fig, axs = plt.subplots(ncols=2,figsize=(12,6))

sns.countplot(dfm.number_project,ax=axs[0])

sns.countplot(dfhr.number_project,ax=axs[1])

axs[0].set_title('Management Department')

axs[1].set_title('HR Department')
df.head()
#Now its time to build our Prediction model

#We are dividing our Dependent and Independent features

X = df.iloc[:,[0,1,2,3,4,5,7,8,9]].values

y = df.iloc[:,6].values
X.shape
y.shape
X.dtypes
#As sales and salary colunm have data type of Object we have to apply label Encoding to it!

from sklearn.preprocessing import LabelEncoder

le_X=LabelEncoder()

X[:,7] = le_X.fit_transform(X[:,7])

X[:,8] = le_X.fit_transform(X[:,8])
#Now we are splittiong our data into train and test set

from sklearn.cross_validation import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
print(X_train.shape)

print(X_test.shape)
#Applying "feature scaling"

#Feature scaling is a method used to standardize the range of independent variables or features of data

from sklearn.preprocessing import StandardScaler as SS

sc_X = SS()

X_train = sc_X.fit_transform(X_train)

X_test = sc_X.fit_transform(X_test)
#We are using Random Forest Classifier as our Prediction Model

from sklearn.ensemble import RandomForestClassifier

classifier = RandomForestClassifier(n_estimators=15,criterion='entropy',random_state=0)

classifier.fit(X_train,y_train)
ypred=classifier.predict(X_test)

rf = metrics.accuracy_score(ypred,y_test) * 100 # to check the accuracy

print(rf)

#WE GET AN ACCURACY OF 99.13%. THATS AMAZING

#IF YOU LIKED MY EFFORT PLEASE GIVE ME AN UPVOTE. 

#THANK YOU!:)