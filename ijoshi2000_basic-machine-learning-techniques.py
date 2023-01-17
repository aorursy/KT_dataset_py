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
# Reading the dataset into a variable called pima

pima = pd.read_csv('../input/pima-indians-diabetes-database/diabetes.csv')

# Get the preview of the dataset

pima.head()

#Get a summary of the dataset

pima.describe()
# PREPROCESSING THE DATA 



# 1.THE HIGHEST NUMBER OF PREGNANCIES THAT WOULD ESSENTIALLY AFFECT THE DATA ARE 8

pima.loc[pima.Pregnancies>8,'Pregnancies'] = 8



# 2. THE AGE FEATURE CAN BE DISTRIBUTED IN 5 CATAGORIES

pima.loc[pima.Age<=30 , 'Age'] = 1

pima.loc[pima.Age.between(31,41), 'Age'] = 2

pima.loc[pima.Age.between(41,51), 'Age'] = 3

pima.loc[pima.Age.between(51,61), 'Age'] = 4

pima.loc[pima.Age>60 ,'Age'] = 5



# 3. X-FEATURES 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI' and 'DiabetesPedigreeFunction'

# CANNOT BE ZERO HENCE CHANGE 0 VALUES TO THE NEXT SMALLEST VALUE



pima.loc[pima.Glucose==0 , 'Glucose'] = 1000

pima.loc[pima.Glucose==1000 , 'Glucose'] = pima.Glucose.min()



pima.loc[pima.BloodPressure==0 , 'BloodPressure'] = 1000

pima.loc[pima.BloodPressure==1000 , 'BloodPressure'] = pima.BloodPressure.min()



pima.loc[pima.SkinThickness==0 , 'SkinThickness'] = 1000

pima.loc[pima.SkinThickness==1000 , 'SkinThickness'] = pima.SkinThickness.min()



pima.loc[pima.Insulin==0 , 'Insulin'] = 1000

pima.loc[pima.Insulin==1000 , 'Insulin'] = pima.Insulin.min()



pima.loc[pima.BMI==0 , 'BMI'] = 1000

pima.loc[pima.BMI==1000 , 'BMI'] = pima.BMI.min()



pima.loc[pima.DiabetesPedigreeFunction==0 , 'DiabetesPedigreeFunction'] = 1000

pima.loc[pima.DiabetesPedigreeFunction==1000 , 'DiabetesPedigreeFunction'] = pima.DiabetesPedigreeFunction.min()

# WE NOW NEED TO SCALE THE DATA USING MINMAX SCALER 

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(copy = False)

pima[['Glucose', 'BloodPressure', 'SkinThickness','Insulin','BMI','DiabetesPedigreeFunction']] = scaler.fit_transform(pima[['Glucose', 'BloodPressure', 'SkinThickness','Insulin','BMI','DiabetesPedigreeFunction']])



print(pima.head())

# SUMMARY OF THE DATA TO SHOW THAT THE DATA IS CHANGED 

pima.describe().T







#HERE WE CAN SEE THAT 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI' AND 'DiabetesPedigreeFunction'

# ARE SCALED BETWEEN 0 AND 1



#'Age' AND 'Pregnancies' COLUMNS ARE CHARACTERIZED 
# GENERATING HEATMAP IN ORDER TO FIND CORRELATIONS BETWEEN THE X-FEATURES

import seaborn as sns

import matplotlib.pyplot as plt

corr=pima.corr()



sns.set(font_scale=1.15)

plt.figure(figsize=(10, 10))



sns.heatmap(corr, vmax=1, linewidths=0.1,

            square=True,annot=True,cmap='Purples',linecolor="red")

plt.title('Correlation between x-features')



#NOW WE CAN EXTRACT THE TWO FEATURES 'DiabetesPedigreeFunction' AND 'BloodPressure' TO MAKE X_total

#AND 'Outcome' TO MAKE y_total



X_total = pima.iloc[:, 0:8].values # CREATE ARRAYS

y_total = pima.iloc[:,8].values #CREATE AN ARRAY



features = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness','Insulin','BMI','DiabetesPedigreeFunction']







# SPLIT THE DATA INTO TRAIN AND TEST 70-30 TO PERFORM LOGISTIC REGRESSION(LR)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X_total, y_total, test_size=0.3,random_state=1)

#LOGISTIC REGRESSION(LR)

#TRAIN THE DATA WITH LR MODEL

from sklearn.linear_model import LogisticRegression

lr = LogisticRegression(C=100, random_state =1)

lr.fit(X_train, y_train)



#PREDICT ON THE X-TEST VALUES

y_pred = lr.predict(X_test)

print(lr.coef_) # W1 AND W2 FOR THE 2 GIVEN FEATURES



#CHECK THE ACCURACY OF THE LR MODEL

from sklearn.metrics import accuracy_score

print('Misclassified samples: %d' % (y_test != y_pred).sum())

print('Accuracy: %.2f' % accuracy_score(y_test, y_pred))

#DECISION TREE CLASSIFIER(DT)

from sklearn.tree import DecisionTreeClassifier

#TRAIN THE DATA WITH DT MODEL

dtree = DecisionTreeClassifier(max_depth=4, random_state=42)

dtree.fit(X_train, y_train)



#PREDICT ON THE X-TEST VALUES

y_pred = dtree.predict(X_test)



#CHECK THE ACCURACY OF THE DT MODEL

print('Misclassified samples: %d' % (y_test != y_pred).sum())

print('Accuracy: %.2f' % accuracy_score(y_test, y_pred))

#KNeighbors Classifier(KNN)

from sklearn.neighbors import KNeighborsClassifier



#TRAIN THE DATA WITH DT MODEL

knn = KNeighborsClassifier(n_neighbors=3, p=2, metric='minkowski')

knn.fit(X_train, y_train)



#PREDICT ON THE X-TEST VALUES

y_pred = knn.predict(X_test)



#CHECK THE ACCURACY OF THE DT MODEL

print('Misclassified samples: %d' % (y_test != y_pred).sum())

print('Accuracy: %.2f' % accuracy_score(y_test, y_pred))

#Gradient Boosting Classifier (GB)

from sklearn.ensemble import GradientBoostingClassifier



#TRAIN THE DATA WITH GB MODEL

gb = model = GradientBoostingClassifier()

gb.fit(X_train, y_train)



#PREDICT ON THE X-TEST VALUES

y_pred = gb.predict(X_test)



#CHECK THE ACCURACY OF THE DT MODEL

print('Misclassified samples: %d' % (y_test != y_pred).sum())

print('Accuracy: %.2f' % accuracy_score(y_test, y_pred))

#Gaussian Naive Bais(NB)

from sklearn.naive_bayes import GaussianNB



#TRAIN THE DATA WITH NB MODEL

nb = GaussianNB()

nb.fit(X_train, y_train)



#PREDICT ON THE X-TEST VALUES

y_pred = nb.predict(X_test)



#CHECK THE ACCURACY OF THE NB MODEL

print('Misclassified samples: %d' % (y_test != y_pred).sum())

print('Accuracy: %.2f' % accuracy_score(y_test, y_pred))

#Support Vector Machines(SVM)

from sklearn import svm



#TRAIN THE DATA WITH SVM MODEL

svc = svm.SVC()

svc.fit(X_train, y_train)



#PREDICT ON THE X-TEST VALUES

y_pred = svc.predict(X_test)



#CHECK THE ACCURACY OF THE SVM MODEL

print('Misclassified samples: %d' % (y_test != y_pred).sum())

print('Accuracy: %.2f' % accuracy_score(y_test, y_pred))
#Random Forrest Classifier(RF)

from sklearn.ensemble import RandomForestClassifier



#TRAIN THE DATA WITH RF MODEL

rf = RandomForestClassifier(n_estimators=10, random_state =0)

rf.fit(X_train, y_train)



#PREDICT ON THE X-TEST VALUES

y_pred = rf.predict(X_test)



#CHECK THE ACCURACY OF THE RF MODEL

print('Misclassified samples: %d' % (y_test != y_pred).sum())

print('Accuracy: %.2f' % accuracy_score(y_test, y_pred))