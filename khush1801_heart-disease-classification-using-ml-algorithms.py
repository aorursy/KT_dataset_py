# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
# print(os.listdir("../input/avani24"))
print(os.listdir("../input/"))

# Any results you write to the current directory are saved as output.
import pandas as pd # Data processing
import numpy as np # For Linear Algebra Calculation
heart_disease = pd.read_csv('../input/heart/heart.csv')
heart_disease.tail()
# Find how many variables and objects in the data set
heart_disease.shape
# view the type of data in the data set
heart_disease.info()
heart_disease= heart_disease.rename(columns= {'cp': 'chest_pain_type' , 'trestbps': 'resting_blood_pressure' , 'chol' : 'cholesterol',
                                             'fbs': 'fasting_blood_sugar' , 'restecg' : 'rest_ecg' ,'thalach' : 'max_heart_rate_achieved',
                                             'exang' : 'exercise_induced_angina' , 'oldpeak' : 'st_depression' , 'slope' : 'st_slope',
                                             'ca' : 'num_major_vessels' , 'thal' : 'thalassemia'})
# View the first 10 rows in data set
heart_disease.head(10)
# View the last 10 rows in the data set
heart_disease.tail(10)
heart_disease.isnull().sum()
# Convert Sex Column data
heart_disease['sex'][heart_disease['sex'] == 0] = 'Female'
heart_disease['sex'][heart_disease['sex'] == 1] = 'Male'
# Convert Chest pain type column data
heart_disease['chest_pain_type'][heart_disease['chest_pain_type'] == 0] = 'typical angina'
heart_disease['chest_pain_type'][heart_disease['chest_pain_type'] == 1] = 'atypical angina'
heart_disease['chest_pain_type'][heart_disease['chest_pain_type'] == 2] = 'non-anginal pain'
heart_disease['chest_pain_type'][heart_disease['chest_pain_type'] == 3] = 'asymptomatic'
# Convert Fast Blood sugar column
heart_disease['fasting_blood_sugar'][heart_disease['fasting_blood_sugar'] == 0] = 'lower than 120mg/ml'
heart_disease['fasting_blood_sugar'][heart_disease['fasting_blood_sugar'] == 1] = 'greater than 120mg/ml'
# Convert rest_ecg column data
heart_disease['rest_ecg'][heart_disease['rest_ecg'] == 0] = 'normal'
heart_disease['rest_ecg'][heart_disease['rest_ecg'] == 1] = 'ST-T wave abnormality'
heart_disease['rest_ecg'][heart_disease['rest_ecg'] == 2] = 'left ventricular hypertrophy'
# Convert exercise_included_angina
heart_disease['exercise_induced_angina'][heart_disease['exercise_induced_angina'] == 0] = 'no'
heart_disease['exercise_induced_angina'][heart_disease['exercise_induced_angina'] == 1] = 'yes'
# Convert solpe column data
heart_disease['st_slope'][heart_disease['st_slope'] == 1] = 'upsloping'
heart_disease['st_slope'][heart_disease['st_slope'] == 2] = 'flat'
heart_disease['st_slope'][heart_disease['st_slope'] == 3] = 'downsloping'
# convert Thalassemia column data
heart_disease['thalassemia'][heart_disease['thalassemia'] == 1] = 'normal'
heart_disease['thalassemia'][heart_disease['thalassemia'] == 2] = 'fixed defect'
heart_disease['thalassemia'][heart_disease['thalassemia'] == 3] = 'reversable defect'
# View the data set after changing it to Categorical
heart_disease.head(10)
heart_disease.describe().transpose()
# Calaculte on individual column count -Sex
heart_disease['sex'].value_counts()
# Calaculte on individual column count -chest_pain_type
heart_disease['chest_pain_type'].value_counts()
# Calculate on individual column count - fasting_blood_sugar
heart_disease['fasting_blood_sugar'].value_counts()
# Calculate on individual column count - rest_ecg
heart_disease['rest_ecg'].value_counts()
# Calculate on individual column count -exercise_induced_angina
heart_disease['exercise_induced_angina'].value_counts()
# Calculate on individual column count -st_slope
heart_disease['st_slope'].value_counts()
# Calculate on individual column count -  thalassemia
heart_disease['thalassemia'].value_counts()
# Import Libraries
from scipy.stats import skew , kurtosis
# Calculate Skewnes and Kurtosis on individual columns -Sex
print("skewness of the age" , skew(heart_disease['age']))
print("Kurtosis of Age ", kurtosis(heart_disease['age']))

# Calculate Skewnes and Kurtosis on individual columns - resting_blood_pressure
print("skewness of the resting_blood_pressure" , skew(heart_disease['resting_blood_pressure']))
print("Kurtosis of resting_blood_pressure ", kurtosis(heart_disease['resting_blood_pressure']))
# Calculate Skewnes and Kurtosis on individual columns - cholesterol
print("skewness of the cholesterol" , skew(heart_disease['cholesterol']))
print("Kurtosis of cholesterol ", kurtosis(heart_disease['cholesterol']))
# Calculate Skewnes and Kurtosis on individual columns - max_heart_rate_achieved
print("skewness of the max_heart_rate_achieved" , skew(heart_disease['max_heart_rate_achieved']))
print("Kurtosis of max_heart_rate_achieved ", kurtosis(heart_disease['max_heart_rate_achieved']))
# Calculate Skewnes and Kurtosis on individual columns - st_depression
print("skewness of the st_depression" , skew(heart_disease['st_depression']))
print("Kurtosis of st_depression ", kurtosis(heart_disease['st_depression']))
# Import Libraries
import matplotlib.pyplot as plt
import seaborn as sns
fig,ax = plt.subplots(figsize=(5,5))
ax = sns.countplot(heart_disease['age'])
plt.show()
fig,ax = plt.subplots(figsize=(15,5))
ax = sns.countplot(heart_disease['sex'])
plt.show()
fig,ax = plt.subplots(figsize=(15,5))
ax = sns.countplot(heart_disease['chest_pain_type'])
plt.show()
fig,ax = plt.subplots(figsize=(15,5))
ax = sns.countplot(heart_disease['target'])
plt.show()
fig,ax = plt.subplots(figsize=(15,5))
ax = sns.countplot(heart_disease['exercise_induced_angina'])
plt.show()
fig,ax = plt.subplots(figsize=(15,5))
ax = sns.countplot(heart_disease['rest_ecg'])
plt.show()
fig,ax = plt.subplots(figsize=(15,5))
ax = sns.countplot(heart_disease['st_slope'])
plt.show()
fig,ax = plt.subplots(figsize=(15,5))
ax = sns.countplot(heart_disease['thalassemia'])
plt.show()
sns.distplot(heart_disease['age'])

heart_disease.head()
sns.distplot(heart_disease['cholesterol'])
sns.distplot(heart_disease['resting_blood_pressure'])
sns.distplot(heart_disease['max_heart_rate_achieved'])

sns.distplot(heart_disease['st_depression'])

sns.distplot(heart_disease['num_major_vessels'])
f,ax = plt.subplots(figsize=(15,6))
ax = sns.boxplot(x='sex',y='age',data=heart_disease)
plt.show()
f,ax = plt.subplots(figsize=(15,6))
ax = sns.boxplot(x='age',y='max_heart_rate_achieved',data=heart_disease)
plt.show()
f,ax = plt.subplots(figsize=(15,6))
ax = sns.boxplot(x='age',y='max_heart_rate_achieved',data=heart_disease)
plt.show()
f,ax = plt.subplots(figsize=(15,6))
ax = sns.boxplot(x='age',y='target',data=heart_disease)
plt.show()
f,ax = plt.subplots(figsize=(15,6))
ax = sns.boxplot(x='age',y='cholesterol',data=heart_disease)
plt.show()
sns.distplot(heart_disease['target'])
pd.crosstab(heart_disease.age,heart_disease.target).plot(kind="bar",figsize=(25,8),color=['gold','brown' ])
plt.title('Heart Disease Frequency for Ages')
plt.xlabel('Sex')
plt.ylabel('Frequency')
plt.show()
pd.crosstab(heart_disease.sex,heart_disease.target).plot(kind="bar",figsize=(10,5),color=['cyan','coral' ])
plt.xlabel('Sex (0 = Female, 1 = Male)')
plt.xticks(rotation=0)
plt.legend(["Haven't Disease", "Have Disease"])
plt.ylabel('Frequency')
plt.show()
sns.pairplot(data=heart_disease)
plt.figure(figsize=(14,10))
sns.heatmap(heart_disease.corr(),annot=True,cmap='hsv',fmt='.3f',linewidths=2)
plt.show()
heart_disease.groupby('chest_pain_type', as_index=False)['target'].mean()
heart_disease.groupby('st_slope',as_index=False)['target'].mean()
heart_disease.groupby('thalassemia',as_index=False)['target'].mean()
heart_disease.groupby('target').mean()
# Convert the data into categorical data type
heart_disease.chest_pain_type = heart_disease.chest_pain_type.astype("category")
heart_disease.exercise_induced_angina = heart_disease.exercise_induced_angina.astype("category")
heart_disease.fasting_blood_sugar = heart_disease.fasting_blood_sugar.astype("category")
heart_disease.rest_ecg = heart_disease.rest_ecg.astype("category")
heart_disease.sex = heart_disease.sex.astype("category")
heart_disease.st_slope = heart_disease.st_slope.astype("category")
heart_disease.thalassemia = heart_disease.thalassemia.astype("category")
# Dummy values
heart_disease1 = pd.get_dummies(heart_disease, drop_first=True)
print(heart_disease1)
heart_disease1.head()
# Import Libraries
from sklearn.preprocessing import scale
scale(heart_disease1)
np.exp(scale(heart_disease1))
x = heart_disease1.drop(['target'], axis = 1)
y = heart_disease1.target.values
# Input values
x
# Output Values
y
# Import Libraries
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,train_size=0.80)
from sklearn.linear_model import LogisticRegression
logmodel = LogisticRegression()
# Fit the model
logmodel.fit(x_train,y_train)
# Predict the model
LR_pred = logmodel.predict(x_test)
LR_pred
# Confusion Matrix
from sklearn.metrics import confusion_matrix
print(confusion_matrix(LR_pred,y_test))
# Accuracy
from sklearn.metrics import accuracy_score
LR_accuracy = accuracy_score(LR_pred,y_test)
LR_accuracy
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors=3)
classifier
# Fit the Model
classifier.fit(x_train,y_train)
# Predict the Model
knn_pred = classifier.predict(x_test)
knn_pred
# Confusion Matrix
from sklearn.metrics import confusion_matrix
print(confusion_matrix(knn_pred,y_test))
# Accuracy Score
from sklearn.metrics import accuracy_score
accuracy_knn=accuracy_score(knn_pred,y_test)
accuracy_knn
from sklearn.naive_bayes import GaussianNB
classifier2 = GaussianNB()
classifier2
# Fit the model
classifier2.fit(x_train,y_train)
# Predict the model
NBC_pred = classifier2.predict(x_test)
NBC_pred
# Confusion Matrix
from sklearn.metrics import confusion_matrix
print(confusion_matrix(NBC_pred,y_test))
# Accuracy
from sklearn.metrics import accuracy_score
NBC_accuracy = accuracy_score(NBC_pred,y_test)
NBC_accuracy
from sklearn.tree import DecisionTreeClassifier
classifier1 = DecisionTreeClassifier(criterion='entropy',random_state=0)
classifier1
# Fit the model
classifier1.fit(x_train,y_train)
# Predict the model
DT_pred = classifier1.predict(x_test)
DT_pred
# Confusion Matrix
from sklearn.metrics import confusion_matrix
print(confusion_matrix(DT_pred,y_test))
# Accuracy
from sklearn.metrics import accuracy_score
accuracy_DT = accuracy_score(DT_pred,y_test)
accuracy_DT
from sklearn.ensemble import RandomForestClassifier
classifier3 = RandomForestClassifier(criterion='entropy',random_state=0)
classifier3
# Fit the model
classifier3.fit(x_train,y_train)
# Predict the model
RF_pred = classifier3.predict(x_test)
RF_pred
# Confusion Matrix
from sklearn.metrics import confusion_matrix
print(confusion_matrix(RF_pred,y_test))
# Accuracy
from sklearn.metrics import accuracy_score
accuracy_RF = accuracy_score(RF_pred,y_test)
accuracy_RF
from sklearn.svm import SVC
classifier4 = SVC(kernel = 'linear', random_state = 0)
classifier4
# Fit the model
classifier4.fit(x_train,y_train)
# Predict the model
SVC_pred = classifier4.predict(x_test)
SVC_pred
# Confusion Matrix
from sklearn.metrics import confusion_matrix
print(confusion_matrix(SVC_pred,y_test))
# Accuracy
from sklearn.metrics import accuracy_score
accuracy_SVC = accuracy_score(SVC_pred,y_test)
accuracy_SVC
from sklearn.ensemble import RandomForestClassifier
clf= RandomForestClassifier()
parameters = {'n_estimators': [4, 6, 9], 
              'max_features': ['log2', 'sqrt','auto'], 
              'criterion': ['entropy', 'gini'],
              'max_depth': [2, 3, 5, 10], 
              'min_samples_split': [2, 3, 5],
              'min_samples_leaf': [1,5,8]
             }

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, accuracy_score
classifier5 = GridSearchCV(clf, parameters, cv=5, scoring='accuracy')
classifier5
# Type of scoring used to compare parameter combinations
acc_scorer = make_scorer(accuracy_score)

# Run the grid search
grid_obj = GridSearchCV(clf, parameters, scoring=acc_scorer)
grid_obj = grid_obj.fit(x_train, y_train)
# Set the clf to the best combination of parameters
clf = grid_obj.best_estimator_
# Fit the best algorithm to the data. 
clf.fit(x_train, y_train)
predictions = clf.predict(x_test)
print(accuracy_score(y_test, predictions))
