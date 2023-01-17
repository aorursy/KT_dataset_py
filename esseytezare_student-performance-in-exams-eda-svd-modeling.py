# Imports

# pandas
import pandas as pd
from pandas import Series,DataFrame

# numpy, matplotlib, seaborn
import numpy as np
import matplotlib.pyplot as plt
import missingno as missing
import seaborn as sns
sns.set_style('whitegrid')
%matplotlib inline

import random
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import learning_curve, validation_curve
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, accuracy_score
import sklearn.metrics
# for providing path
import os
print(os.listdir('../input/'))
# get titanic & test csv files as a DataFrame
df = pd.read_csv('../input/students-performance-in-exams/StudentsPerformance.csv')

# preview the data
df.head()
df.info()
missing.matrix(df, figsize = (20,3))
df.isnull().sum()
df.describe()
# lets check the no. of unique items present in the categorical column

df.select_dtypes('object').nunique()
plt.figure(figsize=(25,6))
plt.subplot(1, 3, 1)
sns.distplot(df['math score'])

plt.subplot(1, 3, 2)
sns.distplot(df['reading score'])

plt.subplot(1, 3, 3)
sns.distplot(df['writing score'])

plt.suptitle('Checking for Skewness', fontsize = 15)
plt.show()



plt.figure(figsize=(25,6))
plt.subplot(1, 3, 1)
sns.boxplot(x="race/ethnicity", y="math score", hue="gender", data=df)
plt.title('MATH SCORES')
plt.subplot(1, 3, 2)
sns.boxplot(x="race/ethnicity", y="reading score", hue="gender", data=df)
plt.title('READING SCORES')
plt.subplot(1, 3, 3)
sns.boxplot(x="race/ethnicity", y="writing score", hue="gender", data=df)
plt.title('WRITING SCORES')
plt.show()
plt.figure(figsize=(25,6))
plt.subplot(1, 3, 1)
sns.boxplot(x="lunch", y="math score", hue="gender", data=df)
plt.title('MATH SCORES')
plt.subplot(1, 3, 2)
sns.boxplot(x="lunch", y="reading score", hue="gender", data=df)
plt.title('READING SCORES')
plt.subplot(1, 3, 3)
sns.boxplot(x="lunch", y="writing score", hue="gender", data=df)
plt.title('WRITING SCORES')
plt.show()
plt.figure(figsize=(25,6))
plt.subplot(1, 3, 1)
sns.boxplot(x="parental level of education", y="math score", hue="gender", data=df)
plt.title('MATH SCORES')
plt.xticks(rotation = 90)
plt.subplot(1, 3, 2)
sns.boxplot(x="parental level of education", y="reading score", hue="gender", data=df)
plt.title('READING SCORES')
plt.xticks(rotation = 90)
plt.subplot(1, 3, 3)
sns.boxplot(x="parental level of education", y="writing score", hue="gender", data=df)
plt.title('WRITING SCORES')
plt.xticks(rotation = 90)
plt.show()

plt.figure(figsize=(25,6))
plt.subplot(1, 3, 1)
sns.boxplot(x="test preparation course", y="math score", hue="gender", data=df)
plt.title('MATH SCORES')
plt.subplot(1, 3, 2)
sns.boxplot(x="test preparation course", y="reading score", hue="gender", data=df)
plt.title('READING SCORES')
plt.subplot(1, 3, 3)
sns.boxplot(x="test preparation course", y="writing score", hue="gender", data=df)
plt.title('WRITING SCORES')
plt.show()
plt.figure(figsize=(25,6))
sns.pairplot(data=df,hue='gender',plot_kws={'alpha':0.2})
plt.show()
df['math_pass']=np.where(df['math score'] >= 55,'P','F')
df['reading_pass']=np.where(df['reading score'] >= 65,'P','F')
df['writing_pass']=np.where(df['writing score'] >= 65,'P','F')
df['Pass'] = df.apply(lambda x :1 if x['math score'] >= 55 and 
                      x['reading score'] >= 65 and 
                      x['writing score'] >= 65 
                      else 0, axis =1)
df.head()
plt.figure(figsize=(20,15))

plt.subplot(4,3,1)
sns.countplot(x='parental level of education', hue='writing_pass', data=df)
plt.xticks(rotation=45)
plt.subplot(4,3,2)
sns.countplot(x='parental level of education', hue='math_pass', data=df)
plt.xticks(rotation=45)
plt.subplot(4,3,3)
sns.countplot(x='parental level of education', hue='reading_pass', data=df)
plt.xticks(rotation=45)

plt.subplot(4,3,4)
sns.countplot(x='gender', hue='writing_pass', data=df)
plt.xticks(rotation=45)
plt.title("Gender - Writing Pass")
plt.subplot(4,3,5)
sns.countplot(x='gender', hue='math_pass', data=df)
plt.xticks(rotation=45)
plt.title("Gender - Math Pass")
plt.subplot(4,3,6)
sns.countplot(x='gender', hue='reading_pass', data=df)
plt.xticks(rotation=45)
plt.title("Gender - Reading Pass")

plt.subplot(4,3,7)
sns.countplot(x='test preparation course', hue='writing_pass', data=df)
plt.xticks(rotation=45)
plt.title("Preparation - Writing Pass")
plt.subplot(4,3,8)
sns.countplot(x='test preparation course', hue='math_pass', data=df)
plt.xticks(rotation=45)
plt.title("Preparation - Math Pass")
plt.subplot(4,3,9)
sns.countplot(x='test preparation course', hue='reading_pass', data=df)
plt.xticks(rotation=45)
plt.title("Preparation - Reading Pass")

plt.subplot(4,3,10)
sns.countplot(x='race/ethnicity', hue='writing_pass', data=df)
plt.xticks(rotation=45)
plt.title("Race - Writing Pass")
plt.subplot(4,3,11)
sns.countplot(x='race/ethnicity', hue='math_pass', data=df)
plt.xticks(rotation=45)
plt.title("Race - Math Pass")
plt.subplot(4,3,12)
sns.countplot(x='race/ethnicity', hue='reading_pass', data=df)
plt.xticks(rotation=45)
plt.title("Race - Reading Pass")

plt.tight_layout()
plt.show()

map1 = {"high school": 1, "some high school": 1,
        "associate's degree": 2,
        "some college": 3,
        "bachelor's degree": 4,
        "master's degree": 5}
df['parental level of education']  = df['parental level of education'].map(map1)

map2 = {"free/reduced": 0,
        "standard": 1}
df['lunch']  = df['lunch'].map(map2)

map3 = {"none": 0,
        "completed": 1}
df['test preparation course']  = df['test preparation course'].map(map3)

map4 = {"female": 0,
        "male": 1}
df['gender']  = df['gender'].map(map4)

map5 = {"group A": 1,
        "group B": 2,
        "group C": 3,
        "group D": 4,
        "group E": 5}
df['race/ethnicity']  = df['race/ethnicity'].map(map5)

plt.figure(figsize=(13,10))

plt.subplot(4,3,1)
sns.barplot(x = "parental level of education" , y="writing score" , data=df)
plt.title("Parental level - Writing Scores")
plt.subplot(4,3,2)
sns.barplot(x = "parental level of education" , y="math score" , data=df)
plt.title("Parental level - Math Scores")
plt.subplot(4,3,3)
sns.barplot(x = "parental level of education" , y="reading score" , data=df)
plt.title("Parental level - Reading Scores")

plt.subplot(4,3,4)
sns.barplot(x = "gender" , y="writing score" , data=df)
plt.title("Gender - Writing Scores")
plt.subplot(4,3,5)
sns.barplot(x = "gender" , y="math score" , data=df)
plt.title("Gender - Math Scores")
plt.subplot(4,3,6)
sns.barplot(x = "gender" , y="reading score" , data=df)
plt.title("Gender - Reading Scores")

plt.subplot(4,3,7)
sns.barplot(x = "test preparation course" , y="writing score" , data=df)
plt.title("Preparation - Writing Scores")
plt.subplot(4,3,8)
sns.barplot(x = "test preparation course" , y="math score" , data=df)
plt.title("Preparation - Math Scores")
plt.subplot(4,3,9)
sns.barplot(x = "test preparation course" , y="reading score" , data=df)
plt.title("Preparation - Reading Scores")

plt.subplot(4,3,10)
sns.barplot(x = "race/ethnicity" , y="writing score" , data=df)
plt.title("Race - Writing Scores")
plt.subplot(4,3,11)
sns.barplot(x = "race/ethnicity" , y="math score" , data=df)
plt.title("Race - Math Scores")
plt.subplot(4,3,12)
sns.barplot(x = "race/ethnicity" , y="reading score" , data=df)
plt.title("Race - Reading Scores")

plt.tight_layout()
plt.show()
plt.subplots(figsize=(15,10)) 
sns.heatmap(df.corr(), annot = True, fmt = ".2f")
plt.show()
dfDrop = df.drop(['math score','reading score','writing score', 'math_pass', 'reading_pass','writing_pass'], axis=1)
dfDrop.head()

dfDrop.info()
plt.subplots(figsize=(15,10)) 
sns.heatmap(dfDrop.corr(), annot = True, fmt = ".2f")
plt.show()
df.columns
feature_df=df[['gender','race/ethnicity','parental level of education','lunch','test preparation course']]

#Independent var
X=np.asarray(feature_df)

#Dependent variable
y=np.asarray(df['Pass'])

X[0:5]
 
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X, y, test_size=0.2, random_state=23)

# 800 x 5
X_train.shape

# 800 x 1
y_train.shape

# 200 x 5
X_test.shape

# 200 x 1
y_test.shape
from sklearn import svm

classifier=svm.SVC(kernel = 'linear' , gamma=0.001, C=100)
classifier.fit(X_train,y_train)

y_predict=classifier.predict(X_test)
from sklearn.metrics import classification_report
from sklearn import metrics
from sklearn.metrics import roc_curve

print(classification_report(y_test,y_predict))
print("Accuracy:",metrics.accuracy_score(y_test, y_predict))
def plotLearningCurves(X_train, y_train, classifier, title):
    train_sizes, train_scores, test_scores = learning_curve(
            classifier, X_train, y_train, cv=5, scoring="accuracy")
    
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    plt.plot(train_sizes, train_scores_mean, label="Training Error")
    plt.plot(train_sizes, test_scores_mean, label="Cross Validation Error")
    
    plt.legend()
    plt.grid()
    plt.title(title, fontsize = 18, y = 1.03)
    plt.xlabel('Data Size', fontsize = 14)
    plt.ylabel('Error', fontsize = 14)
    plt.tight_layout()
def plotValidationCurves(X_train, y_train, classifier, param_name, param_range, title):
    train_scores, test_scores = validation_curve(
        classifier, X_train, y_train, param_name = param_name, param_range = param_range,
        cv=5, scoring="accuracy")

    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    plt.plot(param_range, train_scores_mean, label="Training Error")
    plt.plot(param_range, test_scores_mean, label="Cross Validation Error")

    plt.legend()
    plt.grid()
    plt.title(title, fontsize = 18, y = 1.03)
    plt.xlabel('Complexity', fontsize = 14)
    plt.ylabel('Error', fontsize = 14)
    plt.tight_layout()
plt.figure(figsize=(16,5))
title='Support Vector Machine Learning Curve'
plotLearningCurves(X_train,y_train,classifier,title)
# call general function to fit the classifier and draw the validation curve
title = 'Support Vector Machine Validation Curve'
param_name = 'C'
param_range = [0.1,1, 10]
plt.figure(figsize = (16,5))
plotValidationCurves(X_train, y_train, classifier, param_name, param_range, title)
classifier2=svm.SVC(kernel = 'rbf' , gamma=1, C=2**5)
classifier2.fit(X_train,y_train)

y_predict2=classifier2.predict(X_test)
print(classification_report(y_test,y_predict2))
print("Accuracy:",metrics.accuracy_score(y_test, y_predict2))
plt.figure(figsize=(16,5))
title='Support Vector Machine Learning Curve'
plotLearningCurves(X_train,y_train,classifier2,title)
# call general function to fit the classifier and draw the validation curve
title = 'Support Vector Machine Validation Curve'
param_name = 'C'
param_range = [0.1,1, 10]
plt.figure(figsize = (16,5))
plotValidationCurves(X_train, y_train, classifier2, param_name, param_range, title)
classifier3=svm.SVC(kernel = 'sigmoid' , gamma=1, C=0.5)
classifier3.fit(X_train,y_train)

y_predict3=classifier2.predict(X_test)
print(classification_report(y_test,y_predict3))
print("Accuracy:",metrics.accuracy_score(y_test, y_predict3))
plt.figure(figsize=(16,5))
title='Support Vector Machine Learning Curve'
plotLearningCurves(X_train,y_train,classifier3,title)
# call general function to fit the classifier and draw the validation curve
title = 'Support Vector Machine Validation Curve'
param_name = 'C'
param_range = [0.1,1, 10]
plt.figure(figsize = (16,5))
plotValidationCurves(X_train, y_train, classifier3, param_name, param_range, title)
param_grid = {'C': [0.05,0.1,0.5,1,10,100], 'gamma': [0.001,0.01,0.1,1,1.5,2**5],'kernel': ['sigmoid', 'rbf','linear']}
grid = GridSearchCV(svm.SVC(),param_grid,refit=True,verbose=2)
svclassifier = grid.fit(X_train,y_train)
SvcPredictions = svclassifier.predict(X_test)
print(grid.best_estimator_)
print("Accuracy;",accuracy_score(y_test, y_predict))
print(confusion_matrix(y_test, y_predict))
print(classification_report(y_test, y_predict))
from sklearn.metrics import plot_roc_curve
svm_disp = plot_roc_curve(classifier, X_test, y_test)
plt.show()