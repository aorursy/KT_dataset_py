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
# Import the basic libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
import plotly.express as px

# Import the data processing and model evaluation libraries 
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit, cross_val_score, cross_val_predict
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, precision_recall_curve
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle

# Import the models
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
df = pd.read_csv('../input/heart-failure-clinical-data/heart_failure_clinical_records_dataset.csv')
df.head()
df.info()
# Summary of numerical features

df.describe()
# Histograms for all numerical features 

df.hist(figsize = (20,15), color='pink')
# Shuffle the entire data

df = shuffle(df)

# Split the test set out of the data

df_train_01, df_test = train_test_split(df, test_size=0.2, random_state=42)

# Create a train and validation set

df_train, df_validation = train_test_split(df_train_01, test_size=0.2, random_state=42)
# The features with the highest correlation values will be used as input features for creating a model.

df_train.corr()
mask = np.triu(np.ones_like(df_train.corr(), dtype=np.bool))

f, ax = plt.subplots(figsize=(15, 10))

cmap = sns.diverging_palette(220, 10, as_cmap=True)

sns.heatmap(df_train.corr(), cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})
# Distribution of Anaemia vs Heart failure 

fig, ax = plt.subplots()
sns.distplot(df[df['DEATH_EVENT']==0]['anaemia'], hist=False, label = 'No heart failure')
sns.distplot(df[df['DEATH_EVENT']==1]['anaemia'], hist=False, label = 'Heart failure')
# Proportion of Anaemic vs. Non-anaemic people who undergo heart failure 

df_ad = df_train.groupby('anaemia')['DEATH_EVENT'].value_counts().to_frame()

print('Proportion of Non-anaemic people who have Heart failure:', ((df_ad.loc[(0, 1), :]/((df_ad.loc[(0, 0), :]) + (df_ad.loc[(0, 1), :])))*100).values) # proportion of non anaemic people who have heart failure
print('Proportion of Anaemic people who have Heart failure:', ((df_ad.loc[(1, 1), :]/((df_ad.loc[(1, 0), :]) + (df_ad.loc[(1, 1), :])))*100).values) # proportion of anaemic people who have heart failure 
# Anaemia, Heart failure and other categorical risk factors

fig, ax = plt.subplots(1,4, figsize=(20,5))
x=0

# Ordinal variables such as high blood pressure, diabetes and smoking are mapped on Y axis and hue is kept as anaemia for better interpretation 
for variable in ['high_blood_pressure', 'diabetes', 'smoking']:
    sns.barplot(x='DEATH_EVENT', y=variable, hue='anaemia' , data=df_train, ax=ax[x])
    x+=1

# Sex is a nominal variable. Hence, anaemia is mapped on y axis and hue is kept as sex for better interpretation
sns.barplot(x='DEATH_EVENT', y='anaemia', hue='sex' , data=df_train, ax=ax[3])
# Correlation values 

for variable in ['high_blood_pressure', 'diabetes', 'smoking']:
    for x in [0,1]:
        print('For patients with anaemia = ', x, 'correlation of', variable, 'with heart failure is', 
              ((df_train[df_train['anaemia']==x]).corr()[variable]['DEATH_EVENT']))
        
for x in [0,1]:
    print('For patients with sex = ', x, 'correlation of anaemia with heart failure is', ((df_train[df_train['sex']==x]).corr()['anaemia']['DEATH_EVENT']))
        
# Correlation values

fig, ax = plt.subplots(1,3, figsize=(20,8))
x=0

for variable in ['creatinine_phosphokinase', 'platelets', 'serum_sodium']:
    sns.boxplot(y=variable, x='DEATH_EVENT', hue='anaemia', showfliers=False, data=df_train, ax=ax[x])
    x+=1
# Correlation values 

for variable in ['creatinine_phosphokinase', 'platelets', 'serum_sodium']:
    for x in [0,1]:
        print('For patients with anaemia =', x, 'correlation of heart failure and', variable, 'is',
              ((df_train[df_train['anaemia']==x]).corr()[variable]['DEATH_EVENT']))
# Distribution of High blood pressure vs Heart failure 

fig, ax = plt.subplots()
sns.distplot(df[df['DEATH_EVENT']==0]['high_blood_pressure'], hist=False, label = 'No heart failure')
sns.distplot(df[df['DEATH_EVENT']==1]['high_blood_pressure'], hist=False, label = 'Heart failure')
# Correlation values

df_ad = df_train.groupby('high_blood_pressure')['DEATH_EVENT'].value_counts().to_frame()

print('Proportion of people without high BP people who have Heart failure:', ((df_ad.loc[(0, 1), :]/((df_ad.loc[(0, 0), :]) + (df_ad.loc[(0, 1), :])))*100).values) 
print('Proportion of people with high BP who have Heart failure:', ((df_ad.loc[(1, 1), :]/((df_ad.loc[(1, 0), :]) + (df_ad.loc[(1, 1), :])))*100).values) 
# High BP, Heart failure and other categorical risk factors

fig, ax = plt.subplots(1,4, figsize=(20,5))
x=0

# Ordinal variables such as anaemia, diabetes and smoking are mapped on Y axis and hue is kept as high BP for better interpretation 
for variable in ['anaemia', 'diabetes', 'smoking']:
    sns.barplot(x='DEATH_EVENT', y=variable, hue='high_blood_pressure' , data=df_train, ax=ax[x])
    x+=1

# Sex is a nominal variable. Hence, high BP is mapped on y axis and hue is kept as sex for better interpretation
sns.barplot(x='DEATH_EVENT', y='high_blood_pressure', hue='sex' , data=df_train, ax=ax[3])
# Correlation values 

for variable in ['anaemia', 'diabetes', 'smoking']:
    for x in [0,1]:
        print('For patients with high_blood_pressure = ', x, 'correlation of', variable, 'with heart failure is', 
              ((df_train[df_train['high_blood_pressure']==x]).corr()[variable]['DEATH_EVENT']))
        
for x in [0,1]:
    print('For patients with sex = ', x, 'correlation of high_blood_pressure with heart failure is', ((df_train[df_train['sex']==x]).corr()['high_blood_pressure']['DEATH_EVENT']))
        
# High BP, Heart failure and other numerical risk factors 

fig, ax = plt.subplots(1,3, figsize=(20,8))
x=0

for variable in ['creatinine_phosphokinase', 'platelets', 'serum_sodium']:
    sns.boxplot(y=variable, x='DEATH_EVENT', hue='high_blood_pressure', showfliers=False, data=df_train, ax=ax[x])
    x+=1
# Correlation values

for variable in ['creatinine_phosphokinase', 'platelets', 'serum_sodium']:
    for x in [0,1]:
        print('For patients with high_blood_pressure =', x, 'correlation of heart failure and', variable, 'is',
              ((df_train[df_train['high_blood_pressure']==x]).corr()[variable]['DEATH_EVENT']))
# Distribution of High blood pressure vs Heart failure 

fig, ax = plt.subplots()
sns.distplot(df[df['DEATH_EVENT']==0]['diabetes'], hist=False, label = 'No heart failure')
sns.distplot(df[df['DEATH_EVENT']==1]['diabetes'], hist=False, label = 'Heart failure')
# Correlation values

df_ad = df_train.groupby('diabetes')['DEATH_EVENT'].value_counts().to_frame()

print('Proportion of people without diabetes people who have Heart failure:', ((df_ad.loc[(0, 1), :]/((df_ad.loc[(0, 0), :]) + (df_ad.loc[(0, 1), :])))*100).values) 
print('Proportion of people with diabetes who have Heart failure:', ((df_ad.loc[(1, 1), :]/((df_ad.loc[(1, 0), :]) + (df_ad.loc[(1, 1), :])))*100).values) 
# Diabetes, Heart failure and other categorical risk factors

fig, ax = plt.subplots(1,4, figsize=(20,5))
x=0

# Ordinal variables such as high blood pressure, anaemia and smoking are mapped on Y axis and hue is kept as diabetes for better interpretation 
for variable in ['anaemia', 'high_blood_pressure', 'smoking']:
    sns.barplot(x='DEATH_EVENT', y=variable, hue='diabetes' , data=df_train, ax=ax[x])
    x+=1

# Sex is a nominal variable. Hence, diabetes is mapped on y axis and hue is kept as sex for better interpretation
sns.barplot(x='DEATH_EVENT', y='diabetes', hue='sex' , data=df_train, ax=ax[3])
# Correlation values 

for variable in ['anaemia', 'high_blood_pressure', 'smoking']:
    for x in [0,1]:
        print('For patients with diabetes = ', x, 'correlation of', variable, 'with heart failure is', 
              ((df_train[df_train['diabetes']==x]).corr()[variable]['DEATH_EVENT']))
        
for x in [0,1]:
    print('For patients with sex = ', x, 'correlation of diabetes with heart failure is', ((df_train[df_train['sex']==x]).corr()['diabetes']['DEATH_EVENT']))
# Diabetes, Heart failure and other numerical risk factors 

fig, ax = plt.subplots(1,3, figsize=(20,8))
x=0

for variable in ['creatinine_phosphokinase', 'platelets', 'serum_sodium']:
    sns.boxplot(y=variable, x='DEATH_EVENT', hue='diabetes', showfliers=False, data=df_train, ax=ax[x])
    x+=1
# Correlation values

for variable in ['creatinine_phosphokinase', 'platelets', 'serum_sodium']:
    for x in [0,1]:
        print('For patients with diabetes =', x, 'correlation of heart failure and', variable, 'is',
              ((df_train[df_train['diabetes']==x]).corr()[variable]['DEATH_EVENT']))
# Distribution of Smoking vs Heart failure 

fig, ax = plt.subplots()
sns.distplot(df[df['DEATH_EVENT']==0]['smoking'], hist=False, label = 'No heart failure')
sns.distplot(df[df['DEATH_EVENT']==1]['smoking'], hist=False, label = 'Heart failure')
# Correlation values

df_ad = df_train.groupby('smoking')['DEATH_EVENT'].value_counts().to_frame()

print('Proportion of non-smokers who have Heart failure:', ((df_ad.loc[(0, 1), :]/((df_ad.loc[(0, 0), :]) + (df_ad.loc[(0, 1), :])))*100).values) 
print('Proportion of smokers who have Heart failure:', ((df_ad.loc[(1, 1), :]/((df_ad.loc[(1, 0), :]) + (df_ad.loc[(1, 1), :])))*100).values) 
# Smoking, Heart failure and other categorical risk factors

fig, ax = plt.subplots(1,4, figsize=(20,5))
x=0

# Ordinal variables such as high blood pressure, anaemia and diabetes are mapped on Y axis and hue is kept as smoking for better interpretation 
for variable in ['anaemia', 'high_blood_pressure', 'diabetes']:
    sns.barplot(x='DEATH_EVENT', y=variable, hue='smoking' , data=df_train, ax=ax[x])
    x+=1

# Sex is a nominal variable. Hence, smoking is mapped on y axis and hue is kept as sex for better interpretation
sns.barplot(x='DEATH_EVENT', y='smoking', hue='sex' , data=df_train, ax=ax[3])
# Correlation values 

for variable in ['anaemia', 'high_blood_pressure', 'diabetes']:
    for x in [0,1]:
        print('For patients with smoking = ', x, 'correlation of', variable, 'with heart failure is', 
              ((df_train[df_train['smoking']==x]).corr()[variable]['DEATH_EVENT']))
        
for x in [0,1]:
    print('For patients with sex = ', x, 'correlation of smoking with heart failure is', ((df_train[df_train['sex']==x]).corr()['smoking']['DEATH_EVENT']))
# Smoking, Heart failure and other numerical risk factors 

fig, ax = plt.subplots(1,3, figsize=(20,8))
x=0

for variable in ['creatinine_phosphokinase', 'platelets', 'serum_sodium']:
    sns.boxplot(y=variable, x='DEATH_EVENT', hue='smoking', showfliers=False, data=df_train, ax=ax[x])
    x+=1
# Correlation values

for variable in ['creatinine_phosphokinase', 'platelets', 'serum_sodium']:
    for x in [0,1]:
        print('For patients with smoking =', x, 'correlation of heart failure and', variable, 'is',
              ((df_train[df_train['smoking']==x]).corr()[variable]['DEATH_EVENT']))
# Distribution of Sex vs Heart failure 

fig, ax = plt.subplots()
sns.distplot(df[df['DEATH_EVENT']==0]['sex'], hist=False, label = 'No heart failure')
sns.distplot(df[df['DEATH_EVENT']==1]['sex'], hist=False, label = 'Heart failure')
# Correlation values

df_ad = df_train.groupby('sex')['DEATH_EVENT'].value_counts().to_frame()

print('Proportion of females who have Heart failure:', ((df_ad.loc[(0, 1), :]/((df_ad.loc[(0, 0), :]) + (df_ad.loc[(0, 1), :])))*100).values) 
print('Proportion of males who have Heart failure:', ((df_ad.loc[(1, 1), :]/((df_ad.loc[(1, 0), :]) + (df_ad.loc[(1, 1), :])))*100).values) 
# Relation between smoking, heart failure and other categorical risk factors has already been explored in the previous sections
# Sex, Heart failure and other numerical risk factors 

fig, ax = plt.subplots(1,3, figsize=(20,8))
x=0

for variable in ['creatinine_phosphokinase', 'platelets', 'serum_sodium']:
    sns.boxplot(y=variable, x='DEATH_EVENT', hue='sex', showfliers=False, data=df_train, ax=ax[x])
    x+=1
# Correlation values

for variable in ['creatinine_phosphokinase', 'platelets', 'serum_sodium']:
    for x in [0,1]:
        print('For patients with sex =', x, 'correlation of heart failure and', variable, 'is',
              ((df_train[df_train['sex']==x]).corr()[variable]['DEATH_EVENT']))
# Shuffling

df_train = shuffle(df_train)
df_validation = shuffle(df_validation)
# Standard scaling of training data
X_train = df_train.copy()[['time', 'ejection_fraction', 'serum_creatinine', 'age']]

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)

y_train = df_train['DEATH_EVENT']
# Standard scaling of validation data
X_validation = df_validation.copy()[['time', 'ejection_fraction', 'serum_creatinine', 'age']]

scaler = StandardScaler()

X_validation = scaler.fit_transform(X_validation)

y_validation = df_validation['DEATH_EVENT']
### Model fitting and Evaluation
# SGDclassifier and performance analysis

sgd_clf = SGDClassifier()

# Fit the model on training data
sgd_clf.fit(X_train, y_train)

# Predict the values for validation data
y_validation_pred = sgd_clf.predict(X_validation)

# Confusion matrix for validation data predictions
print('Confusion matrix for validation data:' '\n', confusion_matrix(y_validation, y_validation_pred))

# Accuracy score for validation data predictions 
acc_score = accuracy_score(y_validation, y_validation_pred)
print('Accuracy score for validation data:', acc_score)
      
# Precision score for validation data predictions 
pre_score = precision_score(y_validation, y_validation_pred)
print('Precision score for validation data:', pre_score)
      
# Recall score for validation data predictions 
rec_score = recall_score(y_validation, y_validation_pred)
print('Recall score for validation data:', rec_score)

accu_scores = []
rec_scores = []
accu_scores.append(acc_score)
rec_scores.append(rec_score)
# Logistic regression and performance analysis

logreg = LogisticRegression()

# Fit the model on training data
logreg.fit(X_train, y_train)

# Predict the values for validation data
y_validation_pred = logreg.predict(X_validation)

# Confusion matrix for validation data predictions
print('Confusion matrix for validation data:' '\n', confusion_matrix(y_validation, y_validation_pred))

# Accuracy score for validation data predictions 
acc_score = accuracy_score(y_validation, y_validation_pred)
print('Accuracy score for validation data:', acc_score)
      
# Precision score for validation data predictions 
pre_score = precision_score(y_validation, y_validation_pred)
print('Precision score for validation data:', pre_score)
      
# Recall score for validation data predictions 
rec_score = recall_score(y_validation, y_validation_pred)
print('Recall score for validation data:', rec_score)

accu_scores.append(acc_score)
rec_scores.append(rec_score)
# Selecting the right hyperparameter

accuracyscores = []
recallscores = []

for c in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
    svc = SVC(C=c, random_state=0, kernel='rbf')
    svc.fit(X_train, y_train)
    y_validation_pred = svc.predict(X_validation)
    accuracyscores.append(accuracy_score(y_validation, y_validation_pred))
    recallscores.append(recall_score(y_validation, y_validation_pred))

plt.plot([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], accuracyscores, label='accuracy')
plt.plot([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], recallscores, label='recall')
plt.legend()

# SVC and performance analysis

svc = SVC(C=1.0, random_state=0, kernel='rbf')

# Fit the model on training data
svc.fit(X_train, y_train)

# Predict the values for validation data
y_validation_pred = svc.predict(X_validation)

# Confusion matrix for validation data predictions
print('Confusion matrix for validation data:' '\n', confusion_matrix(y_validation, y_validation_pred))

# Accuracy score for validation data predictions 
acc_score = accuracy_score(y_validation, y_validation_pred)
print('Accuracy score for validation data:', acc_score)
      
# Precision score for validation data predictions 
pre_score = precision_score(y_validation, y_validation_pred)
print('Precision score for validation data:', pre_score)
      
# Recall score for validation data predictions 
rec_score = recall_score(y_validation, y_validation_pred)
print('Recall score for validation data:', rec_score)

accu_scores.append(acc_score)
rec_scores.append(rec_score)
# Selecting the right hyperparameter

accuracyscores = []
recallscores = []

for neighbors in range(3,10):
    knn = KNeighborsClassifier(n_neighbors=neighbors, metric='minkowski')
    knn.fit(X_train, y_train)
    y_validation_pred = knn.predict(X_validation)
    accuracyscores.append(accuracy_score(y_validation, y_validation_pred))
    recallscores.append(recall_score(y_validation, y_validation_pred))

plt.plot(list(range(3,10)), accuracyscores, label='accuracy')
plt.plot(list(range(3,10)), recallscores, label='recall')
plt.legend()
# KNN and performance analysis

knn = KNeighborsClassifier(n_neighbors=5)

# Fit the model on training data
knn.fit(X_train, y_train)

# Predict the values for validation data
y_validation_pred = knn.predict(X_validation)

# Confusion matrix for validation data predictions
print('Confusion matrix for validation data:' '\n', confusion_matrix(y_validation, y_validation_pred))

# Accuracy score for validation data predictions 
acc_score = accuracy_score(y_validation, y_validation_pred)
print('Accuracy score for validation data:', acc_score)
      
# Precision score for validation data predictions 
pre_score = precision_score(y_validation, y_validation_pred)
print('Precision score for validation data:', pre_score)
      
# Recall score for validation data predictions 
rec_score = recall_score(y_validation, y_validation_pred)
print('Recall score for validation data:', rec_score)

accu_scores.append(acc_score)
rec_scores.append(rec_score)
# Selecting the right hyperparameter

accuracyscores = []
recallscores = []

for leaves in range(2,10):
    dt = DecisionTreeClassifier(max_leaf_nodes = leaves, random_state=0, criterion='entropy')
    dt.fit(X_train, y_train)
    y_validation_pred = dt.predict(X_validation)
    accuracyscores.append(accuracy_score(y_validation, y_validation_pred))
    recallscores.append(recall_score(y_validation, y_validation_pred))

plt.plot(list(range(2,10)), accuracyscores, label='accuracy')
plt.plot(list(range(2,10)), recallscores, label='recall')
plt.legend()
# Decision tree and performance analysis

dt = DecisionTreeClassifier(max_leaf_nodes = 4, random_state=0, criterion='entropy')

# Fit the model on training data
dt.fit(X_train, y_train)

# Predict the values for validation data
y_validation_pred = dt.predict(X_validation)

# Confusion matrix for validation data predictions
print('Confusion matrix for validation data:' '\n', confusion_matrix(y_validation, y_validation_pred))

# Accuracy score for validation data predictions 
acc_score = accuracy_score(y_validation, y_validation_pred)
print('Accuracy score for validation data:', acc_score)
      
# Precision score for validation data predictions 
pre_score = precision_score(y_validation, y_validation_pred)
print('Precision score for validation data:', pre_score)
      
# Recall score for validation data predictions 
rec_score = recall_score(y_validation, y_validation_pred)
print('Recall score for validation data:', rec_score)

accu_scores.append(acc_score)
rec_scores.append(rec_score)
# Selecting the right hyperparameter

accuracyscores = []
recallscores = []

for estimators in range(10,30):
    rf = RandomForestClassifier(n_estimators = estimators, random_state=0, criterion='entropy')
    rf.fit(X_train, y_train)
    y_validation_pred = rf.predict(X_validation)
    accuracyscores.append(accuracy_score(y_validation, y_validation_pred))
    recallscores.append(recall_score(y_validation, y_validation_pred))

plt.plot(list(range(10,30)), accuracyscores, label='accuracy')
plt.plot(list(range(10,30)), recallscores, label='recall')
plt.legend()
# Random forest and performance analysis

rf = RandomForestClassifier(n_estimators = 26, random_state=0, criterion='entropy')

# Fit the model on training data
rf.fit(X_train, y_train)

# Predict the values for validation data
y_validation_pred = rf.predict(X_validation)

# Confusion matrix for validation data predictions
print('Confusion matrix for validation data:' '\n', confusion_matrix(y_validation, y_validation_pred))

# Accuracy score for validation data predictions 
acc_score = accuracy_score(y_validation, y_validation_pred)
print('Accuracy score for validation data:', acc_score)
      
# Precision score for validation data predictions 
pre_score = precision_score(y_validation, y_validation_pred)
print('Precision score for validation data:', pre_score)
      
# Recall score for validation data predictions 
rec_score = recall_score(y_validation, y_validation_pred)
print('Recall score for validation data:', rec_score)

accu_scores.append(acc_score)
rec_scores.append(rec_score)
# Plotting the accuracy and recall scores for all models

List = ['SGDRegressor', 'LogisticRegression', 'SVC', 'KNN', 'DecisionTree', 'RandomForest']

plt.figure(figsize=(20,15))
plt.plot(List, accu_scores, label='accuracy')
plt.plot(List, rec_scores, label='recall')
plt.legend()

plt.xlabel("Classifier Models", fontsize = 20 )
plt.ylabel("% of Accuracy/Recall", fontsize = 20)
plt.title("Accuracy/Recall of different Classifier Models", fontsize = 20)

plt.xticks(fontsize = 12, horizontalalignment = 'center', rotation = 8)
plt.yticks(fontsize = 13)
# Preparing the test data

X_test = df_test.copy()[['time', 'ejection_fraction', 'serum_creatinine', 'age']]
X_test = scaler.fit_transform(X_test)

y_test = df_test['DEATH_EVENT']
# Predict test values using SVC

# Predict y values
y_test_pred = svc.predict(X_test)

# Confusion matrix for test data predictions
print('Confusion matrix for test data:' '\n', confusion_matrix(y_test, y_test_pred))

# Accuracy score for test data predictions 
print('Accuracy score for test data:', accuracy_score(y_test, y_test_pred))
      
# Precision score for test data predictions 
print('Precision score for test data:', precision_score(y_test, y_test_pred))
      
# Recall score for test data predictions 
print('Recall score for test data:', recall_score(y_test, y_test_pred))
# Predict test values using KNN

# Predict y values
y_test_pred = knn.predict(X_test)

# Confusion matrix for test data predictions
print('Confusion matrix for test data:' '\n', confusion_matrix(y_test, y_test_pred))

# Accuracy score for test data predictions 
print('Accuracy score for test data:', accuracy_score(y_test, y_test_pred))
      
# Precision score for test data predictions 
print('Precision score for test data:', precision_score(y_test, y_test_pred))
      
# Recall score for test data predictions 
print('Recall score for test data:', recall_score(y_test, y_test_pred))