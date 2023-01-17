# Check the versions of libraries



# Python version

import sys

print('Python: {}'.format(sys.version))

# scipy

import scipy

print('scipy: {}'.format(scipy.__version__))

# numpy

import numpy

print('numpy: {}'.format(numpy.__version__))

# matplotlib

import matplotlib

print('matplotlib: {}'.format(matplotlib.__version__))

# pandas

import pandas

print('pandas: {}'.format(pandas.__version__))

# scikit-learn

import sklearn

print('sklearn: {}'.format(sklearn.__version__))
# data analysis and wrangling

import pandas as pd

import numpy as np

import random as rnd



# visualization

import seaborn as sns

from scipy.stats import norm

import matplotlib.pyplot as plt

%matplotlib inline



# machine learning

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC, LinearSVC

from sklearn.svm import LinearSVC

from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.linear_model import Perceptron

from sklearn.linear_model import SGDClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis



## Load metrics for predictive modeling

from sklearn.model_selection import train_test_split

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import StratifiedKFold

from sklearn.metrics import classification_report

from sklearn.metrics import confusion_matrix

from sklearn.metrics import accuracy_score

from sklearn.metrics import roc_curve, auc
# Load dataset train and test

train_titanic = pd.read_csv('../input/titanic/train.csv')

test_titanic = pd.read_csv('../input/titanic/test.csv')



# Create titanis list with both dataset to use same cleansing methods

titanic_list = [train_titanic, test_titanic]
# Check dataframe structure

for information in titanic_list:

    information.info()

    print('_'*40)
# Check dataframe basic stats data

for stats in titanic_list:

    print(stats)

    print('_'*40)
# Check test dataframe basic stats data

for descrip in titanic_list:

    print(descrip.describe())

    print('_'*40)
# Check null and NA values for both dataset

for nuls in titanic_list:

    print(nuls.isna().sum())

    print('_'*40)
# Table of relative frequency

for nuls in titanic_list:

    print(nuls.isnull().sum()/len(nuls)*100)

    print('_'*40)
# Check first 10 elements

for passenger in titanic_list:

    print(passenger['PassengerId'].head(10))

    print('_'*40)
# Remove PassengerId variable only for train dataset

titanic_list[0].drop(['PassengerId'], axis=1, inplace=True)
# Check train dataset

titanic_list[0].head()
sns.barplot(x="Survived", data=titanic_list[0])
titanic_list[0].describe()['Survived']
titanic_list[0][['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean()
sns.barplot(x="Pclass", y="Survived", data=titanic_list[0])
# Check the survived ratio with sex

titanic_list[0][["Sex", "Survived"]].groupby(['Sex'], as_index=False).mean()
sns.barplot(x="Sex", y="Survived", data=titanic_list[0])
# Convert categorical variable to binary variable - female 1 and male 0

for genre in titanic_list:

    genre['Sex'] = genre['Sex'].map( {'female': 1, 'male': 0} ).astype(int)
# Check Sex features

titanic_list[0].head()
titanic_list[1].head()
titanic_list[0][["SibSp", "Survived"]].groupby(['SibSp'], as_index=False).mean()
sns.barplot(x="SibSp", y="Survived", data=titanic_list[0])
titanic_list[0][["Parch", "Survived"]].groupby(['Parch'], as_index=False).mean()
sns.barplot(x="Parch", y="Survived", data=titanic_list[0])
# Create a new feature

for famsize in titanic_list:

    famsize['FamilySize'] = famsize['SibSp'] + famsize['Parch'] + 1
titanic_list[0][["FamilySize", "Survived"]].groupby(['FamilySize'], as_index=False).mean()
sns.barplot(x="FamilySize", y="Survived", data=titanic_list[0])
for alone in titanic_list:

    alone['IsAlone'] = 0

    alone.loc[alone['FamilySize'] == 1, 'IsAlone'] = 1



# Check new feature with predictor    

titanic_list[0][['IsAlone', 'Survived']].groupby(['IsAlone'], as_index=False).mean()
sns.barplot(x="IsAlone", y="Survived", data=titanic_list[0])
# Check new features in dataset train

titanic_list[1].head()
# We remove Ticket variable in both traing and test dataset

for tick in titanic_list:

    tick.drop(['Ticket'], axis=1, inplace=True)
# We check the dataset again - train

titanic_list[0].head(10)
# ...and test dataset

titanic_list[1].head(10)
# Check ratio Embarked and Survived variable

titanic_list[0][['Embarked', 'Survived']].groupby(['Embarked'], as_index=False).mean()
# Fill na or null values with the most frequent value, C

freq_port = titanic_list[0].Embarked.dropna().mode()[0]

freq_port
# Assign result on the dataset

for port in titanic_list:

    port['Embarked'] = port['Embarked'].fillna(freq_port)
sns.barplot(x="Embarked", y="Survived", data=titanic_list[0])
sns.distplot(titanic_list[0]['Fare'], fit=norm)
for f in titanic_list:

    f['Fare'] = np.log1p(f['Fare'])

sns.distplot(titanic_list[0]['Fare'], fit=norm)
for faregr in titanic_list:

    faregr['FareGroup'] = pd.qcut(faregr['Fare'], 7, labels=['A', 'B', 'C', 'D', 'E', 'F', 'G'])





titanic_list[0][['FareGroup', 'Survived']].groupby(['FareGroup'], as_index=False).mean()
sns.barplot(x="FareGroup", y="Survived", data=titanic_list[0])
# We remove the variable Fare

for fares in titanic_list:

    fares.drop(['Fare'], axis=1, inplace=True)
for cab in titanic_list:

    cab['InCabin'] = ~cab['Cabin'].isnull()
sns.barplot(x="InCabin", y="Survived", data=titanic_list[0])

plt.show()
# We remove the variable Cabin

for cabin in titanic_list:

    cabin.drop(['Cabin'], axis=1, inplace=True)
bins = [-1, 0, 5, 12, 18, 24, 35, 60, np.inf]

labels = ['Unknown', 'Baby', 'Child', 'Teenager', 'Student', 'Young Adult', 'Adult', 'Senior']

for age in titanic_list:

    age["Age"] = age["Age"].fillna(-0.5)

    age['AgeGroup'] = pd.cut(age["Age"], bins, labels = labels)
sns.barplot(x="AgeGroup", y="Survived", data=titanic_list[0])

plt.show()
# We remove the variable Age

#for a in titanic_list:

#    a.drop(['Age'], axis=1, inplace=True)
# Check the names

titanic_list[0]['Name'].head(10)
# Create the function to extract the title

import re

def get_title(name):

    title_search = re.search(' ([A-Za-z]+)\.', name)

    if title_search:

        return title_search.group(1)

    return ""



# Apply get_title function

for title in titanic_list:

    title['Title'] = title['Name'].apply(get_title)



# Check the results

pd.crosstab(titanic_list[0]['Title'], titanic_list[0]['Sex'])
# Create a categorization on train dataset

for t in titanic_list:

    t['Title'] = t['Title'].replace(['Lady', 'Countess','Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')

    t['Title'] = t['Title'].replace('Mlle', 'Miss')

    t['Title'] = t['Title'].replace('Ms', 'Miss')

    t['Title'] = t['Title'].replace('Mme', 'Mrs')



# We create a relative table

titanic_list[0][['Title', 'Survived']].groupby(['Title'], as_index=False).mean()
sns.barplot(x="Title", y="Survived", data=titanic_list[0])

plt.show()
# Remove Name variable

for name in titanic_list:

    name.drop(['Name'], axis=1, inplace=True)
# Check all values and new features

titanic_list[0].head(10)
titanic_list[1].head(10)
correlation_matrix = titanic_list[0].corr()

correlation_matrix



plt.figure(figsize=(10,10))

plt.title('Pearson Correlation of Features', y=1.05, size=15)

sns.heatmap(correlation_matrix, annot=True)
# The procedure is very simple, binarizing categorical variable for training dataset

cols = ['Pclass', 'Embarked', 'FareGroup', 'AgeGroup', 'Title']

titanic_categorical = titanic_list[0][cols]

titanic_categorical = pd.concat([pd.get_dummies(titanic_categorical[col], prefix=col) for col in titanic_categorical], axis=1)

titanic_categorical.head()

train_titanic = pd.concat([titanic_list[0][titanic_list[0].columns[~titanic_list[0].columns.isin(cols)]], titanic_categorical], axis=1)

train_titanic.head()
correlation_matrix = train_titanic.corr()

correlation_matrix



#plt.figure(figsize=(10,10))

fig, ax = plt.subplots(figsize=(20, 10))

plt.title('Pearson Correlation of Features', y=1.05, size=15)

sns.heatmap(correlation_matrix, annot=True)
# Binarizing variable for testing dataset

titanic_categorical = titanic_list[1][cols]

titanic_categorical = pd.concat([pd.get_dummies(titanic_categorical[col], prefix=col) for col in titanic_categorical], axis=1)

test_titanic = pd.concat([titanic_list[1][titanic_list[1].columns[~titanic_list[1].columns.isin(cols)]], titanic_categorical], axis=1)

test_titanic.head()
# Backup train and test dataset

train_bak = train_titanic

test_bak = test_titanic
# Version 1 

# Modeling with only high correlation features - Drop SibSb - Parch

for feature in train_titanic, test_titanic:

    feature.drop(['SibSp'], axis=1, inplace=True)

    feature.drop(['Parch'], axis=1, inplace=True)
train_titanic.head(10)
test_titanic.head(10)
# Split and drop Survived variable

X_train = train_titanic.drop('Survived', axis=1)

y_train = train_titanic['Survived']



# Drop PassengerId variable on test dataset

ids = test_titanic[['PassengerId']] # create a sub-dataset for submission file and saving it 

test_titanic = test_titanic.drop('PassengerId', axis=1).copy()



# Create train and test 80-20 with seed fixed to 42 for validation the model

X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.20, random_state=42)
X_train.shape, y_train.shape, X_test.shape, y_test.shape
X_train.head()
y_train.head()
X_test.head()
y_test.head()
# Create a performance_auc dict

performance_auc = {}
model = LogisticRegression().fit(X_train, y_train)

model
predicted_log = model.predict(X_test)

predicted_log
# Confidence score

logreg_score = round(model.score(X_train,y_train) * 100, 2)



print(logreg_score)



print(classification_report(y_test, predicted_log))
# Create a confusion matrix

matrix = confusion_matrix(y_test, predicted_log)

sns.heatmap(matrix, annot=True, fmt="d", cmap='Blues', square=True)

plt.xlabel("predicted")

plt.ylabel("actual")

plt
# Visualize results by ROC graph

fpr, tpr, thresholds = roc_curve(y_test, predicted_log)

roc_auc = auc(fpr, tpr)

performance_auc['Logistic Regression'] = roc_auc



# Plotting

plt.figure()

lw = 2

plt.plot(fpr, tpr, color='darkorange', lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)

plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')

plt.xlim([0.0, 1.0])

plt.ylim([0.0, 1.05])

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.title('Receiver operating characteristic example')

plt.legend(loc="lower right")

plt.show()
model = DecisionTreeClassifier().fit(X_train, y_train)

model
predicted_dt = model.predict(X_test)

predicted_dt
# Confidence score

dectree_score = round(model.score(X_train,y_train) * 100, 2)

print(dectree_score)

print(classification_report(y_test, predicted_dt))
# Create a confusion matrix

matrix = confusion_matrix(y_test, predicted_dt)

sns.heatmap(matrix, annot=True, fmt="d", cmap='Blues', square=True)

plt.xlabel("predicted")

plt.ylabel("actual")

plt
# Visualize results by ROC graph

fpr, tpr, thresholds = roc_curve(y_test, predicted_dt)

roc_auc = auc(fpr, tpr)

performance_auc['Decision Tree'] = roc_auc



# Plotting

plt.figure()

lw = 2

plt.plot(fpr, tpr, color='darkorange', lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)

plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')

plt.xlim([0.0, 1.0])

plt.ylim([0.0, 1.05])

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.title('Receiver operating characteristic example')

plt.legend(loc="lower right")

plt.show()
pd.concat((pd.DataFrame(X_train.iloc[:, 1:].columns, columns = ['variable']), 

           pd.DataFrame(model.feature_importances_, columns = ['importance'])), 

          axis = 1).sort_values(by='importance', ascending = False)[:20]
model = RandomForestClassifier(1000).fit(X_train, y_train)

model
predicted_rf = model.predict(X_test)

predicted_rf
# Confidence score

randfor_score = round(model.score(X_train,y_train) * 100, 2)

print(randfor_score)

print(classification_report(y_test, predicted_rf))
matrix = confusion_matrix(y_test, predicted_rf)

sns.heatmap(matrix, annot=True, fmt="d", cmap='Blues', square=True)

plt.xlabel("predicted")

plt.ylabel("actual")

plt
fpr, tpr, thresholds = roc_curve(y_test, predicted_rf)

roc_auc = auc(fpr, tpr)

performance_auc['Random Forests'] = roc_auc



plt.figure()

lw = 2

plt.plot(fpr, tpr, color='darkorange', lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)

plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')

plt.xlim([0.0, 1.0])

plt.ylim([0.0, 1.05])

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.title('Receiver operating characteristic example')

plt.legend(loc="lower right")

plt.show()
pd.concat((pd.DataFrame(X_train.iloc[:, 1:].columns, columns = ['variable']), 

           pd.DataFrame(model.feature_importances_, columns = ['importance'])), 

          axis = 1).sort_values(by='importance', ascending = False)[:20]
model = KNeighborsClassifier(3).fit(X_train, y_train)

model
predicted_knn = model.predict(X_test)

predicted_knn
# Confidence score

knn_score = round(model.score(X_train,y_train) * 100, 2)

print(knn_score)

print(classification_report(y_test, predicted_knn))
matrix = confusion_matrix(y_test, predicted_knn)

sns.heatmap(matrix, annot=True, fmt="d", cmap='Blues', square=True)

plt.xlabel("predicted")

plt.ylabel("actual")

plt
fpr, tpr, thresholds = roc_curve(y_test, predicted_knn)

roc_auc = auc(fpr, tpr)

performance_auc['k-nearest neighbours'] = roc_auc



plt.figure()

lw = 2

plt.plot(fpr, tpr, color='darkorange', lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)

plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')

plt.xlim([0.0, 1.0])

plt.ylim([0.0, 1.05])

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.title('Receiver operating characteristic example')

plt.legend(loc="lower right")

plt.show()
model = SVC(probability=True).fit(X_train, y_train)

model
predicted_sv = model.predict(X_test)

predicted_sv
# Confidence score

svm_score = round(model.score(X_train,y_train) * 100, 2)

print(svm_score)

print(classification_report(y_test, predicted_sv))
matrix = confusion_matrix(y_test, predicted_sv)

sns.heatmap(matrix, annot=True, fmt="d", cmap='Blues', square=True)

plt.xlabel("predicted")

plt.ylabel("actual")

plt
fpr, tpr, thresholds = roc_curve(y_test, predicted_sv)

roc_auc = auc(fpr, tpr)

performance_auc['SVM'] = roc_auc



plt.figure()

lw = 2

plt.plot(fpr, tpr, color='darkorange', lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)

plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')

plt.xlim([0.0, 1.0])

plt.ylim([0.0, 1.05])

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.title('Receiver operating characteristic example')

plt.legend(loc="lower right")

plt.show()
perf = pd.DataFrame.from_dict(performance_auc, orient='index')

perf['Model'] = perf.index

perf['AUC'] = perf[0]

plt.xlabel('AUC')

plt.title('Classifier AUC')

sns.set_color_codes("muted")

sns.barplot(x='AUC', y='Model', data=perf, color="b")
models = pd.DataFrame({

    'Model': ['Support Vector Machines', 'KNN', 'Logistic Regression', 

              'Random Forest', 'Decision Tree'],

    'Score': [svm_score, 

              knn_score, 

              logreg_score, 

              randfor_score,

              dectree_score ]})

models.sort_values(by='Score', ascending=False)