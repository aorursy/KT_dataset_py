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
#Import libraries

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

import warnings

warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split

import random

from sklearn.preprocessing import StandardScaler

import time
#Import Training and Test data

train_data = pd.read_csv("../input/titanic/train.csv")

test_data = pd.read_csv("../input/titanic/test.csv")
#Data Analysis

train_data.describe()
train_data.info()
#Check for missing values in the training dataset

train_data.isnull().any()
#Get a count of the missing values in training data

train_data.isnull().sum()
#Analyze test dataset

test_data.describe()
#Check for missing values in the test dataset

test_data.isnull().any()
#Get a count of the missing values in test dataset

test_data.isnull().sum()
# Data Preparation - Handle missing values

#Training dataset

# Drop 2 rows with no embarkment information

# Drop column "Cabin" as it has lot of missing values

# Replace missing ages with mean value of the age columns



#Test dataset

# Drop column "Cabin" as it has lot of missing values

# Replace missing ages with mean value of the age columns

# Replace missing fare with mean value of the fare columns



#Training data

train_data = train_data[pd.notnull(train_data['Embarked'])]

train_data = train_data.drop(columns = ['Cabin'])

train_data['Age'].fillna((train_data['Age'].mean()), inplace=True)



#Test data

test_data = test_data.drop(columns = ['Cabin'])

test_data['Age'].fillna((test_data['Age'].mean()), inplace=True)

test_data['Fare'].fillna((test_data['Fare'].mean()), inplace=True)
#Histogram of Numerical Columns

#Delete passenger id and response variable i.e Survived

train_data2 = train_data.drop(columns=['PassengerId','Survived'])
plt.figure(figsize=(20,15))

plt.suptitle('Histogram of Numerical Columns', fontsize = 20)

for i in range(1,train_data2.shape[1]+1):

    plt.subplot(3,3,i)

    f = plt.gca()

    f.axes.get_yaxis().set_visible(False)

    f.set_title(train_data2.columns.values[i-1])

    

    vals = np.size(train_data2.iloc[:,i-1].unique())

    plt.hist(train_data2.iloc[:, i - 1], bins = vals, color='#3F5D7D')



plt.tight_layout(rect=[0, 0.03, 1, 0.95])
##Pie plots

# Prepare data set that has only binary columns



train_data3 = train_data[['SibSp', 'Sex','Embarked']]



fig = plt.figure(figsize=(15,12))

plt.suptitle('Pie Chart Distributions', fontsize=20)

for i in range(1,train_data3.shape[1]+1):

    plt.subplot(1,3,i)

    f=plt.gca()

    f.axes.get_yaxis().set_visible(False)

    f.set_title(train_data3.columns.values[i-1])

    

    values = train_data3.iloc[:, i - 1].value_counts(normalize = True).values

    index = train_data3.iloc[:, i - 1].value_counts(normalize = True).index

    plt.pie(values, labels = index, autopct='%1.1f%%')

    plt.axis('equal')



plt.tight_layout(rect=[0, 0.03, 1, 0.95])
#Exploring uneven features

train_data2.corrwith(train_data.Survived).plot.bar(figsize=(20,10),

                         title = 'Correlation with Response Variable',

                         fontsize = 15, rot = 45, grid = True)
## Correlation Matrix

sns.set(style="white")



# Compute the correlation matrix

corr = train_data.drop(columns = ['PassengerId', 'Survived']).corr()



# Generate a mask for the upper triangle

mask = np.zeros_like(corr, dtype=np.bool)

mask[np.triu_indices_from(mask)] = True



# Set up the matplotlib figure

f, ax = plt.subplots(figsize=(18, 15))



# Generate a custom diverging colormap

cmap = sns.diverging_palette(220, 10, as_cmap=True)



# Draw the heatmap with the mask and correct aspect ratio

sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,

            square=True, linewidths=.5, cbar_kws={"shrink": .5})
###Data Preprocessing
#Store user identifier and target valriable

train_PassengerId = train_data['PassengerId']

test_PassengerId = test_data['PassengerId']
#Drop the columns which are not required

train_data_updated = train_data.drop(columns=['PassengerId','Name','Ticket','Fare'])

test_data_updated = test_data.drop(columns=['PassengerId','Name','Ticket','Fare'])
#One Hot Encoding to convert categorical values to binary

train_data_updated = pd.get_dummies(train_data_updated)

test_data_updated = pd.get_dummies(test_data_updated)
#Split the data into training and test set

dataset = train_data_updated

X_train,X_test,y_train,y_test = train_test_split(dataset.drop(columns=['Survived']), dataset['Survived'],

                                                test_size = 0.20,

                                                random_state = 0)
#Feature scaling

from sklearn.preprocessing import StandardScaler

sc_X = StandardScaler()

X_train2 = pd.DataFrame(sc_X.fit_transform(X_train))

X_test2 = pd.DataFrame(sc_X.transform(X_test))

test_data_updated2 = pd.DataFrame(sc_X.transform(test_data_updated))

X_train2.columns = X_train.columns.values

X_test2.columns = X_test.columns.values

test_data_updated2.columns = test_data_updated.columns.values

X_train2.index = X_train.index.values

X_test2.index = X_test.index.values

test_data_updated2.index = test_data_updated.index.values



X_train = X_train2

X_test = X_test2

test_data_updated = test_data_updated2
#### Model Building ####



### Comparing Models



## Logistic Regression

from sklearn.linear_model import Lasso,LogisticRegression

classifier = LogisticRegression(random_state = 0, penalty = 'l2')

classifier.fit(X_train, y_train)



# Predicting Test Set

y_pred = classifier.predict(X_test)

from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, precision_score, recall_score

acc = accuracy_score(y_test, y_pred)

prec = precision_score(y_test, y_pred)

rec = recall_score(y_test, y_pred)

f1 = f1_score(y_test, y_pred)



results = pd.DataFrame([['Linear Regression (Lasso)', acc, prec, rec, f1]],

               columns = ['Model', 'Accuracy', 'Precision', 'Recall', 'F1 Score'])





## SVM (Linear)

from sklearn.svm import SVC

classifier = SVC(random_state = 0, kernel = 'linear')

classifier.fit(X_train, y_train)



# Predicting Test Set

y_pred = classifier.predict(X_test)

acc = accuracy_score(y_test, y_pred)

prec = precision_score(y_test, y_pred)

rec = recall_score(y_test, y_pred)

f1 = f1_score(y_test, y_pred)



model_results = pd.DataFrame([['SVM (Linear)', acc, prec, rec, f1]],

               columns = ['Model', 'Accuracy', 'Precision', 'Recall', 'F1 Score'])



results = results.append(model_results, ignore_index = True)





## SVM (rbf)

from sklearn.svm import SVC

classifier = SVC(random_state = 0, kernel = 'rbf')

classifier.fit(X_train, y_train)



# Predicting Test Set

y_pred = classifier.predict(X_test)

acc = accuracy_score(y_test, y_pred)

prec = precision_score(y_test, y_pred)

rec = recall_score(y_test, y_pred)

f1 = f1_score(y_test, y_pred)



model_results = pd.DataFrame([['SVM (RBF)', acc, prec, rec, f1]],

               columns = ['Model', 'Accuracy', 'Precision', 'Recall', 'F1 Score'])



results = results.append(model_results, ignore_index = True)





## SVM (rbf)

from sklearn.ensemble import RandomForestClassifier

classifier = RandomForestClassifier(random_state = 0, n_estimators = 100,

                                    criterion = 'entropy')

classifier.fit(X_train, y_train)



# Predicting Test Set

y_pred = classifier.predict(X_test)

acc = accuracy_score(y_test, y_pred)

prec = precision_score(y_test, y_pred)

rec = recall_score(y_test, y_pred)

f1 = f1_score(y_test, y_pred)



model_results = pd.DataFrame([['Random Forest (n=100)', acc, prec, rec, f1]],

               columns = ['Model', 'Accuracy', 'Precision', 'Recall', 'F1 Score'])



results = results.append(model_results, ignore_index = True)





## K-fold Cross Validation

from sklearn.model_selection import cross_val_score

accuracies = cross_val_score(estimator = classifier, X= X_train, y = y_train,

                             cv = 10)

print("Random Forest Classifier Accuracy: %0.2f (+/- %0.2f)"  % (accuracies.mean(), accuracies.std() * 2))







## EXTRA: Confusion Matrix

cm = confusion_matrix(y_test, y_pred) # rows = truth, cols = prediction

df_cm = pd.DataFrame(cm, index = (0, 1), columns = (0, 1))

plt.figure(figsize = (10,7))

sns.set(font_scale=1.4)

sns.heatmap(df_cm, annot=True, fmt='g')

print("Test Data Accuracy: %0.4f" % accuracy_score(y_test, y_pred))
#Print result table showing the output of various used above

print(results)
#Since SVM (RBF) has better stats than others, will use this to predict Survival on test dataset

## SVM (rbf)

from sklearn.svm import SVC

classifier = SVC(random_state = 0, kernel = 'rbf')

classifier.fit(X_train, y_train)



# Predicting Test Set

y_pred = classifier.predict(X_test)

acc = accuracy_score(y_test, y_pred)

prec = precision_score(y_test, y_pred)

rec = recall_score(y_test, y_pred)

f1 = f1_score(y_test, y_pred)



model_results = pd.DataFrame([['SVM (RBF)', acc, prec, rec, f1]],

               columns = ['Model', 'Accuracy', 'Precision', 'Recall', 'F1 Score'])



## EXTRA: Confusion Matrix

cm = confusion_matrix(y_test, y_pred) # rows = truth, cols = prediction

df_cm = pd.DataFrame(cm, index = (0, 1), columns = (0, 1))

plt.figure(figsize = (10,7))

sns.set(font_scale=1.4)

sns.heatmap(df_cm, annot=True, fmt='g')

print("Test Data Accuracy: %0.4f" % accuracy_score(y_test, y_pred))
#Predict Survival on the test data set

y_preds_final = classifier.predict(test_data_updated)
#### End of Model ####





# Formatting Final Results



final_results = pd.concat([test_data_updated, test_PassengerId], axis = 1).dropna()

final_results['Survived'] = y_preds_final

final_results = final_results[['PassengerId','Survived']]
final_results.head(10)
#Save predictions in a csv file

final_results.to_csv("my_submission.csv", index=False)