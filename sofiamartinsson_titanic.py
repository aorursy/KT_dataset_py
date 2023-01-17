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
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')

train = pd.read_csv('/kaggle/input/titanic/train.csv')
train.head()
train.isnull().sum()
numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
numeric_values = train.select_dtypes(include=numerics)
numeric_values.isnull().sum()
#select all category features and check which of them has nan values
categorical_values = train.select_dtypes(include=object)
categorical_values.isnull().sum() / len(train)
train.drop('Cabin', axis=1, inplace=True)
train['Embarked'].mode()
train['Embarked'] = train['Embarked'].fillna("S")
plt.figure(figsize=(12,6))
sns.distplot(train['Age'].dropna(), bins=60, kde=False)
plt.figure(figsize=(12,6))
sns.distplot(train['Age'].dropna(), bins=4, kde=False, color='r' )
#show the age  groups in a pie chart
age_groups = train['Age'].dropna().value_counts(bins=4, sort=False)
plt.figure(figsize=(12,6))
labels = '0-20','21-40', '41-60', '60+'
plt.pie(age_groups, labels=labels, autopct='%1.1f%%', shadow=True, startangle=140) 
print(X_balanced.shape)
print(X_balanced[0])
def average_age(dataset):
    
    ### Find out how many people are in a certain age group 0-20 , 21-40, 41-60, 60+
    age_group_1 = []
    age_group_2 = []
    age_group_3 = []
    age_group_4 = []

    for i in range(len(dataset)):

        if dataset['Age'][i] < 20:
            age_group_1.append(dataset['Age'][i])
        elif dataset['Age'][i] >= 20 and dataset['Age'][i] < 40:
            age_group_2.append(dataset['Age'][i])
        elif dataset['Age'][i] >= 40 and dataset['Age'][i] < 60:
            age_group_3.append(dataset['Age'][i])
        elif dataset['Age'][i] >= 60 and dataset['Age'][i] < 90:
            age_group_4.append(dataset['Age'][i])

    #calculate the average age for all age groups
    avg_age_1 = round(np.mean(age_group_1),0) 
    avg_age_2 = round(np.mean(age_group_2),0) 
    avg_age_3 = round(np.mean(age_group_3),0)
    avg_age_4 = round(np.mean(age_group_4),0)
    
    #calculate how many people (%) are in each group
    total_people = dataset['Age'].notnull().sum()
    percent_age_1 = round(len(age_group_1) / total_people,2)
    percent_age_2 = round(len(age_group_2) / total_people,2)
    percent_age_3 = round(len(age_group_3) / total_people,2)
    percent_age_4 = round(len(age_group_4) / total_people,2)

    #calculate how many people there should be in each group for the missing values
    total_people_nan = dataset['Age'].isnull().sum()
    ave_age_group_1 = round(percent_age_1 * total_people_nan,0)
    ave_age_group_2 = round(percent_age_2 * total_people_nan,0)
    ave_age_group_3 = round(percent_age_3 * total_people_nan,0)
    ave_age_group_4 = round(percent_age_4 * total_people_nan,0)
    
    
    #setting all nan to 0 and add them to a list
    dataset['Age'] = dataset.fillna(0)['Age']

    indices_with_age_0 = []

    for z in range(len(dataset)):

        if dataset['Age'][z] == 0:
            indices_with_age_0.append(z)

    #setup steps that will be checked in the for loop, first step is all people 0-19 next step is all people 20-39 and so on.
    step_1 = ave_age_group_1              
    step_2 = step_1 + ave_age_group_2      
    step_3 = step_2 + ave_age_group_3
    step_4 = step_3 + ave_age_group_4

    #loop through all rows of age = 0 and add the average age for each age group. 
    for x in range(len(indices_with_age_0)):

        if x <= step_1:
            dataset.at[indices_with_age_0[x], 'Age'] = avg_age_1
        elif x > step_1 and x <= step_2:
             dataset.at[indices_with_age_0[x], 'Age'] = avg_age_2

        elif x > step_2 and x <= step_3:
             dataset.at[indices_with_age_0[x], 'Age'] = avg_age_3
        elif x > step_3 and x <= step_4:
             dataset.at[indices_with_age_0[x], 'Age'] = avg_age_4
    
    return dataset['Age']                                                                                                                                                                                                               
train['Age'] = average_age(train)
#All nan values  are handled.
train.isnull().sum()
sns.countplot(x='Survived', hue='Sex', data=train,palette='viridis_r')
sns.countplot(x='Survived', hue='Pclass', data=train,palette='rainbow')
sns.countplot(x='Pclass', hue='Sex', data=train,palette='rainbow')
sns.countplot(x='Survived', hue='Embarked', data=train,palette='spring')
sns.countplot(x='Embarked', hue='Pclass', data=train,palette='winter')
sns.countplot(x='Embarked', hue='Sex', data=train,palette='YlOrBr_r')
train.head()
categorical_values = train.select_dtypes(include=object)
categorical_values
train.drop(['Name', 'Ticket'], axis=1, inplace=True)
train.head()
categorical_values = train.select_dtypes(include=object)
categorical_values
#create function to loop throug categorical features and add dummy values
def dummy_df(df, todummylist):
    for x in todummylist:
       dummies = pd.get_dummies(df[x], prefix=x, dummy_na = False, drop_first=True)
       df = df.drop(x, 1)
       df = pd.concat([df, dummies], axis = 1)
    return df
#create dummies for all categorical features
dummies = list(categorical_values)
train = dummy_df(train, dummies)
train.rename(columns={'Sex_male':'Male'}, inplace=True)
train.head()
train["Pclass"] = train["Pclass"].astype("category")
train = pd.get_dummies(train, columns = ["Pclass"],prefix="Pc", drop_first=True)
plt.figure(figsize=(12,6))
sns.heatmap(train.corr(), annot=True)
#create function to decide which features that are correlated which eachother.
def correlation(dataset, threshold):
    
    col_corr = set()
    corr_matrix = dataset.corr()
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if abs(corr_matrix.iloc[i, j]) > threshold:
                colname = corr_matrix.columns[i]
                col_corr.add(colname)
    return col_corr 
#check if any feature correlate with eachother.
corr_features = correlation(train, 0.8)
len(set(corr_features))
corr_features
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.feature_selection import SelectFromModel

features = ['Male', 'Fare', 'Embarked_Q', 'Embarked_S', 'Pc_2', 'Pc_3', 'Age']
X = train[features]
y = train['Survived']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=101)
X_train.head()
#scaling the features
scaler_X = MinMaxScaler()
scale_X_train = scaler_X.fit_transform(X_train)
scale_X_test = scaler_X.transform(X_test)
classifier = RandomForestClassifier(max_depth=10, n_estimators=300, criterion='gini')
classifier.fit(scale_X_train, y_train)
## Applying grid search  to find the best model and the best parameters
parameters = [{'n_estimators': [200,300,400, 500,600,800],
               'criterion': ['entropy', 'gini'],
               'max_depth': [3,5,10,15,20]
                }]
grid_search = GridSearchCV(estimator = classifier, param_grid = parameters, scoring = 'accuracy', cv=10)

grid_search = grid_search.fit(scale_X_train, y_train)

best_acc = grid_search.best_score_
best_para = grid_search.best_params_
print(best_acc)
print(best_para)
classifier = DecisionTreeClassifier(criterion='gini', max_depth=3, splitter='best')
classifier.fit(scale_X_train, y_train)
## Decision tree
parameters = [{
               'criterion': ['entropy', 'gini'],
               'max_depth': [3,5,10,15,20],
               'splitter': ['best', 'random'],               
                }]
grid_search = GridSearchCV(estimator = classifier, param_grid = parameters, scoring = 'accuracy', cv=10)

grid_search = grid_search.fit(scale_X_train, y_train)

best_acc = grid_search.best_score_
best_para = grid_search.best_params_

print(best_acc)
print(best_para)
classifier = SVC(kernel = 'rbf', C = 1000, gamma = 0.08)
classifier.fit(scale_X_train, y_train)
parameters = [{'C': [1, 10, 100, 1000], 'kernel': ['linear'], 'gamma': [0.09, 0.08,0.07,0.06,0.05],
               'C': [1, 10, 100, 1000], 'kernel': ['rbf'], 'gamma': [0.09, 0.08,0.07,0.06,0.05]
                }]
grid_search = GridSearchCV(estimator = classifier, param_grid = parameters, scoring = 'accuracy', cv=10)

grid_search = grid_search.fit(scale_X_train, y_train)

best_acc = grid_search.best_score_
best_para = grid_search.best_params_
print(best_acc)
print(best_para)
#check accuracy with Cross val score
accuracies = cross_val_score(estimator = classifier, X = scale_X_train, y = y_train, cv=10)
print(accuracies.mean())
print(accuracies.std())
#predict
y_pred = classifier.predict(scale_X_test)
#evaluate prediction
from sklearn.metrics import confusion_matrix, classification_report
cm = confusion_matrix(y_test, y_pred)
classification = classification_report(y_test, y_pred)

print(cm)
print(classification)