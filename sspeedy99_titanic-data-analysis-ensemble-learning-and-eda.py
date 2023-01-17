# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input/'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
train_data =  '../input/titanic/train.csv'

test_data = '../input/titanic/test.csv'

train_set = pd.read_csv(train_data)

test_set = pd.read_csv(test_data)
print(f'Train dataset has {train_set.shape[0]} rows and {train_set.shape[1]} columns.')

print(f'Test dataset has {test_set.shape[0]} rows and {test_set.shape[1]} columns.')
train_set.head()
train_set.describe()
train_set.info()
train_set.columns
#Checking all the  null values present in every column in the  Dataset

train_set.isnull().sum(axis=0)
train_set.describe()
#data visualaisation

%matplotlib inline

import matplotlib.pyplot as plt

import seaborn as sns

sns.set_style('whitegrid')

import warnings

warnings.filterwarnings('ignore')
#Heatmap of the missing values present in the all feature of the training data.

sns.heatmap(train_set.isnull(), cmap='plasma')
#Correlation heatmap between the predictor variables. 

sns.heatmap(train_set.corr(),annot=True, cmap='BrBG')
sns.countplot(data=train_set, x='Survived', palette='Set1')

print(train_set.Survived.value_counts())
#Passenger survival based on the Sex and Age feature 

sns.countplot(data=train_set, x='Survived', hue='Sex', palette='Set2')
#Number of values in the embarked feature

print(train_set.Embarked.value_counts())

sns.countplot(data=train_set,x='Embarked',palette='Set1')
sns.pairplot(data=train_set)
train_set.columns
train_set.isnull().sum()
#missing values in the Cabin

train_set['Cabin'].isnull().sum()
train_set['Cabin'].value_counts().head()
#inorder to apply same analysis to our test and train data we need to merge them togather

merged = pd.concat([train_set,test_set], sort = False)

merged.head(3)
merged['Cabin'].value_counts().head(3)
#filling the na values in the Cabin column

merged['Cabin'].fillna('X', inplace=True)
merged.head(3)
# Keeping 1st charater from the Cabin

merged['Cabin'] = merged['Cabin'].apply(lambda x: x[0])

merged['Cabin'].value_counts()

sns.countplot(data = merged, x='Cabin')
merged['Name'].head(5)
#Extracting the Title from the names, it will give us the sex of the passengers

merged['Title'] = merged['Name'].str.extract('([A-Za-z]+)\.')

merged['Title'].head()
#There can be also differnt titles present in the dataset apart for the common titles such as Mr, Mrs etc

merged['Title'].value_counts()
# Replacing  Dr, Rev, Col, Major, Capt with 'Officer'

merged['Title'].replace(to_replace = ['Dr', 'Rev', 'Col', 'Major', 'Capt'], value = 'Officer', inplace=True)



# Replacing Dona, Jonkheer, Countess, Sir, Lady with 'Aristocrate'

merged['Title'].replace(to_replace = ['Dona', 'Jonkheer', 'Countess', 'Sir', 'Lady', 'Don'], value = 'Aristocrat', inplace = True)



#  Replace Mlle and Ms with Miss. And Mme with Mrs.

merged['Title'].replace({'Mlle':'Miss', 'Ms':'Miss', 'Mme':'Mrs'}, inplace = True)
merged['Title'].value_counts()

sns.countplot(data=merged, x='Title')
merged['SibSp'].value_counts()

merged['Parch'].value_counts()
# Merging Sibsp and Parch and creating new variable called 'Family_size'

merged['Family_size'] = merged.SibSp + merged.Parch + 1  # Adding 1 for single person

merged['Family_size'].value_counts()
# Create buckets of single, small, medium, and large and then put respective values into them.

merged['Family_size'].replace(to_replace = [1], value = 'single', inplace = True)

merged['Family_size'].replace(to_replace = [2,3], value = 'small', inplace = True)

merged['Family_size'].replace(to_replace = [4,5], value = 'medium', inplace = True)

merged['Family_size'].replace(to_replace = [6, 7, 8, 11], value = 'large', inplace = True)
merged['Family_size'].value_counts()
sns.countplot(data=merged,x='Family_size')
# let's preview the Ticket variable.

merged['Ticket'].head(10)
# Assign N if there is only number and no character. If there is a character, extract the character only.

ticket = []

for x in list(merged['Ticket']):

    if x.isdigit():

        ticket.append('N')

    else:

         ticket.append(x.replace('.','').replace('/','').strip().split(' ')[0])

# Swap values

merged['Ticket'] = ticket
merged['Ticket'].value_counts()
# Keeping 1st charater from the Ticket

merged['Ticket'] = merged['Ticket'].apply(lambda x: x[0])

merged['Ticket'].value_counts()
sns.countplot(data=merged, x='Ticket')
#We define a function which counts number of outliers present in the variable

def outliers(variable):

    global filtered # Global keyword is used inside a function only when we want to do assignments or when we want to change a variable.

    

    # Calculate 1st, 3rd quartiles and iqr.

    q1, q3 = variable.quantile(0.25), variable.quantile(0.75)

    iqr = q3 - q1

    

    # Calculate lower fence and upper fence for outliers

    l_fence, u_fence = q1 - 1.5*iqr , q3 + 1.5*iqr   # Any values less than l_fence and greater than u_fence are outliers.

    

    # Observations that are outliers

    outliers = variable[(variable<l_fence) | (variable>u_fence)]

    print('Total Outliers of', variable.name,':', outliers.count())

    

    # Drop obsevations that are outliers

    filtered = variable.drop(outliers.index, axis = 0)
#Total number of outliers present in the fare

outliers(merged['Fare'])
#Visualization of the outliers in the Fare distribution

plt.figure(figsize=(13,2))

sns.boxplot(x=merged['Fare'], palette='magma')

plt.title('Fare distribution with Outliers', fontsize=15)
#Visualization of Fare districution without Outliers

plt.figure(figsize=(13,2))

sns.boxplot(x=filtered, palette='Blues')

plt.title('Fare distribution with Outliers', fontsize=15)
#Number of outliers in the Age distribution 

plt.figure(figsize=(13,2))

sns.boxplot(x=merged['Age'], palette='Blues')

plt.title('Age distribution with outliers', fontsize=15)
#Number of outliers in the Age distribution

plt.figure(figsize=(13,2))

sns.boxplot(x=filtered, palette='Blues')

plt.title('Age distribution without outliers', fontsize=15)
#Listing all the missing values in the dataset

merged.isnull().sum()
merged.sample(5)

merged['Embarked'].value_counts()
#Here S is the most frequent so we will be putting S in the empty places

merged['Embarked'].fillna(value = 'S', inplace=True)
#Fare is the discriptive value so it will be filled with the median values 

merged['Fare'].fillna(value= merged['Fare'].median(), inplace=True)
#We will put the median group value in the missing values of the NA

#We need to draw heatmap of the all the correlated value, we need to convert categorical data into numerical data

df = merged.loc[:, ['Sex', 'Pclass', 'Embarked', 'Title', 'Family_size', 'Parch', 'SibSp', 'Cabin', 'Ticket']]

from sklearn.preprocessing import LabelEncoder

LE = LabelEncoder()

df = df.apply(LE.fit_transform)

df.head(5)

#Moving Age variable to the Labeled dataframe

df['Age'] = merged['Age']

df.head(2)
#Moving Age variable to the index 0

df = df.set_index('Age').reset_index()

df.head(2)
#Drawing heatmap of the attribute which is correlated to the Age variable.

plt.figure(figsize=(10,6))

sns.heatmap(df.corr(), cmap='BrBG', annot=True)

plt.title('Correlation map of Age', fontsize=15)

plt.show()
#correlation between PCLass and Age

plt.figure(figsize=(10,6))

sns.boxplot(x="Pclass", y="Age", data=merged)
#correlated values of Age with Ticket

plt.figure(figsize=(10,6))

sns.boxplot(x="Title", y="Age",data= merged)
#Imputing the missing valuse of Age variable with median values of Pclass and Title

## Impute Age with median of respective columns (i.e., Title and Pclass)

merged['Age'] = merged.groupby(['Title', 'Pclass'])['Age'].transform(lambda x: x.fillna(x.median()))
merged.sample(5)
#Age Distribution

plt.figure(figsize=(10,6))

sns.distplot(merged['Age'], color='Red')
#binning the Age variable

label_names = ['infant', 'child', 'teenager','young_adult', 'adult', 'aged']



#range for each bin categrories of age

cut_points = [0,5,12,18,35,60,81]



#view categorized Age with original Age.

merged['Age_binned'] = pd.cut(merged['Age'], cut_points, labels = label_names)



#Age with Categorized Age.

merged[['Age', 'Age_binned']].head(3)
#visualisation of the Fare variable

plt.figure(figsize=(10,6))

sns.distplot(merged['Fare'], color='Blue')
#binning the Fare variable 

groups = ['low','medium','high','very high']



# Create range for each bin categories of Fare

cut_points = [-1, 130, 260, 390, 520]



#Create and view categorized Fare with original Fare

merged['Fare_binned'] = pd.cut(merged.Fare, cut_points, labels = groups)



# Fare with Categorized Fare

merged[['Fare', 'Fare_binned']].head(2)
merged.sample(5)
merged.dtypes
# Correcting data types, converting into categorical variables.

merged.loc[:, ['Pclass', 'Sex', 'Embarked', 'Cabin', 'Title', 'Family_size', 'Ticket']] = merged.loc[:, ['Pclass', 'Sex', 'Embarked', 'Cabin', 'Title', 'Family_size', 'Ticket']].astype('category')



# Due to merging there are NaN values in Survived for test set observations.

merged['Survived'] = merged['Survived'].dropna().astype('int') #Converting without dropping NaN throws an error
#verify converted data types

merged.dtypes
merged.head(3)
#Dropping the features which will not be helpfull for us.

# droping the feature that would not be useful anymore

merged.drop(columns = ['Name', 'Age','SibSp', 'Parch','Fare'], inplace = True, axis = 1)

merged.columns
merged.head(2)
# convert categotical data into dummies variables

merged = pd.get_dummies(merged, drop_first=True)

merged.head(2)
#Splitting our test and train data from the merged dataframe.

#Let's split the train and test set to feed machine learning algorithm.

#train will be the first 891 values and test will be the 

train = merged.iloc[:891, :]

test  = merged.iloc[891:, :]
train.head(3)
test.head(3)
#Drop passengerid from train set and Survived from test set.'''

train = train.drop(columns = ['PassengerId'], axis = 1)

test = test.drop(columns = ['Survived'], axis = 1)
# setting the data as input and output for machine learning models

X_train = train.drop(columns = ['Survived'], axis = 1) 

y_train = train['Survived']



# Extract test set

X_test  = test.drop("PassengerId", axis = 1).copy()
X_train.head(3)

y_train.head(3)
X_test.head(3)
# See the dimensions of input and output data set.'''

print('Input Matrix Dimension:  ', X_train.shape)

print('Output Vector Dimension: ', y_train.shape)

print('Test Data Dimension:     ', X_test.shape)
#Initialising the instances of all 5 classifiers



#Logistic Regression

from sklearn.linear_model import LogisticRegression

LR = LogisticRegression()



#K-nearest neighbours

from sklearn.neighbors import KNeighborsClassifier

KNN = KNeighborsClassifier()



#Desicion Tree

from sklearn.tree import DecisionTreeClassifier

DT = DecisionTreeClassifier()



#Random Forest classifier

from sklearn.ensemble import RandomForestClassifier

RF = RandomForestClassifier()



#Support vector machine

from sklearn.svm import SVC

SVM = SVC(gamma='auto')



#XG Boost classifier

from xgboost import XGBClassifier

XGB = XGBClassifier(n_jobs=-1, random_state=42)
#Creating a function which will give us the training score of the differnt types of classifier

def train_accuracy(model):

    model.fit(X_train, y_train)

    train_accuracy = model.score(X_train, y_train)

    train_accuracy = np.round(train_accuracy*100,2)

    return train_accuracy
#creating a summary table of train_accuracy

train_accuracy = pd.DataFrame({'Training accuracy %':[train_accuracy(LR),train_accuracy(KNN),train_accuracy(DT),train_accuracy(RF),train_accuracy(SVM), train_accuracy(XGB)]})

train_accuracy.index = ['Logistic Regression','KNN','Decision Tree','Random Forest','SVM','XGB']

sorted_train_accuracy = train_accuracy.sort_values(by='Training accuracy %', ascending = False)



sorted_train_accuracy
# Create a function that returns mean cross validation score for different models.

def val_score(model):

    from sklearn.model_selection import cross_val_score

    val_score = cross_val_score(model, X_train, y_train, cv = 10, scoring = 'accuracy').mean()

    val_score = np.round(val_score*100, 2)

    return val_score



# making the summary table of cross validation accuracy.

val_score = pd.DataFrame({'val_score(%)':[val_score(LR), val_score(KNN), val_score(DT), val_score(RF), val_score(SVM), val_score(XGB)]})

val_score.index = ['Logistic Regression', 'KNN','Decision Tree', 'Random Forest', 'SVC','XGB']

sorted_val_score = val_score.sort_values(by = 'val_score(%)', ascending = False)



#cross validation accuracy of the Classifiers

sorted_val_score

#Hypertuning the parameters for grid search CV

# define all the model hyperparameters one by one first



# 1. For logistic regression

lr_params = {'penalty':['l1', 'l2'],

             'C': np.logspace(0, 2, 4, 8 ,10)}



# 2. For KNN

knn_params = {'n_neighbors':[4,5,6,7,8,9,10],

              'weights':['uniform', 'distance'],

              'algorithm':['auto', 'ball_tree','kd_tree','brute'],

              'p':[1,2]}



# 3. For DT

dt_params = {'max_features': ['auto', 'sqrt', 'log2'],

             'min_samples_split': [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15], 

             'min_samples_leaf':[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],

             'random_state':[46]}

# 4. For RF

rf_params = {'criterion':['gini','entropy'],

             'n_estimators':[ 10, 30, 200, 400],

             'min_samples_leaf':[1, 2, 3],

             'min_samples_split':[3, 4, 6, 7], 

             'max_features':['sqrt', 'auto', 'log2'],

             'random_state':[46]}

# 5. For SVC

svc_params = {'C': [0.1, 1, 10,100], 

              'kernel': ['linear', 'rbf', 'poly', 'sigmoid'],

              'gamma': [ 1, 0.1, 0.001, 0.0001]}



#6. For XGB

xgb_params = xgb_params_grid = {"learning_rate"    : [0.05, 0.10, 0.15, 0.20, 0.25, 0.30 ] ,

                         "max_depth"        : [ 3, 4, 5, 6, 8, 10, 12, 15],

                         "min_child_weight" : [ 1, 3, 5, 7 ],

                         "gamma"            : [ 0.0, 0.1, 0.2 , 0.3, 0.4 ],

                         "colsample_bytree" : [ 0.3, 0.4, 0.5 , 0.7 ] }
#function to find the tuned hyperprametrs of the model

def tune_hyperparameters(model, param_grid):

    from sklearn.model_selection import GridSearchCV

    global best_params, best_scores

    

    #grid search object with 10 cross folds

    grid = GridSearchCV(model,param_grid, verbose=0, cv=10, scoring='accuracy', n_jobs=-1)

    

    #fitting with the grid object

    grid.fit(X_train, y_train)

    best_params, best_scores = grid.best_params_, np.round(grid.best_score_*100,2)

    return best_params, best_scores

    
#Apply tune hyperparameters in the created function

#Tuning hyperparametes for Logistic Regression

tune_hyperparameters(LR, param_grid=lr_params)

lr_best_params, lr_best_score = best_params,best_scores 

print('Logistic Regression Best Score:', lr_best_score)

print('And Best Parameters:', lr_best_params)
#Tuning hyperparametes for KNN

tune_hyperparameters(KNN, param_grid=knn_params)

knn_best_params, knn_best_score = best_params,best_scores 

print('KNN Best Score:', knn_best_score)

print('And Best Parameters:', knn_best_params)
#Tuning hyperparametes for DT

tune_hyperparameters(DT, param_grid=dt_params)

dt_best_params, dt_best_score = best_params,best_scores 

print('DT Best Score:', dt_best_score)

print('And Best Parameters:', dt_best_params)
#Tuning hyperparametes for RF

tune_hyperparameters(RF, param_grid=rf_params)

rf_best_params, rf_best_score = best_params,best_scores 

print('RF Best Score:', rf_best_score)

print('And Best Parameters:', rf_best_params)
#Tuning hyperparametes for XGB

tune_hyperparameters(XGB, param_grid=xgb_params)

xgb_best_params, xgb_best_score = best_params,best_scores 

print('SVC Best Score:', xgb_best_score)

print('And Best Parameters:', xgb_best_params)
#Tuning hyperparametes for SVC

tune_hyperparameters(SVM, param_grid=svc_params)

svc_best_params, svc_best_score = best_params,best_scores 

print('SVC Best Score:', svc_best_score)

print('And Best Parameters:', svc_best_params)
#create a summary table of best scores after byperparameter tuning

tuned_score = pd.DataFrame({'tuned_score(%)':[lr_best_score, knn_best_score, dt_best_score, rf_best_score, svc_best_score,xgb_best_score]})

tuned_score.index = ['Logistic Regression', 'KNN','Decision Tree', 'Random Forest', 'SVC','XGB']

sorted_tuned_score = tuned_score.sort_values(by = 'tuned_score(%)', ascending = False)



#cross validation accuracy of the Classifiers

sorted_tuned_score
# Instantiate the models with optimized hyperparameters.

lr  = LogisticRegression(**lr_best_params)

knn = KNeighborsClassifier(**knn_best_params)

dt  = DecisionTreeClassifier(**dt_best_params)

rf  = RandomForestClassifier(**rf_best_params)

svc = SVC(**svc_best_params)

xgb = XGBClassifier(**xgb_best_params)
#train all the model with optimized hyperparameters

models = {'LR':lr,'KNN':knn,'DT':dt,'RF':rf,'SVC':svc, 'XGB':xgb}



#10 folds cross validation after optimized hyperparametrs

score = []

for x,(keys, items) in enumerate(models.items()):

    # Train the models with optimized parameters using cross validation.

    # No need to fit the data. cross_val_score does that for us.

    # But we need to fit train data for prediction in the follow session.

    from sklearn.model_selection import cross_val_score

    items.fit(X_train,y_train)

    scores = cross_val_score(items, X_train, y_train, cv = 10, scoring='accuracy')*100

    score.append(scores.mean())

    print('Mean Accuracy: %0.4f (+/- %0.4f) [%s]'  % (scores.mean(), scores.std(), keys))

    
# Make prediction using all the trained models

model_prediction = pd.DataFrame({'LR':lr.predict(X_test), 'KNN':knn.predict(X_test), 'DT':dt.predict(X_test),'RF':rf.predict(X_test), 'SVC':svc.predict(X_test), 'XGB':xgb.predict(X_test)})



#All the Models Prediction 

model_prediction.head()
#Submission with Most accurate random forest classifier

submisson = pd.DataFrame({"PassengerID":test["PassengerId"],"Survived":rf.predict(X_test)})

submisson.to_csv('submisson_rf.csv',index=False)
#Submission with Most accurate SVC classifier

submission = pd.DataFrame({'PassengerId':test['PassengerId'],'Survived':svc.predict(X_test)})

submission.to_csv("submission_svc.csv", index = False)
#submission with the most accurate XGBClassifier

submission = pd.DataFrame({'PassengerId':test['PassengerId'],'Survived':xgb.predict(X_test)})

submission.to_csv("submission_xgb.csv", index = False)