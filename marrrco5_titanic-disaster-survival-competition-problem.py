#This kernel aims to investigate the performance of different approaches

#in solving the Titanic Disaster Survival compeition problem
%xmode Plain

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns;
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import LabelEncoder

from sklearn.preprocessing import  MinMaxScaler

from sklearn.impute import SimpleImputer

from sklearn.model_selection import cross_val_score

from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier
#Data directory

train_dir = '../input/titanic/train.csv'

test_dir = '../input/titanic/test.csv'
#Load the train data into dataframe

df_train = pd.read_csv(train_dir)
#Display the first few lines of data

df_train.head()
#Examine the data structure of train data

df_train.info(),df_train.isna().sum()
#Display the first few entries of columns with null values

Columns_missing_value = ['Age','Cabin','Embarked']

df_train[Columns_missing_value].head(5)
#Remove columns with no useful information and columns with too many null values

df_train = df_train.drop(columns=['PassengerId','Cabin'])
df_train.info(),df_train.isna().sum()
#Impute the missing values 

Imp = SimpleImputer(missing_values=np.nan,strategy='mean')

df_train['Age'] = Imp.fit_transform(df_train['Age'].values.reshape(-1,1))

df_train['Embarked'].fillna(method='backfill',inplace=True)
#Get the numerical and categorical columns

numerical_col = ['Survived','Age','SibSp','Parch','Fare']

categorical_col = ['Name','Sex','Ticket','Embarked','Pclass']
#Display the correlation between numerical features

sns.heatmap(df_train[numerical_col].corr(),annot=True,cmap='Greys');
#Plot the distributions of  numerical features of  Survived and Not Survived

for i,feature in enumerate(numerical_col):

    if feature != 'Survived':

        plt.figure(i)

        sns.distplot(df_train[feature].loc[df_train['Survived']==0],label='Not Survived')

        sns.distplot(df_train[feature].loc[df_train['Survived']==1],label='Survived')

        plt.legend(['Not Survived','Survived'])

        plt.title('Disttribution of {}'.format(feature))

plt.tight_layout()

plt.show();
#Feature Engineering on feature Name

#Examine the names

Names = df_train['Name'].values

Titles = [name.split(',')[1].split('.')[0] for name in Names]

np.unique(Titles)
#Add a new column showing whether that passenger has a normal title

Normal_titles = ['Mr','Mrs','Ms','Miss','Lady','Sir']

df_train['Normal_Title'] = 0

for i,name in enumerate(Names):

    if name.split(',')[1].split('.')[0].strip() in Normal_titles:

         df_train['Normal_Title'][i] = 1
#Remove the features Name and Ticket

df_train = df_train.drop(columns=['Name','Ticket'])
#LabelEncoding

le = LabelEncoder()

df_train['Sex'] = le.fit_transform(df_train['Sex'])

df_train = pd.get_dummies(df_train,columns=['Embarked'])
#Standardization of features Age and Fare

minmax_sc = MinMaxScaler()

minmax_sc.fit(df_train[['Age','Fare']])

df_train[['Age','Fare']] = minmax_sc.transform(df_train[['Age','Fare']])
df_train.head(10)
#Extract features and labels from the dataframe

X,y = df_train.drop(columns=['Survived']).values,df_train['Survived'].values
#Construct Logistic Regression Classifier

LogisticReg = LogisticRegression(penalty='l2')

val_score_lr = cross_val_score(LogisticReg,X=X,y=y,n_jobs=-1)

print('Validation Score: {} +/- {}'.format(np.mean(val_score_lr),np.std(val_score_lr)))
#Construct Decision Tree Classifier

Tree = DecisionTreeClassifier()

val_score_tree = cross_val_score(Tree,X=X,y=y,n_jobs=-1)

print('Validation Score: {} +/- {}'.format(np.mean(val_score_tree),np.std(val_score_tree)))
#Construct RandomForest Classifier

forest = RandomForestClassifier(n_estimators=100,n_jobs=-1,random_state=123)

#Since the dataset is small, we would perform cross-validation to validate the model performance

val_score_rf = cross_val_score(forest,X=X,y=y,cv=5,n_jobs=-1)

print('Validation Score: {} +/- {}'.format(np.mean(val_score_rf),np.std(val_score_rf)))
#RandomForest Classifier has shown the best perfomance among the three classifiers

#In the following, we will perform a GridSearch for tuning the hyperparameter
from sklearn.model_selection import GridSearchCV
param_range_estimator_num = [10, 50, 100, 300, 500, 1000, 2000]

param_grid = [{'n_estimators':param_range_estimator_num}]
GS = GridSearchCV(forest,param_grid = param_grid,scoring='accuracy',cv=5,n_jobs=-1)
GS.fit(X,y)
#Print the optimal hyperparameter and the correesponding accuracy

GS.best_params_,GS.best_score_
#Load the test data into dataframe

df_test = pd.read_csv(test_dir)
df_test.head()
#Examine the data structure of train data

df_test.info(),df_test.isna().sum()
#Remove columns with no useful information and columns with too many null values

df_test = df_test.drop(columns=['Ticket','Cabin'])
#Impute the missing values 

Imp = SimpleImputer(missing_values=np.nan,strategy='mean')

df_test['Age'] = Imp.fit_transform(df_test['Age'].values.reshape(-1,1))

df_test['Fare'] = Imp.fit_transform(df_test['Fare'].values.reshape(-1,1))
#Feature Engineering on feature Name

#Examine the names

Names = df_test['Name'].values

Titles = [name.split(',')[1].split('.')[0] for name in Names]
#Add a new column showing whether that passenger has a normal title

Normal_titles = ['Mr','Mrs','Ms','Miss','Lady','Sir']

df_test['Normal_Title'] = 0

for i,name in enumerate(Names):

    if name.split(',')[1].split('.')[0].strip() in Normal_titles:

         df_test['Normal_Title'][i] = 1
#Remove the feature Name

df_test = df_test.drop(columns=['Name'])
#LabelEncoding

le = LabelEncoder()

df_test['Sex'] = le.fit_transform(df_test['Sex'])

df_test = pd.get_dummies(df_test,columns=['Embarked'])
#Standardization of features Age and Fare in test data

df_test[['Age','Fare']] = minmax_sc.transform(df_test[['Age','Fare']])
#Prepare the test data

PassengerID = df_test['PassengerId']

Xtest = df_test.drop(columns=['PassengerId']).values
df_test.head()
#Predict the test results

forest_best = GS.best_estimator_

forest_best.fit(X,y)

ypred = forest_best.predict(Xtest)
Submission = pd.DataFrame({'PassengerID':PassengerID,'Survived':ypred})

Submission.to_csv('submission.csv',index=False) 