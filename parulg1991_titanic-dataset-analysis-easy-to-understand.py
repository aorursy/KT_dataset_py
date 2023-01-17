#Import libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from collections import Counter

%matplotlib inline
# Importing the dataset
train_df = pd.read_csv('/kaggle/input/titanic/train.csv')
test_df = pd.read_csv('/kaggle/input//titanic/test.csv')
gender_sub = pd.read_csv('/kaggle/input/titanic/gender_submission.csv')
#Analaysis of data
train_df.head()
#checking for null values
train_df.isnull().sum()
sns.set_style('whitegrid')
sns.countplot(x='Survived', hue='Sex',data=train_df)
sns.countplot(x='Survived',hue='Pclass',data=train_df,palette='rainbow')
plt.figure(figsize =(12,6))
sns.heatmap(train_df.corr(),annot =True)

sns.countplot(x='SibSp',hue='Survived',data=train_df,palette='rainbow')
sns.countplot(x='Parch',hue='Survived',data=train_df,palette='rainbow')
# Explore Age vs Survived
g = sns.FacetGrid(train_df, col='Survived')
g = g.map(sns.distplot, "Age")
train_df['Fare'].hist(color='green',bins=40,figsize=(8,4))
def detect_outliers(df,n,features):
    outlier_indices = []
    # iterate over features(columns)
    for col in features:
        # 1st quartile (25%)
        Q1 = np.percentile(df[col], 25)
        # 3rd quartile (75%)
        Q3 = np.percentile(df[col],75)
        # Interquartile range (IQR)
        IQR = Q3 - Q1
        
        
        # outlier step
        outlier_step = 1.5 * IQR
        # Determine a list of indices of outliers for feature col
        outlier_list_col = df[(df[col] < Q1 - outlier_step) | (df[col] > Q3 + outlier_step )].index
        
        # append the found outlier indices for col to the list of outlier indices 
        outlier_indices.extend(outlier_list_col)
        
    # select observations containing more than 2 outliers
    outlier_indices = Counter(outlier_indices)        
    
    multiple_outliers = list( k for k, v in outlier_indices.items() if v > n )
    
    return multiple_outliers   
# detect outliers from Age, SibSp , Parch and Fare
Outliers_to_drop = detect_outliers(train_df, 2 ,["Age","SibSp","Parch","Fare"])

train_df.loc[Outliers_to_drop] # Show the outliers rows
# Drop outliers
train_df = train_df.drop(Outliers_to_drop, axis = 0).reset_index(drop=True)
combine = pd.concat([train_df,test_df],axis=0)
plt.figure(figsize=(12, 7))
g = sns.heatmap(train_df[["Age","SibSp","Parch","Pclass"]].corr(),cmap="BrBG",annot=True)
combine['Age'] = combine.groupby("Pclass")['Age'].transform(lambda x: x.fillna(x.median()))
combine.isnull().sum()
combine['Embarked'] = combine['Embarked'].fillna(value=combine.Embarked.mode()[0])
combine['Fare'] = combine['Fare'].fillna(value=combine['Fare'].median())
combine.loc[ combine['Fare'] <= 7.91, 'Fare'] = 0
combine.loc[(combine['Fare'] > 7.91) & (combine['Fare'] <= 14.454), 'Fare'] = 1
combine.loc[(combine['Fare'] > 14.454) & (combine['Fare'] <= 31), 'Fare']   = 2
combine.loc[ combine['Fare'] > 31, 'Fare'] = 3
combine['Fare'] = combine['Fare'].astype(int)
combine['Fare'].unique()
plt.figure(figsize=(11, 6))
sns.heatmap(combine.corr(),annot =True)
combine =combine.drop(['SibSp','Parch'],axis=1)
dataset_title = [i.split(",")[1].split(".")[0].strip() for i in combine["Name"]]
combine['Title'] = pd.Series(dataset_title)
combine['Title'] = combine['Title'].replace(['Rev', 'Dr', 'Col', 'Ms', 'Mlle', 'Major', 'the Countess', 
                                     'Capt', 'Dona', 'Jonkheer', 'Lady', 'Sir', 'Mme', 'Don'], 'Other')
title_category = {'Mr':1, 'Miss':2, 'Mrs':3, 'Master':4, 'Other':5}
combine['Title'] = combine['Title'].map(title_category)

combine =combine.drop(['Name','Ticket'],axis=1)
combine['Cabin']=combine['Cabin'].fillna('Missing')

combine['Cabin']=combine['Cabin'].str.get(0)

combine['Cabin'].unique()
cabin_category = {'A':1,'B':2, 'C':3, 'D':4, 'E':5,'F':6,'G':7,'T':8,'M':9}

combine['Cabin'] = combine['Cabin'].map(cabin_category)

combine.info()
#converting categorical feature using pd_dummies and also dropping first column in conversion to reduce unnecessary features
sex = pd.get_dummies(combine['Sex'],drop_first=True)
embark = pd.get_dummies(combine['Embarked'],drop_first=True)
combine.drop(['Sex','Embarked'],axis=1,inplace=True)

combine = pd.concat([combine,sex,embark],axis=1)
combine.head()
dataset_train = combine[combine['PassengerId']<=891]
dataset_test = combine[combine['PassengerId']>891]
dataset_train = dataset_train.drop('PassengerId',axis = 1)
dataset_test = dataset_test.drop('PassengerId',axis = 1)

dataset_test.isnull().sum()
#asssigning values to X and y
X = dataset_train.drop('Survived',axis = 1)
y = dataset_train['Survived']
X_test = dataset_test.drop('Survived',axis=1)
y

from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size = 0.2, random_state = 0)
X_train.isnull().sum()
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression()
classifier.fit(X_train,y_train)
y_pred =classifier.predict(X_val)
acc_log = round(classifier.score(X_train, y_train) * 100, 2)
acc_log
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_val, y_pred)
cm
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10)
print(accuracies.mean())
print(accuracies.std())

y_test=classifier.predict(X_test)

from sklearn.metrics import classification_report
print(classification_report(y_val,y_pred))

from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators=500)
classifier.fit(X_train,y_train)
y_pred =classifier.predict(X_val)
acc_log = round(classifier.score(X_train, y_train) * 100, 2)
acc_log
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_val, y_pred)
cm
from sklearn.metrics import classification_report
print(classification_report(y_val,y_pred))
y_test=classifier.predict(X_test)
gender_sub['Survived']=y_test
gender_sub.dtypes
gender_sub.Survived=gender_sub.Survived.astype(int)
gender_sub.head()
gender_sub.dtypes #to check type of dataframe
gender_sub.to_csv("gender_sub_final1.csv",index=False)