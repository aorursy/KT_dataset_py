# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
titanic_train_df = pd.read_csv('/kaggle/input/titanic/train.csv')
titanic_test_df = pd.read_csv('/kaggle/input/titanic/test.csv')
titanic_train_df.head()
# Checking dataset balancing
titanic_train_df['Survived'].value_counts()
titanic_train_df.isnull().sum()
titanic_test_df.isnull().sum()
sns.heatmap(titanic_train_df.isnull())
# Plotting the count plot for survived and not survived passengers
sns.countplot(titanic_train_df.Survived)
# Checking survival rate with respect to the gender

sns.countplot(x='Survived', hue='Sex', data = titanic_train_df, palette='RdBu_r')
# Observing Survival rate with respect to the passenger class
sns.countplot(x='Survived', hue='Pclass', data=titanic_train_df, palette='rainbow')
# Observing the Age range of the passengers
titanic_train_df['Age'].hist(bins=40, color='darkred',alpha=0.5)
# Observing the Sibsp that is 'Number of Siblings/Spouses Aboard' of the passengers
sns.countplot(titanic_train_df['SibSp'])
# Observing the Fare
titanic_train_df['Fare'].hist(bins=40, color='blue', figsize=(9,6))
# Observing passenger class and age  combination
plt.figure(figsize=(12,6))
sns.boxplot(x='Pclass',y='Age', data=titanic_train_df, palette='winter')
# Method to predict the age of the passenger
def predict_age(column_val):
    
    Age    =  column_val[0]
    Pclass =  column_val[1]
    
    if pd.isnull(Age):
        if Pclass == 1:
            return 37
        elif Pclass == 2:
            return 29
        elif Pclass == 3:
            return 24
        
    else:
        return Age

titanic_train_df['Age'] = titanic_train_df[['Age','Pclass']].apply(predict_age,axis=1)
titanic_test_df['Age'] = titanic_test_df[['Age','Pclass']].apply(predict_age,axis=1)
# Again observing the null values in the dataset

sns.heatmap(titanic_train_df.isnull())
titanic_train_df.drop('Cabin',axis=1,inplace=True)
titanic_test_df.drop('Cabin',axis=1,inplace=True)
titanic_train_df.shape
titanic_train_df.dropna(inplace=True)
# We need to convert Sex and Embarked columns into numbers 

sex = pd.get_dummies(titanic_train_df['Sex'],drop_first= True)
embarked = pd.get_dummies(titanic_train_df['Embarked'], drop_first = True)

#For test data
sex_test = pd.get_dummies(titanic_test_df['Sex'],drop_first= True)
embarked_test = pd.get_dummies(titanic_test_df['Embarked'], drop_first = True)
# Now we will drop unwanted columns which will not play significant role in training the model
titanic_train_df.drop(['Sex','Embarked','Name','Ticket'], axis=1, inplace=True)
titanic_test_df.drop(['Sex','Embarked','Name','Ticket'], axis=1, inplace=True)

#Now merging sex, embarked with main dataframe
titanic_train_df = pd.concat([titanic_train_df,sex,embarked],axis=1)
titanic_test_df = pd.concat([titanic_test_df,sex_test,embarked_test],axis=1)
titanic_train_df.head()
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
X_train = titanic_train_df.drop(['Survived'], axis=1)
y_train = titanic_train_df['Survived']
models ={
    'svm': {
        'model' : SVC(gamma='auto'),
        'params': {
            'C' : [1,10,20,30],
            'kernel': ['rbf','linear']
        }
    },
    'logistic_regression':{
        'model': LogisticRegression(solver='liblinear'),
        'params':{
            'C': [1,5,10]
        }
    },
    'knn':{
        'model' : KNeighborsClassifier(3),
        'params': {}
    },
    'decision_tree':{
        'model' : DecisionTreeClassifier(),
        'params': {}
    },
    'random_forest':{
        'model': RandomForestClassifier(),
        'params': {
            'n_estimators': [5,10,15,20]
        }
    },
    'adaboost':{
        'model': AdaBoostClassifier(),
        'params': {}
    },
    'gradient_boosting':{
        'model': GradientBoostingClassifier(),
        'params': {}
    },
}
scores = []

for model_name, model_values in models.items():
    classifier = GridSearchCV(model_values['model'],model_values['params'],cv=5,return_train_score=False)
    classifier.fit(X_train,y_train)
    
    scores.append({'model':model_name,
                  'best score':classifier.best_score_,
                  'best params':classifier.best_params_})
    
analyzed_model_df = pd.DataFrame(scores,columns=['model','best score','best params'])
analyzed_model_df
svm_classifier = GridSearchCV(SVC(gamma='auto'),{
    'kernel':['rbf','linear'],
    'C': [1,5,10,15,20]
},cv=5,return_train_score = False)
svm_classifier.fit(X_train,y_train)
result_df = pd.DataFrame(svm_classifier.cv_results_)
result_df
result_df[['param_C','param_kernel','params','mean_test_score']]
svm_classifier = SVC(gamma='auto',kernel='linear',C=10)
svm_classifier.fit(X_train,y_train)
titanic_test_df.isnull().sum()
titanic_test_df['Fare'].describe()
# Filling null value with mean value
titanic_test_df['Fare'].fillna(titanic_test_df['Fare'].mean(),inplace=True)
titanic_test_df.isnull().sum()
survival_pred = svm_classifier.predict(titanic_test_df)
survival_pred
predicted_results_df = pd.DataFrame({'PassengerId':titanic_test_df['PassengerId'],'Survived':survival_pred})
predicted_results_df.head()
predicted_results_df.to_csv('/kaggle/working/TitanicSurvivalPredictions.csv',index=False)