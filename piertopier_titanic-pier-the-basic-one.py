# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns # visualization 
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
# import xgboost as xgb


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
plt.style.use('seaborn-deep')
%matplotlib inline
Train_df = pd.read_csv('../input/train.csv')
Test_df = pd.read_csv('../input/test.csv')
Train_df.head(10)
Test_df.head()
Train_df.set_index('PassengerId',inplace=True)
Test_df.set_index('PassengerId',inplace=True)
Train_df.head()
Train_df.info()
Test_df.info()
Train_df.describe()
median_age_train=Train_df["Age"].median()
Train_df["Age"].fillna(median_age_train, inplace=True)
# fill NaN Age values with zero , preferred value for XGB 
# Train_df["Age"].fillna(0, inplace=True)
Train_df_clean = Train_df.dropna(subset=["Embarked"]).drop("Cabin",axis=1).drop("Ticket",axis=1)
#median_age_test=Test_df["Age"].median()
Test_df["Age"].fillna(median_age_train, inplace=True)
#Test_df["Age"].fillna(0, inplace=True)
Test_df_clean = Test_df.dropna(subset=["Embarked"]).drop("Cabin",axis=1).drop("Ticket",axis=1)
def extract_title(name):
    if "Mrs" in name :
        title = 0
    elif "Mr" in name :
        title = 1
    elif "Miss" in name :
        title = 2
    elif "Master" in name :
        title = 3
    elif "Dr" in name :
        title = 4
    else :
        title = 5
    return title 

Train_df_clean["Title"] = Train_df_clean["Name"].apply(extract_title)
Test_df_clean["Title"] = Test_df_clean["Name"].apply(extract_title)
Train_df_clean = Train_df_clean.drop("Name",axis=1)
Test_df_clean = Test_df_clean.drop("Name",axis=1)
Train_df_clean["Familysize"] = Train_df_clean["SibSp"]+Train_df_clean["Parch"]+1
Test_df_clean["Familysize"] = Test_df_clean["SibSp"]+Test_df_clean["Parch"]+1
le = LabelEncoder()
Train_df_clean["Sex"] = le.fit_transform(Train_df_clean["Sex"])
print(le.classes_)
Test_df_clean["Sex"] = le.fit_transform(Test_df_clean["Sex"])
le2 = LabelEncoder()
Train_df_clean["Embarked"] = le2.fit_transform(Train_df_clean["Embarked"])
print(le2.classes_)
Test_df_clean["Embarked"] = le2.fit_transform(Test_df_clean["Embarked"])
median_fare=Test_df_clean["Fare"].median()
Test_df_clean["Fare"].fillna(median_fare, inplace=True)
Test_df_clean.info()
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import accuracy_score
#predicting missing values in age using Random Forest
# using function from Poonam LigadeTitanic Survival Prediction End to End ML Pipeline

def fill_missing_age(df):
    
    #Feature set
    age_df = df[['Age','Embarked','Fare', 'Parch', 'SibSp',
                  'Title','Pclass','Familysize'
                 ]]
    # Split datasets into train and test
    train  = age_df.loc[ (df.Age.notnull()) ]# known Age values
    test = age_df.loc[ (df.Age.isnull()) ]# null Ages
    
    # All age values are stored in a target array
    y = train.values[:, 0]
    
    # All the other values are stored in the feature array
    X = train.values[:, 1::]
    
    # Create and fit a model
    rtr = RandomForestRegressor(n_estimators= 2000,n_jobs=-1)
    rtr.fit(X, y)
    
    # Use the fitted model to predict the missing values
    predictedAges = rtr.predict(test.values[:, 1::])

    
    # Assign those predictions to the full data set
    df.loc[ (df.Age.isnull()), 'Age' ] = predictedAges 
    
    return df

#Train_df_clean=fill_missing_age(Train_df_clean)
#Test_df_clean=fill_missing_age(Test_df_clean)

Train_df_clean.info()


Test_df_clean.info()
#Test_df_clean.Age.iloc[[10]] = 20

Train_df_clean.head(20)
Train_df_clean.info()
Test_df_clean.info()
Train_df_clean.head(20)
Train_df_clean.info()
Train_df_clean.head()
corr_matrix = Train_df_clean.corr()
corr_matrix["Survived"].sort_values(ascending=False)
sns.countplot(x="Sex", hue="Survived",data=Train_df_clean);
sns.countplot(x="Title", hue="Survived",data=Train_df_clean);
sns.countplot(x="Pclass", hue="Survived",data=Train_df_clean);
sns.countplot(x="Familysize", hue="Survived",data=Train_df_clean);

sns.set(rc={'figure.figsize':(35.7,16.27)})
sns.countplot(x="Age",data=Train_df_clean);
sns.set(rc={'figure.figsize':(35.7,16.27)})
sns.countplot(x="Fare",data=Train_df_clean);
def agecat(Age):
    Ageclass = 1
    if Age <= 5 :
        Ageclass = 1
    elif ( Age > 5 and Age <= 8 ):
        Ageclass = 2
    elif ( Age > 8 and Age <= 13 ):
        Ageclass = 3
    elif ( Age > 13 and Age <= 20 ):
        Ageclass = 4
    elif ( Age > 20 and Age <= 40 ):
        Ageclass = 5
    elif ( Age > 40 and Age <= 60 ):
        Ageclass = 6
    else : 
        Ageclass = 7
    return Ageclass

def child(Age):
    ischild = 1
    if Age <= 5:
       ischild = 1
    else:
       ischild = 0
    return ischild

def farecat(Fare):
    Fareclass = 1
    if Fare <= 5 :
        Fareclass = 1
    elif ( Fare > 5 and Fare < 15 ):
        Fareclass = 2
    elif ( Fare > 15 and Fare < 30 ):
        Fareclass = 3
    elif ( Fare > 30 and Fare < 50 ):
        Fareclass = 4
    else : 
        Fareclass = 5
    return Fareclass

#Train_df_clean["Ageclass"] = Train_df_clean["Age"].apply(agecat)
#Test_df_clean["Ageclass"] = Test_df_clean["Age"].apply(agecat)
#Train_df_clean["Fareclass"] = Train_df_clean["Fare"].apply(farecat)
#Test_df_clean["Fareclass"] = Test_df_clean["Fare"].apply(farecat)
Train_df_clean["ischild"] = Train_df_clean["Age"].apply(child)
Test_df_clean["ischild"] = Test_df_clean["Age"].apply(child)

Train_df_clean.head(10)
#sns.countplot(x="Ageclass", hue="Survived",data=Train_df_clean);
#sns.countplot(x="Fareclass", hue="Survived",data=Train_df_clean);
corr_matrix = Train_df_clean.corr()
corr_matrix["Survived"].sort_values(ascending=False)
#scaler = MinMaxScaler()

#Train_df_clean[["Age","Fare"]] = scaler.fit_transform(Train_df_clean[["Age","Fare"]])
#Test_df_clean[["Age","Fare"]] = scaler.fit_transform(Test_df_clean[["Age","Fare"]])
#Train_df_clean.head(20)

#Train_df_clean = Train_df_clean.drop("Age",axis=1).drop("Fare",axis=1)
#Test_df_clean = Test_df_clean.drop("Age",axis=1).drop("Fare",axis=1)
y = Train_df_clean["Survived"]
X = Train_df_clean.drop("Survived",axis=1)
X_test = Test_df_clean
from sklearn.model_selection import train_test_split
X_tr,X_te,y_tr,y_te = train_test_split(X,y,random_state = 42,test_size=0.01)
from sklearn import tree
# Initialize our decision tree object
#titanic_tree = tree.DecisionTreeClassifier()

# Train our decision tree (tree induction + pruning)
#titanic_tree.fit(X_tr,y_tr)
#y_tr_pred = titanic_tree.predict(X_tr)
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score

#print(confusion_matrix(y_tr, y_tr_pred))
#print(accuracy_score(y_tr, y_tr_pred))
#print(recall_score(y_tr, y_tr_pred))
#print(precision_score(y_tr, y_tr_pred))
#y_te_pred = titanic_tree.predict(X_te)
#print(confusion_matrix(y_te, y_te_pred))
#print(accuracy_score(y_te, y_te_pred))
#print(recall_score(y_te, y_te_pred))
#print(precision_score(y_te, y_te_pred))
#param_grid = { 
 #'n_estimators': [150,200],
 #'max_features': ['auto', 'sqrt', 'log2'],
 #'max_depth' : [5,8],
 #'criterion' :['gini', 'entropy'],
 #'min_samples_split': [3,5],
 #'min_samples_leaf': [3,5,20]
#}
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
#Initialize our Random Forest object
#titanic_forest = RandomForestClassifier(random_state=42)

#CV_titanic_forest = GridSearchCV(estimator=titanic_forest, param_grid=param_grid, cv= 8)
#CV_titanic_forest.fit(X_tr, y_tr)

#CV_titanic_forest.best_params_

# Initialize our Random Forest object
titanic_forest_final = RandomForestClassifier(random_state=42,max_features=8, n_estimators= 2500, max_depth=8, criterion='gini',min_samples_split=5,min_samples_leaf=1)

# Train our Random Forest model
titanic_forest_final.fit(X_tr,y_tr)
y_tr_pred = titanic_forest_final.predict(X_tr)

print(confusion_matrix(y_tr, y_tr_pred))
print(accuracy_score(y_tr, y_tr_pred))
print(recall_score(y_tr, y_tr_pred))
print(precision_score(y_tr, y_tr_pred))
#y_te_pred = titanic_forest_final.predict(X_te)

#print(confusion_matrix(y_te, y_te_pred))
#print(accuracy_score(y_te, y_te_pred))
#print(recall_score(y_te, y_te_pred))
#print(precision_score(y_te, y_te_pred))

from sklearn.linear_model import LogisticRegression
#titanic_lr = LogisticRegression()

#titanic_lr.fit(X_tr,y_tr)

#X_tr_matrix = X_tr.as_matrix()
#X_te_matrix = X_te.as_matrix()
#X_test_matrix = X_test.as_matrix()
#titanic_gbm = xgb.XGBClassifier(max_depth=3, n_estimators=300, learning_rate=0.03).fit(X_tr_matrix, y_tr)

#y_tr_pred = titanic_gbm.predict(X_tr_matrix)

#print(confusion_matrix(y_tr, y_tr_pred))
#print(accuracy_score(y_tr, y_tr_pred))
#print(recall_score(y_tr, y_tr_pred))
#print(precision_score(y_tr, y_tr_pred))


#y_te_pred = titanic_gbm.predict(X_te_matrix)

#print(confusion_matrix(y_te, y_te_pred))
#print(accuracy_score(y_te, y_te_pred))
#print(recall_score(y_te, y_te_pred))
#print(precision_score(y_te, y_te_pred))

#y_pred_test = titanic_tree.predict(X_test)
y_pred_test = titanic_forest_final.predict(X_test)
#y_pred_test = titanic_gbm.predict(X_test_matrix)
#y_pred_test = titanic_lr.predict(X_test)
y_pred_test_df = pd.DataFrame(y_pred_test)
y_pred_test_df.head()
X_test.info()
X_test=X_test.reset_index()
#X_test.head()

df_submission = pd.concat([X_test["PassengerId"],y_pred_test_df],axis=1)
df_submission.rename(columns={0:'Survived'}, inplace=True)
df_submission.head(10)
list(df_submission)
filename = 'csv_submission.csv'
df_submission.to_csv(filename,index=False,header=True)
print('Saved file: ' + filename)