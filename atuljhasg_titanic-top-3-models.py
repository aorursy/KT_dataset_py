# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# for handling data
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# for visualisation
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns

# for machine learning
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from sklearn.ensemble import (RandomForestClassifier, AdaBoostClassifier,GradientBoostingClassifier, ExtraTreesClassifier)
from sklearn.cross_validation import KFold
from sklearn import preprocessing
from sklearn.metrics import accuracy_score

# importing data
df_train=pd.read_csv('../input/train.csv',sep=',')
df_test=pd.read_csv('../input/test.csv',sep=',')
df_data = df_train.append(df_test) # The entire data: train + test.
print(pd.isnull(df_data).sum())
print(pd.isnull(df_train).sum())
print(pd.isnull(df_test).sum())
df_train.describe()
df_test.describe()
df_data.columns
df_data["Title"] = df_data.Name.str.extract(' ([A-Za-z]+)\.', expand=False) #Creating new column name Title
df_data.Title.head() #Lets see the result.
df_data.Title.tail()
df_data.Title
#classify common titles. 
df_data["Title"] = df_data["Title"].replace('Master', 'Master')
df_data["Title"] = df_data["Title"].replace('Mlle', 'Miss')
df_data["Title"] = df_data["Title"].replace(['Mme', 'Dona', 'Ms'], 'Mrs')
df_data["Title"] = df_data["Title"].replace(['Don','Jonkheer'],'Mr')
df_data["Title"] = df_data["Title"].replace(['Capt','Rev','Major', 'Col','Dr'], 'Millitary')
df_data["Title"] = df_data["Title"].replace(['Lady', 'Countess','Sir'], 'Honor')
# Assign in df_train and df_test:
df_train["Title"] = df_data['Title'][:891]
df_test["Title"] = df_data['Title'][891:]

# convert Title categories to Columns
titledummies=pd.get_dummies(df_train[['Title']], prefix_sep='_') #Title
df_train = pd.concat([df_train, titledummies], axis=1) 
ttitledummies=pd.get_dummies(df_test[['Title']], prefix_sep='_') #Title
df_test = pd.concat([df_test, ttitledummies], axis=1) 
#Fill the na values in Fare
df_data["Embarked"]=df_data["Embarked"].fillna('S') #NAN Values set to S class
df_train["Embarked"] = df_data['Embarked'][:891] # Assign Columns to Train Data
df_test["Embarked"] = df_data['Embarked'][891:] #Assign Columns to Test Data
print('Missing Data Fixed') # Print confirmation (looks good xD)
# convert Embarked categories to Columns
dummies=pd.get_dummies(df_train[["Embarked"]], prefix_sep='_') #Embarked
df_train = pd.concat([df_train, dummies], axis=1) 
dummies=pd.get_dummies(df_test[["Embarked"]], prefix_sep='_') #Embarked
df_test = pd.concat([df_test, dummies], axis=1)
print("Embarked created")
# Fill the na values in Fare based on average fare
import warnings
warnings.filterwarnings('ignore')
df_data["Fare"]=df_data["Fare"].fillna(np.median(df_data["Fare"]))
df_train["Fare"] = df_data["Fare"][:891]
df_test["Fare"] = df_data["Fare"][891:]
titles = ['Master', 'Miss', 'Mr', 'Mrs', 'Millitary','Honor']
for title in titles:
    age_to_impute = df_data.groupby('Title')['Age'].median()[title]
    df_data.loc[(df_data['Age'].isnull()) & (df_data['Title'] == title), 'Age'] = age_to_impute
# Age in df_train and df_test:
df_train["Age"] = df_data['Age'][:891]
df_test["Age"] = df_data['Age'][891:]
print('Missing Ages Estimated')
#map each Sex value to a numerical value
sex_mapping = {"male": 0, "female": 1}
df_train['Sex'] = df_train['Sex'].map(sex_mapping)
df_test['Sex'] = df_test['Sex'].map(sex_mapping)

df_train.head()
df_train = df_train.drop(['Cabin'], axis = 1)
df_test = df_test.drop(['Cabin'], axis = 1)
df_train = df_train.drop(['Name'], axis = 1)
df_test = df_test.drop(['Name'], axis = 1)
from sklearn import preprocessing
def encode_features(df_train, df_test):
    features = ['Title','Embarked','Ticket']
    
    df_combined = pd.concat([df_train[features], df_test[features]])
    
    for feature in features:
        le = preprocessing.LabelEncoder()
        le = le.fit(df_combined[feature])
        df_train[feature] = le.transform(df_train[feature])
        df_test[feature] = le.transform(df_test[feature])
    return df_train, df_test
    
df_train, df_test = encode_features(df_train, df_test)
df_train.head()
from sklearn.model_selection import train_test_split

predictors = df_train.drop(['Survived', 'PassengerId'], axis=1)
target = df_train["Survived"]
X_train, X_val, y_train, y_val = train_test_split(predictors, target, test_size = 0.22, random_state = 0)
#importing Logistic Regression classifier

from sklearn.linear_model import LogisticRegression
model1 = LogisticRegression()
model1.fit(X_train,y_train)
#printing the training score
print('The training score for logistic regression is:',(model1.score(X_train,y_train)*100),'%')
print('Validation accuracy', accuracy_score(y_val, model1.predict(X_val)))
#importing random forest classifier

from sklearn.ensemble import RandomForestClassifier
model2 = RandomForestClassifier(n_estimators=6)
model2.fit(X_train,y_train)
#printing the training score
print('The training score for logistic regression is:',(model2.score(X_train,y_train)*100),'%')
print('Validation accuracy', accuracy_score(y_val, model2.predict(X_val)))
#importing Gradient boosting classifier

from sklearn.ensemble import GradientBoostingClassifier
model3 = GradientBoostingClassifier(n_estimators=7,learning_rate=1.1)
model3.fit(X_train,y_train)
#printing the training score
print('The training score for logistic regression is:',(model3.score(X_train,y_train)*100),'%')
print('Validation accuracy', accuracy_score(y_val, model3.predict(X_val)))
from sklearn.model_selection import KFold #for K-fold cross validation
from sklearn.model_selection import cross_val_score #score evaluation
from sklearn.model_selection import cross_val_predict #prediction
from sklearn import svm #support vector Machine
kfold = KFold(n_splits=10, random_state=22) # k=10, split the data into 10 equal parts
xyz=[]
accuracy=[]
std=[]
df_test = df_test.dropna()
classifiers=['Logistic Regression','Random Forest','GradientBoosting']
models=[LogisticRegression(),RandomForestClassifier(n_estimators=100),GradientBoostingClassifier(n_estimators=7,learning_rate=1.1)]
for i in models:
    model = i
    cv_result = cross_val_score(model,predictors,target, cv = kfold,scoring = "accuracy")
    cv_result=cv_result
    xyz.append(cv_result.mean())
    std.append(cv_result.std())
    accuracy.append(cv_result)
new_models_dataframe2=pd.DataFrame({'CV Mean':xyz,'Std':std},index=classifiers)       
new_models_dataframe2
plt.subplots(figsize=(12,6))
box=pd.DataFrame(accuracy,index=[classifiers])
box.T.boxplot()
new_models_dataframe2['CV Mean'].plot.barh(width=0.8)
plt.title('Average CV Mean Accuracy')
fig=plt.gcf()
fig.set_size_inches(8,5)
plt.show()
prediction = model2.predict(df_test)
passenger_id = df_data[892:].PassengerId
test = pd.DataFrame( { 'PassengerId': passenger_id , 'Survived': prediction } )
test.shape
test.head()
test.to_csv( 'titanic_pred.csv' , index = False )


submission = pd.read_csv('titanic_pred.csv')
print(submission.head())
print(submission.tail())



