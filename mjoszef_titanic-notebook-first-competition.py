#Import Some Basic Packages
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
#Load data as pandas data frames, then check it out
train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')   #Data I will predict on later
gender_submission = pd.read_csv('../input/gender_submission.csv')  #--- Example of submission File
train.head(2)
train.info()
rd = train.drop('Cabin',axis=1)
rd = rd[rd['Embarked'].notnull()]
rd.info()
sns.distplot(rd[(rd['Age'].notnull()) & (rd['Survived'] == 1)]['Age'],label='Survive',color='blue',kde=False)
sns.distplot(rd[(rd['Age'].notnull()) & (rd['Survived'] == 0)]['Age'],label='Not Survive',color='red',kde=False)
plt.legend()
sns.pairplot(rd[rd['Age'].notnull()],hue='Survived')
model_data = rd.drop(['Name','Ticket','PassengerId'],axis=1)
model_data.head(2)
model_data = pd.get_dummies(model_data,columns=['Sex','Embarked'],drop_first=True)
age_rd = model_data[model_data['Age'].notnull()].drop('Survived',axis=1)
from sklearn.cross_validation import train_test_split
X_train_age, X_test_age, y_train_age, y_test_age = train_test_split(age_rd.drop('Age',axis=1), age_rd['Age'], test_size=0.3, random_state=42)
from sklearn.linear_model import LinearRegression
linear_age_model = LinearRegression()
linear_age_model.fit(X_train_age,y_train_age)
preds = linear_age_model.predict(X_test_age)
from sklearn import metrics
print(metrics.mean_absolute_error(y_test_age,preds))
print(metrics.mean_squared_error(y_test_age,preds))
print(metrics.explained_variance_score(y_test_age,preds))
print(metrics.r2_score(y_test_age,preds))
age_rd['Age']=age_rd['Age'].apply(lambda x: 10*(x//10))
age_y = pd.DataFrame(age_rd['Age'])
age_x = age_rd.drop('Age',axis=1)
age_y = pd.get_dummies(age_y,columns=['Age'])
X_train_rfc_age,X_test_rfc_age, y_train_rfc_age, y_test_rfc_age = train_test_split(age_x, age_y, test_size=0.3, random_state=42)
X_train_rfc_age.head()
from sklearn.ensemble import RandomForestClassifier
rfc_age =RandomForestClassifier(n_estimators=150)
rfc_age.fit(X_train_rfc_age,y_train_rfc_age)
rfc_age_preds=rfc_age.predict(X_test_rfc_age)
from sklearn.metrics import classification_report, confusion_matrix
print(classification_report(y_test_rfc_age,rfc_age_preds))
model_data.info()
model_data.head(2)
def impute(X):
    X_Age = X[~X['Age'].isnull()]
    X_Age['Age'] = X_Age['Age'].apply(lambda x: 10*(x//10))
    X_Age = pd.get_dummies(X_Age,columns=['Age'])
    
    X_NoAge = X[X['Age'].isnull()]
    no_age_preds = pd.DataFrame(rfc_age.predict(X_NoAge.drop(['Survived','Age'],axis=1)),columns=['Age_0.0','Age_10.0','Age_20.0','Age_30.0','Age_40.0','Age_50.0','Age_60.0','Age_70.0','Age_80.0'])
    X = pd.concat([X_NoAge.reset_index(),no_age_preds.reset_index()],axis=1)
    
    return pd.concat([X_Age,X.drop(['Age','index'],axis=1)],axis=0)  
    #return X.drop(['Age','index'],axis=1)
    
imputed_data = impute(model_data)
imputed_x = imputed_data.drop('Survived',axis=1)
imputed_y = imputed_data['Survived']
X_train, X_test, y_train, y_test = train_test_split(imputed_x, imputed_y, test_size=0.3, random_state=9)
rfc = RandomForestClassifier(n_estimators = 200)
rfc.fit(X_train,y_train)
rfc_preds = rfc.predict(X_test)
print(classification_report(y_test,rfc_preds))
print(confusion_matrix(y_test,rfc_preds))
rfc.fit(imputed_x,imputed_y)
test.head()
test2 = test.drop(['Name','Ticket','Cabin'],axis = 1)
test2 = pd.get_dummies(test2,columns=['Sex','Embarked'],drop_first=True)
test2.info()
def impute2(X):
    X_Age = X[~X['Age'].isnull()]
    X_Age['Age'] = X_Age['Age'].apply(lambda x: 10*(x//10))
    X_Age = pd.get_dummies(X_Age,columns=['Age'])
    
    X_NoAge = X[X['Age'].isnull()]
    no_age_preds = pd.DataFrame(rfc_age.predict(X_NoAge.drop(['Age','PassengerId'],axis=1)),columns=['Age_0.0','Age_10.0','Age_20.0','Age_30.0','Age_40.0','Age_50.0','Age_60.0','Age_70.0','Age_80.0'])
    X = pd.concat([X_NoAge.reset_index(),no_age_preds.reset_index()],axis=1)
    
    return pd.concat([X_Age,X.drop(['Age','index'],axis=1)],axis=0)  
    #return X.drop(['Age','index'],axis=1)
test3 = impute2(test2)
test3['Age_80.0'] = 0
test3[test3['Fare'].isnull()] = test3['Fare'].median()
test3.head()
predictions = pd.DataFrame(rfc.predict(test3.drop('PassengerId',axis=1)),columns=['Survived'])
predictions = pd.concat([test3['PassengerId'].reset_index(),predictions['Survived'].reset_index()],axis=1)
predictions.drop('index',axis=1,inplace = True)
predictions.head()
gender_submission.head()
