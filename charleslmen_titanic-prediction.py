# Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
# Import train datastet
train = pd.read_csv('../input/train.csv')
train.head()
test=pd.read_csv('../input/test.csv')
test.head()
sns.set_style('whitegrid')
sns.countplot(x='Survived',data=train)
# Have a look with the missing values
sns.heatmap(train.isnull(),yticklabels=False,cbar=False,cmap='viridis')
# Drop the irrelative features from the train set
irrelative_drop=['PassengerId','Name','Ticket','Cabin']
train=train.drop(columns=irrelative_drop)
train.head()
# Taking care of the missing value
train_class1=train[train['Pclass']==1]
train_class2=train[train['Pclass']==2]
train_class3=train[train['Pclass']==3]
X1=train_class1.iloc[:,:].values
X2=train_class2.iloc[:,:].values
X3=train_class3.iloc[:,:].values
from sklearn.impute import SimpleImputer
imputer1=SimpleImputer(missing_values=np.nan,strategy='mean')
imputer2=SimpleImputer(missing_values=np.nan,strategy='mean')
imputer3=SimpleImputer(missing_values=np.nan,strategy='mean')
imputer1.fit(X1[:,3:4])
imputer2.fit(X2[:,3:4])
imputer3.fit(X3[:,3:4])
X1[:,3:4]=imputer1.transform(X1[:,3:4])
X2[:,3:4]=imputer2.transform(X2[:,3:4])
X3[:,3:4]=imputer3.transform(X3[:,3:4])
X1_df=pd.DataFrame(X1,columns=train_class1.columns)
X2_df=pd.DataFrame(X2,columns=train_class2.columns)
X3_df=pd.DataFrame(X3,columns=train_class3.columns)
train=pd.concat([X1_df,X2_df,X3_df])
train.head()
sns.heatmap(train.isnull(),yticklabels=False,cbar=False,cmap='viridis')
# Do the same process for the test set
test=test.drop(columns=irrelative_drop)
test.head()
test_class1=test[test['Pclass']==1]
test_class2=test[test['Pclass']==2]
test_class3=test[test['Pclass']==3]
test_X1=test_class1.iloc[:,:].values
test_X2=test_class2.iloc[:,:].values
test_X3=test_class3.iloc[:,:].values
test_X1[:,2:3]=imputer1.transform(test_X1[:,2:3])
test_X2[:,2:3]=imputer2.transform(test_X2[:,2:3])
test_X3[:,2:3]=imputer3.transform(test_X3[:,2:3])
test_X1_df=pd.DataFrame(test_X1,columns=test_class1.columns)
test_X2_df=pd.DataFrame(test_X2,columns=test_class2.columns)
test_X3_df=pd.DataFrame(test_X3,columns=test_class3.columns)
test=pd.concat([test_X1_df,test_X2_df,test_X3_df])
test.head()
sns.heatmap(test.isnull(),yticklabels=False,cbar=False,cmap='viridis')
embark=pd.get_dummies(train['Embarked'],drop_first=True)
sex=pd.get_dummies(train['Sex'],drop_first=True)
train=pd.concat([train,sex,embark],axis=1)
train.drop(['Sex','Embarked'],axis=1,inplace=True)
train.head()
embark_test=pd.get_dummies(test['Embarked'],drop_first=True)
sex_test=pd.get_dummies(test['Sex'],drop_first=True)
test=pd.concat([test,sex_test,embark_test],axis=1)
test.drop(['Sex','Embarked'],axis=1,inplace=True)
Fare=test.values
from sklearn.impute import SimpleImputer
imputer=SimpleImputer(missing_values=np.nan,strategy='mean')
imputer.fit(Fare[:,4:5])
Fare[:,4:5]=imputer.transform(Fare[:,4:5])
test=pd.DataFrame(Fare,columns=test.columns)
test.head()
sns.heatmap(test.isnull(),yticklabels=False,cbar=False,cmap='viridis')
# Define the X_train, X_test, y_train, y_test,datasets
X_train=train.iloc[:,1:].values.astype('float64')
X_test=test.values.astype('float64')
y_train=train.iloc[:,0:1].values.astype('int64')
# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X=StandardScaler()
X_train=sc_X.fit_transform(X_train)
X_test=sc_X.transform(X_test)
# Using kernel SVC model to see the results
from sklearn.svm import SVC
classifier=SVC(kernel='rbf',gamma=0.1,random_state=0)
classifier.fit(X_train,y_train)
y_pred = classifier.predict(X_test)
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator=classifier, X=X_train,y=y_train,cv=10)
accuracies.mean()
from sklearn.model_selection import GridSearchCV
parameters = [{'C': [1, 10, 100, 1000], 'kernel': ['linear']},
              {'C': [1, 10, 100, 1000], 'kernel': ['rbf'], 'gamma': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]}]
grid_search = GridSearchCV(estimator = classifier,
                           param_grid = parameters,
                           scoring = 'accuracy',
                           cv = 10,
                           n_jobs = -1)
grid_search = grid_search.fit(X_train, y_train)
print(grid_search.best_score_)
print(grid_search.best_params_)
test=pd.read_csv('../input/test.csv')
sub_df=pd.DataFrame({'PassengerID':test['PassengerId'].values})
sub_df['Survived']=y_pred
sub_df.to_csv('submission.csv',index=False)
