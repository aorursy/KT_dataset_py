import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline

import pandas_profiling

import seaborn as sns

from sklearn import metrics

from sklearn.model_selection import GridSearchCV
train = pd.read_csv("/kaggle/input/titanic/train.csv")

test = pd.read_csv("/kaggle/input/titanic/test.csv")

train.head()
test.head()
train.info()
test.info()
train.isnull().sum()
test.isnull().sum()
train_report = pandas_profiling.ProfileReport(train)
train_report
test_report = pandas_profiling.ProfileReport(test)
test_report
train.shape
test.shape
train.describe(include='object')
test.describe(include = 'object')
train.drop(columns=['PassengerId','Name','Ticket','Cabin'],inplace=True)

test.drop(columns=['PassengerId','Name','Ticket','Cabin'],inplace=True)

train.head()
test.head()
train.isnull().sum()
# replacing missing values in 'Age' column with mean values

train.Age.fillna(train.Age.mean(),inplace=True)
# replacing missing values in 'Embarked' column with mode

train.Embarked.value_counts()
train.Embarked.fillna('S',inplace=True)
train.isnull().sum()
test.isnull().sum()
# replacing missing values in 'Age' column with mean values

test.Age.fillna(test.Age.mean(),inplace=True)
# replacing missing values in 'Fare' column with mean

test.Fare.fillna(test.Fare.mean(),inplace=True)
test.isnull().sum()
# UDF to get summary report of continuous variables

def cont_summ(x):

    return pd.Series([x.mean(),x.median(),x.std(),x.min(),x.quantile(0.01),x.quantile(0.05),x.quantile(0.10),x.quantile(0.25),

              x.quantile(0.50),x.quantile(0.75),x.quantile(0.90),x.quantile(0.95),x.quantile(0.99),x.max()],

             index=['Mean','Median','Std','Min','Q1','Q5','Q10','Q25','Q50','Q75','Q90','Q95','Q99','Max'])
# separating continuous and categorical variables in train data

continuous_vars = train[['Age','SibSp','Parch','Fare']]

categorical_vars = train[['Survived','Pclass','Sex','Embarked']]
# using UDF 'cont_summ' to get summary report of continuous variables

continuous_vars.apply(lambda x: cont_summ(x)).T
# clipping columns 'Age' and 'Fare' 

continuous_vars['Age'] = continuous_vars['Age'].clip(lower=continuous_vars['Age'].dropna().quantile(0.05),upper=continuous_vars['Age'].quantile(0.95))

continuous_vars['Fare'] = continuous_vars['Fare'].clip(lower=continuous_vars['Fare'].dropna().quantile(0.05),upper=continuous_vars['Fare'].quantile(0.95))
# using UDF 'cont_summ' to get summary report of continuous variables

continuous_vars.apply(lambda x: cont_summ(x)).T
# separating continuous and categorical variables in train data

continuous_vars_test = test[['Age','SibSp','Parch','Fare']]

categorical_vars_test = test[['Pclass','Sex','Embarked']]
# using UDF 'cont_summ' to get summary report of continuous variables

continuous_vars_test.apply(lambda x: cont_summ(x)).T
# clipping columns 'Age' and 'Fare' 

continuous_vars_test['Age'] = continuous_vars_test['Age'].clip(lower=continuous_vars_test['Age'].dropna().quantile(0.05),upper=continuous_vars_test['Age'].quantile(0.95))

continuous_vars_test['Fare'] = continuous_vars_test['Fare'].clip(lower=continuous_vars_test['Fare'].dropna().quantile(0.05),upper=continuous_vars_test['Fare'].quantile(0.95))
# using UDF 'cont_summ' to get summary report of continuous variables

continuous_vars_test.apply(lambda x: cont_summ(x)).T
categorical_vars.head()
sns.countplot(categorical_vars.Survived)

plt.show()
sns.countplot(categorical_vars.Pclass)

plt.show()
sns.countplot(categorical_vars.Sex)

plt.show()
sns.countplot(categorical_vars.Embarked)

plt.show()
categorical_vars_test.head()
sns.countplot(categorical_vars_test.Pclass)

plt.show()
sns.countplot(categorical_vars_test.Sex)

plt.show()
sns.countplot(categorical_vars_test.Embarked)

plt.show()
# UDF to create dummy variables

def dummy_var(df,colname):

    dummy = pd.get_dummies(df[colname],prefix=colname,drop_first=True)

    df = pd.concat([df,dummy],axis=1)

    df.drop(colname,axis = 1,inplace=True)

    return df

categorical_vars.dtypes
categorical_vars['Pclass'] = categorical_vars['Pclass'].astype('object')

dummy_df = dummy_var(categorical_vars,['Embarked', 'Pclass', 'Sex'])

dummy_df.head()
train = pd.concat([continuous_vars,dummy_df],axis=1)
train.head()
categorical_vars_test.dtypes
categorical_vars_test['Pclass'] = categorical_vars_test['Pclass'].astype('object')

dummy_df_test = dummy_var(categorical_vars_test,['Embarked', 'Pclass', 'Sex'])

dummy_df_test.head()
test = pd.concat([continuous_vars_test,dummy_df_test],axis=1)
test.head()
test = test[['Age', 'Embarked_Q', 'Embarked_S', 'Fare', 'Parch', 'Pclass_2',

       'Pclass_3', 'Sex_male', 'SibSp']]
feature_columns = train.columns.difference(['Survived'])

train_X = train[feature_columns] 

train_Y = train['Survived']
test_X = test.copy()
from sklearn.linear_model import LogisticRegression

log_titanic = LogisticRegression(random_state=123)
log_titanic.fit(train_X,train_Y)
# this is considering that the probability of an event is 50%

train['predicted'] = log_titanic.predict(train_X)
confusion_mat = metrics.confusion_matrix(train_Y,train['predicted'],[1,0])
confusion_mat
sns.heatmap(confusion_mat,annot=True,fmt='.2f')

plt.show()
log_titanic.predict_proba(train_X)[0]
train['predicted probability'] = log_titanic.predict_proba(train_X)[:,1]
metrics.roc_auc_score(train_Y,train['predicted probability'])
# creating dataframe with actual vales and predicted probability

train_values = pd.DataFrame([train_Y,train['predicted probability']]).T

train_values.columns = ['Actual','Prob']
train_values.head()
cut_off_df = pd.DataFrame()

for value in np.linspace(0,1,100):

    train_values['cut_off'] = value

    train_values['Predicted'] = train_values['Prob'].apply(lambda x: 1.0 if x>value else 0.0)

    train_values['tp'] = train_values.apply(lambda x: 1.0 if x['Actual']==1.0 and x['Predicted']==1.0 else 0.0, axis=1)

    train_values['fp'] = train_values.apply(lambda x: 1.0 if x['Actual']==0.0 and x['Predicted']==1.0 else 0.0, axis=1)

    train_values['tn'] = train_values.apply(lambda x: 1.0 if x['Actual']==0.0 and x['Predicted']==0.0 else 0.0, axis=1)

    train_values['fn'] = train_values.apply(lambda x: 1.0 if x['Actual']==1.0 and x['Predicted']==0.0 else 0.0, axis=1)

    sensitivity = train_values['tp'].sum()/(train_values['tp'].sum() + train_values['fn'].sum())

    specificity = train_values['tn'].sum()/(train_values['tn'].sum() + train_values['fp'].sum())

    accuracy = (train_values['tp'].sum() + train_values['tn'].sum())/(train_values['tp'].sum() + train_values['fn'].sum() + train_values['fp'].sum() + train_values['tn'].sum())

    temp = pd.DataFrame([value,sensitivity,specificity,accuracy]).T

    temp.columns = ['cut_off','sensitivity','specificity','accuracy']

    cut_off_df = pd.concat([cut_off_df,temp],axis=0)

cut_off_df['total'] = cut_off_df['sensitivity']+ cut_off_df['specificity']

    
cut_off_df
# cut-off will be where sum of sensitivity and specificity i.e, totlal is max

cut_off_df[cut_off_df['total'] == cut_off_df['total'].max()]
# cut-off is 0.57. So finding new labels using new cut-off

train['new_predicted'] = train['predicted probability'].apply(lambda x : 1.0 if x > 0.57 else 0.0)
train.head(20)
cm_new = metrics.confusion_matrix(train_Y,train['new_predicted'],[1,0])

cm_new
plt.figure(figsize=(10,8))

sns.heatmap(cm_new,annot = True,fmt='1.0f',xticklabels=['Survived','Not Survived'], yticklabels=['Survived','Not Survived'])

plt.xlabel('Predicted')

plt.ylabel('Actual')

plt.show()
test['predicted'] = log_titanic.predict(test_X)
test['predicted probability'] = log_titanic.predict_proba(test_X)[:,1]
test.head(10)
# cut-off is 0.57. So finding new labels using new cut-off

test['new_predicted'] = test['predicted probability'].apply(lambda x : 1.0 if x > 0.57 else 0.0)
test.head(10)
from sklearn.ensemble import RandomForestClassifier

random_titanic = RandomForestClassifier(oob_score=True,random_state=123) 
random_titanic
param_grid = {'max_depth':np.arange(2,6),'max_features':np.arange(5,8),'n_estimators':[100,150,200]}
from sklearn.model_selection import GridSearchCV

random_grid = GridSearchCV(random_titanic,param_grid=param_grid,n_jobs=-1,verbose=1,cv=10)
random_grid.fit(train_X,train_Y)
random_grid.best_params_



random_titanic = RandomForestClassifier(oob_score=True,random_state=123,max_depth=5, max_features= 6

                                        ,n_estimators=150) 


random_titanic.fit(train_X,train_Y)
random_titanic.predict(train_X)
random_titanic.predict(test_X)
metrics.confusion_matrix(train_Y,random_titanic.predict(train_X),[1,0])
metrics.accuracy_score(train_Y,random_titanic.predict(train_X))
from sklearn.ensemble import GradientBoostingClassifier

gradient_titanic = GradientBoostingClassifier()
gradient_titanic
pargrid_gradient = {'n_estimators': [100,150,250,300],

               'learning_rate': [10 ** x for x in range(-3, 1)],

              'max_features':np.arange(3,10)}
gradient_grid = GridSearchCV(gradient_titanic,param_grid=pargrid_gradient,n_jobs=-1,verbose=1,cv=10)
gradient_grid.fit(train_X,train_Y)
gradient_grid.best_params_
gradient_titanic = GradientBoostingClassifier(n_estimators=250,learning_rate=0.1,max_features=9)


gradient_titanic.fit(train_X,train_Y)
gradient_titanic.predict(train_X)
gradient_titanic.predict(test_X)
metrics.confusion_matrix(train_Y,gradient_titanic.predict(train_X),[1,0])
metrics.accuracy_score(train_Y,gradient_titanic.predict(train_X))
from xgboost.sklearn import XGBClassifier

titanic_xgboost = XGBClassifier()
titanic_xgboost
pargrid_xg = {'n_estimators': [100,150,250,300],

               'learning_rate': [10 ** x for x in range(-3, 1)],

              'max_depth':np.arange(3,10)}
titanic_xgboost_grid = GridSearchCV(titanic_xgboost,param_grid=pargrid_xg,n_jobs=-1,verbose=1,cv=10)
titanic_xgboost_grid.fit(train_X,train_Y)
titanic_xgboost_grid.best_params_
titanic_xgboost = XGBClassifier(n_estimators=250,learning_rate=0.01,max_depth=9)


titanic_xgboost.fit(train_X,train_Y)
train_X.columns
test_X.head()
titanic_xgboost.predict(train_X)
xgboost_results = pd.DataFrame(titanic_xgboost.predict(test_X))
xgboost_results.head()
metrics.confusion_matrix(train_Y,titanic_xgboost.predict(train_X),[1,0])
metrics.accuracy_score(train_Y,gradient_titanic.predict(train_X))
from imblearn.under_sampling import RandomUnderSampler 

from collections import Counter



rus = RandomUnderSampler(random_state=500)

X_rus_train, y_rus_train = rus.fit_sample(train_X, train_Y)

print('Original dataset shape {}'.format(Counter(train_Y)))

print('Undersampled dataset shape {}'.format(Counter(y_rus_train)))



from imblearn.over_sampling import RandomOverSampler 

from collections import Counter



ros = RandomOverSampler(random_state=500)

X_ros_train, y_ros_train = ros.fit_sample(train_X, train_Y)

print('Original dataset shape {}'.format(Counter(train_Y)))

print('Oversampled dataset shape {}'.format(Counter(y_ros_train)))
X_ros_train
from sklearn.ensemble import RandomForestClassifier

random_titanic = RandomForestClassifier(oob_score=True,random_state=123) 
random_titanic


param_grid = {'max_depth':np.arange(4,6),'max_features':np.arange(5,8),'n_estimators':[50,100,150]}
from sklearn.model_selection import GridSearchCV

random_grid = GridSearchCV(random_titanic,param_grid=param_grid,n_jobs=-1,verbose=1,cv=10)
random_grid.fit(X_ros_train,y_ros_train)
random_grid.best_params_



random_titanic = RandomForestClassifier(oob_score=True,random_state=123,max_depth=5, max_features= 7

                                        ,n_estimators=100) 


random_titanic.fit(X_ros_train,y_ros_train)
random_titanic.predict(X_ros_train)
metrics.confusion_matrix(train_Y,random_titanic.predict(train_X),[1,0])
metrics.accuracy_score(y_ros_train,random_titanic.predict(X_ros_train))
from sklearn.ensemble import GradientBoostingClassifier

gradient_titanic = GradientBoostingClassifier()
gradient_titanic
pargrid_gradient = {'n_estimators': [300,350,400,450,600,],

               'learning_rate': [0.1,0.5,1,1.5],

              'max_features':np.arange(4,8)}
gradient_grid = GridSearchCV(gradient_titanic,param_grid=pargrid_gradient,n_jobs=-1,verbose=1,cv=10)
gradient_grid.fit(X_ros_train,y_ros_train)
gradient_grid.best_params_
gradient_titanic = GradientBoostingClassifier(n_estimators=350,learning_rate=0.5,max_features=7)


gradient_titanic.fit(X_ros_train,y_ros_train)
gradient_titanic.predict(X_ros_train)
test['gradientPredicted'] = gradient_titanic.predict(test_X)
test['gradientPredicted'].head()
test.to_csv('gradient_balnced.csv')
metrics.confusion_matrix(y_ros_train,gradient_titanic.predict(X_ros_train),[1,0])
metrics.accuracy_score(y_ros_train,gradient_titanic.predict(X_ros_train))