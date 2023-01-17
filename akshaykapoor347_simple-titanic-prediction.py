import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt 
%matplotlib inline
df_train = pd.read_csv('../input/train.csv')
df_test = pd.read_csv('../input/test.csv')
df_gender_submission = pd.read_csv('../input/gender_submission.csv')
df_train.head()
df_train.describe()
df_train.info()
df_train.isnull().sum()
df_train.columns
corr = list(df_train.columns[1:])
corr = df_train[corr].corr()

plt.figure(figsize=(16,8))
ax = sns.heatmap(corr, cmap='coolwarm', annot=True)
a = df_train[['Survived', 'Pclass']].groupby(['Pclass']).mean()
a
a.plot.bar()
s = df_train[['Survived', 'Sex']].groupby(['Sex']).mean()
s
s.plot.bar()
a = df_train[['Survived', 'Sex','Pclass']].groupby(['Sex', 'Pclass']).mean()
a
a.plot.bar()
df_train['AgeRange'] = pd.cut(df_train['Age'], 5, precision=0)
ag = df_train[['AgeRange', 'Survived']].groupby(['AgeRange']).mean()
ag

plt.figure(figsize=(12,7))
ax = ag.plot.bar()
ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right")
plt.show
df_train[['SibSp', 'Survived']].groupby(['SibSp']).mean()
df_train[['Parch', 'Survived']].groupby(['Parch']).mean()
df_train[['Embarked', 'Survived']].groupby(['Embarked']).mean()
df_train[['Embarked', 'Survived']].groupby(['Embarked']).mean().plot.bar()
grid = sns.FacetGrid(df_train, col='Survived', row='Pclass')
grid.map(plt.hist, 'Age', alpha=.6, bins=20)
sns.barplot(x='Survived', y='Fare', alpha=.5, data = df_train)
df_train.columns
df_train.head()
df_train = df_train.drop(['Name','Ticket','Cabin'], axis=1)
df_test = df_test.drop(['Name','Ticket','Cabin'], axis=1)
df_train.head()
di = {'male':0,'female':1}
df_train['Sex'] = df_train['Sex'].map(di)
sns.boxplot(x='Pclass',y='Age',data=df_train)
def agerep(cols):
    Age = cols[0]
    Pclass = cols[1]
    if pd.isnull(Age):
        if Pclass == 1:
          return 38
        elif Pclass == 2:
          return 28
        else:
          return 24
    else:
        return Age
        
        
        
df_train['Age'] = df_train[['Age','Pclass']].apply(agerep,axis = 1)
df_train['Embarked'] = df_train['Embarked'].fillna('S')
df_train.isnull().sum()
df_train.loc[ df_train['Age'] <= 16, 'Age'] = 0
df_train.loc[(df_train['Age'] > 16) & (df_train['Age'] <= 32), 'Age'] = 1
df_train.loc[(df_train['Age'] > 32) & (df_train['Age'] <= 48), 'Age'] = 2
df_train.loc[(df_train['Age'] > 48) & (df_train['Age'] <= 64), 'Age'] = 3
df_train.loc[ df_train['Age'] > 64, 'Age'] = 4
df_train = df_train.drop(['AgeRange'], axis=1)
df_train = df_train.drop(['PassengerId'], axis=1)
df_train['Age']=df_train['Age'].astype(int)
df_train['Embarked'] = df_train['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} )
df_train.loc[ df_train['Fare'] <= 7.91, 'Fare'] = 0
df_train.loc[(df_train['Fare'] > 7.91) & (df_train['Fare'] <= 14.454), 'Fare'] = 1
df_train.loc[(df_train['Fare'] > 14.454) & (df_train['Fare'] <= 31), 'Fare']   = 2
df_train.loc[ df_train['Fare'] > 31, 'Fare'] = 3
df_train['Fare'] = df_train['Fare'].astype(int)
df_test['Embarked'] = df_test['Embarked'].fillna('S')
df_test['Age'] = df_test[['Age','Pclass']].apply(agerep,axis = 1)
df_test['Embarked'] = df_test['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} )
df_test.loc[ df_test['Fare'] <= 7.91, 'Fare'] = 0
df_test.loc[(df_test['Fare'] > 7.91) & (df_test['Fare'] <= 14.454), 'Fare'] = 1
df_test.loc[(df_test['Fare'] > 14.454) & (df_test['Fare'] <= 31), 'Fare']   = 2
df_test.loc[ df_test['Fare'] > 31, 'Fare'] = 3

df_test.loc[ df_train['Age'] <= 16, 'Age'] = 0
df_test.loc[(df_train['Age'] > 16) & (df_test['Age'] <= 32), 'Age'] = 1
df_test.loc[(df_train['Age'] > 32) & (df_test['Age'] <= 48), 'Age'] = 2
df_test.loc[(df_train['Age'] > 48) & (df_test['Age'] <= 64), 'Age'] = 3
df_test.loc[ df_train['Age'] > 64, 'Age'] = 4

df_test['Sex'] = df_test['Sex'].map(di)

df_test['Age']=df_test['Age'].astype(int)
df_test['Fare'] = df_test['Fare'].fillna(2)
df_test['Fare'] = df_test['Fare'].astype(int)
df_test.head()
df_train.head()
test_data = df_gender_submission['Survived']
combined_test = pd.merge(df_test, df_gender_submission, on = 'PassengerId')
from sklearn.model_selection import train_test_split
from sklearn.cross_validation import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn import metrics

X_train = df_train.drop("Survived", axis=1)
Y_train = df_train["Survived"]
#X_test  = df_test.drop("PassengerId", axis=1)
X_test  = combined_test.drop(["PassengerId","Survived"], axis=1)
X_train.shape, Y_train.shape, X_test.shape
# I have created a function to perform k folds cross validation which helps in obtaining a better insight to test the accuracy of the model
# More info at https://www.analyticsvidhya.com/blog/2018/05/improve-model-performance-cross-validation-in-python-r/

def classification_model(model, data, predictors, outcome):
  #Fit the model:
  model.fit(data[predictors],data[outcome])
  
  predictions = model.predict(data[predictors])
  
  accuracy = metrics.accuracy_score(predictions,data[outcome])
  print("Accuracy : %s" % "{0:.3%}".format(accuracy))

  #Perform k-fold cross-validation with 5 folds
  kf = KFold(data.shape[0],n_folds= 5)
  error = []
  for train, test in kf:
    # Filter the training data
    train_predictors = (data[predictors].iloc[train,:])
    train_target = data[outcome].iloc[train]
    model.fit(train_predictors, train_target)
    
    error.append(model.score(data[predictors].iloc[test,:], data[outcome].iloc[test]))
    
    print("Cross-Validation Score : %s" % "{0:.3%}".format(np.mean(error)))
    
  model.fit(data[predictors],data[outcome]) 
model = LogisticRegression()
model.fit(X_train, Y_train)

Y_pred = model.predict(X_test)

predictor_var = list(X_train[1:])
outcome_var='Survived'
classification_model(model,df_train,predictor_var,outcome_var)
print('Accuracy on Test data:')
print(accuracy_score(combined_test['Survived'], Y_pred))
print('\n')
print(classification_report(combined_test['Survived'], Y_pred))
model =  KNeighborsClassifier(n_neighbors = 3)
model.fit(X_train, Y_train)
Y_pred = model.predict(X_test)

predictor_var = list(X_train[1:])
outcome_var='Survived'
classification_model(model,df_train,predictor_var,outcome_var)
print('Accuracy on Test data:')
print(accuracy_score(combined_test['Survived'], Y_pred))
print('\n')
print(classification_report(combined_test['Survived'], Y_pred))
model = RandomForestClassifier()
model.fit(X_train, Y_train)
Y_pred = model.predict(X_test)

predictor_var = list(X_train[1:])
outcome_var='Survived'
classification_model(model,df_train,predictor_var,outcome_var)
print('Accuracy on Test data:')
print(accuracy_score(combined_test['Survived'], Y_pred))
print('\n')
print(classification_report(combined_test['Survived'], Y_pred))
model = SVC()
model.fit(X_train, Y_train)
Y_pred = model.predict(X_test)

predictor_var = list(X_train[1:])
outcome_var='Survived'
classification_model(model,df_train,predictor_var,outcome_var)
print('Accuracy on Test data:')
print(accuracy_score(combined_test['Survived'], Y_pred))
print('\n')
print(classification_report(combined_test['Survived'], Y_pred))
submit = pd.DataFrame({
        "PassengerId": combined_test["PassengerId"],
        "Survived": Y_pred
    })
#submit.to_csv('../submit.csv', index=False)
