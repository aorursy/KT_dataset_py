# Disable warnings in Anaconda

import warnings

warnings.filterwarnings('ignore')



import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

sns.set()
df_train = pd.read_csv('../input/train.csv')

df_test  = pd.read_csv('../input/test.csv')

df_full = df_train.append(df_test)

pred_passid = df_test.PassengerId
print (df_train.shape)

print (df_test.shape)

print (df_full.shape)
df_full.info()
df_full.isna().sum()
# Age Distribution

print ("The mean of age is: %.1f" % df_full['Age'].mean())

print ("The median of age: %.1f" % df_full['Age'].median())
# Age

sns.distplot(df_full['Age'], bins=15);
# Since Age skewes to the right, we will use median for the NA values.

df_full['Age']=df_full['Age'].fillna(df_full['Age'].median())

df_full['Age'].describe()
df_full['Fare']=df_full['Fare'].fillna(df_full['Fare'].median())
df_full.Embarked.value_counts()
df_full.loc[df_full['Embarked'].isna(),'Embarked']='S'
df_full.drop('Cabin', axis =1, inplace = True)

df_full.head()
df_full.isna().sum()
# Imbalanced Classes

df_full['Survived'].value_counts(normalize=True)
#Sex

sns.countplot(x='Sex',hue='Survived',data=df_full);
# Fare vs. Survived

plt.figure(figsize=(15,8))

ax = sns.kdeplot(df_full["Fare"][df_full.Survived == 1], color="darkturquoise", shade=True)

sns.kdeplot(df_full["Fare"][df_full.Survived == 0], color="lightcoral", shade=True)

plt.legend(['Survived', 'Died'])

plt.title('Density Plot of Fare for Surviving Population and Deceased Population')

ax.set(xlabel='Fare')

plt.xlim(-10,85)

plt.show()
# Pclass vs Survived

sns.countplot(x='Pclass',hue='Survived',data=df_full)

plt.show()
df_full['Title']=df_full['Name'].str.split(', ', expand=True)[1].str.split('. ',expand=True)[0]

df_full['Title'].value_counts()
Weird_Title = ['Rev','Mlle','Col','Marjor','Capt','Jonkheer','Mme','th','Lady','Major', 'Dr', 'Dona','Don']

df_full[df_full['Title'].isin(Weird_Title)].sort_values(by=['Sex','Title'], ascending = True)
df_full['Title']=df_full['Title'].replace(['Lady', 'Mlle','Mme','th','Ms', 'Dona'], 'Miss')

df_full['Title']=df_full['Title'].replace(['Rev','Col','Marjor','Capt','Jonkheer','Don','th', 'Sir','Major'], 'Mr')

df_full['Title'].value_counts()
df_full['Fam_num']=df_full['SibSp']+df_full['Parch']+1

df_full.head()
# To remove 

df_full.drop(['PassengerId' ,'Name', 'Ticket'],axis=1, inplace=True)
df_train_cleaned = df_full.iloc[0:891,:]

df_test_cleaned = df_full.iloc[891:,:]
X = df_train_cleaned.drop(['Survived'], axis=1)

y = df_train_cleaned['Survived']
X.describe()
X.describe(exclude='number')
X=pd.get_dummies(X, drop_first=True)

X.head()
#np.corrcoef(X['Sex_female'],X['Sex_male'])

sns.heatmap(X.corr(), annot=True,fmt=".1f");
X.info()
from sklearn.model_selection import train_test_split
X_train, X_valid, y_train, y_valid = train_test_split(X,y, test_size=0.3, random_state=17)
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(random_state=17, class_weight='balanced')

lr.fit(X_train,y_train)
from sklearn.metrics import accuracy_score, confusion_matrix
# Without 'balanced': 0.6902

# with: 0.723; 0.746268656716418

# with Sex and Embarked: 0.7649

print(accuracy_score(y_valid, lr.predict(X_valid)))

print(confusion_matrix(y_valid,lr.predict(X_valid)))
prob = lr.predict_proba(X_train)

prob_df = pd.DataFrame({'prob_no': prob[:,0],

                       'prob_yes': prob[:,1],

                       'actual': y_train}, index=X_train.index)

prob_df.head()
pd.DataFrame({'features': X_valid.columns,

              'coef': lr.coef_.flatten().tolist(),

              'abs_coef': np.abs(lr.coef_.flatten().tolist())}).sort_values(by='abs_coef', ascending=False)
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(random_state=17, class_weight='balanced')

rf.fit(X_train,y_train)
pd.DataFrame({'Feature': X_train.columns,

             'Importance': rf.feature_importances_}).sort_values(by='Importance', ascending=False)
print ('Accuracy (Test): %.3f' % accuracy_score(y_valid,rf.predict(X_valid)))

print (confusion_matrix(y_valid, rf.predict(X_valid)))
from xgboost import XGBClassifier



xgb_model = XGBClassifier(random_state=0)

xgb_model.fit(X_train,y_train)
print ('Accuracy (Test): %.3f' % accuracy_score(y_valid,

                                                xgb_model.predict(X_valid)))
print (accuracy_score(y_valid,

                      xgb_model.predict(X_valid)))
from sklearn.model_selection import GridSearchCV
# Set up the hyperparamter grid

parameters = {'C': (0.0001, 0.001, 0.01, 0.1, 1, 10)}



lr_cv = GridSearchCV(lr, parameters, scoring='accuracy', cv=5)

lr_cv.fit(X_train,y_train)
lr_cv.best_score_, lr_cv.best_params_
accuracy_score(y_valid, lr_cv.predict(X_valid))
parameters_rf = {'n_estimators': [4, 6, 9], 

              'max_features': ['log2', 'sqrt','auto'], 

              'criterion': ['entropy', 'gini'],

              'max_depth': [2, 3, 5, 10], 

              'min_samples_split': [2, 3, 5],

              'min_samples_leaf': [1,5,8]}



rf_cv=GridSearchCV(rf, parameters_rf, scoring='accuracy',cv=5, n_jobs=-1)

rf_cv.fit(X_train,y_train)
rf_cv.best_score_, rf_cv.best_params_
accuracy_score(y_valid, rf_cv.predict(X_valid))
parameters_xgb = {

    "n_estimators": [10,20,30,40,50,60,70,80,90,100],

    "learning_rate": [0.1, 0.2, 0.3,0.4,0.5]

}



xgb_cv=GridSearchCV(xgb_model, parameters_xgb, scoring = 'accuracy',cv=5, n_jobs=-1)

xgb_cv.fit(X_train,y_train)

print (xgb_cv.best_score_)

print (xgb_cv.best_params_)
accuracy_score(y_valid, xgb_cv.predict(X_valid))
df_test_cleaned=pd.get_dummies(df_test_cleaned, drop_first=True)

df_test_cleaned.drop(['Survived'], axis=1, inplace=True)

df_test_cleaned.head()
pred = xgb_cv.predict(df_test_cleaned)

pred = pred.astype(np.int64)
output= pd.DataFrame({'PassengerId': pred_passid,

                     'Survived': pred})
output.to_csv('titanic.csv', index=False)