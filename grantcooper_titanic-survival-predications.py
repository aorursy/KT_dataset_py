import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

from scipy import stats

import seaborn as sns

import matplotlib as plt

import numpy as np

import pandas as pd

from sklearn.model_selection import cross_val_score, GridSearchCV

from sklearn.linear_model import LassoCV, Ridge

from sklearn.svm import SVC

from sklearn.neighbors import KNeighborsClassifier

from sklearn.preprocessing import StandardScaler

from sklearn import preprocessing, svm
train_path = '/kaggle/input/titanic/train.csv'

gender_path = '/kaggle/input/titanic/gender_submission.csv'

test_path = '/kaggle/input/titanic/test.csv'

train = pd.read_csv(train_path)

gender = pd.read_csv(gender_path)

test =pd.read_csv(test_path)

test['Survived'] = np.empty((len(test), 0)).tolist()

Total = train.append(test)

Total.set_index('PassengerId', inplace = True)

Total
for column in Total:

    if Total[column].isnull().sum() != 0:

        print('Missing values in',column,':', Total[column].isnull().sum())
Total = Total.drop(['Ticket','Cabin'], axis = 1)

fare_mode = Total['Fare'].mode().iat[0]

Total['Fare'].fillna(fare_mode,inplace = True)

embarked_mode = Total['Embarked'].mode().iat[0]

Total['Embarked'].fillna(embarked_mode, inplace = True)

for column in Total:

    if Total[column].isnull().sum() != 0:

        print('Missing values in',column,':', Total[column].isnull().sum())
sex_dummy = pd.get_dummies(Total['Sex'])

Total = Total.drop('Sex', axis = 1)

Total = pd.concat([Total, sex_dummy], axis = 1)

embark_dummy = pd.get_dummies(Total['Embarked'])

Total = Total.drop('Embarked', axis = 1)

Total = pd.concat([Total, embark_dummy], axis = 1)

Total
title = lambda x: x.split(',')[1].split('.')[0].strip()

Total['Title']=Total['Name'].map(title)

Total = Total.drop('Name', axis = 1)

Total
title_dummy = pd.get_dummies(Total['Title'])

title_dummy

title_dummy['Mil'] = title_dummy['Capt']+title_dummy['Col']+title_dummy['Major']

title_dummy = title_dummy.drop(['Capt','Col','Major'], axis = 1)

title_dummy['Senior Male Honorific'] = title_dummy['Don']+title_dummy['Sir']

title_dummy = title_dummy.drop(['Don','Sir'], axis = 1)

title_dummy['Senior Female Honorific'] = title_dummy['Dona']+title_dummy['Mme']+title_dummy['Lady']

title_dummy = title_dummy.drop(['Dona','Mme','Lady'], axis = 1)

title_dummy['Ms+Mlle'] = title_dummy['Ms']+title_dummy['Mlle']

title_dummy = title_dummy.drop(['Ms','Mlle'], axis = 1)

title_dummy['Fancy'] = title_dummy['Jonkheer']+title_dummy['the Countess']

title_dummy = title_dummy.drop(['Jonkheer','the Countess'], axis = 1)

Total = Total.drop('Title', axis = 1)

Total = pd.concat([Total, title_dummy], axis = 1)

Total
Age_Total = Total

Age_Total = Age_Total.dropna()

Age_Total

def nans(df): return df[df.isnull().any(axis=1)]

Age_Total_Predict = nans(Total)

Age_Total_Predict
col = list(Age_Total.columns)

col.remove('Age')

col.remove('Survived')

Age_Total[col]
x = Age_Total[col]

x_test = Age_Total_Predict[col]

y_train = Age_Total['Age']

x_train = StandardScaler().fit(x).transform(x)

x_test = StandardScaler().fit(x_test).transform(x_test)
lasso = LassoCV(alphas = [0.01, 0.05, 0.1, 0.15,0.5]).fit(x_train, y_train)

alpha = 0.1

lasso_tuned = LassoCV(alphas = [0.8*alpha, 0.9*alpha, alpha, 1.1*alpha, 1.2*alpha]).fit(x_train, y_train)
svm_age = svm.SVR(kernel = 'rbf')

svm_age.fit(x_train, y_train)
ridge = Ridge()

parameters = {'alpha': [1e-15, 1e-10, 1e-8, 1e-4, 1e-3, 1e-2, 1, 5, 10, 20]}

ridge_age = GridSearchCV(ridge, parameters, scoring = 'neg_root_mean_squared_error', cv=5)

ridge_age.fit(x_train, y_train)

print(ridge_age.best_params_)

print(-ridge_age.best_score_)
print('The average R squared score for the lasso model is:', -cross_val_score(lasso_tuned, x_train, y_train, scoring ='neg_root_mean_squared_error').mean())

print('The average R squared score for the svm model is:', -cross_val_score(svm_age, x_train, y_train, scoring ='neg_root_mean_squared_error').mean())

print('The average R squared score for the ridge model is:', -ridge_age.best_score_)

#Gonna use the lasso model, since it has the lowest RSME
predicted_ages = pd.DataFrame(lasso_tuned.predict(x_test), columns=['Age']) 

Age_Total_Predict.drop('Age',axis = 1)

Age_Total_Predict = Age_Total_Predict.assign(Age = lasso_tuned.predict(x_test).round())

Age_Total_Predict['PassengerId'] = Age_Total_Predict.index

Age_Total_Predict = Age_Total_Predict.reset_index(drop=True)

Age_Total_Predict
Age_Total['PassengerId'] = Age_Total.index

Age_Total = Age_Total.reset_index(drop=True)

Total_Processed = pd.concat([Age_Total_Predict,Age_Total])

Total_Processed.set_index('PassengerId', inplace = True)

Total_Processed = Total_Processed.sort_index()

fix_neg = lambda x: x + 11.148738330664404 if x < 0 else x

Total_Processed['Age'] = Total_Processed['Age'].apply(fix_neg)

Total_Processed
Train_Processed = Total_Processed[:len(train)]

Test_Processed = Total_Processed[len(train):]
col = list(Train_Processed.columns)

col.remove('Survived')

x_train = Train_Processed[col]

y_train = Train_Processed['Survived']

y_train = y_train.astype('int')

x_final = Test_Processed[col]

x_train = preprocessing.StandardScaler().fit(x_train).transform(x_train)

x_final= preprocessing.StandardScaler().fit(x_final).transform(x_final)
k_score = []

k_range = range(1,16)

for k in k_range:

    neigh = KNeighborsClassifier(n_neighbors = k).fit(x_train,y_train)

    

    score =  -cross_val_score(neigh, x_train, y_train, scoring ='neg_root_mean_squared_error').mean()

    

    k_score.append(score)



Scores = pd.DataFrame(k_score, columns = ['Score'])

Scores.index += 1

Scores.sort_values('Score')
neigh = KNeighborsClassifier(n_neighbors = 8).fit(x_train,y_train)

y_KNN = pd.DataFrame(neigh.predict(x_final), columns = ['Survived'])

y_KNN = y_KNN.assign(PassengerId = Test_Processed.index)

cols = y_KNN.columns.tolist()

cols = cols[-1:] + cols[:-1]

y_KNN = y_KNN[cols]

filename = 'Titanic Predictions KNN.csv'

y_KNN.to_csv(filename,index=False)

print('Saved file: ' + filename)
svm_lin = svm.SVC(kernel='linear').fit(x_train, y_train)

print('The average R squared score for the svm_lin is:', cross_val_score(svm_lin, x_train, y_train, scoring ='accuracy').mean())

svm_rbf = svm.SVC(kernel='rbf').fit(x_train, y_train)

print('The average R squared score for the svm_rbf is:', cross_val_score(svm_rbf, x_train, y_train, scoring ='accuracy').mean())
gammas = [0.01,0.1, 1, 10, 100]

for gamma in gammas:

    svm_rbf = svm.SVC(kernel='rbf').fit(x_train, y_train)

    print('The average R squared score for the svm_rbf with a gamma of',gamma,' is:', cross_val_score(svm_rbf, x_train, y_train, scoring ='accuracy').mean())
cs = [0.65,0.67]

for c in cs:

    svm_rbf = svm.SVC(kernel='rbf', C=c).fit(x_train, y_train)

    print('The average R squared score for the svm_rbf with a C of',c,' is:', cross_val_score(svm_rbf, x_train, y_train, scoring ='accuracy').mean())
svm_rbf = svm.SVC(kernel='rbf').fit(x_train, y_train)

y_SVM = pd.DataFrame(svm_rbf.predict(x_final), columns = ['Survived'])

y_SVM = y_SVM.assign(PassengerId = Test_Processed.index)

cols = y_SVM.columns.tolist()

cols = cols[-1:] + cols[:-1]

y_SVM = y_SVM[cols]

filename = 'Titanic Predictions SVM.csv'

y_SVM.to_csv(filename,index=False)

print('Saved file: ' + filename)
print('The average accuracy score for the SVM model is:', cross_val_score(svm_rbf, x_train, y_train, scoring ='accuracy').mean())
