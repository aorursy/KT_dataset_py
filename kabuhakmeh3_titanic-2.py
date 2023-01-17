import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
# Get Scaled data

def scale_data(X_Training, X_Testing):

    sc = StandardScaler()

    sc.fit(X_Training)

    X_Training_std = sc.transform(X_Training)

    X_Testing_std = sc.transform(X_Testing)

    return X_Training_std, X_Testing_std



# Logistic Regression

def run_logistic_regression(X_Training, X_Testing, y_Training):

    logreg = LogisticRegression()

    logreg.fit(X_Training, y_Training)

    y_Predicted = logreg.predict(X_Testing)

    return y_Predicted
test_data = pd.read_csv('../input/test.csv') # 418 rows test_data.describe()

train_data = pd.read_csv('../input/train.csv') # 891 rows train_data.describe()
test_data.head(3)
plt.style.use('fivethirtyeight')

sns.countplot(x='Embarked', data=train_data, hue='Survived') 

# looks like embarking at "S" lowers the likelihood of survival 
sns.countplot(x='Pclass', data=train_data, hue='Survived') 

# 3rd class passengers are less likely to survive
plt.figure()

train_data[train_data['Survived'] == 1]['Age'].hist(alpha=0.5, color='blue',

                                                bins=40, label='Survived')

train_data[train_data['Survived'] == 0]['Age'].hist(alpha=0.5, color='red', 

                                                bins = 40, label='Did Not Survive')

plt.legend()

plt.xlabel('Age')

plt.ylabel('Count')

# Passengers aged ~20-30 seem to be more likely to not survive

# otherwise, the ratio is close to 1:1
# get an idea of where which data are missing

train_data.isnull().sum()
# impute the age based on average within gender, class, and whether they had family (sibsp)

ages = train_data.groupby(['Pclass','Sex','SibSp'])['Age'].mean()



# large families have low age average, replace by smallest value for nan

ages.fillna(method='ffill', inplace=True) # forward fill Nan values ... reasonable for now

train_data.drop(['PassengerId', 'Name', 'Ticket','Cabin'], axis=1, inplace=True)

test_data.drop(['PassengerId', 'Name', 'Ticket','Cabin'], axis=1, inplace=True)
#train_data.fillna(method='ffill',inplace=True) ## OLD method

df_age = train_data[train_data['Age'].isnull()]

df_age.reset_index(drop=True, inplace=True)

imputed_age_list = []

# get ages from averaged list

for i in df_age.index.values:

    imputed_age_list.append(ages[df_age.iloc[i]['Pclass']][df_age.iloc[i]['Sex']][df_age.iloc[i]['SibSp']])

imputed_age_series = pd.Series(imputed_age_list)

df_age['Age'] = imputed_age_series

df_no_nan= train_data.dropna()

all_ages = pd.concat([df_no_nan, df_age])
all_ages.isnull().sum()
# Convert categorical data into 1 or 0

final_data = pd.get_dummies(all_ages, prefix=['Sex', 'Embarked'])

final_data.head(3)
X = final_data.drop('Survived', axis=1)

y = final_data['Survived']
from sklearn.model_selection import train_test_split

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=4)
# let's test out a few models

from sklearn.linear_model import LogisticRegression

from sklearn.neighbors import KNeighborsClassifier

from sklearn.ensemble import RandomForestClassifier
# Logistic Regression

logreg = LogisticRegression()

logreg.fit(X_train, y_train)

y_logreg = logreg.predict(X_test)

print('Classification Report')

print(classification_report(y_test, y_logreg))

print('Confusion Matrix')

print(confusion_matrix(y_test, y_logreg))

print('Accuracy Score')

print(accuracy_score(y_test, y_logreg))
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()

sc.fit(X_train)

X_train_std = sc.transform(X_train)

X_test_std = sc.transform(X_test)
# Logistic Regression w/StandardScaler

logreg_ss = LogisticRegression()

logreg_ss.fit(X_train_std, y_train)

y_logreg_ss = logreg_ss.predict(X_test_std)

print('Classification Report')

print(classification_report(y_test, y_logreg_ss))

print('Confusion Matrix')

print(confusion_matrix(y_test, y_logreg_ss))

print('Accuracy Score')

print(accuracy_score(y_test, y_logreg_ss))
# KNN - try a range of neighbor values

knn_accuracy = []

for i in range(1,101):

        knn = KNeighborsClassifier(n_neighbors=i)

        knn.fit(X_train, y_train)

        y_pred = knn.predict(X_test)

        knn_accuracy.append(accuracy_score(y_test, y_pred))

print("Maximum KNN Accuracy is: ", max(knn_accuracy), ' located at k-value: ', knn_accuracy.index(max(knn_accuracy))+1)
knn_ss_accuracy = []

for i in range(1,101):

        knn = KNeighborsClassifier(n_neighbors=i)

        knn.fit(X_train_std, y_train)

        y_pred = knn.predict(X_test_std)

        knn_ss_accuracy.append(accuracy_score(y_test, y_pred))

print("Maximum KNN Accuracy is: ", max(knn_ss_accuracy), ' located at k-value: ', knn_ss_accuracy.index(max(knn_ss_accuracy))+1)
# plot both knn graphs on one figure

x_plt = range(1,101)

plt.plot(x_plt,knn_accuracy)

plt.plot(x_plt,knn_ss_accuracy)

plt.title('knn comparison')

plt.xlabel('k-value')

plt.ylabel('accuracy score')

plt.legend(['knn','knn w/scaling'])
# Random Forests

rfc_accuracy=[]

n_est = [10,20,30,40,50,60,70,80,90,100]

for i in range(len(n_est)):

    rfc = RandomForestClassifier(n_estimators=n_est[i])

    rfc.fit(X_train, y_train)

    y_pred =rfc.predict(X_test)

    rfc_accuracy.append(accuracy_score(y_test,y_pred))

print("Maximum RFC Accuracy is: ", max(rfc_accuracy), ' located at n-est: ', (rfc_accuracy.index(max(rfc_accuracy))+1)*10)
rfc_ss_accuracy=[]

n_est = [10,20,30,40,50,60,70,80,90,100]

for i in range(len(n_est)):

    rfc = RandomForestClassifier(n_estimators=n_est[i])

    rfc.fit(X_train_std, y_train)

    y_pred =rfc.predict(X_test_std)

    rfc_ss_accuracy.append(accuracy_score(y_test,y_pred))

print("Maximum RFC Accuracy is: ", max(rfc_ss_accuracy), ' located at n-est: ', (rfc_ss_accuracy.index(max(rfc_ss_accuracy))+1)*10)
plt.plot(n_est,rfc_accuracy)

plt.plot(n_est,rfc_ss_accuracy)

plt.title('Random Forest Calibration')

plt.xlabel('n-estimator')

plt.ylabel('accuracy score')

plt.legend(['rfc', 'rfc w/scaling'])
from sklearn.svm import SVC

from sklearn.model_selection import GridSearchCV
param_grid = {'C': [0.1,1, 10, 100, 1000], 

              'gamma': [1,0.1,0.01,0.001,0.0001], 

              'kernel': ['rbf']} 
grid = GridSearchCV(SVC(),param_grid,refit=True,verbose=3)

grid.fit(X_train,y_train)

grid.best_params_

grid.best_estimator_

y_grid = grid.predict(X_test)

print('Classification Report')

print(classification_report(y_test, y_grid))

print('Confusion Matrix')

print(confusion_matrix(y_test, y_grid))

print('Accuracy Score')

print(accuracy_score(y_test, y_grid))
from sklearn.model_selection import StratifiedKFold, cross_val_score
# fix the indices for X and y

X_kf = X.reset_index(drop=True)

y_kf = y.reset_index(drop=True)
kfold = StratifiedKFold(n_splits=10)
knn_kf = KNeighborsClassifier(n_neighbors=20)

logreg_kf = LogisticRegression()
scores = []

scores_lr = []

for train_index, test_index in kfold.split(X_kf,y_kf):

    xtrain, xtest = X_kf.iloc[train_index], X_kf.iloc[test_index]

    ytrain, ytest = y_kf[train_index], y_kf[test_index]

    # knn

    knn_kf.fit(xtrain, ytrain)

    score = knn_kf.score(xtest, ytest)

    scores.append(score)

    # logreg

    logreg_kf.fit(xtrain, ytrain)

    score_lr = logreg_kf.score(xtest, ytest)

    scores_lr.append(score_lr)

print(scores, scores_lr)
print('Mean accuracy for knn using KFold is {:0.2f}%'.format(np.mean(scores)*100))

print('Mean accuracy for logreg using KFold is {:0.2f}%'.format(np.mean(scores_lr)*100))
# Try with scaling

# fix the indices for X and y with standardscaling

sc_cv=StandardScaler()

sc_cv.fit(X)

X_kf_std = sc_cv.transform(X)

y_kf_std = y.reset_index(drop=True)

knn_kf_ss = KNeighborsClassifier(n_neighbors=20)

logreg_kf_ss = LogisticRegression()
# try another way

acc_scores = []

ss_scores_lr = []

for train_index, test_index in kfold.split(X_kf_std,y_kf_std):

    xtrain, xtest = X_kf_std[train_index], X_kf_std[test_index]

    ytrain, ytest = y_kf_std[train_index], y_kf_std[test_index]

    #knn

    knn_kf_ss.fit(xtrain, ytrain)

    score = knn_kf_ss.score(xtest, ytest)

    acc_scores.append(score)

    #logreg

    logreg_kf_ss.fit(xtrain, ytrain)

    score_lr = logreg_kf_ss.score(xtest, ytest)

    ss_scores_lr.append(score_lr)

print(acc_scores, ss_scores_lr)
print('Mean accuracy for knn using KFold is {:0.2f}%'.format(np.mean(acc_scores)*100))

print('Mean accuracy for logreg using KFold is {:0.2f}%'.format(np.mean(ss_scores_lr)*100))
def get_df(score_list, algo_name, scaled):

    temp_df = pd.DataFrame(score_list, columns=['Accuracy'])

    temp_df['Algorithm'] = algo_name

    temp_df['Scaled'] = scaled

    return temp_df
df_01 = get_df(scores, algo_name='KNN', scaled=False)

df_02 = get_df(scores_lr, algo_name='LogReg', scaled=False)

df_03 = get_df(acc_scores, algo_name='KNN', scaled=True)

df_04 = get_df(ss_scores_lr, algo_name='LogReg', scaled=True)

frames = [df_01, df_02, df_03, df_04]

all_scores = pd.concat(frames)

all_scores.reset_index(drop=True,inplace=True)

all_scores.head()
sns.barplot(x='Algorithm', y='Accuracy', hue='Scaled', data=all_scores)