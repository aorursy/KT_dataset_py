# Importing basic libraries

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import warnings

warnings.filterwarnings("ignore")
# Reading the data

df = pd.read_csv('../input/framingham-heart-study-dataset/framingham.csv')



# To display first 5 rows in the dataframe 

df.head()
# To display last 5 rows in the dataframe 

df.tail()
# To know the data type and null values if any

df.info()
# Percentage of null values in each column

(df.isnull().sum()/df.shape[0])*100
df.education.fillna(0,inplace=True)
df.cigsPerDay.fillna(df.cigsPerDay.where(df.currentSmoker==1).median(),inplace=True)
df.BPMeds.fillna(0,inplace=True)
df['totChol'].fillna(df.totChol.median(),inplace=True)
df['BMI'].fillna(df.BMI.median(),inplace=True)
df['heartRate'].fillna(df['heartRate'].where(df['currentSmoker']==1).median(),inplace=True)
df['glucose'].fillna(df['glucose'].where(df['diabetes']==0).median(),inplace=True)
# Checking if there are any misisng values:

(df.isnull().sum()/df.shape[0])*100
# Five point summary of clean data

df.describe().T
# To know the data type of column are affected

df.info()
# Names of columns of dataframe

df.columns
# List of columns names with contineous values

col = ['age','totChol', 'sysBP', 'diaBP', 'BMI', 'heartRate', 'glucose']
# To find outliers

for i in col:

    sns.boxplot(df[i])

    plt.show()
# Distribution of the contineous values column

for i in col:

    sns.distplot(df[i])

    plt.show()
# Making a copy of the clean dataframe

df1 = df.copy()



# To remove outliers

for i in col:

    q1 = df1[i].quantile(q=0.25)

    q2 = df1[i].quantile()

    q3 = df1[i].quantile(q=0.75)

    iqr = q3-q1

    ul = q3+1.5*iqr

    ll = q1-1.5*iqr



    df1 = df1[(df1[i]<ul ) & (df1[i]>ll)] 
# Checking distribution of the contineous values column after outliers treatment



for i in col:

    sns.distplot(df1[i])

    plt.show()
print('There were {} rows before outlier treatment.'.format(df.shape[0]))

print('There are {} rows after outlier treatment.'.format(df1.shape[0]))

print('After outlier treatment number of rows lost are {}.'.format(df.shape[0] - df1.shape[0]))
# Ratio of CHD=1 and CHD=0

df1['TenYearCHD'].value_counts(normalize=True)
# Ploting the ratio

sns.countplot(df['TenYearCHD'],)

plt.show()
# Correlation plot using heatmap

cor = df.corr()

plt.figure(figsize=(15,9))

sns.heatmap(cor,annot=True)

plt.show()
X = df1.drop(['TenYearCHD'], axis=1)

y = df1['TenYearCHD']
import statsmodels.api as sm



X_const = sm.add_constant(X)
model = sm.Logit(y, X)

result = model.fit()

result.summary()
## Backward elimination to drop insignificant variables one by one



cols = list(X.columns)

p = []

while len(cols)>1:

    X = X[cols]

    model= sm.Logit(y, X).fit().pvalues

    p =pd.Series(model.values[1:],index=X.columns[1:])

    pmax = max(p)

    pid = p.idxmax()

    if pmax>0.05:

        cols.remove(pid)

        print('Variable removed:', pid, pmax)

    else:

        break

cols   
# Keeping the significant variables

X_sig = df1[['male', 'age', 'education', 'cigsPerDay', 'prevalentStroke', 'prevalentHyp', 'sysBP', 'diaBP', 'BMI',

     'heartRate', 'glucose']]
X_sig.describe().T
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler() 

X_sig = scaler.fit_transform(X_sig)
# using significant feature

model = sm.Logit(y, X_sig)

result = model.fit()

result.summary()
from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(X_sig, y, test_size=0.30, random_state=1)
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import confusion_matrix, accuracy_score, classification_report



logreg = LogisticRegression(solver='liblinear', fit_intercept=True) 



logreg.fit(X_train, y_train)



y_prob_train = logreg.predict_proba(X_train)[:,1]

y_pred_train = logreg.predict (X_train)



print('Confusion Matrix - Train: ', '\n', confusion_matrix(y_train, y_pred_train))

print('\nOverall accuracy - Train: ', accuracy_score(y_train, y_pred_train))





y_prob = logreg.predict_proba(X_test)[:,1]

y_pred = logreg.predict (X_test)



print('\nConfusion Matrix - Test: ','\n', confusion_matrix(y_test, y_pred))

print('\nOverall accuracy - Test: ','\n', accuracy_score(y_test, y_pred))

print('\nClassification report for test:\n',classification_report(y_test,y_pred))
from imblearn.over_sampling import SMOTE



smote = SMOTE(random_state=1)
X_train_sm, y_train_sm = smote.fit_resample(X_train,y_train)
logreg_sm = LogisticRegression(solver='liblinear', fit_intercept=True) 



logreg_sm.fit(X_train_sm, y_train_sm)



y_prob_train = logreg_sm.predict_proba(X_train_sm)[:,1]

y_pred_train = logreg_sm.predict (X_train_sm)



print('Confusion Matrix - Train: ', '\n', confusion_matrix(y_train_sm, y_pred_train))

print('\nOverall accuracy - Train: ', accuracy_score(y_train_sm, y_pred_train))





y_prob = logreg_sm.predict_proba(X_test)[:,1]

y_pred = logreg_sm.predict (X_test)



print('\nConfusion Matrix - Test: ','\n', confusion_matrix(y_test, y_pred))

print('\nOverall accuracy - Test: ', accuracy_score(y_test, y_pred))

print('\nClassification report for test:\n',classification_report(y_test,y_pred))
from sklearn.tree import DecisionTreeClassifier



dt = DecisionTreeClassifier()

dt.fit(X_train_sm, y_train_sm)



y_pred_train = dt.predict(X_train_sm)

y_pred = dt.predict(X_test)

y_prob = dt.predict_proba(X_test)



print('Classification report for test:\n',classification_report(y_test,y_pred))
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
dt = DecisionTreeClassifier()



params = {'max_depth' : [2,3,4,5,6,7,8],

        'min_samples_split': [2,3,4,5,6,7,8,9,10],

        'min_samples_leaf': [1,2,3,4,5,6,7,8,9,10]}



gsearch = GridSearchCV(dt, param_grid=params, cv=3)



gsearch.fit(X,y)



gsearch.best_params_
# DT using best parameters

dt = DecisionTreeClassifier(**gsearch.best_params_)



dt.fit(X_train_sm, y_train_sm)



y_pred_train = dt.predict(X_train_sm)

y_prob_train = dt.predict_proba(X_train_sm)[:,1]



y_pred = dt.predict(X_test)

y_prob = dt.predict_proba(X_test)[:,1]



print('\nClassification report for test:\n',classification_report(y_test,y_pred))
from scipy.stats import randint as sp_randint



dt = DecisionTreeClassifier(random_state=1)



params = {'max_depth' : sp_randint(2,10),

        'min_samples_split': sp_randint(2,50),

        'min_samples_leaf': sp_randint(1,20),

         'criterion':['gini', 'entropy']}



rand_search = RandomizedSearchCV(dt, param_distributions=params, cv=3, 

                                 random_state=1)



rand_search.fit(X, y)

print(rand_search.best_params_)
# DT using best parameters

dt = DecisionTreeClassifier(**rand_search.best_params_)



dt.fit(X_train_sm, y_train_sm)



y_pred_train = dt.predict(X_train_sm)

y_prob_train = dt.predict_proba(X_train_sm)[:,1]



y_pred = dt.predict(X_test)

y_prob = dt.predict_proba(X_test)[:,1]



print('Classification report for test:\n',classification_report(y_test,y_pred))
from sklearn.ensemble import RandomForestClassifier



rfc = RandomForestClassifier(n_estimators=10, random_state=1)



rfc.fit(X_train_sm, y_train_sm)



y_pred_train = rfc.predict(X_train_sm)

y_prob_train = rfc.predict_proba(X_train_sm)[:,1]



y_pred = rfc.predict(X_test)

y_prob = rfc.predict_proba(X_test)[:,1]



print('Classification report for test:\n',classification_report(y_test,y_pred))

from scipy.stats import randint as sp_randint



rfc = RandomForestClassifier(random_state=1)



params = {'n_estimators': sp_randint(5,25),

    'criterion': ['gini', 'entropy'],

    'max_depth': sp_randint(2, 10),

    'min_samples_split': sp_randint(2,20),

    'min_samples_leaf': sp_randint(1, 20),

    'max_features': sp_randint(2,15)}



rand_search_rfc = RandomizedSearchCV(rfc, param_distributions=params,

                                 cv=3, random_state=1)



rand_search_rfc.fit(X, y)

print(rand_search_rfc.best_params_)
# RFC using best parameters

rfc = RandomForestClassifier(**rand_search_rfc.best_params_)



rfc.fit(X_train_sm, y_train_sm)



y_pred_train = rfc.predict(X_train_sm)

y_prob_train = rfc.predict_proba(X_train_sm)[:,1]



y_pred = rfc.predict(X_test)

y_prob = rfc.predict_proba(X_test)[:,1]



print('Classification report for test:\n',classification_report(y_test,y_pred))
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier()
knn.fit(X_train_sm, y_train_sm)



y_pred_train = knn.predict(X_train_sm)

y_prob_train = knn.predict_proba(X_train_sm)[:,1]



y_pred = knn.predict(X_test)

y_prob = knn.predict_proba(X_test)[:,1]



print('Classification report for test:\n',classification_report(y_test,y_pred))
knn = KNeighborsClassifier()



params = {'n_neighbors': sp_randint(1,25),

        'p': sp_randint(1,5)}



rand_search_knn = RandomizedSearchCV(knn, param_distributions=params,

                                 cv=3, random_state=1)

rand_search_knn.fit(X, y)

print(rand_search.best_params_)
# KNN using best parameters



knn = KNeighborsClassifier(**rand_search_knn.best_params_)



knn.fit(X_train_sm, y_train_sm)



y_pred_train = knn.predict(X_train_sm)

y_prob_train = knn.predict_proba(X_train_sm)[:,1]



y_pred = knn.predict(X_test)

y_prob = knn.predict_proba(X_test)[:,1]



print('Classification report for test:\n',classification_report(y_test,y_pred))
from sklearn.ensemble import VotingClassifier

from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(solver='liblinear')

dt = DecisionTreeClassifier(**rand_search.best_params_)

rfc = RandomForestClassifier(**rand_search_rfc.best_params_)

knn = KNeighborsClassifier(**rand_search_knn.best_params_)
# Without using weights

clf = VotingClassifier(estimators=[('lr',lr), ('dt',dt),('rfc',rfc), ('knn',knn)], 

                       voting='soft')

clf.fit(X_train_sm, y_train_sm)

y_pred_train = clf.predict(X_train_sm)

y_prob_train = clf.predict_proba(X_train_sm)[:,1]



y_pred = clf.predict(X_test)

y_prob = clf.predict_proba(X_test)[:,1]
print('Classification report for test:\n',classification_report(y_test,y_pred))
# Using weights

clf = VotingClassifier(estimators=[('lr',lr),('dt',dt) ,('rfc',rfc), ('knn',knn)], 

                       voting='soft', weights=[4,1,3,2])

clf.fit(X_train_sm, y_train_sm)

y_pred_train = clf.predict(X_train_sm)

y_prob_train = clf.predict_proba(X_train_sm)[:,1]



y_pred = clf.predict(X_test)

y_prob = clf.predict_proba(X_test)[:,1]
print('Classification report for test:\n',classification_report(y_test,y_pred))