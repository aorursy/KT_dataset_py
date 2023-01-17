import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

sns.set()

%matplotlib inline
df = pd.read_csv('/kaggle/input/cardiovascular-disease-dataset/cardio_train.csv', sep=';')

df.head()
df.shape
# Check for missing values

df.isnull().sum()
df['cardio'].unique()
# Convert age from days to years

df['age'] =  df['age'] / 365
df.head()
# Rename columns to make features more clearly understood

df.rename(columns={'ap_hi': 'systolic', 'ap_lo': 'diastolic', 'gluc': 'glucose', 'alco': 'alcohol', 'cardio': 'cardiovascular disease'}, inplace=True)
df.head()
sns.lmplot(x='weight', y='height', hue='gender', data=df, fit_reg=False, height=6)

plt.show()
sns.countplot(x='gender', data=df, hue='cardiovascular disease')

plt.show()



# Not much of a difference between females (1) and males (2) and the chance of getting cardiovascular disease.
df.describe()
df_train = df.drop('id', axis=1)
df_train.head()
df_train.info()
# 24 Duplicated entries

df_train.duplicated().sum()
df_train[df_train.duplicated()]
df_train.drop_duplicates(inplace=True)
df_train.count()
df_train.isnull().sum()
df_train.head()
sns.countplot(x='gender', hue='cardiovascular disease', data=df_train)

plt.show()
sns.countplot(x='cholesterol', hue='cardiovascular disease', data=df_train)

plt.show()

# There appears to be a correlation between higher cholesterol levels and cardiovascular disease

# chloesterol levels: 1 = normal, 2 = above normal, 3 = well above normal
sns.countplot(x='glucose', hue='cardiovascular disease', data=df_train)

plt.show()

# There appears to be another correlation between higher glucose levels and cardiovascular disease

# glucose levels: 1 = normal, 2 = above normal, 3 = well above normal
sns.countplot(x='active', hue='cardiovascular disease', data=df_train)

plt.show()
sns.countplot(x='smoke', hue='cardiovascular disease', data=df_train)

plt.show()
sns.countplot(x='alcohol', hue='cardiovascular disease', data=df_train)

plt.show()
sns.distplot(df_train['weight'], kde=False)

plt.show()
df_train['weight'].sort_values().head()
sns.distplot(df_train['height'], kde=False)

plt.show()
df_train['height'].max()
df_train['height'].sort_values().head()
df_train['BMI'] = df_train['weight'] / df_train['height'] / df_train['height'] * 10000

df_train['pulse pressure'] = df_train['systolic'] - df_train['diastolic']
df_train.head()

# Quick look at the dataframe to make sure these new features have been added
plt.figure(figsize=(8,4))

sns.distplot(df_train['BMI'], bins=50, kde=False)

plt.show()
df_train[df_train['BMI'] > 100].head(10)
df_train[(df_train['pulse pressure'] >= 60 ) & (df_train['cholesterol'] == 3)].head(15)
plt.figure(figsize=(8,4))

sns.distplot(df_train['height'], kde=False)

plt.show()
# Splitting data into training and testing datasets

X = df_train.drop(['weight', 'height', 'cardiovascular disease'], axis=1)

y = df_train['cardiovascular disease']



from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=17)
#Feature Scaling

from sklearn.preprocessing import StandardScaler

sc_X = StandardScaler()

X_train = sc_X.fit_transform(X_train)

X_test = sc_X.transform(X_test)
len(X_train)
len(y_train)
# Random Forest

from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier(n_estimators=100)

rfc.fit(X_train, y_train)
y_pred_rfc = rfc.predict(X_test)
# Random Forest Model Evaluation

from sklearn.metrics import confusion_matrix, classification_report

print(confusion_matrix(y_test, y_pred_rfc))

print(classification_report(y_test, y_pred_rfc))
rfc.score(X_test, y_test)
#Applying k-Fold Cross Validation

from sklearn.model_selection import cross_val_score

accuracies_rfc = cross_val_score(estimator=rfc, X=X_train, y=y_train, cv=10)
accuracies_rfc
accuracies_rfc.mean()
accuracies_rfc.std()
# SVM

from sklearn.svm import SVC

svc = SVC(gamma='auto')

svc.fit(X_train, y_train)
y_pred_svc = svc.predict(X_test)
# SVM Model Evaluation

print(confusion_matrix(y_test, y_pred_svc))

print(classification_report(y_test, y_pred_svc))
svc.score(X_test, y_test)
#Applying k-Fold Cross Validation

accuracies_svc = cross_val_score(estimator=svc, X=X_train, y=y_train, cv=10, n_jobs=4)
accuracies_svc
accuracies_svc.mean()
accuracies_svc.std()
# KNN

from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=100)

knn.fit(X_train, y_train)
y_pred_knn = knn.predict(X_test)
# KNN Model Evaluation

print(confusion_matrix(y_test, y_pred_knn))

print(classification_report(y_test, y_pred_knn))
knn.score(X_test, y_test)
#Applying k-Fold Cross Validation

accuracies_knn = cross_val_score(estimator=knn, X=X_train, y=y_train, cv=10)
accuracies_knn
accuracies_knn.mean()
accuracies_knn.std()
# Naive Bayes

from sklearn.naive_bayes import GaussianNB

nbc = GaussianNB()

nbc.fit(X_train, y_train)
y_pred_nbc = nbc.predict(X_test)
# Naive Bayes Model Evaluation

print(confusion_matrix(y_test, y_pred_nbc))

print(classification_report(y_test, y_pred_nbc))
nbc.score(X_test, y_test)
#Applying k-Fold Cross Validation

accuracies_nbc = cross_val_score(estimator=nbc, X=X_train, y=y_train, cv=10)
accuracies_nbc
accuracies_nbc.mean()
accuracies_nbc.std()
# XGBoost Model

from xgboost import XGBClassifier

xgb = XGBClassifier(learning_rate=0.02, n_estimators=600)

xgb.fit(X_train, y_train)
y_pred_xgb = xgb.predict(X_test)
# XGBoost Model Evaluation

print(confusion_matrix(y_test, y_pred_xgb))

print(classification_report(y_test, y_pred_xgb))
xgb.score(X_test, y_test)
#Applying k-Fold Cross Validation

accuracies_xgb = cross_val_score(estimator=xgb, X=X_train, y=y_train, cv=10)
accuracies_xgb
accuracies_xgb.mean()
accuracies_xgb.std()
#Applying Grid Search to find the best model and best parameters (XGBoost)

#from sklearn.model_selection import GridSearchCV



#define set of parameters that will be investigated by Grid Search

#parameters = {

#            'learning_rate': [0.01, 0.02, 0.05, 0.1],

#            'n_estimators': [100, 200, 300, 500],

#            'min_child_weight': [1, 5, 10],

#            'gamma': [0.5, 1, 1.5, 2, 5],

#            'subsample': [0.6, 0.8, 1.0],

#            'colsample_bytree': [0.6, 0.8, 1.0],

#            'max_depth': [3, 4, 5]

#            }
#grid_search = GridSearchCV(estimator=xgb,

#                          param_grid = parameters,

#                          scoring = 'accuracy',

#                          cv = 10,

#                          n_jobs = -1)
#grid_search = grid_search.fit(X_train, y_train)
#Applying Grid Search to find the best model and best parameters (SVM)

#from sklearn.model_selection import GridSearchCV



#define set of parameters that will be investigated by Grid Search

#parameters = {'C': [1, 10, 100, 1000], 'kernel': ['rbf']}
#grid_search = GridSearchCV(estimator=svc,

#                          param_grid = parameters,

#                          scoring = 'accuracy',

#                          cv = 10,

#                          n_jobs = -1)
#grid_search = grid_search.fit(X_train, y_train)
model = ['Random Forest', 'SVM', 'KNN', 'Naive Bayes', 'XGBoost']

scores = [accuracies_rfc.mean(),accuracies_svc.mean(),accuracies_knn.mean(),accuracies_nbc.mean(),accuracies_xgb.mean()]



summary = pd.DataFrame(data=scores, index=model, columns=['Mean Accuracy'])

summary.sort_values(by='Mean Accuracy', ascending=False)