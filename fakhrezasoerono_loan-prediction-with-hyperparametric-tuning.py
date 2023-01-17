# 00. Packages

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.impute import SimpleImputer

from sklearn.pipeline import Pipeline

from sklearn.model_selection import train_test_split

from sklearn.model_selection import GridSearchCV

from sklearn.neighbors import KNeighborsClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import classification_report

from sklearn.preprocessing import scale
# A.01. Import Data

f_train = '../input/loan-prediction-problem-dataset/train_u6lujuX_CVtuZ9i.csv'

df_train = pd.read_csv(f_train, index_col=0)
# A.02.01. Replace 'strange' values and convert dtype

df_train['Dependents'] = df_train['Dependents'].replace('3+', '3')

df_train['Dependents'] = pd.to_numeric(df_train['Dependents'], errors='coerce')
# A.02.02. Features (Categorical & Numerical)

X = df_train.drop(columns='Loan_Status') # feature

categorical = []

numerical = []

for feature in list(X.columns):

	if X[feature].dtypes == object:

		categorical.append(X[feature])

	else:

		numerical.append(X[feature])

categorical = pd.concat(categorical, axis=1)

numerical = pd.concat(numerical, axis=1)

col_name_cat = categorical.columns # for later use

col_name_num = numerical.columns # for later use
# A.02.03 Fill na values (imputing)

imp_num = SimpleImputer(strategy='mean')

imp_cat = SimpleImputer(strategy='most_frequent')

imp_num.fit(numerical)

imp_cat.fit(categorical)

numerical = pd.DataFrame(imp_num.transform(numerical), index=df_train.index,

columns = col_name_num)

categorical = pd.DataFrame(imp_cat.transform(categorical), index=df_train.index)

print(categorical.isnull().sum())

print(numerical.isnull().sum())
# A.02.04 Encode Categorical (with pd.get_dummies())

categorical = pd.get_dummies(categorical, drop_first=True)

categorical.columns = ['Male', 'Married', 'Not Graduate', 'Self-Employed', 'SemiUrban', 'Urban']
data_train = pd.concat([categorical, numerical, df_train['Loan_Status']], axis=1) # for Part C
X = pd.concat([categorical, numerical], axis=1) # features after imputing and encoding

X = scale(X) # scaled features

y = df_train['Loan_Status'] # target

print(X.shape)

print(y.shape)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=99) # train_test_split
# A.03.01 K-NN Method

param_knn = {'n_neighbors' : np.arange(1,50)}

knn = KNeighborsClassifier()

knn_cv = GridSearchCV(knn, param_knn, cv=5)

knn_cv.fit(X_train, y_train)

print('K-NN Best Parameter & Score:')

print(knn_cv.best_params_)

print(knn_cv.best_score_)

y_pred = knn_cv.predict(X_test)

knn_score = knn_cv.score(X_test, y_test)

print('\nK-NN Accuracy Score: ', knn_score)

print('Classification Report: \n')

print(classification_report(y_test, y_pred), '\n')
# A.03.02 Logistic Regression

param_log = {'C' : np.logspace(-4, 4, 20)}

logreg = LogisticRegression()

logreg_cv = GridSearchCV(logreg, param_log, cv=5)

logreg_cv.fit(X_train, y_train)

print('Log Reg Best Parameter & Score:')

print(logreg_cv.best_params_)

print(logreg_cv.best_score_)

y_pred = logreg_cv.predict(X_test)

logreg_score = logreg_cv.score(X_test, y_test)

print('\nLog Reg Accuracy Score: ', logreg_score)

print('Classification Report: \n')

print(classification_report(y_test, y_pred), '\n')
# A.03.03 Random Forest Method

param_rf = {'n_estimators' : np.arange(100,550,100)}

rf = RandomForestClassifier()

rf_cv = GridSearchCV(rf, param_rf, cv=5)

rf_cv.fit(X_train, y_train)

print('Random Forest Best Parameter & Score:')

print(rf_cv.best_params_)

print(rf_cv.best_score_)

y_pred = rf_cv.predict(X_test)

rf_score = rf_cv.score(X_test, y_test)

print('\n Random Forest Accuracy Score: ', rf_score)

print('Classification Report: \n')

print(classification_report(y_test, y_pred), '\n')
f_new = '../input/loan-prediction-problem-dataset/test_Y3wMUE5_7gLdaTN.csv'

df_new = pd.read_csv(f_new, index_col=0)
# B.02.01. Replace 'strange' values and convert dtype

df_new['Dependents'] = df_new['Dependents'].replace('3+', '3')

df_new['Dependents'] = pd.to_numeric(df_new['Dependents'], errors='coerce')

# B.02.02. Features (Categorical & Numerical)

X = df_new # feature

categorical = []

numerical = []

for feature in list(X.columns):

	if X[feature].dtypes == object:

		categorical.append(X[feature])

	else:

		numerical.append(X[feature])

categorical = pd.concat(categorical, axis=1)

numerical = pd.concat(numerical, axis=1)

# B.02.03 Fill na values (imputing)

imp_num.fit(numerical)

imp_cat.fit(categorical)

numerical = pd.DataFrame(imp_num.transform(numerical), index=df_new.index,

columns = col_name_num)

categorical = pd.DataFrame(imp_cat.transform(categorical), index=df_new.index)

# B.02.04 Encode Categorical (with pd.get_dummies())

categorical = pd.get_dummies(categorical, drop_first=True)

categorical.columns = ['Male', 'Married', 'Not Graduate', 'Self-Employed', 'SemiUrban', 'Urban']
# B.03. Predictions

X = pd.concat([categorical, numerical], axis=1) # features after imputing and encoding

X = scale(X) # scale features

if (knn_score > logreg_score) & (knn_score > rf_score):

	y_pred = knn_cv.predict(X)

	print('Chosen Method : K-NN')

	print('Accuracy Score: ', knn_score)

elif (logreg_score > knn_score) & (logreg_score > rf_score):

	y_pred = logreg_cv.predict(X)

	print('Chosen Method : LogisticRegression')

	print('Accuracy Score: ', logreg_score)

else:

	y_pred = rf_cv.predict(X)

	print('Chosen Method : Random Forest')

	print('Accuracy Score: ', rf_score)
# B.04. Save Results

df_result = df_new

df_result['Loan_Status'] = y_pred

df_result.to_csv('test result.csv')
rejected = df_result[df_result['Loan_Status'] == 'N']

accepted = df_result[df_result['Loan_Status'] == 'Y']

print(rejected['Credit_History'].value_counts())

print(accepted['Credit_History'].value_counts())
data_train.Loan_Status = data_train.Loan_Status.replace({'Y' : 1, 'N' : 0})

corr = data_train.corr()

_ = sns.heatmap(corr)

plt.show()