import sys

import os 

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt



from sklearn.mixture import GaussianMixture 

from sklearn.model_selection import cross_val_predict

from sklearn.model_selection import train_test_split

from sklearn.metrics import matthews_corrcoef

from sklearn.metrics import roc_auc_score

from sklearn.metrics import roc_curve

from sklearn.metrics import accuracy_score

from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier 

from sklearn.datasets import make_classification

DIR='.../input/'
train = pd.read_csv("../input/TrainData.csv", ";", low_memory=False, encoding = 'latin-1')

train.head()
test = pd.read_csv("../input/TestData.csv", ";", low_memory=False, encoding = 'latin-1')

test.head()
train.columns
train.rename(columns={'Art der Anstellung': 'employment_type', 'Ausfall Kredit': 'credit_failure', 

                      'Anruf-ID': 'call_id', 'Anzahl der Ansprachen': 'n_speeches',

                      'Tage seit letzter Kampagne': 'days_last_cmp', 

                      'Anzahl Kontakte letzte Kampagne': 'n_contacts_last_cmp', 'Stammnummer': 'customer_id',

                      'Zielvariable': 'goal_variable', 'Tag': 'day', 'Monat': 'month', 'Dauer': 'duration',

                      'Alter': 'age', 'Geschlecht': 'gender', 'Familienstand': 'marital_status', 

                      'Schulabschluß': 'education', 'Kontostand': 'balance', 'Haus': 'house', 'Kredit': 'credit',

                      'Kontaktart': 'contact_type', 'Ergebnis letzte Kampagne': 'result_last_cmp'},

                      inplace=True)

train.head()
train['employment_type'] = train['employment_type'].str.replace(' ', '_')

train['employment_type'] = train['employment_type'].str.lower()

train['education'] = train['education'].str.lower()

train['contact_type'] = train['contact_type'].str.lower()

train['result_last_cmp'] = train['result_last_cmp'].str.replace(' ', '_')

train['result_last_cmp'] = train['result_last_cmp'].str.lower()
train['month'] = train['month'].map({'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'may': 5, 'jun': 6, 'jul': 7, 

                                     'aug': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12})
# Data analysis. Analyse the values contained in each column.

print('Types of marital status are listed as follows:\n')

print(train.marital_status.value_counts())

print('\nTypes of education are listed as follows:\n')

print(train.education.value_counts())

print('\nTypes of employment are listed as follows:\n')

print(train.employment_type.value_counts())

print('\nTypes of contracts are listed as follows:\n')

print(train.contact_type.value_counts())

print('\nTypes of results of the last campaign are listed as follows:\n')

print(train.result_last_cmp.value_counts())
# Transform the employment type column into binary features

for val in ['arbeiter', 'management', 'technicher_beruf', 'verwaltung', 

            'dienstleistung', 'rentner', 'selbstständig',

           'gründer', 'arbeitslos', 'hausfrau', 'student', 'unbekannt']:

    train["employment_type_"+ val] = (train["employment_type"]==val).astype(int)

train = train.drop('employment_type', 1)
# Transform the gender column into binary features

train["gender_female"] = (train["gender"] == 'w').astype(int)

train = train.drop('gender', 1)
# Transform the age column into binary features

for val in ['18_29', '30_39', '40_49', '50_59', '60_69', '70_79', '80_100']:

    train["age_group_"+ val] = ((train["age"] >= int(int(val.replace('_', ''))/100)) & 

                                (train["age"] <= int(val.replace('_', ''))%100)).astype(int)

train = train.drop('age', 1)
# Transform the education column into binary features

for val in ['abitur', 'studium', 'real-/hauptschule', 'unbekannt']:

    train["education_"+ val] = (train["education"] == val).astype(int)

train = train.drop('education', 1)
# Transform the marital staus column into binary features

for val in ['verheiratet', 'single', 'geschieden', 'unbekannt']:

    train['marital_status_'+ val] = (train['marital_status'] == val).astype(int)

train = train.drop('marital_status', 1)
# Transform the credit failure column into binary features

train['credit_failure_positive'] = (train['credit_failure'] == 'ja').astype(int)

train = train.drop('credit_failure', 1)
# Transform the house ownership column into binary features

train['house_ownership'] = (train['house'] == 'ja').astype(int)

train = train.drop('house', 1)
# Transform the credit presence column into binary features

train['credit_present'] = (train['credit'] == 'ja').astype(int)

train = train.drop('credit', 1)
# Transform the contact type column into binary features

for val in ['handy', 'festnetz', 'unbekannt']:

    train['contact_type_'+ val] = (train['contact_type'] == val).astype(int)

train = train.drop('contact_type', 1)
# Transform the campaign result column into binary features

for val in ['unbekannt', 'kein_erfolg', 'sontiges', 'erfolg']:

    train['result_last_cmp_'+ val] = (train['result_last_cmp'] == val).astype(int)

train = train.drop('result_last_cmp', 1)
# Transform the month column into binary features for the each year quarter

for val in [1, 2, 3, 4]:

    train['quarter_'+ str(val)] = ((train['month'] <= val*3) & (train['month'] > (val-1)*3)).astype(int)

train = train.drop('month', 1)
# Transform the house day column into binary features for beginning and end of the month

train['beginning_of_month'] = (train['day'] <= 15).astype(int)

train = train.drop('day', 1)
train['goal_variable'] = train['goal_variable'].map({'nein': 0, 'ja': 1})

train = train.fillna({'days_last_cmp': -1})
X = train.drop(['customer_id', 'call_id'], 1)
import seaborn as sns

f, ax = plt.subplots(figsize=(25, 20))

corr = X.corr()

sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool), cmap=sns.diverging_palette(250, 10, as_cmap=True), 

            square=True, ax=ax)
X = X.drop(['marital_status_unbekannt', 'age_group_80_100', 'employment_type_technicher_beruf', 

                'result_last_cmp_sontiges', 'employment_type_selbstständig'], 1)
y = X['goal_variable']

X = X.drop('goal_variable', 1)
X.head()
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.8)
logreg = LogisticRegression()

logreg.fit(x_train, y_train)

y_pred_log = logreg.predict(x_test)

y_pred_prob_log = logreg.predict_proba(x_test)[:,1]

score_lr_auc = roc_auc_score(y_test, y_pred_prob_log)

score_lr_mcc = matthews_corrcoef(y_pred_log, y_test)

print('Logistic Regression accuracy score: %f'% logreg.fit(x_train, y_train).score(x_test, y_test))

print('Logistic Regression roc_auc score: %f'%score_lr_auc)

print('Logistic Regression mcc score: %f'%score_lr_mcc)
dectree = DecisionTreeClassifier()

dectree.fit(x_train, y_train)

y_pred_dct = logreg.predict(x_test)

y_pred_prob_dct = dectree.predict_proba(x_test)[:,1]

score_dct_auc = roc_auc_score(y_test, y_pred_prob_dct)

score_dct_mcc = matthews_corrcoef(y_pred_dct, y_test)

print('Decision Tree accuracy score: %f'% dectree.fit(x_train, y_train).score(x_test, y_test))

print('Decision Tree roc_auc score: %f'%score_dct_auc)

print('Decision Tree mcc score: %f'%score_dct_mcc)
clf = RandomForestClassifier(n_estimators=100, random_state=0)

clf.fit(x_train, y_train)

y_pred_rf = clf.predict(x_test)

y_pred_prob_rf = clf.predict_proba(x_test)[:,1]

score_rf_auc = roc_auc_score(y_test, y_pred_prob_rf)

score_rf_mcc = matthews_corrcoef(y_pred_rf, y_test)

print('Random Forest accuracy score: %f'% clf.fit(x_train, y_train).score(x_test, y_test))

print('Random Forest roc_auc score: %f'%score_rf_auc)

print('Random Forest mcc score: %f'%score_rf_mcc)
importances = pd.DataFrame({'feature':x_train.columns,

                            'importance': np.round(clf.feature_importances_,3)})

importances = importances.sort_values('importance', ascending = False).set_index('feature')

importances
# Drop insignificant columns fromt X.

X.drop(['employment_type_unbekannt','employment_type_hausfrau', 'credit_failure_positive', 

        'employment_type_gründer', 'employment_type_arbeitslos', 'age_group_60_69', 'age_group_70_79', 

        'contact_type_festnetz', 'education_unbekannt', 'employment_type_student', 

        'employment_type_dienstleistung', 'result_last_cmp_kein_erfolg', 'education_real-/hauptschule',

        'marital_status_geschieden', 'employment_type_rentner'], 1)

X.head()
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.8)

clf = RandomForestClassifier(n_estimators=100, random_state=0)

clf.fit(x_train, y_train)

y_pred = clf.predict(x_test)

y_pred_prob = clf.predict_proba(x_test)[:,1]

score_rf_auc = roc_auc_score(y_test, y_pred_prob)

score_rf_mcc = matthews_corrcoef(y_pred, y_test)

print('Random Forest accuracy score: %f'% clf.fit(x_train, y_train).score(x_test, y_test))

print('Random Forest roc_auc score: %f'%score_rf_auc)

print('Random Forest mcc score: %f'%score_rf_mcc)
from sklearn.model_selection import cross_val_score

scores = cross_val_score(clf, x_train, y_train, cv=10, scoring = "accuracy")

print("Scores:", scores)

print("Mean:", scores.mean())

print("Standard Deviation:", scores.std())
test.rename(columns={'Art der Anstellung': 'employment_type', 'Ausfall Kredit': 'credit_failure', 

                     'Anruf-ID': 'call_id','Anzahl der Ansprachen': 'n_speeches',

                     'Tage seit letzter Kampagne': 'days_last_cmp', 'Zielvariable':'goal_variable',

                     'Anzahl Kontakte letzte Kampagne': 'n_contacts_last_cmp', 'Stammnummer': 'customer_id',

                     'Tag': 'day', 'Monat': 'month', 'Dauer': 'duration', 'Alter': 'age', 'Geschlecht': 'gender', 

                     'Familienstand': 'marital_status', 'Schulabschluß': 'education', 'Kontostand': 'balance', 

                     'Haus': 'house', 'Kredit': 'credit','Kontaktart': 'contact_type', 

                     'Ergebnis letzte Kampagne': 'result_last_cmp'},

                     inplace=True)





test['employment_type'] = test['employment_type'].str.replace(' ', '_')

test['employment_type'] = test['employment_type'].str.lower()

test['education'] = test['education'].str.lower()

test['contact_type'] = test['contact_type'].str.lower()

test['result_last_cmp'] = test['result_last_cmp'].str.replace(' ', '_')

test['result_last_cmp'] = test['result_last_cmp'].str.lower()

test['month'] = test['month'].map({'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'may': 5, 'jun': 6, 'jul': 7, 

                                   'aug': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12})





# Transform the employment type column into binary features

for val in ['arbeiter', 'management', 'technicher_beruf', 'verwaltung', 

            'dienstleistung', 'rentner', 'selbstständig',

           'gründer', 'arbeitslos', 'hausfrau', 'student', 'unbekannt']:

    test["employment_type_"+ val] = (test["employment_type"]==val).astype(int)

test = test.drop('employment_type', 1)





# Transform the gender column into binary features

test["gender_female"] = (test["gender"] == 'w').astype(int)

test = test.drop('gender', 1)





# Transform the age column into binary features

for val in ['18_29', '30_39', '40_49', '50_59', '60_69', '70_79', '80_100']:

    test["age_group_"+ val] = ((test["age"] >= int(int(val.replace('_', ''))/100)) & 

                              (test["age"] <= int(val.replace('_', ''))%100)).astype(int)

test = test.drop('age', 1)



    

# Transform the education column into binary features





for val in ['abitur', 'studium', 'real-/hauptschule', 'unbekannt']:

    test["education_"+ val] = (test["education"] == val).astype(int)

test = test.drop('education', 1)





# Transform the marital staus column into binary features

for val in ['verheiratet', 'single', 'geschieden', 'unbekannt']:

    test['marital_status_'+ val] = (test['marital_status'] == val).astype(int)

test = test.drop('marital_status', 1)





# Transform the credit failure column into binary features

test['credit_failure_positive'] = (test['credit_failure'] == 'ja').astype(int)

test = test.drop('credit_failure', 1)





# Transform the credit failure column into binary features

test['credit_present'] = (test['credit'] == 'ja').astype(int)

test = test.drop('credit', 1)





# Transform the house ownership column into binary features

test['house_ownership'] = (test['house'] == 'ja').astype(int)

test = test.drop('house', 1)





# Transform the contact type column into binary features

for val in ['handy', 'festnetz', 'unbekannt']:

    test['contact_type_'+ val] = (test['contact_type'] == val).astype(int)

test = test.drop('contact_type', 1)





# Transform the campaign result column into binary features

for val in ['unbekannt', 'kein_erfolg', 'sontiges', 'erfolg']:

    test['result_last_cmp_'+ val] = (test['result_last_cmp'] == val).astype(int)

test = test.drop('result_last_cmp', 1)





# Transform the month column into binary features for the each year quarter

for val in [1, 2, 3, 4]:

    test['quarter_'+ str(val)] = ((test['month'] <= val*3) & (test['month'] > (val-1)*3)).astype(int)

test = test.drop('month', 1)



customer_id = test['customer_id']



# Transform the house day column into binary features for beginning and end of the month

test['beginning_of_month'] = (test['day'] <= 15).astype(int)

test = test.drop('day', 1)

test = test.fillna({'days_last_cmp': -1})
prediction = test.drop(['customer_id', 'call_id', 'marital_status_unbekannt', 'age_group_80_100', 

                     'employment_type_technicher_beruf', 'result_last_cmp_sontiges', 

                     'employment_type_selbstständig', 'goal_variable'], 1)

prediction.head()
score = clf.predict_proba(prediction)[:,1]

goal = clf.predict(prediction)

combi = {'ID': customer_id,

         'Expected': score

        }

solution = pd.DataFrame.from_dict(combi)

solution.head()
solution.to_csv('../Solution.csv', sep=';', encoding='utf-8')