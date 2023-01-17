import numpy as np

import pandas as pd

from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn import metrics

from sklearn.preprocessing import LabelEncoder
bank_data = pd.read_csv('../input/bank-marketing-campaign-eda-prediction/bank-additional-full.csv', sep = ';')

bank_data
bank_data.shape
bank_data.info()
print(bank_data['job'].unique())

print(bank_data['marital'].unique())

print(bank_data['education'].unique())

print(bank_data['default'].unique())

print(bank_data['month'].unique())

print(bank_data['day_of_week'].unique())

print(bank_data['contact'].unique())

print(bank_data['housing'].unique())

print(bank_data['loan'].unique())

print(bank_data['poutcome'].unique())

print(bank_data['previous'].unique())

print(bank_data['pdays'].unique())

print(bank_data['y'].unique())
bank_data.rename(columns = {'default':'Default Credit', 'housing': 'Housing Loan', 'loan': 'Personal Loan', 'contact':'Mode of Contact',

                            'month':'Contact Month', 'day_of_week':'Contact Day', 'duration':'Contact Duration','pdays':'Passed_Days',

                           ' previous':'Performed_Contacts', 'poutcome':'Outcome_of_Campaign','nr.employed':'No._Of_Employees','y':'Term Deposit'}, inplace=True)

bank_data.head(10)
bank_data.replace({'unknown': 'NaN'}, inplace=True)

bank_data.replace(['nonexistent', 'failure', 'success'], [0, 1, 2], inplace=True)

bank_data.replace(['mar','apr','may','jun','jul','aug','sep','oct','nov','dec'], [3, 4, 5, 6, 7, 8, 9, 10, 11, 12], inplace=True)

bank_data
bank_data.replace({'basic.4y':'Basic+4yr', 'high.school':'High_School', 'basic.6y':'Basic+6yr', 'basic.9y':'Basic+9yr',

                   'professional.course':'Professional Course','university.degree':'University Degree','illiterate': 'Illiterate'}, inplace=True)

bank_data
sns.heatmap(bank_data.corr())
bank_data.isnull().sum()
default = bank_data['Default Credit'].values

house_loan = bank_data['Housing Loan'].values

personal_loan = bank_data['Personal Loan'].values

term_deposit = bank_data['Term Deposit'].values
bank_data_df = pd.DataFrame(bank_data)
bank_data_df.drop(['Mode of Contact','emp.var.rate','euribor3m','cons.price.idx','cons.conf.idx'], axis=1, inplace=True)
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

labelencoder=LabelEncoder()

default = labelencoder.fit_transform(bank_data_df['Default Credit'])

house_loan = labelencoder.fit_transform(bank_data_df['Housing Loan'])

personal_loan = labelencoder.fit_transform(bank_data_df['Personal Loan'])

term_deposit = labelencoder.fit_transform(bank_data_df['Term Deposit'])

onehotencoder=OneHotEncoder()
print(default)

print(house_loan)

print(personal_loan)

print(term_deposit)
print(bank_data.head(25))
bank_data_df['Default Credit'] = default

bank_data_df['Housing Loan'] = house_loan 

bank_data_df['Personal Loan'] = personal_loan

bank_data_df['Term Deposit'] = term_deposit
bank_data_df.head(25)
bank_data_df.info()
plt.figure(figsize=(20,15))

sns.countplot(y='job', data=bank_data_df)
plt.figure(figsize=(12,6))

sns.countplot('marital', data=bank_data)
plt.figure(figsize=(15,10))

sns.countplot(y='education', data=bank_data)
plt.figure(figsize=(12,6))

sns.countplot(y='Contact Month', data=bank_data)
plt.figure(figsize=(12,6))

sns.countplot('Contact Day', data=bank_data)
plt.figure(figsize=(12,6))

sns.countplot('Outcome_of_Campaign', data=bank_data)
plt.figure(figsize=(10,6))

sns.barplot('Personal Loan', 'Contact Duration', hue='Default Credit', data=bank_data)
plt.figure(figsize=(10,6))

sns.barplot('Personal Loan', 'Contact Duration', hue='Term Deposit', data=bank_data)
plt.figure(figsize=(12,6))

sns.barplot('Housing Loan', 'age', hue='Term Deposit', palette="Paired", data=bank_data)
plt.figure(figsize=(20,15))

sns.barplot('age', 'No._Of_Employees', hue='job', palette="Pastel1", data=bank_data)
plt.figure(figsize=(20,15))

sns.boxplot('age', 'Contact Duration', hue='Term Deposit', palette="Pastel1", data=bank_data)
plt.figure(figsize=(10, 5))

sns.lineplot('Housing Loan', 'age', hue='Outcome_of_Campaign', size='Personal Loan', style='Term Deposit', data=bank_data_df, palette="dark", markers=True )
from matplotlib.colors import LogNorm

plt.figure(figsize=(20,15))

sns.lineplot('campaign', 'Contact Duration', hue='Passed_Days',size='Outcome_of_Campaign', style='Term Deposit',hue_norm=LogNorm(), data=bank_data_df)
plt.figure(figsize=(20,15))

sns.barplot('Contact Duration', 'marital', hue='Term Deposit', palette="prism", data=bank_data)
plt.figure(figsize=(20,15))

sns.barplot('Contact Duration', 'job', hue='Outcome_of_Campaign', palette="colorblind", data=bank_data)
plt.figure(figsize=(20,15))

sns.catplot('age', 'job', hue='Housing Loan', palette="PiYG", data=bank_data, kind='box')
#bank_data_df.info()
#import os
#%pwd
#bank_data.to_csv('Bank Data.csv')
# month = bank_data_df.sort_values(by='Contact Month', ascending=True, inplace=True)

# day = bank_data_df.sort_values(by='Contact Day', ascending=True, inplace=True)

duration = bank_data_df.sort_values(by='Contact Duration', ascending=True, inplace=True)

#bank_data_df
bank_data_df.groupby(['age','job','Default Credit', 'marital', 'Housing Loan','Personal Loan','Contact Month','Contact Duration','Passed_Days','campaign','Outcome_of_Campaign','No._Of_Employees','Term Deposit']).age.count()

bank_data_df
bank_data_df.replace({'NaN': 0}, inplace=True)
bank_data_df.drop(['Contact Day'], axis=1, inplace=True)
bank_data_df['job']= pd.to_numeric(bank_data_df['job'], errors='coerce')

bank_data_df['marital']= pd.to_numeric(bank_data_df['marital'], errors='coerce')

bank_data_df['education']= pd.to_numeric(bank_data_df['education'], errors='coerce')

bank_data_df['No._Of_Employees'] = bank_data_df['No._Of_Employees'].astype(int)

bank_data_df['Outcome_of_Campaign']= pd.to_numeric(bank_data_df['Outcome_of_Campaign'], errors='coerce')
bank_data_df['marital'].unique
bank_data_df.head()
bank_data_df.info()

#bank_data_df.describe()
bank_data_df['education'].isnull().sum()
bank_data_df.drop(['job', 'marital', 'education'], axis=1, inplace=True)
bank_data_df.info()
bank_data_df.describe()
x = bank_data_df.drop(['Term Deposit'], axis=1)

y = bank_data_df['Term Deposit'].values
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.15, random_state=32)
print(X_train.shape)

print(X_test.shape)

print(y_train.shape)

print(y_test.shape)
from sklearn.linear_model import LogisticRegression

lo_reg = LogisticRegression()

lo_reg.fit(X_train,y_train)
pred = lo_reg.predict(X_test)

pred
pred.shape
score=lo_reg.score(X_test, y_test)

print(score)
from sklearn.metrics import confusion_matrix

cm=confusion_matrix(y_test, pred)

# print(cm)

plt.figure(figsize=(20,9))

sns.heatmap(cm, annot=True, fmt=".3f", linewidths=.5, square=True, cmap='nipy_spectral')

plt.ylabel('Actual Label')

plt.xlabel('Predicted Label')

all_sample_title=f"Accuracy Score: {score}"

plt.title(all_sample_title, size=15)
print("Precision:",metrics.precision_score(y_test, pred))

print("Recall:",metrics.recall_score(y_test, pred))
bank_data.replace({'unknown': 'NaN'}, inplace=True)

bank_data.replace(['nonexistent', 'failure', 'success'], [0, 1, 2], inplace=True)

bank_data.replace(['mar','apr','may','jun','jul','aug','sep','oct','nov','dec'], [3, 4, 5, 6, 7, 8, 9, 10, 11, 12], inplace=True)

bank_data.replace(['married', 'single', 'divorced','unknown'], [2, 1, 3, 0], inplace=True)

bank_data
bank_data_df = pd.DataFrame(bank_data)
sns.heatmap(bank_data.corr())
plt.figure(figsize=(20,15))

sns.countplot(y='job', data=bank_data_df)
plt.figure(figsize=(12,6))

sns.countplot('marital', data=bank_data)
plt.figure(figsize=(15,10))

sns.countplot(y='education', data=bank_data)
job = bank_data_df['job'].values

marital = bank_data_df['marital'].values

edu = bank_data_df['education'].values

default = bank_data['Default Credit'].values

house_loan = bank_data['Housing Loan'].values

personal_loan = bank_data['Personal Loan'].values

term_deposit = bank_data['Term Deposit'].values
bank_data_df.drop(['Mode of Contact','emp.var.rate','euribor3m','cons.price.idx','cons.conf.idx'], axis=1, inplace=True)
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

labelencoder=LabelEncoder()

default = labelencoder.fit_transform(bank_data_df['Default Credit'])

job = labelencoder.fit_transform(bank_data_df['job'])

edu = labelencoder.fit_transform(bank_data_df['education'])

house_loan = labelencoder.fit_transform(bank_data_df['Housing Loan'])

personal_loan = labelencoder.fit_transform(bank_data_df['Personal Loan'])

term_deposit = labelencoder.fit_transform(bank_data_df['Term Deposit'])

onehotencoder=OneHotEncoder()
print(default)

print(job)

print(edu)

print(house_loan)

print(personal_loan)

print(term_deposit)
bank_data_df['Default Credit'] = default

bank_data_df['job'] = job

bank_data_df['education'] = edu

bank_data_df['Housing Loan'] = house_loan 

bank_data_df['Personal Loan'] = personal_loan

bank_data_df['Term Deposit'] = term_deposit
bank_data_df.info()
duration = bank_data_df.sort_values(by='Contact Duration', ascending=True, inplace=True)

bank_data_df.groupby(['age','job','Default Credit', 'marital', 'Housing Loan','Personal Loan','Contact Month','Contact Duration','Passed_Days','campaign','Outcome_of_Campaign','No._Of_Employees','Term Deposit']).age.count()

bank_data_df
bank_data_df.replace({'NaN': 0}, inplace=True)

bank_data_df.drop(['Contact Day'], axis=1, inplace=True)
bank_data_df['No._Of_Employees'] = bank_data_df['No._Of_Employees'].astype(int)

bank_data_df['Outcome_of_Campaign']= pd.to_numeric(bank_data_df['Outcome_of_Campaign'], errors='coerce')
bank_data_df.describe()
x = bank_data_df.drop(['Term Deposit'], axis=1)

y = bank_data_df['Term Deposit'].values
from sklearn.model_selection import train_test_split

from sklearn.model_selection import cross_val_score

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.15, random_state=32)
from sklearn.tree import DecisionTreeClassifier

DT = DecisionTreeClassifier(criterion="entropy", max_depth=3)

DT = DT.fit(X_train, y_train)
pred = DT.predict(X_test)

pred
DT.predict_proba(X_test) 
print("Predicticed Subscription for term deposit: ", pred)  
from sklearn import metrics

print("Accuracy:",metrics.accuracy_score(y_test, pred))
from sklearn import tree

plt.figure(figsize=(20,15))

tree.plot_tree(DT)
plt.scatter(y_test, pred, color = 'red')

plt.plot(y_test, pred, color = 'blue')
from sklearn.metrics import confusion_matrix

cm=confusion_matrix(y_test, pred)

# print(cm)

plt.figure(figsize=(20,9))

sns.heatmap(cm, annot=True, fmt=".3f", linewidths=.5, square=True, cmap='jet')

plt.ylabel('Actual Label')

plt.xlabel('Predicted Label')

all_sample_title=f"Accuracy Score: ",metrics.accuracy_score(y_test, pred)

plt.title(all_sample_title, size=15)
x = bank_data_df.drop(['Term Deposit'], axis=1)

y = bank_data_df['Term Deposit'].values
from sklearn.model_selection import train_test_split

from sklearn.model_selection import cross_val_score

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.10, random_state=32)
from sklearn.ensemble import RandomForestClassifier

rdm_forest = RandomForestClassifier(n_estimators = 100, random_state = 0)

rdm_forest = rdm_forest.fit(X_train,y_train)
pred = rdm_forest.predict(X_test)

pred
train_rf_predictions = rdm_forest.predict(X_train)

train_rf_probs =rdm_forest.predict_proba(X_train)



rf_predictions = rdm_forest.predict(X_test)

rf_probs = rdm_forest.predict_proba(X_test)
print(train_rf_predictions)

print(train_rf_probs)

print(rf_predictions)

print(rf_probs)
from sklearn.metrics import precision_score, recall_score, roc_auc_score, roc_curve

print(f'Accuracy ROC AUC  Score: {roc_auc_score(y_test, pred)}')

print(f'Precision ROC Score: {precision_score(y_test, pred)}')

print(f'Recall ROC Score: {recall_score(y_test, pred)}')

print(f'Recall ROC Score: {roc_curve(y_test, pred)}')
from sklearn.metrics import confusion_matrix

cm=confusion_matrix(y_test, pred)

# print(cm)

plt.figure(figsize=(20,9))

sns.heatmap(cm, annot=True, fmt=".3f", linewidths=.5, square=True, cmap='nipy_spectral')

plt.ylabel('Actual Label')

plt.xlabel('Predicted Label')

all_sample_title=f"Accuracy Score: {roc_auc_score(y_test, pred)}"

plt.title(all_sample_title, size=15)