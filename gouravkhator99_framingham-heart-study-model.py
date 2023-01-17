import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.model_selection import train_test_split
data = pd.read_csv('../input/framingham-heart-study-dataset/framingham.csv')
data.head()
data.info()
data.describe()
# education feature is not required as its not predicting the Ten Year CHD

# target is Ten Year CHD (0 or 1)

data.drop('education', axis=1, inplace=True)
# renaming TenYearCHD to CHD

data.rename(columns={"TenYearCHD": "CHD"}, inplace=True)
data.head()
X_train, X_test, y_train, y_test = train_test_split(data.iloc[:,:-1], data.iloc[:,-1], test_size=0.14, random_state=0)



train_data = pd.concat([X_train, y_train], axis=1)

test_data = pd.concat([X_test, y_test], axis=1)
# age vs CHD

plt.figure(figsize=(10,10))

sns.swarmplot(x='CHD', y='age', data=train_data)
plt.figure(figsize=(10,10))

sns.violinplot(x='CHD', y='age', data=train_data)
# age vs CHD for smokers or non-smoker

plt.figure(figsize=(10,10))

sns.swarmplot(x='CHD', y='age', data=train_data, hue='currentSmoker')
plt.figure(figsize=(10,10))

sns.violinplot(x='CHD', y='age', data=train_data, hue='currentSmoker', split=True)
# male and female countplot

sns.countplot(x=train_data['male'])
# male and female having disease or not

sns.countplot(x=train_data['male'], hue=train_data['CHD'])
train_data.iloc[:,:5]
# To understand correlation between some features, pairplot is used

plt.figure(figsize=(20,15))

sns.pairplot(train_data.loc[:,'totChol': 'glucose'])
plt.figure(figsize=(15,15))

sns.heatmap(train_data.corr(), annot=True, linewidths=0.1)
# dropping features which are highly correlated

features_to_drop = ['currentSmoker', 'diaBP']



train_data.drop(features_to_drop, axis=1, inplace=True)
train_data.head()
missing_values_count = train_data.isnull().sum()

missing_values_count = missing_values_count[missing_values_count > 0]

missing_values_percent = (missing_values_count * 100) / (train_data.shape[0])



print(max(missing_values_percent))
print(missing_values_count)
from sklearn.impute import SimpleImputer

imputer = SimpleImputer(strategy='most_frequent')
new_train_data = pd.DataFrame(imputer.fit_transform(train_data))

new_train_data.columns = train_data.columns

new_train_data.index = train_data.index
train_data.isnull().sum()
new_train_data.isnull().sum()
new_train_data.head()
train_data = new_train_data.copy()
fig, ax = plt.subplots(figsize=(10,10), nrows=3, ncols=4)

ax = ax.flatten()



i = 0

for k,v in train_data.items():

    sns.boxplot(y=v, ax=ax[i])

    i+=1

    if i==12:

        break

plt.tight_layout(pad=1.25, h_pad=0.8, w_pad=0.8)
# Outliers handling

print('Number of training examples to be deleted for outliers removal is ',len(train_data[train_data['sysBP'] > 220]) + len(train_data[train_data['BMI'] > 43]) + len(

    train_data[train_data['heartRate'] > 125]) + len(train_data[train_data['glucose'] > 200]) + len(

    train_data[train_data['totChol'] > 450]))
# deleting outliers



train_data = train_data[~(train_data['sysBP'] > 220)]

train_data = train_data[~(train_data['BMI'] > 43)]

train_data = train_data[~(train_data['heartRate'] > 125)]

train_data = train_data[~(train_data['glucose'] > 200)]

train_data = train_data[~(train_data['totChol'] > 450)]

print(train_data.shape)
# fig, ax = plt.subplots(figsize=(10,10), nrows=3, ncols=4)

# ax = ax.flatten()



# i = 0

# for k,v in train_data.items():

#     sns.distplot(v, ax=ax[i])

#     i+=1

#     if i==12:

#         break

# plt.tight_layout(pad=1.25, h_pad=0.8, w_pad=0.8)
# Standardise some features

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

cols_to_standardise = ['age','totChol','sysBP','BMI', 'heartRate', 'glucose', 'cigsPerDay']

train_data[cols_to_standardise] = scaler.fit_transform(train_data[cols_to_standardise])
train_data.head()
# dropping unwanted features as done in train data

test_data.drop(features_to_drop, axis=1, inplace=True)



# imputing missing values if any

imputer = SimpleImputer(strategy='most_frequent')

new_test_data = pd.DataFrame(imputer.fit_transform(test_data))

new_test_data.columns = test_data.columns

new_test_data.index = test_data.index



test_data = new_test_data.copy()
# Standardising features

scaler = StandardScaler()

test_data[cols_to_standardise] = scaler.fit_transform(test_data[cols_to_standardise])
test_data.head()
from sklearn.linear_model import LogisticRegression

log_reg = LogisticRegression()
X_train = train_data.loc[:,train_data.columns != 'CHD']

y_train = train_data.loc[:,'CHD']

X_test = test_data.loc[:, test_data.columns !='CHD']

y_test = test_data.loc[:, 'CHD']
log_reg.fit(X_train, y_train)

y_pred_log = log_reg.predict(X_test)
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
log_reg_accuracy = accuracy_score(y_pred_log, y_test) * 100

print('Accuracy Score for logistic regression is %f'%log_reg_accuracy)
log_train_score = log_reg.score(X_train, y_train) * 100

print('Train score for Logistic Regression is %f'%log_train_score)
print('Difference between train and test score for Logistic Regression is %f'%(log_train_score - log_reg_accuracy))
confusion_matrix(y_pred_log, y_test)
print(classification_report(y_pred_log, y_test))
from sklearn.tree import DecisionTreeClassifier
dt_clf = DecisionTreeClassifier(min_samples_split=40, random_state=0) 

# that fraction of samples(if float) or that many number(if int) of samples is atleast present in the node 

# before splitting, then only split that node



# for min_samples_split as 180 I got a better accuracy and train score and difference was less

# but f1 score was very bad for positive class

# and setting min_samples_split as 40, we got good results for all metrics
dt_clf.fit(X_train, y_train)

y_pred_dt = dt_clf.predict(X_test)
dt_accuracy = accuracy_score(y_pred_dt, y_test)*100

print('Accuracy score for Decision tree is %f'%dt_accuracy)
dt_train_score = dt_clf.score(X_train, y_train)*100

print('Train score for Decision tree is %f'%dt_train_score)
print('Difference between train and test scores for Decision tree is : %f'%(dt_train_score - dt_accuracy))
confusion_matrix(y_pred_dt, y_test)
print(classification_report(y_pred_dt, y_test))
# Exporting the tree in text format

from sklearn.tree import export_text

dt_text_format = export_text(dt_clf, feature_names=list(train_data.columns[:12]))

print('Decision tree in text format : \n%s'%dt_text_format)
from sklearn.ensemble import RandomForestClassifier
rf_clf = RandomForestClassifier(n_estimators = 150,min_samples_split=10,random_state=0)
rf_clf.fit(X_train, y_train)

y_pred_rf = rf_clf.predict(X_test)
rf_accuracy = accuracy_score(y_pred_rf, y_test)*100

print('Accuracy score for Random Forest is %f'%rf_accuracy)
rf_train_score = rf_clf.score(X_train, y_train)*100

print('Train score for Random Forest is %f'%rf_train_score)
print('Difference between train and test scores for Random Forest is : %f'%(rf_train_score - rf_accuracy))
confusion_matrix(y_pred_rf, y_test)
print(classification_report(y_pred_rf, y_test))
pd.DataFrame(y_pred_dt).to_csv('testPredictions.csv', index=False)