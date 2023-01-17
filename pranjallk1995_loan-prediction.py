# importing librabries



import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

from sklearn import preprocessing

from sklearn.metrics import confusion_matrix
# Loading data



df_train = pd.read_csv('../input/loan-prediction-problem-dataset/train_u6lujuX_CVtuZ9i.csv')

df_test = pd.read_csv('../input/loan-prediction-problem-dataset/test_Y3wMUE5_7gLdaTN.csv')
# inspecting loaded data



df_train.head()
# inspecting size of the dataset

df_train.shape
# making missing value data frame



missing_df_train = df_train.isnull()

missing_df_test = df_test.isnull()
# finding all missing values in training data



for column in missing_df_train.columns.values.tolist():

    print(column)

    print(missing_df_train[column].value_counts())

    print()
# finding all missing values in testing data



for column in missing_df_test.columns.values.tolist():

    print(column)

    print(missing_df_test[column].value_counts())

    print()
# filling missing values in training data



df_train['Gender'].fillna(df_train['Gender'].mode()[0], inplace = True)

df_train['Married'].fillna(df_train['Married'].mode()[0], inplace = True)

df_train['Dependents'].fillna(df_train['Dependents'].mode()[0], inplace = True)

df_train['Self_Employed'].fillna(df_train['Self_Employed'].mode()[0], inplace = True)

df_train['LoanAmount'].fillna(df_train['LoanAmount'].median(), inplace = True)

df_train['Loan_Amount_Term'].fillna(df_train['Loan_Amount_Term'].mode()[0], inplace = True)

df_train['Credit_History'].fillna(df_train['Credit_History'].mode()[0], inplace = True)
# filling missing values in test data



df_test['Gender'].fillna(df_test['Gender'].mode()[0], inplace = True)

df_test['Dependents'].fillna(df_test['Dependents'].mode()[0], inplace = True)

df_test['Self_Employed'].fillna(df_test['Self_Employed'].mode()[0], inplace = True)

df_test['LoanAmount'].fillna(df_test['LoanAmount'].median(), inplace = True)

df_test['Loan_Amount_Term'].fillna(df_test['Loan_Amount_Term'].mode()[0], inplace = True)

df_test['Credit_History'].fillna(df_test['Credit_History'].mode()[0], inplace = True)
sns.countplot(x = df_train['Gender'], hue = df_train['Married'])
sns.countplot(x = df_train['Gender'], hue = df_train['Loan_Status'])
sns.scatterplot(x = df_train['LoanAmount'], y = df_train['ApplicantIncome'], hue = df_train['Loan_Status'])
sns.countplot(x = df_train['Married'], hue = df_train['Loan_Status'])
sns.countplot(x = df_train['Property_Area'], hue = df_train['Loan_Status'])
sns.countplot(x = df_train['Self_Employed'],hue = df_train['Loan_Status'])
sns.countplot(x = df_train['Education'], hue = df_train['Loan_Status'])
sns.heatmap(df_train.corr(),annot=True)
sns.countplot(x = df_train['Loan_Amount_Term'], hue = df_train['Loan_Status']).set_yscale('log')
fig, ax = plt.subplots(2, 4, figsize = (20, 10))

sns.countplot(x = df_train['Gender'], ax = ax[0][0])

sns.countplot(x = df_train['Married'], ax = ax[0][1])

sns.countplot(x = df_train['Dependents'], ax = ax[0][2])

sns.countplot(x = df_train['Education'], ax = ax[0][3])

sns.countplot(x = df_train['Self_Employed'], ax = ax[1][0])

sns.countplot(x = df_train['Loan_Amount_Term'], ax = ax[1][1]).set_yscale('log')

sns.countplot(x = df_train['Credit_History'], ax = ax[1][2])

sns.countplot(x = df_train['Property_Area'], ax = ax[1][3])
status_counts = df_train['Loan_Status'].value_counts()

print("Percentage of Y: ", end = ' ')

print(status_counts[0] / (status_counts[0] + status_counts[1]) * 100)

print("Percentage of N: ", end = ' ')

print(status_counts[1] / (status_counts[0] + status_counts[1]) * 100)
df_train = df_train[df_train['ApplicantIncome'] <= 20000]

df_train = df_train[df_train['LoanAmount'] <= 400]

df_train.shape
sns.scatterplot(x = df_train['LoanAmount'], y = df_train['ApplicantIncome'], hue = df_train['Loan_Status'])
# Dropping coapplicant income data



df_train = df_train.drop(labels = ['CoapplicantIncome'], axis = 1)

df_test = df_test.drop(labels = ['CoapplicantIncome'], axis = 1)
# Label encoding training data since all these features have some priority



label_encoder = preprocessing.LabelEncoder()

df_train['Loan_Status'] = label_encoder.fit_transform(df_train['Loan_Status'])

df_train['Married'] = label_encoder.fit_transform(df_train['Married'])

df_train['Education'] = label_encoder.fit_transform(df_train['Education'])

df_train['Dependents'] = label_encoder.fit_transform(df_train['Dependents'])

df_train['Self_Employed'] = label_encoder.fit_transform(df_train['Self_Employed'])

df_train['Property_Area'] = label_encoder.fit_transform(df_train['Property_Area'])

df_train['Gender'] = label_encoder.fit_transform(df_train['Gender'])
# Label encoding test data since all these features have some priority



label_encoder = preprocessing.LabelEncoder()

df_test['Married'] = label_encoder.fit_transform(df_test['Married'])

df_test['Education'] = label_encoder.fit_transform(df_test['Education'])

df_test['Dependents'] = label_encoder.fit_transform(df_test['Dependents'])

df_test['Self_Employed'] = label_encoder.fit_transform(df_test['Self_Employed'])

df_test['Property_Area'] = label_encoder.fit_transform(df_test['Property_Area'])

df_test['Gender'] = label_encoder.fit_transform(df_test['Gender'])
# inspecting data



df_train.head()
# Dropping unique identifier



df_train.drop('Loan_ID', axis = 1, inplace = True)

test_ID = df_test['Loan_ID']

df_test.drop('Loan_ID', axis = 1, inplace = True)
# Importing SMOTE



from imblearn.over_sampling import SMOTE

sm = SMOTE(sampling_strategy = 1 ,k_neighbors = 15)
x_ch = df_train['Credit_History']

X_ch = df_train.drop('Credit_History', axis = 1)

X_ch, x_ch = sm.fit_sample(X_ch, x_ch)
X_ch.shape
df_train = pd.concat([X_ch, x_ch], axis = 1)
# Splitting target variable



y = df_train['Loan_Status']

X = df_train.drop('Loan_Status', axis = 1)
# Inspecting data



X.tail()
# Normalizing data



normalized_X = preprocessing.normalize(X)

normalized_X_test = preprocessing.normalize(df_test)
# importing model



from sklearn.tree import DecisionTreeClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier

from sklearn.svm import SVC
# fitting Logistic Regression



model = LogisticRegression()

model.fit(normalized_X, y)

ypred_LR = model.predict(normalized_X_test)
# fitting Decision Tree



tree = DecisionTreeClassifier()

tree.fit(normalized_X, y)

ypred_TR = tree.predict(normalized_X_test)
# fitting Random Forest



forest = RandomForestClassifier()

forest.fit(normalized_X, y)

ypred_RF = forest.predict(normalized_X_test)
# fitting Support Vector Classifier



svc = SVC()

svc.fit(normalized_X, y)

ypred_SV = svc.predict(normalized_X_test)
result = []

for value in ypred_LR:

    if value == 1:

        result.append('Y')

    else:

        result.append('N')

df = pd.concat([test_ID, pd.DataFrame(result)], axis = 1)

df.rename(columns = {0:'Loan_Status'}, inplace = True)

df.to_csv('Final_result_LR.csv', index = False)
result = []

for value in ypred_TR:

    if value == 1:

        result.append('Y')

    else:

        result.append('N')

df = pd.concat([test_ID, pd.DataFrame(result)], axis = 1)

df.rename(columns = {0:'Loan_Status'}, inplace = True)

df.to_csv('Final_result_TR.csv', index = False)
result = []

for value in ypred_RF:

    if value == 1:

        result.append('Y')

    else:

        result.append('N')

df = pd.concat([test_ID, pd.DataFrame(result)], axis = 1)

df.rename(columns = {0:'Loan_Status'}, inplace = True)

df.to_csv('Final_result_RF.csv', index = False)
result = []

for value in ypred_SV:

    if value == 1:

        result.append('Y')

    else:

        result.append('N')

df = pd.concat([test_ID, pd.DataFrame(result)], axis = 1)

df.rename(columns = {0:'Loan_Status'}, inplace = True)

df.to_csv('Final_result_SV.csv', index = False)