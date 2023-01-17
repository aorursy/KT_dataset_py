# Data Wrangling 

import numpy as np

import pandas as pd 



# Data Visualisation 

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline 



# Machine Learning Tools 

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC, LinearSVC

from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.linear_model import Perceptron

from sklearn.linear_model import SGDClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import accuracy_score, mean_squared_error
train_data = pd.read_csv('../input/train.txt')

test_data = pd.read_csv('../input/test.txt')

combine = [train_data, test_data]
train_data.head()
print(test_data.head())

test_sub = test_data.copy()

test_sub.head()
train_data['Loan_Status'] = train_data['Loan_Status'].map( {'Y' : 1, 'N' : 0} )

train_data.head()
train_data.describe(percentiles = [.31, .32 ])

# train_data.describe(percentiles = [.15, .2 ])
train_data.describe(include = 'O')
# Dropping Loan_ID

train_data.drop('Loan_ID', axis = 1, inplace = True)

test_data.drop('Loan_ID', axis = 1, inplace = True)

combine = [train_data, test_data]

train_data.head()
print(train_data.isnull().sum())

print('-' * 50)

print(test_data.isnull().sum())
# Plotting a correlation heatmap

sns.heatmap(train_data.corr(), annot = True)
print(train_data[['Married', 'ApplicantIncome']].groupby('Married', as_index = False).median())

print('\n')

print(train_data[['Married', 'ApplicantIncome']].groupby('Married', as_index = False).mean())
print(train_data[['Married', 'CoapplicantIncome']].groupby('Married', as_index = False).median())
train_data[['Married', 'Gender']].groupby('Gender').count()
train_data[train_data['Married'].isnull()]

train_data.set_value(435, 'Married', 'Yes')

train_data.set_value(104, 'Married', 'Yes')

train_data.set_value(228, 'Married', 'No')
train_data[['Loan_Status', 'Gender']].groupby(['Gender']).mean()
train_data.columns
train_data.Married.unique()
train_data[['Married', 'Gender', 'Loan_Status']].groupby(['Married'], as_index = False).mean()
sns.violinplot(x = 'Education', y = 'Loan_Status', data = train_data)
train_data[['Property_Area', 'Loan_Status']].groupby(['Property_Area'], as_index = False).mean().sort_values(['Loan_Status'], ascending = False)
axes = plt.gca()

axes.set_xlim([0,25000])

sns.swarmplot(x = 'ApplicantIncome', y = 'Property_Area', data = train_data, hue = 'Loan_Status')
print(train_data.Dependents.unique())

train_data[['Dependents', 'Loan_Status']].groupby(['Dependents'], as_index = False).mean()
train_data['Dependents'].dtype
# Converting Married to nominal variables. 

combine = [train_data, test_data]

for dataset in combine: 

    dataset['Married'] = dataset['Married'].map({'Yes' : 1, 'No' : 0})

train_data.head()
train_data[['Married', 'Gender', 'Education']].groupby(['Education', 'Married']).count()
combine = [train_data, test_data]
train_data.isnull().sum()

train_data[['Married', 'Dependents', 'Gender']].groupby(['Dependents', 'Gender'], as_index = False).mean() 
sns.swarmplot(x = 'Dependents', y = 'ApplicantIncome', hue = 'Married', data = train_data)
train_data[['Married', 'Dependents', 'ApplicantIncome']].groupby('Dependents').median()
train_data[['Education', 'Dependents', 'Married']].groupby(['Dependents', 'Education'], as_index = False).count()
train_data[['Self_Employed', 'Dependents']].groupby('Dependents', as_index = False).count().pivot(columns = 'Self_Employed', index = 'Dependents')
train_data[['Property_Area', 'Dependents']].groupby(['Dependents'], as_index = False).count()
train_data[['Gender', 'Dependents', 'ApplicantIncome']].groupby(['Dependents', 'Gender'], as_index = False).median()
train_data[train_data['Dependents'].isnull()]
train_data.groupby('Dependents')['Dependents'].count()
# Individual Manual Changes 

train_data.set_value(102, 'Dependents','3+')

train_data.set_value(332, 'Dependents', '0')

train_data.set_value(335, 'Dependents', '2')

train_data.set_value(597, 'Dependents', '0')
for dataset in combine:   

    # For males

    dataset.loc[(dataset['Dependents'].isnull()) & (dataset['ApplicantIncome'] < 3683.5) & (dataset['Gender'] == 'Male'), 'Dependents'] = '0'

    dataset.loc[(dataset['Dependents'].isnull()) & (dataset['ApplicantIncome'] > 3683.5) & (dataset['ApplicantIncome'] <= 3931.5) & (dataset['Gender'] == 'Male'),'Dependents'] = '1'

    dataset.loc[(dataset['Dependents'].isnull()) & (dataset['ApplicantIncome'] > 3931.5) & (dataset['ApplicantIncome'] <= 4200.0) & (dataset['Gender'] == 'Male'),'Dependents'] = '2'

    dataset.loc[(dataset['Dependents'].isnull()) & (dataset['ApplicantIncome'] > 4200.0) & (dataset['Gender'] == 'Male'), 'Dependents'] = '3+'

    

    # For Females 

    dataset.loc[(dataset['Dependents'].isnull()) & (dataset['ApplicantIncome'] < 3416.0) & (dataset['Gender'] == 'Female'), 'Dependents'] = '0'

    dataset.loc[(dataset['Dependents'].isnull()) & (dataset['ApplicantIncome'] > 4608.0) & (dataset['ApplicantIncome'] <= 4200) & (dataset['Gender'] == 'Female'),'Dependents'] = '1'

    dataset.loc[(dataset['Dependents'].isnull()) & (dataset['ApplicantIncome'] > 3427.0) & (dataset['ApplicantIncome'] <= 4608.0) & (dataset['Gender'] == 'Female'),'Dependents'] = '2'

    dataset.loc[(dataset['Dependents'].isnull()) & (dataset['ApplicantIncome'] > 4200.0) & (dataset['Gender'] == 'Female'), 'Dependents'] = '3+'

train_data.isnull().sum()
print(test_data.isnull().sum())

print("\n")

test_data[test_data['Dependents'].isnull()]
freq_gender = train_data.Gender.dropna().mode()[0]

freq_gender
for dataset in combine: 

    dataset['Gender'] = dataset['Gender'].fillna(freq_gender)

    

train_data[['Gender', 'Loan_Status']].groupby('Gender', as_index = False).mean()
for dataset in combine:   

    # For males

    dataset.loc[(dataset['Dependents'].isnull()) & (dataset['ApplicantIncome'] < 3683.5) & (dataset['Gender'] == 'Male'), 'Dependents'] = '0'

    dataset.loc[(dataset['Dependents'].isnull()) & (dataset['ApplicantIncome'] > 3683.5) & (dataset['ApplicantIncome'] <= 3931.5) & (dataset['Gender'] == 'Male'),'Dependents'] = '1'

    dataset.loc[(dataset['Dependents'].isnull()) & (dataset['ApplicantIncome'] > 3931.5) & (dataset['ApplicantIncome'] <= 4200.0) & (dataset['Gender'] == 'Male'),'Dependents'] = '2'

    dataset.loc[(dataset['Dependents'].isnull()) & (dataset['ApplicantIncome'] > 4200.0) & (dataset['Gender'] == 'Male'), 'Dependents'] = '3+'

    

    # For Females 

    dataset.loc[(dataset['Dependents'].isnull()) & (dataset['ApplicantIncome'] < 3416.0) & (dataset['Gender'] == 'Female'), 'Dependents'] = '0'

    dataset.loc[(dataset['Dependents'].isnull()) & (dataset['ApplicantIncome'] > 4608.0) & (dataset['ApplicantIncome'] <= 4200) & (dataset['Gender'] == 'Female'),'Dependents'] = '1'

    dataset.loc[(dataset['Dependents'].isnull()) & (dataset['ApplicantIncome'] > 3427.0) & (dataset['ApplicantIncome'] <= 4608.0) & (dataset['Gender'] == 'Female'),'Dependents'] = '2'

    dataset.loc[(dataset['Dependents'].isnull()) & (dataset['ApplicantIncome'] > 4200.0) & (dataset['Gender'] == 'Female'), 'Dependents'] = '3+'

train_data.isnull().sum()
for dataset in combine: 

    dataset.loc[(dataset['Self_Employed'].isnull()) & (dataset['Education'] == 'Not Graduate'), 'Self_Employed'] = 'No'

train_data.isnull().sum()
train_data['Dependents'] = train_data['Dependents'].map( {'0' : 0, '1' : 1, '2' : 2, '3+' : 3} )

train_data.head(10)
test_data['Dependents'] = test_data['Dependents'].map( {'0' : 0, '1' : 1, '2' : 2, '3+' : 3} )

test_data.head(10)
train_data[['ApplicantIncome', 'Self_Employed']].groupby('Self_Employed', as_index = False).median()
combine = [train_data, test_data]

for dataset in combine: 

    dataset['Self_Employed'] = dataset['Self_Employed'].dropna(0).map({'No' : 0, 'Yes' : 1})

train_data.head(20)
train_data[['Self_Employed', 'Gender']].groupby('Gender', as_index = False).mean()
train_data[['Self_Employed', 'Education']].groupby('Education', as_index = False).mean()
train_data[['Self_Employed', 'Dependents']].groupby('Self_Employed', as_index = False).mean()
train_data[['Self_Employed', 'Married', 'Gender']].groupby(['Married', 'Gender'], as_index = False).mean()
for dataset in combine:

    dataset.loc[(dataset['Self_Employed'].isnull()) & (dataset['ApplicantIncome'] < 5809), 'Self_Employed'] = 0

    dataset.loc[(dataset['Self_Employed'].isnull()) & (dataset['ApplicantIncome'] >= 5809), 'Self_Employed'] = 1

train_data.head()
train_data.describe()

plt.figure(figsize = (15, 10))

sns.heatmap(train_data.corr(), annot = True)
# train_data['IncomeBand'] = pd.cut(train_data['Loan_Amount_Term'], 7)

# train_data[['IncomeBand', 'Loan_Status']].groupby(['IncomeBand'], as_index = False).mean()
# for dataset in combine:    

#     dataset.loc[ dataset['ApplicantIncome'] <= 11700.0, 'ApplicantIncome'] = 0

#     dataset.loc[(dataset['ApplicantIncome'] > 11700.0) & (dataset['ApplicantIncome'] <= 23250.0), 'ApplicantIncome'] = 1

#     dataset.loc[(dataset['ApplicantIncome'] > 23250.0) & (dataset['ApplicantIncome'] <= 34800.0), 'ApplicantIncome'] = 2

#     dataset.loc[(dataset['ApplicantIncome'] > 34800.0) & (dataset['ApplicantIncome'] <= 46350.0), 'ApplicantIncome'] = 3

#     dataset.loc[(dataset['ApplicantIncome'] > 46350.0) & (dataset['ApplicantIncome'] <= 57900.0), 'ApplicantIncome'] = 4

#     dataset.loc[(dataset['ApplicantIncome'] > 57900.0) & (dataset['ApplicantIncome'] <= 69450.0), 'ApplicantIncome'] = 5

#     dataset.loc[ dataset['ApplicantIncome'] > 69450, 'ApplicantIncome'] = 7

# train_data.tail(20)
train_data['LoanAmount'].fillna(train_data['LoanAmount'].median(), inplace = True)

train_data['Loan_Amount_Term'].fillna(train_data['Loan_Amount_Term'].median(), inplace = True)

test_data['LoanAmount'].fillna(train_data['LoanAmount'].median(), inplace = True)

test_data['Loan_Amount_Term'].fillna(train_data['Loan_Amount_Term'].median(), inplace = True)
train_data.isnull().sum()
train_data[['Credit_History', 'Education']].groupby('Education', as_index = False).mean()
train_data[['Credit_History', 'ApplicantIncome']].groupby('Credit_History', as_index = False).mean()
import random

for dataset in combine:   

    # For males

    dataset.loc[(dataset['Credit_History'].isnull()) & (dataset['ApplicantIncome'] < 3683.5) & (dataset['Gender'] == 'Male'), 'Credit_History'] = random.randint(0,1)

    dataset.loc[(dataset['Credit_History'].isnull()) & (dataset['ApplicantIncome'] > 3683.5) & (dataset['ApplicantIncome'] <= 3931.5) & (dataset['Gender'] == 'Male'),'Credit_History'] = random.randint(0,1)

    dataset.loc[(dataset['Credit_History'].isnull()) & (dataset['ApplicantIncome'] > 3931.5) & (dataset['ApplicantIncome'] <= 4200.0) & (dataset['Gender'] == 'Male'),'Credit_History'] = random.randint(0,1)

    dataset.loc[(dataset['Credit_History'].isnull()) & (dataset['ApplicantIncome'] > 4200.0) & (dataset['Gender'] == 'Male'), 'Credit_History'] = random.randint(0,1)

    

    # For Females 

    dataset.loc[(dataset['Credit_History'].isnull()) & (dataset['ApplicantIncome'] < 3416.0) & (dataset['Gender'] == 'Female'), 'Dependents'] = random.randint(0,1)

    dataset.loc[(dataset['Credit_History'].isnull()) & (dataset['ApplicantIncome'] > 4608.0) & (dataset['ApplicantIncome'] <= 4200) & (dataset['Gender'] == 'Female'),'Credit_History'] = random.randint(0,1)

    dataset.loc[(dataset['Credit_History'].isnull()) & (dataset['ApplicantIncome'] > 3427.0) & (dataset['ApplicantIncome'] <= 4608.0) & (dataset['Gender'] == 'Female'),'Credit_History'] = random.randint(0,1)

    dataset.loc[(dataset['Credit_History'].isnull()) & (dataset['ApplicantIncome'] > 4200.0) & (dataset['Gender'] == 'Female'), 'Credit_History'] = random.randint(0,1)

train_data.head(20)
train_data.isnull().sum()
train_data[train_data['Credit_History'].isnull()]
train_data.set_value(198, 'Credit_History', 1) 

train_data.set_value(323, 'Credit_History', 1) 

train_data.set_value(473, 'Credit_History', 1) 

train_data.set_value(544, 'Credit_History', 0) 

train_data.set_value(556, 'Credit_History', 0) 

train_data.set_value(600, 'Credit_History', 0) 
train_data.isnull().any().any()
test_data.isnull().sum()
test_data[test_data['Credit_History'].isnull()]
test_data.set_value(177, 'Credit_History', 1) 

test_data.set_value(259, 'Credit_History', 0) 

test_data.set_value(336, 'Credit_History', 1) 
print(train_data.isnull().any().any())

print(test_data.isnull().any().any())
combine = [train_data, test_data]

for dataset in combine: 

    dataset['Education'] = dataset['Education'].map( {'Graduate' : 1, 'Not Graduate' : 0} )

    dataset['Property_Area'] = dataset['Property_Area'].map( {'Rural' : 0, 'Urban' : 2, 'Semiurban' : 1} )

train_data.head()
gender = pd.get_dummies(train_data['Gender'])

train_data = pd.concat([train_data, gender], axis = 1)

train_data.drop('Gender', axis = 1, inplace = True)

train_data.head()
gender = pd.get_dummies(test_data['Gender'])

test_data = pd.concat([test_data, gender], axis = 1)

test_data.drop('Gender', axis = 1, inplace = True)

test_data.head()
train_data = train_data.astype(int)

train_data.dtypes
train_data['LoanBand'] = pd.cut(train_data['LoanAmount'], 4)

train_data[['LoanBand', 'Loan_Status']].groupby('LoanBand').mean()
for dataset in combine:    

    dataset.loc[ dataset['LoanAmount'] <= 181.75, 'LoanAmount'] = 0

    dataset.loc[(dataset['LoanAmount'] > 181.75) & (dataset['LoanAmount'] <= 354.50), 'LoanAmount'] = 1

    dataset.loc[(dataset['LoanAmount'] > 354.50) & (dataset['LoanAmount'] <= 527.25), 'LoanAmount'] = 2

    dataset.loc[ dataset['LoanAmount'] > 700.00, 'LoanAmount'] = 3

train_data.drop('LoanBand', axis = 1, inplace = True)

train_data.head()
X_train = train_data.drop('Loan_Status', axis = 1)

y_train = train_data['Loan_Status']

X_test = test_data.copy()

print(X_train.shape, y_train.shape, X_test.shape)
# Logistic Regression 



logreg = LogisticRegression()

logreg.fit(X_train, y_train)

y_pred = logreg.predict(X_test)

acc_log = round(logreg.score(X_train, y_train) * 100, 2)

acc_log
coeff_data = pd.DataFrame(train_data.columns.delete(0))

coeff_data.columns = ['Feature']

coeff_data['Correlation'] = pd.Series(logreg.coef_[0])

coeff_data.sort_values(by = 'Correlation', ascending = False)
# Support vector machines 

svc = SVC()

svc.fit(X_train, y_train)

Y_pred = svc.predict(X_test)

acc_svc = round(svc.score(X_train, y_train) * 100, 2)

acc_svc
values = {}

for val in range(1, 51):

    knn = KNeighborsClassifier(n_neighbors = val)

    knn.fit(X_train, y_train)

    Y_pred = knn.predict(X_test)

    acc = round(knn.score(X_train, y_train) * 100, 2)

    values[val] = acc
x_values, y_values = [], []

for val in values: 

    x_values.append(val)

    y_values.append(values[val])

from matplotlib import style

style.use('ggplot')

plt.figure(figsize = (15,8))

plt.title('Accuracy Score vs Neighbours')

plt.xlabel('Number of Neighbours')

plt.ylabel('Accuracy Score')

plt.legend()

sns.barplot(x = x_values, y = y_values)
acc_knn = 0

for val in values:

    if values[val] > acc_knn: 

        acc_knn = values[val]

acc_knn
# Gaussian Naive Bayes



gaussian = GaussianNB()

gaussian.fit(X_train, y_train)

Y_pred = gaussian.predict(X_test)

acc_gaussian = round(gaussian.score(X_train, y_train) * 100, 2)

acc_gaussian
# Perceptron



perceptron = Perceptron()

perceptron.fit(X_train, y_train)

Y_pred = perceptron.predict(X_test)

acc_perceptron = round(perceptron.score(X_train, y_train) * 100, 2)

acc_perceptron
# Linear SVC



linear_svc = LinearSVC()

linear_svc.fit(X_train, y_train)

Y_pred = linear_svc.predict(X_test)

acc_linear_svc = round(linear_svc.score(X_train, y_train) * 100, 2)

acc_linear_svc
# Stochastic Gradient Descent



sgd = SGDClassifier()

sgd.fit(X_train, y_train)

Y_pred = sgd.predict(X_test)

acc_sgd = round(sgd.score(X_train, y_train) * 100, 2)

acc_sgd
# Decision Tree



decision_tree = DecisionTreeClassifier(max_depth = 7)

decision_tree.fit(X_train, y_train)

Y_pred = decision_tree.predict(X_test)

acc_decision_tree = round(decision_tree.score(X_train, y_train) * 100, 2)

acc_decision_tree
# Random Forest



random_forest = RandomForestClassifier(n_estimators = 10)

random_forest.fit(X_train, y_train)

y_pred = random_forest.predict(X_test)

random_forest.score(X_train, y_train)

acc_random_forest = round(random_forest.score(X_train, y_train) * 100, 2)

acc_random_forest
pred_values = pd.DataFrame({

    'Model': ['Support Vector Machines', 'KNN', 'Logistic Regression', 

              'Random Forest', 'Naive Bayes', 'Perceptron', 

              'Stochastic Gradient Decent', 'Linear SVC', 

              'Decision Tree'],

    'Score': [acc_svc, acc_knn, acc_log, 

              acc_random_forest, acc_gaussian, acc_perceptron, 

              acc_sgd, acc_linear_svc, acc_decision_tree]})

pred_values.sort_values(by='Score', ascending=False)
submission = pd.DataFrame({

        "Loan_ID": test_sub["Loan_ID"],

        "Loan_Status": y_pred

    })

submission['Loan_Status'].replace(0, 'N',inplace=True) 

submission['Loan_Status'].replace(1, 'Y',inplace=True)

submission.to_csv('submission_loan.csv', index=False)