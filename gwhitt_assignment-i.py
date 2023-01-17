

import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt



#Setting Seaborn Color Codes

sns.set(color_codes=True)



#Import Dataset

original_data = pd.read_csv("../input/loan.csv", low_memory=False)
#Only Keep Data for Loans who status is Fully Paid or Default

data = original_data[(original_data['loan_status'] =='Fully Paid') | (original_data['loan_status'] == 'Default')]



#Transform  Fully Paid into Binary Target Variable

data['target']=(data['loan_status'] == 'Fully Paid')



#print out number of records

print("Number of Records: " + str(len(data.index)))
#print out and calculate mean, median, maximum, and standard deviations

print("Mean of Loan Amounts: " + str(data['loan_amnt'].mean()))

print("Median of Loan Amounts: " + str(data['loan_amnt'].median()))

print("Maximum of Loan Amounts: " + str(data['loan_amnt'].max()))

print("Standard Deviation of Loan Amounts: " + str(data['loan_amnt'].std()))

#Plot Histogram of Loan Amounts

#plt.hist(data['loan_amnt'], bins = 100)

sns.distplot(data['loan_amnt'], bins = 20)
#Print mean and standard deviations 

short_term_loans = data[(data['term'] == " 36 months")]

long_term_loans = data[(data['term'] == " 60 months")]

print("Mean of Short Term Loans: " + str(short_term_loans['int_rate'].mean()))

print("Standard Deviation of Short Term Loans: " + str(short_term_loans['int_rate'].std()))

print("------------------------------------------------------------------------")

print("Mean of Long Term Loans: " + str(long_term_loans['int_rate'].mean()))

print("Standard Deviation of Long Term Loans: " + str(long_term_loans['int_rate'].std()))





dim = (12, 8)

sns.set(style="whitegrid")

fig, ax = plt.subplots(figsize=dim)

ax = sns.boxplot(x= data['term'], y= data['int_rate'])

ax.set(xlabel='Loan Term', ylabel='Interest Rate')
data.int_rate.groupby(data.grade).mean()
data['ryield'] = np.where(data.term == ' 36 months', np.power(((data['total_pymnt']/data['funded_amnt'])),(1/3))-1, np.power(((data['total_pymnt']/data['funded_amnt'])),(1/5))-1)

data.ryield.groupby(data.grade).max()
print(data['application_type'].unique())

joint_app = data[(data['application_type'] =='Joint App')]

individual = data[(data['application_type'] =='Individual')]

print("Number of Records for Joint App: " + str(len(joint_app.index)))

print("Number of Records for Individual: " + str(len(individual)))
feature_subset = data[['loan_amnt','funded_amnt','funded_amnt_inv','term','int_rate','emp_length','addr_state','verification_status','purpose','policy_code']]

feature_subset.head()

array = ['term','emp_length','addr_state','verification_status','purpose']

dummy_data = pd.get_dummies(data = feature_subset, columns= array, sparse=True)

dummy_data.shape
from sklearn.model_selection import train_test_split

dummy_data = dummy_data.round({'funded_amnt_inv': 0, 'int_rate': 0})

target_data = data[['target']]

target_dummy_data = pd.get_dummies(target_data)

target_data.head()

x_train, x_test,y_train, y_test = train_test_split(dummy_data,target_dummy_data,test_size=0.33, random_state=42)



from sklearn.ensemble import RandomForestClassifier

from sklearn.datasets import make_classification

clf = RandomForestClassifier(n_estimators=100, max_depth=4, random_state=42)
clf.fit(x_train, y_train.values.ravel())
forest = clf

X = x_train

importances = forest.feature_importances_

std = np.std([tree.feature_importances_ for tree in forest.estimators_],

             axis=0)

indices = np.argsort(importances)[::-1]



subset_indices = []

new_indices = []

for f in range(X.shape[1]):

    if importances[f] != 0:

        subset_indices.append(importances[f])

        new_indices.append(f)



# Plot the feature importances of the forest

plt.figure(figsize=(18, 14))

plt.title("Feature Importances")

plt.barh(range(X.shape[1]), importances[indices],color="b", yerr=std[indices], align="center")



#plt.xticks(range(X.shape[1]), new_indices)

plt.ylim([0, 45])

plt.gca().invert_yaxis()

#indices = indices[0:45]

plt.yticks(indices, list(X.columns.values))

plt.show()
y_pred=clf.predict(x_test)
#Import scikit-learn metrics module for accuracy calculation

from sklearn import metrics



# Model Accuracy, how often is the classifier correct?

print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

print("Types of Prediction:", np.unique(y_pred))



# from sklearn.model_selection import KFold 

# kf = RepeatedKFold(n_splits=5, n_repeats=10, random_state=None) 



# for train_index, test_index in kf.split(X):

#       print("Train:", train_index, "Validation:",test_index)

#       X_train, X_test = X[train_index], X[test_index] 

#       y_train, y_test = y[train_index], y[test_index]





#Work with Original Dataset

array = ['term','emp_length','addr_state','verification_status','purpose']

feature_subset = original_data[['loan_amnt','funded_amnt','funded_amnt_inv','term','int_rate','emp_length','addr_state','verification_status','purpose','policy_code']]

dummy_data = pd.get_dummies(data = feature_subset, columns= array, sparse=True)
from sklearn.model_selection import train_test_split

dummy_data = dummy_data.round({'funded_amnt_inv': 0, 'int_rate': 0})

target_data = original_data[['loan_status']]

target_dummy_data = pd.get_dummies(target_data)

target_data.head()

x_train, x_test,y_train, y_test = train_test_split(dummy_data,target_dummy_data,test_size=0.33, random_state=42)
from sklearn.ensemble import RandomForestClassifier

from sklearn.datasets import make_classification

clf = RandomForestClassifier(n_estimators=100, max_depth=4, random_state=42)

target_dummy_data.head()
clf.fit(x_train, y_train)
y_pred=clf.predict(x_test)
#Import scikit-learn metrics module for accuracy calculation

from sklearn import metrics

# Model Accuracy, how often is the classifier correct?

print("Accuracy:",metrics.accuracy_score(y_test, y_pred))