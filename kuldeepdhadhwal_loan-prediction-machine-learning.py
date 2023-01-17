# data analysis and wrangling

import pandas as pd

import numpy as np



# visualization

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline



# machine learning

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC, LinearSVC

from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.linear_model import Perceptron

from sklearn.linear_model import SGDClassifier

from sklearn.tree import DecisionTreeClassifier

train_data = pd.read_csv("../input/train_loan_data.csv")
train_data.shape
target_data = train_data.drop(['Loan_Status'], axis=1)
train_data.head()
test_data = pd.read_csv("../input/test_loan_data.csv")
test_data.head()
test_data.columns
combine = [train_data, test_data]

print(train_data.info())

print("-"*40)

print(test_data.info())



for column in train_data[['Gender', 'Married', 'Dependents', 'Self_Employed', 'LoanAmount', 'Loan_Amount_Term','Credit_History']]:

    train_data[column].fillna(train_data[column].mode()[0], inplace=True)
train_data.describe()
train_data.describe(include=['O'])
train_data.head()
# creating a dict file  

gender = {'Male': 1,'Female': 0} 

married = {'Yes': 1, 'No': 0}

education = {'Graduate' : 1, 'Not Graduate' : 0}

self_employed = {'Yes' : 1, 'No' : 0}

loan_status = {'Y' : 1, 'N' : 0}

property_area = {'Urban' : 1, 'Rural' : 2, 'Semiurban' : 3}

  

# traversing through dataframe 

# Gender, Married, Dependents, Education, Self_Employed, Property_Area column and writing 

# values where key matches 

train_data.Gender = [gender[item] for item in train_data.Gender]

train_data.Married = [married[item] for item in train_data.Married] 

train_data.Education = [education[item] for item in train_data.Education] 

train_data.Self_Employed = [self_employed[item] for item in train_data.Self_Employed] 

train_data.Loan_Status = [loan_status[item] for item in train_data.Loan_Status] 

train_data.Property_Area = [property_area[item] for item in train_data.Property_Area] 

target = train_data['Loan_Status']
print(target)
embarked_dummies = pd.get_dummies(train_data.Property_Area, prefix='Property_Area')

pd.concat([train_data, embarked_dummies], axis=1)

train_data = train_data.replace(to_replace ="3+", 

                 value ="4") 
train_data.head()
train_data = train_data.drop(['Loan_ID'], axis=1)
train_data.head()
corr = train_data.corr()

# Generate a mask for the upper triangle

mask = np.zeros_like(corr, dtype=np.bool)

mask[np.triu_indices_from(mask)] = True



# Set up the matplotlib figure

f, ax = plt.subplots(figsize=(11, 9))



# Generate a custom diverging colormap

cmap = sns.diverging_palette(220, 10, as_cmap=True)



# Draw the heatmap with the mask and correct aspect ratio

sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,

            square=True, linewidths=.5, cbar_kws={"shrink": .5})
train_data.describe()
combine = [train_data]

for dataset in combine:

    dataset.loc[ dataset['ApplicantIncome'] <= 150.0, 'ApplicantIncome'] = 0

    dataset.loc[(dataset['ApplicantIncome'] > 150.0) & (dataset['ApplicantIncome'] <= 2877.50), 'ApplicantIncome'] = 1

    dataset.loc[(dataset['ApplicantIncome'] > 2877.50) & (dataset['ApplicantIncome'] <= 3812.50), 'ApplicantIncome']   = 2

    dataset.loc[ dataset['ApplicantIncome'] > 3812.50, 'ApplicantIncome'] = 3

    dataset['ApplicantIncome'] = dataset['ApplicantIncome'].astype(int)

    

train_data.head(10)
from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(train_data, target, test_size=0.25, random_state=42)

# Logistic Regression



logreg = LogisticRegression()

logreg.fit(X_train, Y_train)

Y_pred = logreg.predict(X_test)

acc_log = round(logreg.score(X_train, Y_train) * 100, 2)

acc_log
# Support Vector Machines

svc = SVC()

svc.fit(X_train, Y_train)

Y_pred = svc.predict(X_test)

acc_svc = round(svc.score(X_train, Y_train) * 100, 2)

acc_svc

    

knn = KNeighborsClassifier(n_neighbors = 3)

knn.fit(X_train, Y_train)

Y_pred = knn.predict(X_test)

acc_knn = round(knn.score(X_train, Y_train) * 100, 2)

acc_knn
# Gaussian Naive Bayes



gaussian = GaussianNB()

gaussian.fit(X_train, Y_train)

Y_pred = gaussian.predict(X_test)

acc_gaussian = round(gaussian.score(X_train, Y_train) * 100, 2)

acc_gaussian
perceptron = Perceptron()

perceptron.fit(X_train, Y_train)

Y_pred = perceptron.predict(X_test)

acc_perceptron = round(perceptron.score(X_train, Y_train) * 100, 2)

acc_perceptron
# Decision Tree



decision_tree = DecisionTreeClassifier()

decision_tree.fit(X_train, Y_train)

Y_pred = decision_tree.predict(X_test)

acc_decision_tree = round(decision_tree.score(X_train, Y_train) * 100, 2)

acc_decision_tree