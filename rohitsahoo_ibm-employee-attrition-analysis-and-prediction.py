import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt
attrition = pd.read_csv("../input/employee/train.csv")
attrition.head() #Top 5 Records
attrition.isnull().any()
attrition.dtypes
categorical = attrition.select_dtypes(include = 'object')

print(categorical.columns)
numerical = attrition.select_dtypes(include=['float64','int64'])
print((numerical.columns))
sns.kdeplot(attrition['Age'])
sns.distplot(attrition['Age'])
fig, ax = plt.subplots(5,2, figsize=(9,9))

sns.distplot(attrition['TotalWorkingYears'], ax = ax[0,0])

sns.distplot(attrition['MonthlyIncome'], ax = ax[0,1])

sns.distplot(attrition['YearsAtCompany'], ax = ax[1,0])

sns.distplot(attrition['DistanceFromHome'], ax = ax[1,1])

sns.distplot(attrition['YearsWithCurrManager'], ax = ax[2,0])

sns.distplot(attrition['YearsSinceLastPromotion'], ax = ax[2,1])

sns.distplot(attrition['PercentSalaryHike'], ax = ax[3,0])

sns.distplot(attrition['YearsAtCompany'], ax = ax[3,1])

sns.distplot(attrition['YearsSinceLastPromotion'], ax = ax[4,0])

sns.distplot(attrition['TrainingTimesLastYear'], ax = ax[4,1])

plt.tight_layout()

plt.show()
sns.factorplot(data = attrition, kind = 'count', aspect = 3, size = 5, x = 'BusinessTravel')
sns.factorplot(data = attrition, kind = 'count', aspect = 3, size = 5, x = 'Department')
sns.factorplot(data = attrition, kind = 'count', aspect = 3, size = 5, x = 'EducationField')
bins = [0, 18, 35, 60, np.inf]

labels = ['Student', 'Freshers/junior', 'Senior', 'Retired']

attrition['AgeGroup'] = pd.cut(attrition["Age"], bins, labels = labels)

sns.factorplot(data = attrition, kind = 'count', aspect = 3, size = 5, x = 'AgeGroup')
sns.factorplot(data = attrition, kind = 'count', aspect = 3, size = 5, x = 'Gender')
sns.factorplot(data = attrition, kind = 'count', aspect = 3, size = 5, x = 'JobRole')
sns.factorplot(data = attrition, kind = 'count', aspect = 3, size = 5, x = 'Over18')
sns.factorplot(data = attrition, kind = 'count', aspect = 3, size = 5, x = 'OverTime')
sns.factorplot(data = attrition, kind = 'count', aspect = 3, size = 5, x = 'MaritalStatus')
sns.factorplot(data = attrition, kind = 'count', aspect = 3, size = 5, x = 'Attrition')
cor_mat = attrition.corr()

mask = np.array(cor_mat)

mask[np.tril_indices_from(mask)]=False

fig = plt.gcf()

fig.set_size_inches(60,12)

sns.heatmap(data = cor_mat, mask = mask, square = True, annot = True, cbar = True)
attrition.columns
continious = ['Age',  'DailyRate', 'HourlyRate', 'MonthlyIncome', 'MonthlyRate', 'TotalWorkingYears', 'YearsAtCompany' ]
for var in continious:

    #boxplot

    plt.figure(figsize = (10,5))

    plt.subplot(1,2,1)

    fig = attrition.boxplot(column = var)

    fig.set_ylabel(var)

    

    #histogram

    plt.subplot(1,2,2)

    fig = attrition[var].hist(bins = 20)

    fig.set_ylabel('No. of Employees')

    fig.set_xlabel(var)

    

    plt.show()

    
attrition['TotalWorkingYears'].describe()
categorical.head()
attrition_cat = pd.get_dummies(categorical)
attrition_cat.head()
numerical.head()
attrition_final = pd.concat([numerical,attrition_cat], axis=1)
attrition_final.head()
attrition_final = attrition_final.drop('Attrition', axis = 1)
attrition_final
target = attrition['Attrition']
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.svm import SVC

from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(attrition_final ,target, test_size = 0.2, random_state = 0)
x_train.shape
x_test.shape
model = RandomForestClassifier()

model.fit(x_train,y_train)

model_predictions = model.predict(x_test)

print("Accuracy: ", accuracy_score(y_test, model_predictions))

print(classification_report(y_test, model_predictions))
model = LogisticRegression()

model.fit(x_train,y_train)

model_predictions = model.predict(x_test)

print("Accuracy: ", accuracy_score(y_test, model_predictions))

print(classification_report(y_test, model_predictions))
model = DecisionTreeClassifier()

model.fit(x_train,y_train)

model_predictions = model.predict(x_test)

print("Accuracy: ", accuracy_score(y_test, model_predictions))

print(classification_report(y_test, model_predictions))
model = KNeighborsClassifier()

model.fit(x_train,y_train)

model_predictions = model.predict(x_test)

print("Accuracy: ", accuracy_score(y_test, model_predictions))

print(classification_report(y_test, model_predictions))
model = SVC()

model.fit(x_train,y_train)

model_predictions = model.predict(x_test)

print("Accuracy: ", accuracy_score(y_test, model_predictions))

print(classification_report(y_test, model_predictions))
from imblearn.over_sampling import SMOTE
oversampler = SMOTE(random_state = 12, sampling_strategy = 1.0)

smote_train, smote_target = oversampler.fit_sample(x_train,y_train)
smote_train.shape
model = RandomForestClassifier()

model.fit(smote_train,smote_target)

model_predictions = model.predict(x_test)

print("Accuracy: ", accuracy_score(y_test, model_predictions))

print(classification_report(y_test, model_predictions))