%matplotlib inline

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

sns.set(context='notebook', style='darkgrid', palette='deep', font='sans-serif')
maths_students = pd.read_csv('../input/student-mat.csv')

maths_students.head()
print("Number of rows:", maths_students.shape[0])
portuguese_students = pd.read_csv('../input/student-por.csv')

portuguese_students.head()
print("Number of rows:", portuguese_students.shape[0])
students = pd.concat([maths_students, portuguese_students])

students = students.drop_duplicates(["school","sex","age","address","famsize","Pstatus","Medu","Fedu","Mjob","Fjob","reason","nursery","internet"])

students.head()
students['school_GP'] = students['school'].apply(lambda x: 1 if x == 'GP' else 0)

students['male'] = students['sex'].apply(lambda x: 1 if x == 'M' else 0)

students['urban'] = students['address'].apply(lambda x: 1 if x == 'U' else 0)

students['family_size > 3'] = students['famsize'].apply(lambda x: 1 if x == 'GT3' else 0)

students['parents_together'] = students['Pstatus'].apply(lambda x: 1 if x == 'T' else 0)

students = pd.concat([students, pd.get_dummies(students['Mjob'], prefix='Mjob')], axis=1)

students = pd.concat([students, pd.get_dummies(students['Fjob'], prefix='Fjob')], axis=1)

students = pd.concat([students, pd.get_dummies(students['reason'], prefix='reason')], axis=1)

students = pd.concat([students, pd.get_dummies(students['guardian'], prefix='guardian')], axis=1)

students['schoolsup'] = students['schoolsup'].apply(lambda x: 1 if x == 'yes' else 0)

students['famsup'] = students['famsup'].apply(lambda x: 1 if x == 'yes' else 0)

students['paid'] = students['paid'].apply(lambda x: 1 if x == 'yes' else 0)

students['activities'] = students['activities'].apply(lambda x: 1 if x == 'yes' else 0)

students['nursery'] = students['nursery'].apply(lambda x: 1 if x == 'yes' else 0)

students['higher'] = students['higher'].apply(lambda x: 1 if x == 'yes' else 0)



students['internet'] = students['internet'].apply(lambda x: 1 if x == 'yes' else 0)

students['romantic'] = students['romantic'].apply(lambda x: 1 if x == 'yes' else 0)



students = students.drop(['school', 'sex', 'address', 'famsize', 'Pstatus', 'Mjob', 'Fjob', 'reason', 'guardian'], axis=1)



students.head()
students.describe()
plt.figure(figsize=(15,15))

sns.heatmap(students.corr(), square=True, linewidths=.5, cmap='seismic')

plt.show()
plt.hist(students['age'], rwidth=0.8, bins=8)

plt.title('Age of students')

plt.xlabel('Age')

plt.ylabel('Number of students')

plt.show()
plt.pie([np.sum(students['male'] == True), np.sum(students['male'] == False)], labels=['Male', 'Female'], colors=['b', 'pink'])

plt.title('Sex')

plt.show()
plt.pie([np.sum(students['urban'] == True), np.sum(students['urban'] == False)], labels=['Urban', 'Rural'])

plt.title('Residence')

plt.show()
plt.hist(students['absences'], rwidth=0.8, bins=6, range=[0, 30])

plt.title('Absences')

plt.xlabel('Number of days')

plt.ylabel('Number of students')

plt.show()
plt.pie([np.sum(students['paid'] == True), np.sum(students['paid'] == False)], labels=['Yes', 'No'], colors=['b', 'r'])

plt.title('Takes Tutitions?')

plt.show()
plt.pie([np.sum(students['higher'] == True), np.sum(students['higher'] == False)], labels=['Yes', 'No'], colors=['b', 'r'])

plt.title('Wants to go for higher education?')

plt.show()
plt.pie([np.sum(students['activities'] == True), np.sum(students['activities'] == False)], labels=['Yes', 'No'], colors=['b', 'r'])

plt.title('Takes part in extra curricular activities?')

plt.show()
plt.pie([np.sum(students['romantic'] == True), np.sum(students['romantic'] == False)], labels=['Yes', 'No'], colors=['b', 'r'])

plt.title('In a relationship?')

plt.show()
plt.pie([np.sum(students['internet'] == True), np.sum(students['internet'] == False)], labels=['Yes', 'No'], colors=['b', 'r'])

plt.title('Has internet at home?')

plt.show()
plt.hist(students['Dalc'], rwidth=0.8, bins=5, range=[1, 5])

plt.title('Daily Alcohol Consumption')

plt.xlabel('Consumption rated on scale of 1 to 5')

plt.ylabel('Number of students')

plt.show()
plt.hist(students['Walc'], rwidth=0.8, bins=5, range=[1, 5])

plt.title('Weekend Alcohol Consumption')

plt.xlabel('Consumption rated on scale of 1 to 5')

plt.ylabel('Number of students')

plt.show()
X = students.drop(['G1', 'G2', 'G3'], axis = 1)

y = students['G3']



print("X.shape:", X.shape)

print("y.shape:", y.shape)
from sklearn.model_selection import train_test_split

X_train, X_other, y_train, y_other = train_test_split(X, y, train_size=0.6)

X_cv, X_test, y_cv, y_test = train_test_split(X_other, y_other, test_size=0.5)



print("Train dataset size: ", X_train.shape[0])

print("CV size: ", X_cv.shape[0])

print("Test size: ", X_test.shape[0])
from sklearn.tree import DecisionTreeRegressor

from sklearn.linear_model import LinearRegression, Lasso, Ridge



clf_list = [DecisionTreeRegressor(), LinearRegression(), Ridge(), Lasso()]
for clf in clf_list:

    clf = clf.fit(X_train, y_train)

    print(clf.__class__.__name__, clf.score(X_cv, y_cv))
clf = Lasso()

clf = clf.fit(X_train, y_train)

print('Final Score (the coefficient of determination R^2 of the prediction):', clf.score(X_test, y_test))