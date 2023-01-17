import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import math
data= pd.read_csv('../input/student-grade-prediction/student-mat.csv')

data
data['G_avg']= round((data['G1']+data['G2']+data['G3'])/3, 2)

data.head()
data.info()
data.isnull().sum().sum()
data.describe()
plt.hist(data['G_avg'], bins=20, range=[0,20])

plt.title('Distribution of grades average of students')

plt.xlabel('Grades average')

plt.ylabel('Count')

plt.show()
ax= sns.boxplot(data['G_avg'])

ax.set_title('Boxplot of average grades of students')

ax.set_xlabel('Average Grades')

plt.show()
ax = sns.countplot('age',hue='sex', data=data)

ax.set_title('Students distribution according to age and sex')

ax.set_xlabel('Age')

ax.set_ylabel('Count')

plt.show()
ax = sns.swarmplot(x='age', y='G_avg',hue='sex', data=data)

ax.set_title('Age and sex relation with average grades')

ax.set_xlabel('Age')

ax.set_ylabel('Average grades')

plt.show()
ax = sns.swarmplot(x='famsize', y='G_avg', data=data)

ax.set_title('Age and sex relation with average grades')

ax.set_xlabel('Age')

ax.set_ylabel('Average grades')

plt.show()
Pedu = data['Fedu'] + data['Medu'] 

ax = sns.swarmplot(x=Pedu,y=data['G_avg'])

ax.set_title('Parents education effect to child grades')

ax.set_xlabel('Parents education')

ax.set_ylabel('Average Grades')

plt.show()
ax = sns.boxplot(x=data['Fjob'],y=data['G_avg'])

ax.set_title('Father job effect to child grades')

ax.set_xlabel('Father job')

ax.set_ylabel('Average Grades')

plt.show()
ax = sns.boxplot(x=data['Mjob'],y=data['G_avg'])

ax.set_title('Mother job effect to child grades')

ax.set_xlabel('Mother job')

ax.set_ylabel('Average Grades')

plt.show()
ax = sns.swarmplot(x=data['Pstatus'],y=data['G_avg'])

ax.set_title('Parents status effect on grades')

ax.set_xlabel('A= apart, T= living together')

ax.set_ylabel('Average Grades')

plt.show()
ax = sns.swarmplot(x=data['famrel'], y=data['G_avg'])

ax.set_title('family relations effect to child grades')

ax.set_xlabel('Relationship with family scale')

ax.set_ylabel('Average Grades')

plt.show()
ax = sns.boxplot(x='traveltime', y='G_avg',hue='address', data=data)

ax.set_title('Address and travel time to school')

ax.set_xlabel('Travel time (1: <15min, 2: 15min-30min, 3: 30min-1h, 4: >1h)')

ax.set_ylabel('Average grades')

plt.show()
ax = sns.swarmplot(x=data['romantic'],y=data['G_avg'])

ax.set_title('Students having a romantic relationship')

ax.set_xlabel('Romantic')

ax.set_ylabel('Average Grades')

plt.show()
b = sns.boxplot(x=data['freetime'], hue=data['activities'], y=data['G_avg'])

b.set_title('Freetime and extra activities')

b.set_xlabel('Freetime')

b.set_ylabel('Average Grades')

plt.show()
ax = sns.boxplot(x=data['goout'],y=data['G_avg'])

ax.set_title('Students going out')

ax.set_xlabel('Go out times per week')

ax.set_ylabel('Average Grades')

plt.show()
alc= data['Walc'] + data['Dalc'] 

ax = sns.swarmplot(x=alc,y=data['G_avg'])

ax.set_title('Alcohol consumption')

ax.set_xlabel('Alcohol')

ax.set_ylabel('Average Grades')

plt.show()
ax= sns.violinplot(x=data['studytime'],y=data['G_avg'])

ax.set_title('Study time in relation with gradess')

ax.set_xlabel('Study time')

ax.set_ylabel('Average Grades')

plt.show()
ax = sns.violinplot(x=data['failures'],y=data['G_avg'])

ax.set_title('Past subjects failures')

ax.set_xlabel('Failures')

ax.set_ylabel('Average Grades')

plt.show()
ax= sns.swarmplot(x=data['absences'],y=data['G_avg'])

ax.set_title('Absence effect on results')

ax.set_xlabel('Number of absence')

ax.set_ylabel('Average Grades')

plt.show()
ax= sns.swarmplot(x=data['schoolsup'],y=data['G_avg'])

ax.set_title('School support and grades')

ax.set_xlabel('School support')

ax.set_ylabel('Average Grades')

plt.show()
ax= sns.swarmplot(x=data['paid'],y=data['G_avg'])

ax.set_title('Extra paid courses effect to grades')

ax.set_xlabel('paid course')

ax.set_ylabel('Average Grades')

plt.show()
ax= sns.swarmplot(x=data['internet'],y=data['G_avg'])

ax.set_title('internet access')

ax.set_xlabel('internet')

ax.set_ylabel('Average Grades')

plt.show()
ax= sns.boxplot(x=data['higher'],y=data['G_avg'])

ax.set_title('Students who aim to go to university')

ax.set_xlabel('higher education')

ax.set_ylabel('Average Grades')

plt.show()
data['school']=data['school'].map({'GP':0, 'MS':1})

data['sex']=data['sex'].map({'M':0 ,'F':1})

data['address']=data['address'].map({'R':0 ,'U':1})

data['famsize']=data['famsize'].map({'LE3':0 ,'GT3':1})

data['Pstatus']=data['Pstatus'].map({'A':0 ,'T':1})

data['Mjob']=data['Mjob'].map({'at_home':0 ,'services':1, 'teacher':2, 'health':3, 'other':4})

data['Fjob']=data['Fjob'].map({'at_home':0 ,'services':1, 'teacher':2, 'health':3, 'other':4})

data['famsup']=data['famsup'].map({'no':0, 'yes':1})

data['reason']=data['reason'].map({'course':0 ,'home':1, 'reputation':2, 'other':3})

data['guardian']=data['guardian'].map({'mother':0 ,'father':1, 'other':2})

data['schoolsup']=data['schoolsup'].map({'no':0, 'yes':1})

data['paid']=data['paid'].map({'no':0, 'yes':1})

data['activities']=data['activities'].map({'no':0, 'yes':1})

data['nursery']=data['nursery'].map({'no':0, 'yes':1})

data['higher']=data['higher'].map({'no':0, 'yes':1})

data['internet']=data['internet'].map({'no':0, 'yes':1})

data['romantic']=data['romantic'].map({'no':0, 'yes':1})
corr = data.corr()

corr.style.background_gradient(cmap='coolwarm').set_precision(2)
data= data.drop(['G1','G2','G3'], axis=1)

data
import sklearn as sk

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression

from sklearn import metrics
X = data[["failures"]]

y = data["G_avg"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=30)

model = LinearRegression()

model.fit(X_train, y_train)

predicted = model.predict(X_test)

print ("MSE :", metrics.mean_squared_error(y_test,predicted))

print("R squared :", metrics.r2_score(y_test,predicted))
plt.scatter(X, y)

plt.title("Linear Regression")

plt.xlabel("Failures")

plt.ylabel("Grade average")

plt.plot(X, model.predict(X), color="r")

plt.show()
from sklearn.preprocessing import PolynomialFeatures

X = data[["failures"]]

y = data["G_avg"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=30)

lg = LinearRegression()

poly = PolynomialFeatures()

X_train_fit = poly.fit_transform(X_train)

lg.fit(X_train_fit, y_train)

X_test_fit = poly.fit_transform(X_test)

predicted = lg.predict(X_test_fit)

print ("MSE :", metrics.mean_squared_error(y_test,predicted))

print("R squared :", metrics.r2_score(y_test,predicted))
X= data.drop(["G_avg"], axis=1)

y= data["G_avg"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=30)

model = LinearRegression()

model.fit(X_train, y_train)

predicted = model.predict(X_test)

print ("MSE :", metrics.mean_squared_error(y_test, predicted))

print("R squared :", metrics.r2_score(y_test, predicted))
data['pass']= np.where(data['G_avg']<10, 0, 1)

data
print(data['pass'].value_counts())
corr = data.corr()

corr.style.background_gradient(cmap='coolwarm').set_precision(2)
import sklearn as sk

from sklearn.model_selection import train_test_split

Y = data["pass"]

X = data[["failures","schoolsup","Medu","Fedu","higher","goout","internet"]]

X_train, X_test, Y_train, Y_test= train_test_split(X, Y, test_size=0.2, random_state=20)
from sklearn.linear_model import LogisticRegression

logreg= LogisticRegression(solver="lbfgs")

logreg.fit(X_train, Y_train)

Y_pred= logreg.predict(X_test)

print("Accuracy = {:.2f}".format(logreg.score(X_test, Y_test)))
Y_pred1= logreg.predict([[0,1,3,4,1,1,1],[3,0,2,3,1,5,1]])

print(Y_pred1)
confusion_matrix= pd.crosstab(Y_test, Y_pred, rownames=["Actual"], colnames=["predict"])

print(confusion_matrix)
from sklearn.metrics import classification_report

print(classification_report(Y_test, Y_pred))