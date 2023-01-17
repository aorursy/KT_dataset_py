import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split

from sklearn import svm

from sklearn.metrics import accuracy_score, confusion_matrix

import seaborn as sns
df = pd.read_csv('../input/students-performance-in-exams/StudentsPerformance.csv', engine = 'python')
df.head()
df.shape
df.info()
# no missing values

df.isna().sum()
# distribution of gender

df['gender'].value_counts()
# distribution of race/ethnicity 

df['race/ethnicity'].value_counts()
# distribution of parental education

df['parental level of education'].value_counts()
# distribution of discounted lunch

df['lunch'].value_counts()
# distribution of test preparation

df['test preparation course'].value_counts()
sns.countplot(df['gender'])

plt.title("Gender Counts")
sns.countplot(df['race/ethnicity'])

plt.title("Race/Enthicity Counts")
chart = sns.countplot(df['parental level of education'])

plt.title("Parents Education")

chart.set_xticklabels(chart.get_xticklabels(), rotation=45)
chart = sns.countplot(df['lunch'])

plt.title("Standard or Free/Reduced")
chart = sns.countplot(df['test preparation course'])

plt.title("Took Test Preparation or Not")
plt.ylim(0, 50)

plt.xlim(0,110)

plt.title("Math Score Histogram")

df['math score'].hist(bins = 100)
plt.ylim(0, 50)

plt.xlim(0,110)

plt.title("Reading Score Histogram")

df['reading score'].hist(bins = 100)
plt.ylim(0, 50)

plt.xlim(0,110)

plt.title("Writing Score Histogram")

df['writing score'].hist(bins = 100)
df.plot(x='math score', y='reading score', style = 'o')

plt.title("Scatter Plot Between Math and Reading Scores")
df.plot(x='math score', y='writing score', style = 'o')

plt.title("Scatter Plot Between Math and Writing Scores")
df.plot(x='reading score', y='writing score', style = 'o')

plt.title("Scatter Plot Between Reading and Writing Scores")
# mean of scores by gender

df1 = df.groupby('gender').mean()
df1
# transpose the grouped df for chart display

df2 = df1.T
df2
ax = df2.plot(kind='bar', title ="Average Grades by Gender", figsize=(10, 7), legend=True, fontsize=12)
df.info()
# one hot encoding of categorical variables

race = pd.get_dummies(df['race/ethnicity'])
race.head()
parents = pd.get_dummies(df['parental level of education'])
lunch = pd.get_dummies(df['lunch'])
test_prep = pd.get_dummies(df['test preparation course'])
# listing the encoded dataframes

dummies = [race, parents, lunch, test_prep]
# concatenate encoded dfs to eachother

concat = pd.concat(dummies, axis = 1)
concat.head()
# concatenate dummy df with original df

df3 = pd.concat([df, concat], axis = 1)
df3.info()
# average of values by gender

df_group = df3.groupby('gender').mean()

df_group
# drop object columns

df4 = df3.drop(['race/ethnicity', 'lunch', 'test preparation course', 'parental level of education'], axis = 1)
df4.head()
# assign score columns to X

X = df4[['math score', 'reading score', 'writing score']]
X.head()
y = df['gender']
# split into train / test data

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .15, random_state = 9)
# standardize X data

scaler = StandardScaler()

scaler.fit(X_train)

X_train = scaler.transform(X_train)

X_test = scaler.transform(X_test)
from sklearn.ensemble import RandomForestClassifier
regr_rf = RandomForestClassifier(max_depth=None, random_state=2, n_estimators=600)

regr_rf.fit(X_train, y_train)
rf_pred = regr_rf.predict(X_test)

rf_pred
accuracy = accuracy_score(y_test, rf_pred)

accuracy
from sklearn import linear_model
log_reg = linear_model.LogisticRegression(solver="newton-cg", C=.750, penalty="l2") 

log_reg.fit(X_train, y_train)

log_pred = log_reg.predict(X_test)
print(accuracy_score(y_test, log_pred))
svc = svm.SVC(kernel = 'rbf', gamma = 2.5, C = 1.0)

svc.fit(X_train, y_train)

svc_pred = svc.predict(X_test)
y_test.value_counts()
print(accuracy_score(y_test, svc_pred))
conf_mat = confusion_matrix(y_test, svc_pred)

conf_mat
ax= plt.subplot()

sns.heatmap(conf_mat, annot=True, ax = ax); #annot=True to annotate cells



# labels, title and ticks

ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels'); 

ax.set_title('Confusion Matrix'); 

ax.xaxis.set_ticklabels(['female', 'male']); ax.yaxis.set_ticklabels(['female', 'male']);