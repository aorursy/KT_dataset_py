import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns
data = pd.read_csv('../input/factors-affecting-campus-placement/Placement_Data_Full_Class.csv')

data.drop('sl_no',axis = 1, inplace = True)
data.head()
data.shape
data.info()
data.isnull().sum()
mask = pd.isnull(data['salary'])
data[mask].head(20)
sns.countplot("gender", hue="status", data=data)

plt.xlabel('Gender')

plt.ylabel('Count')

fig = plt.gcf()

fig.set_size_inches(10, 7)
pd.crosstab(index = data['gender'], columns = data['status'])
# Calculating percentages

pd.crosstab(index = data['gender'], columns = data['status']).apply(lambda r: 100 * r/r.sum(), axis=1)
sns.swarmplot(x='gender',y='salary',data=data)

plt.xlabel('Gender')

plt.ylabel('Salary')

fig = plt.gcf()

fig.set_size_inches(8, 7)
sns.kdeplot(data['salary'][data['gender']=='M'])

sns.kdeplot(data['salary'][data['gender']=='F'])

plt.xlabel('Salary (100K)')

plt.ylabel('Count')

plt.legend(['Male','Female'])

fig = plt.gcf()

fig.set_size_inches(10, 7)
sns.kdeplot(data['ssc_p'][data['gender']=='M'])

sns.kdeplot(data['ssc_p'][data['gender']=='F'])

plt.xlabel('Secondary School Percentage')

plt.ylabel('Count')

plt.legend(['Male','Female'])

fig = plt.gcf()

fig.set_size_inches(10, 7)
sns.kdeplot(data['ssc_p'][data['status']=='Placed'])

sns.kdeplot(data['ssc_p'][data['status']=='Not Placed'])

plt.xlabel('Secondary School Percentage')

plt.ylabel('Count')

plt.legend(['Placed','Not Placed'])

fig = plt.gcf()

fig.set_size_inches(10, 7)
sns.kdeplot(data['salary'][data['ssc_b']=='Central'])

sns.kdeplot(data['salary'][data['ssc_b']=='Others'])

plt.xlabel('Salary (100K)')

plt.ylabel('Count')

plt.legend(['Central','Others'])

fig = plt.gcf()

fig.set_size_inches(10, 7)
sns.countplot('ssc_b',hue = 'status',data=data)

plt.xlabel('Secondary School Board')

plt.ylabel('Count')

fig = plt.gcf()

fig.set_size_inches(10, 7)
pd.crosstab(index = data['ssc_b'], columns = data['status'])
sns.relplot(x='ssc_p', y='salary', hue = 'gender', kind = 'line', data=data)

plt.xlabel('Secondary School Percentage')

plt.ylabel('Salary')

fig = plt.gcf()

fig.set_size_inches(12, 6)
sns.kdeplot(data['hsc_p'][data['gender']=='M'])

sns.kdeplot(data['hsc_p'][data['gender']=='F'])

plt.xlabel("Higher Secondary Percentage")

plt.ylabel("Count")

plt.legend(["Male","Female"])

fig = plt.gcf()

fig.set_size_inches(10,7)
sns.kdeplot(data['hsc_p'][data['status']=="Placed"])

sns.kdeplot(data['hsc_p'][data['status']=='Not Placed'])

plt.xlabel('Higher Secondary Percentage')

plt.ylabel('Count')

plt.legend(["Placed", "Not Placed"])

fig = plt.gcf()

fig.set_size_inches(10,7)
sns.relplot(x='hsc_p', y='salary', hue = 'hsc_s', kind = 'line', data=data)

plt.xlabel('Higher Secondary Percentage')

plt.ylabel('Salary')

fig = plt.gcf()

fig.set_size_inches(12, 6)
sns.countplot('hsc_b',hue = 'status',data=data)

plt.xlabel('Higher Secondary Board')

plt.ylabel('Count')

fig = plt.gcf()

fig.set_size_inches(10, 7)
sns.kdeplot(data['salary'][data['hsc_b']=='Central'])

sns.kdeplot(data['salary'][data['hsc_b']=='Others'])

plt.xlabel('Salary (100K)')

plt.ylabel('Count')

plt.legend(['Central','Others'])

fig = plt.gcf()

fig.set_size_inches(10,7)
sns.countplot('hsc_s', hue = 'gender', data=data)

plt.xlabel('Higher Secindary Stream')

plt.ylabel('Count')

fig = plt.gcf()

fig.set_size_inches(10,7)
pd.crosstab(index = data['hsc_s'], columns = data['gender'])
stream = list(data['hsc_s'].unique())

for s in stream:

    sns.kdeplot(data['hsc_p'][data['hsc_s']==s])

plt.xlabel('Higher Seconary Percentage')

plt.ylabel('Count')

plt.legend(stream)

fig = plt.gcf()

fig.set_size_inches(10,7)
sns.countplot('hsc_s', hue = 'status', data=data)

plt.xlabel('Higher Seconary Stream')

plt.ylabel('Count')

fig = plt.gcf()

fig.set_size_inches(10,7)
stream = list(data['hsc_s'].unique())

for s in stream:

    sns.kdeplot(data['salary'][data['hsc_s']==s])

plt.xlabel('Salary (100K)')

plt.ylabel('Count')

plt.legend(stream)

fig = plt.gcf()

fig.set_size_inches(10,7)
sns.kdeplot(data['degree_p'][data['status']=='Placed'])

sns.kdeplot(data['degree_p'][data['status']=='Not Placed'])

plt.legend(['Placed','Not Placed'])

plt.xlabel('Degree Percentage')

plt.ylabel('COunt')

fig = plt.gcf()

fig.set_size_inches(10,7)
sns.relplot(x='degree_p', y = 'salary', kind = 'line', hue = 'degree_t', data = data)

plt.xlabel('Degree Percentage')

plt.ylabel('Salary (100K)')

fig = plt.gcf()

fig.set_size_inches(12,6)
sns.countplot('degree_t', hue = 'status', data=data)

plt.xlabel('Degree Type')

plt.ylabel('Count')

fig = plt.gcf()

fig.set_size_inches(10,7)
pd.crosstab(index = data['degree_t'], columns = data['status']).apply(lambda r: r/r.sum(), axis=1)
for i in data['degree_t'].unique():

    sns.kdeplot(data['salary'][data['degree_t'] == i])

plt.legend(data.degree_t.unique())

plt.xlabel('Salary (100K)')

plt.ylabel('Count')

fig = plt.gcf()

fig.set_size_inches(10,7)
sns.countplot('workex', hue ='status', data=data)

plt.xlabel('Work Experience')

plt.ylabel('Count')

fig = plt.gcf()

fig.set_size_inches(10,7)
sns.kdeplot(data['salary'][data['workex']=='Yes'])

sns.kdeplot(data['salary'][data['workex']=='No'])

plt.legend(['Yes','No'])

plt.xlabel('Salary (100K)')

plt.ylabel('Count')

fig = plt.gcf()

fig.set_size_inches(10,7)
sns.lmplot(x = 'ssc_p', y = 'etest_p', hue = 'ssc_b', data=data)

plt.xlabel('Seconary School Percentage')

plt.ylabel('Employability Test')

fig = plt.gcf()

fig.set_size_inches(10,7)
sns.lmplot(x = 'hsc_p', y = 'etest_p', hue = 'hsc_s', data=data)

plt.xlabel('Higher Seconary Percentage')

plt.ylabel('Employability Test')

fig = plt.gcf()

fig.set_size_inches(10,7)
sns.lmplot(x = 'degree_p', y = 'etest_p', hue = 'degree_t', data=data)

plt.xlabel('Degree Percentage')

plt.ylabel('Employability Test')

fig = plt.gcf()

fig.set_size_inches(10,7)
sns.lmplot(x = 'mba_p', y = 'etest_p', hue = 'specialisation', data=data)

plt.xlabel('MBA Percentage')

plt.ylabel('Employability Test')

fig = plt.gcf()

fig.set_size_inches(10,7)
sns.kdeplot(data['etest_p'][data['status'] == 'Placed'])

sns.kdeplot(data['etest_p'][data['status'] == 'Not Placed'])

plt.legend(['Placed', 'Not Placed'])

plt.xlabel('Employabilty Test')

plt.ylabel('Count')

fig = plt.gcf()

fig.set_size_inches(10,7)
sns.relplot(x = 'etest_p', y = 'salary', kind = 'line', data = data)

plt.xlabel('Employability Test')

plt.ylabel('Salary')

fig = plt.gcf()

fig.set_size_inches(12,6)
sns.kdeplot(data['mba_p'][data['specialisation'] == 'Mkt&HR'])

sns.kdeplot(data['mba_p'][data['specialisation'] == 'Mkt&Fin'])

plt.legend(['Mkt&HR','Mkt&Fin'])

plt.xlabel('MBA Percentage')

plt.ylabel('Count')

fig = plt.gcf()

fig.set_size_inches(10,7)
sns.boxplot('mba_p','specialisation',data=data)

plt.xlabel('MBA Percentage')

plt.ylabel('Specialisation')

fig = plt.gcf()

fig.set_size_inches(10,7)
sns.countplot('specialisation', hue = 'status', data=data)

plt.xlabel('Specialisation')

plt.ylabel('Count')

fig = plt.gcf()

fig.set_size_inches(10,7)
# Finding percentages

pd.crosstab(index = data['status'], columns = data['specialisation']).apply(lambda r: 100 * r/r.sum(), axis=1)
pd.crosstab(index = data['specialisation'], columns = data['status']).apply(lambda r:100 *  r/r.sum(), axis=1)
sns.kdeplot(data['salary'][data['specialisation'] == 'Mkt&HR'])

sns.kdeplot(data['salary'][data['specialisation'] == 'Mkt&Fin'])

plt.legend(['Mkt&HR','Mkt&Fin'])

plt.xlabel('Salary (100K)')

plt.ylabel('Count')

fig = plt.gcf()

fig.set_size_inches(10,7)
sns.kdeplot(data['mba_p'][data['status'] == 'Placed'])

sns.kdeplot(data['mba_p'][data['status'] == 'Not Placed'])

plt.legend(['Placed','Not Placed'])

plt.xlabel('MBA Percentage')

plt.ylabel('Count')

fig = plt.gcf()

fig.set_size_inches(10,7)
sns.relplot(x = 'mba_p', y ='salary', kind = 'line', hue = 'specialisation', data = data)

plt.xlabel('MBA Percentage')

plt.ylabel('Salary')

fig = plt.gcf()

fig.set_size_inches(18,6)
# Dropping salary column as we do not require it for classification

data.drop('salary', axis = 1, inplace = True)



# Dropping the boards as it was not an important factor in determining placement status

data.drop(['ssc_b','hsc_b'],axis = 1, inplace = True)

data.head()
# Encoding our categorical values

data["gender"] = data.gender.map({'M':0,'F':1})

data["workex"] = data.workex.map({'No':0, 'Yes':1})

data['specialisation'] = data.specialisation.map({'Mkt&Fin':0,'Mkt&HR':1})

data['status'] = data.status.map({'Not Placed':0,'Placed':1})
for column in ['hsc_s', 'degree_t']:

    dummies = pd.get_dummies(data[column])

    data[dummies.columns] = dummies
data.drop(['hsc_s', 'degree_t'],axis = 1, inplace = True)
# Removing one of the three options from 'hsc_s' and one from 'Specialisation' to avoid dummy variable trap

data.drop(['Arts', 'Others'], axis = 1,inplace = True)

data.head()
X = data.copy()

y = data['status']

X.drop('status', axis = 1, inplace = True)
from sklearn.preprocessing import StandardScaler



scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)
from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size = 0.25, random_state = 3)
from sklearn.naive_bayes import GaussianNB

from sklearn.linear_model import LogisticRegression

from sklearn import tree

from sklearn.neighbors import KNeighborsClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.svm import SVC

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
gnb = GaussianNB()

gnb.fit(X_train, y_train)

y_pred = gnb.predict(X_test)

accuracy_score(y_test, y_pred)
lr = LogisticRegression()

lr.fit(X_train, y_train)

y_pred = lr.predict(X_test)

accuracy_score(y_test, y_pred)
dt = tree.DecisionTreeClassifier(random_state = 1)

dt.fit(X_train, y_train)

y_pred = dt.predict(X_test)

accuracy_score(y_test, y_pred)
knn = KNeighborsClassifier()

knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)

accuracy_score(y_test, y_pred)
rf = RandomForestClassifier(random_state = 1)

rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)

accuracy_score(y_test, y_pred)
svc = SVC(probability = True)

svc.fit(X_train, y_train)

y_pred = svc.predict(X_test)

accuracy_score(y_test, y_pred)
print(confusion_matrix(y_test, y_pred))
from xgboost import XGBClassifier

xgb = XGBClassifier(random_state =1)

xgb.fit(X_train, y_train)

y_pred = xgb.predict(X_test)

accuracy_score(y_test, y_pred)
from sklearn.ensemble import VotingClassifier

voting_clf = VotingClassifier(estimators = [('lr',lr),('knn',knn),('rf',rf),('gnb',gnb),('svc',svc),('xgb',xgb)], voting = 'soft')

voting_clf.fit(X_train, y_train)

y_pred = voting_clf.predict(X_test)

accuracy_score(y_test, y_pred)