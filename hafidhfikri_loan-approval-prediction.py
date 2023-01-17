#Import packages

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import cross_val_score

from sklearn.tree import DecisionTreeClassifier

from sklearn.neighbors import KNeighborsClassifier

from  sklearn import svm



#Read CSV data

data = pd.read_csv("../input/train_u6lujuX_CVtuZ9i (1).csv")
#preview data

data.head()
#Preview data information

data.info()
#Check missing values

data.isnull().sum()
# percent of missing "Gender" 

print('Percent of missing "Gender" records is %.2f%%' %((data['Gender'].isnull().sum()/data.shape[0])*100))
print("Number of people who take a loan group by gender :")

print(data['Gender'].value_counts())

sns.countplot(x='Gender', data=data, palette = 'Set2')
# percent of missing "Married" 

print('Percent of missing "Married" records is %.2f%%' %((data['Married'].isnull().sum()/data.shape[0])*100))
print("Number of people who take a loan group by marital status :")

print(data['Married'].value_counts())

sns.countplot(x='Married', data=data, palette = 'Set2')
# percent of missing "Dependents" 

print('Percent of missing "Dependents" records is %.2f%%' %((data['Dependents'].isnull().sum()/data.shape[0])*100))
print("Number of people who take a loan group by dependents :")

print(data['Dependents'].value_counts())

sns.countplot(x='Dependents', data=data, palette = 'Set2')
# percent of missing "Self_Employed" 

print('Percent of missing "Self_Employed" records is %.2f%%' %((data['Self_Employed'].isnull().sum()/data.shape[0])*100))
print("Number of people who take a loan group by self employed :")

print(data['Self_Employed'].value_counts())

sns.countplot(x='Self_Employed', data=data, palette = 'Set2')
# percent of missing "LoanAmount" 

print('Percent of missing "LoanAmount" records is %.2f%%' %((data['LoanAmount'].isnull().sum()/data.shape[0])*100))
ax = data["LoanAmount"].hist(density=True, stacked=True, color='teal', alpha=0.6)

data["LoanAmount"].plot(kind='density', color='teal')

ax.set(xlabel='Loan Amount')

plt.show()
# percent of missing "Loan_Amount_Term" 

print('Percent of missing "Loan_Amount_Term" records is %.2f%%' %((data['Loan_Amount_Term'].isnull().sum()/data.shape[0])*100))
print("Number of people who take a loan group by loan amount term :")

print(data['Loan_Amount_Term'].value_counts())

sns.countplot(x='Loan_Amount_Term', data=data, palette = 'Set2')
# percent of missing "Credit_History" 

print('Percent of missing "Credit_History" records is %.2f%%' %((data['Credit_History'].isnull().sum()/data.shape[0])*100))
print("Number of people who take a loan group by credit history :")

print(data['Credit_History'].value_counts())

sns.countplot(x='Credit_History', data=data, palette = 'Set2')
train_data = data.copy()

train_data['Gender'].fillna(train_data['Gender'].value_counts().idxmax(), inplace=True)

train_data['Married'].fillna(train_data['Married'].value_counts().idxmax(), inplace=True)

train_data['Dependents'].fillna(train_data['Dependents'].value_counts().idxmax(), inplace=True)

train_data['Self_Employed'].fillna(train_data['Self_Employed'].value_counts().idxmax(), inplace=True)

train_data["LoanAmount"].fillna(train_data["LoanAmount"].mean(skipna=True), inplace=True)

train_data['Loan_Amount_Term'].fillna(train_data['Loan_Amount_Term'].value_counts().idxmax(), inplace=True)

train_data['Credit_History'].fillna(train_data['Credit_History'].value_counts().idxmax(), inplace=True)
#Check missing values

train_data.isnull().sum()

train_data
#Convert some object data type to int64

gender_stat = {"Female": 0, "Male": 1}

yes_no_stat = {'No' : 0,'Yes' : 1}

dependents_stat = {'0':0,'1':1,'2':2,'3+':3}

education_stat = {'Not Graduate' : 0, 'Graduate' : 1}

property_stat = {'Semiurban' : 0, 'Urban' : 1,'Rural' : 2}



train_data['Gender'] = train_data['Gender'].replace(gender_stat)

train_data['Married'] = train_data['Married'].replace(yes_no_stat)

train_data['Dependents'] = train_data['Dependents'].replace(dependents_stat)

train_data['Education'] = train_data['Education'].replace(education_stat)

train_data['Self_Employed'] = train_data['Self_Employed'].replace(yes_no_stat)

train_data['Property_Area'] = train_data['Property_Area'].replace(property_stat)
#Preview data information

data.info()

data.isnull().sum()
#Separate feature and target

x = train_data.iloc[:,1:12]

y = train_data.iloc[:,12]



#make variabel for save the result and to show it

classifier = ('Gradient Boosting','Random Forest','Decision Tree','K-Nearest Neighbor','SVM')

y_pos = np.arange(len(classifier))

score = []
clf = GradientBoostingClassifier()

scores = cross_val_score(clf, x, y,cv=5)

score.append(scores.mean())

print('The accuration of classification is %.2f%%' %(scores.mean()*100))
clf = RandomForestClassifier(n_estimators=10)

scores = cross_val_score(clf, x, y,cv=5)

score.append(scores.mean())

print('The accuration of classification is %.2f%%' %(scores.mean()*100))
clf = DecisionTreeClassifier()

scores = cross_val_score(clf, x, y,cv=5)

score.append(scores.mean())

print('The accuration of classification is %.2f%%' %(scores.mean()*100))
clf = KNeighborsClassifier()

scores = cross_val_score(clf, x, y,cv=5)

score.append(scores.mean())

print('The accuration of classification is %.2f%%' %(scores.mean()*100))
clf  =  svm.LinearSVC(max_iter=5000)

scores = cross_val_score(clf, x, y,cv=5)

score.append(scores.mean())

print('The accuration of classification is %.2f%%' %(scores.mean()*100))
plt.barh(y_pos, score, align='center', alpha=0.5)

plt.yticks(y_pos, classifier)

plt.xlabel('Score')

plt.title('Classification Performance')

plt.show()