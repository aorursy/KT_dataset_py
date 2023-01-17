# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import pandas as pd

from sklearn.preprocessing import LabelEncoder

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.preprocessing import StandardScaler

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
data = pd.read_csv('../input/HR_comma_sep.csv')

data.head()
data.salary.value_counts()
data.sales.value_counts()
data.left.value_counts()
data.describe()
data.info()
plt.figure(figsize=(12,8))

sns.set(font_scale=1.5)

sns.countplot(x = "sales", data = data)

plt.xlabel('Department')

plt.ylabel('Number of Employees')

plt.title('No. of Employees in each department')

plt.xticks(rotation="vertical")

plt.show()
sns.countplot(x = 'salary',data=data,hue='left')

plt.show()
count = data.salary.value_counts(normalize=True)

sns.barplot(count.index,count.values, color = 'blue')

plt.ylabel('Proportion of Employees')

plt.xlabel('Salary')

plt.show()
sns.stripplot(x = 'salary', y='average_montly_hours', data=data, jitter = True, hue= 'left')

plt.show()
sns.factorplot(x = 'left', y='average_montly_hours',data=data, hue='salary')

plt.show()
sns.barplot(x = 'number_project', y='left', data=data)

plt.show()
plt.figure(figsize=(12,8))

sns.violinplot(x = 'left', y = 'satisfaction_level', data=data)

plt.show()
sns.barplot(x='time_spend_company', y='left', data=data)

plt.show()

print(data.time_spend_company.value_counts())
left = data[data.left==1]

notleft = data[data.left==0]

f,a = plt.subplots(1,2,sharey=True)

a[0].hist(x='Work_accident', bins=3 ,data=left)

a[0].set_title('left')

a[0].set_xlabel('Work Accident')

a[0].set_ylabel('Frequency')

a[1].hist(x='Work_accident', bins=3, data=notleft)

a[1].set_title('Not left')

a[1].set_xlabel('Work Accident')



print('Work Accident')

print(data.Work_accident.value_counts())

plt.show()
sns.countplot('promotion_last_5years', data=data)

plt.show()

f,a = plt.subplots(1, 2, sharey= True)

a[0].hist(x='promotion_last_5years', data=left)

a[0].set_title('Left')

a[0].set_xlabel('Promotion in lst 5 years')

a[0].set_ylabel('Frequency')

a[1].hist(x='promotion_last_5years', data=notleft)

a[1].set_title('Not Left')

a[1].set_xlabel('Promotion in lst 5 years')

plt.show()
sns.violinplot(x='left', y='last_evaluation', data=data)

plt.show()
le = LabelEncoder()

data.sales = le.fit_transform(data.sales)

data.left = le.fit_transform(data.left)

data.salary = le.fit_transform(data.salary)

y = data.left
data.dtypes
corr = data.corr()

sns.heatmap(corr, xticklabels=corr.columns.values, yticklabels=corr.columns.values)

plt.show()
data.drop(['left'],inplace=True,axis=1)

sc = StandardScaler()

data = sc.fit_transform(data)
from sklearn.cross_validation import cross_val_score

def estimating(estimator1,dat,label):

    accuracies = cross_val_score(estimator = estimator1, X = dat, y = label, cv = 5)

    print(accuracies)

    print(accuracies.mean())
from sklearn.neighbors import KNeighborsClassifier

classifier1 = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)

estimating(classifier1,data,y)
from sklearn.svm import SVC

classifier2 = SVC(kernel = 'rbf', random_state = 0)

estimating(classifier2,data,y)
from sklearn.naive_bayes import GaussianNB

classifier3 = GaussianNB()

estimating(classifier3,data,y)
from sklearn.ensemble import RandomForestClassifier

classifier4 = RandomForestClassifier(n_estimators = 100, criterion = 'entropy', random_state = 0)

estimating(classifier4,data,y)