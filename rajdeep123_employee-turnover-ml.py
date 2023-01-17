# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import pandas as pd

hr = pd.read_csv('../input/HR_comma_sep.csv')

col_names = hr.columns.tolist()

print("Column names:")

print(col_names)

print("\nSample data:")

hr.head()
hr=hr.rename(columns = {'sales':'department'})
hr.isnull().any()
hr.dtypes
hr.shape
hr['department'].unique()
import numpy as np

hr['department']=np.where(hr['department'] =='support', 'technical', hr['department'])

hr['department']=np.where(hr['department'] =='IT', 'technical', hr['department'])
hr['department'].unique()
hr['left'].value_counts()
hr.groupby('left').mean()
hr.groupby('department').mean()
hr.groupby('salary').mean()
%matplotlib inline

import matplotlib.pyplot as plt

pd.crosstab(hr.department,hr.left).plot(kind='bar')

plt.xlabel('Department')

plt.ylabel('Frequency of Turnover')

plt.savefig('department_bar_chart')
table=pd.crosstab(hr.salary, hr.left)

table.div(table.sum(1).astype(float), axis=0).plot(kind='bar', stacked=True)

plt.title('Stacked Bar Chart of Salary Level vs Turnover')

plt.xlabel('Salary Level')

plt.ylabel('Proportion of Employees')

plt.savefig('salary_bar_chart')
num_bins = 10

hr.hist(bins=num_bins, figsize=(20,15))

plt.savefig("hr_histogram_plots")

plt.show()
cat_vars=['department','salary']

for var in cat_vars:

    cat_list='var'+'_'+var

    cat_list = pd.get_dummies(hr[var], prefix=var)

    hr1=hr.join(cat_list)

    hr=hr1
hr.drop(hr.columns[[8, 9]], axis=1, inplace=True)

hr.columns.values
hr_vars=hr.columns.values.tolist()

y=['left']

X=[i for i in hr_vars if i not in y]

print(X)
from sklearn.feature_selection import RFE

from sklearn.linear_model import LogisticRegression

model = LogisticRegression()

rfe = RFE(model, 10)

rfe = rfe.fit(hr[X], hr[y])

print(rfe.support_)

print(rfe.ranking_)
cols=['satisfaction_level', 'last_evaluation', 'time_spend_company', 'Work_accident', 'promotion_last_5years', 

      'department_RandD', 'department_hr', 'department_management', 'salary_high', 'salary_low'] 

X=hr[cols]

y=hr['left']


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

from sklearn.linear_model import LogisticRegression

from sklearn import metrics

logreg = LogisticRegression()

logreg.fit(X_train, y_train)
from sklearn.metrics import accuracy_score

print('Logistic regression accuracy: {:.3f}'.format(accuracy_score(y_test, logreg.predict(X_test))))
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier()

rf.fit(X_train, y_train)
print('Random Forest Accuracy: {:.3f}'.format(accuracy_score(y_test, rf.predict(X_test))))
from sklearn.svm import SVC

svc = SVC()

svc.fit(X_train, y_train)

print('Support vector machine accuracy: {:.3f}'.format(accuracy_score(y_test, svc.predict(X_test))))
from sklearn.metrics import classification_report

print(classification_report(y_test, rf.predict(X_test)))
y_pred = rf.predict(X_test)

from sklearn.metrics import confusion_matrix

import seaborn as sns

forest_cm = metrics.confusion_matrix(y_pred, y_test, [1,0])

sns.heatmap(forest_cm, annot=True, fmt='.2f',xticklabels = ["Left", "Stayed"] , yticklabels = ["Left", "Stayed"] )

plt.ylabel('True class')

plt.xlabel('Predicted class')

plt.title('Random Forest')

plt.savefig('random_forest')