# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
import pandas as pd

import numpy as np

import time

import random

from sklearn import preprocessing



# machine learning

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC, LinearSVC

from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.linear_model import Perceptron

from sklearn.linear_model import SGDClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import classification_report



pd.options.display.max_columns = 999

pd.options.display.max_rows = 999





pd.options.display.max_columns = 999
data = pd.read_csv('../input/loan.csv', low_memory=False)

data.drop(['id', 'member_id', 'emp_title'], axis=1, inplace=True)
data["last_credit_pull_d_month"] = data["last_credit_pull_d"].str.split("-", 1, True)[0]

data["last_credit_pull_d_year"] = data["last_credit_pull_d"].str.split("-", 1, True)[1]

data["earliest_cr_line_month"] = data["earliest_cr_line"].str.split("-", 1, True)[0]

data["earliest_cr_line_year"] = data["earliest_cr_line"].str.split("-", 1, True)[1]

data["last_pymnt_d_month"] = data["last_pymnt_d"].str.split("-", 1, True)[0]

data["last_pymnt_d_year"] = data["last_pymnt_d"].str.split("-", 1, True)[1]

data["next_pymnt_d_month"] = data["next_pymnt_d"].str.split("-", 1, True)[0]

data["next_pymnt_d_year"] = data["next_pymnt_d"].str.split("-", 1, True)[1]

data["issue_d_month"] = data["issue_d"].str.split("-", 1, True)[0]

data["issue_d_year"] = data["issue_d"].str.split("-", 1, True)[1]
a = {"Jan":1,"Feb":2,"Mar":3,"Apr":4,"May":5,"Jun":6,"Jul":7,"Aug":8,"Sep":9,"Oct":10,"Nov":11,"Dec":12}

data["last_credit_pull_d_month"] = data["last_credit_pull_d_month"].map(a)

b = {"Jan":1,"Feb":2,"Mar":3,"Apr":4,"May":5,"Jun":6,"Jul":7,"Aug":8,"Sep":9,"Oct":10,"Nov":11,"Dec":12}

data["earliest_cr_line_month"] = data["earliest_cr_line_month"].map(b)

c = {"Jan":1,"Feb":2,"Mar":3,"Apr":4,"May":5,"Jun":6,"Jul":7,"Aug":8,"Sep":9,"Oct":10,"Nov":11,"Dec":12}

data["earliest_cr_line_year"] = data["earliest_cr_line_year"].map(c)

d = {"Jan":1,"Feb":2,"Mar":3,"Apr":4,"May":5,"Jun":6,"Jul":7,"Aug":8,"Sep":9,"Oct":10,"Nov":11,"Dec":12}

data["last_pymnt_d_month"] = data["last_pymnt_d_month"].map(d)

e = {"Jan":1,"Feb":2,"Mar":3,"Apr":4,"May":5,"Jun":6,"Jul":7,"Aug":8,"Sep":9,"Oct":10,"Nov":11,"Dec":12}

data["next_pymnt_d_month"] = data["next_pymnt_d_month"].map(e)

f = {"Jan":1,"Feb":2,"Mar":3,"Apr":4,"May":5,"Jun":6,"Jul":7,"Aug":8,"Sep":9,"Oct":10,"Nov":11,"Dec":12}

data["issue_d_month"] = data["issue_d_month"].map(f)
data.drop(["last_credit_pull_d","earliest_cr_line", "last_pymnt_d","issue_d","next_pymnt_d"],axis = 1, inplace = True)
data.replace('n/a', np.nan,inplace=True)

data.emp_length.fillna(value=0,inplace=True)



data['emp_length'].replace(to_replace='[^0-9]+', value='', inplace=True, regex=True)

data['emp_length'] = data['emp_length'].astype(int)



data['term'] = data['term'].apply(lambda x: x.lstrip())
data=data.dropna(axis=1, how = "all")

data = data.loc[:,data.apply(pd.Series.nunique) != 1]
data['Default_Binary'] = int(0)

for index, value in data.loan_status.iteritems():

    if value == 'Default':

        data.set_value(index,'Default_Binary',int(1))

    if value == 'Charged Off':

        data.set_value(index, 'Default_Binary',int(1))

    if value == 'Late (31-120 days)':

        data.set_value(index, 'Default_Binary',int(1))    

    if value == 'Late (16-30 days)':

        data.set_value(index, 'Default_Binary',int(1))

    if value == 'Does not meet the credit policy. Status:Charged Off':

        data.set_value(index, 'Default_Binary',int(1)) 

Y = data["Default_Binary"]
a = {"A":1,"B":2,"C":3,"D":4,"E":5,"F":6}

data["grade"] = data["grade"].map(a)



b = {"INDIVIDUAL":1,"0":0}

data["application_type"] = data["application_type"].map(b)
data = pd.get_dummies(data, columns=["loan_status", "purpose",'home_ownership', 'verification_status','sub_grade',"purpose","addr_state"], drop_first=True)
data.drop(['home_ownership', 'verification_status','sub_grade',"purpose","addr_state", "loan_status", "zip_code"], axis = 1, inplace=True)
data.drop(["id","url", "desc","title","pymnt_plan","term","int_rate","revol_util","emp_title","initial_list_status"],axis = 1, inplace = True)
data.fillna(value=-1,inplace=True)
X = data.drop("Default_Binary", axis=1)
Y.head(2)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state=888)
logreg = LogisticRegression()

logreg.fit(X_train, y_train)

Y_pred = logreg.predict(X_test)

acc_log = round(logreg.score(X_train, y_train) * 100, 2)

acc_log
decision_tree = DecisionTreeClassifier()

decision_tree.fit(X_train, y_train)

Y_pred = decision_tree.predict(X_test)

print(classification_report(y_test,Y_pred))
random_forest = RandomForestClassifier(n_estimators=100)

random_forest.fit(X_train, y_train)

Y_pred = random_forest.predict(X_test)

random_forest.score(X_train, y_train)

print(classification_report(y_test,Y_pred))
data.to_csv("111.csv")