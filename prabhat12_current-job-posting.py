# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
data=pd.read_csv('../input/nyc-jobs.csv')

data.head()
label=['min_salary','max_salary']

def salary_for_agencies(agency,salary_freq):

    try:

        posting_type_data=data[data['Salary Frequency']==salary_freq]

        agency_data=posting_type_data[posting_type_data['Agency']==agency]

        min_salary=agency_data['Salary Range From']

        max_salary=agency_data['Salary Range To']

        avg_max_salary=sum(max_salary)/len(max_salary)

        print("Count of "+salary_freq+" Job poosition in the agency:",len(max_salary))

        avg_min_salary=sum(min_salary)/len(min_salary)

        print("Minimum Avg salary for agency:",avg_min_salary)

        print("Maximum Avg salary for agency:",avg_max_salary)

        plt.bar(label,[avg_min_salary,avg_max_salary])

        plt.title("Average Min Max "+salary_freq+" Salary for "+agency)

        plt.show()

    except:

        print("No data")
salary_for_agencies('DEPARTMENT OF BUSINESS SERV.','Annual')
salary_for_agencies('DEPARTMENT OF BUSINESS SERV.','Hourly')
salary_for_agencies('NYC HOUSING AUTHORITY','Annual')
from numpy import loadtxt

from xgboost import XGBClassifier

from xgboost import plot_importance

from matplotlib import pyplot

from sklearn import preprocessing
data=data.dropna(subset = ['Job Category'])

print(data.shape)

data.head()
X_temp = data.iloc[:,[1,2,4,5,7,12]]

y = data.iloc[:,[10,11]]



X=X_temp.apply(preprocessing.LabelEncoder().fit_transform)

X.head()
X['# Of Positions']=data.iloc[:,3]



print(X.shape)

X.head()

model = XGBClassifier()

model.fit(X, y.iloc[:,0])

# plot feature importance

plot_importance(model)

pyplot.show()
from sklearn.model_selection import train_test_split



X_train,X_test,Y_train,Y_test=train_test_split(X.iloc[:,[0,2,3,4,6]],y)
from sklearn.linear_model import LinearRegression



lr=LinearRegression()

lr.fit(X_train,Y_train)

predict=lr.predict(X_test)
lr.score(X_test,Y_test)
from sklearn.ensemble import RandomForestRegressor

# Instantiate model with 1000 decision trees

rf = RandomForestRegressor(n_estimators = 1000, random_state = 42)

# Train the model on training data

rf.fit(X_train, Y_train);
rf.score(X_test,Y_test)