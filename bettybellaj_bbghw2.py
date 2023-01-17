# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
#Basic settings

import pandas as pd

data = pd.read_csv("../input/modeltrap/train.csv", low_memory = False, skipinitialspace=True)

datatest = pd.read_csv("../input/modeltrap/test.csv", low_memory = False, skipinitialspace=True)

pd.options.display.float_format = '{:,.2f}'.format #to standardize the formatting

#Q1 - % of training set loans are in default

default = data[data.default == True]

default_rate = len(default)/len(data)

print('the overall default rate is {0:,.2%}'.format(default_rate))
#Q2 - Which ZIP code has the highest default rate in the training dataset?

ZIP = data.groupby(['ZIP','default'])['minority'].count()

a = list(set(data['ZIP']))

a.sort()

for zipcode in a:

    print(zipcode, ZIP.get((zipcode, True), default=0) / len(data.loc[data.ZIP == zipcode]))
g1 = data.loc[(data.ZIP == 'MT01RA') & (data.default == True)]

defr_g1 = len(g1) / len (data.loc[data.ZIP == 'MT01RA'])

g2 = data.loc[(data.ZIP == 'MT04PA') & (data.default == True)]

defr_g2 = len(g2) / len (data.loc[data.ZIP == 'MT04PA'])

g3 = data.loc[(data.ZIP == 'MT12RA') & (data.default == True)]

defr_g3 = len(g3) / len (data.loc[data.ZIP == 'MT12RA'])

g4 = data.loc[(data.ZIP == 'MT15PA') & (data.default == True)]

defr_g4 = len(g4) / len (data.loc[data.ZIP == 'MT15PA'])



print (defr_g1,defr_g2,defr_g3,defr_g4)
#Q3 default rate in the training set for the first year for which you have data?

Year1 = data[data.year == 1]

Year1_default = Year1[Year1.default == True]

Year1_default_rate = len(Year1_default) / len(Year1)

print('the Year-1 default rate is {0:,.2%}'.format(Year1_default_rate))
#Q4 the correlation between age and income in the training dataset?

cor_ageincome = data['age'].corr(data['income'])

print('the cor betw age and income is {0:,.2%}'.format(cor_ageincome))
#Q5 The in-sample accuracy? - find the accuracy score of the fitted model for predicting the outcomes using the whole training dataset

import numpy as np

from sklearn.model_selection import train_test_split

X_train = pd.get_dummies(data[['loan_size','payment_timing','education','occupation','income','job_stability','ZIP','rent']])

y_train = data["default"] 

X_test = pd.get_dummies(datatest[['loan_size','payment_timing','education','occupation','income','job_stability','ZIP','rent']])

y_test = datatest["default"]



from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score

clf = RandomForestClassifier(n_estimators=100, max_depth=4,random_state=42, oob_score=True)

clf.fit(X_train,y_train)

y_pred = clf.predict(X_test)

accuracy_score(y_test,y_pred)

#Q5 The in-sample accuracy? - find the accuracy score of the fitted model for predicting the outcomes using the whole training dataset

import numpy as np

from sklearn.model_selection import train_test_split

X_train = pd.get_dummies(data[['loan_size','payment_timing','education','occupation','income','job_stability','ZIP','rent']])

y_train = data["default"] 

X_test = pd.get_dummies(datatest[['loan_size','payment_timing','education','occupation','income','job_stability','ZIP','rent']])

y_test = datatest["default"]



from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score

clf = RandomForestClassifier(n_estimators=100, max_depth=4,random_state=42, oob_score=True)

trained_model = clf.fit(X_train,y_train)

#trained_model.fit(X_train,y_train)

y_pred = clf.predict(X_train)

accuracy_score(y_train,y_pred)
print(y_pred)
#Kevin notes1

count_True = 0

count_False = 0

for i in y_test:

    if i == True:

        count_True += 1

    else:

        count_False +=1

print(count_True, count_False)
#Q6 Out-of-bag score of the test

clf.oob_score_
#Q7 out-of-sample accuracy

y_oopred = clf.predict(X_test)

accuracy_score(y_test,y_oopred)
#Q8 The pred. avg default prob for all non-minority members in the test set

Nminor = datatest[datatest.minority == 0] #Non-minority group in the test dataset

Nminor_d = Nminor[Nminor.default == True] 

ENminor_drate = len(Nminor_d) / len(Nminor)

ENminor_drate
#Q9 The pred. avg default prob for all minority members in the test set?

Minor = datatest[datatest.minority == 1] #Minority group in the test dataset

Minor_d = Minor[Minor.default == True] 

EMinor_drate = len(Minor_d) / len(Minor)

EMinor_drate
#Q11 Has the loan granting scheme achieved demographic parity? 

Female = datatest[datatest.sex == 1]

Male = datatest[datatest.sex == 0]



Female.y_pred.value_counts(True)

y_test
data