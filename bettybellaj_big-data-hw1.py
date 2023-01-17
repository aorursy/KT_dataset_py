# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
pd.options.display.float_format = '{:,.2f}'.format #to standardize the formatting
#Notes - how to format

#print(len(data)."{0:,}".format(len(data)), 'records are left')

#print('{0:,.0f}'.format(len(data)))

#print('there are {0:,.0f} records left'.format(len(data)))
import pandas as pd

data = pd.read_csv("../input/lending-club-loan-data/loan.csv", low_memory = False, skipinitialspace=True)

data = data[(data.loan_status == 'Fully Paid') | (data.loan_status == 'Default')] #truncate based on some conditions

data["target"] = (data.loan_status == 'Fully Paid') #add columns



#Notes: 

# data.target is the same with data['target']

# When based on conditions =='Fully Paid';1 condition-->中括号；multiple conditions --> [()|()]
#Notes - data slicing

data2 = data.iloc[5,1]

data2
#Q1

len(data)

print('there are {0:,.0f} records left'.format(len(data)))
len(data.columns)

print('there are {0:,.0f} features left'.format(len(data.columns)))
#teacher version

data.shape
#Q2

import matplotlib as mpl

import matplotlib.pyplot as plt

loan_amnt = data.loan_amnt

import seaborn as sns

sns.set()

ax = plt.subplot(211)

plt.hist(loan_amnt,bins=100,rwidth=2,color='#FF7F00',alpha=0.5)

plt.xlabel("Loan amount")

plt.ylabel("# of Loans")

plt.title('Loan Amount Overview')

ax.get_xaxis().set_major_formatter(mpl.ticker.FuncFormatter(lambda x, p: format(int(x), ',')))

ax.get_yaxis().set_major_formatter(mpl.ticker.FuncFormatter(lambda y,p: format(int(y), ',')))

loan_amnt.mean()

print('the mean of the loan amount is {0:,.2f}'.format(loan_amnt.mean()))
loan_amnt.median()

print('the meidan of the loan amount is {0:,.2f}'.format(loan_amnt.median()))
loan_amnt.std()

print('the std dev of the loan amount is {0:,.2f}'.format(loan_amnt.std()))
#teacher version

data.term.unique().tolist()
term = data.term

print(f'{term} length loans int rate mean is {data[data.term==term].int_rate.mean():.2f} and std dev is {data[data.term==term].int_rate.std():.2f}')
#Q3

ST_data = data.loc[data['term'] == '36 months']

#If two conditions asked: ST_data_1 = data.loc[(data['term'] == '36 months') & (data['loan_amnt'] >= 20000)]

r_ST = ST_data.int_rate

r_ST.mean()

print(f'the mean of the short-term interest rate is', r_ST.mean())

r_ST.std()

print('the std dev of the short-term interest rate is', r_ST.std())
LT_data = data.loc[data['term'] == '60 months']

r_LT = LT_data.int_rate

r_LT.mean()

print('the mean of the long-term interest rate is', r_LT.mean())

r_LT.std()

print('the standard deviation of the long-term interest rate is', r_LT.std())
x = data.groupby('term')

x['int_rate'].agg([np.mean,np.std])
term = data.term

r = data.int_rate

data.boxplot('int_rate',by=('term'),color='#8C4799',patch_artist=True)

#Q4

d1 = data.groupby('grade').mean() #group by grade, and all other variables show the mean values

d1
avg_int = data.groupby('grade')['int_rate'].mean()#group by grade, and select the all other variables show the mean values

print('Debt grade',avg_int.idxmax(),'has an average interest rate of','{0:,.2f}'.format(avg_int.max()))
#Q5 (realized yield)

default = (1-data.target.mean())*100

print('{0:,.4%}'.format(default))

print('a default rate of {0:,.4%}'.format(default))

#Questions: print('a default rate of {(1-data.target.mean())*100:.2f}')
s1 = data.groupby('grade')['total_pymnt'].sum()

s2 = data.groupby('grade')['funded_amnt'].sum()

Rlz_yield = (s1/s2-1)*100

Rlz_yield

print('the max realized yield is',Rlz_yield.max(),'under Grade',Rlz_yield.idxmax())
#Q6 application type

len(data.application_type.unique())
Indiv_gp = data.loc[data['application_type'] == 'Individual']

print('{0:,.0f}'.format(len(Indiv_gp)))
Joint_gp = data.loc[data['application_type'] == 'Joint App']

print('{0:,.0f}'.format(len(Joint_gp)))

IJratio = len(Joint_gp)/len(Indiv_gp)

IJratio 
#Q7 dummy settings 

X = pd.get_dummies(data[['loan_amnt','funded_amnt','funded_amnt_inv','term','int_rate','emp_length','addr_state','verification_status','purpose','policy_code']])

print(f'Matrix width is {X.shape[1]}')

y=data.target

#Use these inputs to predict the fully-paid loans
#Q8 train_test_split

import numpy as np

from sklearn.model_selection import train_test_split

X_train,X_test,y_train, y_test = train_test_split(X,y,train_size=0.33,random_state=42)

X_train.shape
#Q9 train_test_split

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score

clf = RandomForestClassifier(n_estimators=100, max_depth=4,random_state=42)

clf.fit(X_train,y_train)

y_pred = clf.predict(X_test)

y_pred

print(f'Random Forest Accuracy {accuracy_score(y_test,y_pred)*100:.2f}')

#Q10

y_pred1 = np.ones(y_test.shape)

print(f'All Repayment Accuracy {accuracy_score(y_test,y_pred)*100:.2f}')