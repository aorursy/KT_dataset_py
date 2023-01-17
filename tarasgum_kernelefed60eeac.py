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
#import pandas, call them "pd"

import pandas as pd

# create a variable data, using pandas command "readcsv", read the loan portfolio)

data = pd.read_csv("../input/loan.csv", low_memory = False)

data = data[(data.loan_status == 'Fully Paid') | (data.loan_status == 'Default')]

data['target'] = (data.loan_status == 'Fully Paid')
#counting observations - 1041983

data[['loan_amnt']].count()

data.head()

data.hist(column= 'loan_amnt')



print(

data.loan_amnt.median(),

data.loan_amnt.mean(),

data.loan_amnt.max(),

data.loan_amnt.std(),

)

data.term.unique()
data_36 = data[(data.term == ' 36 months')]

data_60 = data[(data.term == ' 60 months')]

print(

data_36.int_rate.mean(),

data_36.int_rate.std(),

data_60.int_rate.mean(),

data_60.int_rate.std(),

)

data.boxplot(column='int_rate', by='term')
#print (data.grade)

data.boxplot(column = 'int_rate', by = 'grade')
data.int_rate[(data.grade == 'G')].mean()