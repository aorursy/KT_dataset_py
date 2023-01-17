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
df=pd.read_csv("../input/loan_data_set.csv")
df.head(10)
df.info()
df.Gender.value_counts()
#Fill nan with male 

df.Gender=df.Gender.fillna('Male')
df.Gender.value_counts()
#similarly filled other categorical variable ,married , dependents,

# Education,Self_Employed ,Credit_History  , Property_Area      

df.Married=df.Married.fillna('Yes')

df.Dependents=df.Dependents.fillna('0')

df.Self_Employed=df.Self_Employed.fillna('No')

df.Credit_History=df.Credit_History.fillna(1.0)

df.Credit_History.value_counts()

# now for numeric value

df.describe()
#Here all numeric colums show and its numeric value such mean, std etc

# fill na with mean value

df.LoanAmount=df.LoanAmount.fillna(146.412162)

df.Loan_Amount_Term=df.Loan_Amount_Term.fillna(342.00000)
df.info()
# now seperate categorical variable and non cate.. variable
