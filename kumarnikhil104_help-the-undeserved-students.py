import numpy as np 
import pandas as pd 


import matplotlib
import matplotlib.pyplot as plt # for plotting

import warnings
warnings.filterwarnings('ignore')
import os
print(os.listdir("../input"))

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
%matplotlib inline
school_explorer = pd.read_csv('../input/2016 School Explorer.csv')
registrations = pd.read_csv('../input/D5 SHSAT Registrations and Testers.csv')
print("2016 School Explorer.csv shape is : {}".format(school_explorer.shape))
print("D5 SHSAT Registrations and Testers.csv shape is : {}".format(registrations.shape))
## Taking a peak of the data in the csv's
school_explorer.head(5)
registrations.head(10)
registrations['Grade level'].value_counts(normalize=1).hist()
registrations.groupby(['School name','Year of SHST']).count()#['Number of students who registered for the SHSAT'].count()

#(registrations['Number of students who took the SHSAT']/registrations['Number of students who registered for the SHSAT']).hist(column =)
table = pd.pivot_table(registrations, values=['Number of students who registered for the SHSAT','Number of students who took the SHSAT'], index=['School name'],columns=['Year of SHST'], aggfunc=np.sum)
table


