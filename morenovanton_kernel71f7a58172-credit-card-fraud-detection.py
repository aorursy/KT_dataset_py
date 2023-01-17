import numpy as np 

import pandas as pd 

import seaborn as sns
creditcard = pd.read_csv('../input/creditcardfraud/creditcard.csv')
creditcard.head()
creditcard.describe()
creditcard_Class0 = creditcard.loc[creditcard.Class == 0]

creditcard_Class1 = creditcard.loc[creditcard.Class == 1]


creditcard_Cl = creditcard.groupby('Class').aggregate({'Class': 'count'}).rename(columns={'Class': 'Class_count'}).reset_index()

print(creditcard_Cl)

sns.barplot('Class', 'Class_count', data = creditcard_Cl)
creditcard_Class1.head(2)
import pandas_profiling

creditcard_Class1.profile_report()