import numpy as np

import pandas as pd
data = pd.read_csv("../input/churn-modelling/Churn_Modelling.csv").dropna()

data = data.drop(labels=["CustomerId", "Surname", "Geography", "Gender", "RowNumber"], axis=1)

data.head()
data.corr()
active = data["IsActiveMember"].value_counts()

print( str(float(active[1])*100/len(data))+" persents of members are active." )
crCard = data["HasCrCard"].value_counts()

print( str(float(crCard[1])*100/len(data))+" persents of members have a credit card." )
salary = data["EstimatedSalary"]

print( "Estimated salary: before "+str(max(0, salary.mean()-salary.std()))+" and "+str(salary.mean()+salary.std())+"." )