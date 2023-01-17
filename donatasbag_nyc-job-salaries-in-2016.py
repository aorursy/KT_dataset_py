import pandas as pd

import numpy as np

import matplotlib.pyplot as plt
data = pd.read_csv('../input/jobpostings.csv')

data.head(1)
data.shape
data.columns
data.info()


data.dropna(subset=['Job Category'], inplace = True)

assert 1==1

data[['Job Category']].info()

data.describe()
a = data['Job Category']

b = data['Salary Range From']

c = data['Salary Range To']

data = pd.concat([a,b,c], axis=1)

data.index = np.arange(1, len(data)+1)

data.head(3)
pd.options.display.max_rows = 999

dt = data[(data['Salary Range From']>1000)]



print()

print("Maximum salary")

print(dt.max())



print()

print()

print("Minimum salary")

print(dt.min())

print()





dt = dt.groupby("Job Category").mean()

dt.reset_index(inplace=True)

dt.iloc[0:32, 0] = 'Administration & Human Resources'

dt.iloc[32:38, 0] = 'Building Operations & Maintenance'

dt.iloc[38:42, 0] = 'Clerical & Administrative Support'

dt.iloc[42:52, 0] = 'Communications & Intergovernmental Affairs'

dt.iloc[52:54, 0] = 'Community & Business Services'

dt.iloc[54:77, 0] = 'Constituent Services & Community Programs'

dt.iloc[77:90, 0] = 'Engineering, Architecture, & Planning'

dt.iloc[90:100, 0] = 'Finance, Accounting, & Procurement'

dt.iloc[100:114, 0] = 'Health'

dt.iloc[114:124, 0] = 'Legal Affairs'

dt.iloc[125:131, 0] = 'Legal Affairs'

dt.iloc[131:133, 0] = 'Public Safety, Inspections, & Enforcement'

dt.iloc[134:138, 0] = 'Technology, Data & Innovation'

dt = dt.groupby("Job Category").mean()



print()

print()

print("Salaries by Job categories - to max")

dt1 = dt.sort_values(by=['Salary Range To'], ascending=False)

dt1





dt1.plot.bar(stacked=True);


rep_plot = dt1.sort_values("Job Category").mean().plot(kind='bar')

rep_plot.set_xlabel("Job Category")

rep_plot.set_ylabel("Salary Range To")
pd.options.display.max_columns = 999

data2 = pd.read_csv('../input/jobpostings.csv')

data2.sort_values(by='Salary Range To', ascending=False).head(2)

pd.options.display.max_columns = 999

data2 = pd.read_csv('../input/jobpostings.csv')

data2.sort_values(by='Salary Range To', ascending=True).head(2)