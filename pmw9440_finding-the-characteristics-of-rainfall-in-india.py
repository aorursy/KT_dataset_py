###importing modules
import numpy as np #linear algebra
import pandas as pd #data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from subprocess import check_output

### import data
data = pd.read_csv("../input/rainfall in india 1901-2015.csv",sep=",")
data.info()
data.head()
data.describe()

# Any results you write to the current directory are saved as output.
###How many subdivisions are in the dataset?
subdivisions = data['SUBDIVISION'].unique()
numberofsubdivisions = subdivisions.size
print('The number of Subdivisions: ' + str(numberofsubdivisions))
### Finding out missing data cells in dataset
data.isnull().sum()

##filling the missing values with the means of each column
data['JAN'].fillna((data['JAN'].mean()), inplace=True)
data['FEB'].fillna((data['FEB'].mean()), inplace=True)
data['MAR'].fillna((data['MAR'].mean()), inplace=True)
data['APR'].fillna((data['APR'].mean()), inplace=True)
data['MAY'].fillna((data['MAY'].mean()), inplace=True)
data['JUN'].fillna((data['JUN'].mean()), inplace=True)
data['JUL'].fillna((data['JUL'].mean()), inplace=True)
data['AUG'].fillna((data['AUG'].mean()), inplace=True)
data['SEP'].fillna((data['SEP'].mean()), inplace=True)
data['OCT'].fillna((data['OCT'].mean()), inplace=True)
data['NOV'].fillna((data['NOV'].mean()), inplace=True)
data['DEC'].fillna((data['DEC'].mean()), inplace=True)
data['Jan-Feb'].fillna((data['Jan-Feb'].mean()), inplace=True)
data['Mar-May'].fillna((data['Mar-May'].mean()), inplace=True)
data['Jun-Sep'].fillna((data['Jun-Sep'].mean()), inplace=True)
data['Oct-Dec'].fillna((data['Oct-Dec'].mean()), inplace=True)
data['ANNUAL'].fillna((data['ANNUAL'].mean()), inplace=True)

data.isnull().sum()

fig = plt.figure(figsize=(15,8))
ax = fig.add_subplot(111)
data.groupby('SUBDIVISION').mean().sort_values(by='ANNUAL', ascending=False)['ANNUAL'].plot('bar',color='b', width=0.3, title='Subdivision wise Average Annual Rain fall',fontsize=20)
print(data.groupby('SUBDIVISION').mean().sort_values(by='ANNUAL', ascending=False)['ANNUAL'])
plt.xticks(rotation = 90)
ax.xaxis.label.set_fontsize(20)
ax.yaxis.label.set_fontsize(20)
plt.ylabel('Average Annual Rainfall (mm)')

data[['JAN', 'FEB', 'MAR', 'APR','MAY', 'JUN', 'AUG', 'SEP', 'OCT','NOV','DEC']].mean().plot('bar',figsize=(13,8))

print(data[['JAN', 'FEB', 'MAR', 'APR','MAY', 'JUN', 'AUG', 'SEP', 'OCT','NOV','DEC']].mean())

fig = plt.figure(figsize = (16, 10))
ax = fig.add_subplot(111)
dfg =data.groupby('YEAR').mean()['ANNUAL']
dfg.plot('line', title = 'Overall Rainfall in Each Year', fontsize =20)
plt.ylabel("Overall Rainfall(mm)")
data['MA10'] = data.groupby('YEAR').mean()['ANNUAL'].rolling(10).mean()
data.MA10.plot()
#Trend test: Kendall-tau test
import scipy.stats as stats
rainfall_rank = data.groupby('YEAR').mean()['ANNUAL']
rainfall_rank = pd.DataFrame({'Year':rainfall_rank.index, 'list':rainfall_rank.values})
rainfall_rank['rank'] = rainfall_rank['list'].rank(ascending = False)
t = list(range(1, len(rainfall_rank['list'])+1))
tau, p_value = stats.kendalltau(t, list(rainfall_rank['rank']))
print("tau: " + str(tau))
print("p-value: " + str(p_value))