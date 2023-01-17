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
data1 = pd.read_csv(r'/kaggle/input/icc-test-cricket-runs/ICC Test Batting Figures.csv',encoding='latin1')
print(data1.shape)
data1.head()

data1.iloc[2113]
data2 = pd.read_excel('/kaggle/input/icc-test-cricket-runs/ICC Test Bat 3001.xlsx')
print(data2.shape)
data2.head()
Players = pd.DataFrame(data1)
Players.head()
Players.info()
Players.columns
Players.describe()
Players.isnull().sum()
# First change the data types for the columns
transform = ['Inn', 'NO', 'Runs', 'HS', 'Avg', '100', '50','0']
Players['HS'] = Players['HS'].str.replace('*', '')
Players[transform] = Players[transform].replace('-','0')
Players[transform] = Players[transform].astype(float,errors='ignore')
Players.head()
Players['Player'] = Players['Player'].astype(str)
Players['Span'] = Players['Span'].astype(str)
Players.info()
def teams(x):
    t = []
    X_ = x.split('(')
    name = X_[0]
#     out.append(name)
    teams = X_[-1]
    teams = teams.replace(')','')
    teams = (teams.split('/'))
    for team in teams:
        if team == 'ICC':
            continue
        else:
            t.append(team)
    country = t[-1]
    return name, country
temp = Players['Player'].apply(teams)
temp
names = []
country = []
for t in temp:
    names.append(t[0][:-1])
    country.append(t[1])
Players['Name'] = pd.Series(names)
Players['Country'] = pd.Series(country)
Players.head()
def start_end(x):
    year = x.split('-')
    year = pd.to_datetime(year)
    return year
year = Players['Span'].apply(start_end)
year
debut = []
last = []
for y in year:
    debut.append(y[0].year)
    last.append(y[1].year)
Players['debut_year'] = debut
Players['last_year'] = last
Players.head()
Players.drop('Span', axis = 1, inplace = True)
Players.drop('Player', axis = 1, inplace = True)
Players.head()
columns = ['Name', 'Country', 'debut_year', 'last_year','Mat', 'Inn', 'NO', 'Runs', 'HS', 'Avg', '100', '50', '0','Player Profile']
Players = Players[columns]
Players.head()
import matplotlib.pyplot as plt
import seaborn as sns
Players.Country.value_counts()
plt.rcParams['figure.figsize'] = (20,9)

sns.countplot(Players['Country'],palette = 'gnuplot')

plt.title('Players per Country', fontweight = 30, fontsize =20)
plt.xticks(rotation = 90)
plt.show()
Players_top = Players[(Players['Avg']>=50) & (Players['Runs']>5000) & (Players['100']>15)]
print(len(Players_top))
Players_top
Avg_weight = Players_top['Avg'].pow(2.5)
run_weight = Players_top['Runs'].mul(Players_top['100'])
# Players_top['weight_factor'] = Players_top['Runs'] * Players_top['Avg'] * Players_top['100'] / 1000
Players_top['weight_factor'] = Avg_weight.mul(run_weight) / 100000000
Players_top = Players_top.sort_values(by = 'weight_factor', ascending = False)
Players_top[:10]
import matplotlib.pyplot as plt

X = np.arange(10)
plt.bar(X + 0.00, Players_top[:10]['Avg'], color = 'b', width = 0.25, label = 'Average')
plt.bar(X + 0.25, Players_top[:10]['100'], color = 'g', width = 0.25, label = '100s')
plt.xticks( X,Players_top[:10]['Name'] )    

plt.show()
X = np.arange(10)
plt.bar(X + 0.00, Players_top[:10]['Runs'], color = 'b', width = 0.25, label = 'Average')
plt.ylabel('Runs')
plt.xlabel('Player')
plt.xticks( X,Players_top[:10]['Name'] )
plt.show()
X = np.arange(27)
plt.bar(X + 0.00, Players_top['weight_factor'], color = 'b', width = 0.25, label = 'Average')
plt.ylabel('Weight')
plt.xlabel('Player')
plt.xticks( X,Players_top['Name'], rotation = 90 )
plt.show()
