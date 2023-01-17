#Assignment Author = Haider Ali
#Code provided by = Genoveva Vargas-Solar
#INP-Grenoble-SGB
#HVDC
#Student id: 42002598 
#Email-id: haider2303@gmail.com , Haider.Ali@grenoble-inp.org
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data = {
'year': [2010, 2011, 2012,
2010, 2011, 2012,
2010, 2011, 2012],
'team': ['FCBarcelona', 'FCBarcelona', 'FCBarcelona',
'RMadrid', 'RMadrid', 'RMadrid',
'ValenciaCF', 'ValenciaCF', 'ValenciaCF'],
'wins': [30, 28, 32, 29, 32, 26, 21, 17, 19],
'draws': [6, 7, 4, 5, 4, 7, 8, 10, 8],
'losses': [2, 3, 2, 4, 2, 5, 9, 11, 11]
}

football = pd.DataFrame(data, columns = ['year', 'team', 'wins', 'draws', 'losses'])
print(football)
edu = pd.read_csv('/kaggle/input/ense32020-ict-lesson-2/files/ch02/educ_figdp_1_Data.csv',
                  na_values=':', usecols=['TIME', 'GEO', 'Value'])
edu
edu.tail()
edu.head(9)
edu.tail()
edu.describe()
edu['Value']
edu[10:14]
edu.iloc[90:94][['TIME','GEO']]
edu['Value'] > 6.5



edu[edu['Value'] > 6.5]
edu[edu["Value"].isnull()].head()
edu.max(axis = 0)
print ('Pandas max function:', edu['Value'].max())
print ('Python max function:', max(edu['Value']))
s = edu["Value"]/100
s.head()
s = edu["Value"].apply(np.sqrt)
s.head()
s = edu["Value"].apply(lambda d: d**2)
s.head()
edu['ValueNorm'] = edu['Value']/edu['Value'].max()
edu.tail()
edu.drop('ValueNorm', axis = 1, inplace = True)
edu.tail()
edu = edu.append({"TIME": 2000, "Value": 5.00, "GEO": 'appended_value'},
                  ignore_index = True)
edu.tail()
edu.drop(max(edu.index), axis = 0, inplace = True)
edu.tail()
eduDrop = edu[~edu["Value"].isnull()].copy()
eduDrop.head()
eduDrop = edu.dropna(how = 'any', subset = ["Value"])
eduDrop.head()
eduFilled = edu.fillna(value = {"Value": 0})
eduFilled.head()
edu.sort_values(by = 'Value', ascending = False,
                inplace = True)
edu.head()
edu.sort_index(axis = 0, ascending = True, inplace = True)
edu.head()
group = edu[["GEO", "Value"]].groupby('GEO').mean()
group.head()
filtered_data = edu[edu["TIME"] > 2005]
pivedu = pd.pivot_table(filtered_data, values = 'Value',
                        index = ['GEO'], columns = ['TIME'])

pivedu
pivedu
pivedu = pivedu.drop(['Euro area (13 countries)',
                      'Euro area (15 countries)',
                      'Euro area (17 countries)',
                      'Euro area (18 countries)',
                      'European Union (25 countries)',
                      'European Union (27 countries)',
                      'European Union (28 countries)'
                      ], axis=0)
pivedu = pivedu.rename(
    index={'Germany (until 1990 former territory of the FRG)': 'Germany'})
pivedu = pivedu.dropna()
pivedu.rank(ascending=False, method='first').head()
totalSum = pivedu.sum(axis = 1).sort_values(ascending = False)
totalSum.plot(kind = 'bar', style = 'b', alpha = 0.4,
              title = "Total Values for Country")
my_colors = ['b', 'r', 'g', 'y', 'm', 'c']
ax = pivedu.plot(kind='barh', stacked=True, color=my_colors, figsize=(12, 6))
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.savefig('Value_Time_Country.png', dpi=300, bbox_inches='tight')