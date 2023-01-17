# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from scipy.stats.mstats import winsorize
import seaborn as sns
from scipy import stats
import matplotlib.pyplot as plt

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
url = os.path.join(dirname, filename)
# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_csv(url)
df.head()
#look at all the variables.
df.describe(include='all').T
#I want to take a look at some of the unique categorical values.
for col in df.describe(include='O'):
    print(df[col].unique())
#data types. There are nulls. Year is reading as an integer. I might have to fix this, for now I take note.
#I'm not noticing any nonesense values.
df.info()
#A summary of missing variables represented as a percentage of the total missing content. 
def missingness_summary(df, print_log=False, sort='ascending'):
  s = df.isnull().sum()*100/df.isnull().count()
  s = s [s > 0]
  if sort.lower() == 'ascending':
    s = s.sort_values(ascending=True)
  elif sort.lower() == 'descending':
    s = s.sort_values(ascending=False)  
  if print_log: 
    print(s)
  
  return pd.Series(s)

suspects = missingness_summary(df, True, 'descending')
#Dropping Life expectancy and adult mortality.
df.dropna(subset=['Life expectancy ', 'Adult Mortality'], inplace=True)

suspects = missingness_summary(df, False)
drop_suspects = suspects[ suspects < 5 ]
drop_suspects
df[df[drop_suspects.keys()].isnull().any(axis=1)]

var = ['Life expectancy '] #variable of interest
lost_development = list(drop_suspects.keys())
lost_development.extend(var)

#Lost Development dataframe. Short name for easy access.
ld = df[lost_development]
ld = ld.dropna()

#No Missing Values for the lost development dataframe. 
missingness_summary(ld)
#Remove the drop_suspect features from the dataset. We will not use them with this dataset as this subset of data is incomplete.
cols = list(df.columns)
cols = [x for x in cols if x not in drop_suspects.keys()]

#My new Life Expectancy dataframe, without those few missing entries.
le = df[cols]
miss = missingness_summary(le, True)
le[le[miss.keys()].isnull().any(axis=1)].sort_values(['Country','Year'])
le[le['Schooling'].isna()]
omit = ['Total expenditure', 'Hepatitis B', 'Income composition of resources', 'Alcohol', 'Population', 'Schooling', 'GDP']

#drop those features you don't want to use right now. 
cols =  [x for x in cols if x not in omit]
le = le[cols]
miss = missingness_summary(le, True)
le
#Country, Year, and Status should not have any outliers. 
out = [x for x in cols if x not in ['Country', 'Year', 'Status']]

le[out[0]].plot.box(whis=3) 
plt.show()
out.pop(0)
le[out[0]].plot.box(whis=3) 
plt.show()
out.pop(0)
le[out[0]].plot.box(whis=3) 
plt.show()
out.pop(0)
#First incident of excessive outliers. Add the variable to an outlier list for future investigation.
outlier = ['infant deaths']
le[out[0]].plot.box(whis=3) 
plt.show()
out.pop(0)
outlier.append('percentage expenditure')
le[out[0]].plot.box(whis=3) 
plt.show()
out.pop(0)
outlier.append('Measles ')
le[out[0]].plot.box(whis=3) 
plt.show()
out.pop(0)
outlier.append('under-five deaths ')
le[out[0]].plot.box(whis=3) 
plt.show()
out.pop(0)
outlier.append(' HIV/AIDS')
outlier
# Tukey's method.
def tukey(field):
  q75, q25 = np.percentile(field, [75 ,25])
  iqr = q75 - q25
 
  for threshold in np.arange(1,5,0.5):
      min_val = q25 - (iqr*threshold)
      max_val = q75 + (iqr*threshold)
      print("The score threshold is: {}".format(threshold))
      print("Number of outliers is: {}".format(
          len((np.where((field > max_val) 
                        | (field < min_val))[0]))
      ))
        
for col in outlier:
    print("TUKEY INFORMATION FOR", col)
    print('____________________________')
    tukey(le[col])
#Trying out winsorize. 
from scipy.stats.mstats import winsorize

# Apply one-way winsorization to the highest end. I went with the 80th percentile. 
print(outlier[0])
wv1 = winsorize(le[outlier[0]], (0, 0.15))
plt.boxplot(wv1)
plt.show()
#Add a column to the datatable for this transformation.
le["w"+ outlier[0]] = wv1

# Apply one-way winsorization to the highest end. I went with the 80th percentile. 
print(outlier[1])
wv2 = winsorize(le[outlier[1]], (0, 0.15))
plt.boxplot(wv2)
plt.show()
#Add a column to the datatable for this transformation.
le["w"+ outlier[1]] = wv2

# Apply one-way winsorization to the highest end. I went with the 80th percentile. 
print(outlier[2])
wv3 = winsorize(le[outlier[2]], (0, 0.2))
plt.boxplot(wv3)
plt.show()
#Add a column to the datatable for this transformation.
le["w"+ outlier[2]] = wv3

# Apply one-way winsorization to the highest end. I went with the 80th percentile. 
print(outlier[3])
wv4 = winsorize(le[outlier[3]], (0, 0.0))
plt.boxplot(wv4)
plt.show()
#Add a column to the datatable for this transformation.
le["w"+ outlier[3]] = wv4

# Apply one-way winsorization to the highest end. I went with the 80th percentile. 
print(outlier[4])
wv5 = winsorize(le[outlier[4]], (0, 1.0))
plt.boxplot(wv4)
plt.show()
le[outlier[4]].describe()
le['Life expectancy '].corr(le[' HIV/AIDS'])
le.plot.scatter(x='Life expectancy ', y=' HIV/AIDS')
plt.show()
cols = list(le.columns)
omit = [' HIV/AIDS', 'Measles ', 'wMeasles ' ]
#drop those features you don't want to use right now. 
cols =  [x for x in cols if x not in omit]
le = le[cols]
le
le.plot()
le.plot.scatter(x='Life expectancy ', y="percentage expenditure")
plt.show()
le.plot.scatter(x='Life expectancy ', y="wpercentage expenditure")
plt.show()
max = 0.0
var = 'Life expectancy '


#slice the columns from five on - since these are numerical data that don't include the area(s) of interest. 
for col in cols[4:]:
    correlation = le[var].corr(le[col])
    print("The correlation score for {} is {} ".format(col, correlation ))
    
    if abs(correlation) >= max:
        max = abs(correlation)
        best = col

print("The greatest correlation of expenditures against the {} is {}".format(var, best)) 

le.corr()
features = cols[-3:]
le[features].corr()
features.append(var)
le[features]
plt.figure(figsize=(20,7))
#Including ci=None because I'm looking at the consequence of including bootstrapping on just two variables when we use all data - on a relatively small dataset. (See summary)
sns.lineplot(data=le[features], x='Life expectancy ', y='wpercentage expenditure', ci=None)
plt.show()
le[['wunder-five deaths ','winfant deaths', var]].plot(x=var, y=['wunder-five deaths ','winfant deaths'], figsize=(20, 8))

plt.figure(figsize=(20,7))
sns.lineplot(data=le[features], x='Life expectancy ', y='wpercentage expenditure')
plt.show()