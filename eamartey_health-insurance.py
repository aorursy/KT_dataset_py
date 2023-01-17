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
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
df = pd.read_csv('/kaggle/input/health-insurance/states.csv')
df.head()
df.isna().sum()
df.info()
df['State Medicaid Expansion (2016)'].value_counts() ## The value_counts will let me know which is the mode 
df['State Medicaid Expansion (2016)'].replace(np.nan,'True', inplace = True )
df['Medicaid Enrollment (2013)'].mean() ## Check for the mean, since it's a float
df['Medicaid Enrollment (2013)'].replace(np.nan,2255699.08, inplace = True ) ## Replace the null with it
df['Medicaid Enrollment Change (2013-2016)'].mean() ## Find the mean of the next column
df['Medicaid Enrollment Change (2013-2016)'].replace(np.nan,644246.28, inplace = True ) ## Replace the null with it
df.isna().sum() ## Check to see if all the nulls are gone
df['State Medicaid Expansion (2016)'].replace('True', 1, inplace = True)
df['State Medicaid Expansion (2016)'].replace('False', 1, inplace = True)

df['Uninsured Rate (2010)'] = list(map(lambda x: x[:-1], df['Uninsured Rate (2010)'].values))
df['Uninsured Rate (2015)'] = list(map(lambda x: x[:-1], df['Uninsured Rate (2015)'].values))
df.head()
df['Uninsured Rate Change (2010-2015)'] = list(map(lambda x: x[:-1], df['Uninsured Rate Change (2010-2015)'].values))
df.head()
df['Uninsured Rate (2010)'] = [float(x) for x in df['Uninsured Rate (2010)'].values]
df['Uninsured Rate (2015)'] = [float(x) for x in df['Uninsured Rate (2015)'].values]
df['Uninsured Rate Change (2010-2015)'] = [float(x) for x in df['Uninsured Rate Change (2010-2015)'].values]
df.head()
plt.figure(figsize = (10, 10))
sns.barplot('Uninsured Rate Change (2010-2015)', 'State', data = df)
                    
plt.figure(figsize = (10, 8))
sns.barplot('Employer Health Insurance Coverage (2015)','State', data = df)

sns.catplot('State Medicaid Expansion (2016)','State',kind = 'swarm', data = df, height = 8)
plt.figure(figsize = (10, 8))
sns.scatterplot('Uninsured Rate (2010)', 'Uninsured Rate (2015)', data = df)
 
plt.figure(figsize = (10, 8))
sns.scatterplot('Health Insurance Coverage Change (2010-2015)','State', data = df)
plt.figure(figsize = (10, 8))
sns.barplot('Marketplace Health Insurance Coverage (2016)','State', data = df) 


plt.figure(figsize = (30,30)) 
sns.barplot('Marketplace Health Insurance Coverage (2016)','Marketplace Tax Credits (2016)', data = df)
a_plot = sns.boxplot('State Medicaid Expansion (2016)','Marketplace Health Insurance Coverage (2016)', data = df)
a_plot.set(xlim=(-1, 2))
a_plot.set(ylim=(0,1000000))

plt.figure(figsize = (12,8))
sns.barplot('Medicaid Enrollment (2013)','Medicaid Enrollment (2016)', data = df)
plt.figure(figsize = (10,8))
sns.barplot('Medicaid Enrollment Change (2013-2016)','State', data = df) 
sns.scatterplot('Medicaid Enrollment (2016)','Medicare Enrollment (2016)', data = df)
a_plot = sns.scatterplot('Medicaid Enrollment (2016)','Medicare Enrollment (2016)', data = df)
a_plot.set(xlim=(0, 5000000))
a_plot.set(ylim=(0, 5000000))


