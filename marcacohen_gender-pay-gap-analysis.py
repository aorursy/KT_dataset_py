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
import seaborn as sns
df = pd.read_csv('/kaggle/input/incomes-by-career-and-gender/inc_occ_gender.csv', na_values="Na")
df.isna().sum()
df = df.dropna()
df = df.reset_index(drop = True)
data = df.sort_values('All_weekly', ascending = False)[:20]
male_data = df.sort_values('M_weekly', ascending = False)[:20]
sns.set_color_codes("pastel")
sns.barplot(x = 'M_weekly', y = 'Occupation', data = male_data, label = 'Male Salaries')
female_data = df.sort_values('F_weekly', ascending = False)[:20]
sns.barplot(x = 'F_weekly', y = 'Occupation', data = female_data, label = 'Female Salaries')
df['difference'] = df['M_weekly'] - df['F_weekly']
sns.barplot(x = 'difference', y = 'Occupation', data = df.sort_values(by = 'difference', ascending = False)[:15])
sns.barplot(x = 'difference', y = 'Occupation', data = df.sort_values(by = 'difference', ascending = True)[:15])
df.F_weekly.hist()
from scipy import stats as stats
stats.ttest_rel(a = data['M_weekly'], b = data['F_weekly'])