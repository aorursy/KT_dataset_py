# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import os
import numpy as np
import pandas as pd
from sklearn.feature_selection import chi2
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import warnings
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import math
import seaborn as sns
from scipy.stats import chi2_contingency

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data = pd.read_excel('/kaggle/input/unemployment-and-mental-illness-survey/Cleaned Data.xlsx', sheet_name="Sheet1", header=0, keep_default_na = False)
df = pd.DataFrame(data, columns=data.columns)
df = df.rename(columns={'Annual income (including any social welfare programs) in USD': 'Total Annual Income', 'Total length of any gaps in my resume in months.': 'Gaps in Employment in Months'})
df.describe()
df.head(5)
fig, axs = plt.subplots(figsize=(10,5), ncols=1, nrows=1)
for i in df['I identify as having a mental illness'].unique():
    word = 'With ' if i == 1 else 'No '
    sns.distplot(df[df['I identify as having a mental illness'] == i]['Total Annual Income'], hist=True, kde_kws={'clip': (0.0, 200000.0), "label": "{} Mental Illness".format(word)})
# Plot the gaps in employment to total reported income
fig2 = sns.jointplot(df['Gaps in Employment in Months'], df['Total Annual Income'], kind='scatter', height=8)
# get ratio of mental illness to no mental illness by region
regions = df['Region'].unique()
df_temp = pd.DataFrame({'Region':regions, 'With Mental Illness': 0, 'No Mental Illness': 0, 'Ratio': 0})
df_temp = df_temp[df_temp['Region'] != '']

for i in regions:
    x1 = len(df[(df['I identify as having a mental illness'] == 1) & (df['Region'] == i)])
    x2 = len(df[(df['I identify as having a mental illness'] == 0) & (df['Region'] == i)])
    
    df_temp.loc[df_temp['Region'] == i, 'With Mental Illness'] = x1
    df_temp.loc[df_temp['Region'] == i, 'No Mental Illness'] = x2
    df_temp.loc[df_temp['Region'] == i, 'Ratio'] = x1 / (x1 + x2)
df_temp = df_temp.sort_values(by=['Ratio'], ascending=False)

# Plot the ration by region
plt.bar(df_temp['Region'], df_temp['Ratio'], align='edge', width=0.3)
plt.xticks(rotation=90)
 
# Show graphic
plt.show()


# Look at the relationship between gender and self identified mental illness
df['Gender_Binary'] = 0
df.loc[df['Gender'] == "Female", 'Gender_Binary'] = 1
gender_binary = df['Gender_Binary']
gender = df['Gender']
illness = df['I identify as having a mental illness']
res = pd.crosstab(gender, illness.eq(1), rownames={'gender'})
print(res)