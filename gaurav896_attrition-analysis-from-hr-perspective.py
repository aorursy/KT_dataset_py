# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
df = pd.read_csv('../input/HR_comma_sep.csv')
df[:5]
sns.heatmap(df.corr(),cmap='viridis', annot=True)
df['sales'].unique()
df.groupby(['left','sales']).mean()
# Prema Facie , People who have low satisfaction score & long working hours, tend to leave the organization, it would have

# been great, if some kind of key resource tagging would have been there. 

# Targets to control can directly with identified. Anyways lets move on.
df['average_montly_hours'].describe();

sns.boxplot(df['average_montly_hours'])
# In any month 22 days working , you would have 176 hours a month to work for. That means most of the employees who are working

# more than 201 are most susceptible to leave. Lets do some visualization
sns.pairplot(df)
df_IT = df[(df['sales'] == 'IT') & (df['left']== 1)]
x = df_IT.shape[0];

y =df.shape[0];

x/y