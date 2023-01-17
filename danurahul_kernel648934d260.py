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
import scipy.stats as stats

import seaborn as sns
import pandas as pd
import numpy as np
dataset=sns.load_dataset('tips')


dataset.head()
dataset_table=pd.crosstab(dataset['sex'],dataset['smoker'])
dataset_table
observed_values=dataset_table.values
print('observed values: \n',observed_values)
val=stats.chi2_contingency(dataset_table)
val
expected_value=val[3]
no_of_rows=len(dataset_table.iloc[0:2,0])
no_of_columns=len(dataset_table.iloc[0,0:2])
ddof=(no_of_rows-1)*(no_of_columns-1)
print('degree of freedom',ddof)
alpha=0.05
from scipy.stats import chi2
chi_square=sum([(o-e)*2./e for o,e in zip(observed_values,expected_value)])
chi_square_statistic=chi_square[0]+chi_square[1]
print('chi-square-statistic:',chi_square_statistic)
critical_value=chi2.ppf(q=1-alpha,df=ddof)
print('critical_value',critical_value)
p_value=1-chi2.cdf(x=chi_square_statistic,df=ddof)
print('p-value',p_value)
print('significance_level',alpha)
if chi_square_statistic>=critical_value:
    print('reject h0 there is a relationship between 2 categorical variable')
else:
    print('retain h0 there is a relationship between 2 categorical variable')
if p_value>=alpha:
    print('there is no relationship')
else:
    print('there is  relationship')
    
