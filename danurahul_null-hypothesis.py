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
df=pd.read_csv("https://raw.githubusercontent.com/krishnaik06/Multiple-Linear-Regression/master/50_Startups.csv")
df.head()
x=df.iloc[:,:-1]
y=df.iloc[:,-1]
ages=[10,28,13,45,38,29,48]
len(ages)
import numpy as np
print(np.mean(ages))
sample_size=4
age_sample=np.random.choice(ages,sample_size)
age_sample
from scipy.stats import ttest_1samp,ttest_ind,ttest_rel
titles,p_value=ttest_1samp(age_sample,30)
print(p_value)
if p_value <0.5:
    print('reject')
else:
    print('accept')
import numpy as np
import pandas as pd
import scipy.stats as stats
import math
np.random.seed(6)
school_ages=stats.poisson.rvs(loc=10,mu=35,size=1500)
classA_ages=stats.poisson.rvs(loc=18,mu=30,size=60)
classA_ages.mean()
_,p_value,ttest_1samp(a=classA_ages,popmean=school_ages.mean())
school_ages.mean()
if p_value<0.5:
    print('we are rejecting null hypthessis')
else:
    print(accept)
np.random.seed(30)
classB_ages=stats.poisson.rvs(loc=18,mu=33,size=60)
classB_ages.mean()
_,p_value=ttest_ind(a=classA_ages,b=classB_ages,equal_var=False)
print(p_value)
if p_value<0.5:
    print('rejecting')
else:
    print('')
weight1=[10,39,40,25,44]
weight2=weight1+stats.norm.rvs(scale=5,loc=-1.25,size=5)
print(weight1)
print(weight2)
df=pd.DataFrame({"weight_10":np.array(weight1),
                "weight_20":np.array(weight2),
                "weight_change":np.array(weight2)-np.array(weight1)})
df
_,p_value=ttest_rel(a=weight1,b=weight2)
print(p_value)
import seaborn as sns
df=sns.load_dataset('iris')
df.corr()
sns.pairplot(df)
