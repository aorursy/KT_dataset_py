# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
%matplotlib inline
data = pd.read_csv('../input/StudentsPerformance.csv')
    
data.describe()
data.head(5)
data.isnull().sum()
passscore = 40

data['math_passed']=np.where(data['math score']>=passscore, 'p','f')
data.math_passed.value_counts()
data.drop(['lunch','test preparation course'],axis=1)
sns.countplot(x='gender', data=data)
plt.show()
sns.countplot(x='math_passed', hue='gender', data= data)
plt.show()
data['reading_passed']=np.where(data['reading score']>passscore, 'p','f')
data['writing_passed']=np.where(data['writing score']>passscore, 'p','f')
data.head(5)
fig, ax = plt.subplots(2,1,figsize=(12,6))
sns.countplot(x='reading_passed', hue='gender', data= data, ax=ax[0])
sns.countplot(x='writing_passed', hue='gender',data=data, ax=ax[1])
plt.show()
sns.countplot(x='parental level of education', hue='math_passed', data=data)
plt.show()
sns.countplot(x='parental level of education', hue='math_passed', data=data)
plt.show()
sns.countplot(x='parental level of education', hue='reading_passed', data=data)
plt.show()

sns.countplot(x='parental level of education', hue='writing_passed', data=data)
plt.show()

data.groupby(['parental level of education', 'math_passed']).size()
data.groupby(['parental level of education', 'reading_passed']).size()
data.groupby(['parental level of education', 'writing_passed']).size()

