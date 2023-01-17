# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import matplotlib.pyplot as plt

import seaborn as sns



data = pd.read_csv("/kaggle/input/titanic/train.csv", index_col='PassengerId', usecols=['PassengerId','Pclass','Survived'])

data.head()
data.info()


plt.figure(figsize=(10,2.5))



plt.subplot(121)

sns.countplot(data.Pclass)

plt.title("Passenger Class")



plt.subplot(122)

sns.countplot(data.Survived)

plt.title("Survived or not")



plt.show()
PClass_survd = pd.pivot_table(data,index=['Pclass'],columns=['Survived'],aggfunc='size')

PClass_survd
sns.heatmap(PClass_survd,annot=True, fmt='g',square=True,cmap='hot')

plt.title('Class Vs Survived',fontsize=20)

plt.show()
pct_class = PClass_survd.sum(axis=1)/891

pct_class
pct_survived = PClass_survd.sum(axis=0)/891

pct_survived
pct_class.to_frame()@(pct_survived.to_frame().T) 
exp = round(pct_class.to_frame()@(pct_survived.to_frame().T)*891)

exp
Chi_table = ((PClass_survd - exp)**2)/exp

Chi_table
from scipy import stats

chi2_stat, p_val, dof, ex = stats.chi2_contingency(PClass_survd)



print("Chi square value is ",chi2_stat)

print("P value is",p_val)

print("Degrees of Freedom:",dof)