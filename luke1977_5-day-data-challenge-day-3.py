# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from scipy.stats import ttest_ind

import seaborn as sns



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.



cereal=pd.read_csv('../input/cereal.csv')

print(cereal.shape)

print(cereal.info())

print(cereal.describe())

print(cereal.head())



print(cereal['type'].unique())

print(cereal['type'].value_counts())

print(cereal['shelf'].value_counts())

_=sns.swarmplot('shelf','sugars',data=cereal)

shelf1=cereal[cereal['shelf']==1]

shelf2=cereal[cereal['shelf']==2]

shelf3=cereal[cereal['shelf']==3]

print("Shape of DataFrames for each shelf category - 1, 2 & 3:")

print(shelf1.shape,shelf2.shape,shelf3.shape)

print("Standard deviation of 'sugars' data in each shelf category:")

print(np.std(shelf1['sugars']),np.std(shelf2['sugars']),np.std(shelf3['sugars']))

ttest12=ttest_ind(shelf1['sugars'],shelf2['sugars'],equal_var=False)

ttest13=ttest_ind(shelf1['sugars'],shelf3['sugars'],equal_var=False)

ttest23=ttest_ind(shelf2['sugars'],shelf3['sugars'],equal_var=False)

print("T-test for 'sugars' data by shelf category:")

print("Between shelves 1 & 2: ",ttest12)

print("Between shelves 1 & 3: ",ttest13)

print("Between shelves 2 & 3: ",ttest23)




