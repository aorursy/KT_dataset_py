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
adult_df = pd.read_csv('../input/train_data.csv', header = None, delimiter=' *, *',engine='python')

adult_df.head()
adult_df.columns = ['age', 'workclass', 'fnlwgt', 'education', 'education_num',
                    'marital_status', 'occupation', 'relationship',
                    'race', 'sex', 'capital_gain', 'capital_loss',
                    'hours_per_week', 'native_country', 'income']

adult_df.head()
import matplotlib.pyplot as plt
%matplotlib inline
adult_df.boxplot() #for plotting boxplots for all the numerical columns in the df
plt.show()
adult_df.boxplot(column='fnlwgt')
plt.show()
adult_df.boxplot(column='capital_gain')
plt.show()
adult_df.boxplot(column='capital_loss')
plt.show()
adult_df.boxplot(column='hours_per_week')
plt.show()
adult_df.boxplot(column='age') 
plt.show()
#for value in colname:
q1 = adult_df['age'].quantile(0.25) #first quartile value
q3 = adult_df['age'].quantile(0.75) # third quartile value
iqr = q3-q1 #Interquartile range
low  = q1-1.5*iqr #acceptable range
high = q3+1.5*iqr #acceptable range
adult_df_include = adult_df.loc[(adult_df['age'] >= low) & \
                                (adult_df['age'] <= high)] # meeting the acceptable range
adult_df_exclude = adult_df.loc[(adult_df['age'] < low) | \
                               (adult_df['age'] > high)] #not meeting the acceptable range

print(adult_df_include.shape)
print(adult_df_exclude.shape)
print(low)
age_mean=int(adult_df_include.age.mean()) #finding the mean of the acceptable range
print(age_mean)
#imputing outlier values with mean value
adult_df_exclude.age=age_mean
#getting back the original shape of df
adult_df_rev=pd.concat([adult_df_include,adult_df_exclude]) #concatenating both dfs to get the original shape
adult_df_rev.shape
#capping approach

adult_df_exclude.loc[adult_df_exclude["age"] <low, "age"] = low
adult_df_exclude.loc[adult_df_exclude["age"] >high, "age"] = high
