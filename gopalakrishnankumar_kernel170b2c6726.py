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
# -*- coding: utf-8 -*-

"""

Created on Fri Apr 24 12:23:53 2020



@author: DEE

"""



import pandas as pd

import numpy as np

import random

import matplotlib.pyplot as plt

#%matplotlib inline



# We are setting the seed to assure you get the same answers on quizzes as we set up



random.seed(42)

df=pd.read_csv('/kaggle/input/ab_data/ab_data.csv')

df

df.head()

df.shape[0]

len(df['user_id'].unique())

df['converted'].sum()

df['converted'].sum() / df.shape[0]

# Mismatch dataframe for the old_page users



no_mismatch_oldpage = df[(df['landing_page'] == 'old_page') & (df['group'] == 'treatment')]



# Number of times that the mismatch occurs for the treatment group users



print(len(no_mismatch_oldpage))



# Mismatch dataframe for the new_page users



no_mismatch_newpage = df[(df['landing_page'] == 'new_page') & (df['group'] == 'control')]



# Number of times that the mismatch occurs for the control group users



print(len(no_mismatch_newpage))



# Total number of times that the mismatch occurs



print(len(no_mismatch_oldpage)+len(no_mismatch_newpage))



df.info()



# Creating an new dataframe that holds all mismatched rows



mismatch_all = pd.concat([no_mismatch_newpage, no_mismatch_oldpage])



# Creating a new dataframe df2 such that it holds all values except mismatched values



df2 = df



# Dropping all mismatched values from the df2 dataframe



df2.drop(mismatch_all.index,inplace=True)



# Double Check all of the correct rows were removed - this should be 0



df2[((df2['group'] == 'treatment') == (df2['landing_page'] == 'new_page')) == False].shape[0]



len(df2['user_id'].unique())



len(df2['user_id']) - len(df2['user_id'].unique())





df2[df2.duplicated(['user_id'],keep=False)]



df2.drop(labels=1899, axis=0, inplace=True)



len(df2[df2.duplicated(['user_id'],keep=False)])



df2['converted'].mean()



df2[df2['group'] == 'control']['converted'].mean()



df2[df2['group'] == 'treatment']['converted'].mean()



len(df2[df2['landing_page'] == 'new_page']) / len(df2)



pnew = df2[df2['landing_page']=='new_page']['converted'].mean()



# Displaying the result



pnew



pold = df2[df2['landing_page']=='old_page']['converted'].mean()



# Displaying the result



pold





# Mean of the probabilities



pmean = np.mean([pnew,pold])



# Displaying the result



pmean



# Calc differences in probability of conversion for new and old page (not under H_0)



pdiff = pnew-pold



# Displaying the result



pdiff



nnew = len(df2[df2['group'] == 'treatment'])



# Displaying the result



nnew



nold = len(df2[df2['group'] == 'control'])



# Displaying the result



nold



new_converted = np.random.choice([1, 0], size=nnew, p=[pmean, (1-pmean)])



# Displaying the mean of the result



new_converted.mean()



old_converted = np.random.choice([1,0], size=nold, p=[pmean, (1-pmean)])



# Displaying the mean of the result



old_converted.mean()



old_converted.mean() - new_converted.mean()



p_diffs = []



for _ in range(10000):

    new_converted = np.random.choice([1, 0], size=nnew, p=[pmean, (1-pmean)]).mean()

    old_converted = np.random.choice([1, 0], size=nold, p=[pmean, (1-pmean)]).mean()

    diff = new_converted - old_converted 

    p_diffs.append(diff)

    

plt.hist(p_diffs)

plt.xlabel('p_diffs')

plt.ylabel('Frequency')

plt.title('Plot of 10K simulated p_diffs');



# Compute difference from original dataset ab_data.csv



actual_diff = df[df['group'] == 'treatment']['converted'].mean() -  df[df['group'] == 'control']['converted'].mean()



# Displaying the actual difference



actual_diff



p_diffs = np.array(p_diffs)



p_diffs



(p_diffs > actual_diff).mean()





import statsmodels.api as sm



convert_old = sum(df2.query("group == 'control'")['converted'])



convert_new = sum(df2.query("group == 'treatment'")['converted'])



n_old = len(df2.query("group == 'control'"))



n_new = len(df2.query("group == 'treatment'"))



print(convert_old, convert_new, n_old, n_new)



z_score, p_value = sm.stats.proportions_ztest([convert_old, convert_new], [n_old, n_new], alternative='smaller')



# Displaying the z_score and p_value



print(z_score, p_value)



from scipy.stats import norm



print(norm.cdf(z_score))

# Tells us how significant our z-score is



# for our single-sides test, assumed at 95% confidence level, we calculate: 

print(norm.ppf(1-(0.05)))

# Tells us what our critical value at 95% confidence is 

# Here, we take the 95% values
