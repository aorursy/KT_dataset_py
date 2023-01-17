import pandas as pd

import numpy as np

import random

import matplotlib.pyplot as plt

%matplotlib inline

random.seed(42)
df = pd.read_csv('../input/ab-data/ab_data.csv')

df.head()
df.shape
df['user_id'].nunique()
df['converted'].mean()
#Adding the two values

len(df.query('group=="treatment" and landing_page!="new_page"')) + len(df.query('group=="control" and landing_page!="old_page"'))
df.isnull().sum()
df.info()
#Finding the mismatch rows

mismatch_1 = df.query('group=="treatment" and landing_page!="new_page"')

mismatch_2 = df.query('group=="control" and landing_page!="old_page"')



#Dropping those columns by their index

df2 = df.drop(mismatch_1.index)

df2 = df2.drop(mismatch_2.index)
# Double Check all of the correct rows were removed - this should be 0

df2[((df2['group'] == 'treatment') == (df2['landing_page'] == 'new_page')) == False].shape[0]
#No. of unique rows

df2['user_id'].nunique()
#Comparing with total size

df2['user_id'].shape
#User ID of the repeated row

df2[df2['user_id'].duplicated()==True]['user_id']
#Info of the repeated column

df2[df2['user_id'].duplicated()==True]
#Dropping the duplicated row

df2.drop_duplicates(subset='user_id',inplace=True)
df2['converted'].mean()
#Probability of control group

df2_control = df2.query('group=="control"')

df2_control['converted'].mean()
#Probability of treatment group

df2_treatment = df2.query('group=="treatment"')

df2_treatment['converted'].mean()
#Split of pages between pages

df2.query('landing_page=="new_page"').shape[0]/df2.shape[0]
#Calculating convert rate

p_new = df2.converted.mean() 

p_new
#Calculating convert rate

p_old = df2.converted.mean()

p_old ## Both are same!!
#size of new page group

n_new = len(df2[df2['group'] =='treatment'])

n_new
#size of new page group

n_old = len(df2[df2['group'] =='control'])

n_old
#Simulating conversion for new page for 0's and 1's

new_page_converted = np.random.binomial(1, p_new, size=n_new)

new_page_converted
#Simulating conversion for old page 0's and 1's 

old_page_converted = np.random.binomial(1, p_old, size=n_old)

old_page_converted
#Single simuation difference in mean

diff = new_page_converted.mean() - old_page_converted.mean()

diff
#Simulating for 10000 times

p_diffs=[]

for _ in range(10000):

    new_converted = np.random.binomial(1, p_new, size=n_new).mean()

    old_converted = np.random.binomial(1, p_old, size=n_old).mean()

    p_diffs.append(new_converted-old_converted)
#Converting to array

p_diffs = np.array(p_diffs)
#Actual difference from dataset

acc_diff = df2_treatment['converted'].mean() - df2_control['converted'].mean() 

acc_diff
plt.hist(p_diffs);

plt.xlabel('Mean of Probability Difference')

plt.ylabel('#Count')

plt.axvline(acc_diff,color='red',label='Actual mean difference')

plt.legend()
#P-Value

(p_diffs > acc_diff).mean()
import statsmodels.api as sm



convert_old = len(df2.query('landing_page == "old_page" and converted ==1 '))

convert_new = len(df2.query('landing_page == "new_page" and converted ==1 '))

n_old = df2.query('landing_page == "old_page"').shape[0]

n_new = df2.query('landing_page == "new_page"').shape[0]
#Values of the above

convert_old,convert_new,n_new,n_old
#smaller as we need to have p1 < p2 for alternative hypothesis

z_score, p_value = sm.stats.proportions_ztest([convert_old, convert_new], [n_old, n_new], alternative='smaller') 

z_score, p_value
df2.head()
df2['intercept'] = 1
df2[['control','ab_page']] = pd.get_dummies(df2['group'])
df2.drop('control',axis=1,inplace=True)
lm = sm.Logit(df2['converted'],df2[['intercept','ab_page']])

result = lm.fit()
result.summary()
countries_df = pd.read_csv('../input/ab-data/countries.csv')

df_new = countries_df.set_index('user_id').join(df2.set_index('user_id'), how='inner')
df_new[['CA','UK','US']] = pd.get_dummies(df_new['country'])
lm = sm.Logit(df_new['converted'],df_new[['CA','US','intercept']])

result = lm.fit()

result.summary()
# Adding Interactions

df_new['US_ab_page'] = df_new['US'] * df_new['ab_page']

df_new['CA_ab_page'] = df_new['CA'] * df_new['ab_page']
lm = sm.Logit(df_new['converted'],df_new[['CA','US','intercept','ab_page','US_ab_page','CA_ab_page']])

result = lm.fit()

result.summary()