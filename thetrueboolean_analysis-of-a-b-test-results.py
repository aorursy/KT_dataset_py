import pandas as pd

import numpy as np

import random

import matplotlib.pyplot as plt

%matplotlib inline



#I am setting a random seed to ensure reproducibility of results

random.seed(42)
df = pd.read_csv('../input/ab-data/ab_data.csv', encoding='utf8', engine='python')

df.head()
df.info()
# convert 'timestamp' to datetime for easier manipulation

df.timestamp = pd.to_datetime(df.timestamp)
# confirm that worked

df.dtypes
print('The website was visited', df.shape[0], 'times.')
print('There are', df.user_id.nunique(), 'unique users.')
print('The A/B test was conducted for', len(df.timestamp.dt.floor('d').value_counts()), 'days.')
df[df.converted == 1].shape[0]/df.shape[0]
df[((df.group=='treatment') & (df.landing_page!='new_page')) | 

   ((df.group!='treatment') & (df.landing_page=='new_page'))].shape[0]

# OR

# df[((df['group'] == 'treatment') ^ (df['landing_page'] == 'new_page'))].shape[0]

# OR

# df[((df['group'] == 'treatment') == (df['landing_page'] == 'new_page')) == False].shape[0]
# Make use of exclusive OR (XOR) to find disalignment between page and corresponding group

df2 = df.drop(df[((df['group'] == 'treatment') ^ (df['landing_page'] == 'new_page'))].index, axis=0)
# Double Check all of the correct rows were removed - this should return 0

df2[((df2['group'] == 'treatment') == (df2['landing_page'] == 'new_page')) == False].shape[0]
df2.user_id.nunique()
df2.user_id.shape
df2[df2.user_id.duplicated(keep=False)]
# drop the first duplicate with index 1899

df2.drop([1899], inplace=True)
df2.converted.sum()/df2.shape[0]
# control group conversion rate

ctrl = df2[df2.group=='control']

p_ctrl = ctrl.converted.sum()/ctrl.shape[0]

p_ctrl
# treatment group conversion rate

treat = df2[df2.group=='treatment']

p_treat = treat.converted.sum()/treat.shape[0]

p_treat
(df2.landing_page=='new_page').sum()/df2.shape[0]
# set conversion rates and sample sizes for both groups under the null

p_old = df2.converted.mean()

p_new = df2.converted.mean()

n_old = df2[df2.landing_page == 'old_page'].shape[0]

n_new = df2[df2.landing_page == 'new_page'].shape[0]

print(f"p_old: {p_old}\np_new: {p_new}\nn_old: {n_old}\nn_new: {n_new}")
# simulation of the two binomial distributions and the difference in their conversion rates

old_page_converted = np.random.binomial(1, p=p_old, size=n_old)

new_page_converted = np.random.binomial(1, p=p_new, size=n_new)

diff = new_page_converted.mean() - old_page_converted.mean()

diff
# simulate the difference between the conversion rate for new and old pages

# make use of binomial distribution since that fits our scenario

new_page_converted = np.random.binomial(n_new, p_new, 10000) #returns no. of successes from n_new trials,performed 10000 times

old_page_converted = np.random.binomial(n_old, p_old, 10000) #returns no. of successes from n_old trials,performed 10000 times

#NB: we cannot use new_page_converted.mean() as above since our simulation returns the no. of successes and not 0s and 1s

p_diffs = new_page_converted/n_new - old_page_converted/n_old

p_diffs = np.array(p_diffs)
plt.hist(p_diffs);
# get observed difference first, then determine the more extreme values in favour of the alternative

obs_diff = (df2[df2.group=='treatment'].converted.mean()) - (df2[df2.group=='control'].converted.mean())

p_val = (p_diffs > obs_diff).mean()

p_val
import statsmodels.api as sm



convert_old = df2[df2.landing_page=='old_page'].converted.sum()

convert_new = df2[df2.landing_page=='new_page'].converted.sum()

n_old = df2[df2.landing_page=='old_page'].shape[0]

n_new = df2[df2.landing_page=='new_page'].shape[0]
test_stat, p_value = sm.stats.proportions_ztest(np.array([convert_old, convert_new]), np.array([n_old, n_new]), alternative='smaller')

print(f"z-score: {test_stat}\np-value: {p_value}") # test_stat is the z-score for our p-value
df2['intercept'] = 1

df2['ab_page'] = pd.get_dummies(df2.group)['treatment']

df2.head()
log_mod_1 = sm.Logit(df2.converted, df2[['intercept', 'ab_page']])

result_1 = log_mod_1.fit()
result_1.summary()
countries_df = pd.read_csv('../input/countries-dataset/countries.csv', encoding='utf8', engine='python')

df_new = countries_df.set_index('user_id').join(df2.set_index('user_id'), how='inner')

df_new.head()
# check unique entries in 'country' column

df_new.country.unique()
### Create the necessary dummy variables.

### I'll create dummies for UK and US alone, leaving CA as the baseline.

df_new[['UK', 'US']] = pd.get_dummies(df_new.country)[['UK','US']]



log_mod_2 = sm.Logit(df_new.converted, df_new[['intercept', 'ab_page', 'UK', 'US']])

result_2 = log_mod_2.fit()

# get model summary

result_2.summary()
# create the additional columns for the interactions

df_new['UK_ab_page'] = df_new['UK'] * df_new['ab_page']

df_new['US_ab_page'] = df_new['US'] * df_new['ab_page']
df_new.head()
### Fit the Linear Model And Obtain the Results

log_mod_3 = sm.Logit(df_new.converted, df_new[['intercept','ab_page', 'UK', 'US', 'UK_ab_page', 'US_ab_page']])

result_3 = log_mod_3.fit()

result_3.summary()