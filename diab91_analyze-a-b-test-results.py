import pandas as pd

import numpy as np

import random

import matplotlib.pyplot as plt

%matplotlib inline

#We are setting the seed to assure you get the same answers on quizzes as we set up

random.seed(42)
df = pd.read_csv('../input/ab_data.csv')

df.head()
df.shape
df.nunique()
len(df.query(' converted== 1').converted)/len(df.converted)
new_page = df.landing_page=='new_page'

treatment = df.group=='treatment'

df_new_xor_treat = df[(new_page) ^ (treatment)]

df_new_xor_treat.head()
df_new_xor_treat.shape
df.info()
df_newpage_treat = df[(df.landing_page == 'new_page') & (df.group == 'treatment')]

df_oldpage_control = df[(df.landing_page == 'old_page') & (df.group == 'control')]

df2 = df_newpage_treat.append(df_oldpage_control, ignore_index = True)
df2.head()
df2.tail()
# Double Check all of the correct rows were removed - this should be 0

df2[((df2['group'] == 'treatment') == (df2['landing_page'] == 'new_page')) == False].shape[0]
df2.user_id.nunique()
df2[df2.user_id.duplicated()]
df2[df2.user_id == 773192]
df2.drop_duplicates("user_id", inplace=True)
# it is the converted users over all the users

len(df2.query(' converted== 1').converted)/len(df2.converted)
df2_control = df2[df2.group == 'control'] 

df2_control_and_converted = df2[(df2.group == 'control') & (df2.converted == 1)] 



proportion_ctrl = df2_control_and_converted.shape[0]/df2_control.shape[0]

proportion_ctrl
df2_treat = df2[df2.group == 'treatment']



df2_treat_and_converted = df2[(df2.group == 'treatment') & (df2.converted == 1)] 



proportion_treat = df2_treat_and_converted.shape[0]/df2_treat.shape[0]

proportion_treat
prob_new_page = len(df2.query("landing_page == 'new_page'"))/len(df2)

prob_new_page
pnew = (df2_treat_and_converted +df2_control_and_converted).shape[0]/ (df2_treat+df2_control).shape[0]

pnew
pold = (df2_treat_and_converted +df2_control_and_converted).shape[0]/ (df2_treat+df2_control).shape[0]

pold
nnew = df2_treat.shape[0]

nnew
nold = df2_control.shape[0]

nold
new_page_converted = np.random.choice([1,0], size=nnew, p=[pnew,1 - pnew])

new_page_converted.sum()
old_page_converted = np.random.choice([1,0], size=nold, p=[pold,1 - pold])

old_page_converted.sum()
new_page_converted.mean() - old_page_converted.mean()
p_diffs = []

for _ in range(10000):

    new_page_converted = np.random.choice([1,0], size = nnew, replace = True, p = (pnew, 1-pnew))

    old_page_converted = np.random.choice([1,0], size = nold, replace = True, p = (pold, 1-pold))

    diff = new_page_converted.mean() - old_page_converted.mean()

    p_diffs.append(diff)
plt.hist(np.array(p_diffs))
p_new_act = df2.query("group == 'treatment'").converted.mean()

p_old_act = df2.query("group == 'control'").converted.mean()



diff_act = p_new_act - p_old_act



diff_act
p_diffs = np.array(p_diffs)

dist = np.random.normal(0, p_diffs.std(), p_diffs.size)

plt.hist(dist);

plt.axvline(diff_act, color = 'r')
pvalue = (dist > diff_act).mean()

pvalue
import statsmodels.api as sm



convert_old = df2.query("group == 'control'").converted.sum()

convert_new = df2.query("group == 'treatment'").converted.sum()

n_old = df2.query("landing_page == 'old_page'").count()[0]

n_new = df2.query("landing_page == 'new_page'").count()[0]
from scipy.stats import norm



z_score, p_value = sm.stats.proportions_ztest([convert_old, convert_new], [n_old, n_new], alternative="smaller")



print(z_score , p_value)
df2['intercept'] = 1

df2  = df2.join(pd.get_dummies(df2['group']))

df2.rename(columns = {"treatment": "ab_page"}, inplace=True)
df2.head()
y = df2["converted"]

x = df2[["intercept", "ab_page"]]



#load model

log= sm.Logit(y,x)



#fit model

result = log.fit()
result.summary()
countries_df = pd.read_csv('../input/countries.csv')

df_new = countries_df.set_index('user_id').join(df2.set_index('user_id'), how='inner')
countries_df.head()
### Create the necessary dummy variables

df2 = df2.merge(countries_df, on='user_id', how='left')

df2.head()

df2.country.unique()
df2 = df2.join(pd.get_dummies(df2['country'],drop_first = True))

df2.head()
### Fit Your Linear Model And Obtain the Results

df2['UK_ab_page'] = df2['UK']*df2['ab_page']

df2['US_ab_page'] = df2['US']*df2['ab_page']

logit3 = sm.Logit(df2['converted'], df2[['intercept', 'ab_page', 'UK', 'US', 'UK_ab_page', 'US_ab_page']])



results = logit3.fit()

results.summary()
#model2 = sm.Logit(df2['converted'], df2[['intercept','ab_page', 'UK', 'US']])

#result2 = model2.fit()
#result2.summary()