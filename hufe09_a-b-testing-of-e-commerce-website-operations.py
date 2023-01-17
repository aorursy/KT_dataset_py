import pandas as pd

import numpy as np

import random

import matplotlib.pyplot as plt

%matplotlib inline

# 设置随机数生成器的种子，以确保在测验中获得与我们设置的答案相同的答案

random.seed(42)
df = pd.read_csv('../input/ab-data.csv')

df.head()
df.shape[0]
df['user_id'].unique()

df['user_id'].nunique()
print(df['converted'].unique())

df['converted'].mean()
df[((df['group'] == 'treatment') == (df['landing_page'] == 'new_page')) == False].shape[0]
print(df.group.value_counts())

df.landing_page.value_counts()
mis_match1 = df[(df['landing_page'] == "new_page") & (df['group'] == "control")].count()[0]

mis_match1
mis_match2 = df[(df['landing_page'] == "old_page") & (df['group'] == "treatment")].count()[0]

mis_match2
mis_match1 + mis_match2
df.isnull().any()
df2 = df[(df['landing_page'] == "new_page") & (df['group'] == 'treatment') | (df['landing_page'] == "old_page") & (df['group'] == 'control')]

df2.head()
df2.info()
df2[((df2['group'] == 'treatment') == (df2['landing_page'] == 'new_page')) == False].shape[0]
df2[((df2['group'] == 'control') == (df2['landing_page'] == 'old_page')) == False].shape[0]
df2.user_id.nunique()
duplicated_user_id = df2[df2.duplicated('user_id', keep= False)]

duplicated_user_id
duplicated_user_id.index
df2 = df2.drop_duplicates('user_id', keep='first')

df2[df2.user_id.duplicated() == True]
df2.converted.mean()
df2.query('group == "control"').converted.mean()
df2[df2['group'] == 'treatment']['converted'].mean()
num_new_page = df2.query('landing_page == "new_page"').user_id.count()

num_users = df2.user_id.count()

num_new_page / num_users
P_new = df2['converted'].mean()

P_new
P_old = P_new

P_old 
n_new = df2[df2['landing_page'] == 'new_page'].count()[0]

n_new
n_old = df2[df2['landing_page'] == 'old_page'].count()[0]

n_old
new_page_converted = np.random.choice([0, 1], size=n_new, p= [(1-P_new), P_new])

new_page_converted
old_page_converted = np.random.choice([0, 1], size=n_old, p=[(1-P_old), P_old])

old_page_converted
p_diff = new_page_converted.mean() - old_page_converted.mean()

p_diff
p_diffs = []



for i in range(10000):

    p_new_converted = np.random.choice([0, 1], size=n_new, p=[(1-P_new), P_new])

    p_old_converted = np.random.choice([0, 1], size=n_old, p=[(1-P_old), P_old])

    p_diff = p_new_converted.mean() - p_old_converted.mean()

    p_diffs.append(p_diff)
p_diffs = np.array(p_diffs)

plt.hist(p_diffs)
plt.hist(p_diffs)

obs_diff = df2[df2['group'] == 'treatment']['converted'].mean() - df2[df2['group'] == 'control']['converted'].mean()

plt.axvline(obs_diff, color='red')
(p_diffs > obs_diff).mean()
import statsmodels.api as sm



convert_old = df2[df2['group'] == 'control']['converted'].sum()

convert_new = df2[df2['group'] == 'treatment']['converted'].sum()

n_old = df2[df2['group'] == 'control'].count()[0]

n_new = df2[df2['group'] == 'treatment'].count()[0]
z_score, p_val = sm.stats.proportions_ztest([convert_old, convert_new], [n_old, n_new], alternative='smaller')

z_score, p_val
from scipy.stats import norm

norm.cdf(z_score), norm.ppf(1-(0.05))
# 创建虚拟列

df2[['control', 'treatment']]= pd.get_dummies(df2['group'])

df2['ab_page'] = df2['treatment']

df2.head()
# 逻辑回归

df2['intercept'] = 1

logit_mod = sm.Logit(df2['converted'], df2[['intercept', 'ab_page']]) # logit模型拟合

results = logit_mod.fit()
print(results.summary())
# 读取国家数据

df_countries = pd.read_csv('../input/countries.csv')

df_countries.country.value_counts()
# 按user_id合并表

df3 = df_countries.set_index('user_id').join(df2.set_index('user_id'), how='inner')

df3.head()
# 创建虚拟列

df3[['CA', 'UK', 'US']] = pd.get_dummies(df3['country'])

df3 = df3.drop(['CA'], axis=1)

df3.head()
# 逻辑回归

df3['intercept'] = 1

logit_mod = sm.Logit(df3['converted'], df3[['intercept', 'US', 'UK']])

results = logit_mod.fit()

print(results.summary())
# 添加 UK_ab_page 列

df3['UK_ab_page'] = df3['UK'] * df3['ab_page']

df3.head()
# 逻辑回归

df3['intercept'] = 1

logit_mod = sm.Logit(df3['converted'], df3[['intercept', 'ab_page', 'US', 'UK', 'UK_ab_page']])

results = logit_mod.fit()   # logit模型拟合

print(results.summary())