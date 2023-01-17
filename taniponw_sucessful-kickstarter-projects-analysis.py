import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline
most_backed = pd.read_csv('../input/most_backed.csv',index_col=0)

most_backed.head()
def stripBracket(text):

    text = text.strip('[')

    text = text.strip(']')

    text = text.split(',')

    return text
most_backed['num_of_tiers'] = most_backed['pledge.tier'].apply(lambda x : len(stripBracket(x)))

most_backed['%_overfunded'] = most_backed['amt.pledged']/most_backed['goal']*100
most_backed['count'] = 1

most_backed.head()
most_backed['currency'].value_counts()
most_backed_us = most_backed[most_backed['currency']=='usd']
most_backed_us.head()
most_backed_us.sort_values(by='%_overfunded',ascending=False).head(20)
most_backed_us['amt.pledged_log'] = most_backed['amt.pledged'].apply(lambda x : np.log10(x))

most_backed_us['goal_log'] = most_backed['goal'].apply(lambda x : np.log10(x))

most_backed_us['%_overfunded_log'] = most_backed['%_overfunded'].apply(lambda x : np.log10(x))
sns.distplot(most_backed_us['amt.pledged_log'])
sns.distplot(most_backed_us['goal_log'])
sns.distplot(most_backed_us['%_overfunded_log'])
sns.jointplot('goal_log','%_overfunded_log',most_backed_us,size=10)
agg_dict = {'amt.pledged':'mean','goal':'mean','num.backers':'mean','%_overfunded':'mean','count':'sum'}

us_groupby_category = most_backed_us.groupby('category',as_index=False).agg(agg_dict).sort_values(by='amt.pledged',ascending=False)
plt.figure(figsize=(10,10))



sns.barplot(x='category',y='count',data=us_groupby_category.sort_values(by='count',ascending=False).head(20))



plt.xticks(rotation=70)
freq_category_us = us_groupby_category[us_groupby_category['count']>10]
plt.figure(figsize=(10,10))



sns.barplot(x='category',y='%_overfunded',data=freq_category_us.sort_values(by='%_overfunded',ascending=False).head(20))



plt.xticks(rotation=70)
plt.figure(figsize=(10,10))



sns.barplot(x='category',y='%_overfunded',data=freq_category_us.sort_values(by='%_overfunded',ascending=False).iloc[2:].head(20))



plt.xticks(rotation=70)
sns.heatmap(most_backed_us.corr(),annot=True)
sns.heatmap(freq_category_us.corr(),annot=True)