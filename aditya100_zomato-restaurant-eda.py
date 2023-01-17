import numpy as np 

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns
df = pd.read_csv('../input/zomato.csv')

df.head()
# Cleaning the rate feature

data = df

data.rate = data.rate.replace("NEW", np.nan)

data.dropna(how ='any', inplace = True)
# converting it from '4.1/5' to '4.1'

data['rate'] = data.loc[:,'rate'].replace('[ ]','',regex = True)

data['rate'] = data['rate'].astype(str)

data['rate'] = data['rate'].apply(lambda r: r.replace('/5',''))

data['rate'] = data['rate'].apply(lambda r: float(r))
# Converting into integer

data['approx_cost(for two people)'] = data['approx_cost(for two people)'].str.replace(',','')

data['approx_cost(for two people)'] = data['approx_cost(for two people)'].astype(int)
data.head()
del data['url']

del data['address']

del data['phone']

del data['listed_in(city)']
# Renaming features for convenience

data.rename(columns={'approx_cost(for two people)': 'approx_cost','listed_in(type)': 'type'}, inplace=True)
data.head()
_ = sns.countplot(data['online_order'])
_ = sns.countplot(data['book_table'])
_ = sns.distplot(data['rate'], color="m")
_ = sns.distplot(data['votes'], hist=False, color="c", kde_kws={"shade": True})
plt.figure(figsize=(15, 10))

p = sns.countplot(data['location'])

_ = plt.setp(p.get_xticklabels(), rotation=90)
plt.figure(figsize=(15, 10))

p = sns.countplot(data['rest_type'])

_ = plt.setp(p.get_xticklabels(), rotation=90)
_ = sns.distplot(data['approx_cost'], hist=False, color="g", kde_kws={"shade": True})
count = data['type'].value_counts().sort_values(ascending=True)

slices = [count[6], count[5], count[4], count[3], count[2], count[1], count[0]]

labels = ['Delivery ', 'Dine-out', 'Desserts', 'Cafes', 'Drinks & nightlife', 'Buffet', 'Pubs and bars']
plt.figure(figsize=(20, 10))

_ = plt.pie(slices, labels=labels, autopct='%1.1f%%', shadow=True, startangle=90)
cost_count = {}



for idx, d in data.iterrows():

    if d['location'] not in cost_count:

        cost_count[d['location']] = [d['approx_cost']]

    else:

        cost_count[d['location']].append(d['approx_cost'])
avg_cost = {}



for key in cost_count.keys():

    avg_cost[key] = sum(cost_count[key])/len(cost_count[key])
fig, axs = plt.subplots(2, 1, figsize=(20, 10), sharex=True)



p = sns.scatterplot(x=list(avg_cost.keys()), y=list(avg_cost.values()), color='r', ax=axs[0])

q = sns.barplot(x=list(avg_cost.keys()), y=list(avg_cost.values()), ax=axs[1])



_ = plt.setp(p.get_xticklabels(), rotation=90)

_ = plt.setp(q.get_xticklabels(), rotation=90)
fig, axs = plt.subplots(2, 1, figsize=(20, 10), sharex=True)

p = sns.scatterplot(x=data['approx_cost'], y=data['votes'], hue=data['rate'], ax=axs[0])

q = sns.scatterplot(x=data['approx_cost'], y=data['rate'], size=data['votes'], sizes=(10, 100), ax=axs[1])
len(data['type'].unique())

types = data['type'].unique()
types_rating = {}



for idx, d in data.iterrows():

    if d['type'] not in types_rating:

        types_rating[d['type']] = [d['rate']]

    else:

        types_rating[d['type']].append(d['rate'])
fig, axs = plt.subplots(3, 3, figsize=(15, 10), sharex=True)



p = sns.distplot(types_rating[types[0]], hist=False, color="g", kde_kws={"shade": True}, ax=axs[0, 0])

_ = p.set_title(types[0])

q = sns.distplot(types_rating[types[1]], hist=False, color="r", kde_kws={"shade": True}, ax=axs[0, 1])

_ = q.set_title(types[1])

r = sns.distplot(types_rating[types[2]], hist=False, color="b", kde_kws={"shade": True}, ax=axs[0, 2])

_ = r.set_title(types[2])

s = sns.distplot(types_rating[types[3]], hist=False, color="c", kde_kws={"shade": True}, ax=axs[1, 0])

_ = s.set_title(types[3])

t = sns.distplot(types_rating[types[4]], hist=False, color="y", kde_kws={"shade": True}, ax=axs[1, 1])

_ = t.set_title(types[4])

u = sns.distplot(types_rating[types[5]], hist=False, color="m", kde_kws={"shade": True}, ax=axs[1, 2])

_ = u.set_title(types[5])

r = sns.distplot(types_rating[types[6]], hist=False, color="c", kde_kws={"shade": True}, ax=axs[2, 0])

_ = r.set_title(types[6])