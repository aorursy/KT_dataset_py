import json

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn import linear_model

from sklearn.model_selection import cross_val_score



with open('../input/yelp_academic_dataset_business.json','r') as f:

    raw_text = f.readlines()

    

business_dataset = pd.DataFrame([json.loads(s) for s in raw_text])



for col in business_dataset.columns:

    print(col)# The column of this table
print(business_dataset['review_count'].describe())
business_dataset.head(10)
fig,(ax1, ax2)=plt.subplots(1,2)

sns.set_style('whitegrid')



business_dataset['review_count'].hist(ax=ax1)

ax1.tick_params(labelsize=14)

ax1.set_xlabel('Review Coount', fontsize=14)

ax1.set_ylabel('Occurence', fontsize=14)



business_dataset['review_count'].hist(ax=ax2)

# The axis scale type to apply. value : {"linear", "log", "symlog", "logit", ...}

ax2.set_yscale('log')

ax2.tick_params(labelsize=14)

ax2.set_xlabel('Review Coount in log scale', fontsize=14)

_=ax2.set_ylabel('Occurence', fontsize=14)
def get_bins_by_fixed_length(fixed_length,display=False):

    floor = np.floor(business_dataset['review_count'].min())

    ceil = np.ceil(business_dataset['review_count'].max())

    bins = int(np.ceil((ceil-floor)/fixed_length))

    if display:

         print("Start from {} to {} with {} bins".format(floor,ceil,bins))

    return bins
bins = get_bins_by_fixed_length(10)

x = pd.cut(business_dataset['review_count'],bins,labels=False)
x2 = pd.cut(business_dataset['review_count'],100,labels=False).value_counts()
deciles = business_dataset['review_count'].quantile([i*0.1 for i in range(10)])

deciles
sns.set_style('whitegrid')

fig, ax = plt.subplots()

business_dataset['review_count'].hist(ax=ax, bins=100)

for pos in deciles:

    handle = plt.axvline(pos, color='r')

ax.legend([handle], ['deciles'], fontsize=14)

ax.set_yscale('log')

ax.set_xscale('log')

ax.tick_params(labelsize=14)

ax.set_xlabel('Review Count', fontsize=14)

ax.set_ylabel('Occurence', fontsize=14)



import numpy as np

small_counts = np.random.randint(0,100,(20,2))

large_counts = np.random.randint(1e5,1e10,(20,2))
mix = np.concatenate([small_counts,large_counts])
plt.scatter(mix[:,0],mix[:,1])
business_dataset['log_review_count'] = np.log(business_dataset['review_count']+1)
model_out_log = linear_model.LinearRegression()

model_with_log = linear_model.LinearRegression()
score_out_log = cross_val_score(model_out_log,business_dataset['review_count'].values.reshape(-1,1),business_dataset['stars'],cv=10)

score_with_log = cross_val_score(model_with_log,business_dataset['log_review_count'].values.reshape(-1,1),business_dataset['stars'],cv=10)
for title, score in zip(['With log','Without log'],[score_with_log,score_out_log]):

    print(title,'mean',score.mean(),'std',score.std())