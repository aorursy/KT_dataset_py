# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import json

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



biz_file = open('../input/yelp_academic_dataset_business.json')

biz_df = pd.DataFrame([json.loads(x) for x in biz_file.readlines()])

biz_file.close()



# Any results you write to the current directory are saved as output.
biz_df.head(10)
import matplotlib.pyplot as plt

import seaborn as sns



sns.relplot(x="review_count", y="stars", data=biz_df);




sns.set_style('whitegrid')

fig, ax = plt.subplots()

biz_df['review_count'].hist(ax=ax, bins=100)

ax.set_yscale('log')

ax.tick_params(labelsize=14)

ax.set_xlabel('Review Count', fontsize=14)

ax.set_ylabel('Occurrence', fontsize=14)

deciles = biz_df['review_count'].quantile([.1, .2, .3, .4, .5, .6, .7, .8, .9])

print(deciles)
biz_df['log_review_count'] = np.log(biz_df['review_count'])

fig, (ax1, ax2) = plt.subplots(2,1)

biz_df['review_count'].hist(ax=ax1, bins=100)

ax1.tick_params(labelsize=14)

ax1.set_xlabel('review_count', fontsize=14)

ax1.set_ylabel('Occurrence', fontsize=14)

biz_df['log_review_count'].hist(ax=ax2, bins=100)

ax2.tick_params(labelsize=14)

ax2.set_xlabel('log10(review_count))', fontsize=14)

ax2.set_ylabel('Occurrence', fontsize=14)

from scipy import stats



biz_df['review_count'].min() #should be positive

rc_log = stats.boxcox(biz_df['review_count'], lmbda=0) #set it to 0 inititally,

rc_bc, bc_params = stats.boxcox(biz_df['review_count']) #find the optimal lambda value to make the transform as close to normal distro as possible

bc_params

#Visualization of the Yelp review_count after using box-cox

biz_df['review_count_box_cox'], bc_params = stats.boxcox(biz_df['review_count'])



fig, (ax3) = plt.subplots(1,1)

biz_df['review_count_box_cox'].hist(ax=ax3, bins=100)

ax3.set_yscale('log')

ax3.tick_params(labelsize=14)

ax3.set_title('Box-Cox Transformed Counts Histogram', fontsize=14)

ax3.set_xlabel('')

ax3.set_ylabel('Occurrence', fontsize=14)
fig2, (ax1, ax2, ax3) = plt.subplots(3,1)

prob1 = stats.probplot(biz_df['review_count'], dist=stats.norm, plot=ax1)

ax1.set_xlabel('')

ax1.set_title('Probplot against normal distribution')

prob2 = stats.probplot(biz_df['log_review_count'], dist=stats.norm, plot=ax2)

ax2.set_xlabel('')

ax2.set_title('Probplot after log transform')

prob3 = stats.probplot(biz_df['review_count_box_cox'], dist=stats.norm, plot=ax3)

ax3.set_xlabel('Theoretical quantiles')

ax3.set_title('Probplot after Box-Cox transform')