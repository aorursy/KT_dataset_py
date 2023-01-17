# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
live_df= pd.read_csv('../input/live.csv')

live_df.head()
backed_df= pd.read_csv('../input/most_backed.csv')

backed_total = backed_df.count()

backed_total_plg = backed_df.sum()['amt.pledged']

backed_total_backers = backed_df.sum()['num.backers']

backed_df.head()
category_df = backed_df.groupby('category')

category_df
category_number = category_df.count()

category_sum = category_df.sum()

category_mean = category_df.mean()

print ("Total number of categories is", category_number['Unnamed: 0'].count())
def plot_category_stat(df,stat):

    plt.figure(figsize=(6,20))

    df.plot(kind='barh') #category name is alreayd in

    plt.ylabel('Category', fontsize=10)

    plt.tick_params(labelsize=10)

    plt.title(stat + 'of most backed projects in 115 categories')
category_number_rank = category_number['Unnamed: 0'].sort_values()

plot_category_stat(category_number_rank,'Total number ')
category_number_rank[-3:].sum()
category_number_rank[-3:].sum()/backed_total[0]
category_sum_rank_by_plg = category_sum['amt.pledged'].sort_values()

plot_category_stat(category_sum_rank_by_plg,'Total amount of pledge ')
category_sum_rank_by_plg[-3:].sum()/backed_total_plg
category_mean_rank_by_plg = category_mean['amt.pledged'].sort_values()

plot_category_stat(category_mean_rank_by_plg,'Average amount of pledge ')
category_mean_rank_by_plg[-3:]
category_number.loc['Television'][0]
category_number.loc['Gaming Hardware'][0]
category_sum_rank_by_backer = category_sum['num.backers'].sort_values()

plot_category_stat(category_sum_rank_by_backer,'Average number of backers ')
category_sum_rank_by_backer[-3:]
category_mean_rank_by_backer = category_mean['num.backers'].sort_values()

plot_category_stat(category_mean_rank_by_backer,'Average number of backers')
backed_df['amt.pledged'].hist(bins=10)