import pandas as pd

import numpy as np



import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

plt.style.use('fivethirtyeight')
#load data

df1 = pd.read_csv('../input/epi_r.csv')
#quick look at data

df1.head(2)
#narrowing dataset & looking at summary stats

recipes = df1.iloc[:,:10]

recipes.drop(['#cakeweek','#wasteless'], axis=1, inplace=True)



recipes.describe()
#checking nan values in each column

for i in recipes.columns:

    print(i, sum(recipes[i].isnull()))
#filling nan values with pseudo-average & removing outliers ---- may not be best method, alternative is to drop nan rows

cal_clean = recipes.loc[recipes['calories'].notnull()]



q1  = cal_clean['calories'].quantile(.25)

q3  = cal_clean['calories'].quantile(.75)

iqr = q3 - q1



for i in recipes.columns[1:6]:

    recipes[i].fillna(cal_clean.loc[(cal_clean['calories'] > q1) & (recipes['calories'] < q3)][i].mean(), inplace=True)

    

recipes = recipes.loc[(recipes['calories'] > q1-(iqr*3)) & (recipes['calories'] < q3+(iqr*3))]
#check summary stats after cleaning data

recipes.describe()
#plotting health metrics against recipe rating

dict_plt = {0:'calories',1:'protein',2:'fat',3:'sodium'}



sns.set(font_scale=.7)



fig, ax = plt.subplots(1,4, figsize=(10,3))



for i in range(4):

    sns.barplot(x='rating',y=dict_plt[i], data=recipes, ax=ax[i], errwidth=1)

    ax[i].set_title('rating by {}'.format(dict_plt[i]), size=15)

    ax[i].set_ylabel('')
five_star = recipes.loc[recipes['rating'] == 5]



print('We have {:,} 5-star recipes to choose from'.format(len(recipes.loc[recipes['rating'] == 5])))
a = pd.qcut(five_star['calories'], [0,.33,.66,1], labels=['low cal','med cal', 'high cal']).rename('cal_bin')



five_star = five_star.join(a)
low_cal = five_star.loc[five_star['cal_bin'] == 'low cal']



plt.scatter(x='calories', y='protein', s=low_cal['fat']*5, data=low_cal)



plt.xlabel('Calories')

plt.ylabel('Protein')

plt.axhspan(ymin=20, ymax=25, xmin=.48, xmax=.6, alpha=.2, color='r')

plt.axhspan(ymin=27, ymax=34, xmin=.7, xmax=.9, alpha=.4, color='r')
#light red box from chart above

low_cal.loc[(low_cal['protein'] > 20) & (low_cal['calories'] < 160)]
#dark red box from chart above

low_cal.loc[low_cal['protein'] > 27]