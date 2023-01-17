import pandas as pd

import seaborn as sns

import numpy as np

import matplotlib.pyplot as plt ## pyplot is a submodule within matplotlib

%matplotlib inline 

## this allows us to print the plots in the Jupyter Notebook
item_data = pd.read_csv('../input/patagoniaclothing/Patagonia_WebScrape_ClothingItems_v1.csv')
item_data.head()
genders = ['womens', 'mens']

for gender in genders:

    subset = item_data[item_data['item_gender'] == gender]

    with sns.color_palette("muted"): ## changes the color palette

        sns.distplot(subset['item_price'], 

                 hist = False, 

                 kde = True, 

                 kde_kws = {'linewidth':3, 'shade': True}, 

                 label = gender)
with sns.color_palette("muted"):

    sns.boxplot(

        x = 'item_gender', 

        y = 'item_price',

        data = item_data)
genders = ['womens', 'mens'] ## to plot the men and womens colors separately

for gender in genders:

    subset = item_data[item_data['item_gender'] == gender]

    with sns.color_palette("muted"): ## changes the color

        sns.distplot(subset['item_colors'], 

                     hist = False, 

                     kde = True, 

                     kde_kws = {'linewidth':3, 'shade': True}, 

                     label = gender)
with sns.color_palette("muted"):

    sns.boxplot(

        x = 'item_gender', 

        y = 'item_colors',

        data = item_data)
## we'll want to do this for prices and colors so we can write a little function



def item_ef(category):

    """Prints mean, std, and effect size of the mens and womens item information"""

    

    w_info = item_data[category].loc[item_data['item_gender'] == 'womens']

    m_info = item_data[category].loc[item_data['item_gender'] == 'mens']

    

    ## the .3f will round the answer to 3 decimal places

    print('Womens Items: Mean=%.3f, Standard Deviation=%.3f' % (np.mean(w_info), np.std(w_info)))

    print('Mens Items: Mean=%.3f, Standard Deviation=%.3f' % (np.mean(m_info), np.std(m_info)))

    print('Difference in means=%.3f' % (np.mean(m_info) - np.mean(w_info)))
item_ef('item_price')
from scipy.stats import mannwhitneyu
def item_mannwhitney(category):

    """Prints statistic and P value for Mann Whitney U test for diff in mens and womens items"""

    

    w_info = item_data[category].loc[item_data['item_gender'] == 'womens']

    m_info = item_data[category].loc[item_data['item_gender'] == 'mens']

    

    ## the .3f will round the answer to 3 decimal places

    stat, p = mannwhitneyu(w_info, m_info)

    print('Statistic=%.3f, p-value=%.3f' % (stat, p))
item_mannwhitney('item_price')
item_ef('item_colors')
item_mannwhitney('item_colors')