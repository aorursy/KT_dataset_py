# Libraries
import pandas as pd
import numpy as np
import geopandas as gpd

import matplotlib.pyplot as plt
from matplotlib import gridspec
import seaborn as sns
sns.set()
%matplotlib inline

from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS
# Data
loans = pd.read_csv("../input/kiva_loans.csv")
loan_themes = pd.read_csv("../input/loan_theme_ids.csv")
world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
world = world.rename(index=str, columns={'name': 'country'})
world = world.set_index('country')
world = world.drop(['pop_est', 'gdp_md_est'], axis=1)

loans_per_country = loans['country'].value_counts()
no_countries = []
for country in loans_per_country.index:
    if country not in world.index: no_countries.append(country)

world = world.rename(index={'Dem. Rep. Congo' : 'The Democratic Republic of the Congo', 'Myanmar': 'Myanmar (Burma)',
                   'Lao PDR': "Lao People's Democratic Republic", 'Solomon Is.': 'Solomon Islands',
                   'Dominican Rep.': 'Dominican Republic', 'S. Sudan': 'South Sudan', "Côte d'Ivoire": "Cote D'Ivoire"})
# No Samoa, Saint Vincent and the Grenadines, Virgin Islands, Guam
world['loans_per_country'] = loans_per_country
world.loc[world['loans_per_country'].isnull(), 'loans_per_country'] = 0

ax = world.plot(edgecolor='black', figsize=(14,14), column='loans_per_country', cmap='OrRd', scheme='quantiles')
ax.set_title('Number of loans by country', fontsize=14)
ax.set_facecolor(sns.color_palette("Blues")[0])
ax.grid(False)
plt.show()
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(11, 8))

# Data
n_loans_8 = loans['country'].value_counts().sort_values(ascending=False)[:8]
lender_count_mean = loans.groupby('country')['lender_count'].mean().loc[n_loans_8.index]
loan_amount_8 = loans.groupby(['country']).sum()['funded_amount'].sort_values(ascending=False)[:8]
funded_amount_mean = loans.groupby('country')['funded_amount'].mean().loc[loan_amount_8.index]

my_pal = {}
for i, key in enumerate(n_loans_8.keys()):
    my_pal[key] = sns.color_palette()[i%len(sns.color_palette())]
for i, key in enumerate(loan_amount_8.keys()):
    my_pal[key] = sns.color_palette()[i%len(sns.color_palette())]

# First row
sns.barplot(x=n_loans_8, y=n_loans_8.keys(), palette=my_pal, ax=axes[0,0])
sns.barplot(x=lender_count_mean, y=lender_count_mean.keys(), palette=my_pal, ax=axes[0,1])

axes[0,0].set_title("Total number of loans by country", fontsize=14)
axes[0,0].set_ylabel('Country')
axes[0,0].set_xlabel('Number of loans')
axes[0,0].tick_params(axis='y', direction='in', pad=-3, labelsize = 13)
axes[0,0].set_yticklabels(n_loans_8.keys(), horizontalalignment = "left", color="white")
axes[0,0].grid(False)
axes[0,0].set_axisbelow('line')

axes[0,1].set_title('Average number of lenders by country', fontsize=14)
axes[0,1].set_xlabel('Average number of lenders')
axes[0,1].set_ylabel('')
axes[0,1].set_yticklabels(lender_count_mean.keys(), fontsize=12)

# Second row
sns.barplot(x=funded_amount_mean, y=funded_amount_mean.keys(), palette=my_pal, ax=axes[1,1])
sns.barplot(x=loan_amount_8, y=loan_amount_8.keys(), palette=my_pal, ax=axes[1,0])

axes[1,0].set_title("Total funded amount by country", fontsize=14)
axes[1,0].set_ylabel('')
axes[1,0].set_xlabel('Total funded amount')
axes[1,0].tick_params(axis='y', direction='in',pad=-3, labelsize = 13)
axes[1,0].set_yticklabels(loan_amount_8.keys(), horizontalalignment = "left", color="white")
axes[1,0].grid(False)
axes[1,0].set_axisbelow('line')
axes[1,0].set_ylabel('Country')

axes[1,1].set_title('Average funded amount per loan by country', fontsize=14)
axes[1,1].set_xlabel('Average funded amount')
axes[1,1].set_ylabel('')
axes[1,1].set_yticklabels(funded_amount_mean.keys(), fontsize=12)

plt.tight_layout()
plt.show()
loans[loans['funded_amount'] < 5000]['funded_amount'].plot(kind='hist', bins=100, figsize=(12, 4))
plt.title('Loans distribution', fontsize=14)
plt.ylabel('Number of loans')
plt.xlabel('Funded amount')
plt.xlim(0, 5000)
plt.show()
fig, axes = plt.subplots(nrows=2, ncols=2)
gs = gridspec.GridSpec(2, 2, width_ratios=[2, 1]) 
axes[0,0] = plt.subplot(gs[0])
axes[0,1] = plt.subplot(gs[1])
axes[1,0] = plt.subplot(gs[2])
axes[1,1] = plt.subplot(gs[3])

# Data
activity = loans['activity'].value_counts().sort_values(ascending=False)[:8][::-1]
sector =  loans['sector'].value_counts().sort_values(ascending=False)[:8][::-1]
sector.plot(kind="barh", figsize=(11,8), fontsize = 11, ax=axes[1,0], width=0.65)
distr_list_sec = []
for act_ind in sector.index:
    ar = np.array(loans[(loans['sector'] == act_ind) & (loans['funded_amount'] < 2000)]['funded_amount'])
    distr_list_sec.append(ar)

activity.plot(kind="barh", figsize=(11,8), fontsize = 11, ax=axes[0,0], width=0.65)
distr_list_act = []
for act_ind in activity.index:
    ar = np.array(loans[(loans['activity'] == act_ind) & (loans['funded_amount'] < 2000)]['funded_amount'])
    distr_list_act.append(ar)
    
# First row
axes[0,0].set_title("Number of loans by activity", fontsize=14)
axes[0,0].set_ylabel('Activity')
axes[0,0].set_xlabel('')
axes[0,0].tick_params(axis='y', direction='in',pad=-3, labelsize = 13)
axes[0,0].set_yticklabels(activity.keys(), horizontalalignment = "left", color="white")
axes[0,0].grid(False)
axes[0,0].set_axisbelow('line')

axes[0,1].boxplot(distr_list_act, 0, 'rs', 0, flierprops={'alpha':0.6, 'markersize': 2, 'markeredgecolor': 'None',
                                                          'marker': '.'}, patch_artist=True, medianprops={'color': 'black'})
axes[0,1].set_title('Funded loan distribution\nby activity', fontsize=14)
axes[0,1].set_ylabel('')
axes[0,1].set_xlabel('')
axes[0,1].set_yticklabels(activity.index, fontsize=12)

# Second row
axes[1,0].set_title("Number of loans by sector", fontsize=14)
axes[1,0].set_ylabel('Sector')
axes[1,0].set_xlabel('Number of loans')
axes[1,0].tick_params(axis='y', direction='in',pad=-3, labelsize = 13)
axes[1,0].set_yticklabels(sector.keys(), horizontalalignment = "left", color="white")
axes[1,0].grid(False)
axes[1,0].set_axisbelow('line')

axes[1,1].boxplot(distr_list_sec, 0, 'rs', 0, flierprops={'alpha':0.6, 'markersize': 2, 'markeredgecolor': 'None',
                                                          'marker': '.'}, patch_artist=True, medianprops={'color': 'black'})
axes[1,1].set_title('Funded loan distribution\nby sector', fontsize=14)
axes[1,1].set_ylabel('')
axes[1,1].set_xlabel('Funded loan')
axes[1,1].set_yticklabels(sector.index, fontsize=12)

plt.tight_layout()
plt.show()
fig, axes = plt.subplots(nrows=1, ncols=2)

loans[loans['lender_count'] < 100]['lender_count'].plot(kind='hist', bins=100, xticks=np.linspace(0, 100, num=11),
                                                       figsize=(10,4), ax=axes[0])
axes[0].set_title('Number of lenders per loan', size=14)
axes[0].set_ylabel('Number of loans')
axes[0].set_xlabel('Number of lenders')
axes[0].set_xlim(0)

lenders_percent = round(loans['lender_count'].value_counts().sort_index()[1:13]/loans['lender_count'].value_counts().sum()*100,2)
axes[1].axis('off')

rowlabels = []
for lenders in lenders_percent.index:
    label = str(lenders) + ' ' + 'lender'
    if lenders > 1: label += 's'
    rowlabels.append(label)
    
celltext = []
for fract in lenders_percent:
    text = str(fract) + '%'
    celltext.append(text)
celltext = np.array(celltext).reshape(12, 1)    
    
table = axes[1].table(cellText=celltext, colLabels=["Loans"], rowLabels=rowlabels, loc='center',
                         cellLoc='center', colWidths = [0.2])
table.set_fontsize(14)
table.scale(1.5, 1.5)
plt.tight_layout()

plt.show()
fig, axes = plt.subplots(nrows=1, ncols=2)
# Data
loans_by_lenders = loans.groupby('lender_count').mean()['funded_amount'][:29]
loans_by_person = loans_by_lenders / loans_by_lenders.index

# First row
loans_by_lenders.plot(kind='bar', color=sns.color_palette("Blues")[-1],
                      figsize=(11,4), ax=axes[0])

axes[0].set_title('Average funded amount per loan', size=14)
axes[0].set_ylabel('Average funded amount')
axes[0].set_xlabel('Number of lenders')

loans_by_person.plot(kind='bar', color=sns.color_palette("Blues")[-1], ax=axes[1], figsize=(11,4))
axes[1].set_title('Average funded amount per person', size=14)
axes[1].set_ylabel('')
axes[1].set_xlabel('Number of lenders')

plt.tight_layout()
plt.show()
repay_interval = loans['repayment_interval'].value_counts()
ax = repay_interval.plot(kind='pie', figsize=(5,5), startangle=90, labels=None)
plt.ylabel('')
plt.title('Repayment interval', size=14)
labels = []
for key in repay_interval.keys():
    percent = int(repay_interval[key]/repay_interval.sum()*100)
    if percent == 0: percent = '<1'
    label = key.capitalize() + ' ' +str(percent) + '%'
    labels.append(label)
ax.legend(labels=labels, prop={'size': 14}, bbox_to_anchor=(1.5,0.7))
plt.show()
loans[loans['term_in_months'] < 45]['term_in_months'].plot(kind='hist', bins=45, xticks=np.linspace(0, 45, num=10), 
                                                           figsize=(13, 4))
plt.title('Terms of loans', fontsize=14)
plt.xlabel('Number of months')
plt.ylabel('Number of loans')
plt.show()
use_15 = loans['use'].str.lower().str.replace('.', '').str.replace(',', '').value_counts()[:15][::-1]
use_15.plot(kind='barh', figsize=(4, 6), fontsize=12, width=0.65)
plt.title('Purpose of loan', fontsize=14)
plt.xlabel('Number of loans')
plt.show()
dictionary = pd.Series(' '.join(loans[loans['use'].notnull()]['use']).replace(',', ' ').replace('.', ' ')
                       .lower().split()).value_counts()[:100]
dictionary = dictionary[(dictionary.index.isin(ENGLISH_STOP_WORDS) == False) & (dictionary > 100)]
dictionary[:30].plot(kind="barh", figsize=(12,8), fontsize = 12, width=0.65)
plt.title('Most common words used', fontsize=14)
plt.xlabel('Times mentioned')
plt.ylabel('Word')
plt.show()
loan_themes = pd.merge(loans[['id', 'funded_amount', 'country', 'lender_count']], 
                       loan_themes[['id', 'Loan Theme Type']], how='left', on=['id'])
loan_themes = loan_themes.rename(index=str, columns={'Loan Theme Type': 'theme'})
loan_themes.head()
# Data
loan_themes_num = loan_themes['theme'].value_counts()[:10]
loan_themes_fund = loan_themes.groupby('theme').sum()['funded_amount'].sort_values(ascending=False)[:10]
loan_themes_10 = loan_themes_fund.index
loan_themes_fund_mean = loan_themes.groupby('theme').mean().loc[loan_themes_10]['funded_amount']
loan_themes_lend_mean = loan_themes.groupby('theme').mean().loc[loan_themes_10]['lender_count']

my_pal = {}
for i, key in enumerate(loan_themes_num.keys()):
    my_pal[key] = sns.color_palette()[i%len(sns.color_palette())]
for i, key in enumerate(loan_themes_fund.keys()):
    my_pal[key] = sns.color_palette()[i%len(sns.color_palette())]

fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(11, 8))

# First row
sns.barplot( x=loan_themes_num, y=loan_themes_num.keys(), palette=my_pal, ax=axes[0,0])
sns.barplot( x=loan_themes_fund, y=loan_themes_fund.keys(), palette=my_pal, ax=axes[0,1])

axes[0,0].set_title('Number of loans\nby loan theme type', fontsize=14)
axes[0,0].set_ylabel('Loan theme type')
axes[0,0].set_xlabel('Number of loans')
axes[0,0].set_xticks(np.linspace(0, 300000, num=4))
axes[0,0].tick_params(labelsize=12)

axes[0,1].set_title('Funded amount\nby loan theme type', fontsize=14)
axes[0,1].set_ylabel('')
axes[0,1].set_xlabel('Funded amount')
axes[0,1].tick_params(labelsize=12)

# Second row
sns.barplot( x=loan_themes_lend_mean, y=loan_themes_lend_mean.keys(), palette=my_pal, ax=axes[1,0])
sns.barplot( x=loan_themes_fund_mean, y=loan_themes_fund_mean.keys(), palette=my_pal, ax=axes[1,1])

axes[1,0].set_title('Average number of lenders\nby loan theme type', fontsize=14)
axes[1,0].set_ylabel('Loan theme type')
axes[1,0].set_xlabel('Average number of lenders')
axes[1,0].tick_params(labelsize=12)

axes[1,1].set_title('Average funded amount\nby loan theme type', fontsize=14)
axes[1,1].set_ylabel('')
axes[1,1].set_xlabel('Average fund')
axes[1,1].tick_params(labelsize=12)

plt.tight_layout()
plt.show()