import numpy as np

import pandas as pd

import datetime as dt

import re

from nltk.util import ngrams



import matplotlib.pyplot as plt

import seaborn as sns

!pip install pywaffle

from pywaffle import Waffle

import matplotlib.dates as mdates



from mlxtend.frequent_patterns import apriori

from mlxtend.frequent_patterns import association_rules

customer = pd.read_csv('../input/quantium-data-analytics-virtual-experience-program/PurchaseBehaviour.csv')

dat = pd.read_csv('../input/quantium-data-analytics-virtual-experience-program/Transactions.csv')



customer.head()
# check for missing values

customer.isnull().sum()
dat.head()
dat.isnull().sum()
# Change excel date to real date

dat['DATE'] = pd.TimedeltaIndex(dat['DATE'], unit='d') + dt.datetime(1899, 12, 30)
dat.head(10)
# Extract product size

dat['PROD_SIZE'] = [re.search(r"[0-9]+(g|G)", p).group(0).replace('G','').replace('g','') for p in dat['PROD_NAME']]

# get unique products

dat['PROD_NAME'].unique()[:10] 
### Remove salsa dips

dat = dat[~dat['PROD_NAME'].isin(['Old El Paso Salsa   Dip Tomato Mild 300g',

'Old El Paso Salsa   Dip Chnky Tom Ht300g',

'Woolworths Mild     Salsa 300g',

'Old El Paso Salsa   Dip Tomato Med 300g',

'Woolworths Medium   Salsa 300g',

'Doritos Salsa Mild  300g',

'Doritos Salsa       Medium 300g'])].reset_index(drop=True)
### Clean up product names

# https://www.guru99.com/python-regular-expressions-complete-tutorial.html



# replace & with space and remove multiple spaces

dat['PROD_NAME'] = [" ".join(p.replace('&',' ').split()) for p in dat['PROD_NAME']]

# remove digits that are followed by grams

dat['PROD_NAME'] = [re.sub(r"\s*[0-9]+(g|G)", r"", p) for p in dat['PROD_NAME']]

def replaceWords(string):

    # specific

    string = re.sub(r"SeaSalt", "Sea Salt", string)

    string = re.sub(r"Frch/Onin", "French Onion", string)

    string = re.sub(r"Cheddr Mstrd", "Cheddar Mustard", string)

    string = re.sub(r"Jlpno Chili", "Jalapeno Chilli", string)

    string = re.sub(r"Swt/Chlli Sr/Cream", "Sweet Chilli Sour Cream", string)

    string = re.sub(r"SourCream", "Sour Cream", string)

    string = re.sub(r"Tmato Hrb Spce", "Tomato Herb Spice", string)

    string = re.sub(r"S/Cream", "Sour Cream", string)

    string = re.sub(r"ChipsFeta", "Chips Feta", string)

    string = re.sub(r"ChpsHny", "Chips Honey", string)

    string = re.sub(r"FriedChicken", "Fried Chicken", string)

    string = re.sub(r"OnionDip", "Onion Dip", string)

    string = re.sub(r"SweetChili", "Sweet Chilli", string)

    string = re.sub(r"PotatoMix", "Potato Mix", string)

    string = re.sub(r"Seasonedchicken", "Seasoned Chicken", string)

    string = re.sub(r"CutSalt/Vinegr", "Cut Salt Vinegar", string)

    string = re.sub(r"ChpsBtroot", "Chips Beetroot", string)

    string = re.sub(r"ChipsBeetroot", "Chips Beetroot", string)

    string = re.sub(r"ChpsFeta", "Chips Feta", string)

    string = re.sub(r"OnionStacked", "Onion Stacked", string)

    string = re.sub(r"Ched", "Cheddar", string)

    string = re.sub(r"Strws", "Straws", string)

    string = re.sub(r"Slt", "Salt", string)

    string = re.sub(r"Chikn", "Chicken", string)

    string = re.sub(r"Rst", "Roast", string)

    string = re.sub(r"Vinegr", "Vinegar", string)

    string = re.sub(r"Mzzrlla", "Mozzarella", string)

    string = re.sub(r"Originl", "Original", string)

    string = re.sub(r"saltd", "Salted", string)

    string = re.sub(r"Swt", "Sweet", string)

    string = re.sub(r"Chli", "Chilli", string)

    string = re.sub(r"Hony", "Honey", string)

    string = re.sub(r"Chckn", "Chicken", string)

    string = re.sub(r"Chp", "Chip", string)

    string = re.sub(r"Btroot", "Beetroot", string)

    string = re.sub(r"Chs", "Cheese", string)

    string = re.sub(r"Crm", "Cream", string)

    string = re.sub(r"Orgnl", "Original", string)



    return string



dat['PROD_NAME'] = [replaceWords(s) for s in dat['PROD_NAME']]



dat['PROD_NAME'].replace('Infzns Crn Crnchers Tangy Gcamole',

'Infuzions Corn Crunchers Tangy Guacamole', inplace=True)
dat['PROD_NAME'].unique()[:10]
def replaceBrands(string):

    # specific

    string = re.sub(r"Red Rock Deli", "RRD", string)

    string = re.sub(r"Dorito", "Doritos", string)

    string = re.sub(r"Doritoss", "Doritos", string)

    string = re.sub(r"Smith", "Smiths", string)

    string = re.sub(r"Smithss", "Smiths", string)

    string = re.sub(r"GrnWves", "Grain Waves", string)

    string = re.sub(r"Woolworths", "WW", string) 

    string = re.sub(r"Snbts", "Sunbites", string) 



    return string



# standardize common brand names

dat['PROD_NAME'] = [replaceBrands(s) for s in dat['PROD_NAME']]



# get brand name from first word

dat['brand'] = [s.split(' ')[0] for s in dat['PROD_NAME']]
dat.head()
dat.describe()
# remove outlier

dat = dat[dat['PROD_QTY'] < 200].reset_index(drop=True)
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15,4))



# Product quantity sales by brand

ax1=plt.subplot(121)

dat.groupby(['brand'], as_index=False).agg({'PROD_QTY': 'sum'}).sort_values('PROD_QTY').plot.barh(x='brand',legend=False, ax=ax1)



ax1.set_xlabel('Quantity Sold')

ax1.set_ylabel('Brand')

ax1.set_title('Quantities Sold by Brand')



ax2=plt.subplot(122)

dat.groupby(['brand'], as_index=False)[['TXN_ID']].count().sort_values('TXN_ID').plot.barh(x='brand',color='#ff7f0e', legend=False, ax=ax2)

ax2.set_xlabel('Number of Transactions')

ax2.set_ylabel('Brand')

ax2.set_title('Transactions by Brand')



plt.show()
## Plot quantities sold by date

bydate = dat.groupby('DATE').agg({'PROD_QTY': 'sum'}).reset_index()



fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15,4))



ax1=plt.subplot(121)

sns.lineplot(x="DATE", y="PROD_QTY", data=bydate, ax=ax1)

ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b-%Y"))

sns.lineplot(x="DATE", y="PROD_QTY", data=bydate[(bydate['DATE'] > '2018-11-30') & (bydate['DATE'] < '2019-01-01')], color='#2ca02c', ax=ax1)

sns.lineplot(x="DATE", y="PROD_QTY", data=bydate[(bydate['DATE'] > '2018-08-10') & (bydate['DATE'] < '2018-08-24')], color='red', ax=ax1)

sns.lineplot(x="DATE", y="PROD_QTY", data=bydate[(bydate['DATE'] > '2019-05-10') & (bydate['DATE'] < '2019-05-24')], color='red', ax=ax1)

plt.ylabel('Quantities Sold')

plt.title('Quantities Sold Throughout Whole Year')





## Plot December quantities sold

# filter december

december = bydate[bydate['DATE'].isin(pd.date_range(start="2018-12-01",end="2018-12-31").tolist())]



# fill in missing dec data

december = december.set_index('DATE').reindex(pd.date_range(start="2018-12-01",end="2018-12-31"), fill_value=0)



ax2=plt.subplot(122)

ax2.bar(december.index,december['PROD_QTY'],color='#2ca02c')

ax2.set_xticks(december.index)

ax2.xaxis.set_major_formatter(mdates.DateFormatter("%b-%d"))

ax2.xaxis.set_minor_formatter(mdates.DateFormatter("%b-%d"))

ax2.tick_params(axis='x', rotation=90) 

plt.ylabel('Quantities Sold')

plt.title('Quantities Sold in December')

plt.show()
# Product Size



fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(15,4))



ax1=plt.subplot(121)

dat.groupby('PROD_SIZE').agg({'PROD_QTY': 'sum'}).sort_values('PROD_QTY').reset_index().plot.barh(x='PROD_SIZE', legend=False, ax=ax1)

ax1.set_ylabel('Product Size (g)')

ax1.set_xlabel('Quantities Sold')



plt.show()
# Function for counting product keywords

def count_keywords(df):

    words_freq = {}

    for c,p in enumerate(df['PROD_NAME']):

        for word in p.split():

            if word in words_freq:

                words_freq[word] += df['PROD_QTY'][c]

            else:

                words_freq[word] = df['PROD_QTY'][c]

    

    return words_freq



# Function for generating ngrams

def generate_ngrams(text, n):

    words = text.split()

    return [' '.join(ngram) for ngram in list(ngrams(words, n))]



# Function for counting product bigrams

def count_bigrams(df):

    bigrams_freq = {}

    for c,p in enumerate(df['PROD_NAME']):

        for ngram in generate_ngrams(p, 2):

            if ngram in bigrams_freq:

                bigrams_freq[ngram] += df['PROD_QTY'][c]

            else:

                bigrams_freq[ngram] = df['PROD_QTY'][c]

    return bigrams_freq



words_freq = count_keywords(dat)

bigrams_freq = count_bigrams(dat)
# get top keywords

topwords = pd.DataFrame(words_freq.items(), columns=['word','freq']).sort_values('freq')

topwords = topwords[~topwords.word.isin(['Chips','Kettle','Smiths','Doritos','Pringles'])]



# get top bigrams

topbigrams = pd.DataFrame(bigrams_freq.items(), columns=['bigram','freq']).sort_values('freq')

topbigrams = topbigrams[~topbigrams.bigram.isin(['Smiths Crinkle','Doritos Corn','Thins Chips','Cobs Popd','Kettle Tortilla','Grain Waves','Kettle Sensations','Kettle Sweet','Tyrrells Crisps','Twisties Cheese'])]



# Plot

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(8,5))



ax1 = plt.subplot(121)

topwords[-10:].plot.barh(x='word', legend=False, ax=ax1)

ax1.set_xlabel('Quantities Sold')

ax1.set_ylabel('Product Keyword')



ax2 = plt.subplot(122)

topbigrams[-10:].plot.barh(x='bigram', color='#ff7f0e', legend=False, ax=ax2)

ax2.set_xlabel('Quantities Sold')

ax2.set_ylabel('Product Bigram')



fig.tight_layout(pad=3.0)

plt.suptitle('Top 10 Most Popular Product Keywords')

plt.show()
# Plot wordcloud of keywords and bigrams

from wordcloud import WordCloud



rem_list = ['Chips','Kettle','Smiths','Doritos','Pringles','Chip', 'Infuzions', 'RRD', 'Thins', 'Twisties', 'Grain', 'Waves']

[words_freq.pop(key) for key in rem_list] 



rem_list = ['Smiths Crinkle','Doritos Corn','Thins Chips','Cobs Popd','Kettle Tortilla','Grain Waves','Kettle Sensations','Kettle Sweet','Tyrrells Crisps','Twisties Cheese']

[bigrams_freq.pop(key) for key in rem_list] 



plt.figure(figsize=(15,4))

plt.subplot(121)

wc = WordCloud(background_color="black").generate_from_frequencies(words_freq)

plt.imshow(wc)

plt.subplot(122)

wc = WordCloud(background_color="black").generate_from_frequencies(bigrams_freq)

plt.imshow(wc)

plt.suptitle('Most Popular Product Keywords')

plt.show()
customer.head()
## Waffle chart for customer class

premium = dict(customer['PREMIUM_CUSTOMER'].value_counts()/len(customer)*100)



plt.figure(figsize=(7,5),

    FigureClass=Waffle, 

    rows=5, 

    values=premium, 

    colors=["#1f77b4", "#ff7f0e", "green"],

    legend={'loc': 'upper left', 'bbox_to_anchor': (1, 1)},

    icons='child', 

    font_size=15, 

    icon_legend=True)

    

plt.show()
# Plot customer lifestage counts

customer.LIFESTAGE.value_counts().plot(kind='barh', alpha=.9, color=sns.color_palette("colorblind"), title='Customer Lifestage Demographics').invert_yaxis()
# Merge

alldat = dat.merge(customer, on='LYLTY_CARD_NBR')



print('No Duplicates:', len(alldat) == len(alldat)) # check same rows, no duplicates

print('Number of Nulls:', alldat.isnull().sum().sum()) # check for nulls
alldat.head()
# Sum up for each group 

life_prem = alldat.groupby(['LIFESTAGE','PREMIUM_CUSTOMER']).agg({'TOT_SALES':'sum','PROD_QTY':'sum', 'TXN_ID':'count'}).reset_index().sort_values('TOT_SALES') # sort by TOT_SALES

life_prem['Group'] = life_prem['LIFESTAGE'] + '_' + life_prem['PREMIUM_CUSTOMER']



# sort by PROD_QTY

life_prem_qty = life_prem.sort_values('PROD_QTY')
# Function for Lollipop chart

def loll_plot(df1,x,y,xlabel,title,firstX):

    

    my_color=np.where(df1[x]==firstX, '#ff7f0e', '#1f77b4')

    my_color[0] = 'red'

    my_size=np.where(df1[x]==firstX, 70, 30)

    my_size[0] = '70'



    plt.hlines(y=np.arange(0,len(df1)),xmin=0,xmax=df1[y],color=my_color)

    plt.scatter(df1[y], np.arange(0,len(df1)), color=my_color, s=my_size)

    plt.yticks(np.arange(0,len(df1)), df1[x])

    plt.xlabel(xlabel)

    plt.title(title)
## Sales and Quantity Sold by each Segment

fig = plt.figure(figsize=(14,6))



ax1 = plt.subplot(121)

loll_plot(life_prem,'Group','TOT_SALES','Sales ($)','Total Sales','OLDER FAMILIES_Budget')

plt.xticks(ticks=[0,50000,100000,150000])



ax2 = plt.subplot(122)

loll_plot(life_prem_qty,'Group','PROD_QTY','Quantity Sold','Total Quantity Sold','OLDER FAMILIES_Budget')



fig.tight_layout(pad=3.0)

plt.show()
# Get number of unique customers in each group

life_prem_pc = alldat[['LYLTY_CARD_NBR','LIFESTAGE','PREMIUM_CUSTOMER']].drop_duplicates('LYLTY_CARD_NBR').reset_index(drop=True).groupby(['LIFESTAGE','PREMIUM_CUSTOMER']).size().reset_index(name='Count').sort_values('Count').merge(life_prem, on=['LIFESTAGE','PREMIUM_CUSTOMER'])



life_prem_pc['SALES_PER_C'] = life_prem_pc['TOT_SALES']/life_prem_pc['TXN_ID']

life_prem_pc['SALES_PER_UC'] = life_prem_pc['TOT_SALES']/life_prem_pc['Count']

life_prem_pc = life_prem_pc.sort_values('SALES_PER_C')
## Sales by each Segment per Customer and per Unique Customer

fig = plt.figure(figsize=(14,6))



ax1 = plt.subplot(121)

loll_plot(life_prem_pc,'Group','SALES_PER_C','Sales ($)','Sales Per Customer','MIDAGE SINGLES/COUPLES_Mainstream')



life_prem_pc = life_prem_pc.sort_values('SALES_PER_UC')

ax2 = plt.subplot(122)

loll_plot(life_prem_pc,'Group','SALES_PER_UC','Sales ($)','Sales Per Unique Customer','OLDER FAMILIES_Mainstream')



fig.tight_layout(pad=3.0)

plt.show()
life_prem_pc['QTY_PER_C'] = life_prem_pc['PROD_QTY']/life_prem_pc['TXN_ID']

life_prem_pc['QTY_PER_UC'] = life_prem_pc['PROD_QTY']/life_prem_pc['Count']

life_prem_pc = life_prem_pc.sort_values('QTY_PER_C')


fig = plt.figure(figsize=(14,6))



ax1 = plt.subplot(121)

loll_plot(life_prem_pc,'Group','QTY_PER_C','Quantity Sold','Quantity Purchased Per Customer','OLDER FAMILIES_Mainstream')



life_prem_pc = life_prem_pc.sort_values('QTY_PER_UC')

ax2 = plt.subplot(122)

loll_plot(life_prem_pc,'Group','QTY_PER_UC','Quantity Sold','Quantity Purchased Per Unique Customer','OLDER FAMILIES_Mainstream')



fig.tight_layout(pad=3.0)

plt.show()
life_prem_pc = life_prem_pc.sort_values('Count')

fig = plt.figure(figsize=(10,6))

ax1 = plt.subplot(121)

loll_plot(life_prem_pc,'Group','Count','Number of Unique Customers','Number of Unique Customers','YOUNG SINGLES/COUPLES_Mainstream')
alldat['PRICE_PER_UNIT'] = alldat['TOT_SALES']/alldat['PROD_QTY'] # get price per unit

# get price per unit of each customer then groupby lifestage and premium_customer to get average per group

price_per_unit = alldat.groupby('LYLTY_CARD_NBR').agg({'PRICE_PER_UNIT':'mean'}).reset_index().merge(alldat[['LYLTY_CARD_NBR','LIFESTAGE','PREMIUM_CUSTOMER']], on='LYLTY_CARD_NBR').groupby(['LIFESTAGE','PREMIUM_CUSTOMER']).agg({'PRICE_PER_UNIT':'mean'}).reset_index().sort_values('PRICE_PER_UNIT')

price_per_unit['Group'] = price_per_unit['LIFESTAGE'] + '_' + price_per_unit['PREMIUM_CUSTOMER']





fig = plt.figure(figsize=(10,6))

ax1 = plt.subplot(121)

loll_plot(price_per_unit,'Group','PRICE_PER_UNIT','Price Per Qty ($)','Price Paid Per Quantity Per Unique Customer','YOUNG SINGLES/COUPLES_Mainstream')

basket = (alldat[(alldat['LIFESTAGE']=='YOUNG SINGLES/COUPLES') & (alldat['PREMIUM_CUSTOMER']=='Mainstream')]

        .groupby(['LYLTY_CARD_NBR','brand'])['PROD_QTY']

        .sum().unstack().reset_index().fillna(0)

        .set_index('LYLTY_CARD_NBR'))



def encode_units(x):

    if x <= 0:

        return 0

    if x >= 1:

        return 1



basket_sets = basket.applymap(encode_units)

basket_sets

frequent_itemsets = apriori(basket_sets, min_support=0.07, use_colnames=True)

frequent_itemsets
rules = association_rules(frequent_itemsets, metric="lift")

rules.head()
## Affinity to Brand



young_mainstream = alldat[(alldat['LIFESTAGE']=='YOUNG SINGLES/COUPLES') & (alldat['PREMIUM_CUSTOMER']=='Mainstream')]

quantity_bybrand = young_mainstream.groupby(['brand'])[['PROD_QTY']].sum().reset_index()

quantity_bybrand.PROD_QTY = quantity_bybrand.PROD_QTY / young_mainstream.PROD_QTY.sum()

quantity_bybrand = quantity_bybrand.rename(columns={"PROD_QTY": "Targeted_Segment"})



other_segments = pd.concat([alldat, young_mainstream]).drop_duplicates(keep=False) # remove young_mainsream

quantity_bybrand_other = other_segments.groupby(['brand'])[['PROD_QTY']].sum().reset_index()

quantity_bybrand_other.PROD_QTY = quantity_bybrand_other.PROD_QTY / other_segments.PROD_QTY.sum()

quantity_bybrand_other = quantity_bybrand_other.rename(columns={"PROD_QTY": "Other_Segment"})



quantity_bybrand = quantity_bybrand.merge(quantity_bybrand_other, on ='brand')

quantity_bybrand['Affinitytobrand'] = quantity_bybrand['Targeted_Segment'] / quantity_bybrand['Other_Segment']

quantity_bybrand = quantity_bybrand.sort_values('Affinitytobrand')



quantity_bybrand.head()

## Affinity to Size



quantity_bysize = young_mainstream.groupby(['PROD_SIZE'])[['PROD_QTY']].sum().reset_index()

quantity_bysize.PROD_QTY = quantity_bysize.PROD_QTY / young_mainstream.PROD_QTY.sum()

quantity_bysize = quantity_bysize.rename(columns={"PROD_QTY": "Targeted_Segment"})



quantity_bysize_other = other_segments.groupby(['PROD_SIZE'])[['PROD_QTY']].sum().reset_index()

quantity_bysize_other.PROD_QTY = quantity_bysize_other.PROD_QTY / other_segments.PROD_QTY.sum()

quantity_bysize_other = quantity_bysize_other.rename(columns={"PROD_QTY": "Other_Segment"})



quantity_bysize = quantity_bysize.merge(quantity_bysize_other, on='PROD_SIZE')

quantity_bysize['Affinitytosize'] = quantity_bysize['Targeted_Segment'] / quantity_bysize['Other_Segment']

quantity_bysize = quantity_bysize.sort_values('Affinitytosize')


fig = plt.figure(figsize=(10,6))

ax1 = plt.subplot(121)

loll_plot(quantity_bybrand,'brand','Affinitytobrand','Affinity','Affinity To Brand','Tyrrells')



ax2 = plt.subplot(122)

loll_plot(quantity_bysize,'PROD_SIZE','Affinitytosize','Affinity','Affinity To Product Size','270')



plt.suptitle('Young Singles/Couples Mainstream')

plt.show()


