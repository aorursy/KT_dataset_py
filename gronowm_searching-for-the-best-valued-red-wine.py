#import the modules we will need throughout this analysis

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

import os

print(os.listdir("../input/wine-reviews"))

wine = pd.read_csv("../input/wine-reviews/winemag-data_first150k.csv")

wine.head()
wine.info()
wine.drop('Unnamed: 0', axis=1, inplace=True) #remove unwanted column



wine[wine['country'].isnull()]
wine.drop_duplicates(subset = 'description', inplace=True)

wine.loc[1133, 'country'] = 'Greece'

wine.loc[1440, 'country'] = 'Turkey'

wine.loc[68226, 'country'] = 'Chile'



print("Number of wine reviews left after removing duplicates: ", '{:,}'.format(len(wine)))
wine['price'].fillna(wine['price'].mean(), inplace=True) #fill in 'price' null values with mean of column

wine_clean = wine.drop(['designation', 'province', 'region_1', 'region_2'], axis=1) #drop columns not needed

wine_clean['variety'] = wine_clean['variety'].str.lower()

wine_clean.head()
wine_types = pd.read_csv("../input/wine-types/wine_types.csv")

wine_types.rename(columns={'grape':'variety'}, inplace=True) #rename column to be consistent with wine_clean column

wine_clean = pd.merge(wine_clean, wine_types, how='left', on='variety') #merge databases on 'variety' column

wine_type_counts = wine_types['wine_type'].value_counts() #print out number of red and white grapes



wine_type_null = wine_clean['wine_type'].isnull().sum() #count the number of null values in the 'wine_type' column



print("There are {:,} red grapes and {:,} white grapes in our wine_type file with {:,} null values in the merged database"

      .format(wine_type_counts[0], wine_type_counts[1], wine_type_null))
wine_clean.head(10)
#list the top 25 null values in the 'wine_type' column

null_wine_types = wine_clean[wine_clean['wine_type'].isnull()]['variety']

print("Top 25 null values in wine_type column:\n", null_wine_types.value_counts().head(25))
import re



#define a function that returns the colour of a grape if the colour is in the name. 

def colour(variety):

    red = r"\b(red)\b"

    white = r"\b(white)\b"

    if re.search(red, variety):

        return 'red'

    elif re.search(white, variety):

        return 'white'



#apply the 'colour' function to the 'variety' column and assign to the 'wine_type' column if the current value is null

wine_clean['wine_type'] = wine_clean['wine_type'].mask(wine_clean['wine_type'].isnull(), wine_clean['variety'].apply(colour))



#print out the 30 most common varieties with null wine_type so we can manually filter out the red and white grape varieties

print("30 most common varieties with null wine_type values:\n", wine_clean[wine_clean['wine_type'].isnull()]['variety'].value_counts().head(30).index)
#create lists of red and white grape varieties which we can use to fill in missing values in the 'wine_type' column



red_varieties = ['shiraz', 'corvina, rondinella, molinara','sangiovese grosso', 'petite sirah', 'carmenère',

                'tempranillo blend', "nero d'avola", 'garnacha', 'meritage', 'cabernet blend', 'primitivo',

                'montepulciano','cabernet sauvignon-merlot','tinta de toro','mourvèdre', 'mencía']

white_varieties = ['pinot grigio','grüner veltliner','gewürztraminer', 'albariño', 'glera', 'sémillon', 'blaufränkisch' ]



#filter wine_clean by red_varieties and assign 'red' to 'wine_type' column

wine_clean.loc[wine_clean['variety'].isin(red_varieties), 'wine_type'] = 'red'



#filter wine_clean by white_varieties and assign 'white' to 'wine_type' column

wine_clean.loc[wine_clean['variety'].isin(white_varieties), 'wine_type'] = 'white'



#delete the remaining rows which have a null value in 'wine_type

wine_clean.dropna(subset=['wine_type'], inplace=True)

wine_clean.head(10)
#create dictionary of same grapes with different names and replace key with value in table

same_grape = {'shiraz': 'syrah', 'garnacha':'grenache', 'durif':'petite sirah', 'primitivo':'zinfandel', 'sangiovese grosso':'sangiovese',

             'tempranillo blend':'tempranillo'}

wine_clean['variety'].replace(same_grape, inplace=True)



#remove rows that have generic blended varieties

wine_clean = wine_clean[(wine_clean['variety'] != 'red blend') & (wine_clean['variety'] != 'white blend')]

wine_clean.info()
sns.set_style("white")



fig, axes = plt.subplots(2,2, figsize=(20,20))

ax1 = wine_clean['wine_type'].value_counts().plot.bar(ax=axes[0,0], color=['r', 'y'])

ax1.set_title("Wine Types", fontsize=20)

ax1.tick_params(axis='both', which='major', labelsize=15, labelrotation=0)

ax1.spines['top'].set_visible(False)

ax1.spines['right'].set_visible(False)



#create a new table for a pie chart; splitting out the top 12 most common countries and binning the rest into an 'other' value.

table_for_pie_chart = wine_clean['country'].value_counts()[:12]

others_count = wine_clean['country'].value_counts()[12:].sum()

table_for_pie_chart['Other'] = others_count

axes[0,1].pie(table_for_pie_chart.values, labels = table_for_pie_chart.index, startangle=40, autopct='%.0f%%', textprops={'fontsize':14})

axes[0,1].set_title("Wines by country of origin", fontsize=20)



red_wine = wine_clean[wine_clean['wine_type'] == 'red']

ax3 = sns.countplot(data=red_wine, x='variety', ax=axes[1,0],

                    order = red_wine['variety'].value_counts().iloc[:10].index, palette='Reds_r')

ax3.set_title("Top 10 Red Wine Varieties", fontsize=20)

ax3.tick_params(axis='x', labelrotation=90)

ax3.tick_params(axis='y', labelsize=15)

ax3.set_ylabel("")

ax3.set_xlabel("")

ax3.spines['top'].set_visible(False)

ax3.spines['right'].set_visible(False)

ax3.spines['left'].set_visible(False)



white_wine = wine_clean[wine_clean['wine_type'] == 'white']

ax4 = sns.countplot(data=white_wine, x='variety', ax=axes[1,1],

                    order = white_wine['variety'].value_counts().iloc[:10].index, palette='YlOrBr')

ax4.set_title("Top 10 White Wine Varieties", fontsize=20)

ax4.tick_params(axis='x', labelrotation=90)

ax4.tick_params(axis='y', labelsize=15)

ax4.set_ylabel("")

ax4.set_xlabel("")

ax4.spines['top'].set_visible(False)

ax4.spines['right'].set_visible(False)

ax4.spines['left'].set_visible(False)



plt.tight_layout()
top_7_countries = ['US', 'France', 'Italy', 'Spain', 'Chile', 'Argentina', 'Australia']

top_10_red_grapes = red_wine['variety'].value_counts().iloc[:10].index



        

fig, ax = plt.subplots(figsize=(16,16))

red_wine_filter = red_wine[(red_wine['variety'].isin(top_10_red_grapes)) & (red_wine['country'].isin(top_7_countries))]

ax = sns.countplot(x='country', hue='variety', data=red_wine_filter)

ax.legend(loc='upper right', prop={'size':15})

ax.xaxis.grid(False)





  
print("Decile breakdown of points and price:\n",red_wine.describe(percentiles=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]))
import matplotlib.gridspec as gridspec

red_wine_under100 = red_wine[red_wine['price'] < 100].copy() #filter out wines under $100



fig = plt.figure(figsize=(20,20))

gs = gridspec.GridSpec(3,2)

ax1 = fig.add_subplot(gs[0,0])

ax2 = fig.add_subplot(gs[0,1])

ax3 = fig.add_subplot(gs[1:,:])



sns.distplot(red_wine_under100['price'], hist=True, ax=ax1, bins=30)

ax1.spines['top'].set_visible(False)

ax1.spines['right'].set_visible(False)

ax1.set_title("Price Density Plot For Red Wines Under $100", fontsize=16)

ax1.tick_params(axis='x', labelsize=12)

ax1.tick_params(axis='y', labelsize=12)

ax1.set_xlabel("Price", fontsize=14)



red_wine_under100.hist(column='points',ax=ax2, bins=10, grid=False,color='purple', alpha=0.5)

ax2.spines['top'].set_visible(False)

ax2.spines['right'].set_visible(False)

ax2.set_title("Points Distribution For Red Wines Under $100", fontsize=16)

ax2.tick_params(axis='x', labelsize=12)

ax2.tick_params(axis='y', labelsize=12)

ax2.set_xlabel("Points", fontsize=14)



top_10_under100 = red_wine_under100[red_wine_under100['variety'].isin(top_10_red_grapes)] #filter out the top 10 grape varieties

ax3 = sns.boxplot(x='variety', y='price', data=top_10_under100)

ax3.set_title("Boxplot of price of top 10 grape varieties", fontsize=16)

ax3.spines['top'].set_visible(False)

ax3.spines['right'].set_visible(False)

ax3.tick_params(axis='x', labelsize=12)

ax3.tick_params(axis='y', labelsize=12)

ax3.set_xlabel("Variety", fontsize=14)

ax3.set_ylabel("Price", fontsize=14)

ax3.xaxis.labelpad = 20

ax3.yaxis.grid(True)


red_wine_under100.loc[:,'price_bin'] = pd.cut(red_wine_under100.loc[:,'price'],10) #create 10 bins by 'price' column



bins = red_wine_under100['price_bin'].value_counts().sort_index().index

best_valued_reds = {} #dictionary which will store the bins(key) and indices(values) of the data with the max points value

best_reds_df = pd.DataFrame(best_valued_reds)

for bin in bins:

    red_bin = red_wine_under100[red_wine_under100['price_bin'] == bin]            #filter out red wines by bin

    best_reds_index = red_bin[red_bin['points'] == red_bin['points'].max()].index #find indices of highest rated wine in each bin

    best_valued_reds[bin] = best_reds_index                                     #update dictionary

    best_reds_df = pd.concat([best_reds_df, red_wine_under100.loc[best_valued_reds[bin]]]) #concatenate bin dataframes

    

best_reds_df