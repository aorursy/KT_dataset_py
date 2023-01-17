import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

sns.set(style="whitegrid")
import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
pd.set_option('display.max_columns', 100)

encoding_latin = 'latin'

ice = pd.read_csv('/kaggle/input/recipeData.csv', low_memory = False, encoding = encoding_latin)

ice.head(9)
ice.shape
ice.info()
ice.describe().T
ice.isnull().sum()
missing = round(100*(ice.isnull().sum()/len('BeerID')), 2)

missing
ice.columns
ice_missing = ice.copy()

ice_missing = ice_missing.T

true = ice_missing.isnull().sum(axis=1)

false = (len(ice_missing.columns) - true)

ice_missing['Valid Count'] = false / len(ice_missing.columns)

ice_missing['NA Count'] = true / len(ice_missing.columns)



ice_missing[['NA Count','Valid Count']].sort_values(

    'NA Count', ascending=False).plot.bar(

    stacked=True,figsize=(12,6))

plt.legend(loc=9)

plt.ylim(0,1.15)

plt.title('Normed Missing Values Count', fontsize=20)

plt.xlabel('Normed (%) count', fontsize=20)

plt.ylabel('Column name', fontsize=20)

plt.xticks(rotation=60)

plt.show()



ice = ice[pd.notnull(ice['Style'])]
gb_style = ice.groupby(['Style']).count().sort_values(['BeerID'], ascending=False).reset_index()[:20]

gb_style['BeerID'] = (gb_style['BeerID'] / len(ice)) * 100



plt.figure(figsize=(12,8))

g = sns.barplot(x=gb_style['BeerID'], y=gb_style['Style'], orient='h')

plt.title('Normed Style Popularity (%) for 20 most popular Styles', fontsize=22)

plt.ylabel('Style Name', fontsize=20)

plt.xlabel('Normed Style Popularity (%)', fontsize=20)



plt.xlim(0,18)



for index, row in gb_style.iterrows():

    g.text(y=index+0.2,x=row['BeerID']+1,s='{:.2f}%'.format(row['BeerID']),

           color='black', ha="center", fontsize=16)



plt.show()
general_styles = ['Amber Ale','Pale Ale','Red Ale','Cider','Spice Beer',

                  'IPA','Lager','Specialty','Porter','Wheat Beer']

general_styles_dict = {'Brown':'Red','Fruit':'Spice', 'Stout':'Porter'}



ice_general_styles = ice.copy()

ice_general_styles['Style_aux'] = 'Other'

for style in general_styles:

    ice_general_styles.loc[ice_general_styles['Style'].str.contains(style), 'Style_aux'] = style

for key in general_styles_dict:

    ice_general_styles.loc[ice_general_styles['Style'].str.contains('{} Ale'.format(key)), 'Style_aux'] = '{} Ale'.format(general_styles_dict[key])



ice_general_styles = ice_general_styles[ice_general_styles['Style_aux']!='Other']

gb_style = ice_general_styles.groupby(['Style_aux']).count().sort_values(['BeerID'], ascending=False).reset_index()[:20]

gb_style['BeerID'] = (gb_style['BeerID'] / len(ice)) * 100



plt.figure(figsize=(12,6))

g=sns.barplot(x=gb_style['BeerID'], y=gb_style['Style_aux'], orient='h')

plt.title('Normed Style Popularity (%) for GENERAL Styles', fontsize=25)

plt.ylabel('Style Name', fontsize=20)

plt.xlabel('Normed Style Popularity (%)', fontsize=20)

plt.xlim(0,22.5)



for index, row in gb_style.iterrows():

    g.text(y=index+0.1,x=row['BeerID']+1,s='{:.2f}%'.format(row['BeerID']),

           color='black', ha="center", fontsize=16)



plt.show()
plt.figure(figsize=(12,12))

count=0

for col, color in zip(['OG', 'FG', 'ABV', 'IBU','Color'],['b','y','c','m','g']):

    count+=1

    if(count==5):

        plt.subplot(3,2,(5,6))

    else:

        plt.subplot(3,2,count)

    sns.distplot(ice[col], bins=100, label=col, color=color)

    plt.title('{} Distribution'.format(col), fontsize=15)

    plt.legend()

    plt.ylabel('Normed Frequency', fontsize=15)

    plt.xlabel(col, fontsize=15)



plt.subplots_adjust(wspace=0.5, hspace=0.5)

plt.show()
plt.figure(figsize=(12,12))

count=0

for col, color in zip(['OG', 'FG', 'ABV', 'IBU','Color'],['b','y','c','m','g']):

    count+=1

    if(count==5):

        plt.subplot(3,2,(5,6))

    else:

        plt.subplot(3,2,count)

    sns.distplot(np.log1p(ice[col]), bins=100, label=col, color=color)

    plt.title('Log(1 + {}) Distribution'.format(col), fontsize=15)

    plt.legend()

    plt.ylabel('Normed Frequency', fontsize=15)

    plt.xlabel('Log(1 + {})'.format(col), fontsize=15)



plt.subplots_adjust(wspace=0.5, hspace=0.5)

plt.show()
general_styles = ['Amber Ale','Pale Ale','Red Ale','Cider','Spice Beer',

                  'IPA','Lager','Specialty','Porter','Wheat Beer']

general_styles_dict = {'Brown':'Red','Fruit':'Spice', 'Stout':'Porter'}



ice_general_styles = ice.copy()

ice['Style_aux'] = 'Other'

for style in general_styles:

    ice_general_styles.loc[ice_general_styles['Style'].str.contains(style), 'Style_aux'] = style

for key in general_styles_dict:

    ice_general_styles.loc[ice_general_styles['Style'].str.contains('{} Ale'.format(key)), 'Style_aux'] = '{} Ale'.format(general_styles_dict[key])



plt.figure(figsize=(12,6))

sns.boxplot(ice_general_styles['Style_aux'], ice_general_styles['ABV'])

plt.xticks(rotation=45)

plt.ylim(0,25)

plt.title('ABV by GENERAL Styles', fontsize=22)

plt.xlabel('Style', fontsize=20)

plt.ylabel('ABV', fontsize=20)

plt.show()
order = ice_general_styles.groupby('Style_aux')['Color'].median().fillna(0).sort_values()[::-1].index



plt.figure(figsize=(12,6))

sns.boxplot(ice_general_styles['Style_aux'], ice_general_styles['Color'])

plt.xticks(rotation=45)

plt.ylim(0,55)

plt.title('Color by GENERAL Styles', fontsize=22)

plt.xlabel('Style', fontsize=20)

plt.ylabel('Color', fontsize=20)

plt.show()
ice_abv_color = ice[(ice['ABV']<=20) & (ice['Color']<=50)]

ice_abv_color = ice_abv_color.sample(int(len(ice_abv_color)/10), random_state=42)



plt.figure(figsize=(12,6))

sns.regplot(ice_abv_color['ABV'],ice_abv_color['Color'])

plt.title('ABV and Color relation', fontsize=22)

plt.xlabel('ABV', fontsize=20)

plt.ylabel('Color', fontsize=20)

plt.show()
plt.figure(figsize=(12,14))

count=0    

for col in ['OG', 'FG', 'IBU']:

    for i in range(1,3):

        count+=1

        plt.subplot(3,2,count)



        if (i==1):

            sns.boxplot(ice_general_styles['Style_aux'], np.log1p(ice_general_styles[col]))

        else:

            sns.violinplot(ice_general_styles['Style_aux'], np.log1p(ice_general_styles[col]))

        plt.xticks(rotation=45)

        plt.title('Log (1+{}) by GENERAL Styles'.format(col), fontsize=14)

        plt.xlabel(' ')

        plt.ylabel('Log (1+{})'.format(col), fontsize=14)



plt.subplots_adjust(hspace=0.4)

plt.show()
col= 'BrewMethod'

gb_brew_method = ice.groupby([col]).count().sort_values(['BeerID'], ascending=False).reset_index()[:20]

gb_brew_method['BeerID'] = (gb_brew_method['BeerID'] / len(ice)) * 100



plt.figure(figsize=(8,6))

g=sns.barplot(gb_brew_method[col], gb_brew_method['BeerID'])

plt.title('{} Distribution'.format(col), fontsize=15)

plt.legend()

plt.ylabel('Normed Frequency (%)', fontsize=15)

plt.xlabel(col, fontsize=15)



for index, row in gb_brew_method.iterrows():

    g.text(x=index,y=row['BeerID']+1,s='{:.2f}%'.format(row['BeerID']), 

           color='black', ha="center", fontsize=16)



plt.show()
col = 'SugarScale'

gb_brew_method = ice.groupby([col]).count().sort_values(['BeerID'], ascending=False).reset_index()[:20]

gb_brew_method['BeerID'] = (gb_brew_method['BeerID'] / len(ice)) * 100



plt.figure(figsize=(8,6))

g=sns.barplot(gb_brew_method[col], gb_brew_method['BeerID'])

plt.title('{} Distribution'.format(col), fontsize=15)

plt.legend()

plt.ylabel('Normed Frequency (%)', fontsize=15)

plt.xlabel(col, fontsize=15)



for index, row in gb_brew_method.iterrows():

    g.text(x=index,y=row['BeerID']+1,s='{:.2f}%'.format(row['BeerID']), 

           color='black', ha="center", fontsize=16)



plt.show()
ice.head()
ice.isnull().sum()
ice.dtypes
ice.columns
beer = ice.drop(['BeerID', 'Name', 'URL', 'Style', 'BrewMethod', 'PrimingMethod', 'PrimingAmount', 'UserId', 'Style_aux'], axis = 1)

beer.head()
beer1 = beer.fillna(beer.mean())

beer1.head()
beer1.columns
beer1 = beer1.reindex(columns=['StyleID', 'Size(L)', 'OG', 'FG', 'ABV', 'IBU', 'Color', 'BoilSize',

       'BoilTime', 'BoilGravity', 'Efficiency', 'MashThickness',

       'PitchRate', 'PrimaryTemp', 'SugarScale'])
beer1.head()
'''from sklearn.preprocessing import LabelEncoder



lb = LabelEncoder() 

beer['SugarScale'] = lb.fit_transform(beer['SugarScale'])

beer.head()'''
sns.pairplot(beer1)
corr = beer1.corr()

sns.heatmap(corr, xticklabels=corr.columns, yticklabels=corr.columns)
X = beer1.iloc[:,:-1]

X.head()
y = beer1.iloc[:,14]

y.head()
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import scale
X_scaled = scale(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size = 0.3, random_state = 100)
from sklearn.svm import SVC
model = SVC()

model.fit(X_train, y_train)
pred = model.predict(X_test)

pred
from sklearn import metrics

from sklearn.metrics import confusion_matrix

from sklearn.model_selection import cross_val_score
# confusion matrix and accuracy



# accuracy

print("accuracy:", metrics.accuracy_score(y_true=y_test, y_pred=pred), "\n")



# cm

print(metrics.confusion_matrix(y_true=y_test, y_pred=pred))
from sklearn.metrics import classification_report
print(classification_report(y_test, pred))