import os

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import scipy.stats as stats

from bokeh.palettes import Spectral4

from bokeh.plotting import figure, output_notebook, show

from bokeh.layouts import gridplot

import squarify

import warnings

warnings.filterwarnings("ignore")

train = pd.read_table('../input/train.tsv')

def splitCat(line):

    try:

        text = line

        txt1, txt2, txt3 = text.split("/")

        return txt1, txt2, txt3

    except: 

        return ("NoLabel", "NoLabel", "NoLabel")

def basicHandling(df):

    #Reorder to follow my instinct

    cols = ['train_id','price', 'name', 'category_name', 'brand_name', 'item_description', 'item_condition_id', 'shipping']

    df = df[cols]

    df.rename(columns = {'item_condition_id' : 'item_condition'}, inplace = True)

    

    #if train_id unique, I convert it as Index

    if df.train_id.nunique() == df.shape[0]:

        df.set_index('train_id', inplace = True)

        print('Since Train Id is unique, it is going to be Index')

    

    #Fill NA value as 'Missing' what we don't know

    #It helps us to easily deal with the variable when we visualize

    df.loc[:, ['brand_name', 'item_description']] = df.loc[:, ['brand_name', 'item_description']].fillna('NoLabel')

    if df.isnull().any().sum() == 1: print('There is no Null')

    # reference: BuryBuryZymon at https://www.kaggle.com/maheshdadhich/i-will-sell-everything-for-free-0-55

    df['supCat'], df['midCat'], df['infCat'] = zip(*df['category_name'].apply(splitCat))

    df['price'] = np.log1p(df['price'])

    return df



train = basicHandling(train)

print('Simple Engineering is Done')
plt.figure(figsize = (12,6))

ax1 = plt.subplot2grid((2,2), (0,0))

sns.countplot('shipping', data = train, ax = ax1, palette = sns.color_palette("BrBG", 2))

ax1.set_title('Count Shipping', fontsize = 12)

ax2 = plt.subplot2grid((2,2), (0,1))

sns.boxplot('shipping', 'price', data = train, ax = ax2, palette = sns.cubehelix_palette(4))

ax2.set_title('BoxPlot with price', fontsize = 12)

ax3 = plt.subplot2grid((2,2), (1,0), colspan = 2)

group = train['price'].groupby(train['shipping'])

for ship, lp in group:

    sns.distplot(lp, kde = False, label = str(ship), ax = ax3, kde_kws = {'alpha' : 0.7})

ax3.legend()

ax3.set_title('Two logPrice Distribution accross Shipping', fontsize = 12)

plt.suptitle('Shipping', fontsize = 15)

plt.subplots_adjust(left=0.2, hspace=0.3, top=0.85)

plt.show()
#trainS = train.loc[train['shipping'] == 0,:]

#trainN = train.loc[train['shipping'] != 0,:]
plt.figure(figsize = (12,6))

sns.despine(left = True)

sns.set_style("darkgrid", {'axes.grid' : False})



ax1 = plt.subplot2grid((2,2), (0,0), colspan = 2)

cntCondition = train['item_condition'].value_counts()

sns.barplot(cntCondition.index, cntCondition.values, color = 'b', ax = ax1)

ax1.set_title('Count of Item Condition', fontsize = 12)



ax2 = plt.subplot2grid((2,2), (1,0))

sns.boxplot(x = 'item_condition', y = 'price', data = train, palette = sns.color_palette('RdBu',5), ax = ax2)

ax2.set_title('Box Plot, item-condition & price', fontsize = 12)



ax3 = plt.subplot2grid((2,2), (1,1))

group = train['price'].groupby(train['item_condition'])

color = sns.color_palette("Paired", 10)

for con, pri in group:

    sns.distplot(pri, kde = True, label = str(con), color = color[con], ax = ax3, kde_kws = {'alpha' : 0.5})

ax3.legend() 

plt.show()
train['item_condition'].replace(5,4, inplace = True)

trainS = train.loc[train['shipping'] == 0,:]

itemCondS = trainS.copy()

trainNS = train.loc[train['shipping'] != 0,:]

itemCondNS = trainNS.copy()



pink = itemCondS.loc[itemCondS['brand_name'] == 'PINK',:]

louis = itemCondS.loc[itemCondS['brand_name'] == 'Louis Vuitton',:]

pinkTmp = pink.loc[pink.loc[:,'infCat'] == 'Crewneck',:]

pinkTmp.sort_values('item_condition', inplace = True)

louisTmp = louis.loc[louis.loc[:,'infCat'] == 'Wallets', :]

louisTmp.sort_values('item_condition', inplace = True)



first30 = pinkTmp.iloc[:30,:]

behind30 = pinkTmp.iloc[-30:,:]

f, ax = plt.subplots(1,4, figsize = (12,4))

sns.distplot(first30.loc[:,'price'], ax = ax[0], label = 'item_condition_1')

sns.distplot(behind30.loc[:,'price'], ax = ax[0], label = 'item_condition_4')

ax[0].set_title('PINK & Crewneck in Shipping', fontsize = 8)

ax[0].legend()



first30 = louisTmp.iloc[:30,:]

behind30 = louisTmp.iloc[-30:,:]

sns.distplot(first30.loc[:,'price'], ax = ax[1], label = 'item_condition_1')

sns.distplot(behind30.loc[:,'price'], ax = ax[1], label = 'item_condition_4')

ax[1].set_title('Louis Vuitton & Wallet in shipping', fontsize = 8)

ax[1].legend()



pink = itemCondNS.loc[itemCondNS['brand_name'] == 'PINK',:]

louis = itemCondNS.loc[itemCondNS['brand_name'] == 'Louis Vuitton',:]

pinkTmp = pink.loc[pink.loc[:,'infCat'] == 'Crewneck',:]

pinkTmp.sort_values('item_condition', inplace = True)

louisTmp = louis.loc[louis.loc[:,'infCat'] == 'Wallets', :]

louisTmp.sort_values('item_condition', inplace = True)



first30 = pinkTmp.iloc[:30,:]

behind30 = pinkTmp.iloc[-30:,:]

sns.distplot(first30.loc[:,'price'], ax = ax[2], color = 'g', label = 'item_condition_1')

sns.distplot(behind30.loc[:,'price'], ax = ax[2], color = 'purple', label = 'item_condition_4')

ax[2].set_title('PINK & Crewneck in Nonshippng', fontsize = 8)

ax[2].legend()



first30 = louisTmp.iloc[:30,:]

behind30 = louisTmp.iloc[-30:,:]

sns.distplot(first30.loc[:,'price'], ax = ax[3], color = 'g',label = 'item_condition_1')

sns.distplot(behind30.loc[:,'price'], ax = ax[3], color = 'purple',label = 'item_condition_4')

ax[3].set_title('Louis Vuitton & Wallet in Nonshipping', fontsize = 8)

ax[3].legend()

plt.show()
train.rename(columns = {'brand_name' : 'brand'}, inplace = True)

trainNM = train.loc[train['brand'] != 'NoLabel',:]

trainS = trainNM.loc[trainNM['shipping'] == 0,:]

trainN = trainNM.loc[trainNM['shipping'] != 0,:]

print('The Number of Brand : {0}'.format(train['brand'].nunique()))

f = plt.figure(figsize = (12,6))

ax1 = plt.subplot2grid((3,3), (0,0), colspan = 3, rowspan = 2)

cntSupCatS = trainS['supCat'].value_counts().to_frame()

squarify.plot(sizes = cntSupCatS.values, label = cntSupCatS.index,

                  color = sns.color_palette('Paired', 11), alpha = 0.5, ax = ax1)

ax1.set_title("TreeMap of SupCat None Shipping Count", fontsize = 13)



ax2 = plt.subplot2grid((3,3), (2,0))

womenS = trainS.loc[trainS['supCat'] == 'Women','midCat'].value_counts().to_frame()

squarify.plot(sizes = womenS.values, label = womenS.index, 

                      color = sns.color_palette("Set2", womenS.index.shape[0]), alpha = 0.7, ax = ax2)

ax2.set_title("Occurences in Women", fontsize = 13)



ax3 = plt.subplot2grid((3,3), (2,1))

kidS = trainS.loc[trainS['supCat'] == 'Kids','midCat'].value_counts().to_frame()

squarify.plot(sizes = kidS.values, label = kidS.index, 

                      color = sns.color_palette("Set1", kidS.index.shape[0]), alpha = 0.7, ax = ax3)

ax3.set_title("Occurences in Kids", fontsize = 13)



ax4 = plt.subplot2grid((3,3), (2,2))

electS = trainS.loc[trainS['supCat'] == 'Electronics','midCat'].value_counts().to_frame()

squarify.plot(sizes = electS.values, label = electS.index, 

                      color = sns.color_palette("muted", electS.index.shape[0]), alpha = 0.7, ax = ax4)

ax4.set_title("Occurences in electS", fontsize = 13)



for axis in [ax1, ax2, ax3, ax4]:

    axis.set_xticklabels([])

    axis.set_yticklabels([])

f.subplots_adjust(0.03, 0.03, 0.85, 0.90)

plt.suptitle('Tree Map of DataSet', fontsize = 14)

plt.show()
pantSet = trainS.loc[trainS['midCat'] == 'Pants', 'infCat']

pantSet.value_counts().head(5)
pantExp = trainS.loc[pantSet.index, ['item_condition', 'brand', 'price']].copy()

pantExp['item_condition'].replace(5, 4, inplace = True)

pantComp = pantExp['brand'].value_counts()[pantExp['brand'].value_counts() > 100].index.values[:5]

number = pantComp

fiveComp = trainS.loc[trainS['brand'].isin(number),['brand', 'item_condition']]

sns.factorplot(x = 'item_condition' ,col = 'brand', col_wrap = 3, kind = 'count', data = fiveComp, size = 2, aspect = 2, sharey = False)

plt.show()
pantExp = pantExp.loc[pantExp['brand'].isin(number), :]

pantExp['predictBrandVal'] = pantExp['brand'].map({'PINK' : 'gd', 'Lululemon' : 'bd', 'Old Navy': 'bd', 'American Eagle':'bd', 'Express' : 'bd'})

sns.factorplot(x = 'price', y = 'brand', col = 'item_condition', kind = 'box', col_order = [1,2,3,4], col_wrap = 2, order = 

               ['PINK', 'Lululemon','Old Navy','American Eagle', 'Express'], data = pantExp, size = 2, aspect = 2)

plt.show()
pantSet = trainNS.loc[trainNS['midCat'] == 'Pants', 'infCat']

trainNS = train.loc[train['shipping'] != 0, :]

pantExp = trainNS.loc[pantSet.index, ['item_condition', 'brand', 'price']].copy()

pantExp['item_condition'].replace(5, 4, inplace = True)

pantComp = pantExp['brand'].value_counts()[pantExp['brand'].value_counts() > 100].index

number = pantComp

fiveComp = trainNS.loc[trainNS['brand'].isin(number),['brand', 'item_condition']]

sns.factorplot(x = 'item_condition' ,col = 'brand', col_wrap = 3, kind = 'count', data = fiveComp, size = 2, aspect = 2, sharey = False)

plt.show()
pantExp = pantExp.loc[pantExp['brand'].isin(number), :]

pantExp['predictBrandVal'] = pantExp['brand'].map({'PINK' : 'gd', 'LuLaRoe' : 'gd', 'Old Navy': 'bd', 'American Eagle':'bd'})



f, ax = plt.subplots(1,1, figsize = (8,4))

sns.boxplot(x = 'price', y = 'predictBrandVal', data = pantExp, ax = ax)

sns.factorplot(x = 'price', y = 'brand', col = 'item_condition', kind = 'box', col_order = [1,2,3,4], col_wrap = 2, order = 

               ['PINK', 'LuLaRoe','Old Navy','American Eagle'], data = pantExp, size = 2, aspect = 2)

plt.show()
trainNM = train.loc[train['brand'] != 'NoLabel',:]

trainS = trainNM.loc[trainNM['shipping'] == 0,:]

trainNsS = trainNM.loc[trainNM['shipping'] == 1,:]



#Horizontal Stacked// Control Color: colormap?

brand_top10S = trainS['brand'].value_counts().index[:10]

brand_top10S = trainS.loc[trainS['brand'].isin(brand_top10S),:]



top10SsupCat = brand_top10S.groupby(['brand','supCat']).size().unstack().fillna(0)

sumLst = top10SsupCat.sum(axis = 0)

#print(sumLst.sort_values(ascending = False))

top5supCat = sumLst.sort_values(ascending = False).index[:5]

top10SsupCat = top10SsupCat.loc[:, top5supCat]

top10SsupCatRatio = top10SsupCat.div(top10SsupCat.sum(axis = 1), axis = 0)



colorSet = sns.color_palette('GnBu', 5)

f, ax = plt.subplots(1,2, figsize = (12,6))

top10SsupCatRatio.plot.barh(stacked = True, color = colorSet, ax = ax[0])

ax[0].legend(bbox_to_anchor = [1.1, 0.9], fontsize = 8)

ax[0].set_title('supCatgory ratio for each Brand', fontsize = 12)

ax[0].set_ylabel('')



supCatMB = train.loc[:,['brand', 'supCat']]

supCatMB['brand'] = np.where(supCatMB['brand']=='NoLabel', 'NoLabel', 'Label')

supCatMB = supCatMB.groupby(['supCat', 'brand']).size().unstack()

supCatMBRatio = supCatMB.div(supCatMB.sum(axis =1), axis =0)

supCatMBRatio.plot.barh(stacked = True, color = sns.light_palette('purple',  4), ax = ax[1])

ax[1].legend(bbox_to_anchor = [1.1, 0.9])

ax[1].set_title('Missing Ratio for Brand', fontsize = 12)

ax[1].set_ylabel('')

plt.subplots_adjust(wspace = 0.9)

plt.show()