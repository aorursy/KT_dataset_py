import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from IPython.display import display ## for multiple display

import matplotlib as mpl

import matplotlib.pyplot as plt

import seaborn as sns

sns.set_style('whitegrid')

#from matplotlib import style 

#plt.style.use("fivethirtyeight")

#print(plt.style.available)
## import the file and display the head

wetlands = pd.read_csv("../input/wetlands.csv")

wetlands.head()
## Display the info about the table

display(wetlands.info())



## We won't need these columns: OBJECTID, GLOBALID for our exercise so drop them.

## OBJECTID is saved weirdly. Anyone knows if there's specific reason? 

## I found the right name with wetlands.columns[0]

wetlands.drop(['\ufeffOBJECTID', 'GLOBALID'], axis=1, inplace=True) 



## There are 3 entries in ACRES without value. Display the ones. We don't do any for now

display(wetlands[wetlands['ACRES'].isnull()])
## Group values of ATTRIBUTE into CATEGORY, and compare with WETLAND_TYPE

att = wetlands.ATTRIBUTE

att.astype('category')  ## ATTRIBUTE has 1076 categories. It is too much to categorize...



## Realize that these categories all start with first letters either L, P, or R

wetlands['CATEGORY'] = att.apply(lambda x: x[0])  ## Create a new column CATEGORY with the first letter

cat = wetlands.CATEGORY

cat.astype('category')

display(cat.groupby(cat).count())  ## It has now only 3 categories



## Let's compare with WETLAND_TYPE.

wtype = wetlands.WETLAND_TYPE

wtype.astype('category')  ## WETLAND_TYPE has 6 categories. 



## Let's group all Freshwater categories into Palustrine. Lake into Lacustrine, Riverine into Riverine

wtype_dic = dict({'Freshwater Emergent Wetland': 'Palustrine', 

                  'Freshwater Forested/Shrub Wetland': 'Palustrine', 

                  'Freshwater Pond': 'Palustrine', 

                  'Lake': 'Lacustrine', 

                  'Other': 'Other', 

                  'Riverine': 'Riverine'})

wtype = wtype.apply(lambda x: wtype_dic[x])

wetlands = wetlands.join(wtype, rsuffix='_SIMPLE') ## Create a new column named WETLAND_TYPE_SIMPLE

wtype = wetlands['WETLAND_TYPE_SIMPLE']

display(wtype.groupby(wtype).count())
## At first glance, ACRES and ShapeSTArea seems quite related to each other.

## 1 acre = 4046.86 square meters

wetlands['AREA_RATIO'] = wetlands.ShapeSTArea / wetlands.ACRES

area_ratio = wetlands['AREA_RATIO'].dropna()



## Prepare a plot

fig, axis1 = plt.subplots(1,1,figsize=(5,5))



## Distribution of area ratio

print("mean: {}, std: {}".format(area_ratio.mean(), area_ratio.std()))

#area_ratio_l.sort_values(ascending=False)  ## We can see a few excentric values

display(area_ratio.quantile([0, 0.025, 0.05, 0.5, 0.95, 0.975, 1]))  ## Peek through distribution



## Take only 95% values near its mean (i.e. drop excentric values)

lbound = area_ratio.quantile(0.025)

ubound = area_ratio.quantile(0.975)

ar = area_ratio[(area_ratio > lbound) & (area_ratio < ubound)]

ar_min = int(np.floor(min(ar)))  ## Minimum value in 95% group

ar_max = int(np.ceil(max(ar)))  ## Maximum value in 95% group

interval = list(map(lambda x:x/2, list(range(ar_min*2, ar_max*2+1, 1))))

ar_bins = [0] + interval + [float('inf')]  ## Prepare bins to display values

cut = pd.cut(area_ratio, bins=ar_bins)

area_ratio = pd.concat([area_ratio, cut], axis=1)

area_ratio.columns = ['AREA_RATIO', 'AREA_RATIO_GROUP']



## Plot 

l = sns.countplot(x='AREA_RATIO_GROUP', data=area_ratio, ax=axis1)

axis1.set_xticklabels(l.get_xticklabels(), rotation=60, ha='right')

axis1.set_title('Ratio: ShapeSTArea / ACRES', fontsize=14)

axis1.set_xlabel(axis1.get_xlabel(), size=12)

axis1.set_ylabel(axis1.get_ylabel(), size=12)

axis1.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda x, p: format(int(x), ',')))

l.tick_params(labelsize=8)



## Remove rows with excentric values outside 95% near the mean

wetlands = wetlands[(wetlands['AREA_RATIO'] > lbound) & (wetlands['AREA_RATIO'] < ubound)]
## Calculate circularity defined as above, and then split the data by each type (L, P, R) 

wlen = wetlands['ShapeSTLength']

warea = wetlands['ShapeSTArea']

wetlands['CIRCULARITY'] = np.sqrt(4*np.pi*warea/np.square(wlen))

circ_l = wetlands[wetlands['CATEGORY']=='L']['CIRCULARITY']

circ_p = wetlands[wetlands['CATEGORY']=='P']['CIRCULARITY']

circ_r = wetlands[wetlands['CATEGORY']=='R']['CIRCULARITY']



## Basic statistics information of each type (L, P, R) 

print('CIRCULARITY MEAN OF L-TYPE: {}, CIRCULARITY STD OF L-TYPE: {}'.format(np.mean(circ_l), np.std(circ_l)))

print('CIRCULARITY MEAN OF P-TYPE: {}, CIRCULARITY STD OF P-TYPE: {}'.format(np.mean(circ_p), np.std(circ_p)))

print('CIRCULARITY MEAN OF R-TYPE: {}, CIRCULARITY STD OF R-TYPE: {}'.format(np.mean(circ_r), np.std(circ_r)))

lq = circ_l.quantile([0, 0.025, 0.05, 0.5, 0.95, 0.975, 1])

pq = circ_p.quantile([0, 0.025, 0.05, 0.5, 0.95, 0.975, 1])

rq = circ_r.quantile([0, 0.025, 0.05, 0.5, 0.95, 0.975, 1])

circ_qt = pd.concat([lq, pq, rq], axis=1)

circ_qt.columns = ['CIRCULARITY_L', 'CIRCULARITY_P', 'CIRCULARITY_R']

circ_qt.index.rename('QUANTILE', inplace=True)

display(circ_qt)



## Plot circularity of each type using a bin [0.0, 0.1, 0.2, ... 0.9, 1.0]

fig, (axis1, axis2, axis3) = plt.subplots(1,3, figsize=(12,5))

bins = list(map(lambda x:x/10, list(range(0,11,1))))

cut_l = pd.cut(circ_l, bins=bins).to_frame()

cut_l.columns = ['CIRC_GROUP']

sns.countplot(x='CIRC_GROUP', data=cut_l, ax=axis1)

cut_p = pd.cut(circ_p, bins=bins).to_frame()

cut_p.columns = ['CIRC_GROUP']

sns.countplot(x='CIRC_GROUP', data=cut_p, ax=axis2)

cut_r = pd.cut(circ_r, bins=bins).to_frame()

cut_r.columns = ['CIRC_GROUP']

sns.countplot(x='CIRC_GROUP', data=cut_r, ax=axis3)



## Title, labels, ticks, annotations

dtype = ['L-type', 'P-type', 'R-type']

axis = [axis1, axis2, axis3]

total = [len(cut_l), len(cut_p), len(cut_r)]

i=0

while i < len(axis):

    axis[i].set_title(dtype[i], fontsize=14)

    axis[i].set_xlabel(axis[i].get_xlabel(), size=12)

    axis[i].set_ylabel(axis[i].get_ylabel(), size=12)

    axis[i].set_xticklabels(axis[i].get_xticklabels(), rotation=60, ha='right', size=8)

    axis[i].yaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda x, p: format(int(x), ',')))

    ## Annotate percentages of patches in each graphs

    for p in axis[i].patches:

        annt = 100*p.get_height()/total[i]

        x = p.get_x()+p.get_width()/2.0

        y = p.get_height()+0.002*total[i]

        axis[i].annotate('{:.1f}%'.format(annt), (x, y), ha='center', size=6)#, va='bottom')p.get_height()

    i=i+1



axis1.set_ylim([0,1200])

axis3.set_ylim([0,350])

plt.tight_layout()

     
## Divide P-type dataset into sub-groups as per WETLAND_TYPE

FE = wetlands[wetlands['WETLAND_TYPE']=='Freshwater Emergent Wetland']['CIRCULARITY']

FF = wetlands[wetlands['WETLAND_TYPE']=='Freshwater Forested/Shrub Wetland']['CIRCULARITY']

FP = wetlands[wetlands['WETLAND_TYPE']=='Freshwater Pond']['CIRCULARITY']

OT = wetlands[wetlands['WETLAND_TYPE']=='Other']['CIRCULARITY']



## Plot circularity of each type using a bin [0.0, 0.1, 0.2, ... 0.9, 1.0]

fig.clf()

fig, (axis1, axis2, axis3, axis4) = plt.subplots(1,4, figsize=(12,5))

bins = list(map(lambda x:x/10, list(range(0,11,1))))

cut_FE = pd.cut(FE, bins=bins).to_frame()

cut_FE.columns = ['CIRC_GROUP']

sns.countplot(x='CIRC_GROUP', data=cut_FE, ax=axis1)

cut_FF = pd.cut(FF, bins=bins).to_frame()

cut_FF.columns = ['CIRC_GROUP']

sns.countplot(x='CIRC_GROUP', data=cut_FF, ax=axis2)

cut_FP = pd.cut(FP, bins=bins).to_frame()

cut_FP.columns = ['CIRC_GROUP']

sns.countplot(x='CIRC_GROUP', data=cut_FP, ax=axis3)

cut_OT = pd.cut(OT, bins=bins).to_frame()

cut_OT.columns = ['CIRC_GROUP']

sns.countplot(x='CIRC_GROUP', data=cut_OT, ax=axis4)



## Title, labels, ticks, annotations

dtype = ['FE-type', 'FF-type', 'FP-type', 'OT-type']

axis = [axis1, axis2, axis3, axis4]

total = [len(FE), len(FF), len(FP), len(OT)]

i=0

while i < len(axis):

    axis[i].set_title(dtype[i], fontsize=12)

    axis[i].set_xlabel(axis[i].get_xlabel(), size=8)

    axis[i].set_ylabel(axis[i].get_ylabel(), size=8)

    axis[i].set_xticklabels(axis[i].get_xticklabels(), rotation=60, ha='right', size=6)

    axis[i].yaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda x, p: format(int(x), ',')))

    i=i+1



plt.tight_layout()
## Entries with category R

rtype = wetlands[wetlands['CATEGORY']=='R']

## Entries with category R with circularity value greater than 0.8 (river but rather circular shape)

rtype08 = wetlands[(wetlands['CATEGORY']=='R') & (wetlands['CIRCULARITY']>0.8)]

print('R-type, for all circulartiy: mean ShapeSTArea = {:.2f}, mean ShapeSTLength = {:.2f}'

      .format(rtype['ShapeSTArea'].mean(), rtype['ShapeSTLength'].mean()))

print('R-type, circulartiy>0.8: mean ShapeSTArea = {:.2f}, mean ShapeSTLength = {:.2f}'

      .format(rtype08['ShapeSTArea'].mean(), rtype08['ShapeSTLength'].mean()))