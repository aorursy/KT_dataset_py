# load packages



import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib as mpl

from matplotlib import pyplot as plt



from sklearn.preprocessing import Normalizer

from sklearn.preprocessing import MinMaxScaler
# read the data

wine_raw = pd.read_csv('../input/wine-quality/winequality.csv')
print("Shape of Wine data:\nrows:", wine_raw.shape[0], '\ncolumns:', wine_raw.shape[1])
wine_raw.head()
wine_raw.describe().T
# Compute the correlation matrix

corr = wine_raw.corr()

corr
#mask to plot part of the matrix

mask = np.zeros_like(corr)

mask[np.triu_indices_from(mask)] = True





with sns.axes_style("white"):

    f, ax = plt.subplots(figsize=(7,7))

    ax = sns.heatmap(corr, mask=mask, cmap="RdGy", vmax=1, square=True)
aw= wine_raw

#Divide de dataset into redwine and white wine

rw = wine_raw.loc[wine_raw.color == 'red', wine_raw.columns]

ww = wine_raw.loc[wine_raw.color == 'white', wine_raw.columns]



#Create a Dataframe that summarized all wine  by  Quality Value order by index

aw_q=aw.quality.value_counts().sort_index()

aw_qd=pd.DataFrame({'Quality': aw_q.index,'Frequency': aw_q.values})



#Create a Dataframe that summarized red wine  by  Quality Value

rw_q=rw.quality.value_counts().sort_index()

rw_qd=pd.DataFrame({'Quality': rw_q.index,'Frequency': rw_q.values})



#Create a Dataframe that summarized white wine  by  Quality Value

ww_q=ww.quality.value_counts().sort_index()

ww_qd=pd.DataFrame({'Quality': ww_q.index,'Frequency': ww_q.values})



#Concatenate the diferente kind of wine quality DataFrame to compare

pd.concat([aw_qd, rw_qd, ww_qd], axis=1, keys=['All Wine','Red Wine', 'White Wine'])
fig = plt.figure(figsize = (15, 5))

title = fig.suptitle("Wine Type Vs Quality", fontsize=16)

fig.subplots_adjust(top=0.85, wspace=0.3)



ax1 = fig.add_subplot(1,3, 1)

ax1.set_title("All Wine")

ax1.set_xlabel("Quality")

ax1.set_ylabel("Frequency") 

aw_q = wine_raw.quality.value_counts()

aw_q = (list(aw_q.index), list(aw_q.values))

ax1.set_ylim([0, 2900])

ax1.tick_params(axis='both', which='major', labelsize=8.5)

bar1 = ax1.bar(aw_q[0], aw_q[1])





ax2 = fig.add_subplot(1,3, 2)

ax2.set_title("White Wine")

ax2.set_xlabel("Quality")

ax2.set_ylabel("Frequency") 

ww_q = wine_raw.quality[wine_raw.color == 'white'].value_counts()

ww_q = (list(ww_q.index), list(ww_q.values))

ax2.set_ylim([0, 2500])

ax2.tick_params(axis='both', which='major', labelsize=8.5)

bar2 = ax2.bar(ww_q[0], ww_q[1])



ax3 = fig.add_subplot(1,3, 3)

ax3.set_title("Red Wine")

ax3.set_xlabel("Quality")

ax3.set_ylabel("Frequency") 

rw_q = wine_raw.quality[wine_raw.color == 'red'].value_counts()

rw_q = (list(rw_q.index), list(rw_q.values))

ax3.set_ylim([0, 2500])

ax3.tick_params(axis='both', which='major', labelsize=8.5)

bar3 = ax3.bar(rw_q[0], rw_q[1])
#Create a new variable call "category", assign "0" when a quality value is less or equal to 6; and 1 whe the value is bigger than 6

wine_raw['category']=(wine_raw['quality']>6)*1



#Create another variable ('bins') that divide the sample into 4 groups

wine_raw['bins']=pd.cut(wine_raw['quality'],4, labels=["bad", "medium", "good","excelent"])

#check the changes

wine_raw.tail()
df = wine_raw.iloc[:,[0,1,2,3,4,5,6,7,8,9,10,13,15]]

sns.pairplot(df, hue="bins")
#To select the subset of samples that are "excelent"

ew=wine_raw.loc[wine_raw['bins']=='excelent']



#To select the subset of samples that are "bad"

bw=wine_raw.loc[wine_raw['bins']=='bad']



dtw=wine_raw.iloc[:,0:11].describe().T

dew=ew.iloc[:,0:11].describe().T

dbw=bw.iloc[:,0:11].describe().T
#Excelent Wine Scatter Matrix 

df = ew.iloc[:,[0,1,2,3,4,5,6,7,8,9,10,13,15]]

sns.pairplot(df, hue="color")
print('Comparison between wine categories')

#Concatenate the diferente kind of wine quality DataFrame to compare

pd.concat([dbw, dew], axis=1, keys=['Bad','Excelent'])
#Transform features by scaling each feature to a given range



x= wine_raw.iloc[:,[0,1,2,3,4,5,6,7,8,9,10]]

scaler = MinMaxScaler()

scaler.fit(x)

wrn=scaler.transform(x)

dwrn=pd.DataFrame(wrn)

dwrn.columns=['FAn','VAn','CAn','RSn','Cn','FSDn','TSDn','Dn','pHn','Suln','Aln']

dwrn.head()



#Concatenate with the raw dataset

wdf=pd.concat([wine_raw, dwrn], axis=1)

wdf.head()

dtw=wdf.iloc[:,15:28].describe().T

dtw
#To select the subset of samples that are "excelent"

ew=wdf.loc[wdf['bins']=='excelent']



#To select the subset of samples that are "bad"

bw=wdf.loc[wdf['bins']=='bad']



#Choose normalized variables

dtw=wdf.iloc[:,15:28].describe().T

dew=ew.iloc[:,15:28].describe().T

dbw=bw.iloc[:,15:28].describe().T



#Comparison between wine categories

pd.concat([dbw, dew], axis=1, keys=['Bad','Excelent'])