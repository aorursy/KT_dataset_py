import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



import PIL

import glob



import matplotlib as mpl

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

import scipy.stats as st

from sklearn import ensemble, tree, linear_model

import missingno as msno
im1 = PIL.Image.open('../input/the-other-cases/Screen Shot 2020-10-13 at 1.13.56 PM.png')

im2 = PIL.Image.open('../input/image-of-4-cases/1-s2.0-S0925231217315886-gr4.jpg')

im3 = PIL.Image.open('../input/the-other-cases/1-s2.0-S0925231217315886-gr5.jpg')



fig, axarr = plt.subplots(nrows=1, ncols=3, figsize=(20, 20))



axarr[0].imshow(im1)

axarr[0].set_title("Case 1")

axarr[0].axis('off')



axarr[1].imshow(im2)

axarr[1].set_title("Cases 4-6")

axarr[1].axis('off')



axarr[2].imshow(im3)

axarr[2].set_title("Cases 6-9")

axarr[2].axis('off')



fig.subplots_adjust(wspace=0.05)
train = pd.read_csv('../input/cee-498-project9-structural-damage-detection/train.csv')



pd.options.display.precision = 15 

# Pandas does not show us all the decimals, and we are interested in high accuracy.





train
plt.plot(train["Condition"]);

# Checking if the data is sorted logically as undamage must become damage.
train.index.name = 'Time'



train.rename({"DA04": "1st_story_01","DA05": "1st_story_02", "DA06": "1st_story_03",

             "DA07": "2nd_story_01", "DA08": "2nd_story_02", "DA09": "2nd_story_03",

             "DA10": "3rd_story_01", "DA11": "3rd_story_02", "DA12": "3rd_story_03",

             "DA13": "4th_story_01", "DA14": "4th_story_02", "DA15": "4th_story_03"}, axis="columns", inplace=True)

train.head(5)



# Columns have been renamed for a better understanding of the analysis.
train.dtypes
grouped_un = train.groupby(train.Condition) 

train_undamaged = grouped_un.get_group(0) 

train_undamaged.shape
train_undamaged.isna().sum()
grouped = train.groupby(train.Condition) 

train_damaged = grouped.get_group(1) 

train_damaged.shape
train_damaged.isna().sum()
train_undamaged = train_undamaged.dropna(axis = 0)
# By grouping these dataframes, the indexes are kept from the original dataframe



train_undamaged = train_undamaged.reset_index();



train_damaged = train_damaged.reset_index();



train_undamaged.head()
train_undamaged = train_undamaged.drop(['Condition'], axis=1)

train_damaged = train_damaged.drop(['Condition'], axis=1)
print(train_undamaged.shape)

print(train_damaged.shape)
print(train_undamaged.columns)

print(train_damaged.columns)
train_undamaged["Time"] = train_undamaged["Time"]/200

train_undamaged = train_undamaged.rename(columns={'Time': 'Time_sec'})



train_damaged["Time"] = train_damaged["Time"]/200

train_damaged = train_damaged.rename(columns={"Time": "Time_sec"})
train_undamaged.head()
train_damaged.head()
train_undamaged.drop(['Time_sec'], axis=1).describe()
train_damaged.drop(['Time_sec'], axis=1).describe()
fig, axs = plt.subplots(1, 2, tight_layout=True)



axs[0].hist(train_undamaged['Time_sec'], bins=80,color="skyblue");

axs[1].hist(train_damaged['Time_sec'], bins=80,color="skyblue");

train.drop(['Condition'], axis=1).plot(subplots=True, figsize=(20, 20)); plt.legend(loc='best');
train_undamaged.drop(['Time_sec'],axis=1).hist(figsize=(20,20),bins=80, edgecolor='black');
train_damaged.drop(['Time_sec'],axis=1).hist(figsize=(20,20),bins=80, edgecolor='black');
train_undamaged
# Difference between acceleration values in the undamaged condition

    # NOTE: The number of values for time (n) is not equal to the obtained number of values from the difference (n-1). 

    # This has been taken into account as follows,

    

fig, ax = plt.subplots(6,1,figsize=(20,10), sharex='all', sharey='all');

plt.subplots_adjust(hspace=0.7)

fig.suptitle('Difference in acceleration values among sensors in first and second floors')



ax[0].plot(train_undamaged['Time_sec'].values[0:260980], np.diff(train_undamaged['1st_story_01']),'tab:orange');

ax[0].set_xlabel("Time_sec");

ax[0].set_title("1st_story_01");



ax[1].plot(train_undamaged['Time_sec'].values[0:260980], np.diff(train_undamaged['1st_story_02']),'tab:orange');

ax[1].set_xlabel("Time_sec");

ax[1].set_title("1st_story_02");



ax[2].plot(train_undamaged['Time_sec'].values[0:260980], np.diff(train_undamaged['1st_story_03']),'tab:orange');

ax[2].set_xlabel("Time_sec");

ax[2].set_title("1st_story_03");



ax[3].plot(train_undamaged['Time_sec'].values[0:260980], np.diff(train_undamaged['2nd_story_01']),'tab:cyan');

ax[3].set_xlabel("Time_sec");

ax[3].set_title("2nd_story_01");



ax[4].plot(train_undamaged['Time_sec'].values[0:260980], np.diff(train_undamaged['2nd_story_02']),'tab:cyan');

ax[4].set_xlabel("Time_sec");

ax[4].set_title("2nd_story_02");



ax[5].plot(train_undamaged['Time_sec'].values[0:260980], np.diff(train_undamaged['2nd_story_03']),'tab:cyan');

ax[5].set_xlabel("Time_sec");

ax[5].set_title("2nd_story_03");

# Difference between acceleration values in the undamaged condition

fig, ax = plt.subplots(6,1,figsize=(20,10), sharex='all', sharey='all');

plt.subplots_adjust(hspace=0.7)

fig.suptitle('Difference in acceleration values among sensors in third and fourth floors')



ax[0].plot(train_undamaged['Time_sec'].values[0:260980], np.diff(train_undamaged['3rd_story_01']),'tab:orange');

ax[0].set_xlabel("Time_sec");

ax[0].set_title("3rd_story_01");



ax[1].plot(train_undamaged['Time_sec'].values[0:260980], np.diff(train_undamaged['3rd_story_02']),'tab:orange');

ax[1].set_xlabel("Time_sec");

ax[1].set_title("3rd_story_02");



ax[2].plot(train_undamaged['Time_sec'].values[0:260980], np.diff(train_undamaged['3rd_story_03']),'tab:orange');

ax[2].set_xlabel("Time_sec");

ax[2].set_title("3rd_story_03");



ax[3].plot(train_undamaged['Time_sec'].values[0:260980], np.diff(train_undamaged['4th_story_01']),'tab:cyan');

ax[3].set_xlabel("Time_sec");

ax[3].set_title("4th_story_01");



ax[4].plot(train_undamaged['Time_sec'].values[0:260980], np.diff(train_undamaged['4th_story_02']),'tab:cyan');

ax[4].set_xlabel("Time_sec");

ax[4].set_title("4th_story_02");



ax[5].plot(train_undamaged['Time_sec'].values[0:260980], np.diff(train_undamaged['4th_story_03']),'tab:cyan');

ax[5].set_xlabel("Time_sec");

ax[5].set_title("4th_story_03");



sns.set_style("white")



t_unt = train_undamaged.drop(['Time_sec'], axis=1)



# Compute the correlation matrix

corr_un = t_unt.corr()



# Generate a mask for the upper triangle

mask = np.triu(np.ones_like(corr_un, dtype=bool))



# Set up the matplotlib figure

f, ax = plt.subplots(figsize=(11, 9))



# Generate a custom diverging colormap

cmap = sns.diverging_palette(230, 20, as_cmap=True)



# Draw the heatmap with the mask and correct aspect ratio

plt.title('Correlation among sensors in different floors',y=1,size=16)

sns.heatmap(corr_un, mask=mask, cmap=cmap, vmax=.3, center=0,

            square=True, linewidths=.5, cbar_kws={"shrink": .5},annot=True);
s = corr_un.unstack() 

# Returns a DataFrame having a new level of column labels whose inner-most level consists of the pivoted index labels.



sc_df = s.sort_values(kind="quicksort")

train_und_corr = sc_df.sort_values(ascending=False)

train_und_corr = pd.DataFrame(train_und_corr, columns=['Correlation Values'])

train_und_corr = train_und_corr.loc[train_und_corr['Correlation Values'] < 1]



train_und_corr
sns.set_style("darkgrid")



_, axes = plt.subplots(2, 6, sharey=True, figsize=(10, 10))



sns.boxplot(data=train_undamaged['1st_story_01'], palette="Set2", ax=axes[0,0]).set(xlabel='1st_story_01', ylabel='Acceleration Recorded');

sns.boxplot(data=train_undamaged['1st_story_02'], palette="Paired", ax=axes[0,1]).set(xlabel='1st_story_02');

sns.boxplot(data=train_undamaged['1st_story_03'], palette="husl", ax=axes[0,2]).set(xlabel='1st_story_03');



sns.boxplot(data=train_undamaged['2nd_story_01'], palette="Set2", ax=axes[0,3]).set(xlabel='2nd_story_01');

sns.boxplot(data=train_undamaged['2nd_story_02'], palette="Paired", ax=axes[0,4]).set(xlabel='2nd_story_02');

sns.boxplot(data=train_undamaged['2nd_story_03'], palette="husl", ax=axes[0,5]).set(xlabel='2nd_story_03');



sns.boxplot(data=train_undamaged['3rd_story_01'], palette="Set2", ax=axes[1,0]).set(xlabel='3rd_story_01', ylabel='Acceleration Recorded');

sns.boxplot(data=train_undamaged['3rd_story_02'], palette="Paired", ax=axes[1,1]).set(xlabel='3rd_story_02');

sns.boxplot(data=train_undamaged['3rd_story_03'], palette="husl", ax=axes[1,2]).set(xlabel='3rd_story_03');



sns.boxplot(data=train_undamaged['4th_story_01'], palette="Set2", ax=axes[1,3]).set(xlabel='4th_story_01');

sns.boxplot(data=train_undamaged['4th_story_02'], palette="Paired", ax=axes[1,4]).set(xlabel='4th_story_02');

sns.boxplot(data=train_undamaged['4th_story_03'], palette="husl", ax=axes[1,5]).set(xlabel='4th_story_03');

train_damaged.head()
train_damaged.shape
fig, ax = plt.subplots(6,1,figsize=(20,10), sharex='all', sharey='all');

plt.subplots_adjust(hspace=0.7)

fig.suptitle('Difference in acceleration values among sensors in first and second floors')



ax[0].plot(train_damaged['Time_sec'].values[0:9511], np.diff(train_damaged['1st_story_01']),'tab:orange');

ax[0].set_xlabel("Time_sec");

ax[0].set_title("1st_story_01");



ax[1].plot(train_damaged['Time_sec'].values[0:9511], np.diff(train_damaged['1st_story_02']),'tab:orange');

ax[1].set_xlabel("Time_sec");

ax[1].set_title("1st_story_02");



ax[2].plot(train_damaged['Time_sec'].values[0:9511], np.diff(train_damaged['1st_story_03']),'tab:orange');

ax[2].set_xlabel("Time_sec");

ax[2].set_title("1st_story_03");



ax[3].plot(train_damaged['Time_sec'].values[0:9511], np.diff(train_damaged['2nd_story_01']),'tab:cyan');

ax[3].set_xlabel("Time_sec");

ax[3].set_title("2nd_story_01");



ax[4].plot(train_damaged['Time_sec'].values[0:9511], np.diff(train_damaged['2nd_story_02']),'tab:cyan');

ax[4].set_xlabel("Time_sec");

ax[4].set_title("2nd_story_02");



ax[5].plot(train_damaged['Time_sec'].values[0:9511], np.diff(train_damaged['2nd_story_03']),'tab:cyan');

ax[5].set_xlabel("Time_sec");

ax[5].set_title("2nd_story_03");
fig, ax = plt.subplots(6,1,figsize=(20,10), sharex='all', sharey='all');

plt.subplots_adjust(hspace=0.7)

fig.suptitle('Difference in acceleration values among sensors in first and second floors')



ax[0].plot(train_damaged['Time_sec'].values[0:9511], np.diff(train_damaged['3rd_story_01']),'tab:orange');

ax[0].set_xlabel("Time_sec");

ax[0].set_title("3rd_story_01");



ax[1].plot(train_damaged['Time_sec'].values[0:9511], np.diff(train_damaged['3rd_story_02']),'tab:orange');

ax[1].set_xlabel("Time_sec");

ax[1].set_title("3rd_story_02");



ax[2].plot(train_damaged['Time_sec'].values[0:9511], np.diff(train_damaged['3rd_story_03']),'tab:orange');

ax[2].set_xlabel("Time_sec");

ax[2].set_title("3rd_story_03");



ax[3].plot(train_damaged['Time_sec'].values[0:9511], np.diff(train_damaged['4th_story_01']),'tab:cyan');

ax[3].set_xlabel("Time_sec");

ax[3].set_title("4th_story_01");



ax[4].plot(train_damaged['Time_sec'].values[0:9511], np.diff(train_damaged['4th_story_02']),'tab:cyan');

ax[4].set_xlabel("Time_sec");

ax[4].set_title("4th_story_02");



ax[5].plot(train_damaged['Time_sec'].values[0:9511], np.diff(train_damaged['4th_story_03']),'tab:cyan');

ax[5].set_xlabel("Time_sec");

ax[5].set_title("4th_story_03");



sns.set_style("white")



t_dam = train_damaged.drop(['Time_sec'], axis=1)



# Compute the correlation matrix

corr_dam = t_dam.corr()



# Generate a mask for the upper triangle

mask = np.triu(np.ones_like(corr_dam, dtype=bool))



# Set up the matplotlib figure

f, ax = plt.subplots(figsize=(11, 9))



# Generate a custom diverging colormap

cmap = sns.diverging_palette(230, 20, as_cmap=True)



# Draw the heatmap with the mask and correct aspect ratio

plt.title('Correlation among sensors in different floors for damage data',y=1,size=16)

sns.heatmap(corr_un, mask=mask, cmap=cmap, vmax=.3, center=0,

            square=True, linewidths=.5, cbar_kws={"shrink": .5},annot=True);
r = corr_dam.unstack() 

# Returns a DataFrame having a new level of column labels whose inner-most level consists of the pivoted index labels.



rc_df = r.sort_values(kind="quicksort")

train_d_corr = rc_df.sort_values(ascending=False)

train_d_corr = pd.DataFrame(train_d_corr, columns=['Correlation Values'])

train_d_corr = train_d_corr.loc[train_d_corr['Correlation Values'] < 1]



train_d_corr
sns.set_style("darkgrid")



_, axes = plt.subplots(2, 6, sharey=True, figsize=(10, 10))



sns.boxplot(data=train_damaged['1st_story_01'], palette="Set2", ax=axes[0,0]).set(xlabel='1st_story_01', ylabel='Acceleration Recorded');

sns.boxplot(data=train_damaged['1st_story_02'], palette="Paired", ax=axes[0,1]).set(xlabel='1st_story_02');

sns.boxplot(data=train_damaged['1st_story_03'], palette="husl", ax=axes[0,2]).set(xlabel='1st_story_03');



sns.boxplot(data=train_damaged['2nd_story_01'], palette="Set2", ax=axes[0,3]).set(xlabel='2nd_story_01');

sns.boxplot(data=train_damaged['2nd_story_02'], palette="Paired", ax=axes[0,4]).set(xlabel='2nd_story_02');

sns.boxplot(data=train_damaged['2nd_story_03'], palette="husl", ax=axes[0,5]).set(xlabel='2nd_story_03');



sns.boxplot(data=train_damaged['3rd_story_01'], palette="Set2", ax=axes[1,0]).set(xlabel='3rd_story_01', ylabel='Acceleration Recorded');

sns.boxplot(data=train_damaged['3rd_story_02'], palette="Paired", ax=axes[1,1]).set(xlabel='3rd_story_02');

sns.boxplot(data=train_damaged['3rd_story_03'], palette="husl", ax=axes[1,2]).set(xlabel='3rd_story_03');



sns.boxplot(data=train_damaged['4th_story_01'], palette="Set2", ax=axes[1,3]).set(xlabel='4th_story_01');

sns.boxplot(data=train_damaged['4th_story_02'], palette="Paired", ax=axes[1,4]).set(xlabel='4th_story_02');

sns.boxplot(data=train_damaged['4th_story_03'], palette="husl", ax=axes[1,5]).set(xlabel='4th_story_03');