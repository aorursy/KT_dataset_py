# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

import statistics as stats

import math



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))

%matplotlib inline



# Any results you write to the current directory are saved as output.
#############################################

###########    Data import    ###############

#############################################



# 2012 to 2017 data is in a seperate csv

# from https://transparentcalifornia.com/salaries/san-francisco/

# import, concat, then export the data to a new csv



#import shutil

#import glob



##import csv files from folder

#path = '/Users/micahshull/Desktop/SF Salary Data Files'

#allFiles = glob.glob(path + "/*.csv")

#with open('/Users/micahshull/Desktop/SF SALARY REPORT 2012 TO 2017/SF SALARY DATA.csv', 'wb') as outfile:

#    for i, fname in enumerate(allFiles):

#        with open(fname, 'rb') as infile:

#            if i != 0:

#                infile.readline()  # Throw away header on all but first file

#            # Block copy rest of file from input to output without parsing

#            shutil.copyfileobj(infile, outfile)

#            print('SF SALARY DATA' + " has been imported.")

df = pd.read_csv('../input/SF SALARY DATA.csv').drop(['Employee Name', 'Overtime Pay', 'Other Pay', 'Total Pay', 'Notes', 'Agency', 'Status'], axis = 1)

# DtypeWarning: Columns (2,3,4,5,11) have mixed types.
    ###########################################

    ###########   mixed data types    #########

    ###########################################



# df.info()

df['Base Pay'].value_counts().head()

# includeds 606 'Not Provided' values 
    ######################################################

    #############   replace 'Not Provided'   #############

    ######################################################



# replace 'Not Provided" with 0.0 for float data type

# df['Base Pay'].replace('Not Provided', '0.0')

df['Base Pay'].replace('Not Provided', 0.0, inplace =True)

#df['Base Pay'].value_counts()

# 'Not Provided' data has been replace with zero base pay
    ##################################################

    ######  changing from object to numeric   ########

    ##################################################



# pd.to_numeric(df['Base Pay'], errors = 'coerce')

df['Base Pay'] = pd.to_numeric(df['Base Pay'], errors = 'coerce')

#df['Base Pay'].value_counts()

#df['Base Pay'].isna().sum()

#df.info()



# pd.to_numeric(df['Benefits'], errors = 'coerce')

df['Benefits'] = pd.to_numeric(df['Benefits'], errors = 'coerce')

# df['Benefits'].isna().sum()

# df.info()
    ###########################################

    ##########     replace NaN's     ##########

    ###########################################



#df.info()

[df['Base Pay'].isna().sum(),

df['Benefits'].isna().sum(),

df['Total Pay & Benefits'].isna().sum()]

# there is one NaN's in Benefits
df['Benefits'].fillna(0.0, inplace = True)

df['Benefits'].isna().sum()
    ##########################################

    ########  City of San Francisco   ########

    ########     Salaries Analysis    ########

    ########       2012 to 2017       ########

    ##########################################



    # Are Salaries increasing or decreasing?

    # Is base pay increasing/decreasing?

    # Are benefits incresing/decreasing?

    # Who is most affected by the changes
    ###########################################

    ############     Base Pay    ##############

    ###########################################



    # what is the distribution for base pay?

plt.figure(figsize=(10,7))

plt.hist(df['Base Pay'], color='g', bins = 30)

plt.title('Base Pay', fontsize = 20)

plt.xlabel('Base Pay in Dollars')
print('\n')

print('#########################')

print('    Base Pay Statistics')

print('#########################\n')

print('    Mean    ',int(stats.mean(df['Base Pay']))),

print('    Median  ',int(stats.median(df['Base Pay']))),

print('    Mode    ',int(stats.mode(df['Base Pay']))),

print('    Std     ',int(stats.stdev(df['Base Pay']))),

print('\n')

# print(df['Base Pay'].describe())
plt.figure(figsize=(10,7))

sns.boxplot(df['Base Pay'], palette='Greens')

plt.title('Base Pay', fontsize = 20)

plt.xlabel('Base Pay in Dollars')
 # who are the outliers at the top of the payscale?

df.groupby('Job Title')['Base Pay'].mean().sort_values(ascending=False).head(20)
    ###########################################

    #########    Base Pay over time   #########

    ###########################################



sns.set(font_scale=1.3)

a = sns.FacetGrid(df, col='Year',height =4,aspect= 0.9, col_wrap=3)

a.map(plt.hist, 'Base Pay', color='G', bins = 40)

plt.suptitle('Base Pay', fontsize = 20)

plt.tight_layout(rect=[0, 0.03, 1, 0.95])


    # how many employees per year?

# df.info()

df.groupby(['Year'])['Job Title'].count()


plt.figure(figsize=(10,7))

sns.boxplot(df['Year'],df['Base Pay'], palette='Greens')

plt.title('Base Pay', fontsize = 20)

plt.xlabel('Base Pay in Dollars')
    ############################################

    ##############    Benefits   ###############

    ############################################



plt.figure(figsize=(10,7))

plt.hist(df['Benefits'], color='g', bins = 30)

plt.title('Benefits', fontsize = 20)

plt.xlabel('Benefits in Dollars')
print('\n')

print('#########################')

print('    Benefits Statistics')

print('#########################\n')

print('    Mean    ',int(stats.mean(df['Benefits']))),

print('    Median  ',int(stats.median(df['Benefits']))),

print('    Mode    ',int(stats.mode(df['Benefits']))),

print('    Std     ',int(stats.stdev(df['Benefits'])))

print('\n')

#print(df['Benefits'].describe())
plt.figure(figsize=(10,7))

sns.boxplot(df['Benefits'], palette='Greens')

plt.title('Benefits', fontsize = 20)

plt.xlabel('Benefits in Dollars')
df[df['Benefits']<0]['Benefits'].value_counts()

    # there are over 30 people with benefits below zero

    # some are over $10,000 below zero!
df.groupby('Job Title')['Benefits'].min().sort_values(ascending=True).head(10)
    ###########################################

    #########    Benefits over time   #########

    ###########################################

    # are Benefits going up or down over time?

    # if they were increasing we would see a shift

    # to the right in the distribution.



    # benefits over time histogram

sns.set(font_scale=1.3)

a = sns.FacetGrid(df,col='Year',height =4,aspect= 0.9, col_wrap=3)

a.map(plt.hist, 'Benefits', color='g', bins = 40)

plt.suptitle('Benefits', fontsize = 20)

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    # benefits over time boxplot

plt.figure(figsize=(10,7))

sns.boxplot(df['Year'],df['Benefits'], palette='Greens')

plt.title('Benefits', fontsize = 20)

plt.xlabel('Benefits in Dollars')
    #####################################################

    ###########   Total Pay plus Benefits   #############

    #####################################################



    # what is the distribution for Total Pay plus Benefits?

plt.figure(figsize=(10,7))

plt.hist(df['Total Pay & Benefits'], color='g', bins = 40)

plt.title('Total Base Pay plus Benefits', fontsize = 20)

plt.xlabel('Total Pay in Dollars')
print('\n')

print('#########################')

print('   Total Pay Statistics')

print('#########################\n')

print('    Mean    ',int(stats.mean(df['Total Pay & Benefits']))),

print('    Median  ',int(stats.median(df['Total Pay & Benefits']))),

print('    Mode    ',int(stats.mode(df['Total Pay & Benefits']))),

print('    Std     ',int(stats.stdev(df['Total Pay & Benefits'])))

print('\n')

#print(df['Total Pay & Benefits'].describe())
    # boxplot

plt.figure(figsize=(10,7))

sns.boxplot(df['Total Pay & Benefits'], palette='Greens')

plt.title('Total Pay plus Benefits', fontsize = 18)
    # total pay over time histogram

sns.set(font_scale=1.3)

a = sns.FacetGrid(df, col='Year', height =4,aspect= 0.9, col_wrap=3)

a.map(plt.hist, 'Total Pay & Benefits', color='g', bins = 40)

plt.suptitle('Total Pay plus Benefits')

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    # total pay over time boxplot

plt.figure(figsize=(10,7))

sns.boxplot(df['Year'],df['Total Pay & Benefits'], palette='Greens')

plt.title('Total Pay plus Benefits', fontsize = 20)
# whose at the top of the pay scale?

df.groupby('Job Title')['Total Pay & Benefits'].max().sort_values(ascending=False).head(15)
    ###############################################

    ###############################################

    ######    above and below average       #######

    ######      Base Pay, Benefits          #######

    ######    Total Pay visulizations       #######

    ###############################################

    ###############################################

       

# create a new column with Above Average Boolean

#df['Base Pay'].mean() # 69297.51868379996

df['Above Average Base Pay'] = df['Base Pay'] > 69297.51868379996

df['AABP'] = df['Base Pay'] > 69297.51868379996



# and split the data into two datasets

ba = df[df['AABP']==False]

aa = df[df['AABP']==True]



[ba['Base Pay'].count(),aa['Base Pay'].count()]



# if we split the data into above and below average

# there are 122,768 poeple below and 111,646 above the mean

    #####################################

    ######  below average Base Pay  #####

    #####################################



b = sns.FacetGrid(ba, col='Year', height =4,aspect= 0.9, col_wrap=3)

b.map(plt.hist, 'Base Pay', color='g', bins = 30)

plt.suptitle('Below Average Base Pay')

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    # boxplot

fig, ax = plt.subplots(1, figsize = (10,6))

sns.boxplot(ba['Year'],ba['Base Pay'], palette='Greens')

plt.title('Below Average Base Pay', fontsize = 18)

plt.xlabel('Year', fontsize =16)

plt.ylabel("Base Pay", fontsize=16)
    #####################################

    ######  above average Base Pay  #####

    #####################################



a = sns.FacetGrid(aa, col='Year', height =4,aspect= 0.9, col_wrap=3)

a.map(plt.hist, 'Base Pay', color='g', bins = 20)

plt.suptitle('Above Average Base Pay')

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
# boxplot

fig, ax = plt.subplots(1, figsize = (10,6))

sns.boxplot(aa['Year'],aa['Base Pay'], palette='Greens')

plt.title('Above Average Base Pay', fontsize = 18)

plt.xlabel('Year', fontsize =16)

plt.ylabel("Base Pay", fontsize=16)
    ################################################

    ######  Below Average Base Pay (Benefits)  #####

    ################################################



b = sns.FacetGrid(ba, col='Year', height =4,aspect= 0.9, col_wrap=3)

b.map(plt.hist, 'Benefits', color='g', bins = 30)

plt.suptitle('Below Average Base Pay (Benefits)')

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    # boxplot

fig, ax = plt.subplots(1, figsize = (10,6))

sns.boxplot(ba['Year'],ba['Benefits'], palette='Greens')

plt.title('Below Average Base Pay (Benefits)', fontsize = 18)

plt.xlabel('Year', fontsize =16)

plt.ylabel('Benefits', fontsize=16)
    ################################################

    ######  Above Average Base Pay (Benefits)  #####

    ################################################



a = sns.FacetGrid(aa, col='Year', height =4,aspect= 0.9, col_wrap=3)

a.map(plt.hist, 'Benefits', color='g', bins = 30)

plt.suptitle('Above Average Base Pay (Benefits)')

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
# above average

fig, ax = plt.subplots(1, figsize = (10,6))

sns.boxplot(aa['Year'],aa['Benefits'], palette='Greens')

plt.title('Above Average Base Pay (Benefits)', fontsize = 18)

plt.xlabel('Year', fontsize =16)

plt.ylabel("Benefits", fontsize=16)

plt.tight_layout()
    #############################################

    #######    Below Average Base Pay    ########

    #######   (Base Pay plus Benefits)   ########

    #############################################



b = sns.FacetGrid(ba, col='Year', height =4,aspect= 0.9, col_wrap=3)

b.map(plt.hist, 'Total Pay & Benefits', color='g', bins = 30)

plt.suptitle('Below Average Base Pay (Pay & Benefits)')

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
fig, ax = plt.subplots(1, figsize = (10,6))

sns.boxplot(ba['Year'],ba['Total Pay & Benefits'], palette='Greens')

plt.title('Below Average Base Pay (Pay & Benefits)', fontsize = 18)

plt.xlabel('Year', fontsize =16)

plt.ylabel("Total Pay & Benefits", fontsize=16)

plt.tight_layout()
    #############################################

    #######    Above Average Base Pay    ########

    #######   (Base Pay plus Benefits)   ########

    #############################################



a = sns.FacetGrid(aa, col='Year', height =4,aspect= 0.9, col_wrap=3)

a.map(plt.hist, 'Total Pay & Benefits', color='g', bins = 30)

plt.suptitle('Above Average Base Pay (Pay & Benefits)')

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
fig, ax = plt.subplots(1, figsize = (10,6))

sns.boxplot(aa['Year'],aa['Total Pay & Benefits'], palette='Greens')

plt.title('Above Average Base Pay (Pay & Benefits)', fontsize = 18)

plt.xlabel('Year', fontsize =16)

plt.ylabel("Total Pay & Benefits", fontsize=16)

plt.tight_layout()
    #################################################

    ########    Change from 2012 to 2017    #########

    #################################################



# df = df[df['Year'].isin([2012,2017])]

ba = ba[ba['Year'].isin([2012,2017])]

aa = aa[aa['Year'].isin([2012,2017])]
    #############################################

    ####   Below Avg Base Pay (2012-2017)   #####

    #############################################



f, axes = plt.subplots(1,2, figsize = (12,7))

ax1 = plt.subplot(121)

plt.hist(df[(df['Above Average Base Pay'] == False)

    & (df['Year'] == 2012)]['Base Pay'], color='g', bins = 30)

plt.title("2012", fontsize = 20)

plt.xlabel('Base Pay', fontsize = 16)

plt.ylabel('Frequency',fontsize = 16)

ax2 = plt.subplot(122, sharex= ax1, sharey = ax1)

plt.hist(df[(df['Above Average Base Pay'] == False)

    & (df['Year'] == 2017)]['Base Pay'], color='g', bins = 30)

plt.title("2017", fontsize = 20)

plt.xlabel('Base Pay', fontsize = 16)

plt.ylabel('Frequency',fontsize = 16)

plt.suptitle('Below Average Base Pay', fontsize = 25)

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
fig, ax = plt.subplots(1, figsize = (10,6))

sns.boxplot(ba['Year'],ba['Base Pay'], palette='Greens')

plt.title('Below Average Base Pay', fontsize = 18)

plt.xlabel('Year', fontsize =16)

plt.ylabel("Base  Pay", fontsize=16)

plt.tight_layout()
    #############################################

    ####   Above Avg Base Pay (2012-2017)   #####

    #############################################



f, axes = plt.subplots(1,2, figsize = (12,7))

ax1 = plt.subplot(121)

plt.hist(df[(df['Above Average Base Pay'] == True)

    & (df['Year'] == 2012)]['Base Pay'], color='g', bins = 30)

plt.title("2012", fontsize = 20)

plt.xlabel('Base Pay', fontsize = 16)

plt.ylabel('Frequency',fontsize = 16)

ax2 = plt.subplot(122, sharex= ax1, sharey = ax1)

plt.hist(df[(df['Above Average Base Pay'] == True)

    & (df['Year'] == 2017)]['Base Pay'], color='g', bins = 30)

plt.title("2017", fontsize = 20)

plt.xlabel('Base P ay', fontsize = 16)

plt.ylabel('Frequency',fontsize = 16)

plt.suptitle('Above Average Base Pay', fontsize = 25)

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
fig, ax = plt.subplots(1, figsize = (10,6))

sns.boxplot(aa['Year'],aa['Base Pay'], palette='Greens')

plt.title('Above Average Base Pay', fontsize = 25)

plt.xlabel('Year', fontsize =16)

plt.ylabel("Base Pay", fontsize=16)

plt.tight_layout()
    ############################################

    ####  Below Avg Base Pay (Benefits)    #####

    ######          (2012-2017)         ########

    ############################################



f, axes = plt.subplots(1,2, figsize = (12,7))

ax1 = plt.subplot(121)

plt.hist(df[(df['Above Average Base Pay'] == False)

    & (df['Year'] == 2012)]['Benefits'], color='g', bins = 30)

plt.title("2012", fontsize = 20)

plt.xlabel('Benefits', fontsize = 16)

plt.ylabel('Frequency',fontsize = 16)

ax2 = plt.subplot(122, sharex= ax1, sharey = ax1)

plt.hist(df[(df['Above Average Base Pay'] == False)

    & (df['Year'] == 2017)]['Benefits'],color='g', bins = 30)

plt.title("2017", fontsize = 20)

plt.xlabel('Benefits', fontsize = 16)

plt.ylabel('Frequency',fontsize = 16)

plt.suptitle('Below Average Base Pay (Benefits)', fontsize = 25)

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
fig, ax = plt.subplots(1, figsize = (10,6))

sns.boxplot(ba['Year'],ba['Benefits'],palette='Greens')

plt.title('Below Average Base Pay (Benefits)', fontsize = 18)

plt.xlabel('Year', fontsize =16)

plt.ylabel("Benefits", fontsize=16)

plt.tight_layout()
    ############################################

    ####  Above Avg Base Pay (Benefits)    #####

    ######          (2012-2017)         ########

    ############################################





f, axes = plt.subplots(1,2, figsize = (12,7))

ax1 = plt.subplot(121)

plt.hist(df[(df['Above Average Base Pay'] == True)

    & (df['Year'] == 2012)]['Benefits'],color='g', bins = 30)

plt.title("2012", fontsize = 20)

plt.xlabel('Benefits', fontsize = 16)

plt.ylabel('Frequency',fontsize = 16)

ax2 = plt.subplot(122, sharex= ax1, sharey = ax1)

plt.hist(df[(df['Above Average Base Pay'] == True)

    & (df['Year'] == 2017)]['Benefits'],color='g', bins = 30)

plt.title("2017", fontsize = 20)

plt.xlabel('Benefits', fontsize = 16)

plt.ylabel('Frequency',fontsize = 16)

plt.suptitle('Above Average Base Pay (Benefits)', fontsize = 25)

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
fig, ax = plt.subplots(1, figsize = (10,6))

sns.boxplot(aa['Year'],aa['Benefits'],palette='Greens')

plt.title('Above Average Base Pay (Benefits)', fontsize = 18)

plt.xlabel('Year', fontsize =16)

plt.ylabel("Benefits", fontsize=16)

plt.tight_layout()
    ###############################################

    #########   below average(2012-2017)  #########

    #########    total pay and benefits   #########

    ###############################################





f, axes = plt.subplots(1,2, figsize = (12,7))

ax1 = plt.subplot(121)

plt.hist(df[(df['Above Average Base Pay'] == False)

    & (df['Year'] == 2012)]['Total Pay & Benefits'],color='g', bins = 30)

plt.title("2012", fontsize = 20)

plt.xlabel('Total Pay & Benefits', fontsize = 16)

plt.ylabel('Frequency',fontsize = 16)

ax2 = plt.subplot(122, sharex= ax1, sharey = ax1)

plt.hist(df[(df['Above Average Base Pay'] == False)

    & (df['Year'] == 2017)]['Total Pay & Benefits'],color='g', bins = 30)

plt.title("2017", fontsize = 20)

plt.xlabel('Total Pay & Benefits', fontsize = 16)

plt.ylabel('Frequency',fontsize = 16)

plt.suptitle('Below Average Base Pay (Pay plus Benefits)', fontsize = 25)

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
fig, ax = plt.subplots(1, figsize = (10,6))

sns.boxplot(ba['Year'],ba['Total Pay & Benefits'],palette='Greens')

plt.title('Below Average Base Pay (Pay & Benefits)', fontsize = 18)

plt.xlabel('Year', fontsize =16)

plt.ylabel("Total Pay & Benefits", fontsize=16)

plt.tight_layout()
    ###############################################

    #########   Above average(2012-2017)  #########

    #########    total pay and benefits   #########

    ###############################################





f, axes = plt.subplots(1,2, figsize = (12,7))

ax1 = plt.subplot(121)

plt.hist(df[(df['Above Average Base Pay'] == True)

    & (df['Year'] == 2012)]['Total Pay & Benefits'],color='g', bins = 30)

plt.title("2012", fontsize = 20)

plt.xlabel('Total Pay & Benefits', fontsize = 16)

plt.ylabel('Frequency',fontsize = 16)

ax2 = plt.subplot(122, sharex= ax1, sharey = ax1)

plt.hist(df[(df['Above Average Base Pay'] == True)

    & (df['Year'] == 2017)]['Total Pay & Benefits'],color='g', bins = 30)

plt.title("2017", fontsize = 20)

plt.xlabel('Total Pay & Benefits', fontsize = 16)

plt.ylabel('Frequency',fontsize = 16)

plt.suptitle('Above Average Base Pay (Pay plus Benefits)', fontsize = 25)

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
fig, ax = plt.subplots(1, figsize = (10,6))

sns.boxplot(aa['Year'],aa['Total Pay & Benefits'],palette='Greens')

plt.title('Above Average Base Pay (Pay & Benefits)', fontsize = 18)

plt.xlabel('Year', fontsize =16)

plt.ylabel("Total Pay & Benefits", fontsize=16)

plt.tight_layout()
    ###########################################

    #########    Base Pay over time   #########

    ###########################################



bp_2012_mean = np.mean(df[df['Year']==2012]['Base Pay'])

bp_2017_mean = np.mean(df[df['Year']==2017]['Base Pay'])

bp_2012_max = np.max(df[df['Year']==2012]['Base Pay'])

bp_2017_max = np.max(df[df['Year']==2017]['Base Pay'])

bp_2012_std = np.std(df[df['Year']==2012]['Base Pay'])

bp_2017_std = np.std(df[df['Year']==2017]['Base Pay'])



    # Average Base Pay

abp = round((bp_2017_mean - bp_2012_mean)/bp_2012_mean * 100,1)

    # Maximum Base Pay max

mbp = round((bp_2017_max - bp_2012_max)/bp_2012_max * 100,1)

    # Base Pay Standard Deviation

bpstd = round((bp_2017_std - bp_2012_std)/bp_2012_std * 100,1)
    ###########################################

    #########    Benefits over time   #########

    ###########################################



b_2012_mean = np.mean(df[df['Year']==2012]['Benefits'])

b_2017_mean = np.mean(df[df['Year']==2017]['Benefits'])

b_2012_max = np.max(df[df['Year']==2012]['Benefits'])

b_2017_max = np.max(df[df['Year']==2017]['Benefits'])

b_2012_std = np.std(df[df['Year']==2012]['Benefits'])

b_2017_std = np.std(df[df['Year']==2017]['Benefits'])



    # Average Benefits

ab = round((b_2017_mean - b_2012_mean)/b_2012_mean * 100,1)

    # Maximum Benefits

mb = round((b_2017_max - b_2012_max)/b_2012_max * 100,1)

    # Benefits Standard Deviation

bstd = round((b_2017_std - b_2012_std)/b_2012_std * 100,1)

    #####################################################

    ###########   Total Pay plus Benefits   #############

    #####################################################



ppb_2012_mean = np.mean(df[df['Year']==2012]['Total Pay & Benefits'])

ppb_2017_mean = np.mean(df[df['Year']==2017]['Total Pay & Benefits'])

ppb_2012_max = np.max(df[df['Year']==2012]['Total Pay & Benefits'])

ppb_2017_max = np.max(df[df['Year']==2017]['Total Pay & Benefits'])

ppb_2012_std = np.std(df[df['Year']==2012]['Total Pay & Benefits'])

ppb_2017_std = np.std(df[df['Year']==2017]['Total Pay & Benefits'])



    # Average Total Pay & Benefits

aTpb = round((ppb_2017_mean - ppb_2012_mean)/ppb_2012_mean * 100,1)

    # Maximum Total Pay & Benefits

mTpb = round((ppb_2017_max - ppb_2012_max)/ppb_2012_max * 100,1)

    # Total Pay & Benefits Standard Deviation

stdTpb = round((ppb_2017_std - ppb_2012_std)/ppb_2012_std * 100,1)

    ###############################################

    ###############################################

    ######    above and below average       #######

    ###############################################

    ###############################################
    ##############################################

    ###########      base  pay      ##############

    ##############################################



# below average base pay (mean)

babp_15 = np.mean(ba[ba['Year']==2012]['Base Pay'])

babp_17 = np.mean(ba[ba['Year']==2017]['Base Pay'])

babp = round((babp_17 - babp_15)/babp_15 * 100,1)

# above average base pay (mean)

aabp_15 = np.mean(aa[aa['Year']==2012]['Base Pay'])

aabp_17 = np.mean(aa[aa['Year']==2017]['Base Pay'])

aabp = round((aabp_17 - aabp_15)/aabp_15 * 100,1)





# below average base pay (max)

babpmax_15 = np.max(ba[ba['Year']==2012]['Base Pay'])

babpmax_17 = np.max(ba[ba['Year']==2017]['Base Pay'])

babpmax = round((babpmax_17 - babpmax_15)/babpmax_15*100,1)



# above average base pay (max)

aabpmax_15 = np.max(aa[aa['Year']==2012]['Base Pay'])

aabpmax_17 = np.max(aa[aa['Year']==2017]['Base Pay'])

aabpmax = round((aabpmax_17-aabpmax_15)/aabpmax_15*100,1)



# below average base pay (std)

babpstd_15 = np.std(ba[ba['Year']==2012]['Base Pay'])

babpstd_17 = np.std(ba[ba['Year']==2017]['Base Pay'])

babpstd = round((babpstd_17 -babpstd_15 )/babpstd_15 *100,1)



# above average base pay (std)

aabpstd_15 = np.std(aa[aa['Year']==2012]['Base Pay'])

aabpstd_17 = np.std(aa[aa['Year']==2017]['Base Pay'])

aabpstd  = round((aabpstd_17-aabpstd_15)/aabpstd_15*100,1)
    ################################################

    ###########        benefits       ##############

    ################################################



# below average base pay (benefits)

babpben_15 = np.mean(ba[ba['Year']==2012]['Benefits'])

babpben_17 = np.mean(ba[ba['Year']==2017]['Benefits'])

babpben = round((babpben_17-babpben_15)/babpben_15*100,1)



# above average base pay (benefits)

aabpben_15 = np.mean(aa[aa['Year']==2012]['Benefits'])

aabpben_17 = np.mean(aa[aa['Year']==2017]['Benefits'])

aabpben = round((aabpben_17-aabpben_15)/aabpben_15*100,1)



# below average base pay (benefits)(max)

babpbenmax_15 = np.max(ba[ba['Year']==2012]['Benefits'])

babpbenmax_17 = np.max(ba[ba['Year']==2017]['Benefits'])

babpbenmax = round((babpbenmax_17-babpbenmax_15)/babpbenmax_15*100,1)



# above average base pay (benefits)(max)

aabpbenmax_15 = np.max(aa[aa['Year']==2012]['Benefits'])

aabpbenmax_17 = np.max(aa[aa['Year']==2017]['Benefits'])

aabpbenmax = round((aabpbenmax_17-aabpbenmax_15)/aabpbenmax_15*100,1)



# below average base pay (benefits)(std)

babpbenstd_15 = np.std(ba[ba['Year']==2012]['Benefits'])

babpbenstd_17 = np.std(ba[ba['Year']==2017]['Benefits'])

babpbenstd = round((babpbenstd_17-babpbenstd_15)/babpbenstd_15*100,1)



# above average base pay (benefits)(std)

aabpbenstd_15 = np.std(aa[aa['Year']==2012]['Benefits'])

aabpbenstd_17 = np.std(aa[aa['Year']==2017]['Benefits'])

aabpbenstd = round((aabpbenstd_17-aabpbenstd_15)/aabpbenstd_15*100,1)
    ################################################

    ########   total pay plus benefits     #########

    ################################################



# below average base pay (Total Pay & Benefits)

babpT_15 = np.mean(ba[ba['Year']==2012]['Total Pay & Benefits'])

babpT_17 = np.mean(ba[ba['Year']==2017]['Total Pay & Benefits'])

babpT = round((babpT_17-babpT_15)/babpT_15*100,1)



# above average base pay (Total Pay & Benefits)

aabpT_15 = np.mean(aa[aa['Year']==2012]['Total Pay & Benefits'])

aabpT_17 = np.mean(aa[aa['Year']==2017]['Total Pay & Benefits'])

aabpT = round((aabpT_17-aabpT_15)/aabpT_15*100,1)



# below average base pay (Total Pay & Benefits)(max)

babpTmax_15 = np.max(ba[ba['Year']==2012]['Total Pay & Benefits'])

babpTmax_17 = np.max(ba[ba['Year']==2017]['Total Pay & Benefits'])

babpTmax = round((babpTmax_17-babpTmax_15)/babpTmax_15*100,1)



# above average base pay (Total Pay & Benefits)(max)

aabpTmax_15 = np.max(aa[aa['Year']==2012]['Total Pay & Benefits'])

aabpTmax_17 = np.max(aa[aa['Year']==2017]['Total Pay & Benefits'])

aabpTmax=round((aabpTmax_17-aabpTmax_15)/aabpTmax_15*100,1)



# below average base pay (Total Pay & Benefits)(std)

babpTstd_15 = np.std(ba[ba['Year']==2012]['Total Pay & Benefits'])

babpTstd_17 = np.std(ba[ba['Year']==2017]['Total Pay & Benefits'])

babpTstd=round((babpTstd_17-babpTstd_15)/babpTstd_15*100,1)



# above average base pay (Total Pay & Benefits)(std)

aabpTstd_15 = np.std(aa[aa['Year']==2012]['Total Pay & Benefits'])

aabpTstd_17 = np.std(aa[aa['Year']==2017]['Total Pay & Benefits'])

aabpTstd=round((aabpTstd_17-aabpTstd_15)/aabpTstd_15*100,1)
print('\n')

print('############################################')

print('############################################\n')

print('       San Francisco Salary Report \n')

print('              2012 to 2017\n')

print('############################################')

print('############################################\n\n')

print('\nPercentage Change from 2012 to 2017\n')

print('______________________________________\n')

print('######     Base Pay    #######')

print('______________________________________\n')

print('Average Base Pay              {0} %'.format(abp))

print('Max Base Pay                  {0} %'.format(mbp))

print('Standard Deviation            {0} %\n'.format(bpstd))

print('______________________________________\n')

print('######     Benefits     ######')

print('______________________________________\n')

print('Average Benefits              {0} %'.format(ab))

print('Max Benefits                  {0} %'.format(mb))

print('Standard Deviation            {0} %\n'.format(bstd))



print('______________________________________\n')

print('Total Salary (Base Pay plus Benefits)')

print('______________________________________\n')

print('Average                       {0} %'.format(aTpb))

print('Max                           {0} %'.format(mTpb))

print('Standard Deviation            {0} %\n'.format(stdTpb))

print('______________________________________\n')

print('  Below and Above Average Base Pay  ')

print('______________________________________\n')

print('Below Average Base Pay        {0} %'.format(babp))

print('Below Average Max Pay         {0} %\n'.format(babpmax))

print('Above Average Base Pay        {0} %'.format(aabp))

print('Above Average Max Pay         {0} %\n'.format(aabpmax))

print('Below Average Benefits        {0} %'.format(babpben))

print('Below Average Max Benefits    {0} %\n'.format(babpbenmax))

print('Above Average Benefits        {0} %'.format(aabpben))

print('Above Average Max Benefits    {0} %\n'.format(aabpbenmax))

print('______________________________________\n')

print('Total Salary (Base Pay plus Benefits)')

print('______________________________________\n')

print('Below Average                 {0} %'.format(babpT))

print('Below Average Max             {0} %\n'.format(babpTmax))

print('Above Average                 {0} %'.format(aabpT))

print('Above Average Max             {0} %\n'.format(aabpTmax))

print('\n')