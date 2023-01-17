import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



data = pd.read_csv('/kaggle/input/phd-stipends/csv')

data_original = data.copy()

data
#searching for missing values in our data! We also notice that 12m,9m,3m gross pay and fees are all included in overallpay column so we might have to reduce our dataframe for simplicity

data.isnull().sum()
#we notice that we have a lot of missing values in the 4 columns mentioned above as well as Comments so we immediately remove them and simplify our dataframe and keep columns we find valuable for our analysis

data = data.loc[:][['University','Department','Overall Pay','LW Ratio','Academic Year','Program Year']]

data
# checking the types of our data!

data.dtypes
#LW ratio is float but Overall Pay is object,we have to make it a float as well in order to procceed to our analysis.

data['Overall Pay'] = data['Overall Pay'].str.replace('$','').str.replace(',','').astype(float)

data['Overall Pay']
#we now have Overall Pay column as we want it for our analysis,quick rename to Pay to make it more practical and a describe statistic:

data.rename(columns={'Overall Pay':'Pay'},inplace=True)

data.Pay.describe()
#we notice negative min value so we quickly investigate how many negative values we have, if large number it indicates outliers,else it might imply a PhD u have to pay to participate in overall

investigate = data[data['Pay']<0].Pay
#We have 74 negative values with a mean of 26,6 thousand dollars which might indicate that in those Universities you have to pay in order to undertake a PhD ,lets quickly plot a distribution to visualize our findings:

plt.hist(investigate,bins=100)

plt.show()
#900 thousand $ is indeed an outlier , let's see which university it is:

data[data.Pay==-900000]
#Further Investigate this University using its name as a search querry:

data[data['University']=='University of California - San Diego (UCSD)']
# This University offers 120 PhD programs, and all of them seem to pay , lets find the average:

data[data['University']=='University of California - San Diego (UCSD)'].Pay.mean()
data[data['University']=='University of California - San Diego (UCSD)'].describe()
#We might want to replace our outlier with the mean value of this University's Pay but let's just remove it from our sample:

data.drop([3350],axis=0,inplace=True)
data[data['University']=='University of California - San Diego (UCSD)'].describe()

#as we see it is now removed , let's now quickly view which of those Programs cost money to participate in with the most expensive one being close to 30000$.
data[(data['University']=='University of California - San Diego (UCSD)') & (data['Pay']<0)]
#Let's see if we can find information on comments of those values:

data[(data['University']=='University of California - San Diego (UCSD)') & (data['Pay']<0)].index.tolist()
#now we got the indexes lets find comments of those from original list:

data_original.iloc[[2447,5395,5821]]['Comments']
#No help from comments, anyway let's find the top 10 highest paying Universities if you wanted to start a PhD and get paid Studying!

top_ten = data['Pay'].nlargest(10)
#Those numbers see a bit high especially the million one! let's investigate on Original Data Set:

index_investigation = top_ten.index.tolist()
data_original.iloc[index_investigation]
#Above we can see that for 2 of those universities we have absolutely no information other than Overall Pay so we must remove all missing University values:

data['University'].isna().sum()
#Let's remove those 263 values

data['University'].isna()
#the missing University values contain the following:

data[data['University'].isna()]
#Let's quickly check how many missing value in total we have in this new dataframe, with a quick glance they see many!

data[data['University'].isna()].isna().sum()
#a very large number, let's remove all from original dataframe!
rem_row = data[data['University'].isna()].index.tolist()

rem_row
data.loc[rem_row]
data.drop(rem_row,axis=0,inplace=True)
#no we can move on to our analysis with highest and lowest paying universities!

highest = data.Pay.nlargest(10)

highest_idx = data.Pay.nlargest(10).index.tolist()
lowest = data.Pay.nsmallest(10)

lowest_idx = data.Pay.nsmallest(10).index.tolist()
#The top 10 Highest Paying Unis are!(Complete List):

data.loc[highest_idx]
#The top 10 Lowest Paying Unis are!:(Complete List):

data.loc[lowest_idx]
#of course we were expecting negative values , let's see if we have any comments on those on our original Dataset!

data_original.loc[lowest_idx]
#After a glance at full table we notice that those are indeed the Universities with the 'lowest' pay since you have to subtract what they pay you from your fees and you get your total net pay which is negative!
top_h = data.loc[highest_idx]['University']

top_h
top_l = data.loc[lowest_idx]['University']

top_l
top_h.reset_index(drop=True,inplace=True)

top_l.reset_index(drop=True,inplace=True)
top_h.rename('HighestPaying_University',inplace=True)

top_l.rename('LowestPaying_University',inplace=True)
df = pd.concat([top_h,top_l],axis=1)
df