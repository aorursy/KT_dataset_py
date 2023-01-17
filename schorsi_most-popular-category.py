import numpy as np

import pandas as pd

import os
app_dat = pd.read_csv('/kaggle/input/google-play-store-apps/googleplaystore.csv')

app_dat.head()

## Notice how the installs column does not have an integer value
##Probably should have seen this mistake coming, gave me a good laugh

##essentially this just adds all the strings from "Installs" into one large superstring for each catagory



agg_df = app_dat.groupby(['Category'])['Installs'].sum()

display(agg_df)

#now to see exactly what string values are in the 'Installs' catagory

app_dat.Installs.unique()
app_dat.info()
app_dat.isnull().sum()
over_list = ['1,000,000,000+', '500,000,000+', '100,000,000+', '50,000,000+', '10,000,000+', '5,000,000+', '1,000,000+', '500,000+', '100,000+', '50,000+', '10,000+', '5,000+', '1,000+', '500+', '100+', '50+', '10+', '5+', '1+', '0+', '0']

#this list orders the bins for easier graphing             
import matplotlib as plt

import seaborn as sns

import matplotlib.pyplot
matplotlib.pyplot.figure(figsize=(22,6))

sns.countplot(app_dat.Installs, order=over_list, orient='v')
#'Free' is a rather wacky value to have here, let's look closer

free = app_dat['Installs'] == 'Free'

display(app_dat[free])

wack = app_dat['Category'] == '1.9'

display(app_dat[wack])



##turns out this value is also unique in catagories as well. clearly this data is not intact

##this is an anomoly in the dataset and should be ignored

##by shifting all entries one column this entry makes sence but we wouldn't have an accurate catagory for it
## removing entry

app_dat.drop(app_dat[wack].index, axis=0, inplace=True)
## I could have used a dictionary and mapping function, which would have been more concise

## This perticular layout just helped me to see the gaps between values better

def installs_min(stng):

    if stng == '1,000,000,000+':

        return 1000000001

    if stng == '500,000,000+':

        return 500000001

    if stng == '100,000,000+':

        return 100000001

    if stng == '50,000,000+':

        return 50000001

    if stng == '10,000,000+':

        return 10000001

    if stng == '5,000,000+':

        return 5000001

    if stng == '1,000,000+':

        return 1000001

    if stng == '500,000+':

        return 500001

    if stng == '100,000+':

        return 100001

    if stng == '50,000+':

        return 50001

    if stng == '10,000+':

        return 10001

    if stng == '5,000+':

        return 5001

    if stng == '1,000+':

        return 1001

    if stng == '500+':

        return 501

    if stng == '100+':

        return 101

    if stng == '50+':

        return 51

    if stng == '10+':

        return 11

    if stng == '5+':

        return 6

    if stng == '1+':

        return 2

    if stng == '0+':

        return 1

    if stng == '0':

        return 0

    if stng == 'Free': ##this entry should be deleted in the final version but lets not take chances

        return 0

    

def installs_mid(stng):

    if stng == '1,000,000,000+': ##I assume 5 billion would come next, yes this is getting silly

        return 2500000000

    if stng == '500,000,000+':

        return 750000000

    if stng == '100,000,000+':

        return 250000000

    if stng == '50,000,000+':

        return 75000000

    if stng == '10,000,000+':

        return 25000000

    if stng == '5,000,000+':

        return 7500000

    if stng == '1,000,000+':

        return 2500000

    if stng == '500,000+':

        return 750000

    if stng == '100,000+':

        return 250000

    if stng == '50,000+':

        return 75000

    if stng == '10,000+':

        return 25000

    if stng == '5,000+':

        return 7500

    if stng == '1,000+':

        return 2500

    if stng == '500+':

        return 750

    if stng == '100+':

        return 250

    if stng == '50+':

        return 75

    if stng == '10+':

        return 25

    if stng == '5+':

        return 8

    if stng == '1+':

        return 3

    if stng == '0+':

        return 1

    if stng == '0':

        return 0

    if stng == 'Free':

        return 

    

def installs_max(stng):#shows max value

    if stng == '1,000,000,000+':

        return 5000000000

    if stng == '500,000,000+':

        return 1000000000

    if stng == '100,000,000+':

        return 500000000

    if stng == '50,000,000+':

        return 100000000

    if stng == '10,000,000+':

        return 50000000

    if stng == '5,000,000+':

        return 10000000

    if stng == '1,000,000+':

        return 5000000

    if stng == '500,000+':

        return 1000000

    if stng == '100,000+':

        return 500000

    if stng == '50,000+':

        return 100000

    if stng == '10,000+':

        return 50000

    if stng == '5,000+':

        return 10000

    if stng == '1,000+':

        return 5000

    if stng == '500+':

        return 1000

    if stng == '100+':

        return 500

    if stng == '50+':

        return 100

    if stng == '10+':

        return 50

    if stng == '5+':

        return 10

    if stng == '1+':

        return 5

    if stng == '0+':

        return 1

    if stng == '0':

        return 0

    if stng == 'Free': ##this entry should be deleted in the final version but lets not take chances

        return 0
app_dat['min_installs'] = pd.Series([installs_min(x) for x in app_dat.Installs], index=app_dat.index)

app_dat['mid_installs'] = pd.Series([installs_mid(x) for x in app_dat.Installs], index=app_dat.index)

app_dat['max_installs'] = pd.Series([installs_max(x) for x in app_dat.Installs], index=app_dat.index)

app_dat.head(3)
#Now to aggrigate and sort the data

full_app = app_dat.groupby(['Category'])['max_installs', 'mid_installs', 'min_installs'].sum()

full_app = full_app.sort_values(by=['min_installs'],ascending=False)

full_app
#mako

matplotlib.pyplot.figure(figsize=(10,10))

sns.barplot(x=full_app['min_installs'], y=full_app.index, palette='mako')



#Additional ideas: bin bottum 5% of apps
#full_app = full_app.sort_values(by=['mid_installs'],ascending=False)

matplotlib.pyplot.figure(figsize=(10,10))

#with sns.palplot(sns.color_palette("BuGn_r")):

sns.barplot(x=full_app['mid_installs'], y=full_app.index, palette='BuGn_r')

#sns.palplot(sns.color_palette("BuGn_r"))
matplotlib.pyplot.figure(figsize=(10,10))

sns.barplot(x=full_app['max_installs'], y=full_app.index, palette='Blues_r')

#when we replace each value with the max for its bin we start seeing communication overtake games as the installed catagory
short_app = full_app.head(10) ##shorter list of the top entries

##one other alternative that I did not try was to group all the entries that have fewer than 5% of the installs
import matplotlib.pyplot as plt

f, ax = plt.subplots(figsize=(10, 8))



sns.set_color_codes("pastel")

sns.barplot(x=short_app['max_installs'], y=short_app.index,

            label="Maximum possible value", color="b")



sns.set_color_codes("muted")

sns.barplot(x=short_app['mid_installs'], y=short_app.index,

            label="Median value", color="b")



sns.set_color_codes("dark")

sns.barplot(x=short_app['min_installs'], y=short_app.index,

            label="Minimum possible value", color="b")



ax.legend(ncol=3, loc="lower right", frameon=True)

ax.set( ylabel="Catagory",

       xlabel="Number of Installs")

sns.despine(left=True, bottom=True)