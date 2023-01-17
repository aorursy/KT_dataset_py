# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import math as mt 

get_ipython().magic(u'pylab inline')

import matplotlib.patches as mpatches

import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



responses_data = pd.read_csv('../input/responses.csv') 

columns_data = pd.read_csv('../input/columns.csv')



# Any results you write to the current directory are saved as output.



## Load columns from csv file to dataframe to make pandas analysis simpler and get an idea of which 

## columns to analyze 

df = pd.DataFrame({'Alcohol':responses_data['Alcohol'], 

                   "Loneliness":responses_data['Loneliness'],

                   "Health":responses_data['Health'],"Age":responses_data['Age'] })

Alcohol = df['Alcohol']

Health = df['Health']

## Rough look at the dataframe 

## Count the different types of alcoholics there are 

print("Different types of Alcohol users and count \n")

group_alcohol = df.Alcohol.value_counts()

print(group_alcohol)



## See statistics in Alcohol vs Health (People that worry about their health on a scale of 1-5) 

grouped_data = df.groupby(['Alcohol','Health']).size()





## Table counting how many people there are in each health rating by alcohol category

print(df.pivot_table(index = ['Health'], columns='Alcohol', values='Loneliness', aggfunc=len)) 



## Proportion/percentage of people in each Health rating by Alcohol usage category, e.g. 

## there are 30 people in Health 1.0 and Alcohol "drink a lot", so that is 13% out of everyone 

## in the "drink a lot" column 

print(" ")

print("Percentage of each Health rating by Alcohol usage category \n")

health_alcohol_percent = df.pivot_table(index = ['Health'], columns='Alcohol', values='Loneliness', aggfunc=len).apply(lambda y: y/y.sum())

print(health_alcohol_percent.applymap(lambda x: "{:.0f}%".format(100*x))) 
## Bar plot for previous pivot table (count of people in each rating by alcohol usage category)

import matplotlib.gridspec as gridspec



fig1 = plt.figure(figsize=[15,8])

gs = GridSpec(100,100)

ax1 = fig1.add_subplot(gs[:70,0:40])

ax2 = fig1.add_subplot(gs[:70,60:100])



plot_1 = df.pivot_table(index = ['Health'], columns='Alcohol', values='Loneliness', aggfunc=len).plot(kind='bar',stacked=False,ax=ax1).set_ylabel("Count")

plot_2 = health_alcohol_percent.plot(kind='bar',grid=True,ax=ax2).set_ylabel("Proportion") 



drinkers = np.array(Alcohol[(Alcohol == 'drink a lot') | (Alcohol == 'social drinker')].values)



df_1 = pd.DataFrame({'Drinkers':Alcohol.isin(drinkers),

                     "Non-drinkers":df['Alcohol'] == 'never',"Health":df['Health']})

df_1_count = df_1.groupby('Health').sum()

df_1_totals = df_1.groupby('Health').sum()

df_1_totals.loc['Total']= df_1_totals.sum()

print(df_1_totals)



print(" ")

print("Percent of people in each Health rating by Drinking category")

df_1_percent = df_1_count.apply(lambda y: y/y.sum())

print(df_1_percent.applymap(lambda x: "{:.0f}%".format(100*x)))

fig2 = plt.figure(figsize=[15,8])

gs = GridSpec(100,100)

ax3 = fig2.add_subplot(gs[:70,0:40])

ax4 = fig2.add_subplot(gs[:70,60:100])

print("Drinkers and Non-drinkers by Health Rating")

df_1_count.plot(kind='bar',ax=ax3).set_ylabel("Count") 

df_1_percent.plot(kind='bar',ax=ax4).set_ylabel("Proportion")



def label_rating(row):

    if row['Health'] == 1 or row['Health'] == 2:

        return "Not-concerned"

    if row['Health'] == 3:

        return "Moderate"

    if row['Health'] == 4 or row['Health'] == 5:

        return "Very-concerned"



def label_alcohol(row):

    if row['Alcohol'] == 'never':

        return "Non-drinker"

    if row['Alcohol'] == 'social drinker' or row['Alcohol'] == 'drink a lot':

        return "Drinker"

    

df['health_label'] = df.apply (lambda row: label_rating (row),axis=1)

df['Alcohol_usage'] = df.apply (lambda row:label_alcohol (row), axis = 1)



df_2 = pd.DataFrame({'Drinkers':Alcohol.isin(drinkers),"Non-drinkers":df['Alcohol'] == 'never',"Health_labels":df['health_label']})



df_2_count = df_2.groupby('Health_labels').sum()

df_2_totals = df_2.groupby('Health_labels').sum()

df_2_totals.loc['Total']= df_2_totals.sum()

print(df_2_totals)



print(" ")

print("Percent of people in each Health rating by Drinking category")

df_2_percent = df_2_count.apply(lambda y: y/y.sum())

print(df_2_percent.applymap(lambda x: "{:.0f}%".format(100*x)))





fig3 = plt.figure(figsize=[15,8])

gs = GridSpec(100,100)

ax5 = fig3.add_subplot(gs[:70,0:40])

ax6 = fig3.add_subplot(gs[:70,60:100])

df_2_count.plot(kind='bar',ax=ax5).set_ylabel("Count") 

df_2_percent.plot(kind='bar',ax=ax6).set_ylabel("Proportion")
## Table counting how many people there are in each health rating by alcohol category

print(df.pivot_table(index = ['Alcohol'], columns='Health', values='Loneliness', aggfunc=len)) 



## Proportion/percentage of people in each Health rating by Alcohol usage category, e.g. 

## there are 30 people in Health 1.0 and Alcohol "drink a lot", so that is 13% out of everyone 

##  in the drink a lot column 

print(" \n Percentage of people in each Alcohol usage category by Health Rating \n")

health_alcohol_percent_1 = df.pivot_table(index = ['Alcohol'], columns='Health', values='Loneliness', aggfunc=len).apply(lambda y: y/y.sum())

print(health_alcohol_percent_1.applymap(lambda x: "{:.0f}%".format(100*x)))

fig4 = plt.figure(figsize=[15,8])

gs = GridSpec(100,100)

ax7 = fig4.add_subplot(gs[:70,0:40])

ax8 = fig4.add_subplot(gs[:70,60:100])



df.pivot_table(index = ['Alcohol'], columns='Health', values='Loneliness', aggfunc=len).plot(kind='bar',ax=ax7).set_ylabel("Count")

health_alcohol_percent_1.plot(kind='bar',ax=ax8,title='Proportion of Alcohol users in each Health rating').set_ylabel("Proportion")

print(" ")

df_4 = pd.DataFrame({"Alcohol_usage":df['Alcohol_usage'], "Health":df['Health'],"Loneliness":df['Loneliness']})

## Table for Drinkers and non-drinkers 

df_4_count = df_4.pivot_table(index = ['Alcohol_usage'], columns='Health', values='Loneliness', aggfunc=len)

print(df_4_count)



print(" ")

## Shows percents of previous table 

print("Percent of people in each Alcohol usage by Health rating")

df_4_percent = df_4_count.apply(lambda y: y/y.sum())

print(df_4_percent.applymap(lambda x: "{:.0f}%".format(100*x)))



fig5 = plt.figure(figsize=[15,8])

gs = GridSpec(100,100)

ax9 = fig5.add_subplot(gs[:70,0:40])

ax10 = fig5.add_subplot(gs[:70,60:100])





df_4_count.plot(kind='bar',ax=ax9,title="Alcohol usage by Health Rating").set_ylabel("Count")

plt.ylabel("Proportion")

df_4_percent.plot(kind='bar',ax=ax10,title="Proportion of each Health rating").set_ylabel("Proportion")




