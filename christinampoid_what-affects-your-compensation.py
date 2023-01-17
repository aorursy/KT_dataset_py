import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt



df = pd.read_csv('../input/multipleChoiceResponses.csv', encoding='ISO-8859-1', low_memory=False)



# keep only rows where the CompensationAmount is not missing

df = df[df['CompensationAmount'].notnull()]

# convert to str and replace the commas

df['CompensationAmountClean'] = df['CompensationAmount'].str.replace(',', '')

df = df.loc[~df['CompensationAmountClean'].str.contains('-')]

df['CompensationAmountClean'] = df['CompensationAmountClean'].astype(float)



# load the conversion rates

rates = pd.read_csv('../input/conversionRates.csv')

df_merged = pd.merge(left=df, right=rates, left_on='CompensationCurrency', right_on='originCountry')

df_merged['CompensationAmountUSD'] = df_merged['CompensationAmountClean'] * df_merged['exchangeRate']



# keep the columns we need

df_clean = df_merged[['CompensationAmountUSD', 'GenderSelect', 'MajorSelect',

                     'CurrentJobTitleSelect', 'FormalEducation', 'Country']]

df_clean.head(10)
# find the values for male - remove the compensation outliers

male_values = df_clean.loc[

    (df_clean['CompensationAmountUSD'] < 2000000) &

    (df_clean['CompensationAmountUSD'] > 1) &

    (df_clean['GenderSelect'] == 'Male')

]['CompensationAmountUSD']



# find the values for female - remove the compensation outliers

female_values = df_clean.loc[

    (df_clean['CompensationAmountUSD'] < 2000000) &

    (df_clean['CompensationAmountUSD'] > 1) &

    (df_clean['GenderSelect'] == 'Female')

]['CompensationAmountUSD']



# define the figure

fig = plt.figure(figsize=(12, 7))

ax = fig.add_subplot(111)

# plot both histograms together

plt.hist(male_values, bins=12, alpha=0.7, label='Male', normed=True, color='blue')

plt.hist(female_values, bins=12, alpha=0.7, label='Female', normed=True, color='grey')

# legend

plt.legend(loc='upper right', fontsize=14)

# title

plt.title('Compensation in USD for Male and Female Data Scientists', fontsize=14, alpha=0.8)

# remove figure border

ax.spines["top"].set_visible(False)

ax.spines["bottom"].set_visible(False)

ax.spines["right"].set_visible(False)

ax.spines["left"].set_visible(False)

# configure xticks & yticks

plt.yticks(fontsize=12, alpha=0.8)

plt.xticks(fontsize=14, alpha=0.8)

# configure ylabel

plt.xlabel('USD', fontsize=20, alpha=0.8)

plt.show()
groups = df_clean.groupby(['FormalEducation'])



fig = plt.figure(figsize=(16, 9))

ax = fig.add_subplot(111)



sorted_medians = groups['CompensationAmountUSD'].median().sort_values()

ax = sorted_medians.plot(kind='barh', color=(117/255., 148/255., 205/255.))

# xaxis and yaxis conf

ax.xaxis.tick_top()

ax.set_xlabel('USD', fontsize=14)

ax.xaxis.set_label_position('top')

ax.yaxis.label.set_visible(False)

# configure ticks

plt.tick_params(

        axis='both',  # changes apply to the x-axis and y-axis

        which='both',  # both major and minor ticks are affected

        bottom='off',  # ticks along the bottom edge are off

        top='on',  # ticks along the top edge are off

        right='off',

        left='off'

    )

# configure xticks & yticks

plt.yticks(fontsize=14, alpha=0.8)

plt.xticks(fontsize=14, alpha=0.8)

# remove border figure

ax.spines["top"].set_visible(False)

ax.spines["bottom"].set_visible(False)

ax.spines["right"].set_visible(False)

ax.spines["left"].set_visible(False)

plt.xticks(alpha=0.8)

plt.yticks(alpha=0.8)

plt.title('Median Compensation in USD by Formal Education', fontsize=16, alpha=0.8, y=1.13, x=0.2)

# source

plt.text(60000, 0.01,

             'Source: Kaggle ML and Data Science Survey, 2017',

             fontsize=11,

             style='italic',

             alpha=0.7)

plt.tight_layout()

plt.show()
groups = df_clean.groupby(['CurrentJobTitleSelect'])



fig = plt.figure(figsize=(16, 9))

ax = fig.add_subplot(111)



sorted_medians = groups['CompensationAmountUSD'].median().sort_values()

ax = sorted_medians.plot(kind='barh', color=(117/255., 148/255., 205/255.))

# xaxis and yaxis conf

ax.xaxis.tick_top()

ax.set_xlabel('USD', fontsize=14)

ax.xaxis.set_label_position('top')

ax.yaxis.label.set_visible(False)

# configure ticks

plt.tick_params(

        axis='both',  # changes apply to the x-axis and y-axis

        which='both',  # both major and minor ticks are affected

        bottom='off',  # ticks along the bottom edge are off

        top='on',  # ticks along the top edge are off

        right='off',

        left='off'

    )

# configure xticks & yticks

plt.yticks(fontsize=14, alpha=0.8)

plt.xticks(fontsize=14, alpha=0.8)

# remove border figure

ax.spines["top"].set_visible(False)

ax.spines["bottom"].set_visible(False)

ax.spines["right"].set_visible(False)

ax.spines["left"].set_visible(False)

plt.xticks(alpha=0.8)

plt.yticks(alpha=0.8)

plt.title('Median Compensation in USD by Job Title', fontsize=16, alpha=0.8, y=1.13, x=0.2)

# source

plt.text(60000, 0.01,

             'Source: Kaggle ML and Data Science Survey, 2017',

             fontsize=11,

             style='italic',

             alpha=0.7)

plt.tight_layout()

plt.show()
groups = df_clean.groupby(['MajorSelect'])



fig = plt.figure(figsize=(16, 9))

ax = fig.add_subplot(111)



sorted_medians = groups['CompensationAmountUSD'].median().sort_values()

ax = sorted_medians.plot(kind='barh', color=(117/255., 148/255., 205/255.))

# xaxis and yaxis conf

ax.xaxis.tick_top()

ax.set_xlabel('USD', fontsize=14)

ax.xaxis.set_label_position('top')

ax.yaxis.label.set_visible(False)

# configure ticks

plt.tick_params(

        axis='both',  # changes apply to the x-axis and y-axis

        which='both',  # both major and minor ticks are affected

        bottom='off',  # ticks along the bottom edge are off

        top='on',  # ticks along the top edge are off

        right='off',

        left='off'

    )

# configure xticks & yticks

plt.yticks(fontsize=14, alpha=0.8)

plt.xticks(fontsize=14, alpha=0.8)

# remove border figure

ax.spines["top"].set_visible(False)

ax.spines["bottom"].set_visible(False)

ax.spines["right"].set_visible(False)

ax.spines["left"].set_visible(False)

plt.xticks(alpha=0.8)

plt.yticks(alpha=0.8)

plt.title('Median Compensation in USD by Undergraduate Major', fontsize=16, alpha=0.8, y=1.13, x=0.2)

# source

plt.text(60000, 0.01,

             'Source: Kaggle ML and Data Science Survey, 2017',

             fontsize=11,

             style='italic',

             alpha=0.7)

plt.tight_layout()

plt.show()
# keep only the ten countries with most respondents

keep_countries = ['United States', 'People \'s Republic of China', 'United Kingdom', 'Russia', 'India', 'Brazil', 'Germany', 'France', 'Canada']



df_clean = df_clean.loc[df_clean['Country'].isin(keep_countries)]

groups = df_clean.groupby(['MajorSelect', 'Country'])

sorted_medians = groups['CompensationAmountUSD'].median().sort_values()

major_df = pd.DataFrame(sorted_medians).reset_index()

major_df.head(10)

#for name, group in groups:

    #print(name)

    #print(group)
import plotly.offline as py

from plotly.offline import init_notebook_mode

import plotly.graph_objs as go

import plotly.figure_factory as ff

init_notebook_mode(connected=True)



trace = go.Heatmap(z=major_df['CompensationAmountUSD'],

                   x=major_df['Country'],

                   y=major_df['MajorSelect'],

                   colorscale=[[0.0, 'rgb(94,79,162)'], [0.09999999999998, 'rgb(50,136,189)'], 

                               [0.19999999999999996, 'rgb(102,194,165)'], [0.30000000000000004, 'rgb(171,221,164)'], 

                               [0.4, 'rgb(230,245,152)'], [0.5, 'rgb(255,255,191)'], 

                               [0.6, 'rgb(254,224,139)'], [0.7, 'rgb(253,174,97)'], 

                               [0.8, 'rgb(244,109,67)'], [0.9, 'rgb(213,62,79)'],

                               [1.0, 'rgb(158,1,66)']

                              ])

figure = dict(data=[trace], layout= dict( margin = dict(t=80,r=80,b=100,l=385)))

py.iplot(figure, filename='labelled-heatmap')