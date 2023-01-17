import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

sns.set(style = 'whitegrid')



import textwrap

from wordcloud import WordCloud, STOPWORDS 
data = pd.read_csv('../input/data-scientist-jobs/DataScientist.csv')

data
# Drop Columns

data.drop(['Unnamed: 0', 'Company Name', 'Headquarters','Competitors'], axis = 1, inplace = True)

data = data.set_index(['index'])

data.head()
# Change data with -1 values into NaN

data = data.replace([-1, -1.0, '-1'], np.nan)

print(data.isnull().sum(axis = 0))
# fill row data contain nan values in columns Easy Apply

data['Easy Apply'].fillna('FALSE', inplace = True)

# Drop row data contain nan values

data.dropna(axis = 0, inplace = True)
comment_words = ''

stopwords = set(STOPWORDS)



for val in data['Job Title']:

    val = str(val)

    tokens = val.split()

    

    for i in range(len(tokens)):

        tokens[i] = tokens[i].lower()

    

    comment_words += " ".join(tokens)+" "



wordcloud = WordCloud(width = 800, height = 400, background_color = 'white'

                      , stopwords = stopwords, min_font_size = 10).generate(comment_words)



fig, ax = plt.subplots(figsize = (16, 16))

ax.grid(False)

ax.imshow((wordcloud))

fig.tight_layout(pad=0)

plt.show()
data= data[data['Job Title'].str.contains('Data Scientist|Data Science')]

data
data['Salary Estimate']= data['Salary Estimate'].str.replace('(', '').str.replace(')', '').str.replace('Glassdoor est.', '').str.replace('Employer est.', '')
data['Min Salary'],data['Max Salary']=data['Salary Estimate'].str.split('-').str

data['Min Salary']=data['Min Salary'].str.strip(' ').str.strip('$').str.strip('K').fillna(0).astype(int)

data['Max Salary'] = data['Max Salary'].str.replace('Per Hour','')

data['Max Salary']=data['Max Salary'].str.strip(' ').str.strip('$').str.strip('K').fillna(0).astype(int)

fig, ax = plt.subplots(1,2, figsize = [16,4])



sns.distplot(ax = ax[0], a = data['Min Salary'])



sns.distplot( ax = ax[1], a = data['Max Salary'])

plt.show()
data['Revenue'].replace(['Unknown / Non-Applicable'], np.nan, inplace = True)

data[['Revenue']]
dataview = data.groupby('Location')['Job Title'].count().reset_index()

dataview = dataview.sort_values('Job Title', ascending = False).head(10)



fig, ax = plt.subplots(figsize = [16,5])

sns.barplot(data = dataview, x = 'Location', y = 'Job Title', ax = ax)

ax.set_ylabel('Count Jobs')

ax.set_yticks(np.arange(0, 200, step = 20))

for index,dataview in enumerate(dataview['Job Title'].astype(int)):

       ax.text(x=index-0.1 , y =dataview+1 , s=f"{dataview}" , fontdict=dict(fontsize=10))

plt.show()
dataview = data.pivot_table(index = 'Location', columns = 'Sector', values = 'Job Title', aggfunc = 'count')

dataview.fillna(0, inplace = True)

dataview['Total'] = dataview.sum(axis = 1)

dataview.reset_index(inplace = True)

dataview.sort_values('Total', ascending = False).head(10)
pd.melt(dataview, 

            id_vars=['Location'], 

            value_vars=list(dataview.columns[1:-1]), 

            var_name='Sector', 

            value_name='Sum of Value').sort_values('Sum of Value', ascending = False)
dataview1 = dataview.sort_values('Total', ascending = False).head(5)

max_width = 15

fig, ax = plt.subplots(5,1 , figsize = [18,20])

for i in range(0,5):

    dataview1 = dataview.sort_values('Total', ascending = False).head(5)

    dataview1 = dataview1[i:i+1]

    dataview1.dropna(axis = 1, inplace = True)

    dataview1 = pd.melt(dataview1, 

                id_vars=['Location'], 

                value_vars=list(dataview1.columns[1:-1]),

                var_name='Sector', 

                value_name='Sum of Value').sort_values('Sum of Value', ascending = False)

    

    dataview1 = dataview1.sort_values('Sum of Value', ascending = False).head(5)

    sns.barplot(ax=ax[i], data = dataview1, x ='Sector', y = 'Sum of Value')

    ax[i].set_title("Top 5 Sector Open Job Data Science in  '"+ dataview1.iloc[0,0]+"'", fontsize = 16)

    ax[i].set_xlabel('Sector', fontsize = 14)

    ax[i].set_ylabel('Count of Jobs', fontsize = 14)

    ax[i].set_xticklabels(textwrap.fill(x.get_text(), max_width) for x in ax[i].get_xticklabels())

    ax[i].set_yticks(np.arange(0, 85, step = 10))

    ax[i].tick_params(labelsize = 12)



    

    for index,dataview1 in enumerate(dataview1['Sum of Value'].astype(int)):

        ax[i].text(x=index-0.05 , y =dataview1+1 , s=f"{dataview1}" , fontdict=dict(fontsize=16))

    fig.subplots_adjust(wspace = 0.1, hspace = 1)

plt.show()
dataview = data.pivot_table(index = 'Location', columns = 'Industry', values = 'Job Title', aggfunc = 'count')

dataview.fillna(0, inplace = True)

dataview['Total'] = dataview.sum(axis = 1)

dataview.reset_index(inplace = True)

dataview.sort_values('Total', ascending = False).head(10)
pd.melt(dataview, 

            id_vars=['Location'], 

            value_vars=list(dataview.columns[1:-1]),

            var_name='Industry', 

            value_name='Sum of Value').sort_values('Sum of Value', ascending = False)
dataview1 = dataview.sort_values('Total', ascending = False).head(5)

max_width = 15

fig, ax = plt.subplots(5,1 , figsize = [18,20])

for i in range(0,5):

    dataview1 = dataview.sort_values('Total', ascending = False).head(5)

    dataview1 = dataview1[i:i+1]

    dataview1.dropna(axis = 1, inplace = True)

    dataview1 = pd.melt(dataview1, 

                id_vars=['Location'], 

                value_vars=list(dataview1.columns[1:-1]), 

                var_name='Industry', 

                value_name='Sum of Value').sort_values('Sum of Value', ascending = False)

    

    dataview1 = dataview1.sort_values('Sum of Value', ascending = False).head(5)

    

    sns.barplot(ax=ax[i], data = dataview1, x ='Industry', y = 'Sum of Value')

    ax[i].set_title("Top 5 Industry Open Job Data Science in  '"+ dataview1.iloc[0,0]+"'", fontsize = 16)

    ax[i].set_xlabel('Industry', fontsize = 14)

    ax[i].set_ylabel('Count of Jobs', fontsize = 14)

    ax[i].set_xticklabels(textwrap.fill(x.get_text(), max_width) for x in ax[i].get_xticklabels())

    ax[i].set_yticks(np.arange(0, 45, step = 5))

    ax[i].tick_params(labelsize = 12)

    

    for index,dataview1 in enumerate(dataview1['Sum of Value'].astype(int)):

        ax[i].text(x=index-0.05 , y =dataview1+1 , s=f"{dataview1}" , fontdict=dict(fontsize=16))

    fig.subplots_adjust(wspace = 0.1, hspace = 1)

plt.show()
dataview = data.pivot_table(index = 'Location', columns = 'Salary Estimate', values = 'Job Title', aggfunc = 'count')

#dataview.replace(0, inplace = True)

dataview['Total'] = dataview.sum(axis = 1)

dataview.reset_index(inplace = True)

dataview.sort_values('Total', ascending = False).head(10)
pd.melt(dataview, 

            id_vars=['Location'], 

            value_vars=list(dataview.columns[1:-1]),

            var_name='Salary Estimate', 

            value_name='Sum of Value').sort_values('Sum of Value', ascending = False)
dataview1 = dataview.sort_values('Total', ascending = False).head(5)
max_width = 15

fig, ax = plt.subplots(5,1 , figsize = [18,20])

for i in range(0,5):

    dataview1 = dataview.sort_values('Total', ascending = False).head(5)

    dataview1 = dataview1[i:i+1]

    dataview1.dropna(axis = 1, inplace = True)

    dataview1 = pd.melt(dataview1, 

                id_vars=['Location'], 

                value_vars=list(dataview1.columns[1:-1]), 

                var_name='Salary Estimate', 

                value_name='Sum of Value').sort_values('Sum of Value', ascending = False)

    

    dataview1 = dataview1.sort_values('Sum of Value', ascending = False).head(5)

    

    sns.barplot(ax=ax[i], data = dataview1, x ='Salary Estimate', y = 'Sum of Value')

    ax[i].set_title("Top 5 Salary Estimate Open Job Data Science in  '"+ dataview1.iloc[0,0]+"'", fontsize = 16)

    ax[i].set_xlabel('Salary Estimate', fontsize = 14)

    ax[i].set_ylabel('Count of Jobs', fontsize = 14)

    ax[i].set_xticklabels(textwrap.fill(x.get_text(), max_width) for x in ax[i].get_xticklabels())

    ax[i].set_yticks(np.arange(0, 35, step = 5))

    ax[i].tick_params(labelsize = 12)



    

    for index,dataview1 in enumerate(dataview1['Sum of Value'].astype(int)):

        ax[i].text(x=index-0.05 , y =dataview1+1 , s=f"{dataview1}" , fontdict=dict(fontsize=16))

    fig.subplots_adjust(wspace = 0.1, hspace = 1)

plt.show()
dataview_top=data.groupby('Location')[['Max Salary','Min Salary']].mean().sort_values(['Max Salary','Min Salary'],ascending=False).head(10)

dataview_top.reset_index(inplace = True)



dataview_bot=data.groupby('Location')[['Max Salary','Min Salary']].mean().sort_values(['Max Salary','Min Salary'],ascending=True).head(10)

dataview_bot.reset_index(inplace = True)



print(dataview_top, '\n')

print(dataview_bot)
max_width = 15

data_salary = [dataview_top, dataview_bot]

data_title = ['Top 10', 'Bottom 10']

fig, ax = plt.subplots(2,1, figsize = (24,14))

fig.subplots_adjust(hspace = 0.5)

for i in range(0,2):

    sns.barplot(ax = ax[i], data = data_salary[i], x = 'Location', y = 'Max Salary', color = 'orangered', label = 'Max Salary')

    sns.barplot(ax = ax[i], data = data_salary[i], x = 'Location', y = 'Min Salary', color = 'darkslateblue', label = 'Min Salary')

    ax[i].legend()

    ax[i].set_title(data_title[i]+' Average Salary in Each Location', fontsize = 20)

    ax[i].set_ylabel('Salary', fontsize = 20)

    ax[i].set_xlabel('Location', fontsize = 20)

    ax[i].set_xticklabels(textwrap.fill(x.get_text(), max_width) for x in ax[i].get_xticklabels())

    ax[i].set_yticks(np.arange(0, 300, step = 50))

    ax[i].tick_params(labelsize = 18)

    

plt.show()
dataview_top=data.groupby('Sector')[['Max Salary','Min Salary']].mean().sort_values(['Max Salary','Min Salary'],ascending=False).head(10)

dataview_top.reset_index(inplace = True)



dataview_bot=data.groupby('Sector')[['Max Salary','Min Salary']].mean().sort_values(['Max Salary','Min Salary'],ascending=True).head(10)

dataview_bot.reset_index(inplace = True)



print(dataview_top, '\n')

print(dataview_bot)
max_width = 15

data_salary = [dataview_top, dataview_bot]

data_title = ['Top 10', 'Bottom 10']

fig, ax = plt.subplots(2,1, figsize = (24,14))

fig.subplots_adjust(hspace = 0.5)

for i in range(0,2):

    sns.barplot(ax = ax[i], data = data_salary[i], x = 'Sector', y = 'Max Salary', color = 'orangered', label = 'Max Salary')

    sns.barplot(ax = ax[i], data = data_salary[i], x = 'Sector', y = 'Min Salary', color = 'darkslateblue', label = 'Min Salary')

    ax[i].legend()

    ax[i].set_title(data_title[i]+' Average Salary in Each Sector', fontsize = 20)

    ax[i].set_ylabel('Salary', fontsize = 20)

    ax[i].set_xlabel('Sector', fontsize = 20)

    ax[i].set_xticklabels(textwrap.fill(x.get_text(), max_width) for x in ax[i].get_xticklabels())

    ax[i].set_yticks(np.arange(0, 175, step = 25))

    ax[i].tick_params(labelsize = 18)

    

plt.show()
dataview_top=data.groupby('Industry')[['Max Salary','Min Salary']].mean().sort_values(['Max Salary','Min Salary'],ascending=False).head(10)

dataview_top.reset_index(inplace = True)



dataview_bot=data.groupby('Industry')[['Max Salary','Min Salary']].mean().sort_values(['Max Salary','Min Salary'],ascending=True).head(10)

dataview_bot.reset_index(inplace = True)



print(dataview_top, '\n')

print(dataview_bot)
max_width = 15

data_salary = [dataview_top, dataview_bot]

data_title = ['Top 10', 'Bottom 10']

fig, ax = plt.subplots(2,1, figsize = (24,14))

fig.subplots_adjust(hspace = 0.5)

for i in range(0,2):

    sns.barplot(ax = ax[i], data = data_salary[i], x = 'Industry', y = 'Max Salary', color = 'orangered', label = 'Max Salary')

    sns.barplot(ax = ax[i], data = data_salary[i], x = 'Industry', y = 'Min Salary', color = 'darkslateblue', label = 'Min Salary')

    ax[i].legend()

    ax[i].set_title(data_title[i]+' Average Salary in Each Industry', fontsize = 20)

    ax[i].set_ylabel('Salary', fontsize = 20)

    ax[i].set_xlabel('Industry', fontsize = 20)

    ax[i].set_xticklabels(textwrap.fill(x.get_text(), max_width) for x in ax[i].get_xticklabels())

    ax[i].set_yticks(np.arange(0, 200, step = 25))

    ax[i].tick_params(labelsize = 18)

    

plt.show()
dataview = data.groupby('Easy Apply')['Job Title'].count().reset_index()

dataview
fig, ax = plt.subplots()

ax =sns.barplot(ax = ax, data = dataview, x = 'Easy Apply', y = 'Job Title' )

ax.set_title('Easy Apply Data Science Job')

ax.set_ylabel('Counts of Jobs')

plt.show()
data['Revenue'].unique().tolist()
dataview = data.copy()

dataview['Revenue'].replace(['Unknown / Non-Applicable'], np.nan, inplace = True)

dataview['Revenue'].dropna(axis = 0, inplace = True)

dataview = dataview.groupby('Revenue')['Job Title'].count().reset_index()

dataview.sort_values('Job Title', ascending = False, inplace = True)

dataview
max_width = 15

fig, ax = plt.subplots(figsize = (16,4))

sns.barplot(ax = ax, data = dataview, x='Revenue', y = 'Job Title', palette = 'deep')

ax.set_title('Count Job Based Revenue')

ax.set_ylabel('Count Jobs')

ax.set_xticklabels(textwrap.fill(x.get_text(), max_width) for x in ax.get_xticklabels())

for index,dataview in enumerate(dataview['Job Title'].astype(int)):

        ax.text(x=index-0.1 , y =dataview+1 , s=f"{dataview}" , fontdict=dict(fontsize=12))

plt.show()
dataview = data.groupby('Rating')['Job Title'].count().reset_index()

dataview.sort_values('Job Title', ascending = False).head()
fig, ax = plt.subplots(figsize = (16, 8))

#sns.barplot(ax = ax, data = dataview, x = 'Rating', y = 'Job Title', order = dataview.sort_values('Job Title', ascending = False).Rating)

sns.barplot(ax = ax, data = dataview, x = 'Rating', y = 'Job Title', palette = 'deep')

ax.set_ylabel('Count Jobs')

for index,dataview in enumerate(dataview['Job Title'].astype(int)):

        ax.text(x=index-0.1 , y =dataview , s=f"{dataview}" , fontdict=dict(fontsize=10))

plt.show()
dataview_top = data.groupby('Location')['Rating'].mean().reset_index()

dataview_top = dataview_top.sort_values('Rating', ascending = False).head(10)



dataview_bot = data.groupby('Location')['Rating'].mean().reset_index()

dataview_bot = dataview_bot.sort_values('Rating', ascending = True).head(10)



print(dataview_top, '\n' )

print(dataview_bot)
max_width = 15

data_rating = [dataview_top, dataview_bot]

data_title = ['Top 10', 'Bottom 10']

fig, ax = plt.subplots(2,1, figsize = (24,14))

fig.subplots_adjust(hspace = 0.5)

for i in range(0,2):

    sns.barplot(ax = ax[i], data = data_rating[i], x = 'Location', y = 'Rating', color = 'orangered', label = 'Rating')

    ax[i].set_title(data_title[i]+' Average Rating Company in Each Location', fontsize = 20)

    ax[i].set_ylabel('Rating', fontsize = 20)

    ax[i].set_xlabel('Location', fontsize = 20)

    ax[i].set_xticklabels(textwrap.fill(x.get_text(), max_width) for x in ax[i].get_xticklabels())

    ax[i].set_yticks(np.arange(0, 5, step = 0.5))

    for index, data_rating[i] in enumerate(np.round(data_rating[i]['Rating'], 2)):

        ax[i].text(x=index-0.1 , y =data_rating[i] , s=f"{data_rating[i]}" , fontdict=dict(fontsize=16))

    ax[i].tick_params(labelsize = 18)

    

plt.show()
dataview_top = data.groupby('Sector')['Rating'].mean().reset_index()

dataview_top = dataview_top.sort_values('Rating', ascending = False).head(10)



dataview_bot = data.groupby('Sector')['Rating'].mean().reset_index()

dataview_bot = dataview_bot.sort_values('Rating', ascending = True).head(10)



print(dataview_top, '\n' )

print(dataview_bot)
max_width = 15

data_rating = [dataview_top, dataview_bot]

data_title = ['Top 10', 'Bottom 10']

fig, ax = plt.subplots(2,1, figsize = (24,14))

fig.subplots_adjust(hspace = 0.5)

for i in range(0,2):

    sns.barplot(ax = ax[i], data = data_rating[i], x = 'Sector', y = 'Rating', color = 'orangered', label = 'Rating')

    ax[i].set_title(data_title[i]+' Average Rating Company in Each Sector', fontsize = 20)

    ax[i].set_ylabel('Rating', fontsize = 20)

    ax[i].set_xlabel('Sector', fontsize = 20)

    ax[i].set_xticklabels(textwrap.fill(x.get_text(), max_width) for x in ax[i].get_xticklabels())

    ax[i].set_yticks(np.arange(0, 5, step = 0.5))

    for index, data_rating[i] in enumerate(np.round(data_rating[i]['Rating'], 2)):

        ax[i].text(x=index-0.1 , y =data_rating[i] , s=f"{data_rating[i]}" , fontdict=dict(fontsize=16))

    ax[i].tick_params(labelsize = 18)

    

plt.show()
dataview_top = data.groupby('Industry')['Rating'].mean().reset_index()

dataview_top = dataview_top.sort_values('Rating', ascending = False).head(10)



dataview_bot = data.groupby('Industry')['Rating'].mean().reset_index()

dataview_bot = dataview_bot.sort_values('Rating', ascending = True).head(10)



print(dataview_top, '\n' )

print(dataview_bot)
max_width = 15

data_rating = [dataview_top, dataview_bot]

data_title = ['Top 10', 'Bottom 10']

fig, ax = plt.subplots(2,1, figsize = (26,14))

fig.subplots_adjust(hspace = 0.5)

for i in range(0,2):

    sns.barplot(ax = ax[i], data = data_rating[i], x = 'Industry', y = 'Rating', color = 'orangered', label = 'Rating')

    ax[i].set_title(data_title[i]+' Average Rating Company in Each Industry', fontsize = 20)

    ax[i].set_ylabel('Rating', fontsize = 20)

    ax[i].set_xlabel('Industry', fontsize = 20)

    ax[i].set_xticklabels(textwrap.fill(x.get_text(), max_width) for x in ax[i].get_xticklabels())

    ax[i].set_yticks(np.arange(0, 5, step = 0.5))

    for index, data_rating[i] in enumerate(np.round(data_rating[i]['Rating'], 2)):

        ax[i].text(x=index-0.1 , y =data_rating[i] , s=f"{data_rating[i]}" , fontdict=dict(fontsize=16))

    ax[i].tick_params(labelsize = 18)

    

plt.show()