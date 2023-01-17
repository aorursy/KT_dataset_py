import pandas as pd ## To work with data.

import numpy as np ## Linear Algebra.

### I am using plotly for data visualization. It has interactive and cool figgures.

import plotly.express as px

import plotly.graph_objects as go

from plotly.offline import init_notebook_mode, iplot ## For offline mode.

from plotly.subplots import make_subplots

from wordcloud import WordCloud, STOPWORDS ## For the wordcloud.

import matplotlib.pyplot as plt ##Again, visualizatoin.
## Reading the data

df = pd.read_csv("../input/grammy-awards/the_grammy_awards.csv")
df.head() ## Take a look into the data.
df1 = df[df['winner']==True] ## Dataset about the winners.
temp = df['winner'].value_counts().reset_index()

temp.loc[:,'index'] = ['Winner', 'Not Winner']

fig = go.Figure(data=[

    go.Pie(labels=temp['index'], values=temp['winner'])

])

fig.update_layout(title='Winners amongst nominees')

iplot(fig)
temp = df.groupby(by='year')['winner'].value_counts().unstack().reset_index().fillna(0)

temp.columns

fig = go.Figure(data=[

    go.Scatter(name='Winners', x=temp['year'], y=temp[True]),

    go.Scatter(name='Not Winners', x=temp['year'], y=temp[False])

])

fig.show()
temp1 = df['nominee'].value_counts()

temp2 = df1['nominee'].value_counts().reset_index()

temp2.columns=['Nominee', 'Wins']

temp2['Nominations'] = temp2['Nominee'].apply(lambda x : temp1.loc[x])

temp2 = temp2.astype('object')

temp2 = temp2.groupby(by=['Nominations', 'Wins']).agg('count').reset_index()



fig = px.scatter(temp2, 'Nominations', 'Wins', hover_data=['Nominee'], labels={'Nominee':'Count'})

fig.show()
temp1 = df['nominee'].value_counts()

temp2 = df1['nominee'].value_counts().reset_index()

temp2.columns=['Nominee', 'Wins']

temp2['Nominations'] = temp2['Nominee'].apply(lambda x : temp1.loc[x])

temp2 = temp2.astype('object')



temp1 = temp1.reset_index()

temp3 = temp1.nominee.value_counts().reset_index()

temp4 = temp2.Nominations.value_counts().reset_index()

temp3.sort_values(by='index', inplace=True)

temp4.sort_values(by='index', inplace=True)



fig = go.Figure(data=[

    go.Line(name='Nominations', x=temp3['index'], y=temp3['nominee']),

    go.Line(name='Wins', x=temp4['index'], y=temp4['Nominations'])

])

fig.update_layout(title='Wins and Nominations:')

iplot(fig)
temp = df.groupby(by='year').category.unique().reset_index()

temp.columns=['Year', 'Count']

temp.loc[:,'Count'] = temp.loc[:,'Count'].apply(lambda x : len(x))

temp.sort_values(by='Count', inplace=True)



fig = go.Figure(data=[

    go.Scatter(name='Least Award Categories', x=temp.head()['Year'], y=temp.head()['Count'], mode='markers'),

    go.Scatter(name='Most Award Categories', x=temp.tail()['Year'], y=temp.tail()['Count'], mode='markers')

])

fig.update_layout(title='Years with most and least Award Categories')

iplot(fig)
temp = df.groupby(by='category').year.unique().reset_index()

temp.columns=['Category', 'Count']

temp.loc[:,'Count'] = temp.loc[:,'Count'].apply(lambda x : len(x))

temp.sort_values(by='Count', inplace=True)



fig = go.Figure(data=[

    go.Scatter(name='Least Introduced Award Categories', x=temp.head()['Category'], y=temp.head()['Count'],

              mode='markers'),

    go.Scatter(name='Most Introduced Award Categories', x=temp.tail()['Category'], y=temp.tail()['Count'],

              mode='markers')

])

fig.update_layout(title='Categories with most awards.')

iplot(fig)
print('Number of categories that have only introduced once is:',len(temp[temp['Count']==1]))
temp = df['nominee'].value_counts().reset_index()

temp.columns = ['Nominee', 'Awards']

temp = temp.head(10)

fig = px.scatter(temp, x='Nominee', y='Awards', color='Awards', size='Awards')

fig.update_layout(title='Most Nominated :')

iplot(fig)
top_category = temp['Nominee'].tolist()



temp = df[df['nominee'].isin(top_category)].groupby(by=['nominee','year'])['year'].count().unstack().fillna(0)

temp = temp.T.cumsum().reset_index()

value_list = list(temp.columns)[1:]

temp = pd.melt(temp, id_vars='year', value_vars=value_list)



fig = px.scatter(temp, x='nominee', y='value', size='value', animation_frame='year',range_y=[0,10], color='nominee')

fig.update_layout(showlegend=False, title='Total Nomiations of top nominees throught the history of Grammy:')

iplot(fig)
temp = df1['nominee'].value_counts().reset_index()

temp.columns = ['nominee', 'Awards']

temp = temp.iloc[:10,:]

fig = px.scatter(temp, x='nominee', y='Awards', color='Awards', size='Awards')

fig.update_layout(title='Nominees with most awards:')

iplot(fig)
top_category = temp['nominee'].tolist()

temp = df1[df1['nominee'].isin(top_category)].groupby(by=['nominee','year'])['year'].count().unstack().fillna(0)

temp = temp.T.cumsum().reset_index()

value_list = list(temp.columns)[1:]



temp = pd.melt(temp, id_vars='year', value_vars=value_list)

fig = px.scatter(temp, x='nominee', y='value', size='value', animation_frame='year',range_y=[0,10], color='nominee')

fig.update_layout(showlegend=False, title='Total Awards of top nominees throught the history of Grammy:')

iplot(fig)
temp = df1.groupby(by=['nominee', 'year'])['year'].count().unstack().fillna(0).stack().sort_values(ascending=False)

temp = temp.reset_index().head()

temp.columns=['Nominee', 'Year', 'Awards']

print('Nominee with most awards in a single year: ')

temp
comment_words = ' '.join(df['workers'].dropna().tolist())



fig = WordCloud(width=400, height=400, stopwords=STOPWORDS, background_color='white',

         min_font_size=10).generate(comment_words)



plt.figure(figsize = (8, 8), facecolor = None) 

plt.imshow(fig) 

plt.axis("off") 

plt.tight_layout(pad = 0) 

plt.show() 


temp = pd.merge(temp1, temp2.drop(columns='Nominations'), how='outer',

                left_on='index', right_on='Nominee').drop(columns='Nominee')

temp.fillna(0, inplace=True)

temp_new = temp.astype('object')

temp_pivot = pd.pivot_table(temp_new, index='nominee', columns='Wins', aggfunc='count').fillna(0)

temp_pivot.columns = temp_pivot.columns.droplevel()



temp_pivot['Wins'] = temp_pivot.iloc[:,1:].sum(axis=1)

temp_pivot = temp_pivot.iloc[:,[0,8]]

temp_pivot['Total'] = temp_pivot.sum(axis=1)

temp_pivot['Lose_Per'] = temp_pivot.iloc[:,0]/temp_pivot['Total']

temp_pivot['Win_Per'] = temp_pivot.iloc[:,1]/temp_pivot['Total']

temp_pivot = temp_pivot.iloc[:,[3,4]]



temp_pivot.reset_index(inplace=True)
temp_pivot
temp_pivot.corr()