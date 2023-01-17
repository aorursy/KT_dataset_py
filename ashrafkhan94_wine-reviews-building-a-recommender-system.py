import warnings

warnings.filterwarnings('ignore')



import numpy as np

import pandas as pd

import re



import matplotlib.pyplot as plt

import seaborn as sns



import plotly

plotly.offline.init_notebook_mode()

import plotly.graph_objs as go

import plotly.express as px



import plotly.figure_factory as ff

import cufflinks as cf



from scipy.stats import kurtosis, skew

from scipy import stats



%matplotlib inline

sns.set_style("whitegrid")

sns.set_context("paper")

plt.style.use('seaborn')
df = pd.read_csv('../input/wine-reviews/winemag-data-130k-v2.csv', index_col=[0])
class color:

   PURPLE = '\033[95m'

   CYAN = '\033[96m'

   DARKCYAN = '\033[36m'

   BLUE = '\033[94m'

   GREEN = '\033[92m'

   YELLOW = '\033[93m'

   RED = '\033[91m'

   BOLD = '\033[1m'

   UNDERLINE = '\033[4m'

   END = '\033[0m'



def DataDesc(df):

    print(f"Dataset Shape: {df.shape}")

    summary = pd.DataFrame(df.dtypes,columns=['dtypes'])

    summary = summary.reset_index()

    summary['Name'] = summary['index']

    summary = summary[['Name','dtypes']]

    summary['Missing'] = df.isnull().sum().values    

    summary['Uniques'] = df.nunique().values

    summary['First Value'] = df.loc[0].values

    summary['Second Value'] = df.loc[1].values

    summary['Third Value'] = df.loc[2].values



    for name in summary['Name'].value_counts().index:

        summary.loc[summary['Name'] == name, 'Entropy'] = round(stats.entropy(df[name].value_counts(normalize=True), base=2),2) 



    return summary



def CalOutliers(df_num): 

    '''

    

    Leonardo Ferreira 20/10/2018

    Set a numerical value and it will calculate the upper, lower and total number of outliers

    It will print a lot of statistics of the numerical feature that you set on input

    

    '''

    # calculating mean and std of the array

    data_mean, data_std = np.mean(df_num), np.std(df_num)



    # seting the cut line to both higher and lower values

    # You can change this value

    cut = data_std * 3



    #Calculating the higher and lower cut values

    lower, upper = data_mean - cut, data_mean + cut



    # creating an array of lower, higher and total outlier values 

    outliers_lower = [x for x in df_num if x < lower]

    outliers_higher = [x for x in df_num if x > upper]

    outliers_total = [x for x in df_num if x < lower or x > upper]



    # array without outlier values

    outliers_removed = [x for x in df_num if x > lower and x < upper]

    

    print(color.BOLD+f'Lower outliers: {len(outliers_lower)}'+ color.END) # printing total number of values in lower cut of outliers

    print(color.BOLD+f'Upper outliers: {len(outliers_higher)}'+ color.END) # printing total number of values in higher cut of outliers

    print(color.BOLD+f'Total outliers: {len(outliers_total)}'+ color.END) # printing total number of values outliers of both sides

    print(color.BOLD+f'Non - outliers: {len(outliers_removed)}'+ color.END) # printing total number of non outlier values

    print(color.BOLD+f'% of Outliers : {round((len(outliers_total) / len(outliers_removed) )*100, 4)}'+ color.END ) # Percentual of outliers in points
DataDesc(df)
display(df['points'].describe())



fig = plt.figure(figsize=(20,5))

plt.suptitle('Points Distribution', fontsize=30)

ax1 = fig.add_subplot(121)

_ = sns.countplot(data=df, x='points', color='#963559', ax=ax1)

#_ = ax1.set_title('Points Distribution', fontsize=30)

_ = ax1.set_ylabel('Count', fontsize=20)

_ = ax1.set_xlabel('Points', fontsize=20)





ax2 = fig.add_subplot(122)

_ = plt.scatter(range(df.shape[0]), np.sort(df.points.values), color='#38A585')

#_ = ax1.set_title('Points Distribution', fontsize=30)

_ = ax2.set_ylabel('Points', fontsize=20)

_ = ax2.set_xlabel('Total', fontsize=20)



print("\n")

display(CalOutliers(df['points']))
display(df['price'].describe())



fig = plt.figure(figsize=(20,5))

plt.suptitle('Price Distribution', fontsize=30)

ax1 = fig.add_subplot(121)

_ = sns.distplot(np.log(df['price'].dropna()), color='#963559', ax=ax1)

#_ = ax1.set_title('Points Distribution', fontsize=30)

_ = ax1.set_ylabel('Frequency Log', fontsize=20)

_ = ax1.set_xlabel('Price(log)', fontsize=20)





ax2 = fig.add_subplot(122)

_ = plt.scatter(range(df.shape[0]), np.sort(df.price.values), color='#38A585')

#_ = ax1.set_title('Points Distribution', fontsize=30)

_ = ax2.set_ylabel('Price', fontsize=20)

_ = ax2.set_xlabel('Total', fontsize=20)



print("\n")

display(CalOutliers(df['price']))
df['price_log'] = np.log(df['price'])

_ = sns.jointplot(data=df, x='price_log', y='points', color='#963559')
country_group = df.groupby('country').size().rename('Wines').reset_index()

fig = px.pie(country_group, 

             values='Wines', names='country', 

             color_discrete_sequence=px.colors.sequential.RdBu,

            title='Country wise distribution of Wine Samples',

            width=800,

            height=500)



fig.update_layout(

    margin=dict(l=25, r=20, t=30, b=50),

    paper_bgcolor="#ECEFF9",

)

fig.show()
groups = df.groupby('country').filter(lambda x: len(x) >= 100).reset_index()



print('Average points = ', np.nanmean(list(groups.points)))



layout = {'title' : 'Wine Ratings across major Countries',

        'xaxis': {'tickangle':-40},

        'yaxis': {'title': 'Points'},

        'margin': dict(l=25, r=10, t=30, b=10),

          'width' : 900,

          'height' : 350,

          'plot_bgcolor': '#ECEFF9',

          'shapes': [{

              'type' :'line',

              'x0': -.5,

              'y0': np.nanmean(list(groups.points)),

              'x1': 19,

              'y1': np.nanmean(list(groups.points)),

              'line': { 'dash': 'dashdot'},

              'line_color': '#38A585'

          }]

          }



data = [{

    'y': df[df.country==country]['points'], 

    'type':'violin',

    'name' : country,

    'showlegend':False,

    'fillcolor' : '#963559',

    'line_color': '#963559'

    #'marker': {'color': 'Set2'}

    } for i,country in enumerate(list(set(groups.country)))]





plotly.offline.iplot({'data': data, 'layout': layout})
groups = df.groupby('country').filter(lambda x: len(x) >= 100).reset_index()



print('Average Price = ', np.nanmean(list(groups.price)))



layout = {'title' : 'Wine Prices across major Countries',

        'xaxis': {'tickangle':-40},

        'yaxis': {'title': 'Price'},

        'margin': dict(l=25, r=10, t=30, b=10),

          'width' : 900,

          'height' : 350,

          'plot_bgcolor': '#ECEFF9',

          'shapes': [{

              'type' :'line',

              'x0': -.5,

              'y0': np.nanmean(list(groups.price)),

              'x1': 19,

              'y1': np.nanmean(list(groups.price)),

              'line': { 'dash': 'dashdot'},

              'line_color' : '#38A585'

          }]

          }



data = [{

    'y': df[df.country==country]['price'], 

    'type':'violin',

    'name' : country,

    'showlegend':False,

    'fillcolor' : '#963559',

    'line_color': '#963559'

    #'marker': {'color': 'Set2'}

    } for i,country in enumerate(list(set(groups.country)))]







plotly.offline.iplot({'data': data, 'layout': layout})
cnt = df.groupby(['country','points'])['price'].agg(['count','min','max','mean']).sort_values(by='mean',ascending=False)[:10]

cnt.reset_index(inplace=True)

cnt.style.background_gradient(cmap='PuBuGn',high=0.5)
variety_group = df.groupby('variety').size().rename('Wines').reset_index()



famous_variety_group = variety_group.query('Wines > 1500').sort_values(by='Wines', ascending=False)



fig  = px.bar(data_frame=famous_variety_group, x='variety', y='Wines', color_discrete_sequence=px.colors.sequential.RdBu)

fig.update_layout(width=900, height=350, title= {'text': "Famous Varities of Wine samples",

                                                'y':0.95,'x':0.5,

                                                'xanchor': 'center','yanchor': 'top'},

                 margin = dict(l=25, r=10, t=35, b=10))
groups = df.groupby('variety').filter(lambda x: len(x) >= 1500).reset_index()



print('Average points = ', np.nanmean(list(groups.points)))



layout = {'title' : 'Wine Ratings of Famous Varieties',

        'xaxis': {'tickangle':-40},

        'yaxis': {'title': 'Points'},

        'margin': dict(l=25, r=10, t=30, b=10),

          'width' : 900,

          'height' : 350,

          'plot_bgcolor': '#ECEFF9',

          'shapes': [{

              'type' :'line',

              'x0': -.5,

              'y0': np.nanmean(list(groups.points)),

              'x1': 19,

              'y1': np.nanmean(list(groups.points)),

              'line': { 'dash': 'dashdot'},

              'line_color': '#38A585'

          }]

          }



data = [{

    'y': df[df.variety==variety]['points'], 

    'type':'violin',

    'name' : variety,

    'showlegend':False,

    'fillcolor' : '#963559',

    'line_color': '#963559'

    #'marker': {'color': 'Set2'}

    } for i,variety in enumerate(list(set(groups.variety)))]





plotly.offline.iplot({'data': data, 'layout': layout})
groups = df.groupby('variety').filter(lambda x: len(x) >= 1500).reset_index()



print('Average price = ', np.nanmean(list(groups.points)))



layout = {'title' : 'Wine Prices from Famous Varieties',

        'xaxis': {'tickangle':-40},

        'yaxis': {'title': 'Prices'},

        'margin': dict(l=25, r=10, t=30, b=10),

          'width' : 900,

          'height' : 350,

          'plot_bgcolor': '#ECEFF9',

          'shapes': [{

              'type' :'line',

              'x0': -.5,

              'y0': np.nanmean(list(groups.price)),

              'x1': 19,

              'y1': np.nanmean(list(groups.price)),

              'line': { 'dash': 'dashdot'},

              'line_color': '#38A585'

          }]

          }



data = [{

    'y': df[df.variety==variety]['price'], 

    'type':'violin',

    'name' : variety,

    'showlegend':False,

    'fillcolor' : '#963559',

    'line_color': '#963559'

    #'marker': {'color': 'Set2'}

    } for i,variety in enumerate(list(set(groups.variety)))]





plotly.offline.iplot({'data': data, 'layout': layout})
taster_group = df.groupby('taster_name').size().rename('Wines').reset_index()



famous_taster_group = taster_group.query('Wines > 800').sort_values(by='Wines', ascending=False)



fig  = px.bar(data_frame=famous_taster_group, x='taster_name', y='Wines', color_discrete_sequence=px.colors.sequential.RdBu)

fig.update_layout(width=900, height=350, title= {'text': "Famous Tasters",

                                                'y':0.95,'x':0.5,

                                                'xanchor': 'center','yanchor': 'top'},

                 margin = dict(l=25, r=10, t=35, b=10))
groups = df.groupby('taster_name').filter(lambda x: len(x) >= 800).reset_index()



print('Average points = ', np.nanmean(list(groups.points)))



layout = {'title' : 'Wine Ratings by Famous Tasters',

        'xaxis': {'tickangle':-40},

        'yaxis': {'title': 'Points'},

        'margin': dict(l=25, r=10, t=30, b=10),

          'width' : 900,

          'height' : 350,

          'plot_bgcolor': '#ECEFF9',

          'shapes': [{

              'type' :'line',

              'x0': -.5,

              'y0': np.nanmean(list(groups.points)),

              'x1': 19,

              'y1': np.nanmean(list(groups.points)),

              'line': { 'dash': 'dashdot'},

              'line_color': '#38A585'

          }]

          }



data = [{

    'y': df[df.taster_name==taster]['points'], 

    'type':'violin',

    'name' : taster,

    'showlegend':False,

    'fillcolor' : '#963559',

    'line_color': '#963559'

    #'marker': {'color': 'Set2'}

    } for i,taster in enumerate(list(set(groups.taster_name)))]





plotly.offline.iplot({'data': data, 'layout': layout})
groups = df.groupby('taster_name').filter(lambda x: len(x) >= 800).reset_index()



print('Average price = ', np.nanmean(list(groups.price)))



layout = {'title' : 'Wine Prices by Famous Tasters',

        'xaxis': {'tickangle':-40},

        'yaxis': {'title': 'Prices'},

        'margin': dict(l=25, r=10, t=30, b=10),

          'width' : 900,

          'height' : 350,

          'plot_bgcolor': '#ECEFF9',

          'shapes': [{

              'type' :'line',

              'x0': -.5,

              'y0': np.nanmean(list(groups.price)),

              'x1': 19,

              'y1': np.nanmean(list(groups.price)),

              'line': { 'dash': 'dashdot'},

              'line_color': '#38A585'

          }]

          }



data = [{

    'y': df[df.taster_name==taster]['price'], 

    'type':'violin',

    'name' : taster,

    'showlegend':False,

    'fillcolor' : '#963559',

    'line_color': '#963559'

    #'marker': {'color': 'Set2'}

    } for i,taster in enumerate(list(set(groups.taster_name)))]





plotly.offline.iplot({'data': data, 'layout': layout})
from wordcloud import WordCloud, STOPWORDS



stopwords = set(STOPWORDS)



newStopWords = ['fruit', "Drink", "black", 'wine', 'drink', 'flavor']



stopwords.update(newStopWords)



wordcloud = WordCloud(

    background_color='white',

    stopwords=stopwords,

    max_words=300,

    max_font_size=200, 

    width=1000, height=800,

    random_state=42,

).generate(" ".join(df['description'].astype(str)))



print(wordcloud)

fig = plt.figure(figsize = (12,14))

plt.imshow(wordcloud)

plt.title("WORD CLOUD - DESCRIPTION",fontsize=25)

plt.axis('off')

plt.show()
wordcloud = WordCloud(

    background_color='white',

    stopwords=stopwords,

    max_words=300,

    max_font_size=200, 

    width=1000, height=800,

    random_state=42,

).generate(" ".join(df['title'].astype(str)))



print(wordcloud)

fig = plt.figure(figsize = (12,14))

plt.imshow(wordcloud)

plt.title("WORD CLOUD - Title",fontsize=25)

plt.axis('off')

plt.show()


from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

import matplotlib.gridspec as gridspec





# Top 10 countries based on number of wine samples

country_list = df.groupby('country').size().rename('Count').reset_index().sort_values(by='Count', ascending=False)[:10]['country']



grid = gridspec.GridSpec(5, 2)

plt.figure(figsize=(16,7*4))



for n, cat in enumerate(country_list):

    

    ax = plt.subplot(grid[n])   



    vectorizer = TfidfVectorizer(ngram_range = (2, 3), min_df=5, 

                                 stop_words='english',

                                 max_df=.5) 

    

    X2 = vectorizer.fit_transform(df.loc[(df.country == cat)]['description']) 

    features = (vectorizer.get_feature_names()) 

    scores = (X2.toarray()) 

    

    # Getting top ranking features 

    sums = X2.sum(axis = 0) 

    data1 = [] 

    

    for col, term in enumerate(features): 

        data1.append( (term, sums[0,col] )) 



    ranking = pd.DataFrame(data1, columns = ['term','rank']) 

    words = (ranking.sort_values('rank', ascending = False))[:15]

    

    sns.barplot(x='term', y='rank', data=words, ax=ax, 

                color='#38A585', orient='v')

    ax.set_title(f"N-Grams for : {cat}", fontsize=19)

    ax.set_xticklabels(ax.get_xticklabels(),rotation=90, color='#963559')

    ax.set_ylabel(' ')

    ax.set_xlabel(" ")



    

plt.suptitle("Top 15 N-Grams based on Wine's Description", fontsize=23)

plt.subplots_adjust(top = 0.95, hspace=.9, wspace=.1)



plt.show()
from sklearn.neighbors import NearestNeighbors

from scipy.sparse import csr_matrix

from sklearn.decomposition import TruncatedSVD



data_recommend = df[['province','variety','points']]

data_recommend.dropna(axis=0, inplace=True)

data_recommend.drop_duplicates(['province','variety'], inplace=True)



data_pivot = data_recommend.pivot(index= 'variety',columns='province',values='points').fillna(0)

data_matrix = csr_matrix(data_pivot)
knn = NearestNeighbors(n_neighbors=10, algorithm= 'brute', metric= 'cosine')

model_knn = knn.fit(data_matrix)
for n in range(5):

    query_index = np.random.choice(data_pivot.shape[0])

    #print(n, query_index)

    distance, indice = model_knn.kneighbors(data_pivot.iloc[query_index].values.reshape(1,-1), n_neighbors=6)

    for i in range(0, len(distance.flatten())):

        if  i == 0:

            print('Recmmendation for ## {0} ##:'.format(data_pivot.index[query_index]))

        else:

            print('{0}: {1} with distance: {2}'.format(i,data_pivot.index[indice.flatten()[i]],distance.flatten()[i]))

    print('\n')