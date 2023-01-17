#Load necessary packages
import pandas as pd
import numpy as np

import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls
import warnings
warnings.filterwarnings('ignore')

import networkx as nx
import matplotlib.pyplot as plt
%matplotlib inline

from datetime import datetime
yelp_business = pd.read_csv('../input/yelp_business.csv')
yelp_review = pd.read_csv('../input/yelp_review.csv')
yelp_users = pd.read_csv('../input/yelp_user.csv')
def adding_column_suffix(df,suffix_value):
    new_names = [(i,i+suffix_value) for i in df.columns.values]
    df.rename(columns = dict(new_names), inplace=True)
    return df
yelp_business = adding_column_suffix(yelp_business,'_business')
yelp_review = adding_column_suffix(yelp_review,'_review')
yelp_users = adding_column_suffix(yelp_users,'_users')
yelp_business.columns.values
yelp_review.columns.values
yelp_users.columns.values
print ('Total Business categories available on Yelp:') , yelp_business.categories_business.nunique()
#Create another MajorCategory column to identify businesses related to Bars
yelp_business['MajorCategory'] = np.where(yelp_business['categories_business'].str.contains(u'Bar|Bars'), 'Bars', 'Other')
yelp_business = yelp_business.loc[yelp_business['MajorCategory'] == 'Bars']
print ('Total Business categories available on Yelp: ') , yelp_business.categories_business.nunique()
#Get Review Counts per state
df = yelp_business.groupby(['state_business'])['review_count_business'].sum().reset_index(name='count')
df[['count']].sum()
graph_title = "Yelp Reviews Available: %dM" % (df[['count']].sum()[0]/1000000)
data = [ dict(
        type = 'choropleth',
        locations = df['state_business'],
        z = (df['count']/ df['count'].sum()) * 100,
        text = df['state_business'],locationmode = 'USA-states',
        autocolorscale =  True,
        reversescale = False,
        marker = dict(line = dict (color = 'rgb(60,60,60)',width = 0.5) ),
        colorbar = dict(autotick = False, tickprefix = '%: ', title = graph_title),
      ) ]

layout = dict(
    title = 'Yelp Review Count by State (Business Category-Bars)',
    geo = dict(scope='usa',showframe = True, showcoastlines = True, projection = dict(type = 'Mercator')))

fig = dict( data=data, layout=layout )
py.iplot( fig, validate=False)
yelp_business = yelp_business[yelp_business['state_business'].str.contains("SC")==True]
print ( 'Bars reviewed in South Carolina: ') , yelp_business.business_id_business.count()  #delete this cell later
#Convert date to datetime object
yelp_review['date_review'] = pd.to_datetime(yelp_review['date_review'],format='%Y-%m-%d')

yelp_review.sort_values(by='date_review',ascending=False)
yelp_review = yelp_review.drop_duplicates('user_id_review') # We just need one user record to map it to business id.
yelp_review.drop(['useful_review', 'funny_review', 'cool_review'], axis = 1, inplace = True)  # drop inidivdual review ratings
yelp_review.info()
business_reviews = pd.merge(yelp_business, yelp_review, left_on='business_id_business',right_on='business_id_review', how='left')
print ('Bars reviewed in South Carolina: '), business_reviews.user_id_review.count()
#Join the business reviews and users to club the business titles and get our interest group of users
user_business_review = pd.merge( yelp_users, business_reviews, left_on='user_id_users',right_on='user_id_review', how='right')
#Since we are interested in a particular cluster of users, so lets get the relevant columns
user_business_review = user_business_review[['business_id_business','state_business','user_id_users','friends_users','yelping_since_users','average_stars_users','stars_business','stars_review','MajorCategory','name_business','name_users']]
user_business_review.columns
#Drop any unknown users and/or users with 0 friends. For now we ll ignore missing data and just focus on yelp users with yelp friends.
user_business_review = user_business_review.dropna()

#Select only people who have friends
user_business_review = user_business_review[user_business_review.friends_users != 'None']
#Working with all data points for finding connections is overwhelming the system right now while attempting to generate a big graph, so lets just work with a smaller subset
fulluser_set = user_business_review
user_set = user_business_review.head(10)
G = nx.Graph()
G.add_nodes_from(user_set['user_id_users']) 
G.nodes()
#function to add edges
def generate_edges(user_set):
    x = list(user_set.user_id_users)
    y = user_set.friends_users.tolist()

    users_dict = {}
    for i in range(len(x)):
        users_dict[x[i]] = y[i].split(',')
    
    edges = []
    for node in users_dict:
        for neighbour in users_dict[node]:
            edges.append((node, neighbour))

    return edges
G.add_edges_from(generate_edges(user_set))
nx.draw(G, pos=nx.spring_layout(G,k=.15),node_color='c',edge_color='k',node_size=20, width=.7)
plt.show()
#Lets look at a bigger sample 
fulluser_set = fulluser_set.head(50)
G = nx.Graph()
G.add_nodes_from(fulluser_set['user_id_users']) 

G.add_edges_from(generate_edges(fulluser_set))
centralScore = nx.betweenness_centrality(G)
#Store the scores in a dataframe
user_score = pd.DataFrame(
    {'user_id': list(centralScore.keys()),
     'centralityScore': list(centralScore.values())
    })
#Lets get our yelp social users sample, if centrality is greater than 0
central_users = user_score[user_score['centralityScore'] > 0 ]
central_users.info()
centralUserReviews = pd.merge( fulluser_set, central_users, left_on='user_id_users',right_on='user_id', how='inner') #add business details to scored users
centralUserReviews  = centralUserReviews.sort_values(by='centralityScore', ascending=False)#add the ratings data
#Look at the average ratings of the central users vs business ratings

trace0 = go.Bar(
    x=centralUserReviews['name_business'],
    y=centralUserReviews['average_stars_users'],
    name = 'central_user_rating')

trace1 = go.Bar(
    x=centralUserReviews['name_business'],
    y=centralUserReviews['stars_business'],
    name = 'average business rating'
)

data = [trace0,trace1]
layout = go.Layout(
    title='Central-Users Ratings Vs Average Business Ratings',
)

fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename='color-bar')
trace0 = go.Bar(
    x=centralUserReviews['name_business'],
    y=centralUserReviews['average_stars_users'],
    name = 'central_user_rating')

trace1 = go.Bar(
    x=centralUserReviews['name_business'],
    y=centralUserReviews['stars_business'],
    name = 'average business rating'
)

trace2 = go.Bar(
    x=centralUserReviews['name_business'],
    y=centralUserReviews['stars_review'],
    name = 'individual user business specific rating'
)

data = [trace0,trace1, trace2]
layout = go.Layout(
    title='Central-Users Individual & Avg Ratings Vs Avg Business Ratings',
)

fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename='color-bar')