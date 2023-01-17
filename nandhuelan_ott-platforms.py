'''Import basic modules.'''

import pandas as pd

import numpy as np





'''Customize visualization

Seaborn and matplotlib visualization.'''

import matplotlib.pyplot as plt

import seaborn as sns

sns.set_style("whitegrid")

%matplotlib inline



'''Plotly visualization .'''

import plotly.offline as py

from plotly.offline import iplot, init_notebook_mode

import plotly.graph_objs as go

py.init_notebook_mode(connected = True) # Required to use plotly offline in jupyter notebook



import cufflinks as cf #importing plotly and cufflinks in offline mode  

import plotly.offline  

cf.go_offline()  

cf.set_config_file(offline=False, world_readable=True)



'''Display markdown formatted output like bold, italic bold etc.'''

from IPython.display import Markdown,HTML

import matplotlib.gridspec as gridspec # to do the grid of plots



from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.metrics.pairwise import linear_kernel

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.metrics.pairwise import cosine_similarity



def bold(string):

    display(Markdown(string))
netdata=pd.read_csv('../input/netflix-shows/netflix_titles_nov_2019.csv')
netdata["date_added"] = pd.to_datetime(netdata['date_added'])

netdata['year_added'] = netdata['date_added'].dt.year

netdata['month_added'] = netdata['date_added'].dt.month



netdata['season_count'] = netdata.apply(lambda x : x['duration'].split(" ")[0] if "Season" in x['duration'] else "", axis = 1)

netdata['duration'] = netdata.apply(lambda x : x['duration'].split(" ")[0] if "Season" not in x['duration'] else "", axis = 1)

netdata['duration'] =netdata.apply(lambda x : '0' if x['duration']=='' else x['duration'],axis=1)

netdata['duration'] =  netdata['duration'].astype(float)
display(HTML(f"""

   

        <ul class="list-group">

          <li class="list-group-item disabled" aria-disabled="true"><h4>Dataset preview</h4></li>

          <li class="list-group-item"><h4>Number of rows in the dataset: <span class="label label-primary">{ netdata.shape[0]:,}</span></h4></li>

          <li class="list-group-item"> <h4>Number of columns in the dataset: <span class="label label-primary">{netdata.shape[1]}</span></h4></li>

          <li class="list-group-item"><h4>Number of unique types in the dataset: <span class="label label-success">{ netdata['type'].nunique():,}</span></h4></li>

        </ul>

  

    """))
temp = netdata['rating'].value_counts()

df = pd.DataFrame({'rating': temp.index,

                   'values': temp.values

                  })

df.iplot(kind='pie',labels='rating',values='values', title='Distribution of ratings in data')
#Credits: https://www.kaggle.com/shivamb/a-visual-and-insightful-journey-donorschoose/data



t = netdata['month_added'].value_counts()



lObjectsALLcnts = list(t.values)



lObjectsALLlbls = list(t.index)

mapp1 = {1:'Jan',2:'Feb',3:'Mar',4:'Apr',5:'May',6:'Jun',7:'Jul',8:'Aug',9:'Sep',10:'Oct',11:'Nov',12:'Dec'}

lObjectsALLlbls = [mapp1[x] for x in lObjectsALLlbls]



iN = len(lObjectsALLcnts)

arrCnts = np.array(lObjectsALLcnts)



theta=np.arange(0, 2*np.pi, 2*np.pi/iN)

width = (2*np.pi)/iN *0.5

bottom = 50



fig = plt.figure(figsize=(8,8))

ax = fig.add_axes([0.2, 0.1, 1, 0.9], polar=True)

fig.suptitle('Month released', fontsize=16)

bars = ax.bar(theta, arrCnts, width=width, bottom=bottom, color='#eb6951')

plt.axis('off')



rotations = np.rad2deg(theta)

for x, bar, rotation, label in zip(theta, bars, rotations, lObjectsALLlbls):

    lab = ax.text(x,bottom+bar.get_height() , label, ha='left', va='center', rotation=rotation, rotation_mode="anchor",)   

plt.show()
df1=netdata[netdata['type']=='TV Show']

df2=netdata[netdata['type']=='Movie']



df1=df1.groupby('date_added')['title'].nunique().sort_values()

df2=df2.groupby('date_added')['title'].nunique().sort_values()



trace1 = go.Scatter(

    x = df1.index,

    y = df1.values,

    mode = 'markers',

    name = 'Tv shows'

)



trace2 = go.Scatter(

    x = df2.index,

    y = df2.values,

    mode = 'markers',

    name = 'Movies'

)



layout = go.Layout(template= "plotly_dark", title = 'TV Shows', xaxis = dict(title = 'Years'))

fig = go.Figure(data = [trace1,trace2], layout = layout)

fig.show()
pd.crosstab(netdata.type,netdata.year_added,margins=True).style.background_gradient(cmap='summer_r')
pd.crosstab(netdata.type,netdata.season_count,margins=True).style.background_gradient(cmap='RdYlGn')

f,ax=plt.subplots(1,1,figsize=(10,5))

netdata[netdata['type']=='Movie'].duration.plot.hist(ax=ax,bins=20,edgecolor='black',color='red')

ax.set_title('Movie')
cols=['country','type','year_added','season_count','release_year','rating','duration','month_added']



sns.heatmap(netdata[cols].corr(),annot=True,cmap='RdYlGn',linewidths=0.2)

fig=plt.gcf()

fig.set_size_inches(10,8)

plt.show()
#Filling nan values with empty string in director,country,cast

nanlist=['cast','director','country']

metalist=['cast','listed_in','director']



nancols=netdata[nanlist].fillna(' ')



netdata=netdata.drop(nanlist,axis=1)

filterdata=pd.concat([netdata,nancols],axis=1)



#Few processing

filterdata['cast']=filterdata['cast'].apply(lambda x: x.split(','))

filterdata['director']=filterdata['director'].apply(lambda x: x.split(','))

filterdata['country']=filterdata['country'].apply(lambda x: x.split(','))
#Define a TF-IDF Vectorizer Object. Remove all english stop words such as 'the', 'a'

tfidf = TfidfVectorizer(stop_words='english')



#Construct the required TF-IDF matrix by fitting and transforming the data

tfidf_matrix = tfidf.fit_transform(filterdata['description'])



#Output the shape of tfidf_matrix

tfidf_matrix.shape
# Compute the cosine similarity matrix

cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
#Construct a reverse map of indices and movie titles

indices = pd.Series(filterdata.index, index=filterdata['title']).drop_duplicates()
# Function that takes in movie title as input and outputs most similar movies

def get_recommendations(title, cosine_sim=cosine_sim):

    

    # Get the index of the movie that matches the title

    idx = indices[title]



    # Get the pairwsie similarity scores of all movies with that movie

    sim_scores = list(enumerate(cosine_sim[idx]))



    # Sort the movies based on the similarity scores

    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)



    # Get the scores of the 10 most similar movies

    sim_scores = sim_scores[1:11]



    # Get the movie indices

    movie_indices = [i[0] for i in sim_scores]



    # Return the top 10 most similar movies

    return filterdata['title'].iloc[movie_indices]

get_recommendations('The Matrix')
def create_meta(x):

    return ' '.join(x['cast'] + x['director'] +x['country'])



filterdata['meta'] = filterdata.apply(create_meta, axis=1)
count = CountVectorizer(stop_words='english')

count_matrix = count.fit_transform(filterdata['meta'])
cosine_sim2 = cosine_similarity(count_matrix, count_matrix)
get_recommendations('The Matrix',cosine_sim+cosine_sim2)