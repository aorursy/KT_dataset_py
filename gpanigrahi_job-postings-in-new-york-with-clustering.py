import numpy as np

import pandas as pd

import matplotlib.pyplot as plt 

import seaborn as sns

init_notebook_mode(connected=True)  

plt.style.use('ggplot')

from collections import Counter

from wordcloud import WordCloud

from PIL import Image

import urllib.request

import random

from sklearn.preprocessing import StandardScaler
import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
df = pd.read_csv("../input/new-york-city-current-job-postings/nyc-jobs.csv")
df.head()

df.info()
def missing_values_table(df):

   

    # Total missing values

    mis_val = df.isnull().sum()

    

    # Percentage of missing values

    mis_val_percent = 100 * df.isnull().sum() / len(df)

    

    # Make a table with the results

    mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)

    

    # Rename the columns

    mis_val_table_columns = mis_val_table.rename(

    columns = {0 : 'Missing Values', 1 : '% of Total Values'})

    

    # Sort the table by percentage of missing descending

    # .iloc[:, 1]!= 0: filter on missing missing values not equal to zero

    mis_val_table_columns = mis_val_table_columns[

        mis_val_table_columns.iloc[:,1] != 0].sort_values(

    '% of Total Values', ascending=False).round(2)  # round(2), keep 2 digits

    

    # Print some summary information

    print("Dataset has {} columns.".format(df.shape[1]) + '\n' + 

    "There are {} columns that have missing values.".format(mis_val_table_columns.shape[0]))

    

    # Return the dataframe with missing information

    return mis_val_table_columns
missing_values_table(df)
df = df.drop(['Recruitment Contact', 'Hours/Shift', 'Post Until', 'Work Location 1'],axis=1)
df = df.drop(['Additional Information'],axis=1)
missing_values_table(df)
for column in ['Job Category','Residency Requirement','Posting Date', 'Posting Updated','Process Date', 'To Apply']:

    df[column] = df[column].fillna(df[column].mode()[0]) 


high_sal_range = (df.groupby('Civil Service Title')['Salary Range To'].mean().nlargest(10)).reset_index()



fig = px.bar(high_sal_range, y="Civil Service Title", x="Salary Range To", orientation='h', title = "Highest High Salary Range",color=  "Salary Range To", color_continuous_scale= px.colors.qualitative.G10).update_yaxes(categoryorder="total ascending")

fig.show()

popular_categories = df['Job Category'].value_counts()[:5]

popular_categories
job_categorydf = df['Job Category'].value_counts(sort=True, ascending=False)[:10].rename_axis('Job Category').reset_index(name='Counts')

job_categorydf = job_categorydf.sort_values('Counts')
trace = go.Scatter(y = job_categorydf['Job Category'],x = job_categorydf['Counts'],mode='markers',

                   marker=dict(size= job_categorydf['Counts'].values/2,

                               color = job_categorydf['Counts'].values,

                               colorscale='Viridis',

                               showscale=True,

                               colorbar = dict(title = 'Opening Counts')),

                   text = job_categorydf['Counts'].values)



data = [(trace)]



layout= go.Layout(autosize= False, width = 1000, height = 750,

                  title= 'Top 10 Job Openings Count',

                  hovermode= 'closest',

                  xaxis=dict(showgrid=False,zeroline=False,

                             showline=False),

                  yaxis=dict(title= 'Job Openings Count',ticklen= 2,

                             gridwidth= 5,showgrid=False,

                             zeroline=True,showline=False),

                  showlegend= False)



fig = go.Figure(data=data, layout=layout)

py.iplot(fig)
num_cols = df._get_numeric_data().columns
num_cols
cat_cols = list(set(df.columns) - set(num_cols))
today = pd.datetime.today()
redudant_cols = ['Job ID', '# Of Positions','Posting Updated','Minimum Qual Requirements','To Apply','Business Title','Level']
df[cat_cols]
df = df.drop(redudant_cols,axis=1)
df
def parse_categories(x):

    l = x.replace('&', ',').split(',')

    l = [x.strip().rstrip(',') for x in l]

    key_categories.extend(l)
def parse_keywords(x, l):

    x = x.lower()

    tokens = nltk.word_tokenize(x)

    stop_words = set(stopwords.words('english'))

    token_l = [w for w in tokens if not w in stop_words and w.isalpha()]

    l.extend(token_l)
def preferred_skills(x):

    kwl = []

    df[df['Job Category'] == x]['Preferred Skills'].dropna().apply(parse_keywords, l=kwl)

    kwl = pd.Series(kwl)

    return kwl.value_counts()[:20]
key_categories = []

df['Job Category'].dropna().apply(parse_categories)

key_categories = pd.Series(key_categories)

key_categories = key_categories[key_categories!='']

popular_categories = key_categories.value_counts().iloc[:25]
key_categories
df['cat'] = key_categories
plt.figure(figsize=(10,10))

sns.countplot(y=key_categories, order=popular_categories.index, palette='YlGn')


salary_table = df[['Civil Service Title', 'Salary Range From', 'Salary Range To']]

jobs_highest_high_range = pd.DataFrame(salary_table.groupby(['Civil Service Title'])['Salary Range To'].mean().nlargest(10)).reset_index()

plt.figure(figsize=(8,6))

sns.barplot(y='Civil Service Title', x='Salary Range To', data=jobs_highest_high_range, palette='Greys')
def plot_wordcloud(text):

    wordcloud = WordCloud(background_color='white',

                     width=1024, height=720).generate(text)

    plt.clf()

    plt.imshow(wordcloud, interpolation="bilinear")

    plt.axis('off')

    plt.show()
job_description_keywords = []

df['Job Description'].apply(parse_keywords, l=job_description_keywords)

plt.figure(figsize=(10, 8))

counter = Counter(job_description_keywords)

common = [x[0] for x in counter.most_common(40)]

plot_wordcloud(' '.join(common))
words = []

counts = []

for letter, count in counter.most_common(10):

    words.append(letter)

    counts.append(count)
import matplotlib.cm as cm

from matplotlib import rcParams

colors = cm.rainbow(np.linspace(0, 1, 10))

rcParams['figure.figsize'] = 20, 10



plt.title('Top words in the Job description vs their count')

plt.xlabel('Count')

plt.ylabel('Words')

plt.barh(words, counts, color=colors)
df['Posting Date'] = pd.to_datetime(df['Posting Date'])
df['Process Date'] = pd.to_datetime(df['Process Date'])
df['years of exprience'] = df['Process Date'] - df['Posting Date']
df['years of exprience'] = df['years of exprience'].dt.days
df_cluster = df[['cat','Salary Range To','years of exprience']]
df_cluster.isna().sum()
df_cluster['cat'].value_counts()
df_cluster['cat'].fillna('Others', inplace=True)
df_cluster=df_cluster.replace('\*','',regex=True)
df_cluster
#Calculating the Hopkins statistic

from sklearn.neighbors import NearestNeighbors

from random import sample

from numpy.random import uniform

import numpy as np

from math import isnan

 

def hopkins(X):

    d = X.shape[1]

    #d = len(vars) # columns

    n = len(X) # rows

    m = int(0.1 * n) 

    nbrs = NearestNeighbors(n_neighbors=1).fit(X.values)

 

    rand_X = sample(range(0, n, 1), m)

 

    ujd = []

    wjd = []

    for j in range(0, m):

        u_dist, _ = nbrs.kneighbors(uniform(np.amin(X,axis=0),np.amax(X,axis=0),d).reshape(1, -1), 2, return_distance=True)

        ujd.append(u_dist[0][1])

        w_dist, _ = nbrs.kneighbors(X.iloc[rand_X[j]].values.reshape(1, -1), 2, return_distance=True)

        wjd.append(w_dist[0][1])

 

    H = sum(ujd) / (sum(ujd) + sum(wjd))

    if isnan(H):

        print(ujd, wjd)

        H = 0

 

    return H
#Let's check the Hopkins measure

hopkin_df = df_cluster

hopkins(hopkin_df.drop(['cat'],axis=1))
df_cluster_std = df_cluster

X_C = df_cluster_std.drop(['cat'],axis=1)

df_cluster_std = StandardScaler().fit_transform(X_C)
df_cluster
#Let's check the silhouette score first to identify the ideal number of clusters

# To perform KMeans clustering 

from sklearn.cluster import KMeans

from sklearn.metrics import silhouette_score

sse_ = []

for k in range(2, 10):

    kmeans = KMeans(n_clusters=k).fit(df_cluster_std)

    sse_.append([k, silhouette_score(df_cluster_std, kmeans.labels_)])
plt.plot(pd.DataFrame(sse_)[0], pd.DataFrame(sse_)[1]);
#The sihouette score reaches a peak at around 4 clusters indicating that it might be the ideal number of clusters.

#Let's use the elbow curve method to identify the ideal number of clusters.

ssd = []

for num_clusters in list(range(1,10)):

    model_clus = KMeans(n_clusters = num_clusters, max_iter=50)

    model_clus.fit(df_cluster_std)

    ssd.append(model_clus.inertia_)



plt.plot(ssd)


#K-means with k=4 clusters

model_clus4 = KMeans(n_clusters = 4, max_iter=50)

model_clus4.fit(df_cluster_std)
dat4=df_cluster

dat4.index = pd.RangeIndex(len(dat4.index))

dat_km = pd.concat([dat4, pd.Series(model_clus4.labels_)], axis=1)

dat_km.columns = ['cat','salary_max','exp','ClusterID']

dat_km
dat_km['ClusterID'].value_counts()
dat_km
#One thing we noticed is all distinct clusters are being formed except cluster 1 with more data points

#Now let's create the cluster means wrt to the various variables mentioned in the question and plot and see how they are related

df_final=pd.merge(df,dat_km,on='cat')
df_final
df_final.info()
#Along Job category and years of exprience

sns.scatterplot(x='cat',y='exp',hue='ClusterID',data=df_final)
#Along Job category and years of exprience

sns_plot = sns.scatterplot(x='Salary Range To',y='exp',hue='cat',data=df_final)
fig = sns_plot.get_figure()

fig.savefig("output.png")
#let's take a look at those Job category clusters and try to make sense if the clustering process worked well.

df_final_on_jobcat = df_final[df_final['ClusterID']==1]
df_final_on_jobcat['cat'].value_counts()