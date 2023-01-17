import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.style as style
from sklearn.feature_extraction import text
import warnings
warnings.filterwarnings("ignore")
style.use('ggplot')
df=pd.read_csv('/kaggle/input/ted-talks/ted_main.csv')
df.columns
df.head()
import datetime
df['film_date'] = df['film_date'].apply(lambda x: datetime.datetime.fromtimestamp( int(x)).strftime('%d-%m-%Y'))
df['published_date'] = df['published_date'].apply(lambda x: datetime.datetime.fromtimestamp( int(x)).strftime('%d-%m-%Y'))
df['duration_hr']=df['duration']/(60*60)
df['duration_hr']=df['duration_hr'].astype(float)
df['duration_hr']=df['duration_hr'].round(decimals=2)
df.head()
df.isnull().sum()
def get_top10(feature):
    popular_talks = df[['title', 'main_speaker', 'views','comments','film_date', 'published_date','duration','duration_hr']].sort_values(by=feature,ascending=False).set_index(feature).reset_index().head(10)
    return popular_talks
get_top10('views')
get_top10('comments')
get_top10('duration')
def get_graph(feature):
    df_top10 = df[['title', 'main_speaker', 'views','comments', 'published_date','duration','duration_hr']].sort_values(feature, ascending=False).head(10)
    x_labels=df_top10['title'].to_list()
    plt.figure(figsize=(12,8))
    ax=sns.barplot(x='title',y=feature,data=df_top10)
    ax.set_title('TOP 10 TED Talks by '+str(feature).upper())
    ax.set_xlabel('TITLE')
    ax.set_ylabel(feature)
    ax.set_xticklabels(x_labels,rotation='vertical')
    
    rects=ax.patches

    
    labels=df_top10[feature].to_list()
    
    for rect,label in zip(rects,labels):
        height=rect.get_height()
        ax.text(rect.get_x()+rect.get_width()/2,height,label,ha='center',va='bottom')
    
    
get_graph('views')
get_graph('comments')
get_graph('duration')
g=sns.pairplot(data=df)
g.map_upper(sns.scatterplot,color='blue')
g.map_lower(sns.scatterplot, color='green')
g.map_diag(plt.hist)
sns.scatterplot(x='duration_hr',y='comments',data=df,color='blue')
df[df['comments']==df['comments'].max()]
df.corr()
month_order = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
df['month'] = df['film_date'].apply(lambda x: month_order[int(x.split('-')[1]) - 1])

df_month = pd.DataFrame(df['month'].value_counts()).reset_index()
df_month.columns = ['month', 'talks']
plt.figure(figsize=(19,8))
sns.barplot(x='month', y='talks', data=df_month, order=month_order)
df['year'] = df['film_date'].apply(lambda x: x.split('-')[2])
df_year = pd.DataFrame(df['year'].value_counts().reset_index())
df_year.columns = ['year', 'talks']
plt.figure(figsize=(19,8))
sns.countplot(x='year',data=df)
df_rec=df[['title','description']]
df_rec
df_rec.columns
import re
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
def remove_tags(string):
    result = re.sub('<.*?>','',string)
    return result
df_rec['description']=df_rec['description'].apply(lambda cw : remove_tags(cw))
df_rec['description']=df['description'].str.lower()
df_rec['title']=df['title'].str.lower()
df_rec
Text=df_rec['description'].tolist()
tfidf=text.TfidfVectorizer(input=Text,stop_words="english")
matrix=tfidf.fit_transform(Text)
print(matrix.shape)
from yellowbrick.text import TSNEVisualizer
tsne = TSNEVisualizer()
tsne.fit(matrix)
### Get Similarity Scores using cosine similarity
from sklearn.metrics.pairwise import cosine_similarity
cosine_sim=cosine_similarity(matrix)
print(cosine_sim)
indices = pd.Series(df_rec['title'])
def recommend_talks(name):
    talks=[]
    idx = indices[indices == name].index[0]
    sort_index = pd.Series(cosine_sim[idx]).sort_values(ascending = False)
    top_10 = sort_index.iloc[1:11]
    for i in top_10.index:
        talks.append(indices[i])
    print(*talks, sep='\n')
def rec():
    try:
        i = 1
        while(i > 0):
            name = input("\n Enter The title of the TED Talk : ")
            if name.lower() == 'quit':
                break
            else:
                print("\n",recommend_talks(name))

    except KeyboardInterrupt:
        print("The TED Talk does not exist\n")
        rec()

    except IndexError:
        print("The TED Talk does not exist\n")
        rec()
        

print("To exit Enter \"quit\" \n")
rec()

