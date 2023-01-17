import numpy as np
import pandas as pd
import plotly.express as px
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse.linalg import svds
import scipy
import sklearn
import numpy as np
import math
import os


import nltk
nltk.download('stopwords')


for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
df_art = pd.read_csv('../input/articles-sharing-reading-from-cit-deskdrop/shared_articles.csv')
df_art
print(df_art.lang.nunique())
print(df_art.lang.unique())
df_art.lang.value_counts()
px.pie(df_art, df_art.lang.value_counts().index, df_art.lang.value_counts().values, title='Article Languages')
df_art[df_art.lang.isin(['la', 'es', 'ja'])]
df_art[df_art.lang.isin(['la', 'es', 'ja'])].text.tolist()
df_art = df_art[df_art.lang.isin(['en', 'pt'])]
df_art
df_int = pd.read_csv('../input/articles-sharing-reading-from-cit-deskdrop/users_interactions.csv')
df_int
df_int.eventType.unique()
px.pie(df_int, df_int.eventType.value_counts().index,df_int.eventType.value_counts().values, title='Person Interaction Type')
stop_words = stopwords.words('english') + stopwords.words('portuguese')
vectorizer = TfidfVectorizer(analyzer='word',
                     ngram_range=(1, 2),
                     min_df=0.005,
                     max_df=0.8,
                     max_features=5000,
                     stop_words=stop_words)

item_ids = df_art['contentId'].tolist()
tfidf_matrix = vectorizer.fit_transform(df_art['title'] + "" + df_art['text'])
tfidf_feature_names = vectorizer.get_feature_names()
tfidf_matrix
event_strength = {
   'VIEW': 1.0,
   'LIKE': 2.0, 
   'BOOKMARK': 3.0, 
   'FOLLOW': 4.0,
   'COMMENT CREATED': 5.0,  
}

df_int['eventStrength'] = df_int['eventType'].apply(lambda x: event_strength[x])
df_int
df_int_count = df_int.groupby(['personId', 'contentId']).size().groupby('personId').size().reset_index().rename(columns={0:'IntCount'})
print('# of people: %d' % len(df_int_count))
df_int_count.sort_values('IntCount', ascending=False)
df_2int_count = df_int_count[df_int_count.IntCount > 2]
print('# of people with at least 3 interactions: %d' % len(df_2int_count))
print('# of interactions: %d' % len(df_int))
df_int_tot = pd.merge(df_int, df_2int_count, how = 'right', left_on = 'personId', right_on = 'personId')
print('# of interactions from people with at least 3 interactions: %d' % len(df_int_tot))
df_int.info()
df_int_tot.info()
df_int_full = df_int_tot.groupby(['personId', 'contentId'])['eventStrength'].sum().apply(lambda x: math.log(1+x, 2)).reset_index()
print('# of unique person/item interactions: %d' % len(df_int_full))
df_int_full.head(15)
int_df = df_int_full[df_int_full['contentId'].isin(df_art['contentId'])].set_index('personId')
person_profiles = {}
for person_id in int_df.index.unique():
    
    int_person_df = int_df.loc[person_id]

    item_profiles_list = [tfidf_matrix[item_ids.index(x):item_ids.index(x)+1] for x in int_person_df['contentId']]
    person_item_profiles = scipy.sparse.vstack(item_profiles_list)

    person_item_strengths = np.array(int_person_df['eventStrength']).reshape(-1,1)
    person_item_strengths_weighted_avg = np.sum(person_item_profiles.multiply(person_item_strengths), axis=0) / np.sum(person_item_strengths)
    person_profile_norm = sklearn.preprocessing.normalize(person_item_strengths_weighted_avg)
    person_profiles[person_id] = person_profile_norm
person_profiles[3609194402293569455]
len(person_profiles)
print( person_profiles[3609194402293569455].shape)
pd.DataFrame(sorted(zip(tfidf_feature_names, 
                        person_profiles[3609194402293569455].flatten().tolist()), key=lambda x: -x[1]),
             columns=['token', 'relevance'])
df_int_full.personId.nunique()
def produce_person_vector(person_id_number):
    person_id =  df_int_full.personId.unique()[person_id_number]
    person_profile = person_profiles[person_id]
    df = pd.DataFrame(sorted(zip(tfidf_feature_names, 
                                 person_profile.flatten().tolist()), key=lambda x: -x[1]),
                      columns=['token', 'relevance'])
    return df
produce_person_vector(150)
df = produce_person_vector(0).rename(columns={'token':'token1', 'relevance':'relevance1'})
df
for i in range(1, df_int_full.personId.nunique()):
    df['token'+str(i+1)] = produce_person_vector(i).token
    df['relevance'+str(i+1)] = produce_person_vector(i).relevance
df