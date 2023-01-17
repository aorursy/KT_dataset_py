# Helper libraries
import os
import math
import warnings
import pickle
import re

import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords
# Read Data
files = [(f[:-4],"../input/%s"%f) for f in os.listdir("../input")]
files = dict(files)
print(os.listdir("../input"))

def read_csv(name):
    if name not in files:
        print("error")
        return None    
    return pd.read_csv(files[name])

# load data
questions = read_csv("questions")
answers = read_csv("answers")
professionals = read_csv("professionals")
### Read questions data
q = questions.copy()
q['qtext'] = list(q.apply(lambda x:'%s %s' %(x['questions_title'],x['questions_body']), axis=1))
q = q.drop(['questions_author_id','questions_date_added','questions_body','questions_title'],axis=1)

### Read answers data
a = answers.copy()
a = a.drop(['answers_date_added'],axis=1)
uri_re = r'(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:\'".,<>?«»“”‘’]))'
tag_re = r'<\/?[a-z]+>'

def strip_html_tag(s):
    temp = re.sub(uri_re, ' ', str(s))
    return re.sub(tag_re, ' ', temp)

a['answers_body'] = a['answers_body'].apply(strip_html_tag)
a.rename(columns={'answers_question_id':'questions_id',
                  'answers_author_id':'professionals_id',
                  "answers_body":"atext"},inplace=True)
          
### Read professionals data
p = professionals.copy()
# Connect these professionals with tags and user_tag tables
tags = read_csv("tags")
tag_users = read_csv("tag_users")

tag_users = tag_users[tag_users["tag_users_user_id"].isin(p['professionals_id'])]
tag_users = tag_users.merge(tags, how="left", left_on="tag_users_tag_id", right_on = "tags_tag_id")

t = tag_users.groupby(["tag_users_user_id"])['tags_tag_name'].apply(lambda x: ' '.join(x))
t = t.to_frame().reset_index()

p = p.merge(t, how="left", left_on="professionals_id", right_on = "tag_users_user_id")

# get text
temp = p[['professionals_location','professionals_industry','professionals_headline','tags_tag_name']].fillna('')

p['ptext'] = temp['professionals_location']+" "+temp['professionals_industry']+" "+\
             temp['professionals_headline']+" "+temp['tags_tag_name']

p = p.drop(['professionals_location','professionals_industry','professionals_headline',\
           'professionals_date_joined','tag_users_user_id','tags_tag_name'],axis=1)

print("Number of professionals: ",len(p['ptext']),",with ",sum(p['ptext']!="   ")," of them having data" )
# Remove general words
noise = ['school','would','like', 'want', 'dont', 
         'become','sure','go', 'get', 'college', 
         'career', 'wanted', 'im', 'ing', 'ive',
         'know', 'high', 'becom', 'job', 'best',
         'day', 'hi', 'name', 'help', 'people',
         'year', 'years', 'next', 'interested', 
         'question', 'questions', 'take', 'even',
         'though', 'please', 'tell']

stop = stopwords.words('english')

def remove_general_words(df, col):
    df[col] = df[col].str.lower().str.split() # convert all str to lowercase    
    df[col] = df[col].apply(lambda x: [item for item in x if item not in stop]) # remove stopwords
    df[col] = df[col].apply(lambda x: [item for item in x if item not in noise]) # remove general words
    df[col] = df[col].apply(' '.join) # convert list to str
    return df

q = remove_general_words(q,'qtext')
p = remove_general_words(p,'ptext')
a = remove_general_words(a,'atext')
'''
# Save the data
pickle.dump(q, open('../input/q.p', 'wb'))
pickle.dump(a, open('../input/a.p', 'wb'))
pickle.dump(p, open('../input/p.p', 'wb'))
'''
'''
# Read saved data
q = pickle.load(open('../input/q.p', mode='rb'))
a = pickle.load(open('../input/a.p', mode='rb'))
p = pickle.load(open('../input/p.p', mode='rb'))
'''
def get_similar_docs(corpus, query_text, threshold=0.0, top=10):
    tfidf = TfidfVectorizer(ngram_range=(1,3), stop_words = 'english', max_features = 500, max_df=0.9)
    corpus_tfidf = tfidf.fit_transform(corpus)
    text_tfidf = tfidf.transform([query_text])
    sim = cosine_similarity(corpus_tfidf, text_tfidf)
    sim_idx = (sim >= threshold).nonzero()[0]
    result = pd.DataFrame({'similarity':sim[sim_idx].reshape(-1,),
                          'text':corpus[sim_idx]},
                          index=sim_idx)
    result = result.sort_values(by=['similarity'], ascending=False).head(top)
    return result
# Example
corpus = q['qtext']
query_text = corpus[2]
print('Example 1 Question:\n', query_text)
sim_questions = get_similar_docs(corpus, query_text)
sim_questions
def get_questions_answers(sim_questions):  
    sim_q_a = sim_questions.merge(q, left_index=True, right_index=True).merge(a)
    return sim_q_a
sim_q_a = get_questions_answers(sim_questions)
sim_q_a.head()
def get_recommendation(df, top_n=5):
    temp_values = df['similarity']/df['questions_id'].apply(lambda x: df.groupby('questions_id').size()[x])
    temp_values = temp_values * df['professionals_id'].apply(lambda x: df.groupby('professionals_id').size()[x])
    df["value"] = temp_values
    top_prof = df.groupby('professionals_id').sum().reset_index().sort_values('value', ascending = False).head(top_n)
    top_prof = top_prof[['professionals_id', 'value']]
    top_prof.columns = ['professional', 'recommendation_score']
    print(top_prof)        
get_recommendation(sim_q_a)
# find the new professionals
pa = p.merge(a, how = "left")
p_new = p[pa["questions_id"].isnull()]
print ("There are",p_new.shape[0],"new professionals")
# Example
corpus_p = p['ptext']
print('Example 2 Question:\n', query_text)
sim_professionals = get_similar_docs(corpus_p, query_text, top=2)
# Recommend new professiors
rec_new = sim_professionals.merge(p, left_index=True, right_index=True)[['professionals_id', 'similarity']]
rec_new.columns = ['professional', 'recommendation_score']
rec_new