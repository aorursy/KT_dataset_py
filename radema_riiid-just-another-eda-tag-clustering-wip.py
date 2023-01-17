# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
!pip install datatable
%%time



# reading the dataset from raw csv file

import datatable as dt

import plotly.express as px

import matplotlib.pyplot as plt



#df = dt.fread("../input/riiid-test-answer-prediction/train.csv").to_pandas()

df = pd.read_csv('../input/riiid-test-answer-prediction/train.csv',nrows = 1000000)

print(df.shape)
df.head(10)
df.info()
d = {'row_id':'int64',

 'timestamp':'int64',

 'user_id':'object',

 'content_id':'object',

 'content_type_id':'int64',

 'task_container_id':'object',

 'user_answer':'object',

 'answered_correctly':'int64',

 'prior_question_elapsed_time':'float64',

 'prior_question_had_explanation':'object'}

df.head(100000).astype(d).describe(include='all')
user_profile = df[df['answered_correctly']!=-1].groupby(

    'user_id',as_index = False

    ).agg(age_as_learner = pd.NamedAgg(column = 'timestamp', aggfunc=max),

          num_contents = pd.NamedAgg(column = 'content_id', aggfunc=pd.Series.nunique),

          num_events = pd.NamedAgg(column = 'row_id', aggfunc = 'count'),

          num_true_answers = pd.NamedAgg(column = 'answered_correctly', aggfunc = sum),

          perc_true_answers = pd.NamedAgg(column = 'answered_correctly', aggfunc = 'mean'),

          std_true_answers = pd.NamedAgg(column = 'answered_correctly', aggfunc = 'std'),

          #skew_true_answers = pd.NamedAgg(column = 'answered_correctly', aggfunc = 'skew'),

          avg_prior_question_elapsed_time = pd.NamedAgg(column = 'prior_question_elapsed_time', aggfunc = 'mean'),

          std_prior_question_elapsed_time = pd.NamedAgg(column = 'prior_question_elapsed_time', aggfunc = 'std'),

          )

user_profile['user_id'] = user_profile['user_id'].astype('object')

user_profile.age_as_learner /=(60*60*24*1000*365)

user_profile.avg_prior_question_elapsed_time /= 60*60*24

user_profile.std_prior_question_elapsed_time /= 60*60*24

print('Number of Users: ', len(user_profile))

user_profile.describe()
pd.plotting.scatter_matrix(user_profile.select_dtypes(include='number'),alpha = 0.1, figsize = (30,30))
n = len(df.content_type_id)

a,b = df.content_type_id.value_counts()

print('The number of lecture events is : {} ({}%)'.format(b,100*b/n))

print('The number of question events is : {} ({}%)'.format(a,100*a/n))
user_profile.sort_values(by='age_as_learner',ascending = False).head(10)
user_profile.sort_values(by='num_contents',ascending = False).head(10)
user_profile.sort_values(by='num_true_answers',ascending = False).head(10)
user_profile[user_profile.std_true_answers.notnull()].sort_values(by='perc_true_answers',ascending = False).head(10)
#user_series = df.groupby(['user_id','timestamp']).agg({'answered_correctly':'sum','row_id':'count'})

#user_series = user_series.groupby(level=0).agg({'answered_correctly':'cumsum','row_id':'cumsum'}).reset_index()

#user_series['avg_correct'] = user_series.answered_correctly.div(user_series.row_id, axis = 0)

user_series
questions = pd.read_csv('../input/riiid-test-answer-prediction/questions.csv')#,nrows = 1000000)
questions.head(10)
questions.info()
questions[questions.tags.isnull()]
dq = {

    'question_id':'object',

    'bundle_id':'object',

    'correct_answer':'object',

    'part':'object'

}



questions.astype(dq).describe()
(questions.correct_answer.value_counts()/len(questions)).plot.bar()
(questions.part.value_counts()/len(questions)).plot.bar()
questions.fillna('-1', inplace = True)
from collections import Counter



tags = Counter(questions.tags.str.split(' ')[0])

for tag in questions.tags.str.split(' '):

    try:

        tags.update(tag)

    except:

        print('Found exception: tags field equal to ', tag)
tags = pd.DataFrame.from_dict(dict(tags),orient='index').reset_index().sort_values(by=0, ascending = False)

tags.columns = ['tag','count_tag']
tags.head(20).plot.bar()
from sklearn.preprocessing import MultiLabelBinarizer

mlb = MultiLabelBinarizer()



tags_of_parts = pd.DataFrame(mlb.fit_transform(questions.tags.str.split(' ')), columns = mlb.classes_, index = questions.question_id)
tags_of_parts.info()
test = pd.merge(questions[['question_id','part']],tags_of_parts, on = 'question_id').drop(['question_id'], axis = 1).groupby('part').sum()
import seaborn as sns



f, ax = plt.subplots(figsize=(34, 5))



ax = sns.heatmap(test)
test.reset_index().melt(id_vars = ['part'], var_name = 'tag',value_name = 'count').astype('int64')
import nltk, re

from sklearn.metrics.pairwise import cosine_similarity

from sklearn.cluster import KMeans





def tokenize_and_stem(text):

    tokens = [word for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]

    filtered_tokens = []

    for token in tokens:

        if re.search('[0-9]', token):

            filtered_tokens.append(token)

    return filtered_tokens



from sklearn.feature_extraction.text import TfidfVectorizer



#define vectorizer parameters

tfidf_vectorizer = TfidfVectorizer(max_df=1.0, max_features=200000,

                                 min_df=0.0,use_idf=True, tokenizer=tokenize_and_stem, ngram_range=(1,3))



%time tfidf_matrix = tfidf_vectorizer.fit_transform(questions.tags) #fit the vectorizer to synopses



print(tfidf_matrix.shape)



tags = tfidf_vectorizer.get_feature_names()





dist = 1 - cosine_similarity(tfidf_matrix)
sse = {}

for k in range(2,30,3):

    km = KMeans(n_clusters=k)

    #print('Number of cluster: {}'.format(k))

    #%time 

    km.fit(tfidf_matrix)

    sse[k] = km.inertia_

plt.figure()

plt.plot(list(sse.keys()), list(sse.values()))

plt.xlabel("Number of cluster")

plt.ylabel("SSE")

plt.show()

sse.keys()
NUM_CLUSTERS = 11





km = KMeans(n_clusters=NUM_CLUSTERS)



%time km.fit(tfidf_matrix)



clusters = km.labels_.tolist()



n_words = 5



order_centroids = km.cluster_centers_.argsort()[:, ::-1] 



for i in range(NUM_CLUSTERS):

    print("Cluster %d words:" % i, end='')

    

    for ind in order_centroids[i, :n_words]: 

        print(' %s' % tags[ind], end=' ')

    print() #add whitespace

    

questions['kmean_cluster'] = clusters
questions.head(10)
dq['kmean_cluster']='object'

questions.astype(dq).info()
questions.kmean_cluster.value_counts()/len(questions)
(pd.pivot_table(

    questions, 

    values = 'question_id', 

    index = ['kmean_cluster'], 

    columns = ['part'],aggfunc='count')

).fillna(0).astype('int64')
questions.groupby('bundle_id').agg({

    'question_id':'count',

    'part':pd.Series.nunique,

    'kmean_cluster':pd.Series.nunique}).describe()
(questions.groupby('bundle_id').agg({

    'question_id':'count',

    'part':pd.Series.nunique,

    'kmean_cluster':pd.Series.nunique})[['question_id','kmean_cluster']]).hist()
lectures = pd.read_csv('../input/riiid-test-answer-prediction/lectures.csv')#,nrows = 1000000)

print(lectures.info())

lectures.head(10)
print(pd.pivot_table(

    lectures,

    values = 'lecture_id', 

    index = ['type_of'], 

    columns = ['part'],aggfunc='count'    

).fillna(0))

print()

(lectures.part.value_counts()).plot.bar()
(lectures.type_of.value_counts()).plot.bar()
print(lectures.groupby('tag').agg({'lecture_id':'count'}).sort_values(by='lecture_id',ascending=False)[:25].plot.bar())
print(lectures.groupby('tag').agg({'lecture_id':'count'}).hist())