import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

import numpy as np

pd.set_option('display.max_colwidth', -1)

pd.set_option('display.max_columns', 50)

from sklearn.feature_extraction.text import TfidfVectorizer 
!ls ../input
comments = pd.read_csv('../input/comments.csv')

print(comments.head(2))

print(comments.info())
#https://www.kaggle.com/jeffd23/visualizing-word-vectors-with-t-sne

from gensim.models import word2vec

from sklearn.manifold import TSNE

corpus = []

stoplist = set("for a of the and to in on is are by its it so at s t am im i m an my be then was me will p we did i".split())

for a,b in (comments['comments_body']).iteritems():

    #print(a)

    #print(b.lower().split())

    #corpus.append([word for word in str(re.sub('[\W_]+', ' ', b)).lower().split() if word not in stoplist])

    #[word for word in str(b).lower().split() if word not in stoplist]

    corpus.append([word for word in str(b).lower().split() if word not in stoplist])





def tsne_plot(model):

    "Creates and TSNE model and plots it"

    labels = []

    tokens = []



    for word in model.wv.vocab:

        tokens.append(model[word])

        labels.append(word)

    

    tsne_model = TSNE(perplexity=40, n_components=2, init='pca', n_iter=2500, random_state=23)

    new_values = tsne_model.fit_transform(tokens)



    x = []

    y = []

    for value in new_values:

        x.append(value[0])

        y.append(value[1])

        

    plt.figure(figsize=(16, 16)) 

    for i in range(len(x)):

        plt.scatter(x[i],y[i])

        plt.annotate(labels[i],

                     xy=(x[i], y[i]),

                     xytext=(5, 2),

                     textcoords='offset points',

                     ha='right',

                     va='bottom')

    plt.show()

   

model = word2vec.Word2Vec(corpus, size=300, window=20, min_count=100, workers=4)

tsne_plot(model)
from wordcloud import WordCloud, STOPWORDS



def wordCloudFunction(df,column,numWords):

    vectorizer = TfidfVectorizer(max_features=numWords, stop_words='english',ngram_range=(1,2))

    vector_comments = vectorizer.fit_transform(df[column].dropna())

    word_string = str(vectorizer.get_feature_names())

    indices = np.argsort(vectorizer.idf_)[::-1]

    top_n = 200

    #print(vectorizer.get_feature_names())

    top_features = [vectorizer.get_feature_names()[i] for i in indices]

    #print(top_features)

    wordcloud = WordCloud(stopwords=STOPWORDS,

                          background_color='white',

                          max_words=numWords,

                          width=2000,height=1000,

                         ).generate(' '.join(top_features))

    plt.clf()

    plt.imshow(wordcloud)

    plt.axis('off')

    plt.show()

wordCloudFunction(comments,'comments_body',200)
vectorizer = TfidfVectorizer(max_features=20, stop_words='english',ngram_range=(1,2))

vector_comments = vectorizer.fit_transform(comments['comments_body'].dropna())

print(vectorizer.get_feature_names())

indices = np.argsort(vectorizer.idf_)[::-1]

print(indices)

top_features = [vectorizer.get_feature_names()[i] for i in indices]

print(top_features)

print(' '.join(top_features))

wordcloud = WordCloud(stopwords=STOPWORDS,

                          background_color='white',

                          max_words=20,

                          width=2000,height=1000,

                         ).generate(' '.join(top_features))

plt.clf()

plt.imshow(wordcloud)

plt.axis('off')

plt.show()
comments['comments_date_added'] = pd.to_datetime(comments['comments_date_added'])
#https://stackoverflow.com/questions/16176996/keep-only-date-part-when-using-pandas-to-datetime

comments['just_date'] = comments['comments_date_added'].dt.date
comments.groupby('just_date')['comments_id'].count().plot(figsize=(20,10))
#https://stackoverflow.com/questions/46181475/python-pandas-groupby-date-and-count-new-records-for-each-period

#fig, ax = plt.subplots(figsize=(10,7))

#plt.figure(figsize=(20,10))

(comments

     .groupby(['comments_author_id'], as_index=False)['just_date']  # Group by `user_id` and get first date.

     .first()

     .groupby(['just_date'])  # Group result on `date` and take counts.

     .count()

     .reindex(comments['just_date'].unique())  # Reindex on original dates.

     .fillna(0)).plot(figsize=(20,10))
school_memberships = pd.read_csv('../input/school_memberships.csv')

print(school_memberships.head(2))

print(school_memberships.info())

#school_memberships.groups_group_type.value_counts().plot(kind='bar')
groups = pd.read_csv('../input/groups.csv')

print(groups.head(2))

_ = groups.groups_group_type.value_counts().plot(kind='bar')
group_memberships = pd.read_csv('../input/group_memberships.csv')

print(group_memberships.head(2))

grouptype_membership = groups.merge(right=group_memberships, how='inner', left_on='groups_id', right_on='group_memberships_group_id')

print(grouptype_membership.head(2))

_ = grouptype_membership[['groups_group_type', 'group_memberships_user_id']].groupby('groups_group_type')['group_memberships_user_id'].nunique().plot(kind='bar')

print(grouptype_membership.info())
tags = pd.read_csv('../input/tags.csv')

tags.head(2)
# matches = pd.read_csv('../input/matches.csv')

# #matches.head(2)

# matches.info()
emails = pd.read_csv('../input/emails.csv')

# #emails.head(2)

emails.info()
_ = emails.emails_frequency_level.value_counts().plot(kind='bar')
#emails['emails_date_sent'] = pd.to_datetime(emails['emails_date_sent'])
emails_frequency = emails[['emails_recipient_id','emails_frequency_level']].drop_duplicates(keep='last')
#emails.groupby('emails_recipient_id')['emails_date_sent'].max().dt.year.unique()
# import datetime

# emails.loc[emails['emails_date_sent']>datetime.date(year=2018,month=1,day=1)].groupby('emails_recipient_id')['emails_frequency_level'].nunique()
# emails['emails_date_sent_month'] = emails['emails_date_sent'].dt.month 

# emails['emails_date_sent_day_name'] = emails['emails_date_sent'].dt.day_name() 

# print(emails.head(2))

# print(emails.info())

# #emails.emails_frequency_level.value_counts().plot(kind='bar')

# #pd.crosstab(emails['month'],emails['day_name'])
# fig, ax = plt.subplots(figsize=(8,8))

# sns.heatmap(pd.crosstab(emails['emails_date_sent_month'],emails['emails_date_sent_day_name']), annot=True, ax=ax, fmt='d', linewidths=0.1, cmap='PiYG')
students = pd.read_csv('../input/students.csv')

students['students_date_joined'] = pd.to_datetime(students['students_date_joined'])

students['students_date_joined_month'] = students['students_date_joined'].dt.month 

students['students_date_joined_day_name'] = students['students_date_joined'].dt.day_name() 

students['students_date_joined_dayofweek'] = students['students_date_joined'].dt.dayofweek 

students['students_date_joined_year'] = students['students_date_joined'].dt.year 

students.head(2)
#students = students.merge(right=group_memberships, how='left', left_on='students_id', right_on='group_memberships_user_id')
students.info()
tag_users = pd.read_csv('../input/tag_users.csv')

tag_users.head(2)

tagname_users = tags.merge(right=tag_users, how='inner', left_on='tags_tag_id', right_on='tag_users_tag_id')

print(tagname_users.head(2))

print('Most common tags followed by students')

print(tagname_users[['tags_tag_name','tag_users_user_id']].groupby('tags_tag_name')['tag_users_user_id'].nunique().nlargest(10))
tagname_students_details = students.merge(right=tagname_users, how='inner', left_on='students_id', right_on='tag_users_user_id')

print(tagname_students_details.head(2))

print(tagname_students_details.info())

tagname_students_details_tagsjoined = tagname_students_details.groupby('students_id')['tags_tag_name'].apply(' , '.join).reset_index()

tagname_students_details_tagsjoined.columns = ['students_id', 'students_tags_tag_name']

print(tagname_students_details_tagsjoined.info())
tagname_students_details_tagsjoined.loc[tagname_students_details_tagsjoined['students_id']=='94f41c4228e2452aa8ddf786d02b06c0']
students = students.merge(right=tagname_students_details_tagsjoined, how='left', on='students_id')

#students['students_tags_tag_name'] = students['students_id'].apply(lambda x: tagname_students_details_tagsjoined.loc[tagname_students_details_tagsjoined['students_id']==x])

students.info()
students.students_location.value_counts().head(10)
#pd.crosstab(students['students_date_joined_month'],students['students_date_joined_day_name'])
fig, ax = plt.subplots(figsize=(8,8))

sns.heatmap(pd.crosstab(students['students_date_joined_month'],students['students_date_joined_day_name']), annot=True, ax=ax, fmt='d', linewidths=0.1, cmap='PiYG')
questions = pd.read_csv('../input/questions.csv')

questions['questions_date_added'] = pd.to_datetime(questions['questions_date_added'])

questions['questions_date_added_month'] = questions['questions_date_added'].dt.month 

questions['questions_date_added_day_name'] = questions['questions_date_added'].dt.day_name() 

questions['questions_date_added_dayofweek'] = questions['questions_date_added'].dt.dayofweek

questions['questions_date_added_year'] = questions['questions_date_added'].dt.year 

questions['questions_date_added_hour'] = questions['questions_date_added'].dt.hour 
questions.shape
questions.info()
#https://www.kaggle.com/jeffd23/visualizing-word-vectors-with-t-sne

#https://radimrehurek.com/gensim/tut1.html

from gensim.models import word2vec

from sklearn.manifold import TSNE

import re

corpus = []

stoplist = set('for a of the and to in on is are by its it so at s t am im i m an my be then was me will p we did'.split())

for a,b in (questions['questions_title'] + ' ' + questions['questions_body']).iteritems():

    #print(a)

    #print(b.lower().split())

    corpus.append([word for word in str(re.sub('[\W_]+', ' ', b)).lower().split() if word not in stoplist])

    #corpus.append(str(re.sub('[\W_]+', ' ', b)).lower().split())

#corpus[0:1]    



def tsne_plot(model):

    "Creates and TSNE model and plots it"

    labels = []

    tokens = []



    for word in model.wv.vocab:

        tokens.append(model[word])

        labels.append(word)

    

    tsne_model = TSNE(perplexity=40, n_components=2, init='pca', n_iter=2500, random_state=23)

    new_values = tsne_model.fit_transform(tokens)



    x = []

    y = []

    for value in new_values:

        x.append(value[0])

        y.append(value[1])

        

    plt.figure(figsize=(16, 16)) 

    for i in range(len(x)):

        plt.scatter(x[i],y[i])

        plt.annotate(labels[i],

                     xy=(x[i], y[i]),

                     xytext=(5, 2),

                     textcoords='offset points',

                     ha='right',

                     va='bottom')

    plt.show()

   

model = word2vec.Word2Vec(corpus, size=50, window=20, min_count=300, workers=4)

tsne_plot(model)
model.wv.most_similar('doctor')
model.wv.most_similar('university')
question_scores = pd.read_csv('../input/question_scores.csv')

question_scores.head(2)

question_scores.columns = ['questions_id','question_score']

questions = questions.merge(right=question_scores, how='left', left_on='questions_id', right_on='questions_id')

print(questions.head(2))

questions.shape

questions.head(2)
tag_questions = pd.read_csv('../input/tag_questions.csv')

print(tag_questions.head(2))

tagname_questions = tags.merge(right=tag_questions, how='inner', left_on='tags_tag_id', right_on='tag_questions_tag_id')

print(tagname_questions.head(2))

print('Most common tags in questions')

print(tagname_questions[['tags_tag_name','tag_questions_question_id']].groupby('tags_tag_name')['tag_questions_question_id'].nunique().nlargest(10))
#tagname_questions = tags.merge(right=tag_questions, how='inner', left_on='tags_tag_id', right_on='tag_questions_tag_id')

#tagname_questions.head(2)

tagname_questions_details = questions.merge(right=tagname_questions, how='inner', left_on='questions_id', right_on='tag_questions_question_id')

#print(tagname_questions_details.head(2))

#print(tagname_questions_details.info())

tagname_questions_details_tagsjoined = tagname_questions_details.groupby('questions_id')['tags_tag_name'].apply(' , '.join).reset_index()

tagname_questions_details_tagsjoined.columns = ['questions_id', 'questions_tags_tag_name']

#print(tagname_questions_details_tagsjoined.info())
# from nltk.tokenize import sent_tokenize, word_tokenize

# from nltk.stem.snowball import SnowballStemmer

# # from nltk.stem import WordNetLemmatizer

# # lemmatizer = WordNetLemmatizer()

# stemmer = SnowballStemmer('english')

# def stemSentence(sentence):

#     token_words=word_tokenize(sentence)

#     token_words

#     stem_sentence=[]

#     for word in token_words:

#         stem_sentence.append(stemmer.stem(word))

#         #stem_sentence.append(lemmatizer.lemmatize(word))

#         stem_sentence.append(" ")

#     return "".join(stem_sentence)

# questions['questions_title_stem'] = questions['questions_title'].apply(stemSentence)

# questions['questions_body_stem'] = questions['questions_body'].apply(stemSentence)

# questions[['questions_title_stem','questions_title']].head(2) 
questions.groupby(['questions_author_id'])['questions_id'].nunique().reset_index().sort_values('questions_id',ascending=False).head(2)
sns.distplot(questions.groupby(['questions_author_id'])['questions_id'].nunique().reset_index()['questions_id'])
questions.groupby(['questions_author_id'])['questions_id'].nunique().describe()
questions.groupby(['questions_date_added_year'])['questions_author_id'].nunique().plot(kind='bar')
#pd.crosstab(questions['questions_date_added_month'],questions['questions_date_added_day_name'])
fig, ax = plt.subplots(figsize=(8,8))

sns.heatmap(pd.crosstab(questions['questions_date_added_month'],questions['questions_date_added_day_name']), ax=ax, annot=True,fmt='d', cmap='PiYG')
answers = pd.read_csv('../input/answers.csv')

answers['answers_date_added'] = pd.to_datetime(answers['answers_date_added'])

answers['answers_date_added_year'] = answers['answers_date_added'].dt.year

answers.info()

answers.head(2)
answer_scores = pd.read_csv('../input/answer_scores.csv')

answer_scores.head(2)

answer_scores.columns = ['answers_id','answer_score']

answers = answers.merge(right=answer_scores, how='left', left_on='answers_id', right_on='answers_id')

print(answers.head(2))

answers.shape
answers.groupby(['answers_date_added_year'])['answers_author_id'].nunique().plot(kind='bar')
sns.distplot(answers.groupby(['answers_author_id'])['answers_id'].nunique().reset_index()['answers_id'])
answers.groupby(['answers_author_id'])['answers_id'].nunique().describe()
professionals = pd.read_csv('../input/professionals.csv')

professionals['professionals_date_joined'] = pd.to_datetime(professionals['professionals_date_joined'])

professionals['professionals_date_joined_year'] = professionals['professionals_date_joined'].dt.year

print(professionals.tail(2))

print(professionals.info())

#professionals['professionals_date_joined_year'].value_counts().sort_index()
professionals.professionals_location.value_counts().head(10)
#professionals.loc[(professionals['professionals_date_joined_year']==2019)].info()
question_answers = questions.merge(right=answers, how='left', left_on='questions_id', right_on='answers_question_id')
question_answers.info()
question_answers['answers_question_date_diff'] = (question_answers['answers_date_added'] - question_answers['questions_date_added']).dt.days

question_answers.head(2)
question_answers_with_answer_isnull = question_answers.copy()
#question_answers_with_answer_isnull.answers_author_id.nunique()
#question_answers.answers_id.isnull().sum()
#question_answers.loc[question_answers.answers_id.isnull()].groupby('question_year')['questions_id'].agg([ 'count']).plot(kind='bar', title='answers_id is null')
question_answers.dropna(inplace=True)

question_answers.info()
professional_year_last_answered = question_answers.groupby('answers_author_id')['answers_date_added_year'].max().astype(int).reset_index()
professional_year_last_answered.columns = ['professionals_id', 'professional_year_last_answered']
professional_year_last_answered.head(2)
answers_author_id_answer_count =  question_answers.groupby('answers_author_id')['answers_id'].agg(['count']).reset_index()
answers_author_id_answer_count.columns = ['answers_author_id','overall_author_answer_count']
answers_author_id_answer_count.info()
answers_author_id_answer_count.head(2)
professionals.info()
professionals = professionals.merge(how='left',right=professional_year_last_answered, left_on='professionals_id', right_on='professionals_id')
professionals.info()
professionals_answers_count = professionals.merge(how='left',right=answers_author_id_answer_count, left_on='professionals_id', right_on='answers_author_id')

professionals_answers_count.head(2)
professionals_answers_count['overall_author_answer_count_isnull']=professionals_answers_count.overall_author_answer_count.isnull()
professionals_answers_count.head(2)
professionals_answers_count.groupby(['overall_author_answer_count_isnull'])['professionals_id'].agg(['count']).reset_index()
professionals['professionals_id'].nunique()
answers['answers_author_id'].nunique()
# sns.catplot(x="professionals_date_joined_year", y="count", 

#             hue="overall_author_answer_count_isnull", 

#             col="overall_author_answer_count_isnull",

#             sharey=False,

#             data=professionals_answers_count.groupby(['professionals_date_joined_year', 'overall_author_answer_count_isnull'])['professionals_id'].agg(['count']).reset_index(),

#             height=6, kind="bar", palette="muted")
#sns.boxplot(x=professionals_answers_count['professionals_date_joined_year'], y=professionals_answers_count['overall_author_answer_count'].fillna(0))
#question_answers = question_answers.merge(how='left',right=questions_author_id_answer_count, left_on='questions_author_id', right_on='questions_author_id')

#question_answers.head(2)
sns.distplot(question_answers.groupby('questions_id')['answers_id'].agg(['count']))
question_answers.groupby('questions_id')['answers_id'].agg(['count']).describe()
question_answers.info()
question_answers[['questions_id','answers_id','questions_date_added_year']].groupby('questions_date_added_year')['answers_id'].agg(['count']).reset_index()

question_answers_grpby_qid = question_answers[['questions_id','answers_id','questions_date_added_year']].groupby(['questions_date_added_year','questions_id'])['answers_id'].agg(['count']).reset_index()

question_answers_grpby_qid.columns=['questions_date_added_year', 'questions_id', 'count_of_answers_to_question']
question_answers_grpby_qid.head(2)
sns.boxplot(x=question_answers_grpby_qid['questions_date_added_year'], 

            y=question_answers_grpby_qid['count_of_answers_to_question'])
# sns.set(style="ticks")



# f, ax = plt.subplots(figsize=(7, 6))

# ax.set_xscale("log")



# sns.boxplot(x="count_of_answers_to_question", y="questions_date_added_year", data=question_answers_grpby_qid,

#             palette="vlag",

#             orient='h')



# ax.xaxis.grid(True)

# ax.set(ylabel="")

# sns.despine(trim=True, left=True)
sns.distplot(question_answers['answers_question_date_diff'])
sns.boxplot(x=question_answers['questions_date_added_year'], y=question_answers['answers_question_date_diff'])
qa_professionals = question_answers.merge(right=professionals, left_on='answers_author_id', right_on='professionals_id')

qa_professionals.head(2)
qa_professionals.info()
qa_professionals.answers_author_id.nunique()
qa_professionals_tags = qa_professionals.merge(how='left',right=tagname_questions_details_tagsjoined, left_on='questions_id', right_on='questions_id')
qa_professionals_tags.head(2)
qa_professionals_tags.info()
qa_professionals_tags_students = qa_professionals_tags.merge(how='left',right=students, left_on='questions_author_id', right_on='students_id')



qa_professionals_tags_students.head(2)
qa_professionals_tags_students.info()
#qa_professionals_tags_students = qa_professionals_tags_students.merge(how='left',right=tagname_students_details_tagsjoined, left_on='students_id', right_on='students_id')

#qa_professionals_tags_students.info()
features_1 = ['questions_date_added_month','questions_date_added_day_name','questions_date_added_year','students_location']

features_2 = ['questions_title', 'questions_body', 'questions_tags_tag_name','students_tags_tag_name']

features_3 = ['questions_author_id', 'questions_date_added_month','questions_date_added_day_name','questions_date_added_year',

             'students_location','students_date_joined_month','students_date_joined_day_name','students_date_joined_year']
vectorizer = TfidfVectorizer(max_features=260, stop_words='english',ngram_range=(1,2))
X = qa_professionals_tags_students['questions_title'] + ' ' + qa_professionals_tags_students['questions_body'] + ' ' + qa_professionals_tags_students['questions_tags_tag_name'].fillna('') + ' ' + qa_professionals_tags_students['students_tags_tag_name'].fillna('')
tfidf_vector = vectorizer.fit_transform(X)

tfidf_features =  np.array(vectorizer.get_feature_names())

tfidf_features[:10]
sorted_array = np.argsort(tfidf_vector.toarray()).flatten()[::-1]
sorted_array
tfidf_features[sorted_array][:25]
tfidf_vector.shape
from sklearn.cluster import MiniBatchKMeans

kmeans_model = MiniBatchKMeans(n_clusters=12)

kmeans_model.fit(tfidf_vector).score(tfidf_vector)

preds = kmeans_model.predict(tfidf_vector)

preds
qa_professionals_tags_students['preds'] = preds
qa_professionals_tags_students['preds'].value_counts()
qa_professionals_tags_students.loc[(qa_professionals_tags_students['preds']==0), ['questions_title']].sample(10)
#qa_professionals_tags_students.loc[(qa_professionals_tags_students['preds']==9), ['questions_title']].sample(10)
#qa_professionals_tags_students.loc[(qa_professionals_tags_students['preds']==1), ['questions_title']].sample(10)
Nc = range(1, 35)

kmeans = [MiniBatchKMeans(n_clusters=i) for i in Nc]

kmeans

score = [kmeans[i].fit(tfidf_vector).score(tfidf_vector) for i in range(len(kmeans))]

score

plt.plot(Nc,score)

plt.xlabel('Number of Clusters')

plt.ylabel('Score')

plt.title('Elbow Curve')

plt.show()
from sklearn.model_selection import train_test_split

X_train, X_test, dataframe_train, dataframe_test = train_test_split(tfidf_vector, qa_professionals_tags_students.copy())
kmeans_model = MiniBatchKMeans(n_clusters=35)

kmeans_model.fit(X_train)

train_preds = kmeans_model.predict(X_train)

dataframe_train.loc[:,'preds'] = train_preds

test_preds = kmeans_model.predict(X_test)

dataframe_test.loc[:,'preds'] = test_preds
a_question = dataframe_test.sample().iloc[0]
a_question
prediction_for_a_question = a_question['preds']
prediction_for_a_question
# dataframe_train.loc[

#     ((dataframe_train['preds']==23) 

#      & (dataframe_train['professional_year_last_answered']>=2018.0))]
dataframe_train.loc[((dataframe_train['preds']==prediction_for_a_question) & (dataframe_train['professional_year_last_answered']>=2018)),['professional_year_last_answered','questions_title','question_score','answer_score','professionals_industry', 'professionals_headline','professionals_id','professionals_location']].sample(10)
# from sklearn.cluster import AgglomerativeClustering

# from sklearn.mixture import GaussianMixture

# from sklearn.cluster import Birch



# Nc = range(1, 26)

# models = [Birch(n_clusters=i) for i in Nc]

# models

# score = [kmeans[i].fit(tfidf_vector).score(tfidf_vector) for i in range(len(models))]

# score

# plt.plot(Nc,score)

# plt.xlabel('Number of Clusters')

# plt.ylabel('Score')

# plt.title('Elbow Curve')

# plt.show()
# from sklearn.cluster import DBSCAN



# db = DBSCAN(eps=0.3, min_samples=10).fit(tfidf_vector)

# labels = db.labels_



# # Number of clusters in labels, ignoring noise if present.

# n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)

# n_clusters_

# from sklearn import metrics

# print("Silhouette Coefficient: %0.3f"

#       % metrics.silhouette_score(tfidf_vector, labels))

# qa_professionals_tags_students['preds'] = labels

# qa_professionals_tags_students['preds'].value_counts()
def cosine(v1, v2):

    #v1 = np.array(v1)

    #v2 = np.array(v2)



    return np.dot(v1, v2) / (np.sqrt(np.sum(v1**2)) * np.sqrt(np.sum(v2**2)))
tfidf_vector.toarray()[0].shape
question_index_value = 2551 #8858
np.argsort(np.dot(tfidf_vector.toarray(),tfidf_vector.toarray()[question_index_value]))[::-1][:10]
similar_questions = np.argsort(cosine(tfidf_vector.toarray(), tfidf_vector.toarray()[question_index_value]))[::-1][:11]
similar_questions = list(similar_questions) 

similar_questions.remove(question_index_value)

similar_questions
#qa_professionals_tags_students.loc[qa_professionals_tags_students['questions_id']=='b39f3ae849944898b29ce02f663efc74'].T
qa_professionals_tags_students.iloc[question_index_value]
qa_professionals_tags_students[['questions_title','professional_year_last_answered','question_score','answer_score']].iloc[similar_questions]
qa_professionals_tags_students[['professional_year_last_answered','professionals_industry', 'professionals_headline','professionals_id','professionals_location']].iloc[similar_questions]
from sklearn.neighbors import NearestNeighbors

knn=NearestNeighbors(n_neighbors=12,algorithm='auto',metric='cosine')

knn_fit=knn.fit(tfidf_vector)

Neighbors = knn_fit.kneighbors(tfidf_vector[question_index_value])
neighbors_without_original = Neighbors[1].flatten().tolist()

neighbors_without_original.remove(question_index_value)

print(neighbors_without_original)
qa_professionals_tags_students[['questions_title','professional_year_last_answered','question_score','answer_score']].iloc[neighbors_without_original]
qa_professionals_tags_students[['professional_year_last_answered','professionals_industry', 'professionals_headline','professionals_location','professionals_id']].iloc[neighbors_without_original]
from sklearn.preprocessing import MinMaxScaler

additional_features_scaled = MinMaxScaler().fit_transform(qa_professionals_tags_students[['students_date_joined_month', 'students_date_joined_dayofweek', 'students_date_joined_year', 'questions_date_added_month', 'questions_date_added_dayofweek', 'questions_date_added_hour', 'professional_year_last_answered']].fillna(0))
tfidf_vector.shape
tfidf_vector.toarray()
additional_features_scaled
tfidf_additional_features_scaled_concat = np.concatenate((tfidf_vector.toarray(),additional_features_scaled), axis=1)
tfidf_additional_features_scaled_concat.shape
from sklearn.neighbors import NearestNeighbors

knn=NearestNeighbors(n_neighbors=12,algorithm='auto',metric='cosine')

#knn_fit=knn.fit(tfidf_vector)

knn_fit=knn.fit(tfidf_additional_features_scaled_concat)
#tfidf_additional_features_scaled_concat[question_index_value].shape

#tfidf_additional_features_scaled_concat[question_index_value].reshape(1, -1).shape
Neighbors = knn_fit.kneighbors(tfidf_additional_features_scaled_concat[question_index_value].reshape(1, -1))
neighbors_without_original = Neighbors[1].flatten().tolist()

neighbors_without_original.remove(question_index_value)

print(neighbors_without_original)
qa_professionals_tags_students[['questions_title','professional_year_last_answered','question_score','answer_score']].iloc[neighbors_without_original]
qa_professionals_tags_students[['professional_year_last_answered','professionals_industry', 'professionals_headline','professionals_location','professionals_id']].iloc[neighbors_without_original]
# from sklearn.decomposition import PCA

# pca = PCA()

# pca_transformed = pca.fit_transform(tfidf_additional_features_scaled_concat)

# print(pca.explained_variance_ratio_)  
# from sklearn.metrics.pairwise import cosine_similarity  

# arr = cosine_similarity(tfidf_vector.toarray(), tfidf_vector.toarray()[26].reshape(1, -1))

# np.argsort(np.reshape(arr,50105))[::-1][:10]
# arr = cosine_similarity(tfidf_vector.toarray(), tfidf_vector.toarray()[25:27])

# np.argsort(arr[:,0])[::-1][:10]

# np.argsort(arr[:,1])[::-1][:10]

# qa_professionals_tags_students[features_2].iloc[25:27]

# qa_professionals_tags_students[['questions_title']].iloc[[

#     24127, 35276, 35284, 35285, 35288, 35289, 16143, 16141, 35275,

#        16140]]

# qa_professionals_tags_students[['questions_title']].iloc[

#     [50104, 42560, 19216, 19212, 19211, 19201, 19199, 19191, 19190,

#        19189]]