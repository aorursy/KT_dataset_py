import numpy as np

import pandas as pd

import os

import re

import spacy

from spacy.lang.en import English

import nltk

from nltk import word_tokenize, pos_tag

from nltk.stem.wordnet import WordNetLemmatizer

from nltk.corpus import wordnet as wn

import gensim

from gensim import corpora

import pickle

from html.parser import HTMLParser

import matplotlib.pyplot as plt

import matplotlib.patches as mpatches

import matplotlib.dates as mdates

from collections import Counter



data_folder = "../input/"



emails = pd.read_csv(os.path.join(data_folder, 'emails.csv'))

questions = pd.read_csv(os.path.join(data_folder, 'questions.csv'))

matches = pd.read_csv(os.path.join(data_folder, 'matches.csv'))



profs = pd.read_csv(os.path.join(data_folder, 'professionals.csv'))

tag_users = pd.read_csv(os.path.join(data_folder, 'tag_users.csv'))

studs = pd.read_csv(os.path.join(data_folder, 'students.csv'))

tagq = pd.read_csv(os.path.join(data_folder, 'tag_questions.csv'))

groups = pd.read_csv(os.path.join(data_folder, 'groups.csv'))

emails = pd.read_csv(os.path.join(data_folder, 'emails.csv'))

groupm = pd.read_csv(os.path.join(data_folder, 'group_memberships.csv'))

answers = pd.read_csv(os.path.join(data_folder, 'answers.csv'))

comments = pd.read_csv(os.path.join(data_folder, 'comments.csv'))

matches = pd.read_csv(os.path.join(data_folder, 'matches.csv'))

tags = pd.read_csv(os.path.join(data_folder, 'tags.csv'))

questions = pd.read_csv(os.path.join(data_folder, 'questions.csv'))

school = pd.read_csv(os.path.join(data_folder, 'school_memberships.csv'))

hearts = pd.read_csv(os.path.join(data_folder, 'answer_scores.csv'))
profs['professionals_date_joined'] = pd.to_datetime(profs['professionals_date_joined'])

profs['join_yr']=profs['professionals_date_joined'].dt.year

profsyr = profs[['join_yr','professionals_id']].groupby('join_yr').count().reset_index()



plt.bar(profsyr.join_yr, profsyr.professionals_id)



plt.xlabel('Year')

plt.ylabel('Number of Professionals Added')

plt.title('Professionals Joining CV')

plt.show()
profs['professionals_industry']=profs['professionals_industry'].fillna("")

profs['professionals_industry'] = profs['professionals_industry'].str.replace('healthcare','health care', flags=re.IGNORECASE, regex=True)

prof_ind = profs['professionals_industry'].tolist()

prof_ind=list(filter(lambda a: a != "", prof_ind))

prof_ind=list(filter(lambda a: a != "and", prof_ind))

prof_ind=list(filter(lambda a: a != "or", prof_ind))

prof_tokens=[]

for i in prof_ind:

    words = word_tokenize(i)

    words=[word.lower() for word in words if word.isalpha()]

    prof_tokens.append(words)

    

prof_tokens = [item for sublist in prof_tokens for item in sublist]

prof_tokens=list(filter(lambda a: a != "and", prof_tokens))

prof_tokens=list(filter(lambda a: a != "or", prof_tokens))
dic=dict(Counter(prof_tokens))

proft = pd.DataFrame.from_dict(dic, orient="index").reset_index()

proft.columns=['type','occur']

proft=proft.sort_values(by=['occur'], ascending=False)

proft.shape
proftx=proft.type[:19].tolist()

proftx.append('others')

profty=proft.occur[:19].tolist()

profty.append(proft.occur.sum()-proft.occur[:19].sum())

fig1, ax1 = plt.subplots(figsize=(10, 8))

ax1.pie(profty, labels=proftx, autopct='%1.1f%%',startangle=90)

ax1.axis('equal')

plt.show()
proft.tail(10)
proftlist=proft.type.tolist()

for i in range(len(proftlist)):

    for j in range(i+1, len(proftlist)):

        word1 = wn.synsets(proftlist[i], 'n')

        word2 = wn.synsets(proftlist[j], 'n')

        if (word1 and word2):

            if (word1[0].wup_similarity(word2[0])>0.95):

                proftlist[j]=proftlist[i]
proft.type=proftlist

proft = proft.groupby('type')['occur'].sum().reset_index()

proft.shape
ansauth=answers.groupby('answers_author_id')['answers_body'].count().reset_index()

profans = pd.merge(profs, ansauth, how="left", left_on='professionals_id', right_on='answers_author_id')

profans.shape, profans[~profans['answers_author_id'].isna()].shape
noinfoauth = np.setdiff1d(answers.answers_author_id,profs.professionals_id)

noinfoauth.shape, answers[answers['answers_author_id'].isin(noinfoauth)].shape
ans = answers[['answers_id','answers_date_added']]

ans['answers_date_added'] = pd.to_datetime(ans['answers_date_added'])

ans['year']=ans['answers_date_added'].dt.year

ans=ans.groupby('year')['answers_id'].count().reset_index()

plt.bar(ans.year, ans.answers_id)



plt.xlabel('Year')

plt.ylabel('Answers Added')

plt.title('Number of Answers')

plt.show()
hearts[hearts['score']!=0].shape
ansh = pd.merge(answers, hearts, how="left", left_on='answers_id', right_on='id')

ansh['answers_date_added'] = pd.to_datetime(ansh['answers_date_added'])

ansh['year']=ansh['answers_date_added'].dt.year

ansh = ansh.groupby('year')['score'].mean().reset_index()

plt.bar(ansh.year, ansh.score)



plt.xlabel('Year')

plt.ylabel('Average Score')

plt.title('Trend of Average Answer Scores')

plt.show()

matches['matches_email_id'].value_counts().head()
maxq = matches[matches['matches_email_id']==569938]

maxq = pd.merge(maxq, questions, how="left", left_on=['matches_question_id'], right_on=['questions_id'])

maxq[['matches_email_id','questions_title','questions_body']].head()
fastans = answers.groupby('answers_question_id')['answers_date_added'].min().reset_index()

qatime=pd.merge(questions, fastans, how="left", left_on='questions_id', right_on='answers_question_id')

qatime['delta']=pd.to_datetime(qatime['answers_date_added']) -pd.to_datetime(qatime['questions_date_added'])

bin1=qatime[qatime['delta']<=pd.Timedelta('1 days')].shape[0]

bin2=qatime[qatime['delta']<=pd.Timedelta('3 days')].shape[0]-bin1

bin3=qatime[qatime['delta']<=pd.Timedelta('7 days')].shape[0]-bin2

bin4=qatime[qatime['delta']>pd.Timedelta('7 days')].shape[0]
labels=['1 day','2-3 days','4-7 days','>1 week']

profty=[bin1, bin2, bin3, bin4]

fig1, ax1 = plt.subplots(figsize=(6, 6))

ax1.pie(profty, labels=labels, autopct='%1.1f%%',startangle=90)

ax1.axis('equal')

plt.show()
class MLStripper(HTMLParser):

    def __init__(self):

        self.reset()

        self.strict = False

        self.convert_charrefs= True

        self.fed = []

    def handle_data(self, d):

        self.fed.append(d)

    def get_data(self):

        return ''.join(self.fed)



def strip_tags(html):

    s = MLStripper()

    s.feed(html)

    return s.get_data()



uri_re = r'(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:\'".,<>?«»“”‘’]))'



def strip_html(s):

    s = s.replace("\n","")

    return re.sub(uri_re, ' ', str(s))
questions['full_text'] = questions['questions_title'] +' '+ questions['questions_body']

questions['full_text'] = questions['full_text'].apply(strip_tags)

answers = answers[['answers_question_id','answers_body']]

answers['answers_body'] = answers['answers_body'].fillna("")

answers['answers_body'] = answers['answers_body'].apply(strip_tags)

answers['answers_per_q'] = answers.groupby('answers_question_id')['answers_body'].transform(lambda x: '.'.join(x))

ansrs=answers[['answers_question_id','answers_per_q']]

ansrs=ansrs.drop_duplicates()

qa = pd.merge(questions, ansrs, how="left", left_on='questions_id', right_on='answers_question_id')

qa['qa'] = qa['full_text'] +' '+ qa['answers_per_q']

qa['qa']=qa['qa'].fillna("")

qa=qa[~qa['answers_per_q'].isna()]
answers = pd.read_csv(os.path.join(data_folder, 'answers.csv'))

answers['answers_body']=answers['answers_body'].fillna("")

answers['answers_body'] = answers['answers_body'].apply(strip_tags)



answers['proans']=answers.groupby('answers_author_id')['answers_body'].transform(lambda x: '.'.join(x))

answers=answers[['answers_author_id','proans']]

answers=answers.drop_duplicates()
profans = pd.merge(profs, answers, how="left", left_on='professionals_id', right_on='answers_author_id')

tag_users = pd.read_csv(os.path.join(data_folder, 'tag_users.csv'))



tag_users=pd.merge(tag_users, tags, how="left", left_on='tag_users_tag_id', right_on='tags_tag_id')

tag_users['alltags']=tag_users.groupby('tag_users_user_id')['tags_tag_name'].transform(lambda x: '. '.join(x))

tag_users = tag_users[['tag_users_user_id','alltags']]

tag_users=tag_users.drop_duplicates()



profanstag = pd.merge(profans, tag_users, how="left", left_on='professionals_id', right_on='tag_users_user_id')

profanstag['allprof']=profanstag['proans']+' '+profanstag['alltags']

profanstag['allprof']=profanstag['allprof'].fillna("")
spacy.load('en')

stoplist=nltk.corpus.stopwords.words('english')

stoplist.append('school')

stoplist.append('college')

stoplist.append('career')

stoplist.append('degree')

stoplist.append('people')

stoplist.append('experience')

en_stop = set(stoplist)

def tokenize(text):

    lda_tokens = [token for token, pos in pos_tag(word_tokenize(text)) if pos.startswith('N')]

    return lda_tokens



def get_lemma(word):

    lemma = wn.morphy(word)

    if lemma is None:

        return word

    else:

        return lemma



def get_lemma2(word):

    return WordNetLemmatizer().lemmatize(word)



def prepare_text_for_lda(text):

    tokens = tokenize(text)

    tokens = [token for token in tokens if len(token) > 4]

    tokens = [token for token in tokens if token not in en_stop]

    tokens = [get_lemma(token) for token in tokens]

    return tokens
fullcorp = qa['qa'].tolist()+profanstag['allprof'].tolist()
text_data = []

for line in fullcorp:

    tokens = prepare_text_for_lda(line[:1000000])

    text_data.append(tokens)

    

dictionary = corpora.Dictionary(text_data)

corpus = [dictionary.doc2bow(text) for text in text_data]



pickle.dump(corpus, open('corpus.pkl', 'wb'))

dictionary.save('dictionary.gensim')
NUM_TOPICS = 30

ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics = NUM_TOPICS, id2word=dictionary, passes=15)

ldamodel.save('model5.gensim')

topics = ldamodel.print_topics(num_words=5)

for topic in topics:

    print(topic)
proftopics = ldamodel.get_document_topics(corpus)

topicdf = pd.DataFrame.from_records([{v:k for v, k in row} for row in proftopics])
topicdf=topicdf.tail(profans.shape[0])
topicdf['prof_id']=profans['professionals_id']
def getTopicForQuery (question):

    temp = question.lower()

    print (temp)



    words = re.findall(r'\w+', temp, flags = re.UNICODE )

    words=[word.lower() for word in words if word.isalpha()]



    important_words = []

    important_words = filter(lambda x: x not in stoplist, words)



    dictionary = corpora.Dictionary.load('dictionary.gensim')



    ques_vec = []

    ques_vec = dictionary.doc2bow(important_words)



    topic_vec = []

    topic_vec = ldamodel[ques_vec]



    word_count_array = np.empty((len(topic_vec), 2), dtype = np.object)

    for i in range(len(topic_vec)):

        word_count_array[i, 0] = topic_vec[i][0]

        word_count_array[i, 1] = topic_vec[i][1]



    idx = np.argsort(word_count_array[:, 1])

    idx = idx[::-1]

    word_count_array = word_count_array[idx]



    return word_count_array
question = 'What is the required education for airline pilots? #airline-industry'

topix_array = getTopicForQuery (question)

topix=[]

for each in topix_array:

    topix.append(each[0])
topicdf=topicdf.sort_values(by=topix, ascending=False)
topicdf.head()['prof_id'].tolist()