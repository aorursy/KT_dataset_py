# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator

from sklearn.model_selection import train_test_split

import warnings

warnings.simplefilter('ignore')

import operator



from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

from sklearn.metrics.pairwise import linear_kernel, cosine_similarity

from sklearn.cluster import KMeans



import re

import string 

from collections import Counter

from nltk.corpus import stopwords

stop = stopwords.words('english')



from plotly.offline import init_notebook_mode, iplot

import plotly.graph_objs as go

import plotly.plotly as py

from plotly import tools

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator



import warnings

warnings.simplefilter('ignore')



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
from nltk.corpus import stopwords

stop = stopwords.words('english')
emails = pd.read_csv('../input/emails.csv')

questions = pd.read_csv('../input/questions.csv')

professionals = pd.read_csv('../input/professionals.csv')

comments = pd.read_csv('../input/comments.csv')

tag_users = pd.read_csv('../input/tag_users.csv')

group_memberships = pd.read_csv('../input/group_memberships.csv')

tags = pd.read_csv('../input/tags.csv')

students = pd.read_csv('../input/students.csv')

groups = pd.read_csv('../input/groups.csv')

tag_questions = pd.read_csv('../input/tag_questions.csv')

matches = pd.read_csv('../input/matches.csv')

answers = pd.read_csv('../input/answers.csv')

school_memberships = pd.read_csv('../input/school_memberships.csv')

question_score = pd.read_csv('../input/question_scores.csv')

Answer_score = pd.read_csv('../input/answer_scores.csv')
print(students.shape)

students.head(5)
students=students.dropna(subset = ['students_id','students_location', 'students_date_joined'])

print(students.shape)

students.head(5)
students['students_location'].nunique()
top10_regions_stud = students['students_location'].value_counts().head(10)

ax = top10_regions_stud.plot.bar(x=top10_regions_stud.index, y=top10_regions_stud.values)

ax.set_xticklabels(ax.get_xticklabels(), rotation=45, horizontalalignment='right')

ax.set_title('students count by location')

#ax.set_facecolor("black")
date=students['students_date_joined'].str.split('-').values

years=[]

for i in range(0,len(date)):

    years.append(date[i][0])



stud_join_years = pd.Series(years)

stud_join_years_counts=stud_join_years.value_counts()

stud_join_years_counts.sort_index(inplace=True)

ax = stud_join_years_counts.plot.bar(x=stud_join_years_counts.index, y=stud_join_years_counts.values)

ax.set_xticklabels(ax.get_xticklabels(), rotation=45, horizontalalignment='right')

ax.set_title('number of student over years')
print(questions.shape)

questions.head(5)
questions.isnull().any()
print(questions.iloc[0]['questions_body'])

print(questions.iloc[30]['questions_body'])

print(questions.iloc[120]['questions_body'])

print(questions.iloc[300]['questions_body'])
all_quest_titles=questions['questions_title'].str.cat(sep=' ')



wordcloud = WordCloud(width=1500, height=1500).generate(all_quest_titles)



plt.figure(figsize=(20, 7))

plt.imshow(wordcloud, interpolation='bilinear')

plt.axis("off")

plt.show()
question_date=questions['questions_date_added'].str.split('-').values

years_questions=[]

#months_questions=[]

for i in range(0,len(question_date)):

    years_questions.append(question_date[i][0])

    #months_questions.append(question_date[i][1])



    

questions_date_added = pd.Series(years_questions)

questions_date_added_counts=questions_date_added.value_counts()



questions_date_added_counts.sort_index(inplace=True)

ax = questions_date_added_counts.plot.bar(x=questions_date_added_counts.index, y=questions_date_added_counts.values)

ax.set_xticklabels(ax.get_xticklabels(), rotation=45, horizontalalignment='right')

ax.set_title('number of questions over years')

students_questions = pd.merge(students, questions, left_on='students_id',right_on='questions_author_id', how='inner')

students_questions.head()
no_questions_Bystudent=students_questions.groupby('students_id').size()

top10_activeStudents=no_questions_Bystudent.sort_values(ascending=False).head(10)

ax = top10_activeStudents.plot.bar(x=top10_activeStudents.index, y=top10_activeStudents.values)

ax.set_xticklabels(ax.get_xticklabels(), rotation=45, horizontalalignment='right')

ax.set_title('number of questions asked by students')
print(professionals.shape)

professionals.head(5)



professionals.isnull().any()
professionals=professionals.dropna(subset = ['professionals_location', 'professionals_industry', 'professionals_headline'])

print(professionals.shape)

professionals.head(5)
professionals['professionals_industry'].value_counts()
top10_prof_industries = professionals['professionals_industry'].value_counts().head(10)

ax = top10_prof_industries.plot.bar(x=top10_prof_industries.index, y=top10_prof_industries.values)

ax.set_xticklabels(ax.get_xticklabels(), rotation=45, horizontalalignment='right')

ax.set_title('count professionals by Industry')
professionals['professionals_location'].nunique()
top10_prof_regions = professionals['professionals_location'].value_counts().head(10)

ax = top10_prof_regions.plot.bar(x=top10_prof_regions.index, y=top10_prof_regions.values)

ax.set_xticklabels(ax.get_xticklabels(), rotation=45, horizontalalignment='right')

ax.set_title('count professionals by location')
professionals['professionals_headline'].value_counts().sort_values(ascending=False).head(10)

professionals=professionals[professionals['professionals_headline'] != '--']

professionals.shape
top10_profHeadlines = professionals['professionals_headline'].value_counts().sort_values(ascending=False).head(10)

ax = top10_profHeadlines.plot.bar(x=top10_profHeadlines.index, y=top10_profHeadlines.values)

ax.set_xticklabels(ax.get_xticklabels(), rotation=45, horizontalalignment='right')

ax.set_title('count professionals by headlines')
comments.head(5)
top10_number_comments =comments['comments_parent_content_id'].value_counts().head(10)

ax = top10_number_comments.plot.bar(x=top10_number_comments.index, y=top10_number_comments.values)

ax.set_xticklabels(ax.get_xticklabels(), rotation=45, horizontalalignment='right')

ax.set_title('count comments number for questions')

ax.set_xlabel("question id")
comments_profs = pd.merge(comments, professionals, left_on='comments_author_id',right_on='professionals_id',how='inner')

comments_profs                          
top10_profID_inComments = comments_profs.groupby('professionals_id').size().sort_values(ascending=False).head(10)

ax = top10_profID_inComments.plot.bar(x=top10_profID_inComments.index, y=top10_profID_inComments.values)

ax.set_xticklabels(ax.get_xticklabels(), rotation=45, horizontalalignment='right')

ax.set_title('Top10 professionals make comments')
top10_prof_inComments_df = professionals.loc[professionals['professionals_id'].isin(top10_profID_inComments.index)]

top10_prof_inComments_df['professionals_industry']
top10_industry_comments=comments_profs.groupby('professionals_industry').size().sort_values(ascending=False).head(10)

#ax = top10_industry_comments.plot.bar(x=top10_industry_comments.index, y=top10_industry_comments.values)

#print(top10_industry_comments)



prof_in_top10Industry_df = professionals.loc[professionals['professionals_industry'].isin(top10_industry_comments.index)]

top10_prof_comments=prof_in_top10Industry_df.groupby('professionals_industry').size()                                         



#print(top10_prof_comments)



commentsVsProf_inIndustry_df=pd.concat([top10_industry_comments.rename('no.comments'), top10_prof_comments.rename('no.profs')], axis=1 )

ax = commentsVsProf_inIndustry_df.plot.bar()

ax.set_xticklabels(ax.get_xticklabels(), rotation=45, horizontalalignment='right')

ax.set_title('no.comments , no.profs VS Industry')

ax.set_xlabel("Industry name")
print(answers.shape)

answers.head(5)
questions.head(5)
answers.head(5)
questions_Answers = pd.merge(questions, answers, left_on='questions_id',right_on='answers_question_id', how='inner')

questions_Answers.head(5)
questions_Answers['questions_date_added'] = pd.to_datetime(questions_Answers['questions_date_added'])

questions_Answers['answers_date_added'] = pd.to_datetime(questions_Answers['answers_date_added'])

questions_Answers['respond_time'] = (questions_Answers['answers_date_added']-questions_Answers['questions_date_added']).dt.days  

#questions_Answers['respond_time'].head(10)

ax=questions_Answers['respond_time'].plot()

ax.set_xlim([0,2000])

ax.set_title('questions respond time')

ax.set_xlabel("respond time in days")
questions_Answers['number_words_inQuestion'] = questions_Answers['questions_body'].apply(lambda x: len(x.split(' ')))

questions_Answers.loc[(questions_Answers['respond_time'] <= 7), 'week'] = 1

questions_Answers.loc[(questions_Answers['respond_time'] > 7) & (questions_Answers['respond_time'] <= 14), 'week'] = 2

questions_Answers.loc[(questions_Answers['respond_time'] > 14) & (questions_Answers['respond_time'] <= 21), 'week'] = 3

questions_Answers.loc[(questions_Answers['respond_time'] > 21) & (questions_Answers['respond_time'] <= 28), 'week'] = 4

questions_Answers.loc[(questions_Answers['respond_time'] > 28), 'week'] = 5





ax = sns.countplot(x="week", data=questions_Answers)

ax.set_title('respond time duration in week')



print(questions_Answers.groupby('week').size())
corr=questions_Answers.plot.scatter(x='number_words_inQuestion',y='respond_time')

corr.set_title('response time VS number_words_inQuestion')
fast_response_words = questions_Answers[ questions_Answers['week']==1]['number_words_inQuestion']

slow_response_words = questions_Answers[ questions_Answers['week']==5]['number_words_inQuestion']



pal = sns.color_palette()



plt.figure(figsize=(18, 8))

plt.hist(fast_response_words, bins=40, range=[0, 80], color=pal[9], label='fast')

plt.hist(slow_response_words, bins=40, range=[0, 80], color=pal[3], alpha=0.4, label='slow')

plt.legend()

plt.xlabel('Number of words', fontsize=15)

plt.ylabel('Number of responses', fontsize=15)

tags.head(5)
tags.nunique()
tag_users.head(5)
tags_tagUsers = pd.merge(tags, tag_users, left_on='tags_tag_id',right_on='tag_users_tag_id', how='inner')

tags_tagUsers.head(5)
all_tags_followed=tags_tagUsers['tags_tag_name'].str.cat(sep=' ')



wordcloud = WordCloud(width=1500, height=800).generate(all_tags_followed)



plt.figure(figsize=(20, 7))

plt.imshow(wordcloud, interpolation='bilinear')

plt.axis("off")

plt.show()
emails.head(5)
emails.isnull().any()
emails_frequency=emails['emails_frequency_level'].value_counts()

print(emails_frequency)

ax = emails_frequency.plot.bar(x=emails_frequency.index, y=emails_frequency.values)

ax.set_xticklabels(ax.get_xticklabels(), rotation=45, horizontalalignment='right')

ax.set_title('Email frequency level count')

ax.set_xlabel("Email freq.Level")
matches.head(5)
top10_mails = matches.groupby('matches_email_id').size().sort_values(ascending=False).head(10)

ax = top10_mails.plot.bar(x=top10_mails.index, y=top10_mails.values)

ax.set_title('Top10 mails containing questions')
X_train, X_test= train_test_split(questions, test_size=0.05, random_state=42)

questions_Answers = pd.merge(X_train, answers, left_on='questions_id',right_on='answers_question_id', how='inner')

questions_Answers_test = pd.merge(X_test, answers, left_on='questions_id',right_on='answers_question_id', how='inner')

t = pd.merge(tags, tag_questions, left_on='tags_tag_id', right_on='tag_questions_tag_id')

t_test = pd.merge(t,questions_Answers_test,left_on='tag_questions_question_id',right_on='answers_question_id')
k=questions_Answers_test['answers_author_id'].isin(questions_Answers['answers_author_id'])

f=questions_Answers_test[k]

f=f.reset_index()

UQ=f['questions_body'].unique()
def process_text(df, col):

    df[col] = df[col].str.replace('[^\w\s]','') # replacing punctuations

    df[col] = df[col].str.replace('-',' ') # replacing dashes

    df[col] = df[col].str.replace('\d+','') # replacing digits

    df[col] = df[col].str.lower().str.split() # convert all str to lowercase    

    df[col] = df[col].apply(lambda x: [item for item in x if item not in stop]) # remove stopwords    

    df[col] = df[col].apply(' '.join) # convert list to str

    return df
def merging(df1, df2, left, right):

    return df1.merge(df2, how="inner", left_on=left, right_on=right)
def combine_authors(df):

    c = df.groupby('questions_id')['answers_author_id'].apply(list)

    df_z = merging(df, pd.DataFrame(c), 'questions_id', 'questions_id')

    df_z.drop('answers_author_id_x', axis=1, inplace=True)

    df_z['answers_author_id_y'] = df_z['answers_author_id_y'].apply(', '.join)

    df_z.drop_duplicates(inplace=True)

    return df_z
Ques_Ans_sub = questions_Answers[['questions_title', 'questions_body', 'answers_author_id', 'questions_id']].copy()



UniqueQues = combine_authors(Ques_Ans_sub)



Ques_Prof = UniqueQues[['questions_id', 'answers_author_id_y']].copy()



UniqueQues.drop('answers_author_id_y', axis=1, inplace=True)

UniquesQues = process_text(UniqueQues, "questions_title") 

UniqueQues = process_text(UniqueQues, "questions_body")
tf = TfidfVectorizer(analyzer='word',

                         ngram_range=(1,2),

                         min_df=3,

                         max_df=0.9,

                         stop_words='english')

lst=UniqueQues['questions_body']

tfidf_matrix = tf.fit(lst)

tfidfTrans= tfidf_matrix.transform(lst)
def CosSim(QuesBody):

    TargetQues= tfidf_matrix.transform([QuesBody])

    tfidfTrans.shape

    cosine_sim = linear_kernel(TargetQues, tfidfTrans)

    q_titles = UniqueQues['questions_title']

    q_ids = UniqueQues['questions_id']

    indices = pd.Series(UniqueQues.index, UniqueQues['questions_title'])

    return cosine_sim,indices,q_titles,q_ids
def get_recommendations_idx(df):

    sim_scores = list(enumerate(cosine_sim[0]))

    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    sim_scores = sim_scores[0:21]

    q_indices = [i[0] for i in sim_scores]

    scores= [i[1] for i in sim_scores]

    df['scores']=scores

    return q_indices



def get_recommendations(df):

    return q_titles.iloc[get_recommendations_idx(df)]

    

def get_questions_id(df):

    ls=q_ids.iloc[get_recommendations_idx(df)] 

    df['Ques_ID']=ls.tolist()

    return df  
df=pd.DataFrame()

test=f['questions_body'][5]

cosine_sim,indices,q_titles,q_ids=CosSim(test)

get_questions_id(df)
y=[i for i, value in enumerate(questions_Answers_test['questions_body']) if value in test]

ExpectedProf=questions_Answers_test['answers_author_id'][y]

ExpectedProf=ExpectedProf.reset_index()

ExpectedProf
z=[i for i, value in enumerate(questions_Answers['answers_author_id']) if value in ExpectedProf['answers_author_id'][0]]

m=questions_Answers.iloc[z]

temp5=m['answers_question_id'].isin(df['Ques_ID'])

RecomProfs5=m[temp5]

RecomProfs5
ExpectedProf['answers_author_id'][0]
temp=answers['answers_question_id'].isin(df['Ques_ID'])

RecomProfs=answers[temp]

temp2=comments_profs['professionals_id'].isin(RecomProfs['answers_author_id'])

ActiveProfs=comments_profs[temp2]

ActiveProfs=ActiveProfs[['professionals_id', 'comments_date_added' ]].copy()

ActiveProfs['comments_date_added'] =pd.to_datetime(ActiveProfs.comments_date_added)

ActiveProfs.sort_values(by=['comments_date_added'], inplace=True, ascending=False)

ActiveProfs.drop_duplicates('professionals_id',inplace=True)

ActiveProfs['Active']=ActiveProfs['comments_date_added']>'2018-01-01 00:00:00 UTC+0000'
RecomProfsk=pd.merge(RecomProfs,ActiveProfs,how='left',left_on='answers_author_id',right_on='professionals_id')

RecomProfs2=RecomProfsk[['answers_question_id','answers_author_id','Active']][:]

RecomProfs2.nunique()
temp3=answers['answers_author_id'].isin(RecomProfs2['answers_author_id'].unique())

ActiveAns=answers[temp3]

ActiveAns['answers_date_added'] =pd.to_datetime(ActiveAns.answers_date_added)

ActiveAns.sort_values(by=['answers_date_added'], inplace=True, ascending=False)

ActiveAns.drop_duplicates('answers_author_id',inplace=True)

ActiveAns['Active']=ActiveAns['answers_date_added']>'2018-01-01 00:00:00'

ActiveAns=ActiveAns[['answers_author_id','answers_question_id','Active']][:]
RecomProfsx=pd.merge(RecomProfs2,ActiveAns,how='left',left_on='answers_author_id',right_on='answers_author_id')

RecomProfsy=RecomProfsx[['answers_question_id_x','answers_author_id','Active_x','Active_y']][:]

RecomProfsy.fillna(False,inplace=True)

RecomProfsy['Active']=RecomProfsy['Active_x'] | RecomProfsy['Active_y']

RecomProfsy.drop(['Active_x','Active_y'], axis=1, inplace=True)

RecomProfsy.rename(columns={'answers_question_id_x' :'questions_id'},inplace=True)



RecomProfsy
Recommended_Prof=pd.merge(RecomProfsy,df,how='inner',left_on='questions_id',right_on='Ques_ID')

Recommended_Prof.drop_duplicates(inplace=True)

Recommended_Prof['FinalWeight']=(5*Recommended_Prof['scores'])+(2*Recommended_Prof['Active'])

Recommended_Prof2=Recommended_Prof.groupby(['answers_author_id'],as_index=False).max()

Recommended_Prof2.sort_values(by=['FinalWeight'], inplace=True, ascending=False)

ToCollaborative=Recommended_Prof2['answers_author_id'].values

ToCollaborative
def getProfessionalTages(t,id):

    indices1 = [i for i, value in enumerate(t['answers_author_id']) if value == id]

    p = t['tags_tag_name'][indices1]

    comm=np.unique(list(p))

    return comm
#elbow method to chosse the best number of clusters

def elbow(df1):

    distortions = []

    k = 50

    K = []

    while k < 101:

        print(k)

        K.append(k)

        kmeanModel = KMeans(n_clusters=k).fit(df1)

        kmeanModel.fit(df1)

        distortions.append(sum(np.min(cdist(df1, kmeanModel.cluster_centers_, 'euclidean'), axis=1)) / df1.shape[0])

        k += 2



    # # Plot the elbow

    plt.plot(K, distortions, 'bx-')

    plt.xlabel('k')

    plt.ylabel('Distortion')

    plt.title('The Elbow Method showing the optimal k')

    plt.show()

    return
#merge the data

t = pd.merge(tags,tag_questions, left_on='tags_tag_id', right_on='tag_questions_tag_id')

t = pd.merge(t,questions_Answers,left_on='tag_questions_question_id',right_on='answers_question_id')

tags=t['tags_tag_name']



#get features form the question using tfidf vectirization

vectorizer = TfidfVectorizer(max_df=0.5, max_features=100, min_df=2, stop_words='english',use_idf=True)

vec = vectorizer.fit(tags)

x=vec.transform(tags)

df1=x.toarray()
#cluster the data using k means

km = KMeans(n_clusters=62, init='k-means++', max_iter=100, n_init=1).fit(x)

labels = km.labels_.tolist()



#now predict the new professional

ids=ToCollaborative

most_common=[]

for id in range(len(ids)):

    out=getProfessionalTages(t,ids[id])

    out = vec.transform(out)

    out=out.toarray()

    pred=km.predict(out)

    indices = [i for i, value in enumerate(labels) if value in pred]

    #print(indices)

    p=t['answers_author_id'][indices]

    comm=Counter(p)

    n=10

    if len(comm)<10:

        n=len(comm)

    arr=[]

    comm=comm.most_common(n)

    for a,b in comm:

        arr.append(a)

    most_common.append(arr)
most_common
output=dict()

for i in range(len(most_common)):

    for j in range (len(most_common[i])):

        if most_common[i][j] in output:

            value=output.get(str(most_common[i][j]))+1

            key=str(most_common[i][j])

            output.update({key: value})

        else:

            output.setdefault(str(most_common[i][j]), 1)

sorted_output = sorted(output.items(), key=operator.itemgetter(1),reverse=True)

n=10

if len(comm) < 10:

    n = len(sorted_output)
sorted_output[0:n]