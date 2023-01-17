import sys

import warnings

if not sys.warnoptions:

    warnings.simplefilter("ignore")

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

import gensim

from nltk.tokenize import word_tokenize

import dateutil.parser

from datetime import datetime

import matplotlib.pyplot as plt

pd.set_option('display.max_colwidth', -1)

%matplotlib inline

from matplotlib.pyplot import figure

from matplotlib_venn import venn2, venn2_circles

from matplotlib_venn import venn3, venn3_circles
print(os.listdir("../input"))

pro= pd.read_csv('../input/professionals.csv')

stu=pd.read_csv("../input/students.csv")



qs=pd.read_csv("../input/questions.csv")

ans= pd.read_csv('../input/answers.csv')

com= pd.read_csv('../input/comments.csv')

email= pd.read_csv('../input/emails.csv')

match= pd.read_csv('../input/matches.csv')



school_mem= pd.read_csv('../input/school_memberships.csv')



tags= pd.read_csv('../input/tags.csv')

qs_tags=pd.read_csv('../input/tag_questions.csv')

user_tag=pd.read_csv('../input/tag_users.csv')
print("Number of students registered:",stu['students_id'].count())

print("Number of professional registered:",pro['professionals_id'].count())
ans_stu= pd.merge(ans,stu, left_on='answers_author_id',right_on='students_id')

ans_pro= pd.merge(ans,pro, left_on='answers_author_id',right_on='professionals_id')



com_stu= pd.merge(com,stu, left_on='comments_author_id',right_on='students_id')

com_pro= pd.merge(com,pro, left_on='comments_author_id',right_on='professionals_id')



active_pro= set(ans_pro['answers_author_id']).union(set(com_pro['comments_author_id']))

active_stu= set(ans_stu['answers_author_id']).union(set(com_stu['comments_author_id']))



figure(figsize=(20,20))

plt.subplot(1, 2, 1)

venn3([set(ans_pro['answers_author_id']), set(pro['professionals_id']),set(com_pro['comments_author_id'])],

      set_labels = ('Answered','PROFESSIONALS','Commented'))



plt.subplot(1, 2, 2)

venn3([set(active_stu), set(stu['students_id']), set(qs['questions_author_id'])],

      set_labels = ('Commented/Answered','STUDENTS','Question Posted' ))



plt.tight_layout()

plt.show()

a_s=pd.read_csv('../input/answer_scores.csv')

an_s= pd.merge(ans,a_s,left_on='answers_id',right_on='id')



ans_score_tab=an_s.pivot_table(values='score',index='answers_author_id',aggfunc='sum')

ans_score_tab.index.names=['user_id']



ans_score_tab.columns=['Hearts Earned']

ans_count_tab=ans.pivot_table(values='answers_id',index='answers_author_id',aggfunc='count')

ans_count_tab.index.names=['user_id']

ans_count_tab.columns=['Questions Answered']



score_tab=ans_score_tab.join(ans_count_tab,how='outer')

score_tab=score_tab.replace(np.NaN,0)

score_tab['Total_score']=score_tab['Hearts Earned']+score_tab['Questions Answered']

score_tab=score_tab.sort_values('Total_score',ascending=False)

print("Professionals' Score Chart")

score_tab.head()
print("Popularity Chart")

ans_score_tab.sort_values('Hearts Earned',ascending=False).head()
print("Activity Chart")

ans_count_tab.sort_values('Questions Answered',ascending=False).head()
print("Number of questions asked:",qs['questions_id'].count())

print("Number of answers given:",ans['answers_id'].count())
print("Number of tags:",tags['tags_tag_id'].count())

print("Number of comments made:",com['comments_id'].count())

print("Number of emails sent:",email['emails_id'].count())
figure(figsize=(20,20))

plt.subplot(1, 2, 1)

venn3([set(ans['answers_id']), set(com['comments_parent_content_id']),set(qs['questions_id'])],

      set_labels = ('ANSWERS','COMMENTS','QUESTIONS' ))



plt.tight_layout()

plt.show()

def to_date(val):

    return datetime.strptime( str(val),'%Y-%m-%d %H:%M:%S UTC+0000')

def to_year(val):

    return val.strftime('%Y')

def to_yr_mon(val):

    return val.strftime('%Y-%m')

    

#Question Timestamp Processing    

qs['ts']=qs.apply(lambda x: to_date(x['questions_date_added']),axis=1)

qs['year']=qs.apply(lambda x: to_year(x['ts']),axis=1)

qs['yr_mon']=qs.apply(lambda x: to_yr_mon(x['ts']),axis=1)



#Answer Timestamp Processing    

ans['ts']=ans.apply(lambda x: to_date(x['answers_date_added']),axis=1)

ans['year']=ans.apply(lambda x: to_year(x['ts']),axis=1)



#Question ids that got answered at some point

answered_id= ans['answers_question_id'].unique()



#Questions which never got answered

n_ans=pd.DataFrame(qs.loc[~qs['questions_id'].isin(answered_id)])

n_ans=n_ans.reset_index(drop=True)

n_ans['ts']=n_ans.apply(lambda x: to_date(x['questions_date_added']),axis=1)

n_ans['y_m']=n_ans.apply(lambda x: to_yr_mon(x['ts']),axis=1)

n_ans['yr']=n_ans.apply(lambda x: to_year(x['ts']),axis=1)



#Questions that got answered at some point

y_ans=pd.DataFrame(qs.loc[qs['questions_id'].isin(answered_id)])



#Time taken for replies

ans_gap= pd.merge(ans,y_ans,left_on='answers_question_id',right_on='questions_id')

ans_gap['time_to_answer']= ans_gap.apply(lambda x: (x['ts_x']-x['ts_y']).days ,axis=1 )



#Reply gap tabulated

tab_gap= ans_gap.groupby('time_to_answer').count()[['answers_id']]



#Questions raised by yr-mon

tab_q= qs.groupby('yr_mon').count()[['questions_id']]

tab_q.sort_index()



#Answers given by yr-mon

tab_a= n_ans.groupby('y_m').count()[['questions_id']]



#Answers by year

tab_a_yr = n_ans.groupby('yr').count()[['questions_id']]

#Questions by year

tab_q_yr= qs.groupby('year').count()[['questions_id']]



#Response rates

tab_response_rate=tab_a_yr.join(tab_q_yr,how='outer', lsuffix='_unanswered', rsuffix='_posted')

tab_response_rate=tab_response_rate.fillna(0)

tab_response_rate['response rate']=round(100- ((tab_response_rate['questions_id_unanswered']/tab_response_rate['questions_id_posted'])*100),2)

tab_response_rate.columns=['Questions Unanswered','Questions Posted','Response Rate']

print("RESPONSE RATES OVER THE YEARS")

tab_response_rate
#Professionals Timestamp Processing    

pro['ts']=pro.apply(lambda x: to_date(x['professionals_date_joined']),axis=1)

pro['year']=pro.apply(lambda x: to_year(x['ts']),axis=1)

tab_pro= pro.groupby('year').count()[['professionals_id']]

tab_pro.columns=['Professionals joined each year']

tab_pro['Total Professionals']= tab_pro['Professionals joined each year'].cumsum()



#Students Timestamp Processing    

stu['ts']=stu.apply(lambda x: to_date(x['students_date_joined']),axis=1)

stu['year']=stu.apply(lambda x: to_year(x['ts']),axis=1)

tab_stu= stu.groupby('year').count()[['students_id']]

tab_stu.columns=['Students joined each year']

tab_stu['Total Students']= tab_stu['Students joined each year'].cumsum()

tab_pro_stu= tab_pro.join(tab_stu,how='inner')



print("Size of the community over the years")

tab_pro_stu[:-1]
tab_pro_stu[:-1].plot()
#Reply gap tabulated

tab_gap= ans_gap.groupby(['time_to_answer']).count()[['answers_id']]

tab_gap['per']=round((tab_gap['answers_id']/tab_gap['answers_id'].sum())*100,2)

print("TIME TO RESPOND TO QUESTIONS")

tab_gap[0:10]
email['ts']= email.apply(lambda x: to_date(x['emails_date_sent']),axis=1)

email['year']= email.apply(lambda x: to_year(x['ts']),axis=1)



emailed_qs = pd.merge(email,match,left_on='emails_id',right_on='matches_email_id')

emailed_qs_ans=pd.merge(emailed_qs,ans,left_on=['emails_recipient_id','matches_question_id'],right_on=['answers_author_id','answers_question_id'],how='inner')



sent     = emailed_qs.groupby(['emails_recipient_id','year']).nunique()[['matches_question_id']] 

responded= emailed_qs_ans.groupby(['emails_recipient_id','year_x']).nunique()[['answers_id']]



sent=sent.reset_index(level=['year'])

responded=responded.reset_index(level=['year_x'])



sent_responded= pd.merge(sent,responded,how='left',left_on=['emails_recipient_id','year'],right_on=['emails_recipient_id','year_x'])

sent_responded=sent_responded.drop('year_x',axis=1)

sent_responded['rate']= np.array((sent_responded['answers_id'] / sent_responded['matches_question_id'])*100)
venn3([set(qs_tags['tag_questions_tag_id']), set(user_tag['tag_users_tag_id']),set(tags['tags_tag_id'])],

      set_labels = ('Question_tags','User_tags','TAGS'))
figure(figsize=(5,5))

venn3([set(user_tag['tag_users_user_id']),set(ans_pro['answers_author_id']),set(pro['professionals_id'])],

      set_labels = ('Tagged Users','Professionals Answered','Professionals'))
qs.head(2)
qs_tags.sort_values('tag_questions_question_id')[:10]
qs_tagnames= pd.merge(qs_tags,tags,left_on='tag_questions_tag_id',right_on='tags_tag_id')

qs_tagnames=qs_tagnames.drop(['tags_tag_id','tag_questions_tag_id'],axis=1)

print(qs_tagnames.sort_values('tag_questions_question_id')[:10])

qs_tag_pivot=qs_tagnames.pivot_table(index='tag_questions_question_id',values='tags_tag_name',aggfunc=lambda x: " ".join(x))

qs_tag_pivot['tag_questions_question_id']=qs_tag_pivot.index

print("\nNumber of questions asked:",qs['questions_id'].count())

print("Number of questions with tags:",len(qs_tag_pivot))

qs_tag_pivot=qs_tag_pivot.reset_index(drop=True)

print("\n",qs_tag_pivot.head())
print("Example:\nQuestion id-", qs_tag_pivot.iloc(0)[0]['tag_questions_question_id'],

      ":\n\n",qs.loc[qs['questions_id']==qs_tag_pivot.iloc(0)[0]['tag_questions_question_id']]['questions_body'],

      "\n\n*************************************************************************\nTag string:",

      qs_tag_pivot.iloc(0)[0]['tags_tag_name'])
qs_with_tags=pd.merge(qs,qs_tag_pivot,left_on='questions_id',right_on='tag_questions_question_id')

print("Number of questions with tags:",len(qs_with_tags))

qs_with_tags.head(2)
raw_documents=qs_with_tags['questions_title']+qs_with_tags['questions_body']+qs_with_tags['tags_tag_name']

raw_documents.head()
print("Number of Questions:",len(raw_documents))

print("Tokenizing data...")

gen_docs = [[w.lower() for w in word_tokenize(text)] 

            for text in raw_documents]

print("Creating dictionary...")

dictionary = gensim.corpora.Dictionary(gen_docs)

print("Creating Document-Term Matrix...")

corpus = [dictionary.doc2bow(gen_doc) for gen_doc in gen_docs]

print("Creating TF-IDF Model...")

tf_idf = gensim.models.TfidfModel(corpus)

print("Creating Similarity Checker...")

similar_qs = gensim.similarities.Similarity("",tf_idf[corpus],num_features=len(dictionary))

print("Processing Completed!")
Query='Can I become data scientist without studying at university?#technology #data-science'

Query
query_doc = [w.lower() for w in word_tokenize(Query)]

query_doc_bow = dictionary.doc2bow(query_doc)

query_doc_tf_idf = tf_idf[query_doc_bow]
q_sim=similar_qs[query_doc_tf_idf]
sim_threshold=0.10
qs_with_tags['Similarity']=q_sim

ques=qs_with_tags[qs_with_tags['Similarity']>=sim_threshold]

ques=ques.sort_values('Similarity',ascending=False)

ques.head()
qlist=ques['questions_id']

qlist_ans=ans[ans['answers_question_id'].isin(qlist)]

prof_answered=set(qlist_ans['answers_author_id'])

#print(prof_answered)

solution1= pro[pro['professionals_id'].isin(prof_answered)]

solution1.head()

print("Number of professionals registered:",len(pro['professionals_id']))

print("Number of users who have answered:",len(ans['answers_author_id'].unique()))

ans_pro=pro[pro['professionals_id'].isin(ans['answers_author_id'])]

print("Number of professionals who have answered:",len(ans_pro))

ans_stu=stu[stu['students_id'].isin(ans['answers_author_id'])]

print("Number of students who have answered:",len(ans_stu))

print("\n***PROFESSIONALS IGNORED VIA THIS SOLUTION***")

print("Number of professionals who haven't answered yet:",len(set(pro['professionals_id']))-len(ans_pro))
print("From the numbers, it is clear that users who identify themselves as neither professionals nor students have answered to questions.\nHow big is this population?\n")

u= set(ans['answers_author_id'])

s= set(stu['students_id'])

p= set(pro['professionals_id'])

st_ansrd= u.intersection(s)

pr_ansrd= u.intersection(p)

all_ansrd= st_ansrd.union(pr_ansrd)

unknwn= u.difference(all_ansrd)

print("Unknown users: ",len(unknwn))
user_tag_exp=pd.merge(tags,user_tag,left_on='tags_tag_id',right_on='tag_users_tag_id')

user_tag_exp=user_tag_exp.drop(['tags_tag_id','tag_users_tag_id'],axis=1)

user_tag_exp.sort_values('tag_users_user_id')[:10]
tag_pivot=user_tag_exp.pivot_table(values='tags_tag_name',index='tag_users_user_id',aggfunc=lambda x: " ".join(x))

tag_pivot['tag_users_user_id']=tag_pivot.index

print("Number of all users with tags:",len(tag_pivot))

tag_pivot=tag_pivot.reset_index(drop=True)

tag_pivot.head()
pro_tagstring= tag_pivot[tag_pivot['tag_users_user_id'].isin(pro['professionals_id'])]

print("Number of professionals with tags:",len(pro_tagstring))
raw_tags=pro_tagstring['tags_tag_name']

print("Tag string table of professionals:")

raw_tags.head()

print("Number of Tags:",len(raw_tags))

print("Tokenizing data...")

gen_docs = [[w.lower() for w in word_tokenize(text)] 

            for text in raw_tags]

print("Creating dictionary...")

dictionary = gensim.corpora.Dictionary(gen_docs)

print("Creating Document-Term Matrix...")

corpus = [dictionary.doc2bow(gen_doc) for gen_doc in gen_docs]

print("Creating TF-IDF Model...")

tf_idf = gensim.models.TfidfModel(corpus)

print("Creating Similarity Checker...")

sims = gensim.similarities.Similarity("",tf_idf[corpus],num_features=len(dictionary))

print("Processing Completed!")



Query='Can I become data scientist without studying at university?#technology #data-science'

print("\nQuestion posted:",Query)

'''

query_doc = [w.lower() for w in word_tokenize(Query)]

query_doc_bow = dictionary.doc2bow(query_doc)

query_doc_tf_idf = tf_idf[query_doc_bow]



sim=sims[query_doc_tf_idf]



sim_threshold=0.10



pro_tagstring['sim']=sim

prof_tag=pro_tagstring[pro_tagstring['sim']>=sim_threshold]

prof_tag=prof_tag.sort_values('sim',ascending=False)

prof_tag.head()



prof_list=prof_tag['tag_users_user_id']

solution2= pro[pro['professionals_id'].isin(prof_list)]

solution2.head()

'''
query_doc = [w.lower() for w in word_tokenize(Query)]

query_doc_bow = dictionary.doc2bow(query_doc)

query_doc_tf_idf = tf_idf[query_doc_bow]
sim=sims[query_doc_tf_idf]
sim_threshold=0.10
pro_tagstring['sim']=sim

prof_tag=pro_tagstring[pro_tagstring['sim']>=sim_threshold]

prof_tag=prof_tag.sort_values('sim',ascending=False)

prof_tag.head()
prof_list=prof_tag['tag_users_user_id']

solution2= pro[pro['professionals_id'].isin(prof_list)]

solution2.head()
print("Number of users with tags:",len(tag_pivot))

print("\nNumber of professionals registered:",len(pro['professionals_id']))



print("Number of professionals with tags:",len(pro_tagstring))



print("\n***PROFESSIONALS IGNORED VIA THIS SOLUTION***")

print("Number of professionals without any tags:",len(set(pro['professionals_id']))-len(pro_tagstring))
answered_pro= set(ans_pro['professionals_id'])

tagged_pro= set(pro_tagstring['tag_users_user_id'])

A1= len(answered_pro.difference(tagged_pro))

B1= len(tagged_pro.difference(answered_pro))

AnB= len(answered_pro.intersection(tagged_pro))

print("Number of professionals ignored via both methods:",len(pro['professionals_id'])-(A1+AnB+B1))
com_stu= pd.merge(com,stu, left_on='comments_author_id',right_on='students_id')

com_stu_q= pd.merge(com_stu,qs, left_on='comments_parent_content_id',right_on='questions_id')

com_stu_a= pd.merge(com_stu,ans, left_on='comments_parent_content_id',right_on='answers_id')



com_pro= pd.merge(com,pro, left_on='comments_author_id',right_on='professionals_id')

com_pro_q= pd.merge(com_pro,qs, left_on='comments_parent_content_id',right_on='questions_id')

com_pro_a= pd.merge(com_pro,ans, left_on='comments_parent_content_id',right_on='answers_id')

print("Number of comments posted:",len(com),

      "\n\nNumber of comments made by professionals:",len(com_pro),

      "\nNumber of comments made by students:",len(com_stu),

      "\n\nNumber of questions commented by professionals:",len(com_pro_q['comments_parent_content_id'].unique()),

      "\nNumber of answers commented by professionals:",len(com_pro_a['comments_parent_content_id'].unique()),

      "\n\nNumber of questions commented by students:",len(com_stu_q['comments_parent_content_id'].unique()),

      "\nNumber of answers commented by students:",len(com_stu_a['comments_parent_content_id'].unique())           

     )

      

      

      
from matplotlib.pyplot import figure

figure(figsize=(15,10))

plt.subplot(1, 2, 1)

venn3([set(com_pro['comments_author_id']), set(pro['professionals_id']),set(ques['questions_author_id'])],

      set_labels = ('Commented','PROFESSIONALS', 'Questions Asked'))



plt.subplot(1, 2, 2)

#figure(figsize=(8,8))

venn3([set(com_stu['comments_author_id']), set(stu['students_id']),set(ques['questions_author_id'])],

      set_labels = ('Commented','STUDENTS', 'Questions Asked'))



plt.tight_layout()

plt.show()
