import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from datetime import datetime

import seaborn as sb

import os

import re

from wordcloud import WordCloud

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

import plotly

import plotly.graph_objs as go

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.cluster import KMeans



init_notebook_mode(connected=True)

%matplotlib inline 



# load dataset

questions = pd.read_csv(os.path.join(os.getcwd(), '../input//questions.csv')) 

answers = pd.read_csv(os.path.join(os.getcwd(), '../input//answers.csv')) 

students = pd.read_csv(os.path.join(os.getcwd(), "../input//students.csv")) 

professionals = pd.read_csv(os.path.join(os.getcwd(), '../input//professionals.csv')) 

tag_questions = pd.read_csv(os.path.join(os.getcwd(), '../input//tag_questions.csv')) 

tags = pd.read_csv(os.path.join(os.getcwd(), '../input//tags.csv'))  

groups = pd.read_csv(os.path.join(os.getcwd(), '../input//groups.csv'))  

group_memberships = pd.read_csv(os.path.join(os.getcwd(), '../input//group_memberships.csv'))  

school_memberships = pd.read_csv(os.path.join(os.getcwd(), '../input//school_memberships.csv')) 

emails = pd.read_csv(os.path.join(os.getcwd(), '../input//emails.csv')) 

matches = pd.read_csv(os.path.join(os.getcwd(), '../input//matches.csv')) 

comments = pd.read_csv(os.path.join(os.getcwd(), '../input//comments.csv'))

# define some utils methods

def draw_bar(dt, s, head=20, sort=True):

    if sort:

        dt.groupby([s])[s].count().sort_values(ascending=False).head(head).plot(kind='bar', figsize=(15,8))

    else:

        dt.groupby([s])[s].count().head(head).plot(kind='bar', figsize=(15,8))

    

def draw_map(dt, s1, s2, sort=False):

    d1 = dt[s1]

    d2 = dt[s2]

    fig, ax = plt.subplots(figsize=(12,12))

    sb.heatmap(pd.crosstab(d1, d2).head(20), annot=True, ax=ax, fmt='d', linewidths=0.1)

    

def clean_body(raw_html):

    cleanr = re.compile('<.*?>|\\W|\\n|\\r')

    cleantext = re.sub(cleanr, ' ', raw_html)

    cleantext = re.sub('\\s{2,}', ' ', cleantext)

    return cleantext



def draw_cloud(s, n=100, what='str'):

    if what == 'str':

        cloud = WordCloud(width=1440, height=1080, max_words=n).generate(" ".join(s.astype(str)))

    else:

        cloud = WordCloud(width=1440, height=1080, max_words=n).generate_from_frequencies(s)

    plt.figure(figsize=(20, 15))

    plt.imshow(cloud)

    

def draw_plotly(x_1, y_1, x_2, y_2, l_name, r_name, title):

    trace0 = go.Bar(

        x=x_1,

        y=y_1,

        name=l_name

    )

    trace1 = go.Bar(

        x=x_2,

        y=y_2,

        name=r_name

    )



    data = [trace0, trace1]

    layout = {'title': title, 'xaxis': {'tickangle': 45}}



    fig = go.Figure(data=data, layout=layout)

    iplot(fig, show_link=False)
professionals_lower = professionals.copy()
professionals_industry = professionals.groupby('professionals_industry')['professionals_industry'].count().sort_values(ascending=False).head(20)

professionals_lower['professionals_industry'] = professionals_lower['professionals_industry'].str.lower()

professionals_industry_lower = professionals_lower.groupby('professionals_industry')['professionals_industry'].count().sort_values(ascending=False).head(20)

draw_plotly(professionals_industry_lower.index, professionals_industry, professionals_industry_lower.index,

            professionals_industry_lower, 'Professionals industry', 'Professionals industry lowercas', 'Professionals Industry')
professionals_location = professionals.groupby('professionals_location')['professionals_location'].count().sort_values(ascending=False).head(20)

professionals_lower['professionals_location'] = professionals_lower['professionals_location'].str.lower()

professionals_location_lower = professionals_lower.groupby('professionals_location')['professionals_location'].count().sort_values(ascending=False).head(20)

draw_plotly(professionals_location_lower.index, professionals_location, professionals_location_lower.index,

           professionals_location_lower, 'Professionals location', 'Professionals location lowercase', 'Professionals Location')
professionals_headline = professionals.groupby('professionals_headline')['professionals_headline'].count().sort_values(ascending=False).head(20)

professionals_lower['professionals_headline'] = professionals_lower['professionals_headline'].str.lower()

professionals_headline_lower = professionals_lower.groupby('professionals_headline')['professionals_headline'].count().sort_values(ascending=False).head(20)

draw_plotly(professionals_headline_lower.index, professionals_headline, professionals_headline_lower.index,

           professionals_headline_lower, 'Professionals headlite', 'Professionals headline lowercase',

           'Professionals Headline')
professionals['year_joined'] = pd.to_datetime(professionals['professionals_date_joined']).dt.year
professionals['month_joined'] = pd.to_datetime(professionals['professionals_date_joined']).dt.month
professionals.shape
draw_map(professionals, 'year_joined', 'month_joined')
draw_bar(professionals, 'year_joined', sort=False)
students['year_joined'] = pd.to_datetime(students['students_date_joined']).dt.year
students['month_joined'] = pd.to_datetime(students['students_date_joined']).dt.month
draw_map(students, 'year_joined', 'month_joined')
draw_bar(students, 'year_joined', sort=False)
students_location = students.groupby('students_location')['students_location'].count().sort_values(ascending=False).head(20)

students_lower = students.copy()

students_lower['students_location'] = students_lower['students_location'].str.lower()

students_location_lower = students_lower.groupby('students_location')['students_location'].count().sort_values(ascending=False).head(20)

draw_plotly(students_location_lower.index, students_location, students_location_lower.index, students_location_lower,

           'Students location', 'Students location lowercase', 'Students Location')
answers['year_added'] = pd.to_datetime(answers['answers_date_added']).dt.year
answers['month_added'] = pd.to_datetime(answers['answers_date_added']).dt.month
draw_map(answers, 'year_added', 'month_added')
draw_bar(answers, 'year_added', sort=False)
prof_answ = pd.merge(answers, professionals, right_on='professionals_id', left_on='answers_author_id', how='left')
draw_bar(prof_answ, 'answers_author_id')
prof_answ['answers_author_id'].unique().size
professionals['professionals_id'].unique().size
u1 = prof_answ.drop(['answers_id', 'answers_author_id', 'answers_question_id', 'answers_date_added', 'answers_body', 'month_added', 'professionals_date_joined', 'year_added'], axis=1).drop_duplicates()
profs = professionals.groupby('year_joined')[['professionals_id']].count().join(

    u1.groupby('year_joined')[['professionals_id']].count(), lsuffix='_left', rsuffix='_right')



draw_plotly(profs.index, profs.professionals_id_left, profs.index, profs.professionals_id_right, 

            'All professionals', 'Active professionals', 'Professionals')
profs = professionals.groupby('professionals_location')[['professionals_id']].count().nlargest(30, columns=['professionals_id']).join(

    u1.groupby('professionals_location')[['professionals_id']].count().nlargest(30, columns=['professionals_id']), lsuffix='_left', rsuffix='_right')

draw_plotly(profs.index, profs.professionals_id_left, profs.index, profs.professionals_id_right, 'All professionals', 

           'Active professionals', "Top 30. Professional's activity by location")
p = professionals.groupby('professionals_location')[['professionals_id']].count().join(

    u1.groupby('professionals_location')[['professionals_id']].count(), lsuffix='_left', rsuffix='_right')

profs = p[p['professionals_id_right'].isna()].sort_values(by=['professionals_id_left'], ascending=False).head(30)

draw_plotly(profs.index, profs.professionals_id_left, profs.index, profs.professionals_id_right, 'All professionals', 

           'Active professionals', "Top 30 not active professionals by location")
profs = professionals.groupby('professionals_industry')[['professionals_id']].count().nlargest(30, columns=['professionals_id']).join(

    u1.groupby('professionals_industry')[['professionals_id']].count().nlargest(30, columns=['professionals_id']), lsuffix='_left', rsuffix='_right')

draw_plotly(profs.index, profs.professionals_id_left, profs.index, profs.professionals_id_right, 'All professionals', 

           'Active professionals', "Top 30. Professional's activity by industry")
p = professionals.groupby('professionals_industry')[['professionals_id']].count().join(

    u1.groupby('professionals_industry')[['professionals_id']].count(), lsuffix='_left', rsuffix='_right')

profs = p[p['professionals_id_right'].isna()].sort_values(by=['professionals_id_left'], ascending=False).head(30)

draw_plotly(profs.index, profs.professionals_id_left, profs.index, profs.professionals_id_right, 'All professionals', 

           'Active professionals', "Top 30 not active professionals by industry")
profs = professionals.groupby('professionals_headline')[['professionals_id']].count().nlargest(30, columns=['professionals_id']).join(

    u1.groupby('professionals_headline')[['professionals_id']].count().nlargest(30, columns=['professionals_id']), lsuffix='_left', rsuffix='_right')

draw_plotly(profs.index, profs.professionals_id_left, profs.index, profs.professionals_id_right, 'All professionals', 

           'Active professionals', "Top 30. Professional's activity by headline")
p = professionals.groupby('professionals_headline')[['professionals_id']].count().join(

    u1.groupby('professionals_headline')[['professionals_id']].count(), lsuffix='_left', rsuffix='_right')

profs = p[p['professionals_id_right'].isna()].sort_values(by=['professionals_id_left'], ascending=False).head(30)

draw_plotly(profs.index, profs.professionals_id_left, profs.index, profs.professionals_id_right, 'All professionals', 

           'Active professionals', "Top 30 not active professionals by headline")
prof_answ['answers_body'] = prof_answ['answers_body'].apply(lambda x: clean_body(str(x)))

draw_cloud(prof_answ['answers_body'])
draw_bar(questions, 'questions_body')
draw_bar(questions, 'questions_title')
questions.groupby(['questions_body', 'questions_author_id'])['questions_body'].count().sort_values(ascending=False).head(10).T
questions[questions['questions_author_id'] == 'c17fb778ae734737b08f607e75a87460'].sort_values(by='questions_date_added').head(20).T
questions['year_added'] = pd.to_datetime(questions['questions_date_added']).dt.year
questions['month_added'] = pd.to_datetime(questions['questions_date_added']).dt.month
draw_map(questions, 'month_added', 'year_added')
draw_bar(questions, 'year_added', sort=False)
a = answers.groupby('year_added')['year_added'].count().head(20)

q = questions.groupby('year_added')['year_added'].count().head(20)

draw_plotly(a.index, a, q.index, q, 'Answers', 'Questions', 'Questions vs Answers')
draw_cloud(questions['questions_title'])
draw_cloud(questions['questions_body'])
stud_quest = pd.merge(questions, students, right_on='students_id', left_on='questions_author_id', how='inner')
draw_bar(stud_quest, 'students_id', 40)
tags.shape
tags['tags_tag_name'].unique().size
tag_questions.shape
tags_with_name = pd.merge(tags, tag_questions, left_on='tags_tag_id', right_on='tag_questions_tag_id', how='inner')
tags_with_name['tags_tag_name'].unique().size
draw_bar(tags_with_name, 'tags_tag_name')
draw_cloud(tags_with_name['tags_tag_name'])
unused_tags_id = set(tags['tags_tag_id'].tolist()) - set(tag_questions['tag_questions_tag_id'].tolist())
len(unused_tags_id)
unused_tags = tags[tags['tags_tag_id'].isin(unused_tags_id)]

draw_cloud(unused_tags['tags_tag_name'])
tags[tags['tags_tag_name'].str.contains('college', regex=False, na=False)].head(20)
unused_tags[unused_tags['tags_tag_name'].str.contains('college', regex=False, na=False)].head(20)
answ_quest = pd.merge(answers, questions, left_on='answers_question_id', right_on='questions_id', how='outer', suffixes=('_answ', '_quest'))

answ_quest_tags = pd.merge(answ_quest, tags_with_name, left_on='questions_id', right_on='tag_questions_question_id', how='outer')
draw_bar(answ_quest, 'questions_id')
without_answer = answ_quest[answ_quest['answers_id'].isna()]
draw_map(without_answer, 'month_added_quest', 'year_added_quest')
unanswered = answ_quest_tags[answ_quest_tags['answers_id'].isna()]
draw_cloud(unanswered['tags_tag_name'])
unansw_tag_list = unanswered['tags_tag_name'].tolist()

freq_unansw_tags = {str(x): unansw_tag_list.count(x) for x in unansw_tag_list}

draw_cloud(freq_unansw_tags, what='freq')


draw_cloud(unanswered['questions_title'])
draw_cloud(unanswered['questions_body'])
groups['groups_group_type'].unique()
draw_bar(groups, 'groups_group_type')
draw_bar(group_memberships, 'group_memberships_user_id')
prof_group = pd.merge(group_memberships, professionals, left_on='group_memberships_user_id', right_on='professionals_id')
stud_group = pd.merge(group_memberships, students, left_on='group_memberships_user_id', right_on='students_id')
draw_bar(prof_group, 'year_joined')
draw_bar(stud_group, 'year_joined')
group_answ = pd.merge(prof_group, answers, left_on='professionals_id', right_on='answers_author_id', how='inner')

prof_group_unansw = set(prof_group['professionals_id'].tolist()) - set(group_answ['answers_author_id'].tolist())

groups_members_and_type = pd.merge(groups, group_memberships, left_on='groups_id', right_on='group_memberships_group_id')
draw_bar(groups_members_and_type[groups_members_and_type['group_memberships_user_id'].isin(prof_group_unansw)], 'groups_group_type')
draw_bar(school_memberships, 'school_memberships_school_id')
prof_school = pd.merge(school_memberships, professionals, left_on='school_memberships_user_id', right_on='professionals_id')
stud_school = pd.merge(school_memberships, students, left_on='school_memberships_user_id', right_on='students_id')
prof_school.shape
stud_school.shape
school_answ = pd.merge(school_memberships, answers, left_on='school_memberships_user_id', right_on='answers_author_id', how='inner')

school_answ.shape
prof_school_unansw = school_answ[school_answ['answers_author_id'].isna()]

prof_school_unansw.shape
draw_bar(school_answ, 'school_memberships_school_id')
emails.shape
emails['year_sent'] = pd.to_datetime(emails['emails_date_sent']).dt.year
emails['month_sent'] = pd.to_datetime(emails['emails_date_sent']).dt.month
draw_map(emails, 'month_sent', 'year_sent')
draw_bar(emails, 'year_sent')
emails['emails_frequency_level'].unique()
draw_bar(emails, 'emails_frequency_level')
draw_bar(emails, 'emails_recipient_id')
top_5 = emails.groupby('emails_recipient_id')['emails_recipient_id'].count().sort_values(ascending=False).head(5).index
professionals[professionals['professionals_id'].isin(top_5)].head()
answers[answers['answers_author_id'].isin(top_5)].groupby('answers_author_id')['answers_author_id'].count()
draw_map(questions[questions['questions_id'].isin(matches['matches_question_id'].unique())], 'month_added', 'year_added')
draw_bar(matches, 'matches_email_id')
draw_bar(matches, 'matches_question_id')
questions.shape
matches['matches_question_id'].unique().size
quest_tags = pd.merge(questions, tags_with_name, left_on='questions_id', right_on='tag_questions_question_id', how='inner')

not_sent_ids = set(quest_tags['questions_id'].tolist()) - set(matches['matches_question_id'].tolist())

not_sent_quest = quest_tags[quest_tags['questions_id'].isin(not_sent_ids)].dropna()
not_sent_quest.head(10).T


not_sent_quest.groupby('tags_tag_name')['tags_tag_name'].count().sort_values(ascending=False).head(20)
q_copy = questions.copy()

a_copy = answers.copy()

p_copy = professionals.copy()

q_copy.drop(['questions_date_added', 'questions_author_id', 'questions_title'], axis=1, inplace=True)

a_copy.drop(['answers_date_added', 'answers_body'], axis=1, inplace=True)

p_copy.drop(['professionals_location', 'professionals_industry', 'professionals_headline', 'professionals_date_joined'], axis=1, inplace=True)

a_p = pd.merge(a_copy, p_copy, left_on='answers_author_id', right_on='professionals_id')

a_p.drop(['answers_id', 'answers_author_id'], axis=1, inplace=True)

t = pd.merge(tags, tag_questions, left_on='tags_tag_id', right_on='tag_questions_tag_id')

t.drop(['tags_tag_id', 'tag_questions_tag_id'], axis=1, inplace=True)
stop_tags = ['college', 'career', 'college-major ', 'career-counseling', 'scholarships', 'jobs', 'college-advice', 

             'double-major', 'chef', 'college-minor', 'college-applications', 'college-student', 'school', 

             'college-admissions', 'career-choice', 'university', 'job', 'college-major', 'any', 'student', 

             'professional', 'graduate-school', 'career-path', 'career-paths', 'college-majors', 'career-details', 

             'work', 'college-bound', 'success', 'studying', 'first-job', 'life', 'classes', 'resume', 'job-search']
most_used_tags = t[~t['tags_tag_name'].isin(stop_tags)].groupby('tags_tag_name')['tags_tag_name'].count().nlargest(200).index
a_q_p = pd.merge(a_p, questions, left_on='answers_question_id', right_on='questions_id')

a_q_p.drop(['answers_question_id'], axis=1, inplace=True)

a_q_p_t = pd.merge(t, a_q_p, left_on='tag_questions_question_id', right_on='questions_id', how='inner')

a_q_p_t.drop(['tag_questions_question_id', 'questions_id'], axis=1, inplace=True)

filtered = a_q_p_t[a_q_p_t['tags_tag_name'].isin(most_used_tags)]

filtered = filtered.copy()

filtered.loc[:, 'questions_body'] = filtered['questions_body'].map(clean_body)

filtered = filtered.fillna('')
labels = filtered['tags_tag_name']

data = filtered['questions_body']

n_clusters = np.unique(labels).shape[0]
test_data_1 = data.iloc[11]

test_tag_1 = labels.iloc[11]

test_data_2 = data.iloc[40]

test_tag_2 = labels.iloc[40]

print(test_data_1)

print(test_tag_1)

print()

print(test_data_2)

print(test_tag_2)
vectorizer = TfidfVectorizer(max_df=0.5, max_features=10000, min_df=2, stop_words='english',use_idf=True)

X = vectorizer.fit_transform(data)
km = KMeans(n_clusters=n_clusters, init='k-means++', max_iter=100, n_init=1)
print("Clustering sparse data with %s" % km)

%time km.fit(X)
clusters = km.labels_.tolist()

quest = { 'tags': filtered['tags_tag_name'].tolist(), 'professionals_id': filtered['professionals_id'].tolist(), 'question': filtered['questions_body'].tolist(), 'cluster': clusters }

frame = pd.DataFrame(quest, index = [clusters] , columns = ['tags', 'professionals_id', 'question', 'cluster'])
result = dict()



for i in range(n_clusters):

    tags = [tag for tag in frame.loc[i]['tags'].unique()]

    profs = [prof for prof in frame.loc[i]['professionals_id'].unique()]

    quests = [quest for quest in frame.loc[i]['question'].unique()]

        

    if test_data_1 in quests:

        result['test_data_1'] = test_data_1

        result['test_tag_1'] = test_tag_1

        result['tags_1'] = tags

        result['profs_1'] = profs

    if test_data_2 in quests:

        result['test_data_2'] = test_data_2

        result['test_tag_2'] = test_tag_2

        result['tags_2'] = tags

        result['profs_2'] = profs
print(result.get('test_data_1'))

print()

print(result.get('test_tag_1'))

print(len(result.get('profs_1')))
a_q_p_t[a_q_p_t['professionals_id'].isin(result.get('profs_1'))].groupby('questions_body')['questions_body'].count().sort_values(ascending=False).head(10)
print(result.get('test_data_2'))

print()

print(result.get('test_tag_2'))

print(len(result.get('profs_2')))
a_q_p_t[a_q_p_t['professionals_id'].isin(result.get('profs_2'))].groupby('questions_body')['questions_body'].count().sort_values(ascending=False).head(10)
print(len(result.get('profs_1')))

print(result.get('profs_1'))

print()

print(len(result.get('profs_2')))

print(result.get('profs_2'))