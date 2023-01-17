# Imports



import numpy as np  # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

pd.set_option('max_colwidth', 200)





from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.decomposition import TruncatedSVD

from sklearn.metrics.pairwise import cosine_similarity



from plotly.offline import iplot, init_notebook_mode

init_notebook_mode(connected=True)



import nltk

from nltk.stem import WordNetLemmatizer

from nltk.corpus import stopwords



import re

import string

import math

import random

from random import choice, choices

import time



import gc





from IPython.display import display



import warnings  

warnings.filterwarnings('ignore')



# Professionals Import



professionals = pd.read_csv('../input/professionals.csv', index_col='professionals_id')

professionals = professionals.rename(columns={'professionals_location': 'location', 'professionals_industry': 'industry', 'professionals_headline': 'headline', 'professionals_date_joined': 'date_joined'})

professionals['headline'] = professionals['headline'].fillna('')

professionals['industry'] = professionals['industry'].fillna('')



# Students Import



students = pd.read_csv('../input/students.csv', index_col='students_id')

students = students.rename(columns={'students_location': 'location', 'students_date_joined': 'date_joined'})



# Questions Import

questions = pd.read_csv('../input/questions.csv', index_col='questions_id', parse_dates=['questions_date_added'], infer_datetime_format=True)

questions = questions.rename(columns={'questions_author_id': 'author_id', 'questions_date_added': 'date_added', 'questions_title': 'title', 'questions_body': 'body', 'questions_processed':'processed'})



# Answers Import

answers = pd.read_csv('../input/answers.csv', index_col='answers_id', parse_dates=['answers_date_added'], infer_datetime_format=True)

answers = answers.rename(columns={'answers_author_id':'author_id', 'answers_question_id': 'question_id', 'answers_date_added': 'date_added', 'answers_body': 'body'})



# Tags Import

tags = pd.read_csv('../input/tags.csv',)

tags = tags.set_index('tags_tag_id')

tags = tags.rename(columns={'tags_tag_name': 'name'})



# Comments Import

comments = pd.read_csv('../input/comments.csv', index_col='comments_id')

comments = comments.rename(columns={'comments_author_id': 'author_id', 'comments_parent_content_id': 'parent_content_id', 'comments_date_added': 'date_added', 'comments_body': 'body' })





# School Memberships

school_memberships = pd.read_csv('../input/school_memberships.csv')

school_memberships = school_memberships.rename(columns={'school_memberships_school_id': 'school_id', 'school_memberships_user_id': 'user_id'})



# Groups Memberships

group_memberships = pd.read_csv('../input/group_memberships.csv')

group_memberships = group_memberships.rename(columns={'group_memberships_group_id': 'group_id', 'group_memberships_user_id': 'user_id'})



# Emails

emails = pd.read_csv('../input/emails.csv')

emails = emails.set_index('emails_id')

emails = emails.rename(columns={'emails_recipient_id':'recipient_id', 'emails_date_sent': 'date_sent', 'emails_frequency_level': 'frequency_level'})



#####################################################

print('Important numbers:')

print('\nThere are:')

print(f'- {len(students)} Students.', end="\t")

print(f'- {len(professionals)} Professionals.')

print(f'- {len(questions)} Questions.', end="\t")

print(f'- {len(answers)} Answers.')

print(f'- {len(tags)} Tags.', end="\t\t")

print(f'- {len(comments)} Comments.')

print(f'- {school_memberships["school_id"].nunique()} Schools.', end="\t\t")

print(f'- {len(pd.read_csv("../input/groups.csv"))} Groups.')

print(f'- {len(emails)} Emails were sent.')

#####################################################



# Questions-related stats

tag_questions = pd.read_csv('../input/tag_questions.csv',)

tag_questions = tag_questions.rename(columns={'tag_questions_tag_id': 'tag_id', 'tag_questions_question_id': 'question_id'})

count_question_tags = tag_questions.groupby('question_id').count().rename(columns={'tag_id': 'count_tags'}).sort_values('count_tags', ascending=False)

print('\nInteresting statistics: ')

print(f'- {(answers["question_id"].nunique()/len(questions))*100:.2f} % of the questions have at least 1 answer.')

print(f'\n- {(len(count_question_tags)/len(questions))*100:.2f}% of questions are tagged by at least {count_question_tags["count_tags"].tail(1).values[0]} tag.')

print(f'- Mean of tags per question: {count_question_tags["count_tags"].mean():.2f} tags per question.')



tag_users = pd.read_csv('../input/tag_users.csv',)

tag_users = tag_users.rename(columns={'tag_users_tag_id': 'tag_id', 'tag_users_user_id': 'user_id'})

users_who_follow_tags = list(tag_users['user_id'].unique())

nbr_pros_tags = len(professionals[professionals.index.isin(users_who_follow_tags)])

nbr_students_tags = len(students[students.index.isin(users_who_follow_tags)])

print(f'\n- {(nbr_pros_tags / len(professionals))*100:.2f} % of the professionals follow at least 1 Tag ({nbr_pros_tags}).')

print(f'- {(nbr_students_tags / len(students))*100:.2f} % of the students follow at least 1 Tag ({nbr_students_tags}).')



question_scores = pd.read_csv('../input/question_scores.csv')

nbr_questions_with_hearts = question_scores[question_scores['score'] > 0]['id'].nunique()

print(f'\n- {(nbr_questions_with_hearts/len(questions))*100:.2f} % of questions were upvoted ({nbr_questions_with_hearts}).')



answer_scores = pd.read_csv('../input/answer_scores.csv')

nbr_answers_with_hearts = answer_scores[answer_scores['score'] > 0]['id'].nunique()

print(f'- {(nbr_answers_with_hearts/len(questions))*100:.2f} % of answers were upvoted ({nbr_answers_with_hearts}).')





# School/Group Related Stats



def is_student(user_id):

    if user_id in students.index.values:

        return 1

    elif user_id in professionals.index.values:

        return 0

    else:

        raise ValueError('User ID not student & not professional')



school_memberships['is_student'] = school_memberships['user_id'].apply(is_student)

school_memberships['is_student'] = school_memberships['is_student'].astype(int)

count_students_professionals = school_memberships.groupby('is_student').count()[['school_id']].rename(columns={'school_id':'count'})

print(f'\n- Only {count_students_professionals.loc[1].values[0]/len(students):.2f} % of the students are members of schools ({count_students_professionals.loc[1].values[0]}).')

print(f'- Only {count_students_professionals.loc[0].values[0]/len(professionals):.2f} % of the professionals are members of schools ({count_students_professionals.loc[0].values[0]}).')



group_memberships['is_student'] = group_memberships['user_id'].apply(is_student)

group_memberships['is_student'] = group_memberships['is_student'].astype(int)

count_students_professionals = group_memberships.groupby('is_student').count()[['group_id']].rename(columns={'group_id':'count'})

print(f'\n- Only {count_students_professionals.loc[1].values[0]/len(students):.2f} % of the students are members of groups ({count_students_professionals.loc[1].values[0]}).')

print(f'- Only {count_students_professionals.loc[0].values[0]/len(professionals):.2f} % of the professionals are members of groups ({count_students_professionals.loc[0].values[0]}).')





print('')
# Professionals with zero answers

nbr_pros_without_answers = len(professionals) - answers['author_id'].nunique()

#print(f'\n- {(nbr_pros_without_answers/len(professionals))*100:.2f} % of the professionals have Zero answers ({nbr_pros_without_answers}).')

fig = {

    'data': [{

        'type': 'pie',

        'labels': ['Zero answers', '> 0 answers'],

        'values': [nbr_pros_without_answers , len(professionals) - nbr_pros_without_answers],

        'textinfo': 'label+percent',

        'showlegend': False,

        'marker': {'colors': [ '#00FF66', '#D9BCDB',], 'line': {'width': 3, 'color': 'white'}},

    }],

    'layout': {

        'title': 'Professionals with Zero Answers'

    }

}

iplot(fig)

# Answers Import

years = questions['date_added'].dt.year.unique()

years = sorted(years)

professionals['date_joined'] = pd.to_datetime(professionals['date_joined'])

activity_per_year = {}



for y in years:

#y = 2013

    limit_date = pd.to_datetime(f'{y}-12-31') - np.timedelta64(200, 'D')

    year_answers = answers[answers['date_added'].dt.year == y]

    professionals_up_to_year = professionals[professionals['date_joined'].dt.year <= y]

    

    nbr_active_pros = year_answers['author_id'].nunique()

    nbr_inactive_pros = len(professionals_up_to_year) - nbr_active_pros

    activity_per_year[y] = (nbr_active_pros, nbr_inactive_pros)





fig = {

    'data': [

        {

        'type': 'bar',

        'name': 'Number of Active Professionals',

        'x': years,

        'y': [e[0] for e in list(activity_per_year.values())],

        'marker': {'color': '#db2d43'}

        },

        {

        'type': 'bar',

        'name': 'Number of Inactive Professionals',

        'x': years,

        'y': [e[1] for e in list(activity_per_year.values())],

        'marker': {'color': '#906FA8'}

        }

    ],

    'layout': {

        'title': 'Number of Active vs Non-Active Professionals each year',

        'xaxis': {'title': 'Years'},

        'yaxis': {'title': 'Number of Professionals',},

        'barmode': 'stack',

        'legend': {'orientation': 'h'},

    }

}

iplot(fig)

answers = answers.rename(columns={'date_added': 'answers_date_added'})

questions = questions.rename(columns={'date_added': 'questions_date_added'})

first_answers = answers[['question_id', 'answers_date_added']].groupby('question_id').min()

answers_questions = first_answers.join(questions[['questions_date_added']])

answers_questions['diff_days'] = (answers_questions['answers_date_added'] - answers_questions['questions_date_added'])/np.timedelta64(1,'D')

vals = [answers_questions[answers_questions['questions_date_added'].dt.year == y]['diff_days'].mean() for y in years]

LINE_COLOR = '#9250B0'

fig = {

    'data': [{

        'type': 'scatter',

        'x': years,

        'y': vals,

        'line': {'color': LINE_COLOR}

    }],

    'layout': {

        'title': 'Evolution of Time to First Response in days',

        'xaxis': {'title': 'Years'},

        'yaxis': {'title': 'Time to First Response'}

    }

}

iplot(fig)

answers = answers.rename(columns={'answers_date_added': 'date_added'})

questions = questions.rename(columns={'questions_date_added': 'date_added'})
# Number of accurate recommendations

emails['date_sent'] = pd.to_datetime(emails['date_sent'], infer_datetime_format=True)

matches = pd.read_csv('../input/matches.csv')

matches = matches.join(emails[['recipient_id', 'date_sent']], on='matches_email_id')



matches = matches.rename(columns={'matches_question_id': 'question_id', 'matches_email_id': 'email_id'})

all_recommendations_per_year = []

accurate_recommendations_per_year = []

matches['author_id'] = matches['recipient_id']

for y in years:

    year_answers = answers[answers['date_added'].dt.year == y]

    year_recommendations = matches[matches['date_sent'].dt.year == y]

    all_recommendations_per_year.append(len(year_recommendations))

    m = year_answers.reset_index().merge(year_recommendations, on=['question_id', 'author_id']).set_index('answers_id')

    nbr_accurate_recommendations = len(m)

    accurate_recommendations_per_year.append(nbr_accurate_recommendations)

    #print(f'- {(nbr_accurate_recommendations/len(matches))*100:.2f} % of recommended questions in emails were accurate (lead to professional answering the recommended question) ({nbr_accurate_recommendations})')



#print(accurate_recommendations_per_year)

LINE_COLOR = '#9250B0'

fig = {

    'data': [{

        'type': 'scatter',

        'x': years,

        'y': accurate_recommendations_per_year,

        'line': {'color': LINE_COLOR}

    }],

    'layout': {

        'title': 'Evolution of Number of Accurate recommendations',

        'xaxis': {'title': 'Years'},

        'yaxis': {'title': 'Time to First Response'}

    }

}

iplot(fig)

proportions_of_accurate_recommendations = np.array(accurate_recommendations_per_year)/np.array(all_recommendations_per_year)

proportions_of_accurate_recommendations = [0 if np.isnan(e) else e for e in proportions_of_accurate_recommendations]

#print(proportions_of_accurate_recommendations)

fig = {

    'data': [{

        'type': 'scatter',

        'x': years,

        'y': proportions_of_accurate_recommendations,

        'line': {'color': LINE_COLOR}

    },

    ],

    'layout': {

        'title': 'Percentage of Accurate recommendations',

        'xaxis': {'title': 'Years'},

        'yaxis': {'title': 'Proportion of Accurate Recommendations', 'tickformat': ',.0%'}

    }

}

iplot(fig)
# Garbadge collect stuff we won't be using for building the recommender.



del m

del emails

del matches

del students

del school_memberships

del group_memberships

del count_question_tags

del users_who_follow_tags

del nbr_pros_tags

del nbr_students_tags

del nbr_pros_without_answers

del nbr_questions_with_hearts

del count_students_professionals

gc.collect()

print('')
# Drop tags that are not used in any question and not followed by any user (it will clean a lot of useless stuff)

useless_tags = tags[~tags.index.isin(tag_questions['tag_id'].unique())]

useless_tags = tags[ (tags.index.isin(useless_tags.index.values)) & (~tags.index.isin(tag_users['tag_id'].values)) ]

tags = tags.drop(useless_tags.index)



print(f'- {len(useless_tags)} useless tags were found and dropped.')
# Preprocessing Tags



nbr_tags = len(tags)



stop_words = set(stopwords.words('english'))

# some common words / mistakes to filter out too

stop_words.update(['want', 'go', 'like', 'aa', 'aaa', 'aaaaaaaaa', 

                   'good', 'best', 'would', 'get', 'as', 'th', 'k',

                   'become', 'know', 'us'])

special_characters = f'[{string.punctuation}]'

lm = WordNetLemmatizer()





tags['name'] = tags['name'].str.lower()

tags.fillna('', inplace=True)

tags['processed'] = tags['name'].str.replace(special_characters, '')

tags['processed'] = tags['processed'].str.replace('^\d+$', '') # tags that are just numbers :-/

tags['processed'] = tags['processed'].apply(lambda x: lm.lemmatize(x)) # avoid having plurals like 'career' and 'careers'

tags['processed'] = tags['processed'].str.replace('^\w$', '') # single letter tags :-/

tags['processed'] = tags['processed'].str.replace(r'(\d+)(yrs?)', r'\1year') #

tags['processed'] = tags['processed'].apply(lambda x: x if x not in stop_words else '')



# Drop tags which are prepositions, pronouns, determiners, wh-adverbs (where, ...)

tags_to_drop = []

for i, t in tags['processed'].iteritems():

    if len(t) > 0 and nltk.pos_tag([t])[0][1] in ['IN', 'PRP', 'WP$', 'PRP$', 'WP', 'DT', 'WRB']:

        tags_to_drop.append(i)

tag_questions = tag_questions.drop(tag_questions[tag_questions['tag_id'].isin(tags_to_drop)].index)

tags = tags.drop(tags_to_drop)



# Drop tags which are just numbers

tags_to_drop = tags[tags['name'].str.contains('^\d+$')].index

tag_questions = tag_questions.drop(tag_questions[tag_questions['tag_id'].isin(tags_to_drop)].index)

tags = tags.drop(tags_to_drop)



# Drop tags which are just stop words ( after, the , with , ...)

tags_to_drop = tags[tags['name'].isin(stop_words)].index

tag_questions = tag_questions.drop(tag_questions[tag_questions['tag_id'].isin(tags_to_drop)].index)

tags = tags.drop(tags_to_drop)



print(f'{nbr_tags - len(tags)} Tags were filtered out.')

tags.sample(2)
# Questions Cleaning



questions['processed'] = questions['title'] + ' ' + questions['body']

questions['processed'] = questions['processed'].str.lower()

questions['processed'] = questions['processed'].str.replace('<.*?>', '') # remove html tags

questions['processed'] = questions['processed'].str.replace('[-_]', '') # remove separators

questions['processed'] = questions['processed'].str.replace(special_characters, ' ') # remove special characters



questions['processed'] = questions['processed'].str.replace('\d+\s?yrs?', ' years') # single letter tags :-/



def lem_question(q):

    return " ".join([lm.lemmatize(w) for w in q.split() if w not in stop_words])

questions['processed'] = questions['processed'].apply(lem_question)



questions['processed'] = questions['processed'].str.replace(r'(\d+)($|\s+)', r'\2') # remove numbers which are not part of words

questions['processed'] = questions['processed'].str.replace(r'(\d+)([th]|k)', r'\2') # remove numbers from before th and k





# Function to preprocess new questions

# TODO: update function to do like above

def preprocess_question(q):

    q = q.lower()

    q = re.sub("<.*?>", "", q)

    q = re.sub("[-_]", "", q)

    q = re.sub("\d+", "", q)

    q = q.translate(q.maketrans('', '', string.punctuation))

    q = " ".join([lm.lemmatize(t) for t in q.split()])

    return q



cnt_answers = answers.groupby('question_id').count()[['body']].rename(columns={'body': 'count_answers'})

questions = questions.join(cnt_answers)

questions['count_answers'] = questions['count_answers'].fillna(0)

questions['count_answers'] = questions['count_answers'].astype(int)



print('Questions preprocessed.')

questions.sample(1)[['title', 'body', 'processed', 'count_answers']]


# Count Answers

print('Counting Answers ...')

pro_answers_count = answers.groupby('author_id').count()[['question_id']].rename(columns={'question_id': 'count_answers'})

professionals = professionals.join(pro_answers_count)

professionals['count_answers'] = professionals['count_answers'].fillna(0)

professionals['count_answers'] = professionals['count_answers'].astype(int)





# Cleaning the headlines

print('Cleaning Headlines ...')

professionals['headline'] = professionals['headline'].fillna('')

professionals['headline'] = professionals['headline'].str.lower()

professionals['headline'] = professionals['headline'].str.replace('--|hello|hello!|hellofresh', '')



# Check if follow tags or not

print('Creating "follow_tags" column ...')

professionals['follow_tags'] = False

followers = list(tag_users['user_id'].unique())

professionals.loc[professionals.index.isin(followers), 'follow_tags'] = True



# Create Last Answer Date Column

print('Creating "last_answer_date" column ... ')

professionals = professionals.join(answers[['author_id', 'date_added']].groupby('author_id').max().rename(columns={'date_added': 'last_answer_date'}))





print('Professionals preprocessed')

professionals.sample(3)

start = time.time()



tfidf_vectorizer = TfidfVectorizer(stop_words=stop_words,)



NUM_TOPICS = 1100

def build_model(qs , nbr_topics=NUM_TOPICS):

    print('Building the Model ...')

    # TF-IDF Transformation



    qs_tfidf = tfidf_vectorizer.fit_transform(qs['processed'])

    terms = tfidf_vectorizer.get_feature_names()

    print(' (1/3) TF-IDF matrix shape: ', qs_tfidf.shape)



    # Dimensionality Reduction with SVD

    model = TruncatedSVD(n_components=nbr_topics)

    transformer_model = model.fit(qs_tfidf)

    qs_transformed = transformer_model.transform(qs_tfidf)

    print(' (2/3) Shape after Dimensionality Reduction:', qs_transformed.shape)



    # Construct Similarity Matrix

    sim_mat = cosine_similarity(qs_transformed, qs_transformed)

    print(' (3/3) Similarity Matrix Shape', sim_mat.shape, '\n')

    return transformer_model, qs_transformed, sim_mat



transformer_model, Qs_transformed, Qs_sim_matrix = build_model(questions)



end = time.time()



print(f'{(end-start)/60:.2f} minutes')
def calculate_score_question_answered(days_elapsed_after_answer):

    eps = 370

    score = np.log10(days_elapsed_after_answer*(1/eps)) / (np.log10(1/eps))

    score = 0.001 if score < 0 else score # questions that got a score lower than 0 are still given a very low score

    return score



def calculate_exploit_threshold(answered_question_scores, nbr_recommendations, alpha=1.35):

    nbr_questions_answered = len([s for s in answered_question_scores if s > 0])

    eps = 0.1 if nbr_questions_answered == 0 else 0

    return np.log10(np.sqrt(nbr_questions_answered) + 1) * alpha + eps

debug = False
# Set current date as the last day of the data

def set_today(d_str):

    d = pd.to_datetime(d_str)

    

    min_for_questions = d - np.timedelta64(600, 'D') # used for the Freezing professional to select the latest questions and for the cold to select the latest questions in followed and suggested tags

    min_for_answers = d - np.timedelta64(400, 'D')   # used for hot professional to select his last answers. if no answers in this period, Hot professional will be treated as Cold

    return d, min_for_questions, min_for_answers



today, min_date_for_questions, min_date_for_answers = set_today('2019-01-31')


def choose_random_answered_question(question_score_dic):

    random_key = choices(list(question_score_dic.keys()), list(question_score_dic.values()))[0]

    return (random_key, question_score_dic[random_key])





def choose_random_followed_tag(pro_id):

    followed_tags = tag_users[tag_users['user_id'] == pro_id]

    return followed_tags.sample(1)['tag_id'].values[0]



def get_similar_questions(qid, nbr_questions=10, except_questions_ids=[], prioritize=False, similarity_threshold=0.4):

    recommendations = pd.DataFrame([])



    #print(len(except_questions_ids))

    #print()

    q_dists_row = list(Qs_sim_matrix[questions.index.get_loc(qid)])

    for eq_id in except_questions_ids:

        #print('removing ', eq_id)

        #print(len(q_dists_row), questions.index.get_loc(eq_id))

        q_dists_row[questions.index.get_loc(eq_id)] = -1

    q_dists_row = pd.Series(q_dists_row).sort_values(ascending=False)[:100]

    q_dists_row = q_dists_row[1:]



    if not prioritize:

        q_dists_row = q_dists_row[:nbr_questions]

        for i, d in q_dists_row.iteritems():

            qid = questions.index.values[i]

            recommendations = recommendations.append(questions.loc[qid])

    else:

        qid_to_score = {}

        for i, d in q_dists_row.iteritems():

            qid = questions.index.values[i]

            if d > similarity_threshold:

                #print(qid)

                q_added = questions.loc[qid, 'date_added']

                days_elapsed = (today - q_added) / np.timedelta64(1, 'D')

                qid_to_score[qid] = d * days_elapsed

        qid_scores = sorted(qid_to_score.items(), key=lambda x: x[1])[:nbr_questions]

        for qid, score in qid_scores:

            print(q_dists_row[questions.index.get_loc(qid)], qid_to_score[qid]) if debug else None

            recommendations = recommendations.append(questions.loc[qid])

    return recommendations







def recommend_questions_to_professional(pro_id, nbr_recommendations=10, silent=False, alpha=1.35):

    print('Professional ID:', pro_id ) if not silent else None



    # tags followed

    tags_followed = tag_users[tag_users['user_id'] == pro_id]['tag_id']

    tags_followed = tags[tags.index.isin(tags_followed)]

    print('Followed Tags: ', tags_followed['name'].values)  if not silent else None



    # Number of answered questions

    cnt_pro_answers = professionals.loc[pro_id, 'count_answers']

    if cnt_pro_answers > 0:

        pros_answers = answers[(answers['author_id'] == pro_id) & (answers['date_added'] < min_date_for_answers)]

        cnt_pro_answers = len(pros_answers)



    # Type of Start

    cold_start = (cnt_pro_answers == 0)

    freezing_start = (cold_start and len(tags_followed) == 0 )



    n = 3 # Nbr of questions per tag

    recommendations = pd.DataFrame([])





    # Freezing Start

    if freezing_start:

        print('Freezing ...')  if not silent else None

        recommendations = recommendations.append(questions[questions['date_added'] > min_date_for_questions].sample(10))



    # Cold Start

    elif cold_start:

        print('Cold', cnt_pro_answers)  if not silent else None



        qids_from_followed_tags  = tag_questions[tag_questions['tag_id'].isin(tags_followed.index.values)]['question_id'].values

        qids_from_followed_tags  = list(questions[(questions.index.isin(qids_from_followed_tags))   & (questions['date_added'] > min_date_for_questions)].sort_values('date_added', ascending=False).index.values)



        tags_suggested = tags[tags['processed'].isin(tags_followed['processed'].values)]

        tags_suggested = tags_suggested[~tags_suggested.index.isin(tags_followed.index.values)]

        print('Suggested Tags: ', tags_suggested['name'].values)  if not silent else None

        suggested_tags_available = len(tags_suggested) > 0

        # If there are suggested tags, we do explore on them while exploiting on the followed tags

        if suggested_tags_available:

            qids_from_suggested_tags = tag_questions[tag_questions['tag_id'].isin(tags_suggested.index.values)]['question_id'].values

            qids_from_suggested_tags = list(questions[(questions.index.isin(qids_from_followed_tags))  & (questions['date_added'] > min_date_for_questions)].sort_values('date_added', ascending=False).index.values)

            exploit_threshold = .6

        # If no suggested tags are available, we just exploit on the followed tags

        else:

            exploit_threshold = 1





        print('Exploit Threshold: ', exploit_threshold) if debug else None

        for i in range(1, nbr_recommendations+1):

            if np.random.rand() < exploit_threshold and len(qids_from_followed_tags) > 0:

                # Exploit followed tags

                print(f'{i}- Exploit followed tags') if debug else None

                random_index = choice(qids_from_followed_tags)

                q = questions.loc[random_index]

                recommendations = recommendations.append(q)

                qids_from_followed_tags.remove(random_index)

            elif suggested_tags_available and len(qids_from_suggested_tags) > 0:

                # Suggest from suggested tags

                print(f'{i}- Explore suggested tags') if debug else None

                random_index = choice(qids_from_suggested_tags)

                q = questions.loc[random_index]

                recommendations = recommendations.append(q)

                qids_from_suggested_tags.remove(random_index)

            else:

                # no more questions from the pool

                pass



    # Hot Start

    else:

        

        questions_answered_ids = list(pros_answers['question_id'].unique())

        questions_answered = questions[questions.index.isin(questions_answered_ids)].sort_values('date_added', ascending=False)

        questions_answered_locs = []

        for qid in questions_answered_ids:

            questions_answered_locs.append(questions.index.get_loc(qid))



        print('Hot, Answered Questions: ', cnt_pro_answers)  if not silent else None

        #print(questions_answered_locs)

        display(questions_answered[['date_added', 'title', 'body', 'count_answers']])  if not silent else None

        

        # calculate answered questions scores

        q_scores = {}

        for i, q in questions_answered.iterrows():

            answer_post_date = pros_answers[pros_answers['question_id'] == i]['date_added'].values[0]

            days_elapsed_after_answer = (today - answer_post_date)/np.timedelta64(1, 'D')

            q_scores[i] = calculate_score_question_answered(days_elapsed_after_answer)

        print('Question-Scores: ', q_scores) if debug else None



        # calculate exploit_threshold

        exploit_threshold = calculate_exploit_threshold(list(q_scores.values()), nbr_recommendations, alpha=alpha)

        print('Exploit Threshold:', exploit_threshold) if debug else None

        except_qs = []

        except_qs += questions_answered_ids

        for i in range(nbr_recommendations):



            if np.random.rand() < exploit_threshold:

                # Exploit

                random_q_score = choose_random_answered_question(q_scores)

                print('\nExploit Question', random_q_score) if debug else None

                recommendations = recommendations.append(get_similar_questions(random_q_score[0], nbr_questions=1, except_questions_ids=except_qs, prioritize=True))

            else:

                # Explore

                

                # Get Latest n questions from all followed tags

                n = 5

                latest_questions = pd.DataFrame([])

                for tid in tags_followed.index.values:

                    qids = tag_questions[tag_questions['tag_id'] == tid]['question_id'].values

                    tag_qs = questions[questions.index.isin(qids)]

                    tag_qs = tag_qs[~tag_qs.index.isin(except_qs)]

                    if len(tag_qs) > 0:

                        tag_qs = tag_qs.sort_values('date_added', ascending=False)

                        latest_questions = latest_questions.append(tag_qs.head(n))

                #display(latest_questions)

                

                # Select the most similar one to the ones answered using the similarity matrix

                best_question_id = 0

                best_distance = float('-inf')

                for qid, r in latest_questions.iterrows():

                    qloc = questions.index.get_loc(qid)

                    for aqloc in questions_answered_locs:

                        d = Qs_sim_matrix[qloc, aqloc]

                        if best_question_id == 0 or d > best_distance:

                            best_question_id = qid

                            best_distance = d



                print('\nExplore Tags', best_question_id, best_distance) if debug else None

                if best_question_id != 0:

                    recommendations = recommendations.append(questions.loc[best_question_id])

            except_qs = list(recommendations.index.values)

            except_qs += questions_answered_ids



    return recommendations
# Random Hot Professional

random_hot_pro_id = professionals[(professionals['count_answers'] > 2) & (professionals['count_answers'] < 5)].sample(1).index.values[0]



# Random Cold Professional ( check if he follows some tag )

random_cold_pro_id = professionals[(professionals['count_answers'] == 0) & (professionals['follow_tags'] == True)].sample(1).index.values[0]





#for random_pro_id in [random_hot_pro_id, random_cold_pro_id]:

for random_pro_id in [random_hot_pro_id, random_cold_pro_id]:

    recs = recommend_questions_to_professional(random_pro_id, nbr_recommendations=10)

    print('Recommendations: ')

    display(recs[['date_added', 'title', 'body', 'count_answers']]) if len(recs) > 0 else None
random_hot_pro_id = 'fbd6566ddf36402abeb031c088096ae4'

recs = recommend_questions_to_professional(random_hot_pro_id, nbr_recommendations=10)

print('Recommendations:')

display(recs[['date_added', 'title', 'body', 'count_answers']])
def recommend_professionals_for_question(qid, nbr_recommendations=10, inactivity_period=60):

    #print(len(questions), len(answers))

    similar_questions = get_similar_questions(qid, nbr_questions=10, except_questions_ids=[], prioritize=False)

    #display(similar_questions)

    answer_author_ids = answers[answers['question_id'].isin(similar_questions.index.values)]['author_id'].values

    answer_author_ids = pd.Series(answer_author_ids).value_counts()

    

    # Step 1: Check how active the the candidates are

    min_last_answer_date = today - np.timedelta64(inactivity_period, 'D')

    candidates = professionals[(professionals.index.isin(answer_author_ids.index.values)) & (professionals['last_answer_date'] > min_last_answer_date)].sort_values('last_answer_date', ascending=False)

    answer_author_ids = answer_author_ids.drop(candidates.index)

    

    # Step 2: if number of candidates is still smaller than nbr_recommendations, fill in with other authors based on how many similar questions they answered.

    if len(candidates) < nbr_recommendations:

        others = answer_author_ids.head(nbr_recommendations-len(candidates)).index.values

        candidates = candidates.append(professionals[professionals.index.isin(others)])

    return candidates[:nbr_recommendations]
random_question_index = choice(questions.index.values)



print('Random Question: ', random_question_index,  questions.loc[random_question_index]['date_added'])

print(questions.loc[random_question_index]['title'])

print(questions.loc[random_question_index]['body'])

recommend_professionals_for_question(random_question_index, nbr_recommendations=8)[['location', 'industry', 'headline', 'count_answers', 'last_answer_date']]

# Analyze processed question and extracts implicit tags ( eg. 'computer science' => 'computerscience')

def get_tag_suggestions(q_p):

    #q_p = preprocess_question(q)

    #print(q_p)

    q_tokens = nltk.word_tokenize(q_p)

    q_tokens_cpy = q_tokens.copy()

    

    qp_tagged = nltk.pos_tag(q_tokens)

    important = []

    for t,pos in qp_tagged:

        if t not in stop_words and pos == 'NN' and len(tags[tags['processed'] == t]) > 0 :

            i = q_tokens.index(t)

            #print(len(q_p), t, i)

            poses_before_after = []

            if i > 0:

                poses_before_after.append(nltk.pos_tag([q_tokens[i-1]])[0])

            if i < (len(q_tokens)-1):

                poses_before_after.append(nltk.pos_tag([q_tokens[i+1]])[0])

            for i, bf in enumerate(poses_before_after):

                #print(t, bf)

                if bf[1] in ['NN', 'NNS', 'JJ', 'JJR', 'VBG']:

                    s = f'{t}{bf[0]}' if i == 1 else f'{bf[0]}{t}'

                    important.append(s)

            q_tokens.remove(t)

    important = set(important)

    for i in set(important):

        if i not in tags['processed'].values or i in q_tokens_cpy:

            important.remove(i)

    #print(len(important),important)

    

    return tags[tags['processed'].isin(important)]
new_question = 'I am a student in computer science and I want to be a data scientist but I dont know how to study machine learning and artificial intelligence. Can anyone give some advice ?' 

p_q = preprocess_question(new_question)

suggestions = get_tag_suggestions(p_q)

print('Question: ', new_question)

print('\nTag Suggestions: ')

suggestions[['name']]
# Generate a random index for adding a question to DB

def gen_test_index():

    length = np.random.randint(10,15)

    letters_digits = string.ascii_lowercase + string.digits

    return ''.join(random.sample(letters_digits, length))





def add_question_to_db(title, body):

    global questions

    global Qs_transformed

    global Qs_sim_matrix

    

    q = title + ' ' + body

    q_p = preprocess_question(q)

    

    tag_suggestions = get_tag_suggestions(q_p)

    q_p = q_p + ' ' + ' '.join(tag_suggestions)

    

    print(q_p)  if debug else None

    

    author_id = 1 # special if for test ( doesn't exist in DB )

    index = gen_test_index()

    questions = questions.append(pd.Series({'author_id': author_id,'date_added': pd.to_datetime('now'), 

                                                  'title': title,

                                                  'body': body, 

                                                  'processed': q_p, 

                                                  'count_answers': 0}, name=index))

    print('Qs Transformed before', qs_transformed.shape) if debug else None

    q_transformed = transformer_model.transform(tfidf_vectorizer.transform([q_p]))

    Qs_transformed = np.append(Qs_transformed, [Qs_transformed[0]], axis=0)

    print('Qs Transformed after', Qs_transformed.shape)  if debug else None

    

    sim_mat_shape = Qs_sim_matrix.shape

    print('Similarity Matrix shape before', sim_mat_shape)  if debug else None

    new_sims = cosine_similarity(Qs_transformed[-1].reshape(1,-1),Qs_transformed)[0]

    print('new_sims', new_sims.shape)  if debug else None

    Qs_sim_matrix = np.hstack((Qs_sim_matrix, np.zeros((sim_mat_shape[0], 1))))

    Qs_sim_matrix = np.vstack((Qs_sim_matrix, np.zeros((sim_mat_shape[0]+1))))

    Qs_sim_matrix[-1] = new_sims

    Qs_sim_matrix[:, -1] = new_sims

    print('Similarity Matrix shape before', sim_mat_shape)  if debug else None

    print('Question Added to DB.')  if debug else None

    return index

# Backup

questions_full = questions.copy()

answers_full = answers.copy()

professionals_full = professionals.copy()

tag_users_full = tag_users.copy()

tag_questions_full = tag_questions.copy()



def run_time_machine(today_str):

    global today

    global min_date_for_questions

    global min_date_for_answers

    global professionals

    global questions

    global answers

    global tag_users

    global tag_questions

    global tfidf_vectorizer

    global transformer_model

    global Qs_transformed

    global Qs_sim_matrix



    

    first_date = pd.to_datetime('2012-01-01') 



    today, min_date_for_questions, min_date_for_answers = set_today(today_str)

    print('Running Time Machine ....', 'Going to', today.strftime('%B %d %Y'), '................\n')



    professionals = professionals_full[professionals_full['date_joined'] < today].copy()

    assert (professionals['date_joined'].max() < today), "Professionals have date_joined > today !"

    questions = questions_full[(questions_full['date_added'] > first_date) & (questions_full['date_added'] < today)].copy()

    answers = answers_full[(answers_full['date_added'] > first_date) & (answers_full['date_added'] < today) & (answers_full['question_id'].isin(questions.index.values))].copy()



    #test_questions = questions_full[(questions_full['date_added'] > today) & (questions_full['count_answers'] > 0)].copy()

    #test_answers   = answers_full[(answers_full['question_id'].isin(test_questions.index.values)) & (answers_full['author_id'].isin(professionals.index.values))].copy()

    #print(len(test_questions), 'Test questions (with at least one answer)')

    #print(len(test_answers), 'Answers were posted for the test questions (from professionals who joined before that date)')



    cnt_answers = answers.groupby('question_id').count()[['body']].rename(columns={'body': 'count_answers'})

    questions = questions.drop('count_answers', axis=1)

    questions = questions.join(cnt_answers)

    questions['count_answers'] = questions['count_answers'].fillna(0)

    questions['count_answers'] = questions['count_answers'].astype(int)





    cnt_answers = answers.groupby('author_id').count()[['question_id']].rename(columns={'question_id': 'count_answers'})

    professionals = professionals.drop('count_answers', axis=1)

    professionals = professionals.join(cnt_answers)

    professionals['count_answers'] = professionals['count_answers'].fillna(0)

    professionals['count_answers'] = professionals['count_answers'].astype(int)



    # Create Last Answer Date Column

    professionals = professionals.drop('last_answer_date', axis=1)

    professionals = professionals.join(answers[['author_id', 'date_added']].groupby('author_id').max().rename(columns={'date_added': 'last_answer_date'}))





    tag_users = tag_users_full[tag_users_full['user_id'].isin(professionals.index.values)].copy()

    tag_questions = tag_questions_full[tag_questions_full['question_id'].isin(questions.index.values)].copy()



    transformer_model, Qs_transformed, Qs_sim_matrix = build_model(questions)

    gc.collect()

    print('##################################\n')
# Start Date and End Date of the Simulation



test_start_date = pd.to_datetime('2018-08-01')

test_end_date = pd.to_datetime('2018-08-22')

nbr_weeks = math.floor((test_end_date - test_start_date)/ np.timedelta64(1, 'W'))



# Number of recommendations to generate for each professional. ( for the other mode, Question->Professional, it's nbr_recs*2)

nbr_recs_pro_to_qs = 5

nbr_recs_q_to_pros = nbr_recs_pro_to_qs * 2



# Exploitation Intensity ( 1.0 ~ 1.7 )

alpha_arg=1.35



# Number of days to consider professional as inactive

inactivity_period_arg=60
## %%time



print('Start First Simulation', test_start_date.strftime('%B %d, %Y'), '--->', test_end_date.strftime('%B %d, %Y'), '(', nbr_weeks, ' weeks )', '\n')

start = time.time()

#run_time_machine(test_start_date.strftime('%Y-%m-%d'))





question_to_pros_recs = {}

pro_to_questions_recs = {}



nbr_accurate_q_to_pros = 0

nbr_accurate_pro_to_qs = 0

nbr_all_q_to_pros = 0

nbr_all_pro_to_qs = 0





correct_question_ids = set([])

all_question_ids = set([])



today = test_start_date

for i in range(1, nbr_weeks+1):

    print('\n----------- ', 'Week', i, ' -----------')

    d_old = today

    run_time_machine((today + np.timedelta64(1, 'W')).strftime('%Y-%m-%d'))



    week_questions= questions[(questions['date_added'] > d_old) & (questions['date_added'] < today)]

    week_answers = answers[(answers['date_added'] > d_old) & (answers['date_added'] < today) & (answers['author_id'].isin(professionals.index.values))].copy()

    

    if len(week_answers) == 0:

        continue

    

    target_questions = week_questions[week_questions.index.isin(week_answers)]

    qs_answered_this_week = list(week_answers['question_id'].unique())

    authors_this_week = week_answers['author_id'].unique()

    all_question_ids.update(qs_answered_this_week)



    print( d_old, ' ~ ', today, ' - Number of answers: ', len(week_answers), ' - Number answered questions: ', len(qs_answered_this_week), ' - Number authors: ', len(authors_this_week))

    

    # Hide answers from the system !!!

    answers = answers.drop(week_answers.index)

    for auth_id in authors_this_week:

        professionals.at[auth_id, 'count_answers'] = len(answers[answers['author_id'] == auth_id])



    # some tests to check if everything is ok

    assert (len(answers[(answers['date_added'] < today) & (answers['date_added'] > d_old) & (answers['author_id'].isin(professionals.index.values)) ]) == 0), "The answers of this week were not all removed"

    random_auth_id = authors_this_week[0]

    assert (professionals.loc[random_auth_id, 'count_answers'] == len(answers[answers['author_id'] == random_auth_id])), "Problem with count of answers for professional"



    print('Making Predictions for the week\'s answered questions ...')

    # Predict pros for questions that were answered ( Question->Pros )

    for qid in qs_answered_this_week:

        question_to_pros_recs[qid] = recommend_professionals_for_question(qid, nbr_recommendations=nbr_recs_q_to_pros, inactivity_period=inactivity_period_arg)

        recommended_pro_ids = set(question_to_pros_recs[qid].index.values)

        nbr_all_q_to_pros += len(recommended_pro_ids)

        target_pro_ids = set(week_answers[week_answers['question_id'] == qid]['author_id'].unique())

        union_len = len(target_pro_ids.union(recommended_pro_ids))

        sum_len = len(recommended_pro_ids) + len(target_pro_ids)

        if union_len < sum_len:

            nbr_accurate_q_to_pros += (sum_len - union_len)

            correct_question_ids.update([qid])

            

    # Predict questions for pros who answered ( Pro->Questions )

    for auth_id in authors_this_week:

        pro_to_questions_recs[auth_id] = recommend_questions_to_professional(auth_id, nbr_recommendations=nbr_recs_pro_to_qs, silent=True, alpha=alpha_arg)

        recommended_question_ids = set(pro_to_questions_recs[auth_id].index.values)

        nbr_all_pro_to_qs += len(recommended_question_ids)

        target_question_ids = set(week_answers[week_answers['author_id'] == auth_id]['question_id'].unique())

        union_len = len(target_question_ids.union(recommended_question_ids))

        sum_len = len(recommended_question_ids) + len(target_question_ids)

        if union_len < sum_len:

            nbr_accurate_pro_to_qs += (sum_len - union_len)

            correct_question_ids.update([e for e in recommended_question_ids if e in target_question_ids])

    

    #print('Number of Accurate Recommendations (Question -> Pros): ', nbr_accurate_q_to_pros)

    #print('Number of Accurate Recommendations (Pro -> Questions): ', nbr_accurate_pro_to_qs)



end = time.time()

print(f'\n-------- End of Simulation ({(end-start)/60:.2f} minutes) --------')
print('Results of the Test:')

print(f"- Percentage of Answered Questions that got accurate recommendations: {len(correct_question_ids)/len(all_question_ids)*100:.2f}%", f'( {len(correct_question_ids)} out of {len(all_question_ids)} questions  )' )

print('\n- Percentage of accurate recommendations ( out of all sent ones ):')

print(f'\t- Question-to-Professionals Mode:  {(nbr_accurate_q_to_pros/nbr_all_q_to_pros)*100:.2f}% ',  f'( {nbr_accurate_q_to_pros} out of {nbr_all_q_to_pros} recommendations were accurate )')

print(f'\t- Professional-to-Questions Mode: {(nbr_accurate_pro_to_qs/nbr_all_pro_to_qs)*100:.2f}% ', f'( {nbr_accurate_pro_to_qs} out of {nbr_all_pro_to_qs} recommendations were accurate )')
