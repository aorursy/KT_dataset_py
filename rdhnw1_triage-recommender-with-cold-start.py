%%javascript 

IPython.OutputArea.auto_scroll_threshold = 9999;
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

df_recommender_results = pd.DataFrame([['','','','',''],[13,16,18,15,12],[4,1,2,4,3],[4,1,0,3,3],['','','','',''],['','','','',''],[10,10,10,10,10], [10,10,10,6,10], [0,6,9,1,7],[0,0,0,1,0],[6,10,10,6,10],

                                          [0,0,0,0,0], [0,0,3,0,0], [0,7,6,4,6],[10,10,10,10,10],[10,10,10,10,10],

                                          [5,10,10,3,10], [8,9,10,10,9], [7,6,6,5,1],[10,10,9,10,10],[10,10,9,9,9],

                                          [10,10,9,10,10], [10,9,9,8,10], [10,10,10,10,9],[7,10,10,8,10],[8,8,10,8,8],[131,155, 160,129,149],

                                       [66,78,80,65,75]],

    

                                      columns=['tfidf', 'Word2Vec','FastText','GloVec','UniSE'],

                index=['Similar Q for last 20 queries','similar questions to students available','false positives','false negatives','','Relevance of Top10 Qs','query0', 'query1','query2', 'query3','query4', 'query5','query6', 'query7',

                      'query8', 'query9','query10', 'query11','query12', 'query13','query14', 'query15','query16', 'query17',

                      'query18', 'query19','Total','%'])

display (df_recommender_results)
"""Read in the data"""

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

glove_path = '../input/glove-global-vectors-for-word-representation/glove.6B.200d.txt'





professionals = pd.read_csv('../input/data-science-for-good-careervillage/professionals.csv')

emails = pd.read_csv('../input/data-science-for-good-careervillage/emails.csv')

matches = pd.read_csv('../input/data-science-for-good-careervillage/matches.csv')

questions = pd.read_csv('../input/data-science-for-good-careervillage/questions.csv')

answers = pd.read_csv('../input/data-science-for-good-careervillage/answers.csv')

tag_questions = pd.read_csv('../input/data-science-for-good-careervillage/tag_questions.csv')

tags = pd.read_csv('../input/data-science-for-good-careervillage/tags.csv')

tag_users = pd.read_csv('../input/data-science-for-good-careervillage/tag_users.csv')

comments = pd.read_csv('../input/data-science-for-good-careervillage/comments.csv')



question_scores = pd.read_csv('../input/data-science-for-good-careervillage/question_scores.csv')

answer_scores = pd.read_csv('../input/data-science-for-good-careervillage/answer_scores.csv')



group_memberships = pd.read_csv('../input/data-science-for-good-careervillage/group_memberships.csv')

groups = pd.read_csv('../input/data-science-for-good-careervillage/groups.csv')

school_memberships = pd.read_csv('../input/data-science-for-good-careervillage/school_memberships.csv')

students = pd.read_csv('../input/data-science-for-good-careervillage/students.csv')



import numpy as np # linear algebra

import time

import os

import nltk, string

import random

from nltk.corpus import stopwords 

from nltk.stem import WordNetLemmatizer



random_state = 21



w_tokenizer = nltk.tokenize.WhitespaceTokenizer()

lemmatizer = nltk.stem.WordNetLemmatizer()



stop_words = set(stopwords.words('english'))

stop_words.update(['gives'])



"""number of columns in the results from a recommender run"""

sample_len = 100



'''remove punctuation, lowercase, stem'''

remove_punctuation_map = dict((ord(char), ' ') for char in string.punctuation)    

def normalize(text):

    return nltk.word_tokenize(text.lower().translate(remove_punctuation_map))





def cosine_sim(text1, text2):

    tfidf = vectorizer.fit_transform([text1, text2])

    return ((tfidf * tfidf.T).A)[0,1]



def takeSecond(elem):

    return elem[1]



def clean_text(text):

    text = text.lower().translate(remove_punctuation_map)

    

    return ' '.join(lemmatizer.lemmatize(w) for w in w_tokenizer.tokenize(text))





print(os.listdir("../input"))





#pd.options.display.max_colwidth = -1
questions_tags = questions.merge(right=tag_questions, how = 'left',

                                            left_on ='questions_id',

                                            right_on ='tag_questions_question_id')



questions_tagwords = questions_tags.merge(right=tags, how = 'left',

                                            left_on ='tag_questions_tag_id',

                                            right_on ='tags_tag_id')



questions_tagwords =questions_tagwords.sort_values('questions_id')

questions_tagwords = questions_tagwords.drop (['tag_questions_tag_id','tag_questions_question_id','tags_tag_id','questions_author_id'], axis = 1)



questions_tagwords_tb = questions_tagwords.copy()

questions_tagwords_tb['q_tb'] = questions_tagwords['questions_title'] + " " + questions_tagwords['questions_body']



questions_tagwords_tb = questions_tagwords_tb.drop (['questions_title','questions_body'], axis = 1)

questions_tagwords_tb_str = questions_tagwords_tb.copy()

questions_tagwords_tb_str ['tags'] = questions_tagwords_tb ['tags_tag_name'].map (str)

questions_tagwords_tb_str = questions_tagwords_tb_str.drop (['tags_tag_name'], axis = 1)



foo =lambda x:', '.join(x)

agg_f = {'questions_id':'first', 'questions_date_added' : 'first' ,'q_tb': 'first','tags' : foo}



questions_q_tb_tags  = questions_tagwords_tb_str.groupby(by='questions_id').agg(agg_f)



questions_q_tb_tags  = questions_q_tb_tags.drop(['questions_id'], axis = 1)

questions_q_tb_tags  =questions_q_tb_tags .sort_values ('questions_date_added', ascending = False).reset_index()







questions_q_tb_tags.head(1)

"""Tags are repeated in the body of the question and so stop """

"""questions_bow  = questions_q_tb_tags.copy()

questions_bow ['bow'] = questions_bow['q_tb'] + " " + questions_bow['tags']

questions_bow  = questions_bow.drop(['q_tb','tags','questions_id'], axis = 1)

questions_bow  =questions_bow .sort_values ('questions_date_added', ascending = False).reset_index()

pd.options.display.max_colwidth = -1"""

start = time.time()

questions_bow  = questions_q_tb_tags.copy()

questions_bow = questions_bow.rename(columns={'q_tb': 'bow_f'})



questions_bow['bow'] = questions_bow.bow_f.apply(clean_text)

questions_bow['bow'] = questions_bow['bow'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop_words)]))

questions_bow  = questions_bow.drop(['tags'], axis = 1)

#questions_bow  =questions_bow .sort_values ('questions_date_added', ascending = False).reset_index()





end = time.time()

print('run time',end - start)

pd.options.display.max_colwidth = 500

questions_bow.head(5)





q_with_answers_bow = questions_bow.merge(right=answers, how = 'left',

                                            left_on ='questions_id',

                                            right_on ='answers_question_id')

q_with_answers_bow = q_with_answers_bow.dropna(how='any')

q_with_answers_bow = q_with_answers_bow.drop (['answers_id','answers_author_id','answers_question_id','answers_date_added','answers_body'], axis=1)

q_with_answers_bow  =q_with_answers_bow .sort_values ('questions_date_added', ascending = False)



q_with_answers_bow.drop_duplicates( inplace = True)

q_with_answers_bow = q_with_answers_bow.reset_index()

q_with_answers_bow = q_with_answers_bow.drop (['index'], axis=1)



q_with_answers_bow.head(1)


questions_bow['questions_date'] = pd.to_datetime(questions_bow['questions_date_added'])

q_bow_n = questions_bow.copy()

q_bow_n['questions_date'] = questions_bow['questions_date'].dt.normalize()

q_bow_n = q_bow_n.drop (['questions_date_added'], axis = 1)

"""Change date to include more questions"""

"""last month is 304"""

last_qbow_full = q_bow_n[questions_bow.questions_date > '2018-01-31']

last_qbow_full.describe()

"""Produce a subset of the questions data for performance testing"""



half_qbow_full = q_bow_n[questions_bow.questions_date > '2017-01-1']

half_qbow_full.describe()
"""Run this cell to get a smaller random sample of the smaller question data set"""



last_qbow = last_qbow_full.sample(n=10, random_state = 42).reset_index()

last_qbow = last_qbow.drop (['index'], axis = 1)

last_qbow.head(1)
"""Produce a subset of the questions data, using the x most recent"""



#last_qbow = q_with_answers_bow[0:50]



last_qbow = questions_bow[0:sample_len]

last_qbow.head(1)
professionals_tags = professionals.merge(right=tag_users, how = 'left',

                                            left_on ='professionals_id',

                                            right_on ='tag_users_user_id')



professionals_tagwords = professionals_tags.merge(right=tags, how = 'left',

                                            left_on ='tag_users_tag_id',

                                            right_on ='tags_tag_id')



professionals_tagwords =professionals_tagwords.sort_values('professionals_id')

professionals_tagwords = professionals_tagwords.drop (['professionals_location','professionals_date_joined','tag_users_tag_id','tag_users_user_id','tags_tag_id'], axis=1)



professionals_tagwords.head(1)
"""Convert the columns to strings, even though they already look like strings? 

    This is to make concatenation possible later..."""

df_p_q_str = professionals_tagwords.copy()

df_p_q_str ['tag'] = df_p_q_str ['tags_tag_name'].map (str)

df_p_q_str = df_p_q_str.drop (['tags_tag_name'], axis = 1)

df_p_q_str ['industry'] = df_p_q_str ['professionals_industry'].map (str)

df_p_q_str = df_p_q_str.drop (['professionals_industry'], axis = 1)

df_p_q_str ['job'] = df_p_q_str ['professionals_headline'].map (str)

df_p_q_str = df_p_q_str.drop (['professionals_headline'], axis = 1)



df_p_q_str.head(1)
"""merge the tags"""



foo =lambda x:', '.join(x)

agg_f = {'professionals_id':'first', 'industry': 'first','job': 'first','tag' : foo}



df_p_q= df_p_q_str.groupby(by='professionals_id').agg(agg_f)

df_p_q = df_p_q.drop (['professionals_id'], axis = 1).reset_index()





df_p_q.head(1)
"""Merge the questons answered by the professional to the professionals dataframe"""



df_p_a = df_p_q.merge(right=answers, how = 'left',

                                            left_on ='professionals_id',

                                            right_on ='answers_author_id')

df_p_a = df_p_a.drop (['answers_author_id','answers_date_added','answers_body'], axis=1)



df_p = df_p_a.merge(right=questions_bow, how = 'left',

                                            left_on ='answers_question_id',

                                            right_on ='questions_id')



df_p = df_p.drop (['answers_id','answers_question_id','questions_id'], axis=1)

df_p ['qbow'] = df_p ['bow'].map (str)

df_p = df_p.drop (['bow'], axis = 1)

df_p ['qbow_f'] = df_p ['bow_f'].map (str)

df_p = df_p.drop (['bow_f'], axis = 1)





df_p.head(1)
"""merge the tags"""



foo =lambda x:', '.join(x)

agg_f = {'professionals_id':'first', 'industry': 'first','job': 'first','tag' : foo}



df_p_q= df_p_q_str.groupby(by='professionals_id').agg(agg_f)

df_p_q = df_p_q.drop (['professionals_id'], axis = 1).reset_index()





df_p_q.head(1)
"""Merge the questions"""



Foo =lambda x:', '.join(x)

agg_f = {'professionals_id':'first',  'industry': 'first','job': 'first','tag' : 'first', 'qbow' : foo, 'qbow_f' : foo}



df_p_bow  = df_p.groupby(by='professionals_id').agg(agg_f)

df_p_bow = df_p_bow.drop (['professionals_id'], axis = 1).reset_index()

df_p_bow = df_p_bow.sort_values('professionals_id')





df_p_bow['bow_f'] = df_p_bow['industry'] + " " + df_p_bow['job']+ " " + df_p_bow['tag']+ " " + df_p_bow['qbow_f']



df_p_bow['bow'] = df_p_bow['industry'] + " " + df_p_bow['job']+ " " + df_p_bow['tag']+ " " + df_p_bow['qbow']

df_p_bow = df_p_bow.drop (['industry','job','tag','qbow','qbow_f'], axis = 1).reset_index()



df_p_bow.head(1)
"""drop the professionals who have not answered a question"""



df_p_nonan = df_p.dropna(how='any')



df_p_bow_nonan  = df_p_nonan.groupby(by='professionals_id').agg(agg_f)

df_p_bow_nonan = df_p_bow_nonan.drop (['professionals_id'], axis = 1).reset_index()

df_p_bow_nonan = df_p_bow_nonan.sort_values('professionals_id')







df_p_bow_nonan['bow'] = df_p_bow_nonan['industry'] + " " + df_p_bow_nonan['job']+ " " + df_p_bow_nonan['tag']+ " " + df_p_bow_nonan['qbow']

df_p_bow_nonan = df_p_bow_nonan.drop (['industry','job','tag','qbow'], axis = 1).reset_index()





df_p_bow_nonan.head(1)


"""Find the tfidf cos between one set of questions and another"""

"""Allow for the first array will be the total sample and the second a smaller sample"""



from sklearn.feature_extraction.text import CountVectorizer 

from sklearn.feature_extraction.text import TfidfTransformer

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.metrics.pairwise import cosine_similarity



   

def get_sim_q_array (q_total,q_query):

    #vectorizer = TfidfVectorizer(tokenizer=normalize, stop_words='english')

    vectorizer = TfidfVectorizer(tokenizer=normalize)



    """q_query could be passed 1 or more queries"""

    vectorizer.fit(q_total)

    q_total_tfidf = vectorizer.transform(q_total)

    q_query_tfidf = vectorizer.transform(q_query)

    q_sim_array = cosine_similarity(q_total_tfidf, q_query_tfidf)

    

    return (q_sim_array)
"""Single query to get tfidf similiarities"""

start = time.time()



q_total = ["".join(x) for x in (q_with_answers_bow['bow'])]

q_queries = [last_qbow.loc [0]['bow']]

q_sim = get_sim_q_array (q_total,q_queries)



end = time.time()

print('run time',end - start)

#print(q_sim)
"""Single query to get tfidf similiarities for half questions for performance test"""

start = time.time()



q_total = ["".join(x) for x in (half_qbow_full['bow'])]

q_queries = [last_qbow.loc [0]['bow']]

q_sim = get_sim_q_array (q_total,q_queries)



end = time.time()

print('run time',end - start)

#print(q_sim)
"""Multiple queries to get tfidf similiarities"""



start = time.time()



q_total = ["".join(x) for x in (q_with_answers_bow['bow'])]

q_queries = ["".join(x) for x in (q_with_answers_bow['bow'])]

q_sim_tfidf_array = get_sim_q_array (q_total,q_queries)



end = time.time()

print('run time',end - start)

#print(q_sim_m_array)
"""function to produce dataframe of results of similarity tests"""

def get_sim_results_with_threshold (column_head,index,sim_array,questions,query,h_threshold,l_threshold):



    col_h = column_head + str(index)

    

    df_sim_q = pd.DataFrame({'Cosine':sim_array[:,index], col_h:questions['bow_f']})



    df_sim_q_sorted = df_sim_q.sort_values('Cosine',ascending = False )

    if df_sim_q_sorted.iloc[0]['Cosine'] > .9999:

        df_sim_q_sorted = df_sim_q_sorted.drop(df_sim_q_sorted.index[0])



    h_num = 0

    l_num = 0

    worst_h_num = -1

    i = 0

    questions_len = len(questions)

    while i< questions_len and df_sim_q_sorted.iloc[i]['Cosine'] > l_threshold:

        #print ('i, df_sim_q_sorted.iloc[i]['Cosine']')

        if df_sim_q_sorted.iloc[i]['Cosine'] > l_threshold:

            l_num += 1

            worst_match_to_profs= i

        if df_sim_q_sorted.iloc[i]['Cosine'] > h_threshold:

            worst_h_num = i

            h_num += 1

        i += 1

    

    df_sim_q_sample = df_sim_q_sorted[:10]

        

    best_cos_0 = df_sim_q_sample.iloc[0]['Cosine']

    best_cos_9 = df_sim_q_sample.iloc[9]['Cosine']

    

    df_sim_q_sample = df_sim_q_sample.drop ('Cosine', axis=1).reset_index()

    df_sim_q_sample = df_sim_q_sample.drop ( 'index', axis=1)



    df_sim_q_sample_T = df_sim_q_sample.T

    df_sim_q_sample_T.insert(loc=0, column='query_id', value=[query.iloc[index]['questions_id']] )

    df_sim_q_sample_T.insert(loc=1, column='query_bow', value=[query.iloc[index]['bow_f']]  )

    df_sim_q_sample_T.insert(loc=2, column='best_cos', value=best_cos_0)

    df_sim_q_sample_T.insert(loc=3, column='10th_best_cos', value=best_cos_9)

    df_sim_q_sample_T.insert(loc=4, column='similar Q to students', value= h_num)

    df_sim_q_sample_T.insert(loc=5, column='Qs to profs', value=l_num)

    df_sim_q_sample_T.insert(loc=6, column='best matches', value=' ')



    if worst_h_num > -1:

        df_sim_q_sample_T.insert(loc=17, column='worst match to students', value=df_sim_q_sorted.iloc[worst_h_num][col_h])

    else:

        df_sim_q_sample_T.insert(loc=17, column='worst match to students', value='not available')

    

    if l_num > 0:

        df_sim_q_sample_T.insert(loc=18, column='worst match to profs', value=df_sim_q_sorted.iloc[worst_match_to_profs][col_h])

    else:

        df_sim_q_sample_T.insert(loc=18, column='worst match to profs', value='not available')

    

    

    

    return ( df_sim_q_sample_T)
"""Compare  q with q using tfidf"""

h_threshold =0.45

l_threshold =0.225





results_T = get_sim_results_with_threshold ('query',0,q_sim_tfidf_array,q_with_answers_bow,q_with_answers_bow,h_threshold,l_threshold)



for i in range(1,sample_len):

    next_result = get_sim_results_with_threshold ('query',i,q_sim_tfidf_array,q_with_answers_bow,q_with_answers_bow,h_threshold,l_threshold)

    results_T = pd.concat([results_T,next_result])

results_tfidf = results_T.T

pd.options.display.max_colwidth = 500

display(results_tfidf) 
import numpy as np

import matplotlib.pyplot as plt



q_num =[]

for i in range (sample_len):

    q_num.append(i)



area = np.pi*3

plt.figure(figsize=(10,10))

plt.xlim(0, sample_len)

plt.ylim(0,200)



# Plot

plt.scatter(q_num, results_T['similar Q to students'], s=25, c='red', alpha=0.5, label = "> high threshold")

plt.scatter(q_num, results_T['Qs to profs'], s=25, c='blue', alpha=0.5, label = "> low threshold")



plt.title('Scatter plot showing similiar questions found for each query ')

plt.xlabel('query')

plt.ylabel('similiar questions')

plt.legend()

plt.show()
area = np.pi*3

plt.figure(figsize=(10,10))

plt.xlim(0, sample_len)

plt.ylim(0,20)



# Plot

plt.scatter(q_num, results_T['similar Q to students'], s=25, c='red', alpha=0.5, label = "> high threshold")



plt.title('Scatter plot showing similiar questions found for each query ')

plt.xlabel('query')

plt.ylabel('similiar questions')

plt.legend()

plt.show()
def get_sim_questions_id (column_head,index,sim_array,questions,query):



    col_h = column_head + str(index)

    

    df_sim_q = pd.DataFrame({'Cosine':sim_array[:,index], col_h:questions['questions_id']})

    

    

    df_sim_q_sorted = df_sim_q.sort_values('Cosine',ascending = False )

    if df_sim_q_sorted.iloc[0]['Cosine'] > .9999:

        df_sim_q_sorted = df_sim_q_sorted.drop(df_sim_q_sorted.index[0])

        

    df_sim_q_sample = df_sim_q_sorted[:10]

    

    df_sim_q_sample = df_sim_q_sample.drop ('Cosine', axis=1).reset_index()

    df_sim_q_sample = df_sim_q_sample.drop ( 'index', axis=1)



    

    df_sim_q_sample_T = df_sim_q_sample.T

    df_sim_q_sample_T.insert(loc=0, column='id', value=[query.iloc[index]['questions_id']] )

    df_sim_q_sample_T.insert(loc=1, column='bow', value=[query.iloc[index]['bow_f']]  )    

    

    

    return ( df_sim_q_sample_T)
def find_top_q_ids (q_sim_array):

    q_id_results_T = get_sim_questions_id ('query',0,q_sim_array,q_with_answers_bow,q_with_answers_bow)



    for i in range(1,sample_len):

        next_result = get_sim_questions_id ('query',i,q_sim_array,q_with_answers_bow,q_with_answers_bow)

        q_id_results_T = pd.concat([q_id_results_T,next_result])

    return q_id_results_T.T

def professionals_to_ask (sim_q_results, index):

    col = 'query' + str(index)



    df_prof_with_a = sim_q_results[[col]]

    df_prof_with_a = df_prof_with_a.rename(columns={col: 'questions_id'})

    df_prof_with_a = df_prof_with_a.drop(['id','bow'], axis = 0)   

    df_prof_with_a = df_prof_with_a.merge(right=answers, how = 'left',

                                            left_on ='questions_id',

                                            right_on ='answers_question_id')

    df_prof_with_a = df_prof_with_a.merge(right=professionals, how = 'left',

                                            left_on ='answers_author_id',

                                            right_on ='professionals_id')

    df_prof_with_a =  df_prof_with_a.drop (['questions_id','answers_author_id','professionals_headline','professionals_date_joined',

                                        'answers_id','answers_date_added','answers_body',

                                    'answers_question_id','professionals_location','professionals_industry'], axis = 1)

    tot = df_prof_with_a.shape[0]

    

     

    df_prof_with_a_T = df_prof_with_a.T

    #df_prof_with_a_T.insert(loc=0, column='total', value = tot)

    df_prof_with_a = df_prof_with_a_T.T

    #df_prof_with_a = df_prof_with_a.rename(columns={'professionals_id': col})

    df_prof_with_a.insert(loc=1, column='total', value = 0)

    

    return df_prof_with_a

#tfidfx.head(20)
"""Compare  q with q using tfidf to get q_ids then prof_ids"""

q_id_results = find_top_q_ids (q_sim_tfidf_array)

pd.options.display.max_colwidth = 500

q_id_results.head(20) 

"""check that we have right ids!"""

"""x = q_with_answers_bow.count()

print (x.questions_id)

i =0 

while i<x.questions_id:

    row = q_with_answers_bow.iloc[i]

    

    if  row.questions_id == 'c0c9260091b3443f9c712d5ff2d2c2e0':

        print ('q_with_answers_bow.iloc[i][questions_id]',row.questions_id)

        print ('q_with_answers_bow.iloc[i][bow_f]',row.bow_f)

    i += 1"""
df_tfifd_prof_with_a = professionals_to_ask (q_id_results, 1)

df_tfifd_prof_with_a.head()
df_tfifd_prof_with_a = professionals_to_ask (q_id_results, 0)





for i in range(1,sample_len):

    next_p_to_a = professionals_to_ask (q_id_results, i)

    df_tfifd_prof_with_a = pd.concat([df_tfifd_prof_with_a,next_p_to_a],axis=0, sort=True)

    

pd.options.display.max_colwidth = 500

pd.options.display.max_seq_items = 2000



df_tfidf_p_grouped = df_tfifd_prof_with_a.groupby('professionals_id').count()

df_tfidf_p_grouped = df_tfidf_p_grouped.sort_values('total',ascending = False )

next_p_to_a.head()
df_tfidf_p_grouped.sum(axis = 0, skipna = True) 
df_tfidf_p_grouped.describe()


q_num =[]

for i in range (df_tfidf_p_grouped.shape[0]):

    q_num.append(i)



area = np.pi*3

plt.figure(figsize=(10,10))

plt.xlim(0, df_tfidf_p_grouped.shape[0])

plt.ylim(0,100)



# Plot

plt.scatter(q_num, df_tfidf_p_grouped['total'], s=25, c='red', alpha=1)



plt.title('Scatter plot showing number of queries answered by professionals  ')

plt.xlabel('professional')

plt.ylabel('queries answered')

plt.legend()

plt.show()
q_sim_tfidf_array = []


start = time.time()



total_q_bow = ["".join(x) for x in (q_with_answers_bow['bow'])]

#vectorizer = TfidfVectorizer(tokenizer=normalize, stop_words='english')

vectorizer = TfidfVectorizer(tokenizer=normalize)

tfidf = vectorizer.fit_transform(total_q_bow)



cachedStopWords = stopwords.words("english")

#print(total_q_bow)

total_q_bow_l  = [x.lower() for x in total_q_bow]

#print(total_q_bow_l)

all_words = [nltk.word_tokenize(x.translate(remove_punctuation_map)) for x in total_q_bow_l]



for i in range(len(all_words)):  

    all_words[i] = [w for w in all_words[i] if w not in cachedStopWords]



#print(all_words)



end = time.time()

print('run time',end - start)
"""w2v_model= Word2Vec(all_words, min_count=2)

"""

from gensim.models import Word2Vec



embed_size = 300



#the model is set up as with the Kaggle paper above with the noted exceptions:

# min_count =2 because the corpus is realtively small

# window = 5 to capture more context around a word

w2v_model = Word2Vec(min_count=2,

                     window=5,

                     size=embed_size,

                     sample=6e-5, 

                     alpha=0.03, 

                     min_alpha=0.0007, 

                     negative=20,

                     workers=1)



start = time.time()



w2v_model.build_vocab(all_words, progress_per=10000)



print('Time to build vocab: {} mins'.format(round((time.time() - start) / 60, 2)))



w2v_model.train(all_words, total_examples=w2v_model.corpus_count, epochs=30, report_delay=1)



print('Time to train the model: {} mins'.format(round((time.time() - start) / 60, 2)))







print('w2v_model.corpus_count',w2v_model.corpus_count)

vocabulary = w2v_model.wv.vocab  

#print(vocabulary)

w2v_model.wv.most_similar(positive=["police"])
"""sentence embedding for questions usig Word2Vec"""

start = time.time()



rows, cols = tfidf.nonzero()

print (rows)

print (cols)

rows_l = len(rows)



s_embed = []

s_embeds = []

dividend = []

atStart = True

oldr = -1

w_cnt = 0

vocab = vectorizer.get_feature_names()



for i in range (rows_l):

    r = rows[i]

    c = cols[i]

    if (oldr != r):

        if (atStart == False):

            #calc embedding for questions

            s_embed = np.divide(dividend, divisor)

            s_embeds.append(s_embed.flatten())

            

        else: 

            atStart = False

        oldr = r

        w_cnt = 0

        dividend = np.zeros((1, embed_size))

        divisor = 0



       

    #print('r,c,w_cnt',r,c,w_cnt)

    word = vocab[c]

    if word in w2v_model.wv.vocab:

        wt = tfidf[r,c]

        #print (wt, word)

        w_embed = w2v_model.wv[word]

        #print(w_embed)

        #print(w_embed * wt)

        dividend = np.add(dividend, w_embed * wt)

        divisor += wt

        w_cnt +=1

#    else:

#        print (word, " not in vocab")

s_embed = np.divide(dividend, divisor)

s_embeds.append(s_embed.flatten())

#print (s_embeds)

end = time.time()

print('run time',end - start)

"""Dataframe used to store embeddings"""



df_embed = pd.DataFrame({'col':s_embeds})

df_q_s_embed = pd.merge( questions_bow,df_embed, left_index=True, right_index=True)

df_q_s_embed.head(1)
"""Cosine similarity for all q v one q"""



start = time.time()



q_embed_array = cosine_similarity(s_embeds, [s_embeds[0]])

end = time.time()

print('run time',end - start)



"""This should be compared against the tfidf method that takes about 12 secs to get the array"""
"""Cosine similarity for all q v all q"""



start = time.time()



q_embed_array = cosine_similarity(s_embeds, s_embeds)

end = time.time()

print('run time',end - start)

start = time.time()

"""Compare  q with q using sentence embedding"""

h_threshold =0.84

l_threshold =0.75





results_T = get_sim_results_with_threshold ('query',0,q_embed_array,q_with_answers_bow,q_with_answers_bow,h_threshold,l_threshold)



for i in range(1,sample_len):

    next_result = get_sim_results_with_threshold ('query',i,q_embed_array,q_with_answers_bow,q_with_answers_bow,h_threshold,l_threshold)

    results_T = pd.concat([results_T,next_result])

results_word2vec = results_T.T

pd.options.display.max_colwidth = 500

pd.options.display.max_seq_items = 2000

end = time.time()

print('df run time',end - start)

display (results_word2vec) 
"""Compare  q with q using Word2Vec to get q_ids then prof_ids"""

q_id_results = find_top_q_ids (q_embed_array)

pd.options.display.max_colwidth = 500

q_id_results.head(20) 
"""check that we have right ids!"""

x = q_with_answers_bow.count()

print (x.questions_id)

i =0 

while i<x.questions_id:

    row = q_with_answers_bow.iloc[i]

    

    if  row.questions_id == 'f46e757e38fa4243805534133ff9cb5b':

        print ('q_with_answers_bow.iloc[i][questions_id]',row.questions_id)

        print ('q_with_answers_bow.iloc[i][bow_f]',row.bow_f)

    i += 1
df_w2v_prof_with_a = professionals_to_ask (q_id_results, 0)





for i in range(1,sample_len):

    next_p_to_a = professionals_to_ask (q_id_results, i)

    df_w2v_prof_with_a = pd.concat([df_w2v_prof_with_a,next_p_to_a],axis=0, sort=True)

    

pd.options.display.max_colwidth = 500

pd.options.display.max_seq_items = 2000



df_w2v_p_grouped = df_w2v_prof_with_a.groupby('professionals_id').count()

df_w2v_p_grouped = df_w2v_p_grouped.sort_values('total',ascending = False )

df_w2v_p_grouped.describe()
df_w2v_p_grouped.sum(axis = 0, skipna = True) 
q_embed_array.shape
q_embed_array =[]
"""uncomment to see all_words"""

#print (all_words)
from gensim.models import FastText

start = time.time()

embed_size = 300

"""all_words is a list of all the questions with the words separated and cleaned"""

ft_model = FastText(all_words, size=embed_size, window=5, min_count=2, workers=1

                    ,sg=1)

print('Time to build FastText model: {} mins'.format(round((time.time() - start) / 60, 2)))





ft_model.wv.most_similar(positive=["police"])
"""Uncomment this to see how tdidfs are stored"""

#print (tfidf)
"""sentence embedding for questions usig FastText"""

start = time.time()

"""tfidf is calculated in the Word2Vec section"""

"""There a tfidf value for every word in all_words"""

rows, cols = tfidf.nonzero()

print (rows)

print (cols)

rows_l = len(rows)



s_embed = []

s_embeds = []

dividend = []

atStart = True

oldr = -1

w_cnt = 0

"""using vectorization calculated in the Word2Vec section"""

vocab = vectorizer.get_feature_names()



#this method of calculating the embeddings is a bit ugly but takes advantage of how tfidfs are stored

#for every question

for i in range (rows_l):

    r = rows[i]

    c = cols[i]

    if (oldr != r):

        #new questions and so store last embeddings

        if (atStart == False):

            #calc embedding for last questions

            s_embed = np.divide(dividend, divisor)

            s_embeds.append(s_embed.flatten())

            

        else: 

            atStart = False

        oldr = r

        w_cnt = 0

        dividend = np.zeros((1, embed_size))

        divisor = 0



       

    #find the next word

    word = vocab[c]

    if word in ft_model.wv.vocab:

        #word is in the vocab and so calculate its contribution to the question vector

        wt = tfidf[r,c]

        #print (wt, word)

        w_embed = ft_model.wv[word]

        #print(w_embed)

        #print(w_embed * wt)

        dividend = np.add(dividend, w_embed * wt)

        divisor += wt

        w_cnt +=1

#    else:

#        print (word, " not in vocab")

s_embed = np.divide(dividend, divisor)

s_embeds.append(s_embed.flatten())

#print (s_embeds)

end = time.time()

print('Sentence embedding run time',end - start)

start = time.time()



q_embed_array = cosine_similarity(s_embeds, s_embeds)

end = time.time()

print('cosine sim time',end - start)

"""Compare  q with q using sentence embedding"""

start = time.time()



h_threshold =0.94

l_threshold =0.9



results_T = get_sim_results_with_threshold ('query',0,q_embed_array,q_with_answers_bow,q_with_answers_bow,h_threshold,l_threshold)

for i in range(1,sample_len):

#for i in range(1,20):



    next_result = get_sim_results_with_threshold ('query',i,q_embed_array,q_with_answers_bow,q_with_answers_bow,h_threshold,l_threshold)

    results_T = pd.concat([results_T,next_result])

results_FastText = results_T.T

pd.options.display.max_colwidth = 500

pd.options.display.max_seq_items = 2000

end = time.time()

print('df time',end - start)



display (results_FastText) 
"""Compare  q with q using FastText to get q_ids then prof_ids"""

q_id_results = find_top_q_ids (q_embed_array)

pd.options.display.max_colwidth = 500

#q_id_results.head(20) 
df_FT_prof_with_a = professionals_to_ask (q_id_results, 0)





for i in range(1,sample_len):

    next_p_to_a = professionals_to_ask (q_id_results, i)

    df_FT_prof_with_a = pd.concat([df_FT_prof_with_a,next_p_to_a],axis=0, sort=True)

    

pd.options.display.max_colwidth = 500

pd.options.display.max_seq_items = 2000



df_FT_p_grouped = df_FT_prof_with_a.groupby('professionals_id').count()

df_FT_p_grouped = df_FT_p_grouped.sort_values('total',ascending = False )

df_FT_p_grouped.describe()
df_FT_p_grouped.sum(axis = 0, skipna = True) 
q_embed_array = []
start = time.time()



glove_embed= dict()

glove_data= open(glove_path)

for line in glove_data:

    data = line.split(' ')

    word = data[0]

    vectors = np.asarray(data[1:], dtype='float32')

    glove_embed[word] = vectors

    

glove_data.close()

end = time.time()

print('run time',end - start)
"""sentence embedding for questions using Global Vectors"""

start = time.time()



embed_size = 200

rows, cols = tfidf.nonzero()

print (rows)

print (cols)

rows_l = len(rows)



s_embed = []

s_embeds = []

dividend = []

atStart = True

oldr = -1

w_cnt = 0

vocab = vectorizer.get_feature_names()

tot_words = 0

words_not_in_gv = 0

for i in range (rows_l):

    r = rows[i]

    c = cols[i]

    if (oldr != r):

        if (atStart == False):

            #calc embedding for questions

            s_embed = np.divide(dividend, divisor)

            s_embeds.append(s_embed.flatten())

            

        else: 

            atStart = False

        oldr = r

        w_cnt = 0

        dividend = np.zeros((1, embed_size))

        divisor = 0



       

    #print('r,c,w_cnt',r,c,w_cnt)

    word = vocab[c]

    #print (word)

    wt = tfidf[r,c]

    #print (wt, word)

    if word in glove_embed:

        w_embed = glove_embed[word]

        dividend = np.add(dividend, w_embed * wt)

        divisor += wt

    else:

        words_not_in_gv += 1

    tot_words += 1    

    w_cnt +=1

s_embed = np.divide(dividend, divisor)

s_embeds.append(s_embed.flatten())

print ('The following figures show the benefit of an auto correct or spell check on input')

print ('Number of words not in global vectors', words_not_in_gv,'total words', tot_words)

#print (s_embeds)

end = time.time()

print('run time',end - start)
"""Cosine similarity for all q v all q"""



start = time.time()



q_embed_array = cosine_similarity(s_embeds, s_embeds)

end = time.time()

print('run time',end - start)
"""Compare  q with q using global vecs"""

h_threshold =0.85

l_threshold =0.75





results_T = get_sim_results_with_threshold ('query',0,q_embed_array,q_with_answers_bow,q_with_answers_bow,h_threshold,l_threshold)



for i in range(1,sample_len):

    next_result = get_sim_results_with_threshold ('query',i,q_embed_array,q_with_answers_bow,q_with_answers_bow,h_threshold,l_threshold)

    results_T = pd.concat([results_T,next_result])

results_glovec = results_T.T

pd.options.display.max_colwidth = 500

pd.options.display.max_seq_items = 2000



display (results_glovec) 
"""Compare  q with q using GloVec to get q_ids then prof_ids"""

q_id_results = find_top_q_ids (q_embed_array)

pd.options.display.max_colwidth = 500

q_id_results.head(20) 
df_glo_prof_with_a = professionals_to_ask (q_id_results, 0)





for i in range(1,sample_len):

    next_p_to_a = professionals_to_ask (q_id_results, i)

    df_glo_prof_with_a = pd.concat([df_glo_prof_with_a,next_p_to_a],axis=0, sort=True)

    

pd.options.display.max_colwidth = 500

pd.options.display.max_seq_items = 2000



df_glo_p_grouped = df_glo_prof_with_a.groupby('professionals_id').count()

df_glo_p_grouped = df_glo_p_grouped.sort_values('total',ascending = False )

df_glo_p_grouped.sum(axis = 0, skipna = True) 

df_glo_p_grouped.describe()

q_embed_array =[]
"""Edit 1"""



"""import tensorflow as tf

import tensorflow_hub as hub

import numpy as np

import os, sys

from sklearn.metrics.pairwise import cosine_similarity



# get cosine similairty matrix

def cos_sim(input_vectors):

    similarity = cosine_similarity(input_vectors)

    return similarity



# get topN similar sentences



module_url = "https://tfhub.dev/google/universal-sentence-encoder/2" #@param ["https://tfhub.dev/google/universal-sentence-encoder/2", "https://tfhub.dev/google/universal-sentence-encoder-large/3"]

# Import the Universal Sentence Encoder's TF Hub module

embed = hub.Module(module_url)"""

"""Edit 2"""



"""q_total = ["".join(x) for x in (q_with_answers_bow['bow'])]





start = time.time()

with tf.Session() as session:



  session.run([tf.global_variables_initializer(), tf.tables_initializer()])

  sentences_embeddings = session.run(embed(q_total))



similarity_matrix = cos_sim(np.array(sentences_embeddings))



print (similarity_matrix)

end = time.time()

print('run time',end - start)"""
"""Edit 3"""

"""h_threshold =0.75

l_threshold =0.75





uni_results_T = get_sim_results_with_threshold ('query',0,similarity_matrix,q_with_answers_bow,q_with_answers_bow,h_threshold,l_threshold)



for i in range(1,sample_len):

    next_result = get_sim_results_with_threshold ('query',i,similarity_matrix,q_with_answers_bow,q_with_answers_bow,h_threshold,l_threshold)

    uni_results_T = pd.concat([uni_results_T,next_result])

uni_results = uni_results_T.T

pd.options.display.max_colwidth = 500

pd.options.display.max_seq_items = 2000



display (uni_results)"""

uni_results = results_glovec
"""Edit 4"""

"""Compare  q with q using USE to get q_ids then prof_ids

q_id_results = find_top_q_ids (similarity_matrix)

pd.options.display.max_colwidth = 500

q_id_results.head(20) """
df_uni_prof_with_a = professionals_to_ask (q_id_results, 0)





for i in range(1,sample_len):

    next_p_to_a = professionals_to_ask (q_id_results, i)

    df_uni_prof_with_a = pd.concat([df_uni_prof_with_a,next_p_to_a],axis=0, sort=True)

    

pd.options.display.max_colwidth = 500

pd.options.display.max_seq_items = 2000



df_uni_p_grouped = df_uni_prof_with_a.groupby('professionals_id').count()

df_uni_p_grouped = df_uni_p_grouped.sort_values('total',ascending = False )

df_uni_p_grouped.sum(axis = 0, skipna = True) 
df_uni_p_grouped.describe()
similarity_matrix = []
def get_q_prof_with_threshold (column_head,index,sim_array,questions,query,h_threshold,l_threshold):



    col_h = column_head + str(index)

    

    df_sim_q = pd.DataFrame({'Cosine':sim_array[:,index], col_h:questions['bow_f']})



    df_sim_q_sorted = df_sim_q.sort_values('Cosine',ascending = False )

    if df_sim_q_sorted.iloc[0]['Cosine'] > .9999:

        df_sim_q_sorted = df_sim_q_sorted.drop(df_sim_q_sorted.index[0])



    h_num = 0

    l_num = 0

    worst_h_num = -1

    i = 0

    questions_len = len(questions)

    while i< questions_len and df_sim_q_sorted.iloc[i]['Cosine'] > l_threshold:

        #print ('i, df_sim_q_sorted.iloc[i]['Cosine']')

        if df_sim_q_sorted.iloc[i]['Cosine'] > l_threshold:

            l_num += 1

            worst_match_to_profs= i

        if df_sim_q_sorted.iloc[i]['Cosine'] > h_threshold:

            worst_h_num = i

            h_num += 1

        i += 1

    

        

    df_sim_q_sample = df_sim_q_sorted[:10]

    

        

    best_cos_0 = df_sim_q_sample.iloc[0]['Cosine']

    best_cos_9 = df_sim_q_sample.iloc[9]['Cosine']

    

    df_sim_q_sample = df_sim_q_sample.drop ('Cosine', axis=1).reset_index()

    df_sim_q_sample = df_sim_q_sample.drop ( 'index', axis=1)



    df_sim_q_sample_T = df_sim_q_sample.T

    df_sim_q_sample_T.insert(loc=0, column='id', value=[query.iloc[index]['questions_id']] )

    df_sim_q_sample_T.insert(loc=1, column='bow', value=[query.iloc[index]['bow_f']]  )

    df_sim_q_sample_T.insert(loc=2, column='best_cos', value=best_cos_0)

    df_sim_q_sample_T.insert(loc=3, column='10th_best_cos', value=best_cos_9)

    df_sim_q_sample_T.insert(loc=4, column='similar Q to students', value= h_num)

    df_sim_q_sample_T.insert(loc=5, column='Qs to profs', value=l_num)

    df_sim_q_sample_T.insert(loc=6, column='best matches', value=' ')



    if worst_h_num > -1:

        df_sim_q_sample_T.insert(loc=17, column='worst match to students', value=df_sim_q_sorted.iloc[worst_h_num][col_h])

    else:

        df_sim_q_sample_T.insert(loc=17, column='worst match to students', value='not available')

    

    if l_num > 0:

        df_sim_q_sample_T.insert(loc=18, column='worst match to profs', value=df_sim_q_sorted.iloc[worst_match_to_profs][col_h])

    else:

        df_sim_q_sample_T.insert(loc=18, column='worst match to profs', value='not available')

    

    

    

    return ( df_sim_q_sample_T)


"""Compare  q with prof using tfidf"""

h_threshold =0.45

l_threshold =0.225





start = time.time()



q_total = ["".join(x) for x in (df_p_bow['bow'])]

q_queries = ["".join(x) for x in (q_with_answers_bow['bow'])]

q_sim_p_array = get_sim_q_array (q_total,q_queries)



end = time.time()

print('run time',end - start)

#print(q_sim_m_array)



results_prof_T = get_q_prof_with_threshold ('query',0,q_sim_p_array,df_p_bow,q_with_answers_bow,h_threshold,l_threshold)

#for i in range(1,sample_len):

#reduced to make ui managable

for i in range(1,20):

    next_result = get_q_prof_with_threshold ('query',i,q_sim_p_array,df_p_bow,q_with_answers_bow,h_threshold,l_threshold)

    results_prof_T = pd.concat([results_prof_T,next_result])

results_prof_tfidf = results_prof_T.T

pd.options.display.max_colwidth = 700

display(results_prof_tfidf) 

def get_sim_p_id (column_head,index,sim_array,questions,query):



    col_h = column_head + str(index)

    

    df_sim_q = pd.DataFrame({'Cosine':sim_array[:,index], col_h:questions['professionals_id']})

    

    

    df_sim_q_sorted = df_sim_q.sort_values('Cosine',ascending = False )

    if df_sim_q_sorted.iloc[0]['Cosine'] > .9999:

        df_sim_q_sorted = df_sim_q_sorted.drop(df_sim_q_sorted.index[0])

        

    df_sim_q_sample = df_sim_q_sorted[:20]

    

    df_sim_q_sample = df_sim_q_sample.drop ('Cosine', axis=1).reset_index()

    df_sim_q_sample = df_sim_q_sample.drop ( 'index', axis=1)



    

    df_sim_q_sample_T = df_sim_q_sample.T

    """    df_sim_q_sample_T.insert(loc=0, column='id', value=[query.iloc[index]['questions_id']] )

    df_sim_q_sample_T.insert(loc=1, column='bow', value=[query.iloc[index]['bow_f']]  )    

    """    

    

    return ( df_sim_q_sample_T)
def find_top_p_ids (q_sim_array):

    q_id_results_T = get_sim_p_id ('query',0,q_sim_array,df_p_bow,q_with_answers_bow)

    for i in range(1,sample_len):

        next_result = get_sim_p_id ('query',i,q_sim_array,df_p_bow,q_with_answers_bow)

        q_id_results_T = pd.concat([q_id_results_T,next_result])

    return q_id_results_T.T
"""Compare  q with q using tfidf to get q_ids then prof_ids"""

p_id_results= find_top_p_ids (q_sim_p_array)

pd.options.display.max_colwidth = 500

p_id_results.head(2000) 
"""check that we have right ids!"""

"""x = df_p_bow.count()

print (x.professionals_id)

i =0 

while i<x.professionals_id:

    row = df_p_bow.iloc[i]

    

    if  row.professionals_id == '57a497a3dd214fe6880816c376211ddb':

        print ('df_p_bow.iloc[i][professionals_id]',row.professionals_id)

        print ('df_p_bow.iloc[i][bow_f]',row.bow_f)

    i += 1"""
p_id_results_T0 = get_sim_p_id ('query',0,q_sim_p_array,df_p_bow,q_with_answers_bow)

p_id_results =p_id_results_T0.T

for i in range(1,sample_len):

    p_id_results_Tx = get_sim_p_id ('query',i,q_sim_p_array,df_p_bow,q_with_answers_bow)

    p_id_resultsx = p_id_results_Tx.T

    col_h = 'query' + str(i)

    p_id_resultsx = p_id_resultsx.rename(columns={col_h: 'query0'})



    p_id_results = pd.concat([p_id_results,p_id_resultsx],axis=0, sort=True)



p_id_results.describe()   
p_id_results = p_id_results.reset_index()

p_id_results = p_id_results.rename(columns={'query0': 'professionals_id'})



#p_id_results_r = p_id_results_r.drop('index')

p_id_results.head(10)   
p_id_results_grouped = p_id_results.groupby('professionals_id').count()

p_id_results_grouped = p_id_results_grouped.rename(columns={'index': 'total'})

p_id_results_grouped = p_id_results_grouped.sort_values('total',ascending = False )



p_id_results_grouped.head()
p_id_results_grouped.describe()
print (p_id_results.shape)
q_sim_p_array = []
"""Change the index number between 0 and sample_len """



index = 0



def compare_methods(index):



    col = 'query' + str(index)



    tfidfx = results_tfidf[[col]]

    tfidfx = tfidfx.rename(columns={col: 'tfidf'})

    

    senembedx = results_word2vec[[col]]

    senembedx = senembedx.rename(columns={col: 'Word2Vec'})

   

    

    senembedFTx = results_FastText[[col]]

    senembedFTx = senembedFTx.rename(columns={col: 'FastText'})

    



    glovecx = results_glovec[[col]]

    glovecx = glovecx.rename(columns={col: 'GloVe'})



    univecx = uni_results[[col]]

    univecx = univecx.rename(columns={col: 'USE'})



    df_queryx = pd.concat([tfidfx,senembedx],axis=1, sort=False)

    df_queryx = pd.concat([df_queryx,senembedFTx],axis=1, sort=False)

    df_queryx = pd.concat([df_queryx,glovecx],axis=1, sort=False)

    df_queryx = pd.concat([df_queryx,univecx],axis=1, sort=False)



    

    return df_queryx



display(compare_methods(index))
index = 4

compare_methods(index)

index = 2

compare_methods(index)

df_p_with_a = df_tfidf_p_grouped.merge(right=df_w2v_p_grouped, how = 'outer',

                                            left_on ='professionals_id',

                                            right_on ='professionals_id')



df_p_with_a = df_p_with_a.rename(columns={'total_x': 'tfidf'})

df_p_with_a = df_p_with_a.rename(columns={'total_y': 'Word2Vec'})





df_p_with_a = df_p_with_a.merge(right=df_FT_p_grouped, how = 'outer',

                                            left_on ='professionals_id',

                                            right_on ='professionals_id')

df_p_with_a = df_p_with_a.rename(columns={'total': 'FastText'})







df_p_with_a = df_p_with_a.merge(right=df_glo_p_grouped, how = 'outer',

                                            left_on ='professionals_id',

                                            right_on ='professionals_id')



df_p_with_a = df_p_with_a.rename(columns={'total': 'GloVe'})





df_p_with_a = df_p_with_a.merge(right=df_uni_p_grouped, how = 'outer',

                                            left_on ='professionals_id',

                                            right_on ='professionals_id')

df_p_with_a = df_p_with_a.rename(columns={'total': 'USE'})





df_p_with_a.describe()
df_p_with_a.shape
df_p_with_a = df_tfidf_p_grouped.merge(right=df_w2v_p_grouped, how = 'inner',

                                            left_on ='professionals_id',

                                            right_on ='professionals_id')

df_p_with_a = df_p_with_a.rename(columns={'total_x': 'tfidf'})





df_p_with_a = df_p_with_a.merge(right=df_FT_p_grouped, how = 'inner',

                                            left_on ='professionals_id',

                                            right_on ='professionals_id')







df_p_with_a = df_p_with_a.merge(right=df_glo_p_grouped, how = 'inner',

                                            left_on ='professionals_id',

                                            right_on ='professionals_id')



df_p_with_a = df_p_with_a.merge(right=df_uni_p_grouped, how = 'inner',

                                            left_on ='professionals_id',

                                            right_on ='professionals_id')





df_p_with_a.describe()
df_p_with_a.shape
df_p_with_a = df_tfidf_p_grouped.merge(right=df_w2v_p_grouped, how = 'outer',

                                            left_on ='professionals_id',

                                            right_on ='professionals_id')

df_p_with_a.shape
df_p_with_a = df_tfidf_p_grouped.merge(right=df_w2v_p_grouped, how = 'inner',

                                            left_on ='professionals_id',

                                            right_on ='professionals_id')

df_p_with_a.shape
df_p_with_a = df_tfidf_p_grouped.merge(right=df_glo_p_grouped, how = 'outer',

                                            left_on ='professionals_id',

                                            right_on ='professionals_id')

df_p_with_a.shape
df_p_with_a = df_tfidf_p_grouped.merge(right=df_glo_p_grouped, how = 'inner',

                                            left_on ='professionals_id',

                                            right_on ='professionals_id')

df_p_with_a.shape
df_p_with_a = df_tfidf_p_grouped.merge(right=df_uni_p_grouped, how = 'outer',

                                            left_on ='professionals_id',

                                            right_on ='professionals_id')

df_p_with_a.shape
df_p_with_a = df_tfidf_p_grouped.merge(right=df_uni_p_grouped, how = 'inner',

                                            left_on ='professionals_id',

                                            right_on ='professionals_id')

df_p_with_a.shape
df_p_with_a = df_w2v_p_grouped.merge(right=df_glo_p_grouped, how = 'outer',

                                            left_on ='professionals_id',

                                            right_on ='professionals_id')

df_p_with_a.shape
df_p_with_a = df_w2v_p_grouped.merge(right=df_glo_p_grouped, how = 'inner',

                                            left_on ='professionals_id',

                                            right_on ='professionals_id')

df_p_with_a.shape
df_p_with_a = df_w2v_p_grouped.merge(right=df_uni_p_grouped, how = 'outer',

                                            left_on ='professionals_id',

                                            right_on ='professionals_id')

df_p_with_a.shape
df_p_with_a = df_w2v_p_grouped.merge(right=df_uni_p_grouped, how = 'inner',

                                            left_on ='professionals_id',

                                            right_on ='professionals_id')

df_p_with_a.shape
df_p_with_a = df_glo_p_grouped.merge(right=df_uni_p_grouped, how = 'outer',

                                            left_on ='professionals_id',

                                            right_on ='professionals_id')

df_p_with_a.shape
df_p_with_a = df_glo_p_grouped.merge(right=df_uni_p_grouped, how = 'inner',

                                            left_on ='professionals_id',

                                            right_on ='professionals_id')

df_p_with_a.shape
p_id_results_grouped.head()
df_tfidf_p_grouped.head()
df_p_with_a = df_tfidf_p_grouped.merge(right=p_id_results_grouped, how = 'outer',

                                            left_on ='professionals_id',

                                            right_on ='professionals_id')

df_p_with_a.shape
df_p_with_a = df_tfidf_p_grouped.merge(right=p_id_results_grouped, how = 'inner',

                                            left_on ='professionals_id',

                                            right_on ='professionals_id')

df_p_with_a.shape

def get_prof_q_results (prof_index,dfs_p_bow,q_sim_p_array,q_with_answers_bow):



    prof = 'prof ' + str(prof_index)

    df_profs_q = pd.DataFrame({'Cosine':q_sim_p_array[:,prof_index], prof:q_with_answers_bow['bow_f']})



    df_profs_q_sorted = df_profs_q.sort_values('Cosine',ascending = False )

    if df_profs_q_sorted.iloc[0]['Cosine'] > .9999:

        df_profs_q_sorted = df_profs_q_sorted.drop(df_profs_q_sorted.index[0])



    

    df_profs_q_sample = df_profs_q_sorted[:10]

    best_cos_0 = df_profs_q_sample.iloc[0]['Cosine']

    best_cos_9 = df_profs_q_sample.iloc[9]['Cosine']



    df_profs_q_sample = df_profs_q_sample.drop ('Cosine', axis=1).reset_index()

    df_profs_q_sample = df_profs_q_sample.drop ( 'index', axis=1)



    df_profs_q_sample_T = df_profs_q_sample.T

    df_profs_q_sample_T.insert(loc=0, column='professionals_id', value=[dfs_p_bow.iloc[prof_index]['professionals_id']] )

    df_profs_q_sample_T.insert(loc=1, column='professionals_bow', value=[dfs_p_bow.iloc[prof_index]['bow']]  )

    df_profs_q_sample_T.insert(loc=2, column='best_cos', value=best_cos_0)

    df_profs_q_sample_T.insert(loc=3, column='10th_best_cos', value=best_cos_9)

    df_profs_q_sample_T.insert(loc=4, column='Best Matches', value=' ')



    return ( df_profs_q_sample_T)

"""Produce a subset of the professional data of all professionals"""



dfs_p_bow = df_p_bow.sample(n=10, random_state = 21).reset_index()

dfs_p_bow = dfs_p_bow.drop (['index','level_0'], axis = 1)



"""Compare questions against the professionals bow"""



q_total = ["".join(x) for x in (q_with_answers_bow['bow'])]

q_queries = ["".join(x) for x in (dfs_p_bow['bow'])]

q_sim_p_array = get_sim_q_array (q_total,q_queries)



"""get results for all professionals sample"""

results_T = get_prof_q_results (0,dfs_p_bow,q_sim_p_array,q_with_answers_bow)

prof_sample_len = 10

for i in range(1, prof_sample_len):

    next_result = get_prof_q_results (i,dfs_p_bow,q_sim_p_array,q_with_answers_bow)

    results_T = pd.concat([results_T,next_result])

results = results_T.T

pd.options.display.max_colwidth = 500

results.head(15)    



"""Compare nonan sample of professionals bow against the professionals bow"""



dfs_p_bow_nonan = df_p_bow_nonan.sample(n=10, random_state = 21).reset_index()

dfs_p_bow_nonan = dfs_p_bow_nonan.drop (['index','level_0'], axis = 1)



pd.options.display.max_colwidth = -1



q_total = ["".join(x) for x in (df_p_bow['bow'])]

q_queries = ["".join(x) for x in (dfs_p_bow_nonan['bow'])]

q_sim_pp_array = get_sim_q_array (q_total,q_queries)



"""get results for  p v p comparison"""

results_T = get_prof_q_results (0,dfs_p_bow_nonan,q_sim_pp_array,df_p_bow)

prof_sample_len = 10

for i in range(1, prof_sample_len):

    next_result = get_prof_q_results (i,dfs_p_bow_nonan,q_sim_pp_array,df_p_bow)

    results_T = pd.concat([results_T,next_result])

results = results_T.T

pd.options.display.max_colwidth = 500

results.head(15)    
"""Merge matches and emails to find total questions asked"""

"""There are about 1.8m emails and 4.3m matches and so soe emails contain more than one question """



match_q_p = matches.merge(right=emails, how = 'left',

                                            left_on ='matches_email_id',

                                            right_on ='emails_id')



match_q_p.head()
match_q_p_simple = match_q_p.drop (['emails_id','emails_date_sent','emails_frequency_level'], axis=1)

match_q_p_simple.head()
match_q_p_simple = match_q_p_simple.sort_values ('emails_recipient_id')

match_recipents = match_q_p_simple.groupby('emails_recipient_id').count()

match_recipents = match_recipents.sort_values ('emails_recipient_id')

match_recipents = match_recipents.reset_index()



match_recipents = match_recipents.drop ('matches_question_id', axis=1)

match_recipents = match_recipents.rename(columns={'matches_email_id': 'questions_received'})





match_recipents.head()




email_recipents = emails[['emails_id' ,'emails_recipient_id']]

email_recipents = email_recipents.groupby('emails_recipient_id').count()

sorted_email_recipents = email_recipents.sort_values ('emails_recipient_id')

sorted_email_recipents = sorted_email_recipents.reset_index()



df_profs_emails = professionals.copy()



df_profs_emails = df_profs_emails.sort_values('professionals_id')

df_profs_emails.reset_index(inplace =True, drop =True)

df_profs_emails = df_profs_emails.merge(right=sorted_email_recipents, how = 'left',

                                            left_on ='professionals_id',

                                            right_on ='emails_recipient_id')



df_profs_emails = df_profs_emails.drop ('emails_recipient_id', axis=1)

df_profs_emails = df_profs_emails.rename(columns={'emails_id': 'emails_received'})

df_profs_emails = df_profs_emails.fillna(0)

df_profs_emails.head() 



 

df_profs_emails_q = df_profs_emails.merge(right=match_recipents, how = 'left',

                                            left_on ='professionals_id',

                                            right_on ='emails_recipient_id')

df_profs_emails_q = df_profs_emails_q.drop ('emails_recipient_id', axis = 1)

df_profs_emails_q_sorted = df_profs_emails_q.sort_values ('emails_received' ,ascending = False)



df_profs_emails_q_sorted.head() 


"""Using groupby to speed up data processing"""



answers_cut = answers[['answers_author_id' ,'answers_question_id']]

answer_count = answers_cut.groupby('answers_author_id').count()

sorted_answer_count = answer_count.sort_values ('answers_author_id')

sorted_answer_count = sorted_answer_count.reset_index()



sorted_answer_count.head()



"""Merge the info on answered questions to the professional df"""



df_profs_emails_answers = df_profs_emails_q.copy()

df_profs_emails_answers = df_profs_emails_answers.sort_values('professionals_id')

df_profs_emails_answers.reset_index(inplace =True, drop =True)



df_profs_emails_answers = df_profs_emails_answers.merge(right=sorted_answer_count, how = 'left',

                                            left_on ='professionals_id',

                                            right_on ='answers_author_id')

df_profs_emails_answers = df_profs_emails_answers.drop ('answers_author_id', axis=1)

df_profs_emails_answers = df_profs_emails_answers.rename(columns={'answers_question_id': 'questions_answered'})

df_profs_emails_answers = df_profs_emails_answers.fillna(0)

df_profs_emails_answers_sorted = df_profs_emails_answers.sort_values ('questions_answered' ,ascending = False)



df_profs_emails_answers_sorted.head() 
"""Scatter Plot for q answered v q asked"""

import matplotlib.pyplot as plt

import math





"""Need to use log to get data spreading"""

df_profs_emails_answers ['log_questions_received'] = df_profs_emails_answers ['questions_received']

df_profs_emails_answers ['log_questions_answered'] = df_profs_emails_answers ['questions_answered']



def getlog (x):

    if (x == 0):

        x= 'NaN'

    else:

        x = math.log10(x)

    return x

   



df_profs_emails_answers['log_questions_received'] = df_profs_emails_answers['log_questions_received'].map(getlog)

df_profs_emails_answers['log_questions_answered'] = df_profs_emails_answers['log_questions_answered'].map(getlog)

df_profs_emails_truncatedanswers = df_profs_emails_answers.copy()



plt.figure(figsize=(10,10))

plt.scatter(df_profs_emails_answers['questions_received'],df_profs_emails_answers['questions_answered'],  color='k', s=25, alpha=0.2)

plt.xlim(-5, 50)

plt.ylim(-5,50)

plt.plot([-5,50], [-5,50], 'k-', color = 'red')



plt.xlabel('questions_received')

plt.ylabel('questions_answered')

plt.title('CareerVillage Q v A truncated at 50')

plt.legend()

plt.show()
plt.figure(figsize=(10,10))

plt.scatter(df_profs_emails_answers['log_questions_received'],df_profs_emails_answers['log_questions_answered'],  color='k', s=25, alpha=0.2)

plt.plot([0,3], [0,3], 'k-', color = 'red'), plt.xlim(0, 3), plt.ylim(0,3)

plt.xlabel('log_questions_received'),plt.ylabel('log_questions_answered')

plt.title('CareerVillage Questions Chart')

plt.legend()

plt.show()
df_profs_emails_answers ['a_q_not_asked'] =  df_profs_emails_answers['questions_answered'] - df_profs_emails_answers['questions_received']

df_profs_emails_answers ['a_q_not_asked'] = df_profs_emails_answers ['a_q_not_asked'].apply(lambda x: 0 if x < 0 else x)

print (df_profs_emails_answers ['a_q_not_asked'].sum())

df_profs_emails_answers.describe()


df_profs_emails_answers['DateTime'] = pd.to_datetime(df_profs_emails_answers['professionals_date_joined'])

df_profs_emails_answers['date_joined'] = df_profs_emails_answers['DateTime'].dt.normalize()

df_profs_emails_answers = df_profs_emails_answers.drop(['professionals_date_joined','DateTime'],axis = 1)

#df_profs_emails_answers['Day Joined'] = [2011-01-01]

df_profs_emails_answers.head()

"""Start processing emails to get first and last emails sent dates"""

"""Perhaps don't need to sort but useful for manualchecking"""

sorted_email_recipents = emails.sort_values  (['emails_recipient_id','emails_date_sent'])

sorted_email_recipents.reset_index(inplace =True, drop =True)



sorted_email_recipents.head()
"""This section finds the days that professionals receive emails"""

"""Very slow, working on 1.8m emails"""

sorted_email_recipents_dates = sorted_email_recipents.copy()

sorted_email_recipents_dates['email_date'] = pd.to_datetime(sorted_email_recipents['emails_date_sent'])

sorted_email_recipents_dates['email_date'] = sorted_email_recipents_dates['email_date'].dt.normalize()

sorted_email_recipents_dates = sorted_email_recipents_dates.drop ('emails_date_sent', axis = 1)



sorted_email_recipents_dates_min = sorted_email_recipents_dates.groupby('emails_recipient_id').min()

sorted_email_recipents_dates_min = sorted_email_recipents_dates_min.rename(columns={'email_date':'first_email_date'})

sorted_email_recipents_dates_min = sorted_email_recipents_dates_min.drop(['emails_id','emails_frequency_level'], axis = 1)





sorted_email_recipents_dates_min.head()
sorted_email_recipents_dates_max = sorted_email_recipents_dates.groupby('emails_recipient_id').max()

sorted_email_recipents_dates_max = sorted_email_recipents_dates_max.rename(columns={'email_date':'last_email_date'})

sorted_email_recipents_dates_max = sorted_email_recipents_dates_max.drop(['emails_id','emails_frequency_level'], axis = 1)



sorted_email_recipents_dates_max.head()


sorted_email_recipents_dates_min_max = sorted_email_recipents_dates_min.merge(right=sorted_email_recipents_dates_max, how = 'left',

                                            left_on ='emails_recipient_id',

                                            right_on ='emails_recipient_id')





sorted_email_recipents_dates_min_max.head()
df_profs_emails_answers = df_profs_emails_answers.merge(right=sorted_email_recipents_dates_min_max, how = 'left',

                                            left_on ='professionals_id',

                                            right_on ='emails_recipient_id')



df_profs_emails_answers.head()
#df_profs_emails_answers['Days to 1st email'] = pd.to_datetime(df['date'])

df_profs_emails_answers['days_before_1st_email'] = df_profs_emails_answers['first_email_date'] - df_profs_emails_answers['date_joined']

df_profs_emails_answers['days_ns'] = df_profs_emails_answers['last_email_date'] - df_profs_emails_answers['first_email_date']

df_profs_emails_answers['days_emailed'] = df_profs_emails_answers['days_ns'].apply(lambda x: x.days)

df_profs_emails_answers = df_profs_emails_answers.drop('days_ns', axis = 1)

df_profs_emails_answers.head()

df_profs_emails_answers['emails_per_day'] = df_profs_emails_answers['emails_received'] / df_profs_emails_answers['days_emailed']

df_profs_emails_answers['answers_per_day'] = df_profs_emails_answers['questions_answered'] / df_profs_emails_answers['days_emailed']



df_profs_emails_answers['log_answers_per_day'] = df_profs_emails_answers['answers_per_day'] 

df_profs_emails_answers['log_answers_per_day'] = df_profs_emails_answers['log_answers_per_day'].map(getlog)



df_profs_emails_answers['log_emails_per_day'] = df_profs_emails_answers['emails_per_day'] 

df_profs_emails_answers['log_emails_per_day'] = df_profs_emails_answers['log_emails_per_day'].map(getlog)



df_profs_emails_answers.head()


plt.figure(figsize=(10,10))

plt.scatter(df_profs_emails_answers['days_emailed'],df_profs_emails_answers['questions_answered'],  color='red', s=25, alpha=0.2)



plt.xlim(-5, 250)

plt.ylim(-5,50)



plt.xlabel('days_emailed')

plt.ylabel('questions_answered')

plt.title('CareerVillage Questions Answered')

plt.legend()

plt.show()
plt.figure(figsize=(10,10))

plt.scatter(df_profs_emails_answers['days_emailed'],df_profs_emails_answers['log_questions_answered'],  color='red', s=25, alpha=0.2)



plt.xlabel('days_emailed')

plt.ylabel('log_questions_answered')

plt.title('CareerVillage Questions Answered')

plt.legend()

plt.show()
"""This section finds the days that professionals are active"""



answers_author_date = answers[['answers_author_id' ,'answers_date_added']]

answers_author_date = answers_author_date.sort_values(['answers_author_id' ,'answers_date_added'])

answers_author_date.reset_index(inplace =True, drop =True)

answers_author_date['answer_date'] = pd.to_datetime(answers_author_date['answers_date_added'])

answers_author_date['answer_date'] = answers_author_date['answer_date'].dt.normalize()

answers_author_date = answers_author_date.drop ('answers_date_added', axis = 1)



answers_author_date_min = answers_author_date.groupby('answers_author_id').min()

answers_author_date_max = answers_author_date.groupby('answers_author_id').max()

answers_author_date_min_max = answers_author_date_min.merge(right=answers_author_date_max, how = 'left',

                                            left_on ='answers_author_id',

                                            right_on ='answers_author_id')



answers_author_date_min_max = answers_author_date_min_max.rename(columns={'answer_date_x':'first_answer'})

answers_author_date_min_max = answers_author_date_min_max.rename(columns={ 'answer_date_y' :'last_answer'})



answers_author_date_min_max.head()
plt.figure(figsize=(10,10))

plt.scatter(df_profs_emails_answers['days_emailed'],df_profs_emails_answers['log_answers_per_day'],  color='red', s=25, alpha=0.2)



plt.xlabel('days_emailed')

plt.ylabel('log_answers_per_day')

plt.title('CareerVillage Response ')

plt.legend()

plt.show()
df_profs_emails_answers_active = df_profs_emails_answers.merge(right=answers_author_date_min_max, how = 'left',

                                            left_on ='professionals_id',

                                            right_on ='answers_author_id')

df_profs_emails_answers_active.head()
df_profs_emails_answers_active['days_active_ns'] = df_profs_emails_answers_active['last_answer'] - df_profs_emails_answers_active['first_answer']

df_profs_emails_answers_active['days_active'] = df_profs_emails_answers_active['days_active_ns'].apply(lambda x: x.days)

df_profs_emails_answers_active = df_profs_emails_answers_active.drop('days_active_ns', axis = 1)

df_profs_emails_answers_active.head(10)
df_profs_emails_answers_active['log_questions_answered'] = df_profs_emails_answers_active['questions_answered'] 

df_profs_emails_answers_active['log_questions_answered'] = df_profs_emails_answers_active['log_questions_answered'].map(getlog)

plt.figure(figsize=(10,10))

plt.scatter(df_profs_emails_answers_active['days_active'],df_profs_emails_answers_active['log_questions_answered'],  color='red', s=25, alpha=0.2)

plt.xlabel('days_active')

plt.ylabel('log_questions_answered')

plt.title('CareerVillage Activity ')

plt.legend()

plt.show()
plt.figure(figsize=(10,10))

plt.scatter(df_profs_emails_answers_active['days_active'],df_profs_emails_answers_active['questions_answered'],  color='red', s=25, alpha=0.2)



plt.xlim(-50, 500)

plt.ylim(-5,50)



plt.xlabel('days_active')

plt.ylabel('questions_answered')

plt.title('CareerVillage Activity ')

plt.legend()

plt.show()
df_profs_emails_answers_active.describe(include = 'all')
df_profs_emails_answers_active.sum()
questions.describe()
questions_answers = questions.merge(right=answers, how = 'left',

                                            left_on ='questions_id',

                                            right_on ='answers_question_id')

questions_answers.describe()
questions_answers.head(2)
print(questions_answers['answers_id'].isna().sum() )
answers_v_professionals = df_profs_emails_answers_active.copy()

answers_v_professionals = answers_v_professionals[['questions_answered','professionals_id']]

answers_v_professionals = answers_v_professionals.groupby('questions_answered').count()

match_recipents = answers_v_professionals.sort_values ('questions_answered')

answers_v_professionals = answers_v_professionals.reset_index()

answers_v_professionals = answers_v_professionals.rename(columns={'professionals_id': 'professionals'})

print (answers_v_professionals.sum())

answers_v_professionals.head(50)
plt.figure(figsize=(10,10))

plt.scatter(answers_v_professionals['questions_answered'],answers_v_professionals['professionals'],  color='red', s=25, alpha=0.2)



plt.xlabel('questions_answered')

plt.ylabel('professionals')

plt.title('Professional Activity ')

plt.legend()

plt.show()

plt.figure(figsize=(10,10))

plt.scatter(answers_v_professionals['questions_answered'],answers_v_professionals['professionals'],  color='red', s=25, alpha=0.2)

plt.xlim(-10, 100),plt.ylim(-50,200)

plt.xlabel('questions_answered'),plt.ylabel('professionals')

plt.title('Professional Activity ')

plt.legend()

plt.show()


answers_profs = answers.merge(right=professionals, how = 'left',

                                            left_on ='answers_author_id',

                                            right_on ='professionals_id')

answers_profs['DateTime'] = pd.to_datetime(answers_profs['professionals_date_joined'])

answers_profs['date_joined'] = answers_profs['DateTime'].dt.normalize()

answers_profs['DateTime'] = pd.to_datetime(answers_profs['answers_date_added'])

answers_profs['answer_date'] = answers_profs['DateTime'].dt.normalize()

answers_profs = answers_profs.drop(['DateTime'],axis = 1)



answers_profs['days_ns'] = answers_profs['answer_date'] - answers_profs['date_joined']

answers_profs['days_to_answer'] = answers_profs['days_ns'].apply(lambda x: x.days)

answers_profs = answers_profs.drop('days_ns', axis = 1)





answers_profs.describe()
answers_v_professionals = answers_v_professionals.groupby('questions_answered').count()

sorted_answers_profs = answers_profs.sort_values ('days_to_answer')

sorted_answers_profs = sorted_answers_profs.reset_index()

#sorted_answers_profs.head()

sorted_answers_profs_g = sorted_answers_profs.groupby('days_to_answer').count()

sorted_answers_profs_g.head()
comments.describe ()
answer_scores.describe()
grp_answer_scores = answer_scores.groupby('score').count()

grp_answer_scores.head()
df_answer_scores_comments= answer_scores.merge(right=comments, how = 'left',

                                            left_on ='id',

                                            right_on ='comments_parent_content_id')

df_answer_scores_comments.head()
df_answer_scores_comments.describe()
df_qa= questions_bow.merge(right=answers, how = 'left',

                                            left_on ='questions_id',

                                            right_on ='answers_question_id')

df_qahc= df_qa.merge(right=df_answer_scores_comments, how = 'left',

                                            left_on ='answers_id',

                                            right_on ='id')



df_qahc_s = df_qahc.drop (['answers_id','answers_author_id',

                           'answers_question_id','answers_date_added','id','comments_id','comments_author_id',

                           'comments_parent_content_id','comments_date_added'], axis=1)

df_qahc_s = df_qahc_s.sort_values ('score' ,ascending = False)

df_qahc_s.head(1)
group_memberships.describe()
group_memberships_group  = group_memberships.groupby(by='group_memberships_group_id').count()



group_memberships_group  =group_memberships_group.sort_values ('group_memberships_user_id', ascending = False).reset_index()

group_memberships_group.head(50)

groups.head()
school_memberships.head()
school_memberships_group  = school_memberships.groupby(by='school_memberships_school_id').count()



school_memberships_group  =school_memberships_group.sort_values ('school_memberships_user_id', ascending = False).reset_index()

school_memberships_group.head()
school_memberships_group.describe()
school_student = school_memberships.merge(right=students, how = 'outer',

                                            left_on ='school_memberships_user_id',

                                            right_on ='students_id')

school_student_prof = school_student.merge(right=professionals, how = 'outer',

                                            left_on ='school_memberships_user_id',

                                            right_on ='professionals_id')

school_student_prof_group  = school_student_prof.groupby(by='school_memberships_school_id').count()



school_student_prof_group  =school_student_prof_group.sort_values ('school_memberships_user_id', ascending = False).reset_index()

school_student_prof_group.head()