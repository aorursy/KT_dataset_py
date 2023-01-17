import numpy as np
import pandas as pd
import os
import pickle
from bs4 import BeautifulSoup

# for training and prediction
from sklearn import model_selection
from sklearn.ensemble import RandomForestClassifier

# for calculating similarities among text
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from gensim import corpora, models, similarities
# dataset and pre-stored object files
print(os.listdir("../input/pythonquestions"))
print(os.listdir("../input/object-files-for-computed-matrices"))
# Functions for save and load derived data objects
def save_obj(obj, fpath):
    with open(fpath, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(fpath):
    if os.path.exists(fpath):
        with open(fpath, 'rb') as f:
            return pickle.load(f)
    else:
        return None
# load the data
col_types = {'Id': 'int', 
             'OwnerUserId': 'float', 
             'CreationDate': 'str', 
             'ParentId': 'int', 
             'Score': 'int',
             'Title': 'str',
             'Body':'str'}

questions = pd.read_csv('../input/pythonquestions/Questions.csv', encoding = "ISO-8859-1", dtype=col_types)
answers = pd.read_csv('../input/pythonquestions/Answers.csv', encoding = "ISO-8859-1", dtype=col_types)
# question to answer: (ans_id, score, owner_id)
q_to_a = load_obj('../input/object-files-for-computed-matrices/q_to_a.pkl')
if not q_to_a:
    q_to_a = dict()
    for _, row in answers[['Id', 'ParentId', 'Score', 'OwnerUserId']].iterrows():
        q_id = row['ParentId']
        a_id = row['Id']
        a_score = row['Score']
        a_owner_id = row['OwnerUserId'] if not np.isnan(row['OwnerUserId']) else None
        if q_id not in q_to_a:
            q_to_a[q_id] = [(a_id, a_score, a_owner_id)]
        else:
            q_to_a[q_id].append((a_id, a_score, a_owner_id))
            
    save_obj(q_to_a, 'obj/q_to_a.pkl')

# Keep only the questions with 4-10 answers and a distinguishable answer
q_to_a = {k:v for k, v in q_to_a.items() if len(v)>3 and len(v)<11 and max(v, key=lambda x: x[1])[1]>0}
    
# keep only qualified questions
questions = questions[questions['Id'].isin(q_to_a)]
# Keep only answers related to a qualified question
def answer_in_use(answer):
    if answer['ParentId'] in q_to_a:
        for item in q_to_a[answer['ParentId']]:
            if answer['Id'] == item[0]:
                return True
    return False

answers = answers[answers.apply(lambda x: answer_in_use(x), axis=1)]
questions.info()
answers.info()
def strip_code(html):
    bs = BeautifulSoup(html, 'html.parser')
    [s.extract() for s in bs('code')]
    return bs.get_text()

answers = answers.assign(BodyEnglishText = answers['Body'].apply(strip_code))
answers = answers.assign(EnglishCount = answers['BodyEnglishText'].apply(lambda x: len(x.split())))
def count_links(html):
    bs = BeautifulSoup(html, 'html.parser')
    return len(bs.find_all('a'))

answers = answers.assign(LinkCount = answers['Body'].apply(count_links))
def code_length(html):
    bs = BeautifulSoup(html, 'html.parser')
    contents = [tag.text.split() for tag in bs.find_all('code')]
    return sum(len(item) for item in contents)

answers = answers.assign(CodeLength = answers['Body'].apply(code_length))
answers = answers.assign(TotalLength = answers.apply(lambda row: row['EnglishCount'] + row['CodeLength'], axis=1))
def get_order(qid, aid):
    ids = [i[0] for i in q_to_a[qid]]
    return ids.index(aid)+1

answers = answers.assign(PostOrder = answers.apply(lambda row: get_order(row['ParentId'], row['Id']), axis=1))
# calculate reputation scores for users
user_rep = dict()
for ans_list in q_to_a.values():
    for _, score, owner_id in ans_list:
        if owner_id:
            if owner_id in user_rep:
                user_rep[owner_id] += score
            else:
                user_rep[owner_id] = score
def get_reputation(userId):
    if not np.isnan(userId) and userId in user_rep:
        return user_rep[userId]
    else:
        return 0

answers = answers.assign(Reputation = answers['OwnerUserId'].apply(get_reputation))
answers.head(3)
def compute_sim(q_to_a, df_questions, df_answers):
    a_sim = dict()
    tokenizer = RegexpTokenizer(r'\w+')
    print(len(q_to_a))
    c = 0
    for q_id, a_list in q_to_a.items():
        # show progress
        c+=1
        print(str(c) + ' ' + str(len(a_list)), end='\r')
        
        # get split body text for a question and the answers
        q_body = df_questions[df_questions['Id']==q_id].iloc[0]['Body']
        q_body = BeautifulSoup(q_body, 'html.parser').get_text()#.split()
        q_body = tokenizer.tokenize(q_body.lower())
        q_body = [w for w in q_body if w not in stopwords.words('english')]
        
        a_bodies = list()
        for a_id, _, _ in a_list:
            a_body = df_answers[df_answers['Id']==a_id].iloc[0]['Body']
            a_body = BeautifulSoup(a_body, 'html.parser').get_text()#.split()
            a_body = tokenizer.tokenize(a_body.lower())
            a_body = [w for w in a_body if w not in stopwords.words('english')]
            a_bodies.append(a_body)
        
        # apply a series of transformations to the answers: bag-of-word, tf-idf, and lsi
        dictionary = corpora.Dictionary(a_bodies)
        corpus = [dictionary.doc2bow(a) for a in a_bodies]
        tfidf = models.TfidfModel(corpus)
        corpus_tfidf = tfidf[corpus]
        lsi = models.LsiModel(corpus_tfidf, id2word=dictionary, num_topics=4)
        corpus_lsi = lsi[corpus_tfidf]
        
        index = similarities.MatrixSimilarity(corpus_lsi)
        
        # question-to-answer similarity
        q_bow = dictionary.doc2bow(q_body)
        q_lsi = lsi[q_bow]
        q_to_a_sim = index[q_lsi]
        
        # ans-to-ans similarity, excluding the answer itself
        for idx, a_lsi in enumerate(corpus_lsi):
            a_to_a_sim = index[a_lsi]
            a_to_a_sim = [a_to_a_sim[i] for i in range(len(a_to_a_sim)) if i != idx]  # exclude itself
            
            # construct the dictionary a_sim
            a_id = a_list[idx][0]
            sim_to_q = q_to_a_sim[idx]
            max_sim_to_a = max(a_to_a_sim)
            min_sim_to_a = min(a_to_a_sim)
            a_sim[a_id] = (sim_to_q, max_sim_to_a, min_sim_to_a)
    
    return a_sim

# a_sim: a_id -> (sim_to_question, max_sim_to_other_ans, min_sim_to_other_ans)
a_sim = load_obj('../input/object-files-for-computed-matrices/a_sim.pkl')
if not a_sim:
    a_sim = compute_sim(q_to_a, questions, answers)
    save_obj(a_sim, '../input/object-files-for-computed-matrices/a_sim.pkl')
answers = answers.assign(SimToQ = answers['Id'].apply(lambda a_id: a_sim[a_id][0]))
answers = answers.assign(MaxSimToA = answers['Id'].apply(lambda a_id: a_sim[a_id][1]))
answers = answers.assign(MinSimToA = answers['Id'].apply(lambda a_id: a_sim[a_id][2]))
answers.head(3)
def is_answer(qid, aid):
    if aid == max(q_to_a[qid], key=lambda item: item[1])[0]:
        return 1
    else:
        return 0
    
answers = answers.assign(IsAnswer = answers.apply(lambda row: is_answer(row['ParentId'], row['Id']), axis=1))
answers.head(3)
q_train, q_test = model_selection.train_test_split(questions, test_size=0.2, random_state=42)
a_train = answers[answers['ParentId'].isin(q_train['Id'])]
a_test = answers[answers['ParentId'].isin(q_test['Id'])]
x_train = a_train[['TotalLength','LinkCount', 'CodeLength', 'PostOrder', 'Reputation', 'SimToQ', 'MaxSimToA', 'MinSimToA']]
y_train = a_train[['IsAnswer']]
x_test = a_test[['Id', 'TotalLength','LinkCount', 'CodeLength', 'PostOrder', 'Reputation', 'SimToQ', 'MaxSimToA', 'MinSimToA']]
y_test = a_test[['IsAnswer']]
rf_classifier = RandomForestClassifier(
    n_estimators=1000, min_samples_leaf=4, n_jobs=-1, oob_score=True, random_state=42)

rf_classifier.fit(x_train, y_train.values.ravel())
y_pred = rf_classifier.predict_proba(x_test.iloc[:, 1:])
# question-based accuracy, i.e., the percentage that whether the question is correctly answered
def get_accuracy(q_ids, a_ids, prob_pred):
    a_to_prob = dict()
    for idx, a_id in enumerate(a_ids):
        prob = prob_pred[:, 1][idx]
        a_to_prob[a_id] = prob
        
    count = 0
    for q_id in q_ids:
        right_answer = max(q_to_a[q_id], key=lambda item: item[1])[0]
        predict_answer = 0
        highest_score = 0
        for a_id, score, _ in q_to_a[q_id]:
            pred_score = a_to_prob[a_id]
            if pred_score > highest_score:
                predict_answer = a_id
                highest_score = pred_score
        if right_answer==predict_answer:
            count += 1
    return count/len(q_ids)

print('Random Forest accuracy:', get_accuracy(q_test['Id'].tolist(), a_test['Id'].tolist(), y_pred))
def baseline(q_ids, a_val):
    count_first = 0
    count_last = 0
    count_long = 0
    for q_id in q_ids:
        right_answer = max(q_to_a[q_id], key=lambda item: item[1])[0]
        
        first_answer = q_to_a[q_id][0][0]
        if right_answer==first_answer:
            count_first += 1
        
        last_answer = q_to_a[q_id][-1][0]
        if right_answer==last_answer:
            count_last += 1
            
        longest_answer = -1
        max_length = 0
        for a_id, score, _ in q_to_a[q_id]:
            leng = a_val[a_val.Id==a_id].iloc[0]['TotalLength']
            if leng > max_length:
                longest_answer = a_id
                max_length = leng
        if right_answer==longest_answer:
            count_long += 1
    print('Baseline for the first answer:', count_first/len(q_ids))
    print('Baseline for the last answer:', count_last/len(q_ids))
    print('Baseline for the longest answer:', count_long/len(q_ids))

baseline(q_test['Id'].tolist(), a_test)
importances = rf_classifier.feature_importances_
std = np.std([tree.feature_importances_ for tree in rf_classifier.estimators_], axis=0)
indices = np.argsort(importances)[::-1]

print("Feature ranking:")
for i in range(x_test.shape[1]-1):
    print("%d. feature %d (%f) (%s)" % (i + 1, indices[i], importances[indices[i]], list(x_test.columns.values)[indices[i]+1]))