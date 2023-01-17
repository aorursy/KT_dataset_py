import numpy as np
import pandas as pd
df_train = pd.read_csv('../input/quora-question-pairs/train.csv', encoding='utf-8')
df_train['id'] = df_train['id'].apply(str)
df_test = pd.read_csv('../input/quora-question-pairs/test.csv', encoding='utf-8')
df_test['test_id'] = df_test['test_id'].apply(str)
df_all = pd.concat((df_train, df_test))
df_all['question1'].fillna('', inplace=True)
df_all['question2'].fillna('', inplace=True)
from sklearn.feature_extraction.text import CountVectorizer
import itertools
counts_vectorizer = CountVectorizer(max_features=10000-1).fit(
    itertools.chain(df_all['question1'], df_all['question2']))
other_index = len(counts_vectorizer.vocabulary_)
import re
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
words_tokenizer = re.compile(counts_vectorizer.token_pattern)
def create_padded_seqs(texts, max_len=10):
    seqs = texts.apply(lambda s: 
        [counts_vectorizer.vocabulary_[w] if w in counts_vectorizer.vocabulary_ else other_index
         for w in words_tokenizer.findall(s.lower())])
    return pad_sequences(seqs, maxlen=max_len)
X1_train, X1_val, X2_train, X2_val, y_train, y_val = \
    train_test_split(create_padded_seqs(df_all[df_all['id'].notnull()]['question1']), 
                     create_padded_seqs(df_all[df_all['id'].notnull()]['question2']),
                     df_all[df_all['id'].notnull()]['is_duplicate'].values,
                     stratify=df_all[df_all['id'].notnull()]['is_duplicate'].values,
                     test_size=0.3, random_state=1989)
import keras.layers as lyr
from keras.models import Model
input1_tensor = lyr.Input(X1_train.shape[1:])
input2_tensor = lyr.Input(X2_train.shape[1:])

words_embedding_layer = lyr.Embedding(X1_train.max() + 1, 100)
seq_embedding_layer = lyr.LSTM(256, activation='tanh')

seq_embedding = lambda tensor: seq_embedding_layer(words_embedding_layer(tensor))

merge_layer = lyr.multiply([seq_embedding(input1_tensor), seq_embedding(input2_tensor)])

dense1_layer = lyr.Dense(16, activation='sigmoid')(merge_layer)
ouput_layer = lyr.Dense(1, activation='sigmoid')(dense1_layer)

model = Model([input1_tensor, input2_tensor], ouput_layer)

model.compile(loss='binary_crossentropy', optimizer='adam')
model.summary()
model.fit([X1_train, X2_train], y_train, 
          validation_data=([X1_val, X2_val], y_val), 
          batch_size=128, epochs=6, verbose=2)
features_model = Model([input1_tensor, input2_tensor], merge_layer)
features_model.compile(loss='mse', optimizer='adam')
F_train = features_model.predict([X1_train, X2_train], batch_size=128)
F_val = features_model.predict([X1_val, X2_val], batch_size=128)
import xgboost as xgb
dTrain = xgb.DMatrix(F_train, label=y_train)
dVal = xgb.DMatrix(F_val, label=y_val)
xgb_params = {
    'objective': 'binary:logistic',
    'booster': 'gbtree',
    'eval_metric': 'logloss',
    'eta': 0.1, 
    'max_depth': 9,
    'subsample': 0.9,
    'colsample_bytree': 1 / F_train.shape[1]**0.5,
    'min_child_weight': 5,
    'silent': 1
}
bst = xgb.train(xgb_params, dTrain, 1000,  [(dTrain,'train'), (dVal,'val')], 
                verbose_eval=10, early_stopping_rounds=10)
import joblib
#save model
joblib.dump(model, 'model') 
joblib.dump(bst, 'bst_model') 
joblib.dump(features_model, 'f_model') 

import joblib
model = joblib.load('../input/question-similarity-using-lstm-embedding-bd119d/model')
features_model = joblib.load('../input/question-similarity-using-lstm-embedding-bd119d/f_model')
bst = joblib.load('../input/question-similarity-using-lstm-embedding-bd119d/bst_model')
#ignore
X1_test = create_padded_seqs(df_all[df_all['test_id'].notnull()]['question1'])
X2_test = create_padded_seqs(df_all[df_all['test_id'].notnull()]['question2'])
#ignore
question2 = df_all[df_all['test_id'].notnull()]['question2']
for i in range(0,question2.size):
	question2.values[i] = "How do you charge a laptop without a charger?"
X2_test = create_padded_seqs(question2)
#ignore
df_so1 = pd.read_csv('../input/posts2017to20181/posts2017to20182Fposts000000000000.csv', names=['Id', 'Body'], encoding='utf-8')
df_so1['Id'] = df_so1['Id'].apply(str)
df_so2 = pd.read_csv('../input/posts2017to20181/posts2017to20182Fposts000000000001.csv', names=['Id', 'Body'], encoding='utf-8')
df_so2['Id'] = df_so2['Id'].apply(str)
df_so3 = pd.read_csv('../input/posts2017to20181/posts2017to20182Fposts000000000002.csv', names=['Id', 'Body'], encoding='utf-8')
df_so3['Id'] = df_so3['Id'].apply(str)
df_so = pd.concat([df_so1,df_so2,df_so3])
df_so['Id'] = df_so['Id'].apply(str)
df_so['Body'] = df_so['Body'].apply(str)
df_so = pd.read_csv('../input/posts2017to20181/RegexPostsTitle.csv', encoding='utf-8')

df_so['Id'] = df_so['Id'].apply(str)
df_so['Body'] = df_so['Body'].apply(str)
df_so['Title'] = df_so['Title'].apply(str)
df_so = pd.read_csv('../input/posts2017to20181/RegexComments.csv', encoding='utf-8')

df_so['c_Id'] = df_so['c_Id'].apply(str)
df_so['c_Text'] = df_so['c_Text'].apply(str)
X1_test = create_padded_seqs(df_so[df_so['Id'].notnull()]['Title'])
X2_test = X1_test
question_comment = "Thanks for all your input.  I am looking for a regex to validate my user's password.  My users are allowed to enter a password between 8-15 characters long and it could be either alphamumeric or nonalphanumberic character.  Does that make more sense now?"
question2 = df_so[df_so['Id'].notnull()]['Title']
for i in range(0,question2.size):
	question2.values[i] = question_comment
X2_test = create_padded_seqs(question2)

F_test = features_model.predict([X1_test, X2_test], batch_size=128)
import xgboost as xgb
dTest = xgb.DMatrix(F_test)
import spacy
nlp = spacy.load('en')
ques = nlp(question_comment)
#doc2 = nlp(u'Hello hi there!')
#doc3 = nlp(u'Hey whatsup?')

#print doc1.similarity(doc2) # 0.999999954642
#print doc2.similarity(doc3) # 0.699032527716
#print doc1.similarity(doc3) # 0.699032527716
df_sub = pd.DataFrame({
        'test_id': df_so[df_so['Id'].notnull()]['Id'].values,
        'is_duplicateLSTM': bst.predict(dTest, ntree_limit=bst.best_ntree_limit),
        'is_duplicateCos': bst.predict(dTest, ntree_limit=bst.best_ntree_limit),
        'question' : df_so['Title'].values
    })#.set_index('Id')
def addCosine(v):
    v['is_duplicateCos'] = ques.similarity(nlp(v['question']))
    return v

def get_pair_score(terms1, terms2):
    sims = []
    for word1 in terms1:
        word1_sim = []
        try:
            syn1 = wn.synsets(word1)[0]
        except:  #if wordnet is not able to find a synset for word1
            sims.append([0 for i in range(0, len(terms2))])
            continue
        for word2 in terms2:
            try:
                syn2 = wn.synsets(word2)[0]
            except: #if wordnet is not able to find a synset for word2
                word1_sim.append(0)
                continue
            word_similarity = syn1.wup_similarity(syn2)
            word1_sim.append(word_similarity)
        sims.append(word1_sim)
        
    word1_score = 0
    for i in range(0, len(terms1), 1):
        try:
            word1_score += max(sims[i])
        except:
            continue
    word1_score /= len(terms1) #Averaging over all terms
        
    word2_score = 0
    for i in range(0, len(terms2), 1):
        try:
            word2_score += max([j[i] for j in sims])
        except:
            continue
    word2_score /= len(terms2)
    pair_score = (word1_score + word2_score)/2
    #print(pair_score)
    return pair_score

def addWordnet(v):
    v['is_duplicateCos'] = ques.similarity(nlp(v['question']))
    return v

df_cp = df_sub
df_cp.apply(addCosine, axis=1)
df_cp=df_sub.apply(addCosine, axis=1)
df_cp.head()
df_sub['is_duplicate'].hist(bins=100)
df_cp['is_duplicate'].hist(bins=100)

p = df_sub[df_sub['is_duplicate'] >= 0.6]
#for each_id in df_all['test_id'].values:
#        if each_id in p.index.values:
#            print(df_all[df_all['test_id'] == each_id]['question1'])
print(question_comment)
print("====================================================")
for each_id in df_so['Id'].values:
        if each_id in p['test_id'].values:
            print(df_so[df_so['Id'] == each_id]['Title'].values)
            print("===")

df_cp
p = df_cp[df_cp['is_duplicate'] >= 0.70]
#for each_id in df_all['test_id'].values:
#        if each_id in p.index.values:
#            print(df_all[df_all['test_id'] == each_id]['question1'])
print(question_comment)
print("====================================================")
print(p['question'].values)
print(question_comment)
print("====================================================")
p = df_cp[df_cp['is_duplicateCos'] >= 0.70]
print("Cosine similarity")
print(p['question'].values)
p = df_cp[df_cp['is_duplicateLSTM'] >= 0.70]
print("====================================================")
print("Similarity using LSTM Embedding")
print(p['question'].values)
df_sub['is_duplicateLSTM'].hist(bins=100)
df_sub['is_duplicateCos'].hist(bins=100)