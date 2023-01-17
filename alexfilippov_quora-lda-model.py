import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords 

from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize 


from nltk.stem import WordNetLemmatizer
wordnet_lemmatizer = WordNetLemmatizer()

import gensim

from gensim import models, corpora

from sklearn.decomposition import TruncatedSVD

from sklearn.tree import DecisionTreeClassifier
data_train_gen = pd.read_csv("../input/train.csv")
data_test = pd.read_csv("../input/test.csv")
print("Train datasets shape:", data_train_gen.shape)
print("Test datasets shape:", data_test.shape)
num = 653826
bad_df = data_train_gen[data_train_gen.target == 1]
good_df = data_train_gen[data_train_gen.target == 0][:num]
data_train = pd.concat([bad_df, good_df])
from sklearn.utils import shuffle
data_train = shuffle(data_train)
data_train = data_train[:136520]
quiestion_words = ['what','when','why','which','who','how', 'whose', 'whome', 'people', 'i', 
                  'n\'t','\'s','like','get','would','would']
stop_signs = [',', '.', '"', ':', ')', '(', '-', '!', '?', '|', ';', "'", '$', '&', '/', '[', ']', '>', '%', '=', '#', '*', '+', '\\', '•',  '~', '@', '£', 
 '·', '_', '{', '}', '©', '^', '®', '`',  '<', '→', '°', '€', '™', '›',  '♥', '←', '×', '§', '″', '′', 'Â', '█', '½', 'à', '…', 
 '“', '★', '”', '–', '●', 'â', '►', '−', '¢', '²', '¬', '░', '¶', '↑', '±', '¿', '▾', '═', '¦', '║', '―', '¥', '▓', '—', '‹', '─', 
 '▒', '：', '¼', '⊕', '▼', '▪', '†', '■', '’', '▀', '¨', '▄', '♫', '☆', 'é', '¯', '♦', '¤', '▲', 'è', '¸', '¾', 'Ã', '⋅', '‘', '∞', 
 '∙', '）', '↓', '、', '│', '（', '»', '，', '♪', '╩', '╚', '³', '・', '╦', '╣', '╔', '╗', '▬', '❤', 'ï', 'Ø', '¹', '≤', '‡', '√']
stop_words = stopwords.words('english')
# stop_words = stop_words.extend(quiestion_words)
for w in quiestion_words:
    stop_words.append(w)
for w in stop_signs:
    stop_words.append(w)
cleaned_questions_train = []
 
for sentence in data_train['question_text']:
    new_sentence = [wordnet_lemmatizer.lemmatize(w).lower() for w in word_tokenize(sentence)]
    new_sentence = [w for w in new_sentence if w not in stop_words]
    new_sentence = [w for w in new_sentence if len(w)>3]
         
    clean = ' '.join(new_sentence)    
   
    cleaned_questions_train.append(clean)

cleaned_questions_test = []
for sentence in data_test['question_text']:
    new_sentence = [wordnet_lemmatizer.lemmatize(w).lower() for w in word_tokenize(sentence)]
    new_sentence = [w for w in new_sentence if w not in stop_words]
    new_sentence = [w for w in new_sentence if len(w)>3]
         
    clean = ' '.join(new_sentence)    
   
    cleaned_questions_test.append(clean)
data_train.insert(loc=0, column="debugged_questions", value=cleaned_questions_train)
data_test.insert(loc=0, column="debugged_questions", value=cleaned_questions_test)
data_train.fillna('', inplace = True)
data_test.fillna('', inplace = True)
train = data_train.ix[:, (1,0, 3)]
test = data_test.ix[:, (1,0)]
train.shape
train['tokens_list'] = train.debugged_questions.apply(lambda x: x.split())
test['tokens_list'] = test.debugged_questions.apply(lambda x: x.split())
word2vec_path = "../input/embeddings/GoogleNews-vectors-negative300/GoogleNews-vectors-negative300.bin"
word2vec = gensim.models.KeyedVectors.load_word2vec_format(word2vec_path, binary=True)
len(test.tokens_list)
def get_average_word2vec(tokens_list, vector, generate_missing=False, k=300):
    if len(tokens_list)<1:
        return np.zeros(k)
    if generate_missing:
        vectorized = [vector[word] if word in vector else np.random.rand(k) for word in tokens_list]
    else:
        vectorized = [vector[word] if word in vector else np.zeros(k) for word in tokens_list]
    length = len(vectorized)
    summed = np.sum(vectorized, axis=0)
    averaged = np.divide(summed, length)
    return averaged

def get_word2vec_embeddings(vectors, clean_questions, generate_missing=False):
    embeddings = clean_questions['tokens_list'].apply(lambda x: get_average_word2vec(x, vectors, 
                                                                                generate_missing=generate_missing))
    return embeddings.tolist()


test_embeddings = get_word2vec_embeddings(word2vec, test)
train_embeddings = get_word2vec_embeddings(word2vec, train)
test_embeddings = pd.DataFrame(test_embeddings)
train_embeddings = pd.DataFrame(train_embeddings)
# ну или импортирую заранее обученную
# ldamodel = models.ldamodel.LdaModel.load("ldamodel3_lkcd")

gen = pd.concat([train[train.columns[[0,1,3]]], test], axis = 0)
tokens = gen.tokens_list.tolist()
dictionary = corpora.Dictionary(tokens)
corpus = []
for token in tokens: corpus.append(dictionary.doc2bow(token))

tetok = test.tokens_list.tolist()
trtok = train.tokens_list.tolist()
# wtok = weird_tok.tolist()
tecor = []
trcor = []
# wecor = []
for token in tetok: tecor.append(dictionary.doc2bow(token))
for token in trtok: trcor.append(dictionary.doc2bow(token))
# for token in wtok: wecor.append(dictionary2.doc2bow(token))
np.random.seed(76543)
%time ldamodel = models.ldamodel.LdaModel(corpus, id2word=dictionary, num_topics=200, passes=3)
test_topics = ldamodel.get_document_topics(tecor)
train_topics = ldamodel.get_document_topics(trcor)
# weird_topics = ldamodel.get_document_topics(wecor)
nums1 = []
for list in [ldamodel.get_document_topics(corp) for corp in trcor]:
    num1 = []
    for tup in list:
        num1.append(tup)
    nums1.append(num1)
nums2 = []
nn = []
for list in [ldamodel.get_document_topics(corp) for corp in tecor]:
    num2 = []
    for tup in list:
        num2.append(tup)
    nums2.append(num2)
X_ps_tr = np.ndarray([len(trcor), 200])
X_ps_te = np.ndarray([len(tecor), 200])
for i in range(len(nums1)):
    if i == 136523: pass
    for j in range(len(nums1[i])):
        X_ps_tr[i, nums1[i][j][0]] = nums1[i][j][1]
        
        
for i in range(len(nums2)):
    for j in range(len(nums2[i])):
        X_ps_te[i, nums2[i][j][0]] = nums2[i][j][1]
X_ps_te = pd.DataFrame(X_ps_te)
X_ps_tr = pd.DataFrame(X_ps_tr)
train.shape, train_embeddings.shape, X_ps_tr.shape, test.shape, test_embeddings.shape, X_ps_te.shape
train.index = pd.RangeIndex(0, train.shape[0])
trainig = pd.concat([train.qid, train.target, train_embeddings, X_ps_tr], axis = 1)
testing = pd.concat([test.qid, test_embeddings, X_ps_te], axis = 1)
trainig.shape, testing.shape
testing.head()
lsa = TruncatedSVD(n_components=100)
lsa.fit(trainig[trainig.columns[2:]])
lsa_scores_train = lsa.transform(trainig[trainig.columns[2:]])
lsa_scores_test = lsa.transform(testing[testing.columns[1:]])
lsa_scores_train = pd.DataFrame(lsa_scores_train)
lsa_scores_test = pd.DataFrame(lsa_scores_test)
y = trainig.target
clf = DecisionTreeClassifier(max_depth = 12, min_samples_leaf = 20)
clf.fit(lsa_scores_train,y)
sub_df = pd.DataFrame({'qid':testing.qid.values})
sub_df['prediction'] = clf.predict(lsa_scores_test)
sub_df.to_csv('submission.csv', index=False)


