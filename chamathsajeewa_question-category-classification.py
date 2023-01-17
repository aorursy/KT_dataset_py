#print paths to input files
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
        
for dirname, _, filenames in os.walk('/kaggle/working'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
#imports
import nltk
from nltk import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk import pos_tag

import pandas
import matplotlib.pyplot as plt
import re
import numpy as np
import sys

import tensorflow as tf
from scipy.sparse import hstack

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer 
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score

from gensim.models import Word2Vec
from gensim.models.fasttext import FastText
from gensim.models.keyedvectors import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.callbacks import EarlyStopping

np.set_printoptions(threshold=sys.maxsize)

####################################################
#load data
####################################################

data_frame = pandas.read_csv('/kaggle/input/travelquestionsdataset/5000TravelQuestionsDataset.csv',encoding='ISO-8859-1',header=None, names=['Question','Category','SubCategory']);
print(data_frame.head());
#######################################################
#data preprocessing
#######################################################

#clean categorical values
data_frame.replace(to_replace='TGU\n', value='TGU', inplace=True)
data_frame.replace(to_replace='TTD\n', value='TTD', inplace=True)
data_frame.replace(to_replace='\nENT', value='ENT', inplace=True)

#explore categories
fig = plt.figure(figsize=(6,8));
data_frame.groupby('Category').Question.count().plot.bar(ylim=0);
plt.show();

#clean sub categorical values
data_frame.replace(to_replace='WTHTMP\n', value='WTHTMP', inplace=True)
data_frame.replace(to_replace='\nTGULAU', value='TGULAU', inplace=True)
data_frame.replace(to_replace='TRSOTH\n', value='TRSOTH', inplace=True)
data_frame.replace(to_replace='FODBAK\n', value='FODBAK', inplace=True)
data_frame.replace(to_replace='TRSAIR\n', value='TRSAIR', inplace=True)
data_frame.replace(to_replace='TGUCIG\n', value='TGUCIG', inplace=True)
data_frame.replace(to_replace='TTDOTH\n', value='TTDOTH', inplace=True)
data_frame.replace(to_replace='WTHOTH\n', value='WTHOTH', inplace=True)
data_frame.replace(to_replace='TTDSIG\n', value='TTDSIG', inplace=True)
data_frame.replace(to_replace='TGUOTH\n', value='TGUOTH', inplace=True)
data_frame.replace(to_replace='TTDSHP\n', value='TTDSHP', inplace=True)
data_frame.replace(to_replace='TRSROU\n', value='TRSROU', inplace=True)
data_frame.replace(to_replace='TTDSPO\n', value='TTDSPO', inplace=True)
data_frame.replace(to_replace='\nACMOTH', value='ACMOTH', inplace=True)
data_frame.replace(to_replace='ACMOTH\n', value='ACMOTH', inplace=True)
data_frame.replace(to_replace='\nWTHOTH', value='WTHOTH', inplace=True)

#explore sub categories
fig = plt.figure(figsize=(6,8));
data_frame.groupby('SubCategory').Question.count().plot.bar(ylim=0);
plt.show();

#category values encoding
encoder = LabelEncoder();
category_labels = data_frame["Category"].tolist();
encoded_category_labels = encoder.fit_transform(category_labels).tolist();
print(category_labels[:10]);
print(print(set(encoded_category_labels)));
print(len(encoded_category_labels))

#sub category values encoding
encoder = LabelEncoder();
sub_category_labels = data_frame["SubCategory"].tolist();
encoded_sub_category_labels = encoder.fit_transform(sub_category_labels).tolist();
print(sub_category_labels[:10]);
print(print(set(encoded_sub_category_labels)));
print(len(encoded_sub_category_labels))


class TextPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.stemmer = PorterStemmer()

    def fit(self, X, y=None):
        return self

    def inverse_transform(self, X):
        return X

    def transform(self, X):
        return [self.preprocess(question) for question in X]

    def preprocess(self, question):
        # Remove special characters
        question = re.sub(r'\W', ' ', question)

        # Remove single characters
        question = re.sub(r'\s+[a-zA-Z]\s+', ' ', question)

        # Remove single characters from the start of sentences
        question = re.sub(r'\^[a-zA-Z]\s+', ' ', question) 

        # Substitute multiple spaces with single space
        question = re.sub(r'\s+', ' ', question, flags=re.I)

        # Convert to Lowercase
        question = question.lower()
        
        # Stop words are not removed

        #Stemming/lemmatization
        question=word_tokenize(question)
        
        #document = [self.stemmer.stem(word) for word in document]
        question = [self.lemmatizer.lemmatize(word) for word in question] #stemming can often create non-existent words, whereas lemmas are actual words.
        question = ' '.join(question)
        return question
    
# pre process questions using Text preprocessor
preprocessor = TextPreprocessor()
preprocessed_questions = preprocessor.transform(data_frame["Question"])
print(preprocessed_questions[:5])

#POS -https://stackoverflow.com/questions/24002485/python-how-to-use-pos-part-of-speech-features-in-scikit-learn-classfiers-svm
#preprocess + add pos tags
class POSNLTKPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.stemmer = PorterStemmer()
        self.basePreprocessor = TextPreprocessor()

    def fit(self, X, y=None):
        return self

    def inverse_transform(self, X):
        return X

    def transform(self, X):
        return [self.preprocess(question) for question in X]

    def preprocess(self, question):

        question = self.basePreprocessor.preprocess(question)
        
        question=word_tokenize(question)
        pos_tagged_question=pos_tag(question) 
        
        pos_combined_question =[]
        for word in pos_tagged_question:
            pos_combined_question.append(word[0] + "_" + word[1])
        pos_combined_question = ' '.join(pos_combined_question)
        return pos_combined_question;
    
#preprocess questions using POSNLTK Preprocessor
preprocessor = POSNLTKPreprocessor()
pos_tagged_preprocessed_questions = preprocessor.transform(data_frame["Question"])
print(pos_tagged_preprocessed_questions[:5])

#extract named entities of each question
class NENLTKPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.stemmer = PorterStemmer()

    def fit(self, X, y=None):
        return self

    def inverse_transform(self, X):
        return X

    def transform(self, X):
        return [self.preprocess(question) for question in X]

    def preprocess(self, question):
        # Remove special characters
        question = re.sub(r'\W', ' ', question)

        #below not done for named entity
        # remove single characters
        #question = re.sub(r'\s+[a-zA-Z]\s+', ' ', question)

        # Remove single characters from the start of sentences
        #question = re.sub(r'\^[a-zA-Z]\s+', ' ', question) 

        # Substitute multiple spaces with single space
        #question = re.sub(r'\s+', ' ', question, flags=re.I)

        # Convert to Lowercase - not perform as this affect to the NER
        #question = question.lower()
        
        # Stop words are not removed

        #Stemming/lemmatization
        question=word_tokenize(question)
        
        #document = [self.stemmer.stem(word) for word in document]
        question = [self.lemmatizer.lemmatize(word) for word in question]
        question = ' '.join(question)
        
        question=word_tokenize(question)
        pos_tagged_question=pos_tag(question) 
        ne_tagged_question = nltk.ne_chunk(pos_tagged_question, binary = False)
        
        named_entities = []
        for tagged_tree in ne_tagged_question:
            if hasattr(tagged_tree, 'label'):
              entity_name = ' '.join(c[0] for c in tagged_tree.leaves()) 
              entity_type = tagged_tree.label() # get NE category
              named_entities.append((entity_type,entity_name))

        ne_combined_question =[]
        for word in named_entities:
            ne_combined_question.append(word[0])
        ne_combined_question = ' '.join(ne_combined_question)
        return ne_combined_question;
    
#preprocess questions using POSNLTK Preprocessor
preprocessor = NENLTKPreprocessor()
ne_tagged_preprocessed_questions = preprocessor.transform(data_frame["Question"])
print(ne_tagged_preprocessed_questions[:5])

#extract word shape of each question
def wordshape(text):
    t1 = re.sub('[A-Z]', 'X',text)
    t2 = re.sub('[a-z]', 'x', t1)
    return re.sub('[0-9]', 'd', t2)

class WSNLTKPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.stemmer = PorterStemmer()

    def fit(self, X, y=None):
        return self

    def inverse_transform(self, X):
        return X

    def transform(self, X):
        return [self.preprocess(question) for question in X]

    def preprocess(self, question):
        # Remove special characters
        question = re.sub(r'\W', ' ', question)

        # remove single characters
        question = re.sub(r'\s+[a-zA-Z]\s+', ' ', question)

        # Remove single characters from the start of sentences
        question = re.sub(r'\^[a-zA-Z]\s+', ' ', question) 

        # Substitute multiple spaces with single space
        question = re.sub(r'\s+', ' ', question, flags=re.I)

        # Convert to Lowercase
        #question = question.lower()
        
        # Stop words are not removed

        #Stemming/lemmatization
        question=word_tokenize(question)
        
        #document = [self.stemmer.stem(word) for word in document]
        question = [self.lemmatizer.lemmatize(word) for word in question]
        question = ' '.join(question)
        
        question=word_tokenize(question)
        
        word_shape_question =[]
        for word in question:
            word_shape_question.append(wordshape(word))
        word_shape_question = ' '.join(word_shape_question)
        return word_shape_question;
    
#preprocess questions using POSNLTK Preprocessor
preprocessor = WSNLTKPreprocessor()
ws_preprocessed_questions = preprocessor.transform(data_frame["Question"])
print(ws_preprocessed_questions[:5])
####################################################################################
#  1. A traditional ML classifier
####################################################################################

#################################
# feature extraction
#################################

#extract text features
#1. word occurances (uni grams and bi grams and trigrams)
count_vect = CountVectorizer(analyzer = 'word',ngram_range=(1,3));
word_occurance_counts = count_vect.fit_transform(preprocessed_questions);

#word occurances (uni grams with pos tags)
count_vect = CountVectorizer(analyzer = 'word',ngram_range=(1,1));
pos_word_occurance_counts = count_vect.fit_transform(pos_tagged_preprocessed_questions);

#word occurances (named entities)
count_vect = CountVectorizer(analyzer = 'word',ngram_range=(1,1));
ne_word_occurance_counts = count_vect.fit_transform(ne_tagged_preprocessed_questions);

#word occurances (word shapes)
count_vect = CountVectorizer(analyzer = 'word',ngram_range=(1,1));
ws_word_occurance_counts = count_vect.fit_transform(ws_preprocessed_questions);

#extract text features
# 2. tf-idf
#Transform a count matrix to a normalized tf or tf-idf representation
tfidf_transformer = TfidfTransformer()
word_tf_idf_counts = tfidf_transformer.fit_transform(word_occurance_counts)
print(word_tf_idf_counts.shape)

tfidf_transformer = TfidfTransformer()
word_pos_tf_idf_counts = tfidf_transformer.fit_transform(pos_word_occurance_counts)
print(word_pos_tf_idf_counts.shape)

tfidf_transformer = TfidfTransformer()
word_ne_tf_idf_counts = tfidf_transformer.fit_transform(ne_word_occurance_counts)
print(word_ne_tf_idf_counts.shape)

tfidf_transformer = TfidfTransformer()
word_ws_tf_idf_counts = tfidf_transformer.fit_transform(ws_word_occurance_counts)
print(word_ws_tf_idf_counts.shape)

combined_features = hstack([word_tf_idf_counts,word_pos_tf_idf_counts,word_ne_tf_idf_counts,word_ws_tf_idf_counts],format='csr')
print(combined_features.shape)


#####################
#grid search
#####################

#find best parameters for categories and sub categories
parameters = {'alpha':[1e-2,1e-3,1e-4],'penalty':['l2', 'l1'],'loss':['hinge','log'],'max_iter':[5,10,15,20]}

classifier = SGDClassifier(random_state=42, tol=None)
grid = GridSearchCV(classifier, parameters, cv = 10, scoring = 'accuracy')
grid.fit(combined_features, encoded_category_labels)

print('======Best Parameters for Category Labels=====')
print(grid.best_params_)
print(grid.best_score_)
print('==============================================')

classifier = SGDClassifier(random_state=42, tol=None)
grid = GridSearchCV(classifier, parameters, cv = 10, scoring = 'accuracy')
grid.fit(combined_features, encoded_sub_category_labels)

print('======Best Parameters for Sub Category Labels=====')
print(grid.best_params_)
print(grid.best_score_)
print('==================================================')
np.set_printoptions(precision=2)
np.set_printoptions(formatter={'int': '{:,}'.format})
np.set_printoptions(linewidth=200)
###########
#model evaluation
##########

# SGD with tradtional features

print("===================================================================")
print("SGD for Category classification with traditional features")
print("====================================================================")

kfold = StratifiedKFold(n_splits=10, shuffle=True)
iteration = 0;
cv_accuracy_scores = []
cv_conf_matrices = []
cv_recall_scores = []
cv_precision_scores = []
cv_f1_scores = []

for train_index, test_index in kfold.split(combined_features,encoded_category_labels):

    iteration = iteration + 1
    
    print("Cross Validation =================================================================== ", iteration)

    train_questions,test_questions=combined_features[train_index],combined_features[test_index]
    train_category_labels,test_category_labels=np.array(encoded_category_labels)[train_index],np.array(encoded_category_labels)[test_index]
    
    question_category_classification_pipeline = Pipeline([
     ('Classifier', SGDClassifier(loss='hinge', penalty='l2',
                          alpha=0.0001, random_state=42,
                           max_iter=15, tol=None)),])
    
    question_category_classification_pipeline.fit(train_questions, train_category_labels) 
    test_predicted_category_labels=question_category_classification_pipeline.predict(test_questions)
    
    accuracy = accuracy_score(test_category_labels, test_predicted_category_labels)
    matrix = confusion_matrix(test_category_labels, test_predicted_category_labels)
    recall = recall_score(test_category_labels, test_predicted_category_labels,average=None)
    precision = precision_score(test_category_labels, test_predicted_category_labels,average=None)
    f1 = f1_score(test_category_labels, test_predicted_category_labels,average=None)
    
    cv_accuracy_scores.append(accuracy)
    cv_conf_matrices.append(matrix)
    cv_recall_scores.append(recall)
    cv_precision_scores.append(precision)
    cv_f1_scores.append(f1)

print("Results for categories:================================================================")
print("Accuracy: %.2f" % np.mean(cv_accuracy_scores))
print("Recall:")
print(np.mean(np.array(cv_recall_scores), axis=0)) 
print("Precision:")
print(np.mean(np.array(cv_precision_scores), axis=0))
print("F1 Score:")
print(np.mean(np.array(cv_f1_scores), axis=0))
print("Confusion Matrix:")
print(np.sum(cv_conf_matrices, axis=0))

print("=======================================================================================")
print("SGD for Sub Category classification with traditional features")
print("=======================================================================================")

kfold = StratifiedKFold(n_splits=10, shuffle=True)
iteration = 0;
cv_accuracy_scores = []
cv_conf_matrices = []
cv_recall_scores = []
cv_precision_scores = []
cv_f1_scores = []


for train_index, test_index in kfold.split(combined_features,encoded_sub_category_labels):

    iteration = iteration + 1
    
    print("Cross Validation =================================================================== ", iteration)

    train_questions,test_questions=combined_features[train_index],combined_features[test_index]
    train_sub_category_labels,test_sub_category_labels=np.array(encoded_sub_category_labels)[train_index],np.array(encoded_sub_category_labels)[test_index]
    
    question_category_classification_pipeline = Pipeline([
     ('Classifier', SGDClassifier(loss='hinge', penalty='l2',
                          alpha=0.0001, random_state=42,
                           max_iter=20, tol=None)),], verbose=True)
    
    question_category_classification_pipeline.fit(train_questions, train_sub_category_labels) 
    test_predicted_sub_category_labels=question_category_classification_pipeline.predict(test_questions)
    
    accuracy = accuracy_score(test_sub_category_labels, test_predicted_sub_category_labels)
    required_labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62]
    matrix = confusion_matrix(test_sub_category_labels, test_predicted_sub_category_labels, labels = required_labels)
    recall = recall_score(test_sub_category_labels, test_predicted_sub_category_labels,average=None, labels = required_labels, zero_division = 0)
    precision = precision_score(test_sub_category_labels, test_predicted_sub_category_labels,average=None, labels = required_labels, zero_division = 0)
    f1 = f1_score(test_sub_category_labels, test_predicted_sub_category_labels,average=None, labels = required_labels, zero_division = 0)
    
    cv_accuracy_scores.append(accuracy)
    cv_conf_matrices.append(matrix)
    cv_recall_scores.append(recall)
    cv_precision_scores.append(precision)
    cv_f1_scores.append(f1)

print("Results for sub categories:==============================================================")
print("Accuracy: %.2f" % np.mean(cv_accuracy_scores))
print("Recall:")
print(np.mean(np.array(cv_recall_scores), axis=0)) 
print("Precision:")
print(np.mean(np.array(cv_precision_scores), axis=0))
print("F1 Score:")
print(np.mean(np.array(cv_f1_scores), axis=0))
print("Confusion Matrix:")
print(np.sum(cv_conf_matrices, axis=0))
####################################################################################
#  2. A traditional ML classifier with word embeddings
####################################################################################


#################################
# word embeddings
#################################

print(preprocessed_questions[:5])
preprocessed_questions_list = list();
for sentence in preprocessed_questions:
    word_list = sentence.split();
    preprocessed_questions_list.append(word_list);

print(preprocessed_questions_list[:5])

#https://stats.stackexchange.com/questions/221715/apply-word-embeddings-to-entire-document-to-get-a-feature-vector
#create doc2vec
class DocEmbeddingVectorizer(object):
    def __init__(self, model):
        self.model = model

    def fit(self, X, y):
        return self

    def transform(self, X):
        word_list_list = list();
        for sentence in X:
            word_list = sentence.split();
            word_list_list.append(word_list);
        return np.array([
            np.mean([self.model[word] for word in question if word in self.model], axis=0)for question in word_list_list
        ])

###########
#from scratch
############
#1. word2vec model from scratch
skip_gram_w2v_model = Word2Vec(min_count=1,window=5,size=300,sg=1)
skip_gram_w2v_model.build_vocab(preprocessed_questions_list, progress_per=5000)
skip_gram_w2v_model.train(preprocessed_questions_list, total_examples=skip_gram_w2v_model.corpus_count, epochs=30, report_delay=1)
skip_gram_w2v_model.init_sims(replace=True)#to remove initial vectors and keep only normalized vectors
    
doc_vectorizer = DocEmbeddingVectorizer(skip_gram_w2v_model)
word2vec_question_embeddings = doc_vectorizer.transform(preprocessed_questions)

parameters = {'alpha':[1e-2,1e-3,1e-4],'penalty':['l2', 'l1'],'loss':['hinge','log'],'max_iter':[5,10,15,20]}

classifier = SGDClassifier(random_state=42, tol=None)
grid = GridSearchCV(classifier, parameters, cv = 10, scoring = 'accuracy')
grid.fit(word2vec_question_embeddings, encoded_category_labels)

print('======word2vec_question_embeddings Best Parameters=====')
print(grid.best_params_)
print(grid.best_score_)
print('==========================')


#2. fasttext model from scratch - fast text perf not good for small data sets
fast_text_model = FastText(min_count=1,window=5,size=300)
fast_text_model.build_vocab(preprocessed_questions_list)
fast_text_model.train(sentences=preprocessed_questions_list, total_examples=fast_text_model.corpus_count, epochs=30)  # train
fast_text_model.init_sims(replace=True)#to remove initial vectors and keep only normalized vectors
    
doc_vectorizer = DocEmbeddingVectorizer(fast_text_model)
fasttext_question_embeddings = doc_vectorizer.transform(preprocessed_questions)

parameters = {'alpha':[1e-2,1e-3,1e-4],'penalty':['l2', 'l1'],'loss':['hinge','log'],'max_iter':[5,10,15,20]}

classifier = SGDClassifier(random_state=42, tol=None)
grid = GridSearchCV(classifier, parameters, cv = 10, scoring = 'accuracy')
grid.fit(fasttext_question_embeddings, encoded_category_labels)

print('======fasttext_question_embeddings Best Parameters=====')
print(grid.best_params_)
print(grid.best_score_)
print('==========================')
###########
#pre trained
############

#3. pre-trained google word embeddings
google_word_embedding_model = KeyedVectors.load_word2vec_format('/kaggle/input/googlewordembediings/GoogleNews-vectors-negative300.bin', binary=True)

doc_vectorizer = DocEmbeddingVectorizer(google_word_embedding_model)
google_pre_question_embeddings = doc_vectorizer.transform(preprocessed_questions)

parameters = {'alpha':[1e-2,1e-3,1e-4],'penalty':['l2', 'l1'],'loss':['hinge','log'],'max_iter':[5,10,15,20]}

classifier = SGDClassifier(random_state=42, tol=None)
grid = GridSearchCV(classifier, parameters, cv = 10, scoring = 'accuracy')
grid.fit(google_pre_question_embeddings, encoded_category_labels)

print('======google_pre_question_embeddings Best Parameters=====')
print(grid.best_params_)
print(grid.best_score_)
print('==========================')

#4. pre-trained fast text word embeddings
fasttext_word_embedding_model = KeyedVectors.load_word2vec_format('/kaggle/input/fasttext-crawl-300d-2m/crawl-300d-2M.vec')
 
doc_vectorizer = DocEmbeddingVectorizer(fasttext_word_embedding_model)
fasttext_pre_question_embeddings = doc_vectorizer.transform(preprocessed_questions)

parameters = {'alpha':[1e-2,1e-3,1e-4],'penalty':['l2', 'l1'],'loss':['hinge','log'],'max_iter':[5,10,15,20]}

classifier = SGDClassifier(random_state=42, tol=None)
grid = GridSearchCV(classifier, parameters, cv = 10, scoring = 'accuracy')
grid.fit(fasttext_pre_question_embeddings, encoded_category_labels)

print('======fasttext_pre_question_embeddings Best Parameters=====')
print(grid.best_params_)
print(grid.best_score_)
print('==========================')
#5. pre-trained glove text word embeddings
glove2word2vec(glove_input_file="/kaggle/input/glove6b300dtxt/glove.6B.300d.txt", word2vec_output_file="gensim_glove_vectors.txt")
glove_word_embedding_model = KeyedVectors.load_word2vec_format('/kaggle/working/gensim_glove_vectors.txt')
 
doc_vectorizer = DocEmbeddingVectorizer(glove_word_embedding_model)
glove_pre_question_embeddings = doc_vectorizer.transform(preprocessed_questions)

parameters = {'alpha':[1e-2,1e-3,1e-4],'penalty':['l2', 'l1'],'loss':['hinge','log'],'max_iter':[5,10,15,20]}

classifier = SGDClassifier(random_state=42, tol=None)
grid = GridSearchCV(classifier, parameters, cv = 10, scoring = 'accuracy')
grid.fit(glove_pre_question_embeddings, encoded_category_labels)

print('======glove_pre_question_embeddings Best Parameters=====')
print(grid.best_params_)
print(grid.best_score_)
print('==========================')
#######
#fine tuned from scrach models with pre-trained vectors
#########

#6. fine tune with google news vectors
#fine tune skip gram with google word embeddings
fine_tuned_skip_gram_w2v_model = Word2Vec(min_count=1,window=5,size=300,sg=1)
fine_tuned_skip_gram_w2v_model.build_vocab(preprocessed_questions_list, progress_per=5000)
fine_tuned_skip_gram_w2v_model.intersect_word2vec_format('/kaggle/input/googlewordembediings/GoogleNews-vectors-negative300.bin', binary=True, lockf=1.0)
fine_tuned_skip_gram_w2v_model.train(preprocessed_questions_list, total_examples=fine_tuned_skip_gram_w2v_model.corpus_count, epochs=30, report_delay=1)

doc_vectorizer = DocEmbeddingVectorizer(fine_tuned_skip_gram_w2v_model)
word2vec_tuned_question_embeddings = doc_vectorizer.transform(preprocessed_questions)

parameters = {'alpha':[1e-2,1e-3,1e-4],'penalty':['l2', 'l1'],'loss':['hinge','log'],'max_iter':[5,10,15,20]}

classifier = SGDClassifier(random_state=42, tol=None)
grid = GridSearchCV(classifier, parameters, cv = 10, scoring = 'accuracy')
grid.fit(word2vec_tuned_question_embeddings, encoded_category_labels)

print('======fine_tuned_skip_gram_w2v_model Best Parameters For Categories=====')
print(grid.best_params_)
print(grid.best_score_)
print('==========================')

classifier = SGDClassifier(random_state=42, tol=None)
grid = GridSearchCV(classifier, parameters, cv = 10, scoring = 'accuracy')
grid.fit(word2vec_tuned_question_embeddings, encoded_sub_category_labels)

print('======fine_tuned_skip_gram_w2v_model Best Parameters For Sub Categories=====')
print(grid.best_params_)
print(grid.best_score_)
print('==========================')
#7 transfer learning
#transfer learning with fast text - not success due to insufficient memory
"""
tl_fasttext_word_embedding_model = load_facebook_model('fasttext-common-crawl-bin-model/cc.en.300.bin')
tl_fasttext_word_embedding_model.build_vocab(preprocessed_questions_list, update=True)
tl_fasttext_word_embedding_model.train(preprocessed_questions_list, total_examples=tl_fasttext_word_embedding_model.corpus_count, epochs=30)

doc_vectorizer = DocEmbeddingVectorizer(tl_fasttext_word_embedding_model)
tl_fasttext_question_embeddings = doc_vectorizer.transform(preprocessed_questions)
"""
np.set_printoptions(precision=2)
np.set_printoptions(formatter={'int': '{:,}'.format})
np.set_printoptions(linewidth=200)

###########
#model evaluation
##########

# SGD with tradtional features

print("=====================================================================================")
print("SGD for Category classification with word embeddings")
print("=====================================================================================")

kfold = StratifiedKFold(n_splits=10, shuffle=True)
iteration = 0;
cv_accuracy_scores = []
cv_conf_matrices = []
cv_recall_scores = []
cv_precision_scores = []
cv_f1_scores = []

for train_index, test_index in kfold.split(word2vec_tuned_question_embeddings,encoded_category_labels):

    iteration = iteration + 1
    print("Cross Validation =================================================================== ", iteration)

    train_questions,test_questions=word2vec_tuned_question_embeddings[train_index],word2vec_tuned_question_embeddings[test_index]
    train_category_labels,test_category_labels=np.array(encoded_category_labels)[train_index],np.array(encoded_category_labels)[test_index]
    
    question_category_classification_pipeline = Pipeline([
     ('Classifier', SGDClassifier(loss='log', penalty='l2',
                          alpha=0.0001, random_state=42,
                           max_iter=20, tol=None)),], verbose=True)
    
    question_category_classification_pipeline.fit(train_questions, train_category_labels) 
    test_predicted_category_labels=question_category_classification_pipeline.predict(test_questions)
    
    accuracy = accuracy_score(test_category_labels, test_predicted_category_labels)
    matrix = confusion_matrix(test_category_labels, test_predicted_category_labels)
    recall = recall_score(test_category_labels, test_predicted_category_labels,average=None)
    precision = precision_score(test_category_labels, test_predicted_category_labels,average=None)
    f1 = f1_score(test_category_labels, test_predicted_category_labels,average=None)
    
    cv_accuracy_scores.append(accuracy)
    cv_conf_matrices.append(matrix)
    cv_recall_scores.append(recall)
    cv_precision_scores.append(precision)
    cv_f1_scores.append(f1)

print("Results for categories:==============================================================================")
print("Accuracy: %.2f" % np.mean(cv_accuracy_scores))
print("Confusion Matrix:")
print(np.sum(cv_conf_matrices, axis=0))
print("Recall:")
print(np.mean(np.array(cv_recall_scores), axis=0)) 
print("Precision:")
print(np.mean(np.array(cv_precision_scores), axis=0))
print("F1 Score:")
print(np.mean(np.array(cv_f1_scores), axis=0))

print("=====================================================================================================")
print("SGD for Sub Category classification with word embeddings")
print("======================================================================================================")


kfold = StratifiedKFold(n_splits=10, shuffle=True)
iteration = 0;
cv_accuracy_scores = []
cv_conf_matrices = []
cv_recall_scores = []
cv_precision_scores = []
cv_f1_scores = []

for train_index, test_index in kfold.split(word2vec_tuned_question_embeddings,encoded_sub_category_labels):

    iteration = iteration + 1
    print("Cross Validation =================================================================== ", iteration)

    train_questions,test_questions=word2vec_tuned_question_embeddings[train_index],word2vec_tuned_question_embeddings[test_index]
    train_sub_category_labels,test_sub_category_labels=np.array(encoded_sub_category_labels)[train_index],np.array(encoded_sub_category_labels)[test_index]
    
    question_category_classification_pipeline = Pipeline([
     ('Classifier', SGDClassifier(loss='hinge', penalty='l2',
                          alpha=0.0001, random_state=42,
                           max_iter=15, tol=None)),], verbose=True)
    
    question_category_classification_pipeline.fit(train_questions, train_sub_category_labels) 
    test_predicted_sub_category_labels=question_category_classification_pipeline.predict(test_questions)
    
    accuracy = accuracy_score(test_sub_category_labels, test_predicted_sub_category_labels)
    required_labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62]
    matrix = confusion_matrix(test_sub_category_labels, test_predicted_sub_category_labels, labels = required_labels)
    recall = recall_score(test_sub_category_labels, test_predicted_sub_category_labels,average=None, labels = required_labels, zero_division = 0)
    precision = precision_score(test_sub_category_labels, test_predicted_sub_category_labels,average=None, labels = required_labels, zero_division = 0)
    f1 = f1_score(test_sub_category_labels, test_predicted_sub_category_labels,average=None, labels = required_labels, zero_division = 0)
    
    cv_accuracy_scores.append(accuracy)
    cv_conf_matrices.append(matrix)
    cv_recall_scores.append(recall)
    cv_precision_scores.append(precision)
    cv_f1_scores.append(f1)

print("Results for sub categories:==============================================================================")
print("Accuracy: %.2f" % np.mean(cv_accuracy_scores))
print("Recall:")
print(np.mean(np.array(cv_recall_scores), axis=0)) 
print("Precision:")
print(np.mean(np.array(cv_precision_scores), axis=0))
print("F1 Score:")
print(np.mean(np.array(cv_f1_scores), axis=0))
print("Confusion Matrix:")
print(np.sum(cv_conf_matrices, axis=0))
####################################################################################
#  3. A NN Classifier
####################################################################################


embedding_model = KeyedVectors.load_word2vec_format('/kaggle/input/googlewordembediings/GoogleNews-vectors-negative300.bin', binary=True)

word_tokenizer = Tokenizer(num_words = 6000)
word_tokenizer.fit_on_texts(preprocessed_questions)
word_index = word_tokenizer.word_index

weight_matrix = np.zeros((len(word_index)+1, 300))
exist_word_count = 0;
for word, i in word_index.items():
    if word in embedding_model:
        embedding_vector = embedding_model[word]
        weight_matrix[i] = embedding_vector
        exist_word_count = exist_word_count + 1

preprocessed_questions_sequences = word_tokenizer.texts_to_sequences(preprocessed_questions)
preprocessed_questions_sequences_padded = pad_sequences(preprocessed_questions_sequences, maxlen=300, padding='post', truncating='post')

print("==========================================================================================")
print("LSTM for Category classification")
print("==========================================================================================")

kfold = StratifiedKFold(n_splits=10, shuffle=True)
cvscores = []
iteration = 0

cv_accuracy_scores = []
cv_conf_matrices = []
cv_recall_scores = []
cv_precision_scores = []
cv_f1_scores = []

questions_list = np.array(preprocessed_questions_sequences_padded.tolist());
category_list = np.array(encoded_category_labels);
for train_index, test_index in kfold.split(questions_list,category_list):

    iteration = iteration + 1
    print("Cross Validation =================================================================== ", iteration)

    train_questions,test_questions=questions_list[train_index],questions_list[test_index]
    train_category_labels,test_category_labels=category_list[train_index],category_list[test_index]
    
    #keras LSTM with weight matrix
    #softmax convert outputs layers into a probability distribution.
    #build model
    model = tf.keras.Sequential([
    tf.keras.layers.Embedding(input_dim=len(word_index)+1, input_length=300, output_dim=300,weights=[weight_matrix], trainable=False),
    tf.keras.layers.SpatialDropout1D(0.3),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(100)),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(1024, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(7, activation='softmax')
    ])
    # Compile model
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    # Fit the model
    earlystopCallback = EarlyStopping(monitor='val_acc', min_delta=0, patience=3, verbose=0, mode='auto')
    model.fit(train_questions, train_category_labels, epochs=10, batch_size=100, verbose=1, callbacks=[earlystopCallback])
    # Evaluate the model
    predicted_category_labels = model.predict_classes(test_questions, verbose=0)  
    
    accuracy = accuracy_score(test_category_labels, predicted_category_labels)
    recall = recall_score(test_category_labels, predicted_category_labels,average=None, zero_division = 0)
    precision = precision_score(test_category_labels, predicted_category_labels,average=None, zero_division = 0)
    f1 = f1_score(test_category_labels, predicted_category_labels,average=None, zero_division = 0)
    
    cv_accuracy_scores.append(accuracy)
    cv_recall_scores.append(recall)
    cv_precision_scores.append(precision)
    cv_f1_scores.append(f1)
    
print("Results for Categories:==============================================================================")
print("Accuracy: %.2f" % np.mean(cv_accuracy_scores))
print("Recall:")
print(np.mean(np.array(cv_recall_scores), axis=0)) 
print("Precision:")
print(np.mean(np.array(cv_precision_scores), axis=0))
print("F1 Score:")
print(np.mean(np.array(cv_f1_scores), axis=0))


print("====================================================================================================")
print("LSTM for Sub Category classification")
print("=====================================================================================================")

kfold = StratifiedKFold(n_splits=10, shuffle=True)
cvscores = []
iteration = 1

cv_accuracy_scores = []
cv_conf_matrices = []
cv_recall_scores = []
cv_precision_scores = []
cv_f1_scores = []

questions_list = np.array(preprocessed_questions_sequences_padded.tolist());
category_list = np.array(encoded_sub_category_labels);
for train_index, test_index in kfold.split(questions_list,category_list):

    iteration = iteration + 1
    print("Cross Validation =================================================================== ", iteration)

    train_questions,test_questions=questions_list[train_index],questions_list[test_index]
    train_category_labels,test_category_labels=category_list[train_index],category_list[test_index]
    
    #keras LSTM with weight matrix
    #softmax convert outputs layers into a probability distribution.
    #build model
    model = tf.keras.Sequential([
    tf.keras.layers.Embedding(input_dim=len(word_index)+1, input_length=300, output_dim=300,weights=[weight_matrix], trainable=False),
    tf.keras.layers.SpatialDropout1D(0.3),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(100)),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(1024, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(63, activation='softmax')
    ])
    # Compile model
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    # Fit the model
    earlystopCallback = EarlyStopping(monitor='val_acc', min_delta=0, patience=3, verbose=0, mode='auto')
    model.fit(train_questions, train_category_labels, epochs=10, batch_size=100, verbose=1, callbacks=[earlystopCallback])
    # Evaluate the model
    predicted_category_labels = model.predict_classes(test_questions, verbose=0)  
    
    accuracy = accuracy_score(test_category_labels, predicted_category_labels)
    
    required_labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62]

    recall = recall_score(test_category_labels, predicted_category_labels,average=None, zero_division = 0, labels = required_labels)
    precision = precision_score(test_category_labels, predicted_category_labels,average=None, zero_division = 0, labels = required_labels)
    f1 = f1_score(test_category_labels, predicted_category_labels,average=None, zero_division = 0, labels = required_labels)
    
    cv_accuracy_scores.append(accuracy)
    cv_recall_scores.append(recall)
    cv_precision_scores.append(precision)
    cv_f1_scores.append(f1)
    
print("Results for Sub Categories:==============================================================================")
print("Accuracy: %.2f" % np.mean(cv_accuracy_scores))
print("Recall:")
print(np.mean(np.array(cv_recall_scores), axis=0)) 
print("Precision:")
print(np.mean(np.array(cv_precision_scores), axis=0))
print("F1 Score:")
print(np.mean(np.array(cv_f1_scores), axis=0))