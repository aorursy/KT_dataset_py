# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import random
import sklearn
import scipy
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse.linalg import svds
import matplotlib.pyplot as plt
import math
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
interactions_df = pd.read_csv('../input/articles-sharing-reading-from-cit-deskdrop/users_interactions.csv')
interactions_df.head(10)

articles_df = pd.read_csv('../input/articles-sharing-reading-from-cit-deskdrop/shared_articles.csv')
articles_df = articles_df[articles_df['eventType'] == 'CONTENT SHARED']
articles_df.head(5)
### Evaluate interaction length by eventType 
event_type_strength = {
   'VIEW': 1.0,
   'LIKE': 2.0, 
   'BOOKMARK': 2.5, 
   'FOLLOW': 3.0,
   'COMMENT CREATED': 4.0,  
}

users_interactions_count_df = interactions_df.groupby(['personId', 'contentId']).size().groupby('personId').size()
print('# users: %d' % len(users_interactions_count_df))
users_with_enough_interactions_df = users_interactions_count_df[users_interactions_count_df >= 5].reset_index()[['personId']]
print('# users with at least 5 interactions: %d' % len(users_with_enough_interactions_df))

# merge w/ the original dataframe to retrieve the interaction whose users have >5 interactions 
print('# of interactions: %d' % len(interactions_df))
interactions_from_selected_users_df = interactions_df.merge(users_with_enough_interactions_df, 
               how = 'right',
               left_on = 'personId',
               right_on = 'personId')
print('# of interactions from users with at least 5 interactions: %d' % len(interactions_from_selected_users_df))

interactions_df['eventStrength'] = interactions_df['eventType'].apply(lambda x: event_type_strength[x])

interactions_from_selected_users_df = interactions_df.merge(users_with_enough_interactions_df, 
               how = 'right',
               left_on = 'personId',
               right_on = 'personId')

def smooth_user_preference(x):
    return math.log(1+x, 2)
    
interactions_full_df = interactions_from_selected_users_df \
                    .groupby(['personId', 'contentId'])['eventStrength'].sum() \
                    .apply(smooth_user_preference).reset_index()
print('# of unique user/item interactions: %d' % len(interactions_full_df))
interactions_full_df.head(10)
interactions_df[interactions_df.personId==-9212075797126931087]
interactions_from_selected_users_df.head(5)
interactions_train_df, interactions_test_df = train_test_split(interactions_full_df,
                                   stratify=interactions_full_df['personId'], 
                                   test_size=0.20,
                                   random_state=42)

print('# interactions on Train set: %d' % len(interactions_train_df))
print('# interactions on Test set: %d' % len(interactions_test_df))
interactions_full_indexed_df = interactions_full_df.set_index('personId')
interactions_train_indexed_df = interactions_train_df.set_index('personId')
interactions_test_indexed_df = interactions_test_df.set_index('personId')
from keras.utils import get_file
from keras.preprocessing.text import Tokenizer

num_tokens = 5000

docs = articles_df['title'] + "" + articles_df['text']
docs = docs.tolist()

class keras_tokenizer:
    def __init__(self,filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n', num_tokens=1000 +1):
        self.tokenizer =  Tokenizer(lower=True, filters=filters, num_words = num_tokens + 1)
        self.num_tokens = num_tokens
        self.word_idx = None
        self.idx_word = None

    def stopwords_removal(self, docs, stop_words):
        for i, doc in enumerate(docs):
            new_doc = [word for word in doc.split() if word not in stop_words]
            docs[i] = ' '.join(new_doc)
        return docs   
    
    def fit_on_texts(self,docs, stopwords_list=list()):
        stopwords_list = stopwords.words('english') + stopwords.words('portuguese')
        #print(self.stopwords_removal(docs, stopwords_list))
        self.tokenizer.fit_on_texts(self.stopwords_removal(docs, stopwords_list))
        self.word_idx = {p:q for (p, q) in self.tokenizer.word_index.items() if q <=self.num_tokens}
        self.idx_word = {q:p for (p, q) in self.tokenizer.word_index.items() if q <=self.num_tokens}
        return None
    
    def texts_to_sequences(self,text):    
        return self.tokenizer.texts_to_sequences(text)
    


interactions_indexed_df = interactions_full_df[interactions_full_df['contentId'] \
                                               .isin(articles_df['contentId'])].set_index('personId')
#interactions_indexed_df[interactions_indexed_df.index==-9223121837663643404]

from scipy.sparse import csr_matrix
class vectorizer:
    def __init__(self, keras_tokenizer, word_lookup=None):
        self.tokenizer = tokenizer
        self.word_lookup = word_lookup
        
    def vectorize_docs(self, docs, mode=1):
        '''
        model==1: plain tf-idf; mod==2: word2vec weighted by sklearn tf-idf'
        output is in Compressed Sparse Row format.
        '''
        seqs = self.tokenizer.texts_to_sequences(docs)
        
        # vocabulary based on tokenizer
        voc = self.tokenizer.word_idx.copy()
        voc[''] = 0
        
        # Build sklearn tfidf vectorizer based on vocabulary of tokenizer
        vectzer = TfidfVectorizer(vocabulary = voc)
        score_tfidf = vectzer.fit_transform(docs)
        tfidf_feature_names = vectzer.get_feature_names()
        
        if mode == 2:
            if word_lookup != None:
                embed_dim = 100  # this part has to be improved!
                tfidf_word2vec = np.zeros((score_tfidf.shape[0], embed_dim))

                for i in range(score_tfidf.shape[0]):
                    doc_word2vec = np.zeros(embed_dim)
                    #print(np.nonzero(tfidf_matrix[i, :].toarray())[1].shape)
                    for nonzero_idx in np.nonzero(score_tfidf[i, :].toarray())[1]:
                        #print(nonzero_idx)
                        if tfidf_feature_names[nonzero_idx] in self.word_lookup.keys():
                            w2v = self.word_lookup[tfidf_feature_names[nonzero_idx]]
                            doc_word2vec += score_tfidf[i, nonzero_idx] * w2v
                        
                    tfidf_word2vec[i, :] = doc_word2vec       
                return csr_matrix(tfidf_word2vec)
            else:
                return None
        else:
            return score_tfidf 
    
    
    
    def vectorize_preference(self,content_vectors, interact_df):
        '''
        vectorize the preference for each user: average over the doc vectors weighted by interaction strength"
        interact_df: the dataframe w/ three columns 'person', 'content' and 'interaction strength'
        '''
        def get_item_profile(item_id):
            idx = item_ids.index(item_id)
            item_profile = content_vectors[idx:idx+1]
            return item_profile

        def get_item_profiles(ids):
            item_profiles_list = [get_item_profile(x) for x in ids]
            item_profiles = scipy.sparse.vstack(item_profiles_list)
            return item_profiles

        def build_users_profile(person_id, interactions_indexed_df):
            interactions_person_df = interactions_indexed_df.loc[person_id]
            user_item_profiles = get_item_profiles(interactions_person_df['contentId'])

            user_item_strengths = np.array(interactions_person_df['eventStrength']).reshape(-1,1)
            #Weighted average of item profiles by the interactions strength
            #print(user_item_profiles.shape, user_item_strengths.shape)
            user_item_strengths_weighted_avg = np.sum(user_item_profiles.multiply(user_item_strengths), axis=0) / np.sum(user_item_strengths)
            user_profile_norm = sklearn.preprocessing.normalize(user_item_strengths_weighted_avg)
            return user_profile_norm

        def build_users_profiles(): 
            #interactions_indexed_df = interactions_full_df[interactions_full_df['contentId'] \
            #                                               .isin(articles_df['contentId'])].set_index('personId')
            user_profiles = {}
            for person_id in interact_df.index.unique():
                user_profiles[person_id] = build_users_profile(person_id, interact_df)
            return user_profiles
        user_profiles = build_users_profiles()

        return user_profiles
    



item_ids = articles_df['contentId'].tolist()
print(len(item_ids), len(docs))

class ContentBasedRecommender:
    
    MODEL_NAME = 'Content-Based'
    
    def __init__(self, vectorizer,item_ids, items_df=None):
        self.item_ids = item_ids
        self.items_df = items_df
        self.vectorizer = vectorizer
        self.content_vectors = None
        self.preference_vectors = None
        
    def get_model_name(self):
        return self.MODEL_NAME
    
    def build_content_vectors(self, raw_contents, mode =1):
        self.content_vectors = self.vectorizer.vectorize_docs(raw_contents, mode)
        return None
    
    def build_preference_vectors(self, interact_df):
        self.preference_vectors = self.vectorizer.vectorize_preference(self.content_vectors, interact_df)
        return None
    
    def _get_similar_items_to_user_profile(self, person_id, topn=1000):
        #Computes the cosine similarity between the user profile and all item profiles
        # cosine_similarities = cosine_similarity(user_profiles[person_id], tfidf_matrix)
        cosine_similarities = cosine_similarity(self.preference_vectors[person_id], self.content_vectors)
        #Gets the top similar items
        similar_indices = cosine_similarities.argsort().flatten()[-topn:]
        #Sort the similar items by similarity
        similar_items = sorted([(item_ids[i], cosine_similarities[0,i]) for i in similar_indices], key=lambda x: -x[1])
        return similar_items
        
    def recommend_items(self, user_id, items_to_ignore=[], topn=10, verbose=False):
        similar_items = self._get_similar_items_to_user_profile(user_id)
        #Ignores items the user has already interacted
        similar_items_filtered = list(filter(lambda x: x[0] not in items_to_ignore, similar_items))
        
        recommendations_df = pd.DataFrame(similar_items_filtered, columns=['contentId', 'recStrength']) \
                                    .head(topn)

        if verbose:
            if self.items_df is None:
                raise Exception('"items_df" is required in verbose mode')

            recommendations_df = recommendations_df.merge(self.items_df, how = 'left', 
                                                          left_on = 'contentId', 
                                                          right_on = 'contentId')[['recStrength', 'contentId', 'title', 'url', 'lang']]
        return recommendations_df
    


EVAL_RANDOM_SAMPLE_NON_INTERACTED_ITEMS = 100



class ModelEvaluator:
    def get_items_interacted(self, person_id, interactions_df):
        # Get the user's data and merge in the movie information.
        interacted_items = interactions_df.loc[person_id]['contentId']
        return set(interacted_items if type(interacted_items) == pd.Series else [interacted_items])
    def get_not_interacted_items_sample(self, person_id, sample_size, seed=42):
        interacted_items = self.get_items_interacted(person_id, interactions_full_indexed_df)
        all_items = set(articles_df['contentId'])
        non_interacted_items = all_items - interacted_items

        random.seed(seed)
        non_interacted_items_sample = random.sample(non_interacted_items, sample_size)
        return set(non_interacted_items_sample)

    def _verify_hit_top_n(self, item_id, recommended_items, topn):        
        try:
            index = next(i for i, c in enumerate(recommended_items) if c == item_id)
        except:
            index = -1
        hit = int(index in range(0, topn))
        return hit, index

    def evaluate_model_for_user(self, model, person_id):
        #Getting the items in test set
        interacted_values_testset = interactions_test_indexed_df.loc[person_id]
        if type(interacted_values_testset['contentId']) == pd.Series:
            person_interacted_items_testset = set(interacted_values_testset['contentId'])
        else:
            person_interacted_items_testset = set([int(interacted_values_testset['contentId'])])  
        interacted_items_count_testset = len(person_interacted_items_testset) 

        #Getting a ranked recommendation list from a model for a given user
        person_recs_df = model.recommend_items(person_id, 
                                               items_to_ignore=self.get_items_interacted(person_id, 
                                                                                    interactions_train_indexed_df), 
                                               topn=10000000000)

        hits_at_5_count = 0
        hits_at_10_count = 0
        #For each item the user has interacted in test set
        for item_id in person_interacted_items_testset:
            #Getting a random sample (100) items the user has not interacted 
            #(to represent items that are assumed to be no relevant to the user)
            non_interacted_items_sample = self.get_not_interacted_items_sample(person_id, 
                                                                          sample_size=EVAL_RANDOM_SAMPLE_NON_INTERACTED_ITEMS, 
                                                                          seed=item_id%(2**32))

            #Combining the current interacted item with the 100 random items
            items_to_filter_recs = non_interacted_items_sample.union(set([item_id]))

            #Filtering only recommendations that are either the interacted item or from a random sample of 100 non-interacted items
            valid_recs_df = person_recs_df[person_recs_df['contentId'].isin(items_to_filter_recs)]                    
            valid_recs = valid_recs_df['contentId'].values
            #Verifying if the current interacted item is among the Top-N recommended items
            hit_at_5, index_at_5 = self._verify_hit_top_n(item_id, valid_recs, 5)
            hits_at_5_count += hit_at_5
            hit_at_10, index_at_10 = self._verify_hit_top_n(item_id, valid_recs, 10)
            hits_at_10_count += hit_at_10

        #Recall is the rate of the interacted items that are ranked among the Top-N recommended items, 
        #when mixed with a set of non-relevant items
        recall_at_5 = hits_at_5_count / float(interacted_items_count_testset)
        recall_at_10 = hits_at_10_count / float(interacted_items_count_testset)

        person_metrics = {'hits@5_count':hits_at_5_count, 
                          'hits@10_count':hits_at_10_count, 
                          'interacted_count': interacted_items_count_testset,
                          'recall@5': recall_at_5,
                          'recall@10': recall_at_10}
        return person_metrics

    def evaluate_model(self, model):
        #print('Running evaluation for users')
        people_metrics = []
        for idx, person_id in enumerate(list(interactions_test_indexed_df.index.unique().values)):
            #if idx % 100 == 0 and idx > 0:
            #    print('%d users processed' % idx)
            person_metrics = self.evaluate_model_for_user(model, person_id)  
            person_metrics['_person_id'] = person_id
            people_metrics.append(person_metrics)
        print('%d users processed' % idx)

        detailed_results_df = pd.DataFrame(people_metrics) \
                            .sort_values('interacted_count', ascending=False)
        
        global_recall_at_5 = detailed_results_df['hits@5_count'].sum() / float(detailed_results_df['interacted_count'].sum())
        global_recall_at_10 = detailed_results_df['hits@10_count'].sum() / float(detailed_results_df['interacted_count'].sum())
        
        global_metrics = {'modelName': model.get_model_name(),
                          'recall@5': global_recall_at_5,
                          'recall@10': global_recall_at_10}    
        return global_metrics, detailed_results_df

filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n'
tokenizer = keras_tokenizer(filters = filters, num_tokens= 5000)
tokenizer.fit_on_texts(docs)
tokenizer.texts_to_sequences(docs)

vectorizer1 = vectorizer(tokenizer)
content_vec = vectorizer1.vectorize_docs(docs)
vectorizer1.vectorize_preference(content_vectors=content_vec, interact_df = interactions_indexed_df)

recommender1 = ContentBasedRecommender(vectorizer1,item_ids)
recommender1.build_content_vectors(docs)
recommender1.build_preference_vectors(interactions_indexed_df) #Must build content vectors for recommander before building preference vectors
recommender1.recommend_items(-9223121837663643404)
    
evaluator = ModelEvaluator()  
print('Evaluating Content-Based Filtering model w/ tf-idf vectorization...')
cb_global_metrics, cb_detailed_results_df = evaluator.evaluate_model(recommender1)
print('\nGlobal metrics:\n%s' % cb_global_metrics)
cb_detailed_results_df.head(10)
import os

# Load pre-trained embedding
glove_vectors = '../input/glove6b100dtxt/glove.6B.100d.txt'
glove = np.loadtxt(glove_vectors, dtype='str', comments=None)
glove.shape
vectors = glove[:, 1:].astype('float')
words = glove[:, 0]

del glove

vectors[100], words[100]
print(f'There are {num_tokens} unique words.')

word_lookup = {word: vector for word, vector in zip(words, vectors)}

not_found = 0
'''
for i, word in enumerate(tfidf_feature_names):
    # Look up the word embedding
    vector = word_lookup.get(word, None)

    # Record in matrix
    if vector is None:
        not_found += 1

print(f'There were {not_found} words without pre-trained embeddings.')
'''
import gc
gc.enable()
del vectors
gc.collect()
vectorizer2 = vectorizer(tokenizer, word_lookup = word_lookup)
content_vec = vectorizer2.vectorize_docs(docs, mode=2)
content_vec

vectorizer2.vectorize_preference(content_vectors=content_vec, interact_df = interactions_indexed_df)

recommender2 = ContentBasedRecommender(vectorizer2,item_ids)
recommender2.build_content_vectors(docs, mode=2)
recommender2.build_preference_vectors(interactions_indexed_df)
recommender2.recommend_items(-9223121837663643404)
    
evaluator = ModelEvaluator()  
print('Evaluating Content-Based Filtering model w/ tf-idf vectorization...')
cb_global_metrics, cb_detailed_results_df = evaluator.evaluate_model(recommender2)
print('\nGlobal metrics:\n%s' % cb_global_metrics)
cb_detailed_results_df.head(10)

import gensim

corpus = [[tokenizer.idx_word[token] for token in doc] for doc in tokenizer.texts_to_sequences(docs)]

# train the word2vec model
model_w2v = gensim.models.Word2Vec(corpus, size=100, min_count=5, sg=1, negative=5, workers=2)
# Build look up dictionary from word2vec model
word_lookup2 = {word:model_w2v[word] for word in model_w2v.wv.vocab.keys()}
vectorizer3 = vectorizer(tokenizer, word_lookup = word_lookup2)
content_vec = vectorizer3.vectorize_docs(docs, mode=2) # mode 2: apply word2vec model
content_vec

vectorizer3.vectorize_preference(content_vectors=content_vec, interact_df = interactions_indexed_df)

recommender3 = ContentBasedRecommender(vectorizer3,item_ids)
recommender3.build_content_vectors(docs, mode=2)
recommender3.build_preference_vectors(interactions_indexed_df)
recommender3.recommend_items(-9223121837663643404)
    
print('Evaluating Content-Based Filtering model w/ my own word2vec vectorization...')
cb_global_metrics, cb_detailed_results_df = evaluator.evaluate_model(recommender3)
print('\nGlobal metrics:\n%s' % cb_global_metrics)
cb_detailed_results_df.head(10)
