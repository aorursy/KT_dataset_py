import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')
#from vivadata.datasets.common import get_path_for_dataset
#base_path = get_path_for_dataset('quora')
X_train_filepath = os.path.join('..', 'input', 'train.csv')
X_test_filepath = os.path.join('..', 'input', 'test.csv')
sample_filepath = os.path.join('..', 'input', 'sample_submission.csv')
X_train_filepath, X_test_filepath, sample_filepath
df_train = pd.read_csv(X_train_filepath)
df_test = pd.read_csv(X_test_filepath)
sample = pd.read_csv(sample_filepath)
df_train.shape, df_test.shape, sample.shape
df_train.sample(10)
df_test.sample(10)
df_train.info()
df_train.describe()
ax, fig = plt.subplots(figsize=(10, 7))
sns.countplot(x='target', data=df_train)
plt.title('Reparition of question by insincerity');
import gensim
import nltk
from nltk.corpus import stopwords
# Define the target and the variable.
y_train = df_train.loc[:, 'target']
X_train = df_train.loc[:, 'question_text']
X_train.shape, y_train.shape
X_train.head()
# Create a variable with all the sincere questions.
X_train_sincere = X_train[df_train['target'] == 0]
X_train_sincere[:5]
# Create a variable with all insincere questions.
X_train_insincere = X_train[df_train['target'] == 1]
X_train_insincere[:5]
# First i load the list of stopwords, the words which aren't useful to understand the meaning.
stop_words = stopwords.words('english')
stop_words[:10]
# I transform the sincere questions in a list of words.
sincere_prepro_questions = [gensim.utils.simple_preprocess(question) 
                            for question in X_train_sincere]
sincere_prepro_questions[:2]
# I transform the insincere questions in a list of words.
insincere_prepro_questions = [gensim.utils.simple_preprocess(question) 
                              for question in X_train_insincere]
insincere_prepro_questions[:2]
# Verifying the length of the both lists.
len(sincere_prepro_questions), len(insincere_prepro_questions)
# I remove the stopwords from the sincere questions list of words.
clear_sincere_questions = [[word for word in question if word not in stop_words] 
                             for question in sincere_prepro_questions]
clear_sincere_questions[:2]
# I do the same for insincere one.
clear_insincere_questions = [[word for word in question if word not in stop_words] 
                             for question in insincere_prepro_questions]
clear_insincere_questions[:2]
# Creation of a dictionnary of the words and their id for sincere questions
sincere_questions_dictionary = gensim.corpora.Dictionary(clear_sincere_questions)
sincere_token = sincere_questions_dictionary.token2id
# Association of the dictonary words and their frequency for sincere questions
sincere_dict_frequency = {sincere_questions_dictionary[k]: v for k,v in sincere_questions_dictionary.dfs.items()}
# Creation of a dictionnary of the words and their id for insincere questions
insincere_questions_dictionary = gensim.corpora.Dictionary(clear_insincere_questions)
insincere_token = insincere_questions_dictionary.token2id
# Association of the dictonary words and their frequency for insincere questions
insincere_dict_frequency = {sincere_questions_dictionary[k]: v for k,v in sincere_questions_dictionary.dfs.items()}
# I transform the elements of the sincere dictionary in vectors. 
sincere_corpus = [sincere_questions_dictionary.doc2bow(question) 
                  for question in clear_sincere_questions]
sincere_corpus[:2]
# I do the same for the insincere questions. 
insincere_corpus = [insincere_questions_dictionary.doc2bow(question) 
                    for question in clear_insincere_questions]
insincere_corpus[:2]
# Creating a lda model, which create 10 topics from the sincere questions
lda_model_sincere = gensim.models.ldamodel.LdaModel(
    corpus=sincere_corpus, num_topics=10, id2word=sincere_questions_dictionary,
    random_state=25, passes=5)
# Creating a lda model, which create 10 topics from the insincere questions
lda_model_insincere = gensim.models.ldamodel.LdaModel(
    corpus=insincere_corpus, num_topics=10, id2word=insincere_questions_dictionary,
    random_state=25, passes=5)
from pprint import pprint
# Printing the result of the modeling for sincere questions.
pprint(lda_model_sincere.print_topics()[0])
# Same for insincere ones. 
pprint(lda_model_insincere.print_topics()[0])
import pyLDAvis.gensim
pyLDAvis.enable_notebook()
pyLDAvis.gensim.prepare(lda_model_sincere, sincere_corpus, sincere_questions_dictionary)
pyLDAvis.gensim.prepare(lda_model_insincere, insincere_corpus, 
                        insincere_questions_dictionary)
from gensim.models import Phrases
from gensim.models.phrases import Phraser
# Create the bigram for sincere questions.
sincere_bigram_model = Phrases(sincere_prepro_questions, min_count=1, threshold=2)
# Same for insincere ones. 
insincere_bigram_model = Phrases(insincere_prepro_questions, min_count=1, threshold=2)
# Creation of the phrasers, which is needed to use the bigrams. First for sincere questions...
sincere_bigram_phraser = Phraser(sincere_bigram_model)
# Then for the insincere. 
insincere_bigram_phraser = Phraser(sincere_bigram_model)
sincere_bigrams = [sincere_bigram_model[question] for question in sincere_prepro_questions]
sincere_bigrams[:2]
insincere_bigrams = [insincere_bigram_model[question] for question in insincere_prepro_questions]
insincere_bigrams[:2]
# Creation of a dictionary with the words and their id for sincere bigrams
sincere_bigrams_dictionary = gensim.corpora.Dictionary(sincere_bigrams)
sincere_bigrams_token = sincere_bigrams_dictionary.token2id
# Association of the dictonary words and their frequency for sincere bigrams
sincere_bigrams_frequency = {sincere_bigrams_dictionary[k]: 
                             v for k,v in sincere_bigrams_dictionary.dfs.items()}
# Creation of a dictionary with the words and their id for insincere bigrams
insincere_bigrams_dictionary = gensim.corpora.Dictionary(insincere_bigrams)
insincere_bigrams_token = insincere_bigrams_dictionary.token2id
# Association of the dictonary words and their frequency for insincere bigrams
insincere_bigrams_frequency = {insincere_bigrams_dictionary[k]: 
                               v for k,v in insincere_bigrams_dictionary.dfs.items()}
sincere_bigrams_corpus = [sincere_bigrams_dictionary.doc2bow(question) 
                  for question in clear_sincere_questions]
sincere_bigrams_corpus[:2]
insincere_bigrams_corpus = [insincere_bigrams_dictionary.doc2bow(question) 
                  for question in clear_insincere_questions]
insincere_bigrams_corpus[:2]
# Creating a lda model, which create 10 topics from the sincere bigrams
lda_model_sincere_bigrams = gensim.models.ldamodel.LdaModel(
    corpus=sincere_bigrams_corpus, num_topics=10, id2word=sincere_bigrams_dictionary,
    random_state=25, passes=5)
# Creating a lda model, which create 10 topics from the insincere bigrams
import warnings
warnings.filterwarnings('ignore')

lda_model_insincere_bigrams = gensim.models.ldamodel.LdaModel(
    corpus=insincere_bigrams_corpus, num_topics=10, id2word=insincere_bigrams_dictionary,
    random_state=25, passes=5)
pprint(lda_model_sincere_bigrams.print_topics()[0])
pprint(lda_model_insincere_bigrams.print_topics()[0])
pyLDAvis.gensim.prepare(lda_model_sincere_bigrams, sincere_bigrams_corpus, 
                        sincere_bigrams_dictionary)
pyLDAvis.gensim.prepare(lda_model_insincere_bigrams, insincere_bigrams_corpus, 
                        insincere_bigrams_dictionary)
# Creating a lda model, which create 4 topics from the sincere questions
import warnings
warnings.filterwarnings('ignore')
lda_model_sincere_4 = gensim.models.ldamodel.LdaModel(
    corpus=sincere_corpus, num_topics=4, id2word=sincere_questions_dictionary,
    random_state=25, passes=5)
# Creating a lda model, which create 4 topics from the insincere questions
lda_model_insincere_4 = gensim.models.ldamodel.LdaModel(
    corpus=insincere_corpus, num_topics=4, id2word=insincere_questions_dictionary,
    random_state=25, passes=5)
# Printing the result of the modeling for sincere questions.
pprint(lda_model_sincere_4.print_topics()[0])
# Printing the result of the modeling for insincere questions.
pprint(lda_model_insincere_4.print_topics()[0])
# Display the topic modeling for sincere questions.
pyLDAvis.gensim.prepare(lda_model_sincere_4, sincere_corpus, sincere_questions_dictionary)
# # Display the topic modeling for insincere questions.
pyLDAvis.gensim.prepare(lda_model_insincere_4, insincere_corpus, insincere_questions_dictionary)
# I import all the functions from sklearn.
from sklearn.metrics import classification_report
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
# I split the date into train and test, variable and target. 
X_train, X_test, y_train, y_test = train_test_split(X_train, y_train)
X_train.shape, y_train.shape, X_test.shape, y_test.shape
# I create of the transformer and the estimator and insertion in a pipeline.
tfidv = TfidfVectorizer(lowercase=True, stop_words='english')
multinomialnb = MultinomialNB()
pipe = make_pipeline(tfidv, multinomialnb)
pipe
# I fit of the pipeline on X_train and y_train,
# then predict on the y_test and estimate of the predictions. 
pipe.fit(X_train, y_train)
y_pred = pipe.predict(X_test)
print(classification_report(y_test, y_pred))
