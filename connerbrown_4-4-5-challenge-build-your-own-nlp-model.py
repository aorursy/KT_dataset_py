import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from nltk.corpus import gutenberg, stopwords
import spacy
import time
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression

doc_length = 50000 #None
# pick out 'chesterton-ball.txt', 'chesterton-brown.txt', 'chesterton-thursday.txt' and try to classify the stories
ball = gutenberg.raw('chesterton-ball.txt')[:doc_length]
brown = gutenberg.raw('chesterton-brown.txt')[:doc_length]
thursday = gutenberg.raw('chesterton-thursday.txt')[:doc_length]
nlp = spacy.load('en')
ball_doc = nlp(ball)
print("tokens in ball",len(ball_doc))
brown_doc = nlp(brown)
print("tokens in brown",len(brown_doc))
thursday_doc = nlp(thursday)
print("tokens in thursday",len(thursday_doc))
ball_sents = [[sent, "Ball"] for sent in ball_doc.sents]
brown_sents = [[sent, "Brown"] for sent in brown_doc.sents]
thursday_sents = [[sent, "Thursday"] for sent in thursday_doc.sents]

# Combine the sentences from the two novels into one data frame.
sentences = pd.DataFrame(ball_sents + brown_sents + thursday_sents)
def span_to_str(span):
    return span.as_doc().text
str_sentences = sentences[0].apply(span_to_str)
def bag_of_words(text,most_common_length = 1000):
    allwords = [token.lemma_ for token in text
               if not token.is_punct
                and not token.is_stop]
    return [item[0] for item in Counter(allwords).most_common(most_common_length)]
def bow_features(sentences, common_words):
    df = pd.DataFrame(columns=common_words)
    df['text_sentence'] = sentences[0]
    df['text_source'] = sentences[1]
    df.loc[:, common_words] = 0
    for i, sentence in enumerate(df['text_sentence']):
        words = [token.lemma_
                 for token in sentence
                 if (
                     not token.is_punct
                     and not token.is_stop
                     and token.lemma_ in common_words)
                ]
        
        # Populate the row with word counts.
        for word in words:
            df.loc[i, word] += 1
        
        # This counter is just to make sure the kernel didn't hang.
        if i % 500 == 0:
            print("Processing row {}".format(i))
            
    return df

# Set up the bags.
ball_words = bag_of_words(ball_doc)
brown_words = bag_of_words(brown_doc)
thursday_words = bag_of_words(thursday_doc)
# Combine bags to create a set of unique words.
common_words = set(ball_words + brown_words + thursday_words)
start_time=time.time()
word_counts = bow_features(sentences, common_words)
print("done\nProcessing time {:0.1f} seconds".format(time.time()-start_time))
X_bow = word_counts.drop(['text_sentence','text_source'],axis=1)
print("Number of BoW features",len(X_bow.columns))
word_counts.head(3)
# initialize tfidf
vectorizer = TfidfVectorizer(max_df=0.5, # drop words that occur in more than 95% of sentences
                             min_df=2, # only use words that appear at least twice
                             stop_words='english', 
                             lowercase=True, #convert everything to lower case
                             use_idf=True,#we definitely want to use inverse document frequencies in our weighting
                             norm=u'l2', #Applies a correction factor so that longer paragraphs and shorter paragraphs get treated equally
                             smooth_idf=True #Adds 1 to all document frequencies, as if an extra document existed that used every word once.  Prevents divide-by-zero errors
                            )


#Applying the vectorizer
sentences_tfidf=vectorizer.fit_transform(str_sentences)
print("Number of tfidf features: %d" % sentences_tfidf.get_shape()[1])
X_tfidf = sentences_tfidf.toarray()
# Naive Bayes
gnb = GaussianNB()
# Logistic Regression
lr = LogisticRegression(random_state=42)
# SVM
svc = SVC(gamma='auto',random_state=42)
# Gradient Boost
gbc = GradientBoostingClassifier(random_state=42)
# cross_val_score each model
model_names = {gnb:"Naive Bayes",
              lr: "Logistic Regression",
              svc: "Support Vector Machine",
              gbc: "Gradient Boost"}
models = [gnb,lr,svc,gbc]
for model in models:
    print("\n\n",model_names[model])
    start_bow = time.time()
    bow_score = cross_val_score(model,X_bow,word_counts['text_source'],cv=5,n_jobs=-1)
    print("BOW\t{:0.4f} +/- {:0.4f}\t({:0.4f} seconds)".format(bow_score.mean(),bow_score.std(),time.time()-start_bow))
    start_tfidf = time.time()
    tfidf_score = cross_val_score(model,X_tfidf,word_counts['text_source'],cv=5,n_jobs=-1)
    print("TFIDF\t{:0.4f} +/- {:0.4f}\t({:0.4f} seconds)".format(tfidf_score.mean(),tfidf_score.std(),time.time()-start_tfidf))
def bag_of_words_improve(text):
    allwords = [token.lemma_
                for token in text
                if not token.is_punct
                and not token.is_stop]
    return [item[0] for item in Counter(allwords).most_common(100)]
def bow_features_improve(sentences, common_words):
    df = pd.DataFrame(columns=common_words)
    df['text_sentence'] = sentences[0]
    df['text_source'] = sentences[1]
    df.loc[:,'word_count'] = 0 # add word count
    df.loc[:,'entity_count'] = 0 # add entity count
    df.loc[:, common_words] = 0
    for i, sentence in enumerate(df['text_sentence']):
        words = [token.lemma_
                 for token in sentence
                 if (
                     not token.is_punct
                     and not token.is_stop
                     and token.lemma_ in common_words
                 )]
        
        # Populate the row with word counts.
        for word in words:
            df.loc[i, word] += 1
        df.loc[i,'entity_count'] = len(sentence.as_doc().ents)
        df.loc[i,'word_count'] = len(words)        
        # This counter is just to make sure the kernel didn't hang.
        if i % 500 == 0:
            print("Processing row {}".format(i))
            
    return df

# Set up the bags.
ball_words_improve = bag_of_words_improve(text=ball_doc)
brown_words_improve = bag_of_words_improve(text=brown_doc)
thursday_words_improve = bag_of_words_improve(text=thursday_doc)
# Combine bags to create a set of unique words.
common_words_improve = set(ball_words_improve + brown_words_improve + thursday_words_improve)
start_time=time.time()
df_improve = bow_features_improve(sentences, common_words_improve)
print("done\nProcessing time {:0.1f} seconds".format(time.time()-start_time))
X_improve = df_improve.drop(['text_sentence','text_source'],axis=1)
print("Number of Improved BoW features",len(X_improve.columns))
bow_score = cross_val_score(gnb,X_bow,word_counts['text_source'],cv=5,n_jobs=-1)
print("Original Score\n{:0.4f} +/- {:0.4f}".format(bow_score.mean(),bow_score.std()))
start_gbc_tune = time.time()
gnb_improve = GaussianNB()
improve_score = cross_val_score(gnb_improve,X_improve,df_improve['text_source'],cv=5,n_jobs=-1)
print("Improved Score\n{:0.4f} +/- {:0.4f}\t({:0.4f} seconds)".format(improve_score.mean(),improve_score.std(),time.time() - start_gbc_tune))
