import pandas as pd

df = pd.read_csv('../input/Hotel_Reviews.csv')

df['Tags'].head()

col = ['Negative_Review', 'Positive_Review', 'Tags'] 

df = df[col]
nr = df.Negative_Review

pr = df.Positive_Review

nr.replace({'No Negative' : ''}, inplace = True)

pr.replace({'No Positive' : ''}, inplace = True)
lt = df['Tags'].str.contains(' Leisure trip ')

bt = df['Tags'].str.contains(' Business trip ')

denull = (lt != bt)

#i hav checked that no review tagged 'leisure trip' and 'business trip' at the same time.
data = pd.DataFrame({'Review': nr + pr, 'TripType': lt})

data = data[denull].reset_index(drop = True)

data.replace({True: 'Leisure', False: 'Business'}, inplace = True)
data[10:15]

#500717 reviews
import numpy as np

import nltk

data.dropna(inplace = True)

data.reset_index(drop = True, inplace = True)



#Tokenization: 

from nltk.tokenize import word_tokenize

word_tokenized = data.Review.apply(word_tokenize)

data.insert(0, "WordToken", word_tokenized)





#Anomaly, Weird records: Drop the empty review(or we could remove the review with less than n words?)

word_count = data.WordToken.apply(lambda x: len(x))  

#maybe word count could be took as a feature?

#data.insert(0, 'WordCount', data.WordToken.apply(lambda x: len(x)))

filter_count = (word_count >= 1) #if 5, 462350 remained 

data = data[filter_count]

data.reset_index(drop = True, inplace = True)

#500487 remained.





#StopWordRemoval: Remove the NLTK build-in stopwords.

#from nltk.corpus import stopwords

#stop_words = set(stopwords.words('english'))

#wosw = data.WordToken.apply(lambda x:  [item for item in x if item not in stop_words])



#Lemmatization

#from nltk.stem import WordNetLemmatizer



#wnl = WordNetLemmatizer()

#def lemmatize_text(text):

#    return [wnl.lemmatize(w) for w in text]

#data.insert(0, 'Lemmatized',data.WordToken.apply(lemmatize_text))
#Split randomly with the test size 0.33 (same distribution of classes)

from sklearn.model_selection import train_test_split

train, test = train_test_split(data, stratify = data['TripType'], test_size = 0.33, random_state = 1)



train.reset_index(drop = True, inplace = True)

test.reset_index(drop = True, inplace = True)



#335K v.s. 165K

#with 0.83 of Leisure Trip



#Labeling

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

lbl_train, lbl_test = le.fit_transform(train.TripType), le.transform(test.TripType)
# A-1: TFIDF

def dum(doc):

    return doc



from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(

                             ngram_range = (1,1),

                             tokenizer = dum, 

                             preprocessor = dum,

                             min_df = 0.01)



#x_train = vectorizer.fit_transform(train.WordToken).toarray()

#x_test = vectorizer.transform(test.WordToken).toarray()
#from sklearn.preprocessing import StandardScaler

#idf_dict = vectorizer.idf_[:, np.newaxis]

#ss = StandardScaler()

#weight = ss.fit_transform(idf_dict)

#dictionary = dict(zip(vectorizer.get_feature_names(), weight.flatten()))
# A-2: W2V, with using the mean of word vectors to represent a review.



from gensim.models import Word2Vec, KeyedVectors

from gensim.test.utils import common_texts, get_tmpfile

model = Word2Vec(size = 300, window = 2, min_count = 1)



#Building the dictionary.

model.build_vocab(train.WordToken)



#Train the w2v model with WordTokens

model.train(train.WordToken, total_examples = len(train.WordToken), epochs = 3)

model.init_sims(replace = True)



#To lower the memory usage, save/load in KV model is necessary...

fpath = get_tmpfile("w2v.kv")

model.wv.save(fpath)

del model

wv = KeyedVectors.load(fpath, mmap='r')



#Document vector: By simply get the average of the wordvec in a review.

def doc_vec(doc, mean = np.zeros(wv.vector_size)):  

    doc = [word for word in doc] # target input list of words

    try:

        return np.mean(wv[doc], axis = 0)

       # return np.mean(np.multiply(wv[doc] , np.array(idfweight(doc))[:, np.newaxis]), axis = 0)

    except:

        return mean  #13168/165161 is zero vector



#x_train = np.vstack(train.WordToken.apply(doc_vec))  #not sure abou whether a more efficient way

#x_test = np.vstack(test.WordToken.apply(lambda x : doc_vec(x, np.mean(x_train, axis = 0)) ))  
def dum(doc):

    return doc



from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(

                             ngram_range = (1,2),  # uni-to-bigram perform better, but out of memory on kaggle's notebook.

                             tokenizer = dum, 

                             preprocessor = dum,

                             min_df = 0.001,

                             max_df = 0.4)  



bt = train[train.TripType == 'Business']  

vectorizer.fit(bt.WordToken)

#fit only the instances with Business only.

x_train = vectorizer.transform(train.WordToken).toarray()

x_test = vectorizer.transform(test.WordToken).toarray()
'''

from sklearn.preprocessing import StandardScaler, KBinsDiscretizer



#a = scaler.fit_transform(train.WordCount[:, np.newaxis])  # fit does nothing.

#b = scaler.transform(test.WordCount[:, np.newaxis])

a = np.array(train.WordCount).reshape(len(train.WordCount), 1) # fit does nothing.

b = np.array(test.WordCount).reshape(len(test.WordCount), 1)



kbin = KBinsDiscretizer(n_bins = 17, encode = 'onehot-dense', strategy = 'quantile')

a = kbin.fit_transform(a) 

b = kbin.transform(b)



#Cobiine the wordcount in to be treated as the feature.

x_train = np.append(x_train, a, axis = 1)

x_test = np.append(x_test, b , axis = 1)

'''
#POS filter: filtered the other word without given POS tag.

#Warning: It takes time.



def pos_tagging(sent):

    tagfilter = {'NOUN'}  # e.g. 'NOUN', 'VERB'...

    target = [item for (item, tag) in nltk.pos_tag(sent, tagset = 'universal') if tag in tagfilter]

    return target



data.insert(0, 'POS', data.Lemmatized.apply(pos_tagging))



#then apply on the TFIDF or W2V later.
#Execute each of the vectorization approach to two variables

#Assign the new feature vectors(combination) to x_train, x_test



#x_train_t, x_test_t = ...(A-1)

#x_train_w, x_test_t = ...(A-2)

x_train = np.append(x_train_t, x_train_w, axis = 1)

x_test = np.append(x_test_t, x_test_w, axis = 1)



#Dimentionality is 300+434 = 734
#Feature Selection: Fitting the ANOVA stats on training data.

from sklearn.feature_selection import f_classif, SelectKBest, VarianceThreshold, chi2



def FeatureSelect_f(feature, target, d):

    fs = SelectKBest(f_classif, k = d).fit(feature, target)

    return fs



#Supervised approach

fs = FeatureSelect_f(x_train, lbl_train, 500)

#Transform both training/testing set into new dimensionality.

x_train = fs.transform(x_train)

x_test = fs.transform(x_test)
from sklearn.naive_bayes import GaussianNB

clf_gnb = GaussianNB()

clf_gnb.fit(x_train, lbl_train)
#Evaluation function:

from sklearn.metrics import precision_recall_curve, auc, confusion_matrix, accuracy_score, classification_report

#from imblearn.metrics import classification_report_imbalanced

import matplotlib.pyplot as plt

def evaluating(truth, pred, ax=object):

  

    print(accuracy_score(truth, pred))

    print(classification_report(truth, pred))    

    print(confusion_matrix(truth, pred))

    precision, recall, threshold = precision_recall_curve(truth, pred)



    ax.step(recall, precision, color='b', alpha=1, where='post')

    ax.fill_between(recall, precision, step='post', alpha=0.5, color='b')

    ax.set_xlabel('Recall')

    ax.set_ylabel('Precision')

    ax.set_ylim([0.0, 1.05])

    ax.set_xlim([0.0, 1.0])

    ax.set_title('Precision-Recall curve')

    return ax

#ref: https://scikit-learn.org/stable/auto_examples/model_selection/plot_precision_recall.html
f, (ax1) = plt.subplots(1, 1, figsize=(10,10))



pred_gnb = clf_gnb.predict(x_test)

evaluating(lbl_test, pred_gnb, ax1)



#Take as an imbalanced Problem??