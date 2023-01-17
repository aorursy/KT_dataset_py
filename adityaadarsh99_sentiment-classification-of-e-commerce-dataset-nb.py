# Any results you write to the current directory are saved as output.

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Ignore all warnings

import warnings

warnings.filterwarnings("ignore")

        

# Importing Libraries

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from pandas_profiling import ProfileReport

import matplotlib.pyplot as plt

import seaborn as sn

from sklearn.model_selection import train_test_split # for spliting dataset

from sklearn.feature_extraction.text import CountVectorizer # bow-->1gram and 2 gram

from sklearn.feature_extraction.text import TfidfVectorizer # tf-idf

from gensim.models import Word2Vec  # w2v

from gensim.models import KeyedVectors # to understanding w2v using google pre trained model

from sklearn.metrics import accuracy_score # to check the accuracy of model

from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import cross_val_score # k-fold cv

from sklearn.metrics import classification_report

import pickle

from wordcloud import WordCloud

from collections import Counter

from tqdm import tqdm

import re

import nltk

from nltk.probability import FreqDist

from nltk.tokenize import word_tokenize

from nltk.corpus import stopwords

import string

eng_stopwords = stopwords.words('english')

from nltk.stem import PorterStemmer

from sklearn.metrics import confusion_matrix

from sklearn.metrics import balanced_accuracy_score

from sklearn.preprocessing import StandardScaler
# Loading CSV file

data = pd.read_csv("../input/womens-ecommerce-clothing-reviews/Womens Clothing E-Commerce Reviews.csv",index_col=0)



# Using only 'Review text' and 'Rating' and descarding other columns

data = data[['Review Text','Rating']] 



# Converting into binary classification problem

data = data[data['Rating']!=3]

data['Rating'] = data['Rating'].apply(lambda x: 0 if x<3 else 1)



# Shape of dataset

print("Shape of the dataset:",data.shape)



# Overview of data

print("\nOverview of data: ")

data.info
# Pandas Profiling : Really good library to get the overview EDA.

profile = ProfileReport(data, title='Pandas Profiling Report',minimal=False, html={'style':{'full_width':True}})

profile.to_widgets()
# Finding missing values

print(f"Number of Missing values: \n{data.isnull().sum()}\n")



print(f"number of duplicated reviews: {sum(data[data['Review Text'].notnull()].duplicated(['Review Text'],keep='first'))}")
# Duplicate Review text example

data[data['Review Text'].notnull()][data[data['Review Text'].notnull()].duplicated(['Review Text'],keep=False)]
# Removing datapoints having missing values and duplicate Review text

data_after_drop = data[data['Review Text'].notnull()]

data_after_drop = data_after_drop.drop_duplicates(['Review Text'],keep='first')



print(f"percentage of data remaing after dopping missing values and duplicate reviews: { round((data_after_drop.shape[0]/data.shape[0])*100,3)} %")
# Classlabel value counts



temp  = data_after_drop['Rating'].value_counts()

print(pd.DataFrame({'Class label(sentiment)':temp.index, "values_counts":temp.values,"distribution percentage":temp.values/sum(temp.values) }))



# Ploting distribution of classlabel

sn.countplot(data_after_drop['Rating'])

plt.show()
num_of_words = data_after_drop['Review Text'].apply(lambda x: len(str(x).split()))



# Ploting

sn.distplot(num_of_words[data_after_drop['Rating']==1],label = 'Positive Sentiments')

sn.distplot(num_of_words[data_after_drop['Rating']==0],label = 'Negative Sentiments')

plt.legend()

plt.title("Distribution of Number of words in Reviews text ")

plt.show()
"""sn.boxplot(num_of_words,hue=data_after_drop['Rating'])

plt.show()"""
# refer: https://www.datacamp.com/community/tutorials/wordcloud-python



text_train = " ".join(word for word in data_after_drop['Review Text'])



# Create and generate a word cloud image:

wordcloud = WordCloud().generate(text_train)



# Display the generated image:

plt.figure(figsize=(11,7))

plt.imshow(wordcloud, interpolation='bilinear')

plt.title("WordCloud of Review Text\n")

plt.axis("off")

plt.show()
# WordCloud hue by class label



f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 6))



# For Positive Sentiment 

text = " ".join(word for word in data_after_drop[data_after_drop['Rating']==1]['Review Text'])

# Create and generate a word cloud image:

wordcloud = WordCloud().generate(text)



# Display the generated image:

ax1.imshow(wordcloud, interpolation='bilinear')

ax1.set(title='WordCloud of Positive Text\n')

ax1.axis("off")



# -------------------------------------------------------------------------------------------------



# For Negative Sentiment 

text = " ".join(word for word in data_after_drop[data_after_drop['Rating']==0]['Review Text'])

# Create and generate a word cloud image:

wordcloud = WordCloud().generate(text)



# Display the generated image:

ax2.imshow(wordcloud, interpolation='bilinear')

ax2.set(title='WordCloud of Negative Review Text\n')

ax2.axis("off")

plt.show()
# Preprocessing Functions

# credit : https://www.kaggle.com/urvishp80/quest-encoding-ensemble



#======================================================================================================================================

# Return the number of links and text without html tags 

# Also return the counts of 'number of lines'  and remove it

def strip_html(text):

    """ Return theclean text (without html tags) """

    

    # Removing HTML tags

    text = re.sub(r'http[s]?://\S+'," ",text)

    

    # finding number of lines using regex and counting it and remove it

    text = re.sub(r'\n', " ",text)

    

    return  text





#======================================================================================================================================

mispell_dict = {"aren't" : "are not","can't" : "cannot","couldn't" : "could not","couldnt" : "could not","didn't" : "did not","doesn't" : "does not",

                "doesnt" : "does not","don't" : "do not","hadn't" : "had not","hasn't" : "has not","haven't" : "have not","havent" : "have not",

                "he'd" : "he would","he'll" : "he will","he's" : "he is","i'd" : "i would","i'd" : "i had","i'll" : "i will","i'm" : "i am",

                "isn't" : "is not","it's" : "it is","it'll":"it will","i've" : "I have","let's" : "let us","mightn't" : "might not",

                "mustn't" : "must not","shan't" : "shall not","she'd" : "she would","she'll" : "she will","she's" : "she is","shouldn't" : "should not",

                "shouldnt" : "should not","that's" : "that is","thats" : "that is","there's" : "there is","theres" : "there is","they'd" : "they would",

                "they'll" : "they will","they're" : "they are","theyre":  "they are","they've" : "they have","we'd" : "we would","we're" : "we are",

                "weren't" : "were not","we've" : "we have","what'll" : "what will","what're" : "what are","what's" : "what is","what've" : "what have",

                "where's" : "where is","who'd" : "who would","who'll" : "who will","who're" : "who are","who's" : "who is","who've" : "who have",

                "won't" : "will not","wouldn't" : "would not","you'd" : "you would","you'll" : "you will","you're" : "you are","you've" : "you have",

                "'re": " are","wasn't": "was not","we'll":" will","didn't": "did not","tryin'":"trying"}



def _get_mispell(mispell_dict):

    mispell_re = re.compile('(%s)' % '|'.join(mispell_dict.keys()))

    return mispell_dict, mispell_re



def replace_typical_misspell(text):

    

    text = text.lower()

    

    

    """De-Concatenation of words and correction of misspelled words"""

    mispellings, mispellings_re = _get_mispell(mispell_dict)



    def replace(match):

        return mispellings[match.group(0)]



    return mispellings_re.sub(replace, text)





#======================================================================================================================================

# removing non_alpha_numeric character and removing all the special character words

def non_alpha_numeric_remove(text):  

    

    # removing all non alpha char 

    text = re.sub(r"[^A-Za-z]", " ",text)

    

    return text 



#======================================================================================================================================



# function to remove all the stopwords and words having lengths less than 3

def remove_stop_words_and_punc(text) :

    

    """ 

    Remove all the stopwords 

    

    """

    # removing the words from the stop words list: 'no', 'nor', 'not'

    stops = set(stopwords.words("english"))

    stops.remove('no')

    stops.remove('nor')

    stops.remove('not')

    

    clean_text = []

    for word in text.split():

        if word not in stops and len(word)>3:        

            clean_text.append(word)

        

    clean_text = " ".join(clean_text)

    

    return(clean_text)



#======================================================================================================================================

# function for stemming of words in text

def stem(text):

    stemmer = PorterStemmer()

    result = " ".join([ stemmer.stem(word) for word in text.split(" ")])

    return result



#======================================================================================================================================

#======================================================================================================================================

# Final text cleaning funtion  

def clean_text(text):

    """

    This function sequentially execute all the cleaning and preprocessing function and finaly gives cleaned text.

    Input: Boolean values of extra_features, strip_html, count_all_cap_words_and_lower_it, replace_typical_misspell, count_non_alpha_numeric_and_remove, remove_stop_words_and_punc, stem

            (by default all the input values = True)

    

    return: clean text

    

    """

    

    # remove html tags

    clean_text = strip_html(text)  

    

    # de-concatenation of words

    clean_text = replace_typical_misspell(clean_text)

     

    # Count the number of non alpha numeric character and remove it

    clean_text = non_alpha_numeric_remove(clean_text)

    

    # removing Stopwords and the words length less than 3(As these words mostly tend to redundant words) excpect 'C' and 'R'and 'OS' <-- programing keywords

    clean_text = remove_stop_words_and_punc(clean_text)

    

    # stemming ( use only for BOW or TFIDF represention. Not effective for word embedding like w2v or glove)

    clean_text = stem(clean_text)



    return clean_text
# Preprocessing 

cleaned_review_text = data_after_drop['Review Text'].apply(lambda x: clean_text(x))



# Sample

i=15

print(f"\nBefore Preprocessing\n{'='*20}")

print(data_after_drop['Review Text'].iloc[i])



print(f"\nAfter Preprocessing\n{'='*20}")

print(cleaned_review_text.iloc[i])
# Calculating the length of text before and after preprocessing

len_after_cleaning = cleaned_review_text.apply(lambda x : len(x.split()))

len_before_cleaning = data_after_drop['Review Text'].apply(lambda x : len(x.split()))

    

# ploting

plt.figure(figsize=(9,6))

sn.distplot(len_before_cleaning, label=f'Review text before cleaning')

sn.distplot(len_after_cleaning, label=f'Review text after cleaning')

plt.title(f" Distribution of number of words of Review text before v/s after preprocessing\n",fontsize=15)

plt.ylabel("distribtion")

plt.xlabel(f"number of words in Review text")

plt.legend()

plt.grid()

plt.show()
"""# Calculating the length of text before and after preprocessing

len_after_cleaning = cleaned_review_text[data_after_drop['Rating']==1].apply(lambda x : len(x.split()))

len_before_cleaning = cleaned_review_text[data_after_drop['Rating']==0].apply(lambda x : len(x.split()))



# ploting

plt.figure(figsize=(9,6))

sn.distplot(len_before_cleaning, label=f'Review text before cleaning')

sn.distplot(len_after_cleaning, label=f'Review text after cleaning')

plt.title(f" Distribution of number of words of Review text before v/s after preprocessing\n",fontsize=15)

plt.ylabel("distribtion")

plt.xlabel(f"number of words in Review text")

plt.legend()

plt.grid()

plt.show()"""



print("number of words after preprocessing hue by class label")
# WordCloud hue by class label



f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 6))



# For Positive Sentiment 

text = " ".join(word for word in cleaned_review_text[data_after_drop['Rating']==1])

# Create and generate a word cloud image:

wordcloud = WordCloud().generate(text)



# Display the generated image:

ax1.imshow(wordcloud, interpolation='bilinear')

ax1.set(title='WordCloud of Positive Text\n')

ax1.axis("off")



# -------------------------------------------------------------------------------------------------



# For Negative Sentiment 

text = " ".join(word for word in cleaned_review_text[data_after_drop['Rating']==0])

# Create and generate a word cloud image:

wordcloud = WordCloud().generate(text)



# Display the generated image:

ax2.imshow(wordcloud, interpolation='bilinear')

ax2.set(title='WordCloud of Negative Review Text\n')

ax2.axis("off")

plt.show()
top = Counter([item for sublist in cleaned_review_text[data_after_drop['Rating']==1] for item in str(sublist).split()])

temp = pd.DataFrame(top.most_common(20))

temp = temp.iloc[1:,:]

temp.columns = ['Common_words','count']

temp.style.background_gradient(cmap='Purples')
top = Counter([item for sublist in cleaned_review_text[data_after_drop['Rating']==0] for item in str(sublist).split()])

temp = pd.DataFrame(top.most_common(20))

temp = temp.iloc[1:,:]

temp.columns = ['Common_words','count']

temp.style.background_gradient(cmap='Purples')
# train test split



X = cleaned_review_text

y = data_after_drop['Rating']

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.33, random_state=42)



print(f"Shape of X_train: {X_train.shape}")

print(f"Shape of y_train: {y_train.shape}")

print(f"Shape of X_val: {X_val.shape}")

print(f"Shape of y_val: {y_val.shape}")
from sklearn.feature_extraction.text import CountVectorizer

count_vect = CountVectorizer(ngram_range=(1, 1), max_df=1.0, min_df=1, max_features=None, vocabulary=None)

x_train_bow_unigram = count_vect.fit_transform(X_train)

x_val_bow_unigram = count_vect.transform(X_val)



print(f"shape of features after BOW Feture extraction: {x_train_bow_unigram.shape}")



# Sparsity of BOW-unigram

sparsiry_bow = (len(x_train_bow_unigram.toarray().nonzero()[0]) / len(np.nonzero(x_train_bow_unigram.toarray()==0)[0]))*100

print(f"Sparsity of BOW: {round(sparsiry_bow,5)}%")
# BOW feature representaiona

bow_unigram_feature_representation = pd.DataFrame(data = x_train_bow_unigram.toarray(), columns = count_vect.get_feature_names())

bow_unigram_feature_representation.head()
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf_vect = TfidfVectorizer(ngram_range=(1, 1), max_df=1.0, min_df=1, max_features=None, vocabulary=None)

x_train_tfidf_unigram = tfidf_vect.fit_transform(X_train)

x_val_tfidf_unigram = tfidf_vect.transform(X_val)



print(f"shape of features after BOW Feture extraction: {x_train_tfidf_unigram.shape}")



# Sparsity of TFIDF-unigram

sparsiry_tfidf = (len(x_train_tfidf_unigram.toarray().nonzero()[0]) / len(np.nonzero(x_train_tfidf_unigram.toarray()==0)[0]))*100

print(f"Sparsity of TFIDF: {round(sparsiry_tfidf,5)}%")
# Tf-IDF feature representaion

tfidf_unigram_feature_representation = pd.DataFrame(data = x_train_tfidf_unigram.toarray(), columns = tfidf_vect.get_feature_names())

tfidf_unigram_feature_representation.head()
# Text Preprocessing funtion for word2vec

def clean_text_for_embedding(text):

    """

    This function sequentially execute all the cleaning and preprocessing function and finaly gives cleaned text.

    Input: Boolean values of extra_features, strip_html, count_all_cap_words_and_lower_it, replace_typical_misspell, count_non_alpha_numeric_and_remove, remove_stop_words_and_punc, stem

            (by default all the input values = True)

    

    return: clean text """

    

    # remove html tags

    clean_text = strip_html(text)  

    

    # de-concatenation of words

    clean_text = replace_typical_misspell(clean_text)

     

    # Count the number of non alpha numeric character and remove it

    clean_text = non_alpha_numeric_remove(clean_text)

    

    # removing Stopwords and the words length less than 3(As these words mostly tend to redundant words) excpect 'C' and 'R'and 'OS' <-- programing keywords

    clean_text = remove_stop_words_and_punc(clean_text)



    return clean_text
# Preprocessing for word2vec embedding for train and test review text

X_train_review_text_for_embedding = X_train.apply(lambda x: clean_text_for_embedding(x))

X_val_review_text_for_embedding = X_val.apply(lambda x: clean_text_for_embedding(x))





# Sample

i=15

print(f"\nBefore Preprocessing\n{'='*20}")

print(X_train.iloc[i])



print(f"\nAfter Preprocessing\n{'='*20}")

print(X_train_review_text_for_embedding.iloc[i])
import operator 

import gensim

from gensim.models import KeyedVectors



# Train the genisim word2vec model with our own custom corpus

# CBOW -> sg = 0



# Convering text in list of list of train reviews text

list_of_sent_train = X_train_review_text_for_embedding.apply(lambda x: x.split()).values



# Convering text in list of list of val reviews text

list_of_sent_val = X_val_review_text_for_embedding.apply(lambda x: x.split()).values



# Traing W2V

model_cbow = Word2Vec(sentences= list_of_sent_train, min_count=3, sg=0, workers= 3,size=100) # Default setting
# Vocab after training

words = model_cbow.wv.vocab.keys()

print("Number of words in vocab",len(words),"\n\n")

print(words,sep='\n')
# Top similar word

model_cbow.similar_by_word("good")
'''

    -->procedure to make avg w2v of each reviews

    

    1. find the w2v of each word

    2. sum-up w2v of each word in a sentence

    3. divide the total w2v of sentence by total no. of words in the sentence

'''



# vocablary of w2v model of e-commerce dataset

vocab=model_cbow.wv.vocab





#------------------------------------------------------------------------------------------------------------

## average Word2Vec for train reviews

# compute average word2vec for each review.

train_w2v_cbow = [] # the avg-w2v for each sentence/review in train dataset is stored in this list



list_of_sent_train = X_train_review_text_for_embedding.apply(lambda x: x.split()).values



for sent in list_of_sent_train: # for each review/sentence

    sent_vec = np.zeros(100) # as word vectors are of zero length

    cnt_words =0; # num of words with a valid vector in the sentence/review

    for word in sent: # for each word in a review/sentence

        if word in vocab:

            vec = model_cbow.wv[word]

            sent_vec += vec

            cnt_words += 1

    if cnt_words != 0:

        sent_vec /= cnt_words

    train_w2v_cbow.append(sent_vec)



print("Number of datapoints in train: ",len(train_w2v_cbow))





#------------------------------------------------------------------------------------------------------------



## average Word2Vec for val reviews

# compute average word2vec for each review.

val_w2v_cbow = [] # the avg-w2v for each sentence/review in train dataset is stored in this list



list_of_sent_train = X_val_review_text_for_embedding.apply(lambda x: x.split()).values



for sent in list_of_sent_val: # for each review/sentence

    sent_vec = np.zeros(100) # as word vectors are of zero length

    cnt_words =0; # num of words with a valid vector in the sentence/review

    for word in sent: # for each word in a review/sentence

        if word in vocab:

            vec = model_cbow.wv[word]

            sent_vec += vec

            cnt_words += 1

    if cnt_words != 0:

        sent_vec /= cnt_words

    val_w2v_cbow.append(sent_vec)



print("Number of datapoints in val: ",len(val_w2v_cbow))
# Standard scaling of W2V

sc = StandardScaler()

train_w2v_sc = sc.fit_transform(train_w2v_cbow)

val_w2v_sc = sc.transform(val_w2v_cbow)





## Example : Review text is encoded into 100 dim vector space

print(f"\n Before encoding: \n{'='*20}\n {X_train_review_text_for_embedding.iloc[0]}")

print(f"\n After encoding: \n{'='*20}\n {train_w2v_sc[0]}")
from sklearn.preprocessing import StandardScaler

from sklearn.decomposition import PCA

import seaborn as sns



# PCA for visualisation

pca = PCA(n_components=2)

principalComponents = pca.fit_transform(train_w2v_sc)

principalDf = pd.DataFrame(data = principalComponents

             , columns = ['principal component 1', 'principal component 2'])



# Ploting

sns.scatterplot(x='principal component 1', y='principal component 2', hue=y_train.values, style=None, size=None, data=principalDf)

plt.show()
# Loading Glove(pretrained) model

GLOVE_EMBEDDING_PATH = '../input/glove840b300dtxt/glove.840B.300d.txt'



def get_coefs(word, *arr):

    return word, np.asarray(arr, dtype='float32')



def load_embeddings(path):

    with open(path) as f:

        return dict(get_coefs(*line.strip().split(' ')) for line in tqdm(f))

    

embeddings_index = load_embeddings(GLOVE_EMBEDDING_PATH)
## Building vocubulary from our Quest Data

def build_vocab(sentences, verbose =  True):

    """

    :param sentences: list of list of words

    :return: dictionary of words and their count

    """

    vocab = {}

    for sentence in tqdm(sentences, disable = (not verbose)):

        for word in sentence:

            try:

                vocab[word] += 1

            except KeyError:

                vocab[word] = 1

    return vocab



#=========================================================================================================

import operator 

## This is a common function to check coverage between our quest data and the word embedding

def check_coverage(vocab,embeddings_index):

    a = {}

    oov = {}

    k = 0

    i = 0

    for word in tqdm(vocab):

        try:

            a[word] = embeddings_index[word]

            k += vocab[word]

        except:



            oov[word] = vocab[word]

            i += vocab[word]

            pass



    print('Found embeddings for {:.2%} of vocab'.format(len(a) / len(vocab)))

    print('Found embeddings for  {:.2%} of all text'.format(k / (k + i)))

    sorted_x = sorted(oov.items(), key=operator.itemgetter(1))[::-1]



    return sorted_x
##Apply the vocab function to get the words and the corresponding counts

sentences = X_train_review_text_for_embedding.apply(lambda x: x.split()).values

vocab = build_vocab(sentences)



print(f"\nFor cleaned_question_body_for_embedding: \n{'-'*40}")

oov = check_coverage(vocab,embeddings_index)



## List 10 out of vocabulary word

print(f"\nTop 10 out of vocabulary word: \n{'-'*30}")

oov[:10]
#------------------------------------------------------------------------------------------------------------

## average Word2Vec usnig pretrained model(GLOVE) for train reviews

# compute average word2vec for each review.

train_w2v_pretrained = [] # the avg-w2v for each sentence/review in train dataset is stored in this list



list_of_sent_train = X_train_review_text_for_embedding.apply(lambda x: x.split()).values



for sent in list_of_sent_train: # for each review/sentence

    sent_vec = np.zeros(300) # as word vectors are of zero length

    cnt_words =0; # num of words with a valid vector in the sentence/review

    for word in sent: # for each word in a review/sentence

        if word in vocab:

            try:

                vec = embeddings_index[word]

                sent_vec += vec

                cnt_words += 1

            

            except:

                pass

            

    if cnt_words != 0:

        sent_vec /= cnt_words

    train_w2v_pretrained.append(sent_vec)



print("Number of datapoints in train: ",len(train_w2v_pretrained))





#------------------------------------------------------------------------------------------------------------



## average Word2Vec for val reviews

# compute average word2vec for each review.

val_w2v_pretrained = [] # the avg-w2v for each sentence/review in train dataset is stored in this list



list_of_sent_train = X_val_review_text_for_embedding.apply(lambda x: x.split()).values



for sent in list_of_sent_val: # for each review/sentence

    sent_vec = np.zeros(300) # as word vectors are of zero length

    cnt_words =0; # num of words with a valid vector in the sentence/review

    for word in sent: # for each word in a review/sentence

        if word in vocab:

            try:     

                vec = embeddings_index[word]

                sent_vec += vec

                cnt_words += 1

                

            except:

                pass

    if cnt_words != 0:

        sent_vec /= cnt_words

    val_w2v_pretrained.append(sent_vec)



print("Number of datapoints in val: ",len(val_w2v_pretrained))
# Standard scaling of W2V

sc = StandardScaler()

train_w2v_pretrained_sc = sc.fit_transform(train_w2v_pretrained)

val_w2v_pretrained_sc = sc.transform(val_w2v_pretrained)





## Example : Review text is encoded into 100 dim vector space

print(f"\n Before encoding: \n{'='*20}\n {X_train_review_text_for_embedding.iloc[0]}")

print(f"\n After encoding: \n{'='*20}\n {train_w2v_pretrained_sc[0]}")
from sklearn.preprocessing import StandardScaler

from sklearn.decomposition import PCA

import seaborn as sns



# PCA for visualisation

pca = PCA(n_components=2)

principalComponents = pca.fit_transform(train_w2v_pretrained_sc)

principalDf = pd.DataFrame(data = principalComponents

             , columns = ['principal component 1', 'principal component 2'])



# Ploting

sns.scatterplot(x='principal component 1', y='principal component 2', hue=y_train.values, style=None, size=None, data=principalDf)

plt.show()
from sklearn.naive_bayes import MultinomialNB

from sklearn.metrics import log_loss,classification_report

import matplotlib.pyplot as plt



"""

y_true : array, shape = [n_samples] or [n_samples, n_classes]

True binary labels or binary label indicators.



y_score : array, shape = [n_samples] or [n_samples, n_classes]

Target scores, can either be probability estimates of the positive class, confidence values, or non-thresholded measure of

decisions (as returned by “decision_function” on some classifiers). 

For binary y_true, y_score is supposed to be the score of the class with greater label.



"""



train_loss = []

cv_loss = []

alpha_range = [10e-5,10e-4,10e-3,10e-2,1,10,10e1,10e2,10e3]

for i in alpha_range:

    

    # Training

    mnb = MultinomialNB(alpha=i, fit_prior=False, class_prior=None)

    mnb.fit(x_train_bow_unigram, y_train)

    

    # Predicting

    y_train_pred = mnb.predict_proba(x_train_bow_unigram)

    y_cv_pred = mnb.predict_proba(x_val_bow_unigram)

    

    # Loss metric storing

    train_loss.append(log_loss(y_train, y_train_pred))

    cv_loss.append(log_loss(y_val, y_cv_pred))

    



    

# Visualising and finding optimal parameter 

plt.plot(np.arange(1,10,1), train_loss, label='Train loss')

plt.plot(np.arange(1,10,1), cv_loss, label='CV loss')

plt.xticks( np.arange(1,10,1), (10e-5, 10e-4, 10e-3, 10e-2, 10e-1, 10e0, 10e1, 10e2, 10e3))

plt.legend()

plt.xlabel("alpha: hyperparameter")

plt.ylabel("log loss")

plt.title("ERROR PLOTS")

plt.grid()

plt.show()







## Training using Optimal hyperparemeter

# using optimum_k to find generalistion loss



optimum_alpha = alpha_range[np.argmin(cv_loss)] #optimum 'alpha'



# Naive Bayes training

print(f"Traing using optimal alpha:  {alpha_range[np.argmin(cv_loss)]}\n")

clf=MultinomialNB(alpha=optimum_alpha, fit_prior=False, class_prior=None)

clf.fit(x_train_bow_unigram,y_train)



y_pred = clf.predict(x_val_bow_unigram)

y_pred_proba = clf.predict_proba(x_val_bow_unigram)



# Result track

accuracy = accuracy_score(y_val,y_pred)

bal_accuracy = balanced_accuracy_score(y_val,y_pred)

logloss = log_loss(y_val,y_pred_proba)

print(f'\nGenearalisation log_loss: {logloss:.3f}')

print(f"\nGeneralisation Accuracy: {(round(accuracy,2))*100}%")

print(f"\nGeneralisation Balance accuracy: {(round(bal_accuracy,2))*100}%")

print(f'\nmisclassification percentage: {(1-accuracy)*100:.2f}%')





#ploting confusion matrix

sn.heatmap(confusion_matrix(y_pred,y_val),annot=True, fmt="d",linewidths=.5)

plt.title('Confusion Matrix')

plt.xlabel('Predicted values')

plt.ylabel('Actual values')

plt.show()

# Classification Report

print("\n\nclassification report:\n",classification_report(y_val,y_pred)) 

# imbalanced-learn API: https://imbalanced-learn.readthedocs.io/en/stable/generated/imblearn.over_sampling.SMOTE.html

from imblearn.over_sampling import SMOTE



# transform the dataset

oversample = SMOTE(sampling_strategy='auto',random_state=None,k_neighbors=5,n_jobs=None)

X_res_bow, y_res = oversample.fit_resample(x_train_bow_unigram, y_train)



# summarize the new class distribution

counter = Counter(y_res)

print("After applying SMOTE: ",counter)
train_loss = []

cv_loss = []

alpha_range = [10e-5,10e-4,10e-3,10e-2,1,10,10e1,10e2,10e3]

for i in alpha_range:

    

    # Training

    mnb = MultinomialNB(alpha=i, fit_prior=False, class_prior=None)

    mnb.fit(X_res_bow, y_res)

    

    # Predicting

    y_train_pred = mnb.predict_proba(X_res_bow)

    y_cv_pred = mnb.predict_proba(x_val_bow_unigram)

    

    # Loss metric storing

    train_loss.append(log_loss(y_res,y_train_pred))

    cv_loss.append(log_loss(y_val, y_cv_pred))



    

# Visualising and finding optimal parameter 

plt.plot(np.arange(1,10,1), train_loss, label='Train loss')

plt.plot(np.arange(1,10,1), cv_loss, label='CV loss')

plt.xticks( np.arange(1,10,1), (10e-5, 10e-4, 10e-3, 10e-2, 10e-1, 10e0, 10e1, 10e2, 10e3))

plt.legend()

plt.xlabel("alpha: hyperparameter")

plt.ylabel("log loss")

plt.title("ERROR PLOTS")

plt.grid()

plt.show()





## Training using Optimal hyperparemeter

# using optimum_k to find generalistion loss



optimum_alpha = alpha_range[np.argmin(cv_loss)] #optimum 'alpha'



# Naive Bayes training

print(f"Traing using optimal alpha:  {alpha_range[np.argmin(cv_loss)]}\n")

clf=MultinomialNB(alpha=optimum_alpha, fit_prior=False, class_prior=None)

clf.fit(X_res_bow, y_res)



y_pred = clf.predict(x_val_bow_unigram)

y_pred_proba = clf.predict_proba(x_val_bow_unigram)



# Result track

accuracy = accuracy_score(y_val,y_pred)

bal_accuracy = balanced_accuracy_score(y_val,y_pred)

logloss = log_loss(y_val,y_pred_proba)

print(f'\nGenearalisation log_loss: {logloss:.3f}')

print(f"\nGeneralisation Accuracy: {(round(accuracy,2))*100}%")

print(f"\nGeneralisation Balance accuracy: {(round(bal_accuracy,2))*100}%")

print(f'\nmisclassification percentage: {(1-accuracy)*100:.2f}%')





#ploting confusion matrix

sn.heatmap(confusion_matrix(y_pred,y_val),annot=True, fmt="d",linewidths=.5)

plt.title('Confusion Matrix')

plt.xlabel('Predicted values')

plt.ylabel('Actual values')

plt.show()

# Classification Report

print("\n\nclassification report:\n",classification_report(y_val,y_pred)) 
from sklearn.naive_bayes import MultinomialNB

from sklearn.metrics import log_loss,classification_report

import matplotlib.pyplot as plt





train_loss = []

cv_loss = []

alpha_range = [10e-5,10e-4,10e-3,10e-2,1,10,10e1,10e2,10e3]

for i in alpha_range:

    

    # Training

    mnb = MultinomialNB(alpha=i, fit_prior=False, class_prior=None)

    mnb.fit(x_train_tfidf_unigram, y_train)

    

    # Predicting

    y_train_pred = mnb.predict_proba(x_train_tfidf_unigram)

    y_cv_pred = mnb.predict_proba(x_val_tfidf_unigram)

    

    # Loss metric storing

    train_loss.append(log_loss(y_train,y_train_pred))

    cv_loss.append(log_loss(y_val, y_cv_pred))



    

# Visualising and finding optimal parameter 

plt.plot(np.arange(1,10,1), train_loss, label='Train loss')

plt.plot(np.arange(1,10,1), cv_loss, label='CV loss')

plt.xticks( np.arange(1,10,1), (10e-5, 10e-4, 10e-3, 10e-2, 10e-1, 10e0, 10e1, 10e2, 10e3))

plt.legend()

plt.xlabel("alpha: hyperparameter")

plt.ylabel("log loss")

plt.title("ERROR PLOTS")

plt.grid()

plt.show()



# Training using Optimal hyperparemeter

# using optimum_k to find generalistion loss



optimum_alpha = alpha_range[np.argmin(cv_loss)] #optimum 'alpha'



clf=MultinomialNB(alpha=optimum_alpha, fit_prior=False, class_prior=None)

clf.fit(x_train_tfidf_unigram,y_train)





y_pred = clf.predict(x_val_tfidf_unigram)

y_pred_proba = clf.predict_proba(x_val_tfidf_unigram)



# Result track

accuracy = accuracy_score(y_val,y_pred)

bal_accuracy = balanced_accuracy_score(y_val,y_pred)

logloss = log_loss(y_val,y_pred_proba)

print(f'\nGenearalisation log_loss: {logloss:.3f}')

print(f"\nGeneralisation Accuracy: {(round(accuracy,2))*100}%")

print(f"\nGeneralisation Balance accuracy: {(round(bal_accuracy,2))*100}%")

print(f'\nmisclassification percentage: {(1-accuracy)*100:.2f}%')



#ploting confusion matrix

sn.heatmap(confusion_matrix(y_pred,y_val),annot=True, fmt="d",linewidths=.5)

plt.title('Confusion Matrix')

plt.xlabel('Predicted values')

plt.ylabel('Actual values')

plt.show()

# Classification Report

print("\n\nclassification report:\n",classification_report(y_val,y_pred)) 

# imbalanced-learn API: https://imbalanced-learn.readthedocs.io/en/stable/generated/imblearn.over_sampling.SMOTE.html



# SMOTE

from imblearn.over_sampling import SMOTE



# transform the dataset

oversample = SMOTE(sampling_strategy='auto',random_state=None,k_neighbors=5,n_jobs=None)

X_res_tfidf, y_res = oversample.fit_resample(x_train_tfidf_unigram, y_train)



# summarize the new class distribution

counter = Counter(y_res)

print("After applying SMOTE: ",counter)
from sklearn.naive_bayes import MultinomialNB

from sklearn.metrics import log_loss,classification_report

import matplotlib.pyplot as plt





train_loss = []

cv_loss = []

alpha_range = [10e-5,10e-4,10e-3,10e-2,1,10,10e1,10e2,10e3]

for i in alpha_range:

    

    # Training

    mnb = MultinomialNB(alpha=i, fit_prior=False, class_prior=None)

    mnb.fit(X_res_tfidf, y_res)

    

    # Predicting

    y_train_pred = mnb.predict_proba(X_res_tfidf)

    y_cv_pred = mnb.predict_proba(x_val_tfidf_unigram)

    

    # Loss metric storing

    train_loss.append(log_loss(y_res,y_train_pred))

    cv_loss.append(log_loss(y_val, y_cv_pred))



    

# Visualising and finding optimal parameter 

plt.plot(np.arange(1,10,1), train_loss, label='Train loss')

plt.plot(np.arange(1,10,1), cv_loss, label='CV loss')

plt.xticks( np.arange(1,10,1), (10e-5, 10e-4, 10e-3, 10e-2, 10e-1, 10e0, 10e1, 10e2, 10e3))

plt.legend()

plt.xlabel("alpha: hyperparameter")

plt.ylabel("log loss")

plt.title("ERROR PLOTS")

plt.grid()

plt.show()







## Training using Optimal hyperparemeter

# using optimum_k to find generalistion loss

optimum_alpha = alpha_range[np.argmin(cv_loss)] #optimum 'alpha'



# Naive Bayes Training

print(f"Traing using optimal alpha:  {alpha_range[np.argmin(cv_loss)]}\n")

clf=MultinomialNB(alpha=optimum_alpha, fit_prior=False, class_prior=None)

clf.fit(X_res_tfidf,y_res)



y_pred = clf.predict(x_val_tfidf_unigram)

y_pred_proba = clf.predict_proba(x_val_tfidf_unigram)



# Result track

accuracy = accuracy_score(y_val,y_pred)

bal_accuracy = balanced_accuracy_score(y_val,y_pred)

logloss = log_loss(y_val,y_pred_proba)

print(f'\nGenearalisation log_loss: {logloss:.3f}')

print(f"\nGeneralisation Accuracy: {(round(accuracy,2))*100}%")

print(f"\nGeneralisation Balance accuracy: {(round(bal_accuracy,2))*100}%")

print(f'\nmisclassification percentage: {(1-accuracy)*100:.2f}%')



#ploting confusion matrix

sn.heatmap(confusion_matrix(y_pred,y_val),annot=True, fmt="d",linewidths=.5)

plt.title('Confusion Matrix')

plt.xlabel('Predicted values')

plt.ylabel('Actual values')

plt.show()

# Classification Report

print("\n\nclassification report:\n",classification_report(y_val,y_pred)) 

from sklearn.naive_bayes import GaussianNB

from sklearn.metrics import log_loss,classification_report

import matplotlib.pyplot as plt





train_loss = []

cv_loss = []

smooting_var_range = [1e-07,1e-06,1e-05,1e-04,1e-03,1e-02,1e-01,1,1e1,1e2,1e3]

for i in smooting_var_range:

    

    # Training

    mnb = GaussianNB(var_smoothing=i) # Input data is continious so multinomialNaiveBayes will not run. therefore use Gaussian Naive Bayes

    mnb.fit(train_w2v_sc, y_train)

    

    # Predicting

    y_train_pred = mnb.predict_proba(train_w2v_sc)

    y_cv_pred = mnb.predict_proba(val_w2v_sc)

    

    # Loss metric storing

    train_loss.append(log_loss(y_train,y_train_pred))

    cv_loss.append(log_loss(y_val, y_cv_pred))



    

# Visualising and finding optimal parameter 

plt.plot(np.arange(1,12,1), train_loss, label='Train loss')

plt.plot(np.arange(1,12,1), cv_loss, label='CV loss')

plt.xticks( np.arange(1,12,1), (smooting_var_range))

plt.legend()

plt.xlabel("alpha: hyperparameter")

plt.ylabel("log loss")

plt.title("ERROR PLOTS")

plt.grid()

plt.show()





#----------------------------------------------------------------------------------------------------------------------------------------

# Training using Optimal hyperparemeter

# using optimum_k to find generalistion loss



optimum_smooting_var = smooting_var_range[np.argmin(cv_loss)] #optimum 'alpha'



# Naive Bayes Training

print(f"Traing using optimal alpha:  {smooting_var_range[np.argmin(cv_loss)]}\n")

clf = GaussianNB(var_smoothing = optimum_smooting_var)

clf.fit(train_w2v_sc, y_train)



y_pred = clf.predict(val_w2v_sc)

y_pred_proba = clf.predict_proba(val_w2v_sc)



# Result track

accuracy = accuracy_score(y_val,y_pred)

bal_accuracy = balanced_accuracy_score(y_val,y_pred)

logloss = log_loss(y_val,y_pred_proba)

print(f'\nGenearalisation log_loss: {logloss:.3f}')

print(f"\nGeneralisation Accuracy: {(round(accuracy,2))*100}%")

print(f"\nGeneralisation Balance accuracy: {(round(bal_accuracy,2))*100}%")

print(f'\nmisclassification percentage: {(1-accuracy)*100:.2f}%')



#ploting confusion matrix

sn.heatmap(confusion_matrix(y_pred,y_val),annot=True, fmt="d",linewidths=.5)

plt.title('Confusion Matrix')

plt.xlabel('Predicted values')

plt.ylabel('Actual values')

plt.show()

# Classification Report

print("\n\nclassification report:\n",classification_report(y_val,y_pred)) 

# imbalanced-learn API: https://imbalanced-learn.readthedocs.io/en/stable/generated/imblearn.over_sampling.SMOTE.html



# SMOTE

from imblearn.over_sampling import SMOTE



# transform the dataset

oversample = SMOTE(sampling_strategy='auto',random_state=None,k_neighbors=5,n_jobs=None)

train_res_w2v, y_res = oversample.fit_resample(train_w2v_sc, y_train)



# summarize the new class distribution

counter = Counter(y_res)

print("After applying SMOTE: ",counter)
from sklearn.naive_bayes import GaussianNB

from sklearn.metrics import log_loss,classification_report

import matplotlib.pyplot as plt





train_loss = []

cv_loss = []

smooting_var_range = [1e-07,1e-06,1e-05,1e-04,1e-03,1e-02,1e-01,1,1e1,1e2,1e3]

for i in smooting_var_range:

    

    # Training

    mnb = GaussianNB(var_smoothing=i) # Input data is continious so multinomialNaiveBayes will not run. therefore use Gaussian Naive Bayes

    mnb.fit(train_res_w2v, y_res)

    

    # Predicting

    y_train_pred = mnb.predict_proba(train_res_w2v)

    y_cv_pred = mnb.predict_proba(val_w2v_sc)

    

    # Loss metric storing

    train_loss.append(log_loss(y_res,y_train_pred))

    cv_loss.append(log_loss(y_val, y_cv_pred))



    

# Visualising and finding optimal parameter 

plt.plot(np.arange(1,12,1), train_loss, label='Train loss')

plt.plot(np.arange(1,12,1), cv_loss, label='CV loss')

plt.xticks( np.arange(1,12,1), (smooting_var_range))

plt.legend()

plt.xlabel("alpha: hyperparameter")

plt.ylabel("log loss")

plt.title("ERROR PLOTS")

plt.grid()

plt.show()





#----------------------------------------------------------------------------------------------------------------------------------------

# Training using Optimal hyperparemeter

# using optimum_k to find generalistion loss



optimum_smooting_var = smooting_var_range[np.argmin(cv_loss)] #optimum 'alpha'



# Naive Bayes Training

print(f"Traing using optimal alpha:  {smooting_var_range[np.argmin(cv_loss)]}\n")

clf = GaussianNB(var_smoothing = optimum_smooting_var)

clf.fit(train_res_w2v, y_res)



y_pred = clf.predict(val_w2v_sc)

y_pred_proba = clf.predict_proba(val_w2v_sc)



# Result track

accuracy = accuracy_score(y_val,y_pred)

bal_accuracy = balanced_accuracy_score(y_val,y_pred)

logloss = log_loss(y_val,y_pred_proba)

print(f'\nGenearalisation log_loss: {logloss:.3f}')

print(f"\nGeneralisation Accuracy: {(round(accuracy,2))*100}%")

print(f"\nGeneralisation Balance accuracy: {(round(bal_accuracy,2))*100}%")

print(f'\nmisclassification percentage: {(1-accuracy)*100:.2f}%')



#ploting confusion matrix

sn.heatmap(confusion_matrix(y_pred,y_val),annot=True, fmt="d",linewidths=.5)

plt.title('Confusion Matrix')

plt.xlabel('Predicted values')

plt.ylabel('Actual values')

plt.show()

# Classification Report

print("\n\nclassification report:\n",classification_report(y_val,y_pred)) 

from sklearn.naive_bayes import GaussianNB

from sklearn.metrics import log_loss,classification_report

import matplotlib.pyplot as plt





train_loss = []

cv_loss = []

smooting_var_range = [1e-07,1e-06,1e-05,1e-04,1e-03,1e-02,1e-01,1,1e1,1e2,1e3]

for i in smooting_var_range:

    

    # Training

    mnb = GaussianNB(var_smoothing=i) # Input data is continious so multinomialNaiveBayes will not run. therefore use Gaussian Naive Bayes

    mnb.fit(train_w2v_pretrained_sc, y_train)

    

    # Predicting

    y_train_pred = mnb.predict_proba(train_w2v_pretrained_sc)

    y_cv_pred = mnb.predict_proba(val_w2v_pretrained_sc)

    

    # Loss metric storing

    train_loss.append(log_loss(y_train,y_train_pred))

    cv_loss.append(log_loss(y_val, y_cv_pred))



    

# Visualising and finding optimal parameter 

plt.plot(np.arange(1,12,1), train_loss, label='Train loss')

plt.plot(np.arange(1,12,1), cv_loss, label='CV loss')

plt.xticks( np.arange(1,12,1), [1e-07,1e-06,1e-05,1e-04,1e-03,1e-02,1e-01,1,1e1,1e2,1e3])

plt.legend()

plt.xlabel("alpha: hyperparameter")

plt.ylabel("log loss")

plt.title("ERROR PLOTS")

plt.grid()

plt.show()





#----------------------------------------------------------------------------------------------------------------------------------------

# Training using Optimal hyperparemeter

# using optimum_k to find generalistion loss



optimum_smooting_var = smooting_var_range[np.argmin(cv_loss)] #optimum 'alpha'



# Naive Bayes Training

print(f"Traing using optimal alpha:  {smooting_var_range[np.argmin(cv_loss)]}\n")

clf = GaussianNB(var_smoothing = optimum_smooting_var)

clf.fit(train_w2v_pretrained_sc, y_train)



y_pred = clf.predict(val_w2v_pretrained_sc)

y_pred_proba = clf.predict_proba(val_w2v_pretrained_sc)



# Result track

accuracy = accuracy_score(y_val,y_pred)

bal_accuracy = balanced_accuracy_score(y_val,y_pred)

logloss = log_loss(y_val,y_pred_proba)

print(f'\nGenearalisation log_loss: {logloss:.3f}')

print(f"\nGeneralisation Accuracy: {(round(accuracy,2))*100}%")

print(f"\nGeneralisation Balance accuracy: {(round(bal_accuracy,2))*100}%")

print(f'\nmisclassification percentage: {(1-accuracy)*100:.2f}%')



#ploting confusion matrix

sn.heatmap(confusion_matrix(y_pred,y_val),annot=True, fmt="d",linewidths=.5)

plt.title('Confusion Matrix')

plt.xlabel('Predicted values')

plt.ylabel('Actual values')

plt.show()

# Classification Report

print("\n\nclassification report:\n",classification_report(y_val,y_pred)) 

# imbalanced-learn API: https://imbalanced-learn.readthedocs.io/en/stable/generated/imblearn.over_sampling.SMOTE.html



# SMOTE

from imblearn.over_sampling import SMOTE



# transform the dataset

oversample = SMOTE(sampling_strategy='auto',random_state=11,k_neighbors=5,n_jobs=None)

train_res_w2v_pretrained, y_res = oversample.fit_resample(train_w2v_pretrained_sc, y_train)



# summarize the new class distribution

counter = Counter(y_res)

print("After applying SMOTE: ",counter)
from sklearn.naive_bayes import GaussianNB

from sklearn.metrics import log_loss,classification_report

import matplotlib.pyplot as plt





train_loss = []

cv_loss = []

smooting_var_range = [1e-07,1e-06,1e-05,1e-04,1e-03,1e-02,1e-01,1,1e1,1e2,1e3]

for i in smooting_var_range:

    

    # Training

    mnb = GaussianNB(var_smoothing=i) # Input data is continious so multinomialNaiveBayes will not run. therefore use Gaussian Naive Bayes

    mnb.fit(train_res_w2v_pretrained, y_res)

    

    # Predicting

    y_train_pred = mnb.predict(train_res_w2v_pretrained)

    y_cv_pred = mnb.predict(val_w2v_pretrained_sc)

    

    # Loss metric storing

    train_loss.append(accuracy_score(y_res,y_train_pred))

    cv_loss.append(accuracy_score(y_val, y_cv_pred))



    

# Visualising and finding optimal parameter 

plt.plot(np.arange(1,12,1), train_loss, label='Train loss')

plt.plot(np.arange(1,12,1), cv_loss, label='CV loss')

plt.xticks( np.arange(1,12,1), (smooting_var_range))

plt.legend()

plt.xlabel("alpha: hyperparameter")

plt.ylabel("log loss")

plt.title("ERROR PLOTS")

plt.grid()

plt.show()





#----------------------------------------------------------------------------------------------------------------------------------------

# Training using Optimal hyperparemeter

# using optimum_k to find generalistion loss



optimum_smooting_var = 10#optimum 'alpha'



# Naive Bayes Training

print(f"Traing using optimal alpha:  {smooting_var_range[np.argmin(cv_loss)]}\n")

clf = GaussianNB(var_smoothing = optimum_smooting_var)

clf.fit(train_res_w2v_pretrained, y_res)



y_pred = clf.predict(val_w2v_pretrained_sc)

y_pred_proba = clf.predict_proba(val_w2v_pretrained_sc)



# Result track

accuracy = accuracy_score(y_val,y_pred)

bal_accuracy = balanced_accuracy_score(y_val,y_pred)

logloss = log_loss(y_val,y_pred_proba)

print(f'\nGenearalisation log_loss: {logloss:.3f}')

print(f"\nGeneralisation Accuracy: {(round(accuracy,2))*100}%")

print(f"\nGeneralisation Balance accuracy: {(round(bal_accuracy,2))*100}%")

print(f'\nmisclassification percentage: {(1-accuracy)*100:.2f}%')



#ploting confusion matrix

sn.heatmap(confusion_matrix(y_pred,y_val),annot=True, fmt="d",linewidths=.5)

plt.title('Confusion Matrix')

plt.xlabel('Predicted values')

plt.ylabel('Actual values')

plt.show()

# Classification Report

print("\n\nclassification report:\n",classification_report(y_val,y_pred)) 





print( "wrong result")
from prettytable import PrettyTable

x = PrettyTable()

x.field_names = ["S.No","Model(text featurization)", "generalization log loss", "generalization AUC(%age) ", "generalization balance accuracy",]



x.add_row([1, "NB(BOW)", 0.258, 91, 85])

x.add_row([2, "NB(BOW) + SMOTE", 0.318, 91, 82])



x.add_row([3, "NB(TFIDF)", 0.235, 91, 69])

x.add_row([4, "NB(TFIDF) + SMOTE", 0.303, 88, 83])



x.add_row([5, "NB(W2V_train) ", 0.332, 85, 66])

x.add_row([6, "NB(W2V_train)+ SMOTE", 0.693, 45, 67])



x.add_row([7, "NB(W2V_pretrain)", 0.296, 88, 56.9])

x.add_row([8, "NB(W2V_pretrain)+ SMOTE", 3.47, 16, 52]) # Something is wrong



print(x)