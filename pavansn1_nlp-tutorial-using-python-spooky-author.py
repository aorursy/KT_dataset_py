import numpy as np

import pandas as pd

from sklearn import preprocessing

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from sklearn.model_selection import train_test_split, KFold

from nltk.corpus import stopwords

from nltk.stem.snowball import SnowballStemmer

import matplotlib

from matplotlib import pyplot as plt

# import seaborn as sns

%matplotlib inline

%config InlineBackend.figure_format = 'retina'
data = pd.read_csv("../input/train.csv")
data.head(10)
data.shape
# extracting the number of examples of each class

EAP_len = data[data['author'] == 'EAP'].shape[0]

HPL_len = data[data['author'] == 'HPL'].shape[0]

MWS_len = data[data['author'] == 'MWS'].shape[0]

print(EAP_len,HPL_len, MWS_len, "    total = ", EAP_len+HPL_len+MWS_len)
# bar plot of the 3 classes

plt.bar(10,EAP_len,3, label="EAP")

plt.bar(15,HPL_len,3, label="HPL")

plt.bar(20,MWS_len,3, label="MWS")

plt.legend()

plt.ylabel('Number of examples')

plt.title('Propoertion of examples')

plt.show()
def remove_punctuation(text):

    '''a function for removing punctuation'''

    import string

    # replacing the punctuations with no space, 

    # which in effect deletes the punctuation marks 

    translator = str.maketrans('', '', string.punctuation)

    # return the text stripped of punctuation marks

    return text.translate(translator)
data['text'] = data['text'].apply(remove_punctuation)

data.head(10)
# extracting the stopwords from nltk library

sw = stopwords.words('english')

# displaying the stopwords

np.array(sw)
print("Number of stopwords: ", len(sw))
def stopwords(text):

    '''a function for removing the stopword'''

    # removing the stop words and lowercasing the selected words

    text = [word.lower() for word in text.split() if word.lower() not in sw]

    # joining the list of words with space separator

    return " ".join(text)
data['text'] = data['text'].apply(stopwords)

data.head(10)
# create a count vectorizer object

count_vectorizer = CountVectorizer()

# fit the count vectorizer using the text data

count_vectorizer.fit(data['text'])

# collect the vocabulary items used in the vectorizer

dictionary = count_vectorizer.vocabulary_.items()  
# lists to store the vocab and counts

vocab = []

count = []

# iterate through each vocab and count append the value to designated lists

for key, value in dictionary:

    vocab.append(key)

    count.append(value)

# store the count in panadas dataframe with vocab as index

vocab_bef_stem = pd.Series(count, index=vocab)

# sort the dataframe

vocab_bef_stem = vocab_bef_stem.sort_values(ascending=False)
top_vacab = vocab_bef_stem.head(20)

top_vacab.plot(kind = 'barh', figsize=(5,10), xlim= (25230, 25260))
# create an object of stemming function

stemmer = SnowballStemmer("english")



def stemming(text):    

    '''a function which stems each word in the given text'''

    text = [stemmer.stem(word) for word in text.split()]

    return " ".join(text) 
data['text'] = data['text'].apply(stemming)

data.head(10)
# create the object of tfid vectorizer

tfid_vectorizer = TfidfVectorizer("english")

# fit the vectorizer using the text data

tfid_vectorizer.fit(data['text'])

# collect the vocabulary items used in the vectorizer

dictionary = tfid_vectorizer.vocabulary_.items()  
# lists to store the vocab and counts

vocab = []

count = []

# iterate through each vocab and count append the value to designated lists

for key, value in dictionary:

    vocab.append(key)

    count.append(value)

# store the count in panadas dataframe with vocab as index

vocab_after_stem = pd.Series(count, index=vocab)

# sort the dataframe

vocab_after_stem = vocab_after_stem.sort_values(ascending=False)

# plot of the top vocab

top_vacab = vocab_after_stem.head(20)

top_vacab.plot(kind = 'barh', figsize=(5,10), xlim= (15120, 15145))
def length(text):    

    '''a function which returns the length of text'''

    return len(text)
data['length'] = data['text'].apply(length)

data.head(10)
EAP_data = data[data['author'] == 'EAP']

HPL_data = data[data['author'] == 'HPL']

MWS_data = data[data['author'] == 'MWS']
matplotlib.rcParams['figure.figsize'] = (12.0, 6.0)

bins = 500

plt.hist(EAP_data['length'], alpha = 0.6, bins=bins, label='EAP')

plt.hist(HPL_data['length'], alpha = 0.8, bins=bins, label='HPL')

plt.hist(MWS_data['length'], alpha = 0.4, bins=bins, label='MWS')

plt.xlabel('length')

plt.ylabel('numbers')

plt.legend(loc='upper right')

plt.xlim(0,300)

plt.grid()

plt.show()
# create the object of tfid vectorizer

EAP_tfid_vectorizer = TfidfVectorizer("english")

# fit the vectorizer using the text data

EAP_tfid_vectorizer.fit(EAP_data['text'])

# collect the vocabulary items used in the vectorizer

EAP_dictionary = EAP_tfid_vectorizer.vocabulary_.items()



# lists to store the vocab and counts

vocab = []

count = []

# iterate through each vocab and count append the value to designated lists

for key, value in EAP_dictionary:

    vocab.append(key)

    count.append(value)

# store the count in panadas dataframe with vocab as index

EAP_vocab = pd.Series(count, index=vocab)

# sort the dataframe

EAP_vocab = EAP_vocab.sort_values(ascending=False)

# plot of the top vocab

top_vacab = EAP_vocab.head(20)

top_vacab.plot(kind = 'barh', figsize=(5,10), xlim= (9700, 9740))
# create the object of tfid vectorizer

HPL_tfid_vectorizer = TfidfVectorizer("english")

# fit the vectorizer using the text data

HPL_tfid_vectorizer.fit(HPL_data['text'])

# collect the vocabulary items used in the vectorizer

HPL_dictionary = HPL_tfid_vectorizer.vocabulary_.items()

# lists to store the vocab and counts

vocab = []

count = []

# iterate through each vocab and count append the value to designated lists

for key, value in HPL_dictionary:

    vocab.append(key)

    count.append(value)

# store the count in panadas dataframe with vocab as index    

HPL_vocab = pd.Series(count, index=vocab)

# sort the dataframe

HPL_vocab = HPL_vocab.sort_values(ascending=False)

# plot of the top vocab

top_vacab = HPL_vocab.head(20)

top_vacab.plot(kind = 'barh', figsize=(5,10), xlim= (9300, 9330))
# create the object of tfid vectorizer

MWS_tfid_vectorizer = TfidfVectorizer("english")

# fit the vectorizer using the text data

MWS_tfid_vectorizer.fit(MWS_data['text'])

# collect the vocabulary items used in the vectorizer

MWS_dictionary = MWS_tfid_vectorizer.vocabulary_.items()

# lists to store the vocab and counts

vocab = []

count = []

# iterate through each vocab and count append the value to designated list

for key, value in MWS_dictionary:

    vocab.append(key)

    count.append(value)

# store the count in panadas dataframe and vocab as index    

MWS_vocab = pd.Series(count, index=vocab)

# sort the dataframe

MWS_vocab = MWS_vocab.sort_values(ascending=False)

# plot of the top vocab

top_vacab = MWS_vocab.head(20)

top_vacab.plot(kind = 'barh', figsize=(5,10), xlim= (7010, 7040))
# extract the tfid representation matrix of the text data

tfid_matrix = tfid_vectorizer.transform(data['text'])

# collect the tfid matrix in numpy array

array = tfid_matrix.todense()
# store the tf-idf array into pandas dataframe

df = pd.DataFrame(array)

df.head(10)
df['output'] = data['author']

df['id'] = data['id']

df.head(10)
features = df.columns.tolist()

output = 'output'

# removing the output and the id from features

features.remove(output)

features.remove('id')
from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB

from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import accuracy_score, log_loss

from sklearn.model_selection import GridSearchCV
alpha_list1 = np.linspace(0.006, 0.1, 20)

alpha_list1 = np.around(alpha_list1, decimals=4)

alpha_list1
# parameter grid

parameter_grid = [{"alpha":alpha_list1}]
# classifier object

classifier1 = MultinomialNB()

# gridsearch object using 4 fold cross validation and neg_log_loss as scoring paramter

gridsearch1 = GridSearchCV(classifier1,parameter_grid, scoring = 'neg_log_loss', cv = 4)

# fit the gridsearch

gridsearch1.fit(df[features], df[output])
results1 = pd.DataFrame()

# collect alpha list

results1['alpha'] = gridsearch1.cv_results_['param_alpha'].data

# collect test scores

results1['neglogloss'] = gridsearch1.cv_results_['mean_test_score'].data
matplotlib.rcParams['figure.figsize'] = (12.0, 6.0)

plt.plot(results1['alpha'], -results1['neglogloss'])

plt.xlabel('alpha')

plt.ylabel('logloss')

plt.grid()
print("Best parameter: ",gridsearch1.best_params_)
print("Best score: ",gridsearch1.best_score_) 
alpha_list2 = np.linspace(0.006, 0.1, 20)

alpha_list2 = np.around(alpha_list2, decimals=4)

alpha_list2
parameter_grid = [{"alpha":alpha_list2}]
# classifier object

classifier2 = MultinomialNB()

# gridsearch object using 4 fold cross validation and neg_log_loss as scoring paramter

gridsearch2 = GridSearchCV(classifier2,parameter_grid, scoring = 'neg_log_loss', cv = 4)

# fit the gridsearch

gridsearch2.fit(df[features], df[output])
results2 = pd.DataFrame()

# collect alpha list

results2['alpha'] = gridsearch2.cv_results_['param_alpha'].data

# collect test scores

results2['neglogloss'] = gridsearch2.cv_results_['mean_test_score'].data
matplotlib.rcParams['figure.figsize'] = (12.0, 6.0)

plt.plot(results2['alpha'], -results2['neglogloss'])

plt.xlabel('alpha')

plt.ylabel('logloss')

plt.grid()
print("Best parameter: ",gridsearch2.best_params_)
print("Best score: ",gridsearch2.best_score_)