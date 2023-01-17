# Import all required modules for the analysis(make sure that you installed all these modules prior to importing)
import pandas as pd
import numpy as np
import nltk
import matplotlib.pyplot as plt
import math
from IPython.display import display,HTML
from patsy import dmatrices
import seaborn as sns; sns.set()

import warnings
warnings.filterwarnings('ignore')
%pylab inline
# Reading the train data
train_df = pd.read_csv('../input/Train_rev1.csv')
# Checking the data types in the train data
print (train_df.info())

# Let's look at the unique values present in the data frame to have a general understanding of the data
names = train_df.columns.values
uniq_vals = {}
for name in names:
    uniq_vals[name] = train_df.loc[:,name].unique()
    print("Count of %s : %d" %(name,uniq_vals[name].shape[0]))
# Distribution of salaries based on the train data
pylab.rcParams['figure.figsize'] = (20,10)
plt.hist(train_df['SalaryNormalized'], bins='auto')
plt.xlabel('Salaries')
plt.ylabel('Number of postings')
plt.title('Histogram of Salaries')
# Randomly selecting 2500 rows to train the classifier
import random
random.seed(1)
indices = list(train_df.index.values)
random_2500 = random.sample(indices,2500)

# Subsetting the train data based on the random indices
train_df1 = train_df.loc[random_2500].reset_index()
pylab.rcParams['figure.figsize'] = (20,10)
plt.hist(train_df1['SalaryNormalized'], bins='auto')
plt.xlabel('Salaries')
plt.ylabel('Number of postings')
plt.title('Histogram of Salaries')
import re
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize, sent_tokenize 
stop_words = set(stopwords.words('english')) 
from string import punctuation
import collections
# To obtain the full width of a cell in a dataframe
pd.set_option('display.max_colwidth', -1)
desc = train_df1.loc[1,'FullDescription']

# Creating a list of words from all the job descriptions in train_df1 data
all_desc = []
for i in range(0,train_df1.shape[0]):
    desc = train_df1.loc[i,'FullDescription']
    desc1 = desc.lower()
    # Removing numbers, *** and www links from the data
    desc2 = re.sub('[0-9]+\S+|\s\d+\s|\w+[0-9]+|\w+[\*]+.*|\s[\*]+\s|www\.[^\s]+','',desc1)
    # Removing punctuation
    for p in punctuation:
        desc2 = desc2.replace(p,'')
    all_desc.append(desc2)

# Creating word tokens for all the descriptions
final_list = []
for desc in all_desc:
    word_list = word_tokenize(desc)
    final_list.extend(word_list)
# 3. Tagging parts of speech
pos_tagged = nltk.pos_tag(final_list)

# 4. Identifying the most common parts of speech
tag_fd = nltk.FreqDist(tag for (word, tag) in pos_tagged)
tag_fd.most_common()[:5]
# Excluding stopwords from the analysis
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english')) 

list_wo_stopwords = []
for w in final_list:
    if w not in stop_words:
        list_wo_stopwords.append(w)
        
# 3. Tagging parts of speech
pos_tagged_wo_sw = nltk.pos_tag(list_wo_stopwords)

# 4. Identifying the most common parts of speech
tag_fd_wo_sw = nltk.FreqDist(tag for (word, tag) in pos_tagged_wo_sw)
tag_fd_wo_sw.most_common()[:5]
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()

# Lemmatization without specifying parts of speech
list_lemmatized = []
for word in list_wo_stopwords:
    list_lemmatized.append(lemmatizer.lemmatize(word))

word_freq_lem = dict(collections.Counter(list_lemmatized))
keys = list(word_freq_lem.keys())
values = list(word_freq_lem.values())
df_lem = pd.DataFrame({'words':keys,'freq':values})
display(df_lem.sort_values(by = 'freq',ascending = False)[:10])

from wordcloud import WordCloud
from collections import Counter
word_could_dict=Counter(word_freq_lem)
wordcloud = WordCloud(width = 1000, height = 500).generate_from_frequencies(word_could_dict)

plt.figure(figsize=(15,8))
plt.imshow(wordcloud)

p_75 = np.percentile(train_df1['SalaryNormalized'], 75)
train_df1['target'] = train_df1['SalaryNormalized'].apply(lambda x: 1 if x>=p_75 else 0)
costly_cities = ['London','Brighton','Edinburgh','Bristol','Southampton','Portsmouth','Exeter','Cardiff','Manchester',
                 'Birmingham','Leeds','Aberdeen','Glasgow','Newcastle','Sheffield','Liverpool']
costly_cities_lower = [x.lower() for x in costly_cities]

# More robust if lower() is applied
train_df1['location_flag'] = train_df1['LocationNormalized'].apply(lambda x: 1 if x in costly_cities else 0)
# Dropping job description column from the dataset
train_x = train_df1.drop(['FullDescription','index','Id','LocationRaw','Title','Company','LocationNormalized','SalaryRaw','SalaryNormalized',
                    'target'],axis=1)

train_x1 = pd.get_dummies(train_x,drop_first=True)
X_n = np.array(train_x1)
y_n = np.array(train_df1['target'])

from sklearn.model_selection import train_test_split
X_train_num, X_val_num, y_train_num, y_val_num = train_test_split(X_n, y_n, test_size=0.3, random_state=1)

# Bernoulli
from sklearn.naive_bayes import BernoulliNB
clf = BernoulliNB()
clf.fit(X_train_num, y_train_num)

from sklearn import metrics
prediction_train = clf.predict(X_val_num)
mat_n = metrics.confusion_matrix(y_val_num, prediction_train)
mat_n
print (metrics.accuracy_score(y_val_num, prediction_train))
# Baseline accuracy
1-(sum(y_val_num)/len(y_val_num))
# sum(prediction_train)
def models(l):
    # Counting the occurence of each word in the corpus
    from sklearn.feature_extraction.text import CountVectorizer
    count_vect = CountVectorizer()
    X_train_counts = count_vect.fit_transform(l)
    count_vect.get_feature_names()
    X_matrix= X_train_counts.todense()

    y = np.array(train_df1['target'])

    # Creating the train and test split
    from sklearn.model_selection import train_test_split
    X_train_m, X_val_m, y_train_m, y_val_m = train_test_split(X_train_counts, y, test_size=0.3, random_state=1)

    #Multinomial

    from sklearn.naive_bayes import MultinomialNB
    clf = MultinomialNB().fit(X_train_m, y_train_m)
    labels_m = clf.predict(X_val_m)

    from sklearn.metrics import confusion_matrix
    from sklearn.metrics import accuracy_score
    mat_m = confusion_matrix(y_val_m, labels_m)

    # Bernoulli
    # Changing the data to binary to input BernoulliNB
    x_train_b1 = X_train_counts.todense()
    X_train_counts_ber = np.where(x_train_b1 >=1 ,1,0)

    # Creating the train and test split for bernoulli
    from sklearn.model_selection import train_test_split
    X_train_b, X_val_b, y_train_b, y_val_b = train_test_split(X_train_counts_ber, y, test_size=0.3, random_state=1)

    from sklearn.naive_bayes import BernoulliNB
    clf = BernoulliNB().fit(X_train_b, y_train_b)
    labels_b = clf.predict(X_val_b)

    from sklearn.metrics import confusion_matrix
    from sklearn.metrics import accuracy_score
    mat_b = confusion_matrix(y_val_b, labels_b)
    print ('Confusion matrix:',mat_b)
    print ('Accuracy using BernoulliNB:',accuracy_score(y_val_b, labels_b))


    print ('Confusion matrix:',mat_m)
    print ('Accuracy using MultinomialNB:',accuracy_score(y_val_m, labels_m))
models(all_desc)
# Removing stopwords
def remove_stopwords(s):
    big_regex = re.compile(r'\b%s\b' % r'\b|\b'.join(map(re.escape, stop_words)))
    return big_regex.sub('',s)

all_desc_wo_sw = [remove_stopwords(s) for s in all_desc]
models(all_desc_wo_sw)
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()

from nltk.corpus import wordnet

def get_wordnet_pos(treebank_tag):

    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return None 

# Lemmatizing the data
all_desc_lemm = []
for i in range(0,len(all_desc_wo_sw)):
    desc = all_desc_wo_sw[i]
    desc2 = re.sub('[0-9]+\S+|\s\d+\s|\w+[0-9]+|\w+[\*]+.*|\s[\*]+\s|www\.[^\s]+','',desc)
    for p in punctuation:
        desc2 = desc2.replace(p,'')
    tagged = nltk.pos_tag(word_tokenize(desc2))
    list_lemmatized = []
    for word, tag in tagged:
        wntag = get_wordnet_pos(tag)
        if wntag is None:# not supply tag in case of None
            list_lemmatized.append(lemmatizer.lemmatize(word)) 
        else:
            list_lemmatized.append(lemmatizer.lemmatize(word, pos=wntag))
    k = ' '.join(list_lemmatized)   
    all_desc_lemm.append(k)

models(all_desc_lemm)