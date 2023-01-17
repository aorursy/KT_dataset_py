# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
texto_input_classify = '/kaggle/input/coronavirus-covid19-tweets/'

texto_input_created = '/kaggle/input/tweets-categories/'
df_to_categorize1 = pd.read_csv(texto_input_classify+'2020-03-00 Coronavirus Tweets (pre 2020-03-12).CSV')
df_to_categorize1 = df_to_categorize1.loc[(df_to_categorize1['country_code'].isin(['US','IN','GB','ZA','CA','NG','AU','PH','CN','PK'])) & (df_to_categorize1['lang'] == 'en'),:]

len(df_to_categorize1)
df_to_categorize1['tweet_id'] = df_to_categorize1.apply(lambda x: str(x['status_id'])+'_'+ str(x['user_id']), axis = 1)

df_to_categorize1 = df_to_categorize1.drop(['status_id', 'user_id', 'screen_name','source', 'reply_to_status_id', 'reply_to_user_id',

       'reply_to_screen_name', 'is_quote', 'is_retweet',

       'favourites_count', 'retweet_count','place_full_name', 'place_type', 'followers_count',

       'friends_count', 'account_lang', 'account_created_at', 'verified',

       'lang'], axis=1).reset_index(drop=True)

df_to_categorize1[:5]
# df_to_categorize2 = pd.read_csv(texto_input_classify+'2020-03-19 Coronavirus Tweets.CSV')

# df_to_categorize2 = df_to_categorize2.append(pd.read_csv(texto_input_classify+'2020-03-20 Coronavirus Tweets.CSV'), ignore_index= True)

# df_to_categorize2 = df_to_categorize2.append(pd.read_csv(texto_input_classify+'2020-03-21 Coronavirus Tweets.CSV'), ignore_index= True)

# df_to_categorize2 = df_to_categorize2.append(pd.read_csv(texto_input_classify+'2020-03-22 Coronavirus Tweets.CSV'), ignore_index= True)

# df_to_categorize2 = df_to_categorize2.append(pd.read_csv(texto_input_classify+'2020-03-23 Coronavirus Tweets.CSV'), ignore_index= True)

# len(df_to_categorize2)
# df_to_categorize2 = df_to_categorize2.loc[(df_to_categorize2['country_code'].isin(['US','IN','GB','ZA','CA','NG','AU','PH','CN','PK'])) & (df_to_categorize2['lang'] == 'en'),:]

# len(df_to_categorize2)
# df_to_categorize2['tweet_id'] = df_to_categorize2.apply(lambda x: str(x['status_id'])+'_'+ str(x['user_id']), axis = 1)

# df_to_categorize2 = df_to_categorize2.drop(['status_id', 'user_id', 'screen_name','source', 'reply_to_status_id', 'reply_to_user_id',

#        'reply_to_screen_name', 'is_quote', 'is_retweet',

#        'favourites_count', 'retweet_count','place_full_name', 'place_type', 'followers_count',

#        'friends_count', 'account_lang', 'account_created_at', 'verified',

#        'lang'], axis=1).reset_index(drop=True)

# df_to_categorize2[:5]
df_to_categorize3 = pd.read_csv(texto_input_classify+'2020-03-24 Coronavirus Tweets.CSV')

df_to_categorize3 = df_to_categorize3.append(pd.read_csv(texto_input_classify+'2020-03-25 Coronavirus Tweets.CSV'), ignore_index= True)

df_to_categorize3 = df_to_categorize3.append(pd.read_csv(texto_input_classify+'2020-03-26 Coronavirus Tweets.CSV'), ignore_index= True)

df_to_categorize3 = df_to_categorize3.append(pd.read_csv(texto_input_classify+'2020-03-27 Coronavirus Tweets.CSV'), ignore_index= True)

df_to_categorize3 = df_to_categorize3.append(pd.read_csv(texto_input_classify+'2020-03-28 Coronavirus Tweets.CSV'), ignore_index= True)

len(df_to_categorize3)
df_to_categorize3 = df_to_categorize3.loc[(df_to_categorize3['country_code'].isin(['US','IN','GB','ZA','CA','NG','AU','PH','CN','PK'])) & (df_to_categorize3['lang'] == 'en'),:]

len(df_to_categorize3)
df_to_categorize3['tweet_id'] = df_to_categorize3.apply(lambda x: str(x['status_id'])+'_'+ str(x['user_id']), axis = 1)

df_to_categorize3 = df_to_categorize3.drop(['status_id', 'user_id', 'screen_name','source', 'reply_to_status_id', 'reply_to_user_id',

       'reply_to_screen_name', 'is_quote', 'is_retweet',

       'favourites_count', 'retweet_count','place_full_name', 'place_type', 'followers_count',

       'friends_count', 'account_lang', 'account_created_at', 'verified',

       'lang'], axis=1).reset_index(drop=True)

df_to_categorize3[:5]
df_categorized = pd.read_excel(texto_input_created+'Tweets.xlsx')

df_categorized[:5]
df_us = df_to_categorize1.loc[df_to_categorize1['country_code']=='US',:]

print(len(df_us))

df_us = df_us.append(df_to_categorize3.loc[df_to_categorize3['country_code']=='US',:], ignore_index = True)

print(len(df_us))

df_us = df_us.sample(n=400)

df_us[:5]
import nltk

#nltk.download('punkt')

from nltk import wordpunct_tokenize

#nltk.download('stopwords')

from nltk.corpus import stopwords
stopwords_english = set(stopwords.words('english'))
#!pip install -U spacy
import spacy

nlp = spacy.load("en_core_web_sm")
def first_part_process(sentence):

    sentence = sentence.lower()

    

    # stopword and remove puntuationcs

    stopword_applied = [word for word in nltk.word_tokenize(sentence) if not word in stopwords_english and word.isalnum()] 

    stopword_applied = " ".join(stopword_applied) # I change it back to full sentence



    # lemma and pos

    spacy_applied = nlp(stopword_applied)

    spacy_applied = [word.lemma_ for word in spacy_applied if word.pos_ in ['ADJ','VERB']] # rember fix emojis



    return " ".join(spacy_applied)



from sklearn.feature_extraction.text import TfidfTransformer

from sklearn.feature_extraction.text import CountVectorizer



def countWords(tweets):

    stop_words_included = frozenset(['14th','1st'])

    cv = CountVectorizer(analyzer="word",stop_words=stop_words_included)

    word_count_vector=cv.fit_transform(tweets)

    return word_count_vector, cv



def define_idfs(word_count_vector):

    tfidf_transformer=TfidfTransformer(smooth_idf=True,use_idf=True)

    tfidf_transformer.fit(word_count_vector)

    return tfidf_transformer



df_categorized['text_processed'] = df_categorized.apply(lambda x: first_part_process(x['text']), axis=1)

print(len(df_categorized))

df_categorized[:5]
df_us['text_processed'] = df_us.apply(lambda x: first_part_process(x['text']), axis=1)

print(len(df_us))

df_us[:5]
# count frequency of terms

word_count_vector, tf_v = countWords(df_categorized['text_processed'])

word_count_vector.shape
df_tf_v1 = pd.DataFrame(data=word_count_vector.toarray(),index=np.arange(1, 81),columns=tf_v.get_feature_names())

df_tf_v1['cat'] = df_categorized['cat']

print(len(df_tf_v1))

df_tf_v1[:1]
feature_names = tf_v.get_feature_names()

dic_terms = {} 

dic_terms_pos = {} 

dic_terms_neg = {} 



for f in feature_names:

    dic_terms[f] = 0

    dic_terms_pos[f] = 0

    dic_terms_neg[f] = 0



def count_tf_terms(row):

    for key in feature_names:

        dic_terms[key] += row[key]

    return 0



def count_tf_terms_pos(row):

    if row['cat'] == 'positive ':

        for key in feature_names:

            dic_terms_pos[key] += row[key]

    return 0



def count_tf_terms_neg(row):

    if row['cat'] == 'negative':

        for key in feature_names:

            dic_terms_neg[key] += row[key]

    return 0

    

b = df_tf_v1.apply(lambda x: count_tf_terms(x), axis=1)

b = df_tf_v1.apply(lambda x: count_tf_terms_pos(x), axis=1)

b = df_tf_v1.apply(lambda x: count_tf_terms_neg(x), axis=1)

df_terms_counts= pd.DataFrame(dic_terms.items(), columns=['term', 'tf'])

df_terms_counts['positive'] = df_terms_counts['term'].map(dic_terms_pos) 

df_terms_counts['negative'] = df_terms_counts['term'].map(dic_terms_neg) 

df_terms_counts[:10]
# top ten general

df_terms_counts.nlargest(10, 'tf')[:10]
# top ten negative

df_terms_counts.nlargest(10, 'negative')[:10]
# top ten negative

df_terms_counts.nlargest(10, 'positive')[:10]
df_terms_counts.nlargest(10, 'positive').loc[:,['term','positive']][:10].positive
df_terms_counts.positive.max()
import matplotlib.pyplot as plt



data_upper = df_terms_counts.positive.max() + 5

fig, ax = plt.subplots(figsize=(15,7))

ax.set_ylabel('Tf', fontsize=20)



df_terms_counts.nlargest(10, 'positive').loc[:,['term','positive']].positive.plot(kind = 'bar', 

                                           ylim = [0, data_upper], 

                                           rot = 0, fontsize = 16, ax=ax)

ax.set_xlabel('Terms', fontsize=20)

ax.set_title('Positive Term Frequency - 80 Tweets', fontsize=24)

ax.set_xticklabels(df_terms_counts.nlargest(10, 'positive').loc[:,['term','positive']].term)

ax.grid(True)
tfidf_transformer = define_idfs(word_count_vector)
# count matrix

count_vector = tf_v.transform(df_categorized['text_processed'])

 

# tf-idf scores

tf_idf_vector = tfidf_transformer.transform(count_vector)
df_tf_idf_v1 = pd.DataFrame(data=tf_idf_vector.toarray(),index=np.arange(0, len(df_categorized)),columns=tf_v.get_feature_names())

print(tf_idf_vector.shape)

print(len(df_categorized))

df_tf_idf_v1['tweet_id'] = df_categorized['status_id']

df_tf_idf_v1['code'] = df_categorized['code']

df_tf_idf_v1['created_at'] = df_categorized['created_at']

df_tf_idf_v1['cat_original'] = df_categorized['cat']

df_tf_idf_v1[75:]
len(tf_v.get_feature_names())
df_tf_idf_v1.cat_original.unique()
df_tf_idf_v1['cat_ori_en'] = df_tf_idf_v1.apply(lambda x: 1 if str(x['cat_original']) == 'positive ' else 0, axis=1)

df_tf_idf_v1[40:50]
df_tf_idf_v1.cat_ori_en.unique()
df_tf_idf_v1.iloc[:5,0:562]
from sklearn.model_selection import StratifiedShuffleSplit

from sklearn.tree import DecisionTreeClassifier

from sklearn import ensemble



from sklearn import metrics



# define 10-fold cross validation test harness

seed = 7

np.random.seed(seed)

kfold = StratifiedShuffleSplit(n_splits=10, test_size=0.35, random_state=7)
history = {}

history['acc'] = []

history['precision_p'] = []

history['recall_p'] = []

history['precision_n'] = []

history['recall_n'] = []

history['true_values_x'] = []

history['true_values_y'] = []

history['pred_values_y'] = []



models_list = []



def classifier_tree(train_x, valid_x, train_y, valid_y):

    classifier_tree = DecisionTreeClassifier(random_state= 7)

#     classifier_tree = ensemble.RandomForestClassifier(random_state= 1)

    classifier_tree = classifier_tree.fit(train_x, train_y)



    # predict the labels on validation dataset

    valid_y_pred_tree = classifier_tree.predict(valid_x)

    

    accuracy_tree = metrics.accuracy_score(valid_y, valid_y_pred_tree)

    

    recall_tree_p = metrics.recall_score(valid_y, valid_y_pred_tree,pos_label=1,zero_division = 0 )

    precision_tree_p = metrics.precision_score(valid_y, valid_y_pred_tree,pos_label=1,zero_division = 0)

    

    recall_tree_n = metrics.recall_score(valid_y, valid_y_pred_tree,pos_label=0, zero_division = 0)

    precision_tree_n = metrics.precision_score(valid_y, valid_y_pred_tree,pos_label=0, zero_division = 0)

    

    history['acc'].append(accuracy_tree)

    history['precision_p'].append(precision_tree_p)

    history['recall_p'].append(recall_tree_p)

    history['precision_n'].append(precision_tree_n)

    history['recall_n'].append(recall_tree_n)

    history['true_values_x'].append(valid_x)

    history['true_values_y'].append(valid_y)

    history['pred_values_y'].append(valid_y_pred_tree)

    models_list.append(classifier_tree)
df_X = df_tf_idf_v1.iloc[:,0:294]

df_Y = df_tf_idf_v1['cat_ori_en']
for train, test in kfold.split(df_X, df_Y):

    classifier_tree(df_X.iloc[train],df_X.iloc[test],df_Y[train],df_Y[test])



print("Accuracy kfold=10", sum(history['acc'])/len(history['acc']))
len(models_list)
len(history['true_values_y'])
fig, ax = plt.subplots(figsize=(12, 12))

for i, mod in enumerate(models_list):

    koko = metrics.plot_roc_curve(mod,history['true_values_x'][i],history['true_values_y'][i],name='ROC fold {}'.format(i+1), lw=3, ax=ax)



plt.show()       
fig, ax = plt.subplots(figsize=(12, 12))

for i, mod in enumerate(models_list):

    koko = metrics.plot_precision_recall_curve(mod,history['true_values_x'][i],history['true_values_y'][i],name='ROC fold {}'.format(i+1), lw=3, ax=ax)



plt.show()  
from sklearn.metrics import confusion_matrix



cm = confusion_matrix(y_true=history['true_values_y'][9], y_pred=history['pred_values_y'][9]) 
import itertools

import matplotlib.pyplot as plt

import seaborn as sns



def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix',

                          cmap=sns.cubehelix_palette(as_cmap=True)):

    """

    This function is modified from: 

    http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html

    """

    if normalize:

        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        print("Normalized confusion matrix")

    else:

        print('Confusion matrix, without normalization')

    classes.sort()

    tick_marks = np.arange(len(classes))    

    plt.figure(figsize=(4, 4),dpi=115)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)

    plt.title(title)

    plt.colorbar()

    plt.xticks(tick_marks, classes, rotation=45)

    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'

    thresh = cm.max() / 2.

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):

        plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')

    plt.xlabel('Predicted label')

    plt.tight_layout()
plot_confusion_matrix(cm, classes=[0,1], title='Confusion matrix')
count_vector = tf_v.transform(df_us['text_processed'])

tf_idf_vector = tfidf_transformer.transform(count_vector)

df_us['cat_pre'] = models_list[7].predict(tf_idf_vector)

df_us.loc[:,['text','cat_pre']][:5]
df_us.to_csv('./df_us.csv', index=False)