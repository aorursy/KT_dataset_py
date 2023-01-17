import warnings

warnings.filterwarnings("ignore")

import numpy as np

import pandas as pd

import string as s

import matplotlib.pyplot as plt

import re

import nltk

from nltk.corpus import stopwords

%matplotlib inline

from sklearn.feature_extraction.text  import TfidfVectorizer

from sklearn.metrics  import f1_score, accuracy_score, multilabel_confusion_matrix, confusion_matrix, recall_score, precision_score

from lightgbm import LGBMClassifier

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import LabelEncoder

import sklearn

from keras.utils import to_categorical

import itertools
train_df = pd.read_json("../input/github-bugs-prediction-challenge-machine-hack/embold_train.json").reset_index(drop=True)

train_df.head()
test_df = pd.read_json("../input/github-bugs-prediction-challenge-machine-hack/embold_test.json").reset_index(drop=True)

test_df.head()
train_ex_df = pd.read_json("../input/github-bugs-prediction-challenge-machine-hack/embold_train_extra.json")

train_ex_df.head()
# Mapping news_id with category_id

final_df = pd.concat([train_df, train_ex_df], ignore_index=True)

final_df = train_df

final_df.head()
# Printing Sample Title

final_df.iloc[7][0]
# Printing Sample Body

final_df.iloc[7][1]
# Checking for missing snippets/titles/descriptions

final_df.info()
# Check for duplicates

final_df.drop_duplicates(keep='first').count()
categories = ['Bug','Feature','Question']
# Converting each of title and body into lower case.

final_df['title'] = final_df['title'].apply(lambda title: str(title).lower())

final_df['body'] = final_df['body'].apply(lambda body: str(body).lower())

test_df['title'] = test_df['title'].apply(lambda title: str(title).lower())

test_df['body'] = test_df['body'].apply(lambda body: str(body).lower())
#calculating the length of title and body

final_df['title_len'] = final_df['title'].apply(lambda x: len(str(x).split()))

final_df['body_len'] = final_df['body'].apply(lambda x: len(str(x).split()))
final_df.describe()
def fx(x):

    return x['title'] + " " + x['body']   

final_df['text']=final_df.apply(lambda x : fx(x),axis=1)

test_df['text']=test_df.apply(lambda x : fx(x),axis=1)
final_df.head()
def tokenization(text):

    lst=text.split()

    return lst
def remove_new_lines(lst):

    new_lst=[]

    for i in lst:

        i=i.replace(r'\n', ' ').replace(r'\r', ' ').replace(r'\u', ' ')

        new_lst.append(i.strip())

    return new_lst
def remove_punctuations(lst):

    new_lst=[]

    for i in lst:

        for  j in s.punctuation:

            i=i.replace(j,' ')

        new_lst.append(i.strip())

    return new_lst
def remove_numbers(lst):

    nodig_lst=[]

    new_lst=[]

    for i in  lst:

        for j in  s.digits:

            i=i.replace(j,' ')

        nodig_lst.append(i.strip())

    for i in  nodig_lst:

        if  i!='':

            new_lst.append(i.strip())

    return new_lst
def remove_stopwords(lst):

    stop=stopwords.words('english')

    new_lst=[]

    for i in lst:

        if i not in stop:

            new_lst.append(i.strip())

    return new_lst
lemmatizer=nltk.stem.WordNetLemmatizer()

def lemmatization(lst):

    new_lst=[]

    for i in lst:

        i=lemmatizer.lemmatize(i)

        new_lst.append(i.strip())

    return new_lst
def remove_urls(text):

    return re.sub(r'http\S+', ' ', text)
def split_words(text):

    return ' '.join(text).split()
def remove_single_chars(lst):

    new_lst=[]

    for i in lst:

        if len(i)>1:

            new_lst.append(i.strip())

    return new_lst
# Cleaning Text

def denoise_text(text):

    text = remove_urls(text)

    text = tokenization(text)

    text = remove_new_lines(text)

    text = remove_punctuations(text)

    text = remove_numbers(text)

    text = remove_stopwords(text)

    text = split_words(text)

    text = remove_single_chars(text)

    text = lemmatization(text)

    return text



final_df['text'] = final_df['text'].apply(lambda x: denoise_text(x))

test_df['text'] = test_df['text'].apply(lambda x: denoise_text(x))
# Word Corpus

def get_corpus(text):

    words = []

    for i in text:

        for j in i:

            words.append(j.strip())

    return words

corpus = get_corpus(final_df.text)

corpus[:5]
corpus += get_corpus(test_df.text)
# Most common words

from collections import Counter

counter = Counter(corpus)

most_common = counter.most_common(10)

most_common = dict(most_common)

most_common
from sklearn.feature_extraction.text import CountVectorizer

def get_top_text_ngrams(corpus, n, g):

    vec = CountVectorizer(ngram_range=(g, g)).fit(corpus)

    bag_of_words = vec.transform(corpus)

    sum_words = bag_of_words.sum(axis=0) 

    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]

    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)

    return words_freq[:n]
#label encoding the categories. After this each category would be mapped to an integer.

encoder = LabelEncoder()

final_df['categoryEncoded'] = encoder.fit_transform(final_df['label'])
X_train, X_test, y_train, y_test = train_test_split(final_df['text'], final_df['categoryEncoded'], random_state = 43, test_size = 0.2)
train_x=X_train.apply(lambda x: ''.join(i+' ' for i in x))

test_x=X_test.apply(lambda x: ''.join(i+' '  for i in x))

test_df_final = test_df['text'].apply(lambda x: ''.join(i+' '  for i in x))
tfidf=TfidfVectorizer(max_features=10000,min_df=6)

train_1=tfidf.fit_transform(train_x)

test_1=tfidf.transform(test_x)

test_2=tfidf.transform(test_df_final)

print("No. of features extracted:", len(tfidf.get_feature_names()))

print(tfidf.get_feature_names()[:20])



train_arr=train_1.toarray()

test_arr=test_1.toarray()
test_arr1=test_2.toarray()
def eval_model(y,y_pred):

    print("Recall score of the model:", round(recall_score(y_test, pred, average='weighted'), 3))

    print("Precision score of the model:", round(precision_score(y_test, pred, average='weighted'), 3))

    print("F1 score of the model:", round(f1_score(y,y_pred,average='micro'), 3))

    print("Accuracy of the model:", round(accuracy_score(y,y_pred),3))

    print("Accuracy of the model in percentage:", round(accuracy_score(y,y_pred)*100,3),"%")
def plot_confusion_matrix(cm,

                          target_names,

                          title='Confusion matrix',

                          cmap=None,

                          normalize=True):

    """

    given a sklearn confusion matrix (cm), make a nice plot



    Arguments

    ---------

    cm:           confusion matrix from sklearn.metrics.confusion_matrix



    target_names: given classification classes such as [0, 1, 2]

                  the class names, for example: ['high', 'medium', 'low']



    title:        the text to display at the top of the matrix



    cmap:         the gradient of the values displayed from matplotlib.pyplot.cm

                  see http://matplotlib.org/examples/color/colormaps_reference.html

                  plt.get_cmap('jet') or plt.cm.Blues



    normalize:    If False, plot the raw numbers

                  If True, plot the proportions



    Usage

    -----

    plot_confusion_matrix(cm           = cm,                  # confusion matrix created by

                                                              # sklearn.metrics.confusion_matrix

                          normalize    = True,                # show proportions

                          target_names = y_labels_vals,       # list of names of the classes

                          title        = best_estimator_name) # title of graph



    Citiation

    ---------

    http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html



    """

    accuracy = np.trace(cm) / float(np.sum(cm))

    misclass = 1 - accuracy



    if cmap is None:

        cmap = plt.get_cmap('Blues')



    plt.figure(figsize=(8, 6))

    plt.imshow(cm, interpolation='nearest', cmap=cmap)

    plt.title(title)

    plt.colorbar()



    if target_names is not None:

        tick_marks = np.arange(len(target_names))

        plt.xticks(tick_marks, target_names, rotation=45)

        plt.yticks(tick_marks, target_names)



    if normalize:

        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]





    thresh = cm.max() / 1.5 if normalize else cm.max() / 2

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):

        if normalize:

            plt.text(j, i, "{:0.4f}".format(cm[i, j]),

                     horizontalalignment="center",

                     color="white" if cm[i, j] > thresh else "black")

        else:

            plt.text(j, i, "{:,}".format(cm[i, j]),

                     horizontalalignment="center",

                     color="white" if cm[i, j] > thresh else "black")





    plt.tight_layout()

    plt.ylabel('True label')

    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))

    plt.show()
def confusion_mat(color):

    cm=confusion_matrix(y_test, pred)

    plot_confusion_matrix(cm,

                          categories,

                          title='Confusion matrix')

    
lgbm=LGBMClassifier()

lgbm.fit(train_arr,y_train)

pred=lgbm.predict(test_arr)



print("first 20 actual labels")

print(y_test.tolist()[:20])

print("first 20 predicted labels")

print(pred.tolist()[:20])
eval_model(y_test,pred)

b=round(accuracy_score(y_test,pred)*100,3)
confusion_mat('Blues')
pred=lgbm.predict(test_arr1)

#create a submission dataframe

submission_df = pd.DataFrame(pred, columns=['label'])

#write a .csv file for submission

submission_df.to_csv('lgbm_submission.csv', index=False)