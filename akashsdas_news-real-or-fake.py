import re

import string

import itertools



import numpy as np

import pandas as pd

import matplotlib.pyplot as plt



import nltk

from wordcloud import WordCloud, STOPWORDS



from sklearn.feature_extraction.text import CountVectorizer

from sklearn.feature_extraction.text import TfidfTransformer

from sklearn.metrics import classification_report, confusion_matrix

from sklearn.metrics import f1_score, precision_score, recall_score

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, StratifiedKFold



from sklearn.svm import SVC, LinearSVC

from sklearn.naive_bayes import MultinomialNB

from sklearn.linear_model import LogisticRegression



from sklearn.pipeline import Pipeline
true_news_df = pd.read_csv('/kaggle/input/fake-and-real-news-dataset/Fake.csv')

fake_news_df = pd.read_csv('/kaggle/input/fake-and-real-news-dataset/True.csv')
true_news_df.head()
fake_news_df.head()
stopwords = set(STOPWORDS)
# True news word cloud

true_news_wc = WordCloud(

    background_color='black', 

    max_words=200, 

    stopwords=stopwords

).generate(''.join(true_news_df['text']))





f, ax = plt.subplots(nrows=1, ncols=1, figsize=(15, 18))

ax.imshow(true_news_wc, interpolation='bilinear')

ax.axis('off')

ax.set_label('True News')
# Fake news word cloud

fake_news_wc = WordCloud(

    background_color='black', 

    max_words=200, 

    stopwords=stopwords

).generate(''.join(fake_news_df['text']))



f, ax = plt.subplots(nrows=1, ncols=1, figsize=(20, 10))

ax.imshow(fake_news_wc, interpolation='bilinear')

ax.axis('off')

ax.set_label('Fake News')
def convert_from_list_to_text(_list):

    text = ' '.join(_list)

    return text





def plot_confusion_matrix(cm, classes,

                          normalize=False,

                          title='Confusion matrix',

                          cmap=plt.cm.Purples):

    """

    This function prints and plots the confusion matrix.

    Normalization can be applied by setting `normalize=True`.

    """

    if normalize:

        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        print("Normalized confusion matrix")

    else:

        print('Confusion matrix, without normalization')



    print(cm)



    plt.imshow(cm, interpolation='nearest', cmap=cmap)

    plt.title(title)

    plt.colorbar()

    tick_marks = np.arange(len(classes))

    plt.xticks(tick_marks, classes, rotation=45)

    plt.yticks(tick_marks, classes)



    fmt = '.2f' if normalize else 'd'

    thresh = cm.max() / 2.

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):

        plt.text(j, i, format(cm[i, j], fmt),

                 horizontalalignment="center",

                 color="white" if cm[i, j] > thresh else "black")



    plt.tight_layout()

    plt.ylabel('True label')

    plt.xlabel('Predicted label')
def unique(text):

    text = text.split()

    return list(set(text))
true_news_df['text']  = true_news_df['text'].apply(lambda x: unique(x))

fake_news_df['text']  = fake_news_df['text'].apply(lambda x: unique(x))



true_news_df.head(1)
# Remove punctuation from a word if it exist

def rm_punc_from_word(word):

    clean_word = ''                # word without punctuation

    for alpha in word:

        # checking if alphabet is punctuation or not

        if alpha in string.punctuation:

            continue

        clean_word += alpha

        

    return clean_word





# Remove any punctuation and clean words having punctuation

def clean_punc(words_list):

    for idx, word in enumerate(words_list):

        if word in string.punctuation:

            words_list.remove(word)

        else:

            words_list[idx] = rm_punc_from_word(word)

            words_list[idx] = re.sub('[0-9]+', '', words_list[idx])

            

    return words_list
true_news_df['text']  = true_news_df['text'].apply(lambda x: clean_punc(x))

fake_news_df['text']  = fake_news_df['text'].apply(lambda x: clean_punc(x))



true_news_df.head(1)
def rm_num(words_list):

    text = ' '.join(words_list)

    text = ''.join([i for i in text if not i.isdigit()])

    return text.split()
true_news_df['text']  = true_news_df['text'].apply(lambda x: rm_num(x))

fake_news_df['text']  = fake_news_df['text'].apply(lambda x: rm_num(x))



true_news_df.head(1)
def tokenization(words_list):

    tmp = words_list.copy()

    words_list = []

    

    for idx, word in enumerate(tmp):

        for split_word in re.split('\W+', word):

            words_list.append(split_word)



    words_list = ' '.join(words_list).split()  # removing any white spaces

    return words_list
true_news_df['text']  = true_news_df['text'].apply(lambda x: tokenization(x))

fake_news_df['text']  = fake_news_df['text'].apply(lambda x: tokenization(x))



true_news_df.head(1)
def remove_URL(words_list):

    url = re.compile(r'https?://\S+|www\.\S+')

    for idx, word in enumerate(words_list):

        words_list[idx] = url.sub(r'',word)

    return words_list
true_news_df['text']  = true_news_df['text'].apply(lambda x: remove_URL(x))

fake_news_df['text']  = fake_news_df['text'].apply(lambda x: remove_URL(x))



true_news_df.head(1)
def remove_HTML(words_list):

    html = re.compile(r'<.*?>')

    for idx, word in enumerate(words_list):

        words_list[idx] = html.sub(r'',word)

    return words_list
true_news_df['text']  = true_news_df['text'].apply(lambda x: remove_HTML(x))

fake_news_df['text']  = fake_news_df['text'].apply(lambda x: remove_HTML(x))



true_news_df.head(1)
def remove_emoji(words_list):

    emoji_pattern = re.compile("["

                           u"\U0001F600-\U0001F64F"  # emoticons

                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs

                           u"\U0001F680-\U0001F6FF"  # transport & map symbols

                           u"\U0001F1E0-\U0001F1FF"  # flags (iOS)

                           u"\U00002702-\U000027B0"

                           u"\U000024C2-\U0001F251"

                           "]+", flags=re.UNICODE)



    for idx, word in enumerate(words_list):

        words_list[idx] = emoji_pattern.sub(r'',word)

    return words_list
true_news_df['text']  = true_news_df['text'].apply(lambda x: remove_emoji(x))

fake_news_df['text']  = fake_news_df['text'].apply(lambda x: remove_emoji(x))



true_news_df.head(1)
stopwords = set(nltk.corpus.stopwords.words())



def clean_stopwords(words_list):

    for word in words_list:

        if word in stopwords:

            words_list.remove(word)

    return words_list
true_news_df['text']  = true_news_df['text'].apply(lambda x: clean_stopwords(x))

fake_news_df['text']  = fake_news_df['text'].apply(lambda x: clean_stopwords(x))



true_news_df.head(1)
true_news_df['is_fake'] = 0

fake_news_df['is_fake'] = 1



true_news_df.head()
true_news_df['text']  = true_news_df['text'].apply(lambda x: convert_from_list_to_text(x))

fake_news_df['text']  = fake_news_df['text'].apply(lambda x: convert_from_list_to_text(x))



true_news_df.head()
# True news word cloud

true_news_wc = WordCloud(

    background_color='black', 

    max_words=200, 

    stopwords=stopwords

).generate(''.join(true_news_df['text']))





f, ax = plt.subplots(nrows=1, ncols=1, figsize=(15, 18))

ax.imshow(true_news_wc, interpolation='bilinear')

ax.axis('off')

ax.set_label('True News')
# Fake news word cloud

fake_news_wc = WordCloud(

    background_color='black', 

    max_words=200, 

    stopwords=stopwords

).generate(''.join(fake_news_df['text']))



f, ax = plt.subplots(nrows=1, ncols=1, figsize=(20, 10))

ax.imshow(fake_news_wc, interpolation='bilinear')

ax.axis('off')

ax.set_label('Fake News')
# ✨ Concatenating true and fake news dataframes and suffling them



df = pd.concat([true_news_df, fake_news_df], axis='index')

df = df.sample(frac=1).reset_index(drop=True)

df.head()
# 🔪 Splitting the dataset into train & test



skf = StratifiedKFold(n_splits=10)



X = df.text

Y = df.is_fake



X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.1, random_state=212)
count_vec = CountVectorizer()

X_train_counts = count_vec.fit_transform(X_train)



X_train_counts.shape
tfidf_transformer = TfidfTransformer()

X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)



X_train_tfidf.shape
score = cross_val_score(MultinomialNB(), X_train_tfidf, y_train, cv=skf)

print(f'MultinomialNB mean of cross validation score: {score.mean()}')



score = cross_val_score(LogisticRegression(), X_train_tfidf, y_train, cv=skf)

print(f'LogisticRegression mean of cross validation score: {score.mean()}')



score = cross_val_score(LinearSVC(), X_train_tfidf, y_train, cv=skf)

print(f'LinearSVC mean of cross validation score: {score.mean()}')
model = Pipeline([

    ('vect', CountVectorizer()),

    ('tfidf', TfidfTransformer()),

    ('clf', LinearSVC())

])
model.fit(X_train, y_train)
y_test_pred = model.predict(X_test)

np.mean(y_test_pred == y_test)
print(f'Model Score: {model.score(X_test, y_test)}')

print(f'f1-score: {f1_score(y_test, y_test_pred, average="weighted")}')

print(f'precision score: {precision_score(y_test, y_test_pred, average="weighted")}')

print(f'recall score: {recall_score(y_test, y_test_pred, average="weighted")}')
cnf_matrix = confusion_matrix(y_test, y_test_pred, labels=[0,1])

np.set_printoptions(precision=2)



print(classification_report(y_test, y_test_pred))



# Plot non-normalized confusion matrix

plt.figure(figsize=(6, 6))

plot_confusion_matrix(cnf_matrix, classes=['True(0)', 'Fake(1)'],normalize= False,  title='Confusion matrix', cmap=plt.cm.RdPu)