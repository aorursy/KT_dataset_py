import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn import metrics
# from keras.preprocessing.text import Tokenizer
# from keras.preprocessing.sequence import pad_sequences
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

import seaborn as sns
# import the metrics class
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score,roc_curve,auc

import matplotlib.pyplot as plt

import numpy as np
import os

pd.set_option('display.max_colwidth', -1)
# import nltk
# nltk.download('stopwords')
# nltk.download('wordnet')
DATA_PATH = '/kaggle/input/tweet-sentiment-extraction/'
train = pd.read_csv(DATA_PATH+'train.csv')
test = pd.read_csv(DATA_PATH+'test.csv')
submission_df = pd.read_csv(DATA_PATH+'sample_submission.csv')
train.head()
import warnings; warnings.simplefilter('ignore')
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from nltk.stem import WordNetLemmatizer
train['text'] = train['text'].apply(lambda x: " ".join(x.lower() for x in str(x).split()))
train['text'] = train['text'].str.replace('[^\w\s]','')
stop = stopwords.words('english')
stop = stop + ['information','page','number','please']


train['text'] = train['text'].apply(lambda x: " ".join(x for x in x.split() if x not in stop))
train['text'] = train['text'].str.replace('\d+', '')

stemmer = SnowballStemmer('english')
train['text'].apply(lambda x: " ".join([stemmer.stem(word) for word in x.split()]))

wordnet_lemmatizer = WordNetLemmatizer()
train['text'] = train['text'].apply(lambda x: " ".join([wordnet_lemmatizer.lemmatize(word) for word in x.split()]))


# test
test['text'] = test['text'].apply(lambda x: " ".join(x.lower() for x in str(x).split()))
test['text'] = test['text'].str.replace('[^\w\s]','')
stop = stopwords.words('english')
stop = stop + ['information','page','number','please']


test['text'] = test['text'].apply(lambda x: " ".join(x for x in x.split() if x not in stop))
test['text'] = test['text'].str.replace('\d+', '')

stemmer = SnowballStemmer('english')
test['text'].apply(lambda x: " ".join([stemmer.stem(word) for word in x.split()]))

wordnet_lemmatizer = WordNetLemmatizer()
test['text'] = test['text'].apply(lambda x: " ".join([wordnet_lemmatizer.lemmatize(word) for word in x.split()]))
# from wordcloud import WordCloud, STOPWORDS
# stopwords = set(STOPWORDS)


# wordcloud = WordCloud(
#                           background_color='white',
#                           stopwords=stopwords,
#                           max_words=200,
#                           max_font_size=40, 
#                           random_state=42
#                          ).generate(str(data[data.error==1]['text']))

# print(wordcloud)
# fig = plt.figure(1)
# plt.imshow(wordcloud)
# plt.axis('off')
# plt.show()
# fig.savefig("word1.png", dpi=900)
# from wordcloud import WordCloud, STOPWORDS
# stopwords = set(STOPWORDS)


# wordcloud = WordCloud(
#                           background_color='white',
#                           stopwords=stopwords,
#                           max_words=200,
#                           max_font_size=40, 
#                           random_state=42
#                          ).generate(str(data[data.error==2]['text']))

# print(wordcloud)
# fig = plt.figure(1)
# plt.imshow(wordcloud)
# plt.axis('off')
# plt.show()
# fig.savefig("word1.png", dpi=900)
# documents=[[word for word in document.text.split()] for document in data['text'].tolist()]
# dictionary = corpora.Dictionary(documents)
# n_items = len(dictionary)
# corpus = [dictionary.doc2bow(text) for text in documents]
# tfidf = models.TfidfModel(corpus)
# Splitting the dataset into the Training set and Test set
features = ['text']
target = ['sentiment']

X = train[features]
y = train[target]
from sklearn.model_selection import train_test_split
X_train, val_X, train_y, val_y = train_test_split(X, y, test_size = 0.2, stratify=y, random_state = 0)
print("train test split done")
from sklearn.externals import joblib
# from components.classifier.text_utils import preprocess
from gensim.corpora import Dictionary
from gensim.models import TfidfModel

## fill up the missing values
## fill up the missing values
train_X = X_train["text"].fillna("_na_")
val_X = val_X["text"].fillna("_na_")

test_X = test["text"].fillna("_na_")


# ## some config values 
# # # Get the tfidf vectors #
tfidf_vec = TfidfVectorizer(stop_words='english', ngram_range=(1,3))
tfidf_vec.fit_transform(train_X.values.tolist()+val_X.values.tolist()+test_X.values.tolist())
train_tfidf = tfidf_vec.transform(train_X.values.tolist())
val_tfidf = tfidf_vec.transform(val_X.values.tolist())
test_tfidf = tfidf_vec.transform(test_X.values.tolist())
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn import metrics

def plot_cm(y_true, y_pred, names_list, figsize=(10,10)):

    cm = confusion_matrix(y_true, y_pred, labels=np.unique(y_true))
    cm_sum = np.sum(cm, axis=1, keepdims=True)
    cm_perc = cm / cm_sum.astype(float) * 100
    annot = np.empty_like(cm).astype(str)
    nrows, ncols = cm.shape

    for i in range(nrows): 
        for j in range(ncols):
            c = cm[i, j]
            p = cm_perc[i, j]
            if i == j:
                s = cm_sum[i]
                annot[i, j] = '%.1f%%\n%d/%d' % (p, c, s)
            elif c == 0:
                annot[i, j] = ''
            else:
                annot[i, j] = '%.1f%%\n%d' % (p, c)

    cm = pd.DataFrame(cm, index=np.unique(y_true), columns=np.unique(y_true))
    cm.index.name = 'Actual'
    cm.columns.name = 'Predicted'
    sns.set(font_scale=1.25)
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(cm, cmap= "Blues", annot=annot, fmt='', ax=ax)

def get_metrics(validation_label,predicted_label,names_list):
    cm = confusion_matrix(validation_label,predicted_label)
    print("Accuracy:",metrics.accuracy_score(validation_label,predicted_label))
#     names_list = list(data['document_page_type'].unique())
    print(metrics.classification_report(validation_label,predicted_label, target_names=names_list))
names_list = list(train['sentiment'].unique())


# from sklearn.svm import LinearSVC
# # model =  LinearSVC(C=3)
# from sklearn.naive_bayes import MultinomialNB
# from sklearn.externals import joblib
# from sklearn.model_selection import GridSearchCV


# from sklearn.model_selection import GridSearchCV
# from sklearn import svm

# model = svm.SVC(kernel='linear', probability=True)
# grid_param = {
#     'C': [1],
#     'gamma': [0.001,0.1]
# }

# gd_sr = GridSearchCV(estimator=model,
#                      param_grid=grid_param,
#                      scoring='accuracy',
#                      cv=2,
#                      n_jobs=-1)

# gd_sr.fit(train_tfidf, train_y)

# y_pred_val_svm = gd_sr.predict(val_tfidf)

# get_metrics(val_y, y_pred_val_svm, names_list)
# plot_cm(val_y, y_pred_val_svm, names_list)
# print(gd_sr.best_params_)
# # plt.savefig(path + 'Fax Cover BInary-Apri-poc- SVM.png', dpi=300)
# y_pred_val_svm = gd_sr.predict_proba(val_tfidf)
# y_pred_val_svm
# get_metrics(val_y, y_pred_val_svm, names_list)
# plot_cm(val_y, y_pred_val_svm, names_list)
# print(gd_sr.best_params_)
# pd.Series(y_pred_val_svm)

from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import MultinomialNB
model = MultinomialNB(alpha=.01)

grid_param = {
    'alpha': [.001]
}

gd_sr = GridSearchCV(estimator=model,
                     param_grid=grid_param,
                     scoring='accuracy',
                     cv=2,
                     n_jobs=-1)

gd_sr.fit(train_tfidf, train_y)

y_pred_val_nb = gd_sr.predict(val_tfidf)

get_metrics(val_y, y_pred_val_nb, names_list)
plot_cm(val_y, y_pred_val_nb, names_list)
print(gd_sr.best_params_)
# plt.savefig(path + 'Fax Cover BInary-Apri-poc- NB.png', dpi=300)