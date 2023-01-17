%matplotlib inline



import sqlite3

import pandas as pd

import numpy as np

import nltk

import string

import matplotlib.pyplot as plt

import matplotlib as mpl

import numpy as np



from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix

from sklearn import metrics

from sklearn.metrics import roc_curve, auc

from nltk.stem.porter import PorterStemmer
import os

from IPython.core.display import display, HTML

    

if not os.path.isfile('../input/database.sqlite'):

    display(HTML("""

        <h3 style='color: red'>Dataset database missing!</h3><h3> Please download it

        <a href='https://www.kaggle.com/snap/amazon-fine-food-reviews'>from here on Kaggle</a>

        and extract it to the current directory.

          """))

    raise(Exception("missing dataset"))

        
con = sqlite3.connect('../input/database.sqlite')



pd.read_sql_query("SELECT * FROM Reviews LIMIT 3", con)
messages = pd.read_sql_query("""

SELECT 

  Score, 

  Summary, 

  HelpfulnessNumerator as VotesHelpful, 

  HelpfulnessDenominator as VotesTotal

FROM Reviews 

WHERE Score != 3""", con)
messages.head(5)
messages["Sentiment"] = messages["Score"].apply(lambda score: "positive" if score > 3 else "negative")

messages["Usefulness"] = (messages["VotesHelpful"]/messages["VotesTotal"]).apply(lambda n: "useful" if n > 0.8 else "useless")



messages.head(5)
messages[messages.Score == 5].head(10)
messages[messages.Score == 1].head(10)
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer



import re

import string

import nltk



cleanup_re = re.compile('[^a-z]+')

def cleanup(sentence):

    sentence = sentence.lower()

    sentence = cleanup_re.sub(' ', sentence).strip()

    #sentence = " ".join(nltk.word_tokenize(sentence))

    return sentence



messages["Summary_Clean"] = messages["Summary"].apply(cleanup)



train, test = train_test_split(messages, test_size=0.2)

print("%d items in training data, %d in test data" % (len(train), len(test)))
from wordcloud import WordCloud, STOPWORDS



# To cleanup stop words, add stop_words = STOPWORDS

# But it seems to function better without it

count_vect = CountVectorizer(min_df = 1, ngram_range = (1, 4))

X_train_counts = count_vect.fit_transform(train["Summary_Clean"])



tfidf_transformer = TfidfTransformer()

X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)



X_new_counts = count_vect.transform(test["Summary_Clean"])

X_test_tfidf = tfidf_transformer.transform(X_new_counts)



y_train = train["Sentiment"]

y_test = test["Sentiment"]



prediction = dict()
from wordcloud import WordCloud, STOPWORDS

stopwords = set(STOPWORDS)



#mpl.rcParams['figure.figsize']=(8.0,6.0)    #(6.0,4.0)

mpl.rcParams['font.size']=12                #10 

mpl.rcParams['savefig.dpi']=100             #72 

mpl.rcParams['figure.subplot.bottom']=.1 





def show_wordcloud(data, title = None):

    wordcloud = WordCloud(

        background_color='white',

        stopwords=stopwords,

        max_words=200,

        max_font_size=40, 

        scale=3,

        random_state=1 # chosen at random by flipping a coin; it was heads

    ).generate(str(data))

    

    fig = plt.figure(1, figsize=(8, 8))

    plt.axis('off')

    if title: 

        fig.suptitle(title, fontsize=20)

        fig.subplots_adjust(top=2.3)



    plt.imshow(wordcloud)

    plt.show()

    

show_wordcloud(messages["Summary_Clean"])
show_wordcloud(messages[messages.Score == 1]["Summary_Clean"], title = "Low scoring")
show_wordcloud(messages[messages.Score == 5]["Summary_Clean"], title = "High scoring")
from sklearn.naive_bayes import MultinomialNB

model = MultinomialNB().fit(X_train_tfidf, y_train)

prediction['Multinomial'] = model.predict(X_test_tfidf)
from sklearn.naive_bayes import BernoulliNB

model = BernoulliNB().fit(X_train_tfidf, y_train)

prediction['Bernoulli'] = model.predict(X_test_tfidf)
from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression(C=1e5)

logreg_result = logreg.fit(X_train_tfidf, y_train)

prediction['Logistic'] = logreg.predict(X_test_tfidf)
def formatt(x):

    if x == 'negative':

        return 0

    return 1

vfunc = np.vectorize(formatt)



cmp = 0

colors = ['b', 'g', 'y', 'm', 'k']

for model, predicted in prediction.items():

    false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test.map(formatt), vfunc(predicted))

    roc_auc = auc(false_positive_rate, true_positive_rate)

    plt.plot(false_positive_rate, true_positive_rate, colors[cmp], label='%s: AUC %0.2f'% (model,roc_auc))

    cmp += 1



plt.title('Classifiers comparaison with ROC')

plt.legend(loc='lower right')

plt.plot([0,1],[0,1],'r--')

plt.xlim([-0.1,1.2])

plt.ylim([-0.1,1.2])

plt.ylabel('True Positive Rate')

plt.xlabel('False Positive Rate')

plt.show()
print(metrics.classification_report(y_test, prediction['Logistic'], target_names = ["positive", "negative"]))
def plot_confusion_matrix(cm, title='Confusion matrix', cmap=plt.cm.Blues, labels=["positive", "negative"]):

    plt.imshow(cm, interpolation='nearest', cmap=cmap)

    plt.title(title)

    plt.colorbar()

    tick_marks = np.arange(len(labels))

    plt.xticks(tick_marks, labels, rotation=45)

    plt.yticks(tick_marks, labels)

    plt.tight_layout()

    plt.ylabel('True label')

    plt.xlabel('Predicted label')

    

# Compute confusion matrix

cm = confusion_matrix(y_test, prediction['Logistic'])

np.set_printoptions(precision=2)

plt.figure()

plot_confusion_matrix(cm)    



cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

plt.figure()

plot_confusion_matrix(cm_normalized, title='Normalized confusion matrix')

plt.show()
words = count_vect.get_feature_names()

feature_coefs = pd.DataFrame(

    data = list(zip(words, logreg_result.coef_[0])),

    columns = ['feature', 'coef'])



feature_coefs.sort_values(by='coef')
def test_sample(model, sample):

    sample_counts = count_vect.transform([sample])

    sample_tfidf = tfidf_transformer.transform(sample_counts)

    result = model.predict(sample_tfidf)[0]

    prob = model.predict_proba(sample_tfidf)[0]

    print("Sample estimated as %s: negative prob %f, positive prob %f" % (result.upper(), prob[0], prob[1]))



test_sample(logreg, "The food was delicious, it smelled great and the taste was awesome")

test_sample(logreg, "The whole experience was horrible. The smell was so bad that it literally made me sick.")

test_sample(logreg, "The food was ok, I guess. The smell wasn't very good, but the taste was ok.")
test_sample(logreg, "The smell reminded me of ammonia")
show_wordcloud(messages[messages.Usefulness == "useful"]["Summary_Clean"], title = "Useful")

show_wordcloud(messages[messages.Usefulness == "useless"]["Summary_Clean"], title = "Useless")
messages_ufn = messages[messages.VotesTotal >= 10]

messages_ufn.head()
show_wordcloud(messages_ufn[messages_ufn.Usefulness == "useful"]["Summary_Clean"], title = "Useful")

show_wordcloud(messages_ufn[messages_ufn.Usefulness == "useless"]["Summary_Clean"], title = "Useless")
from sklearn.pipeline import Pipeline



train_ufn, test_ufn = train_test_split(messages_ufn, test_size=0.2)



ufn_pipe = Pipeline([

    ('vect', CountVectorizer(min_df = 1, ngram_range = (1, 4))),

    ('tfidf', TfidfTransformer()),

    ('clf', LogisticRegression(C=1e5)),

])



ufn_result = ufn_pipe.fit(train_ufn["Summary_Clean"], train_ufn["Usefulness"])



prediction['Logistic_Usefulness'] = ufn_pipe.predict(test_ufn["Summary_Clean"])

print(metrics.classification_report(test_ufn["Usefulness"], prediction['Logistic_Usefulness']))
ufn_scores = [a[0] for a in ufn_pipe.predict_proba(train_ufn["Summary"])]

ufn_scores = zip(ufn_scores, train_ufn["Summary"], train_ufn["VotesHelpful"], train_ufn["VotesTotal"])

ufn_scores = sorted(ufn_scores, key=lambda t: t[0], reverse=True)



# just make this into a DataFrame since jupyter renders it nicely:

pd.DataFrame(ufn_scores)
cm = confusion_matrix(test_ufn["Usefulness"], prediction['Logistic_Usefulness'])

cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

np.set_printoptions(precision=2)

plt.figure()

plot_confusion_matrix(cm_normalized, labels=["useful", "useless"])
from sklearn.pipeline import FeatureUnion

from sklearn.base import BaseEstimator, TransformerMixin



# Useful to select only certain features in a dataset for forwarding through a pipeline

# See: http://scikit-learn.org/stable/auto_examples/hetero_feature_union.html

class ItemSelector(BaseEstimator, TransformerMixin):

    def __init__(self, key):

        self.key = key

    def fit(self, x, y=None):

        return self

    def transform(self, data_dict):

        return data_dict[self.key]



train_ufn2, test_ufn2 = train_test_split(messages_ufn, test_size=0.2)



ufn_pipe2 = Pipeline([

   ('union', FeatureUnion(

       transformer_list = [

           ('summary', Pipeline([

               ('textsel', ItemSelector(key='Summary_Clean')),

               ('vect', CountVectorizer(min_df = 1, ngram_range = (1, 4))),

               ('tfidf', TfidfTransformer())])),

          ('score', ItemSelector(key=['Score']))

       ],

       transformer_weights = {

           'summary': 0.2,

           'score': 0.8

       }

   )),

   ('model', LogisticRegression(C=1e5))

])



ufn_result2 = ufn_pipe2.fit(train_ufn2, train_ufn2["Usefulness"])

prediction['Logistic_Usefulness2'] = ufn_pipe2.predict(test_ufn2)

print(metrics.classification_report(test_ufn2["Usefulness"], prediction['Logistic_Usefulness2']))
cm = confusion_matrix(test_ufn2["Usefulness"], prediction['Logistic_Usefulness2'])

cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

np.set_printoptions(precision=2)

plt.figure()

plot_confusion_matrix(cm_normalized, labels=["useful", "useless"])
len(ufn_result2.named_steps['model'].coef_[0])
ufn_summary_pipe = next(tr[1] for tr in ufn_result2.named_steps["union"].transformer_list if tr[0]=='summary')

ufn_words = ufn_summary_pipe.named_steps['vect'].get_feature_names()

ufn_features = ufn_words + ["Score"]

ufn_feature_coefs = pd.DataFrame(

    data = list(zip(ufn_features, ufn_result2.named_steps['model'].coef_[0])),

    columns = ['feature', 'coef'])

ufn_feature_coefs.sort_values(by='coef')
print("And the coefficient of the Score variable: ")

ufn_feature_coefs[ufn_feature_coefs.feature == 'Score']