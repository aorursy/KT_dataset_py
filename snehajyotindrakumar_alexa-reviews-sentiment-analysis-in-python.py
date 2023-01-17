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
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import matplotlib as mpl

import nltk.classify.util

from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix

from sklearn import metrics

from sklearn.metrics import roc_curve, auc

import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning) 

from nltk.classify import NaiveBayesClassifier

import re

import string

import nltk

%matplotlib inline
temp=pd.read_csv('/kaggle/input/amazon-alexa-reviews/amazon_alexa.tsv',delimiter='\t',encoding='utf-8')
temp.describe(include = 'all')

temp.info()
temp.head()
temp.rating.value_counts().plot(kind='pie',figsize=(10,10), title='Rating')
temp.rating.value_counts().plot(kind='bar', figsize=(10,10), title='Rating')
print(temp['rating'].hist(by=temp.feedback, bins=range(1,6,1)))
star = temp['rating'].value_counts()

print("*** Rating distribution ***")

print(star)

star.sort_index(inplace=True)

star.plot(kind='bar',title='Amazon customer ratings',figsize=(6,6),style='Solarize_Light2')
NPS_score = round (100*((star.loc[5])-sum(star.loc[1:3]))/sum(star.loc[:]),2)

print (" NPS score of Amazon is : "  + str(NPS_score))
print(temp.isnull().sum())

temp.head(2)
perm = temp.copy()
perm["sentiment"]=perm["rating"]>=4
perm["sentiment"]=perm["sentiment"].replace([True,False],["pos","neg"])
perm["sentiment"].value_counts().plot.bar()
#Unbalanced data

#Cleaning the data

import nltk.classify.util

from nltk.classify import NaiveBayesClassifier

import numpy as np

import re

import string

import nltk
cleanup_re = re.compile('[^a-z]+')

def cleanup(sentence):

    sentence = str(sentence)

    sentence = sentence.lower()

    sentence = cleanup_re.sub(' ', sentence).strip()

    return sentence
perm["Summary_Clean"] = perm["verified_reviews"].apply(cleanup)
perm.loc[:,["Summary_Clean", "verified_reviews" ]].head()
#Spliting train and test data in 80:20 ratio

split = perm[["Summary_Clean" , "sentiment"]]

train=split.sample(frac=0.8,random_state=5)

test=split.drop(train.index)
train.shape
test.shape
def word_feats(words):

    features = {}

    for word in words:

        features [word] = True

    return features
train["words"] = train["Summary_Clean"].str.lower().str.split()

test["words"] = test["Summary_Clean"].str.lower().str.split()
train.index = range(train.shape[0])

test.index = range(test.shape[0])

prediction =  {} ## For storing results of different classifiers
train_naive = []

test_naive = []
for i in range(train.shape[0]):

    train_naive = train_naive +[[word_feats(train["words"][i]) , train["sentiment"][i]]]

for i in range(test.shape[0]):

    test_naive = test_naive +[[word_feats(test["words"][i]) , test["sentiment"][i]]]
len(train["words"][1])
len(train_naive[1][0])
train_naive[0]
classifier = NaiveBayesClassifier.train(train_naive)

print("NLTK Naive bayes Accuracy : {}".format(nltk.classify.util.accuracy(classifier , test_naive)))

classifier.show_most_informative_features(5)
y =[]

only_words= [test_naive[i][0] for i in range(test.shape[0])]

for i in range(test.shape[0]):

    y = y + [classifier.classify(only_words[i] )]

prediction["Naive"]= np.asarray(y)
from wordcloud import STOPWORDS
from sklearn.feature_extraction.text import TfidfTransformer

from sklearn.feature_extraction.text import CountVectorizer

stopwords = set(STOPWORDS)

stopwords.remove("not")



count_vect = CountVectorizer(min_df=2 ,stop_words=stopwords , ngram_range=(1,2))

tfidf_transformer = TfidfTransformer()



X_train_counts = count_vect.fit_transform(train["Summary_Clean"])        

X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)



X_new_counts = count_vect.transform(test["Summary_Clean"])

X_test_tfidf = tfidf_transformer.transform(X_new_counts)
from sklearn.naive_bayes import MultinomialNB

model1 = MultinomialNB().fit(X_train_tfidf , train["sentiment"])

prediction['Multinomial'] = model1.predict_proba(X_test_tfidf)[:,1]

print("Multinomial Accuracy : {}".format(model1.score(X_test_tfidf , test["sentiment"])))
from sklearn.naive_bayes import BernoulliNB

model2 = BernoulliNB().fit(X_train_tfidf,train["sentiment"])

prediction['Bernoulli'] = model2.predict_proba(X_test_tfidf)[:,1]

print("Bernoulli Accuracy : {}".format(model2.score(X_test_tfidf , test["sentiment"])))
from sklearn import linear_model

logreg = linear_model.LogisticRegression(solver='lbfgs' , C=1000)

logistic = logreg.fit(X_train_tfidf, train["sentiment"])

prediction['LogisticRegression'] = logreg.predict_proba(X_test_tfidf)[:,1]

print("Logistic Regression Accuracy : {}".format(logreg.score(X_test_tfidf , test["sentiment"])))
type(logistic.coef_)

logistic.coef_[0]
words = count_vect.get_feature_names()

feature_coefs = pd.DataFrame(

    data = list(zip(words, logistic.coef_[0])),

    columns = ['feature', 'coef'])

feature_coefs.sort_values(by="coef")
def formatt(x):

    if x == 'neg':

        return 0

    if x == 0:

        return 0

    return 1

vfunc = np.vectorize(formatt)



cmp = 0

colors = ['b', 'g', 'y', 'm', 'k']

for model, predicted in prediction.items():

    if model not in 'Naive':

        false_positive_rate, true_positive_rate, thresholds = roc_curve(test["sentiment"].map(vfunc), predicted)

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
test.sentiment = test.sentiment.replace(["pos" , "neg"] , [True , False] )
keys = prediction.keys()

for key in ['Multinomial', 'LogisticRegression', 'Bernoulli']:

    print(" {}:".format(key))

    print(metrics.classification_report(test["sentiment"], prediction.get(key) > 0.5, target_names = ["positive", "negative"]))

    print("\n")
def test_sample(model, sample):

    sample_counts = count_vect.transform([sample])

    sample_tfidf = tfidf_transformer.transform(sample_counts)

    result = model.predict(sample_tfidf)[0]

    prob = model.predict_proba(sample_tfidf)[0]

    print("Sample estimated as %s: negative prob %f, positive prob %f" % (result.upper(), prob[0], prob[1]))



test_sample(logreg, "The product was good and easy to  use")

test_sample(logreg, "the whole experience was horrible and product is worst")

test_sample(logreg, "product is not good")
from wordcloud import WordCloud, STOPWORDS

stopwords = set(STOPWORDS)





mpl.rcParams['font.size']=12                #10 

mpl.rcParams['savefig.dpi']=100             #72 

mpl.rcParams['figure.subplot.bottom']=.1 



def show_wordcloud(data, title = None):

    wordcloud = WordCloud(

        background_color='white',

        stopwords=stopwords,

        max_words=300,

        max_font_size=40, 

        scale=3,

        random_state=1 # chosen at random by flipping a coin; it was heads

        

    ).generate(str(data))

    

    fig = plt.figure(1, figsize=(15, 15))

    plt.axis('off')

    if title: 

        fig.suptitle(title, fontsize=20)

        fig.subplots_adjust(top=2.3)



    plt.imshow(wordcloud)

    plt.show()

    

show_wordcloud(perm["Summary_Clean"])
show_wordcloud(perm["Summary_Clean"][perm.sentiment == "pos"] , title="Postive Words")
show_wordcloud(perm["Summary_Clean"][perm.sentiment == "neg"] , title="Negative Words")