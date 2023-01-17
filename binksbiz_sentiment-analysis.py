import numpy as np

import pandas as pd

import matplotlib.pyplot as plt



from sklearn.model_selection import train_test_split

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

from sklearn.metrics import confusion_matrix

from sklearn import metrics

from sklearn.metrics import roc_curve, auc
dataset = pd.read_csv("../input/MrTrumpSpeeches.csv", header=0, delimiter="\~", quoting=3, engine='python')
sent_th = 10

dataset['sentiment'] = (dataset["like_count"]/dataset["dislike_count"]).apply(lambda r: 'popular' if r > sent_th else 'unpopular')



#print(len(dataset[dataset.sentiment == 1]))

#print(len(dataset[dataset.sentiment == 0]))

#print(dataset.head(3))

dataset[dataset.sentiment == 'popular'].head(5)
import re

import string

import nltk



def cleanup(sentence):

    sentence = sentence.lower()

    sentence = re.sub("[\[].*?[\]]", "", sentence)

    return sentence



dataset['subtitles_clean'] = dataset['subtitles'].apply(cleanup)



train, test = train_test_split(dataset, test_size=0.2)

print("%d items in training data, %d in test data" % (len(train), len(test)))



dataset.head(3)

#print(dataset['subtitles_clean'][0])

#print(dataset['subtitles'][0])
count_vect = CountVectorizer(min_df = 1, max_features = 500)

X_train_counts = count_vect.fit_transform(train['subtitles_clean'])



X_train_counts = X_train_counts.toarray()

print(X_train_counts.shape)



# Take a look at the words in the vocabulary

vocab = count_vect.get_feature_names()

print(vocab)
# Sum up the counts of each vocabulary word

dist = np.sum(X_train_counts, axis=0)



# For each, print the vocabulary word and the number of times it 

# appears in the training set

for tag, count in zip(vocab, dist):

    print(count, tag)
count_vect = CountVectorizer(min_df = 1, max_features = 500, ngram_range = (1, 4))

X_train_counts = count_vect.fit_transform(train['subtitles_clean'])



X_train_counts_array = X_train_counts.toarray()

print(X_train_counts_array.shape)



tfidf_transformer = TfidfTransformer()

X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)



X_new_counts = count_vect.transform(test['subtitles_clean'])

X_test_tfidf = tfidf_transformer.transform(X_new_counts)



y_train = train['sentiment']

y_test = test['sentiment']



prediction = dict()
from sklearn.naive_bayes import MultinomialNB

model = MultinomialNB().fit(X_train_tfidf, y_train)

prediction['Multinomial'] = model.predict(X_test_tfidf)
from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression(C=1e5)

logreg_result = logreg.fit(X_train_tfidf, y_train)

prediction['Logistic'] = logreg.predict(X_test_tfidf)
from sklearn.ensemble import RandomForestClassifier

forest = RandomForestClassifier(n_estimators = 100)



model = forest.fit(X_train_tfidf, y_train)

prediction['RandomForest'] = model.predict(X_test_tfidf)
def formatt(x):

    if x == 'unpopular':

        return 0

    return 1

vfunc = np.vectorize(formatt)



cmp = 0

colors = ['b', 'g', 'y', 'm', 'k', 'r', 'c']

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
print(metrics.classification_report(y_test, prediction['Multinomial'], target_names = ['1', '0']))

print(metrics.classification_report(y_test, prediction['Logistic'], target_names = ['1', '0']))

print(metrics.classification_report(y_test, prediction['RandomForest'], target_names = ['1', '0']))
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

    print("Sample estimated as %s: popular prob %f, unpopular prob %f" % (result, prob[0], prob[1]))



test_sample(logreg, "our country")

test_sample(logreg, "thank you")

test_sample(logreg, "we are")

test_sample(logreg, "you you")