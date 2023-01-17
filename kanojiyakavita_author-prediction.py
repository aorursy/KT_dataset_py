# !pip install wordcloud
# !pip install nltk

# Load libraries #

import csv
import json

import numpy
import pandas
from time import time

from matplotlib import pyplot
import seaborn as sns

import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import word_tokenize

from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import CountVectorizer
# Load all data #

authored_contents = pandas.read_csv("../input/train.csv")

unauthored_contents = pandas.read_csv("../input/train.csv")
authored_contents.head()
unauthored_contents.head()
authored_contents.shape
training_records = len(authored_contents)

author_eap, author_mws, author_hpl = authored_contents.author.value_counts()

print("Total number of authored contents: ", training_records)
print("Total number of authored contents by EAP: ", author_eap)
print("Total number of authored contents by MWS: ", author_mws)
print("Total number of authored contents by HPL: ", author_hpl)
# grab text length of each contents

authored_contents['text_length'] = authored_contents['text'].str.len()
authored_contents.head()
pyplot.figure(figsize=(14,5))
sns.countplot(authored_contents['author'],)
pyplot.xlabel('Author')
pyplot.title('Target variable distribution')
pyplot.show()
authored_contents.groupby('author').size()
# examine the same in test data

testing_records = len(unauthored_contents)

unauthored_contents['text_length'] = unauthored_contents['text'].str.len()
unauthored_contents.head()
def text_len(df):
    df['num_words'] = df['text'].apply(lambda x: len(str(x).split()))
    df['num_uniq_words'] = df['text'].apply(lambda x: len(set(str(x).split())))
    df['num_chars'] = df['text'].apply(lambda x: len(str(x)))
    df['num_stopwords'] = df['text'].apply(lambda x: len([w for w in str(x).lower().split() 
                                                          if w in set(stopwords.words('english'))]))
    df['num_punctuations'] = df['text'].apply(lambda x: len([w for w in str(x) if w in string.punctuation]))
    df['num_words_upper'] = df['text'].apply(lambda x: len([w for w in str(x).split() if w.isupper()]))
    df['num_words_title'] = df['text'].apply(lambda x: len([w for w in str(x).split() if w.istitle()]))
    df['mean_word_len'] = df['text'].apply(lambda x: numpy.mean([len(w) for w in str(x).split()]))
    df['num_character_len'] = df['text'].apply(lambda x: len(x))
text_len(authored_contents)
text_len(unauthored_contents)
def plot_heatmap(df):
    pyplot.figure(figsize=(14,6))

    pyplot.subplot(211)
    sns.heatmap(pandas.crosstab(df['author'], df['num_words']), cmap='gist_earth', xticklabels=True)
    pyplot.xlabel('Original text word count')
    pyplot.ylabel('Author')
    pyplot.tight_layout()
    pyplot.show()


    pyplot.subplot(212)
    sns.heatmap(pandas.crosstab(df['author'], df['num_uniq_words']), cmap='gist_heat', xticklabels=True)
    pyplot.xlabel('Unique text word count')
    pyplot.ylabel('Author')
    pyplot.tight_layout()
    pyplot.show()


    pyplot.subplot(212)
    sns.heatmap(pandas.crosstab(df['author'], df['num_punctuations']), cmap='gist_heat', xticklabels=True)
    pyplot.xlabel('Punctuations')
    pyplot.ylabel('Author')
    pyplot.tight_layout()
    pyplot.show()


    pyplot.subplot(212)
    sns.heatmap(pandas.crosstab(df['author'], df['mean_word_len']), cmap='gist_heat', xticklabels=False)
    pyplot.xlabel('Mean word length')
    pyplot.ylabel('Author')
    pyplot.tight_layout()
    pyplot.show()
plot_heatmap(authored_contents)
eap_documents = authored_contents[authored_contents.author == 'EAP']['text'].values
hpl_documents = authored_contents[authored_contents.author == 'HPL']['text'].values
mws_documents = authored_contents[authored_contents.author == 'MWS']['text'].values
eap_words = " ".join(eap_documents)
hpl_words = " ".join(hpl_documents)
mws_words = " ".join(mws_documents)
from wordcloud import WordCloud, STOPWORDS
pyplot.figure(figsize=(16,13))

wordcloud = WordCloud(relative_scaling = 1.0, stopwords = STOPWORDS, max_font_size= 35)
wordcloud.generate(eap_words)
pyplot.imshow(wordcloud.recolor(colormap= 'Pastel2' , random_state=17), alpha=0.98)
pyplot.axis('off')
pyplot.show()
pyplot.figure(figsize=(16,13))

wordcloud = WordCloud(relative_scaling = 1.0, stopwords = STOPWORDS, max_font_size= 35)
wordcloud.generate(hpl_words)
pyplot.imshow(wordcloud.recolor(colormap= 'Pastel2' , random_state=17), alpha=0.98)
pyplot.axis('off')
pyplot.show()
pyplot.figure(figsize=(16,13))

wordcloud = WordCloud(relative_scaling = 1.0, stopwords = STOPWORDS, max_font_size= 35)
wordcloud.generate(mws_words)
pyplot.imshow(wordcloud.recolor(colormap= 'Pastel2' , random_state=17), alpha=0.98)
pyplot.axis('off')
pyplot.show()
authored_contents['numerical_author'] = authored_contents.author.map({ 'EAP': 0, 'HPL': 1, 'MWS': 2 })
# Quick view of preprocessing

authored_contents[['text', 'author', 'numerical_author']].head()
authored_contents.head()
all_stopwords = stopwords.words('english')
ps = PorterStemmer()

def scrub_text(data_frame):
    sentences = []
    for i in data_frame.values:
#         sentence = unicode(i[1], 'utf-8')
        sentence = i[1]

        # remove all punctuations
        sentence = sentence.translate(string.punctuation)

        # break sentence into words
        array_of_words = word_tokenize(sentence)

        # removes all English stopwords
        array_of_words = [word for word in array_of_words if word.lower() not in all_stopwords]

        # singularise words in the array_of_words
        array_of_words = [ps.stem(word) for word in array_of_words]
        cleaned_sentence = ' '.join(array_of_words)

        sentences.append(cleaned_sentence)

    return sentences
# Run the #scrub_text over the text in the training and testing datasets.

training_cleaned_texts = scrub_text(authored_contents)
testing_cleaned_texts = scrub_text(unauthored_contents)
authored_contents['scrubbed_text'] = training_cleaned_texts

unauthored_contents['scrubbed_text'] = testing_cleaned_texts
# Define labels and features set

X = authored_contents['text']
Y = authored_contents['numerical_author']
from sklearn.cross_validation import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X,
                                                    Y,
                                                    test_size = 0.2,
                                                    random_state = 0)
# Show the results of the split

# Features
print("Training set has " + str(X_train.shape[0]) + " features.")
print("Testing set has " + str(X_test.shape[0]) + " features.")

# Labels
print("Training set has " + str(Y_train.shape[0]) + " labels.")
print("Testing set has " + str(Y_test.shape[0]) + " labels.")

print("\nPrinting labels set...")
print(Y_train.value_counts())

print(Y_test.value_counts())
X2_train, X2_test, Y2_train, Y2_test = train_test_split(X,
                                                        Y,
                                                        stratify = Y,
                                                        test_size = 0.2,
                                                        random_state = 42)
# Show the results of the split

# Features
print("Training set has " + str(X2_train.shape[0]) + " features.")
print("Testing set has " + str(X2_test.shape[0]) + " features.")

# Labels
print("\nTraining set has " + str(Y2_train.shape[0]) + " labels.")
print("Testing set has " + str(Y2_test.shape[0]) + " labels.")

print("\nPrinting labels set...")
print(Y_train.value_counts())

print(Y_test.value_counts())
from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer()
vectorizer.fit(X2_train)

training_vectorizer = vectorizer.transform(X2_train)
# Quick view of vectors from the texts

training_vectorizer.toarray()
print(len(vectorizer.get_feature_names()))
# run vectorizer for X2_test

testing_vectorizer = vectorizer.transform(X2_test)
# Include libraries to evaluate performances on the attached dataset

from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import MultinomialNB

kfold = KFold(n_splits = 10, random_state = 7)

model = MultinomialNB()
start = time()
results = cross_val_score(model, training_vectorizer, Y2_train, cv=kfold)
end = time()

print("Mean value: " + str(results.mean()))
print("Training time: " + str((end - start)))
from sklearn.linear_model import LogisticRegression

kfold = KFold(n_splits = 10, random_state = 7)

model = LogisticRegression()
start = time()
results = cross_val_score(model, training_vectorizer, Y2_train, cv=kfold)
end = time()

print("Mean value: " + str(results.mean()))
print("Training time: " + str(end - start))
from sklearn.svm import SVC

kfold = KFold(n_splits = 10, random_state = 7)

model = SVC()
start = time()
results = cross_val_score(model, training_vectorizer, Y2_train, cv=kfold)
end = time()

print("Mean value: " + str(results.mean()))
print("Training time: " + str(end - start))
from sklearn.linear_model import SGDClassifier

kfold = KFold(n_splits = 10, random_state = 7)

model = SGDClassifier()
start = time()
results = cross_val_score(model, training_vectorizer, Y2_train, cv=kfold)
end = time()

print("Mean value: " + str(results.mean()))
print("Training time: " + str(end - start))
from xgboost import XGBClassifier

model = XGBClassifier()
start = time()
results = cross_val_score(model, training_vectorizer, Y2_train, cv=kfold)
end = time()

print("Mean value: " + str(results.mean())).format(results.mean())
print("Training time: " + str(end - start))
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import fbeta_score, make_scorer
from sklearn.naive_bayes import MultinomialNB
model = MultinomialNB()

parameters = { 'alpha': [0.01, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0] }

scorer = make_scorer(fbeta_score, beta=0.5)

grid_obj = GridSearchCV(model, parameters)

grid_fit = grid_obj.fit(training_vectorizer, Y2_train)

# Get the estimator
best_clf = grid_fit.best_estimator_

grid_fit.best_params_
mnb = MultinomialNB(alpha = 0.5)
mnb.fit(training_vectorizer, Y2_train)
mnb.predict_proba(training_vectorizer)
mnb.feature_log_prob_
Y_train_prediction = mnb.predict(training_vectorizer)
Y_test_prediction = mnb.predict(testing_vectorizer)
# calculate score for predicted data against testing data
from sklearn import metrics

# compare predicted resultset with the test set
metrics.accuracy_score(Y2_test, Y_test_prediction)
# calculate score for predicted data against training data

metrics.accuracy_score(Y2_train, Y_train_prediction)
# Calculate confusion matrix

metrics.confusion_matrix(Y2_test, Y_test_prediction)
# calculate predicted probabilities for X_test_dtm
y_pred_prob = mnb.predict_proba(testing_vectorizer)
y_pred_prob[:10]
print(classification_report(Y2_train, Y_train_prediction, target_names=['EAP', 'HPL', 'MWS']))
fpr, tpr, thresholds = metrics.roc_curve(Y2_train, Y_train_prediction, pos_label = 1)

print("Multinomial naive bayes AUC: " + str(metrics.auc(fpr, tpr)))
# vectorise the unauthored_contents

unpredicted_texts = unauthored_contents['text']

unpredicted_texts_vectorizer = vectorizer.transform(unpredicted_texts)
unpredicted_texts_vectorizer
unpredicted_texts_prediction = mnb.predict(unpredicted_texts_vectorizer)
# calculate predicted probabilities for X_test_dtm
predicted_prob = mnb.predict_proba(unpredicted_texts_vectorizer)
predicted_prob[:10]
unpredicted_texts_prediction
len(unauthored_contents)
numerical_authors = pandas.DataFrame(unpredicted_texts_prediction, columns=['num_author'])

predicted_unauthored_contents = pandas.concat([unauthored_contents, numerical_authors], axis=1)
predicted_unauthored_contents['author'] = predicted_unauthored_contents.num_author.map({ 0: 'EAP', 1: 'HPL', 2: 'MWS' })
predicted_unauthored_contents.groupby('author').size()
predicted_unauthored_contents.head()
# Plotting over the test.csv that we ran the prediction above to study the analysis.

plot_heatmap(predicted_unauthored_contents)
# Comparing the heatmap that we plotted above using `authored_contents`

plot_heatmap(authored_contents)
predicted_eap_documents = predicted_unauthored_contents[predicted_unauthored_contents.author == 'EAP']['text'].values
predicted_hpl_documents = predicted_unauthored_contents[predicted_unauthored_contents.author == 'HPL']['text'].values
predicted_mws_documents = predicted_unauthored_contents[predicted_unauthored_contents.author == 'MWS']['text'].values
predicted_eap_words = " ".join(predicted_eap_documents)
predicted_hpl_words = " ".join(predicted_hpl_documents)
predicted_mws_words = " ".join(predicted_mws_documents)
pyplot.figure(figsize=(16,13))

wordcloud = WordCloud(relative_scaling = 1.0, stopwords = STOPWORDS, max_font_size= 35)
wordcloud.generate(predicted_eap_words)
pyplot.imshow(wordcloud.recolor(colormap= 'Pastel2' , random_state=17), alpha=0.98)
pyplot.axis('off')
pyplot.show()
pyplot.figure(figsize=(16,13))

wordcloud = WordCloud(relative_scaling = 1.0, stopwords = STOPWORDS, max_font_size= 35)
wordcloud.generate(predicted_hpl_words)
pyplot.imshow(wordcloud.recolor(colormap= 'Pastel2' , random_state=17), alpha=0.98)
pyplot.axis('off')
pyplot.show()
pyplot.figure(figsize=(16,13))

wordcloud = WordCloud(relative_scaling = 1.0, stopwords = STOPWORDS, max_font_size= 35)
wordcloud.generate(predicted_mws_words)
pyplot.imshow(wordcloud.recolor(colormap= 'Pastel2' , random_state=17), alpha=0.98)
pyplot.axis('off')
pyplot.show()
## Closing notes to come...
