import pandas as pd
import re, nltk
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline

import warnings
warnings.filterwarnings('ignore')
newsDF = pd.read_json("../input/News_Category_Dataset.json", lines = True)
newsDF.columns
newsDF['final text'] = newsDF['headline'] + newsDF['short_description']
newsDF.head(2)
len(newsDF[newsDF['short_description'] == ''])
newsDF.drop(newsDF[newsDF['short_description'] == '' ].index, inplace=True)
figure, ax = plt.subplots(1, 1, figsize = (20, 5))

ax = sns.countplot(newsDF['category'])
ax.set_title("Category Counts")
ax.set_xlabel("Category")
# Manipulate the labels to make them more readable
ax.set_xticklabels([x for x in ax.get_xticklabels()], rotation=80)
plt.show()
figure, ax = plt.subplots(1, 1, figsize = (10, 10))
newsDF['category'].value_counts().plot.pie()
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(newsDF['final text'], 
                                                    newsDF['category'], 
                                                    test_size = 0.33)
X_train = np.array(X_train)
y_train = np.array(y_train)
X_test = np.array(X_test)
y_test = np.array(y_test)
train_set_len = len(X_train)
test_set_len = len(X_test)
from nltk.stem import WordNetLemmatizer
from nltk import word_tokenize
from nltk.corpus import stopwords

lemmatizer = WordNetLemmatizer()
def process_sentences(main_text):
    headlines_without_numbers = re.sub('[^a-zA-Z]', ' ', main_text)
    words = word_tokenize(headlines_without_numbers.lower())
    stop_words_english = set(stopwords.words('english'))
    final_words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words_english]
    return(' '.join(final_words))
finalHeadlineTrain = []
finalHeadlineTest = []
for i in range(0, train_set_len):
    finalHeadlineTrain.append(process_sentences(X_train[i]))
for i in range(0, test_set_len):
    finalHeadlineTest.append(process_sentences(X_test[i]))
from sklearn.feature_extraction.text import CountVectorizer

vectorize = CountVectorizer(analyzer = "word", max_df=0.5,min_df=2, ngram_range=(1, 2), max_features=10000)

bagOfWords_train = vectorize.fit_transform(finalHeadlineTrain)
X_train = bagOfWords_train.toarray()

bagOfWords_test = vectorize.transform(finalHeadlineTest)
X_test = bagOfWords_test.toarray()
# A function to find train and test set accuracy
def report_accuracy(trained_clf):
    train_score = trained_clf.score(X_train, y_train)
    test_score = trained_clf.score(X_test, y_test)
    print("Training set accuracy score is: {}".format(train_score))
    print("Test set accuracy score is: {}".format(test_score))
from sklearn.linear_model import LogisticRegression
from sklearn import metrics

logisticModel = LogisticRegression(n_jobs = -1)
logisticModel.fit(X_train, y_train)

report_accuracy(logisticModel)
# prediction = logisticModel.predict(X_test)
# accuracyVal = metrics.accuracy_score(prediction, y_test) * 100
# print(accuracyVal)
from sklearn.svm import LinearSVC

modelSVM = LinearSVC()
modelSVM.fit(X_train,y_train)

report_accuracy(modelSVM)
# y_predict = model.predict(X_test)
# accuracy = metrics.accuracy_score(y_test,y_predict)*100
# print(accuracy)
newsDF['authors'].nunique()
newsDF.drop(newsDF[newsDF['authors'] == '' ].index, inplace=True)
authorNewsCount = dict()
for nCat in newsDF.iterrows():
    if nCat[1]['category'] not in authorNewsCount.keys():
        authorNewsCount[nCat[1]['category']] = {}
        
    if nCat[1]['authors'] not in authorNewsCount[nCat[1]['category']].keys():
        authorNewsCount[nCat[1]['category']][nCat[1]['authors']] = 1
    
    else:
        authorNewsCount[nCat[1]['category']][nCat[1]['authors']] += 1
authorsSpecificCount = newsDF.authors.value_counts()
sumOfMajorContributors = 0
sumOfMajorContributions = 0
authorCount = newsDF.authors.nunique()
for authRole in authorsSpecificCount:
    if authRole < 100:
        break
    sumOfMajorContributors += 1
    sumOfMajorContributions += authRole
print("Count of Major Contributors is: " + str(sumOfMajorContributors))
print("Total number of Contributors is: " + str(authorCount))

print("% of major contributors: " + str(sumOfMajorContributors/authorCount*100))

print("% of the news contributed by major contributors: " + str(sumOfMajorContributions/newsDF.shape[0]*100))
lemmatizer = WordNetLemmatizer()
def process_sentences_with_author(main_text):
    headlines_without_numbers = re.sub('[^a-zA-Z]', ' ', main_text[0])
    author_name = [x for x in main_text[1].lower().replace('and',',').replace(' ', '').split(',') if x != '']
    words = word_tokenize(headlines_without_numbers.lower())
    stop_words_english = set(stopwords.words('english'))
    final_words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words_english]
    return(' '.join(final_words+author_name))
X_train, X_test, y_train, y_test = train_test_split(newsDF[['final text', 'authors']], 
                                                    newsDF['category'], 
                                                    test_size = 0.33)
X_train = np.array(X_train)
y_train = np.array(y_train)
X_test = np.array(X_test)
y_test = np.array(y_test)
train_set_len = len(X_train)
test_set_len = len(X_test)

finalHeadlineAuthorTrain = []
finalHeadlineAuthorTest = []
for i in range(0, train_set_len):
    finalHeadlineAuthorTrain.append(process_sentences_with_author(X_train[i]))
for i in range(0, test_set_len):
    finalHeadlineAuthorTest.append(process_sentences_with_author(X_test[i]))
from sklearn.feature_extraction.text import CountVectorizer

vectorize = CountVectorizer(analyzer = "word", max_df=0.5,min_df=2, ngram_range=(1, 2), max_features=10000)

bagOfWords_train = vectorize.fit_transform(finalHeadlineAuthorTrain)
X_train = bagOfWords_train.toarray()

bagOfWords_test = vectorize.transform(finalHeadlineAuthorTest)
X_test = bagOfWords_test.toarray()
# A function to find train and test set accuracy
def report_accuracy(trained_clf):
    train_score = trained_clf.score(X_train, y_train)
    test_score = trained_clf.score(X_test, y_test)
    print("Training set accuracy score is: {}".format(train_score))
    print("Test set accuracy score is: {}".format(test_score))
from sklearn.linear_model import LogisticRegression
from sklearn import metrics

logisticModel = LogisticRegression(n_jobs = -1)
logisticModel.fit(X_train, y_train)

report_accuracy(logisticModel)
# prediction = logisticModel.predict(X_test)
# accuracyVal = metrics.accuracy_score(prediction, y_test) * 100
# print(accuracyVal)
from sklearn.svm import LinearSVC

modelSVM = LinearSVC()
modelSVM.fit(X_train,y_train)

report_accuracy(modelSVM)
# y_predict = model.predict(X_test)
# accuracy = metrics.accuracy_score(y_test,y_predict)*100
# print(accuracy)
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizerTf = TfidfVectorizer(analyzer = 'word', max_features = 30000)

bagOfWords_train_tf = vectorize.fit_transform(finalHeadlineAuthorTrain)
X_train = bagOfWords_train_tf.toarray()

bagOfWords_test_tf = vectorize.transform(finalHeadlineAuthorTest)
X_test = bagOfWords_test_tf.toarray()
from sklearn.linear_model import LogisticRegression
from sklearn import metrics

logisticModel = LogisticRegression(n_jobs = -1)
logisticModel.fit(X_train, y_train)

report_accuracy(logisticModel)
# prediction = logisticModel.predict(X_test)
# accuracyVal = metrics.accuracy_score(prediction, y_test)*100
# print(accuracyVal)
from sklearn.svm import LinearSVC

modelSVM = LinearSVC()
modelSVM.fit(X_train,y_train)

report_accuracy(modelSVM)
# y_predict = model.predict(X_test)
# accuracy = metrics.accuracy_score(y_test,y_predict)*100
# print(accuracy)
from nltk import word_tokenize, PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

def tokenize(s, lemmatize=True, decode=False):
    try:
        if decode:
            s = s.decode("utf-8")
        tokens = word_tokenize(s.lower())
    except LookupError:
        nltk.download('punkt')
        tokenize(s)
    

    ignored = stopwords.words("english") + [punct for punct in string.punctuation]
    clean_tokens = [token for token in tokens if token not in ignored]
    
    if lemmatize:
        lemmatizer = WordNetLemmatizer()
        return [lemmatizer.lemmatize(token) for token in clean_tokens]
    return clean_tokens
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
import string
# Bag of Words Representation using our own tokenizer.

newsDF['testCol'] = newsDF['final text'] + newsDF['authors']

vectorizer = CountVectorizer(lowercase=False, tokenizer=tokenize)
x = vectorizer.fit_transform(newsDF['testCol'])

# Create numerical labels.
encoder = LabelEncoder()
y = encoder.fit_transform(newsDF['category'])

# Let's keep this in order to interpret our results later,
encoder_mapping = dict(zip(encoder.transform(encoder.classes_), encoder.classes_))

# Split into a training and test set. Classifiers will be trained on the former and the final
# results will be reported on the latter.
seed = 42
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=seed)
# A function to find train and test set accuracy
def report_accuracy(trained_clf):
    train_score = trained_clf.score(x_train, y_train)
    test_score = trained_clf.score(x_test, y_test)
    print("Training set accuracy score is: {}".format(train_score))
    print("Test set accuracy score is: {}".format(test_score))
from sklearn.linear_model import LogisticRegression
from sklearn import metrics

logisticModel = LogisticRegression(n_jobs = -1)
logisticModel.fit(x_train, y_train)

report_accuracy(logisticModel)
from sklearn.svm import LinearSVC

modelSVM = LinearSVC()
modelSVM.fit(x_train,y_train)

report_accuracy(modelSVM)
# y_predict = model.predict(X_test)
# accuracy = metrics.accuracy_score(y_test,y_predict)*100
# print(accuracy)
