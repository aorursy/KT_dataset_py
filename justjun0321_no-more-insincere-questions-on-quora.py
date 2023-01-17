import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns



plt.style.use('ggplot')



from wordcloud import WordCloud, ImageColorGenerator

from PIL import Image



import datetime

from string import punctuation
train = pd.read_csv('../input/quora-insincere-questions-classification/train.csv')
train.head()
sum(train.target)/len(train)
train = train.drop('qid',axis=1)
# preprocess text for further use

train['question_text'] = train.question_text.apply(lambda x: x.lower())

train['question_text'] = train.question_text.apply(lambda x: ''.join([c for c in x if c not in punctuation]))
train['question_length'] = train.question_text.apply(lambda x: len(x))
print('The max length of question is:',train.question_length.max())

print('The minimum length of question is:',train.question_length.min())

print('The mean length of question is:',train.question_length.mean())

print('The max standard deviation of question is:',train.question_length.std())
train.question_length.hist(bins=50)

plt.title('The distribution of length of questions')

plt.axvline(np.mean(train.question_length),color='y')
train['question_length_scaled'] = train.question_length.apply(lambda x: np.log(x+1))
train.describe()
train.question_length_scaled.hist(bins=50)
sns.boxplot(train.target,train.question_length_scaled)

plt.title('The distribution of length of target or not')
Q = np.array(Image.open('../input/quora-logo1/quora-logo-rubber-stamp.png'))
np.random.seed(321)

sns.set(rc={'figure.figsize':(14,8)})

reviews = ' '.join(train['question_text'].tolist())



wordcloud = WordCloud(mask=Q,background_color="white").generate(reviews)

plt.figure()

plt.imshow(wordcloud, interpolation="bilinear")

plt.axis("off")

plt.margins(x=0, y=0)

plt.title('Questions',size=20)

plt.show()
train_1 = train[train.target == 1]

train_0 = train[train.target == 0]
reviews = ' '.join(train_1['question_text'].tolist())



wordcloud = WordCloud(mask=Q,background_color="white").generate(reviews)

plt.figure()

plt.imshow(wordcloud, interpolation="bilinear")

plt.axis("off")

plt.margins(x=0, y=0)

plt.title('Target 1',size=20)

plt.show()
reviews = ' '.join(train_0['question_text'].tolist())



wordcloud = WordCloud(mask=Q,background_color="white").generate(reviews)

plt.figure()

plt.imshow(wordcloud, interpolation="bilinear")

plt.axis("off")

plt.margins(x=0, y=0)

plt.title('Target 0',size=20)

plt.show()
from collections import Counter



text = ' '.join(train['question_text'].tolist())

question_word = text.split(' ')

all_question = ' '.join(question_word)

words = all_question.split()



# words wrong datatype

counts = Counter(words)

vocab = sorted(counts, key=counts.get, reverse=True)

vocab_to_int = {word: ii for ii, word in enumerate(vocab, 1)}



questions_ints = []

for questions in question_word:

    questions_ints.append([vocab_to_int[word] for word in questions.split()])
print('Unique words: ', len((vocab_to_int)))
counts.most_common(20)
text = ' '.join(train_0['question_text'].tolist())

question_word = text.split(' ')

all_question = ' '.join(question_word)

words = all_question.split()



# words wrong datatype

counts = Counter(words)

vocab = sorted(counts, key=counts.get, reverse=True)

vocab_to_int = {word: ii for ii, word in enumerate(vocab, 1)}



questions_ints = []

for questions in question_word:

    questions_ints.append([vocab_to_int[word] for word in questions.split()])
counts.most_common(20)
text = ' '.join(train_1['question_text'].tolist())

question_word = text.split(' ')

all_question = ' '.join(question_word)

words = all_question.split()



# words wrong datatype

counts = Counter(words)

vocab = sorted(counts, key=counts.get, reverse=True)

vocab_to_int = {word: ii for ii, word in enumerate(vocab, 1)}



questions_ints = []

for questions in question_word:

    questions_ints.append([vocab_to_int[word] for word in questions.split()])
counts.most_common(20)
train.head()
from nltk.corpus import stopwords 

from nltk.tokenize import word_tokenize 



stop_words = set(stopwords.words('english')) 



train['question_text'] = train.question_text.apply(lambda x: word_tokenize(x))



train['question_text'] = train.question_text.apply(lambda x: [w for w in x if w not in stop_words])
train.head()
train['question_text'] = train.question_text.apply(lambda x: ' '.join(x))
train.head()
text = ' '.join(train['question_text'].tolist())

question_word = text.split(' ')

all_question = ' '.join(question_word)

words = all_question.split()



# words wrong datatype

counts = Counter(words)

vocab = sorted(counts, key=counts.get, reverse=True)

vocab_to_int = {word: ii for ii, word in enumerate(vocab, 1)}



questions_ints = []

for questions in question_word:

    questions_ints.append([vocab_to_int[word] for word in questions.split()])
print('Unique words: ', len((vocab_to_int)))
counts.most_common(20)
train['question_text'] = train['question_text'].apply(lambda x: x.replace('’', ""))
train_1 = train[train.target == 1]

train_0 = train[train.target == 0]
text = ' '.join(train_0['question_text'].tolist())

question_word = text.split(' ')

all_question = ' '.join(question_word)

words = all_question.split()



# words wrong datatype

counts = Counter(words)

vocab = sorted(counts, key=counts.get, reverse=True)

vocab_to_int = {word: ii for ii, word in enumerate(vocab, 1)}



questions_ints = []

for questions in question_word:

    questions_ints.append([vocab_to_int[word] for word in questions.split()])
counts.most_common(20)
text = ' '.join(train_1['question_text'].tolist())

question_word = text.split(' ')

all_question = ' '.join(question_word)

words = all_question.split()



# words wrong datatype

counts = Counter(words)

vocab = sorted(counts, key=counts.get, reverse=True)

vocab_to_int = {word: ii for ii, word in enumerate(vocab, 1)}



questions_ints = []

for questions in question_word:

    questions_ints.append([vocab_to_int[word] for word in questions.split()])
counts.most_common(20)
dup_words = ['people','would','get','like','india','think', 'many']



train['question_text'] = train.question_text.apply(lambda x: word_tokenize(x))



train['question_text'] = train.question_text.apply(lambda x: [w for w in x if w not in dup_words])



train['question_text'] = train.question_text.apply(lambda x: ' '.join(x))



train_1 = train[train.target == 1]

train_0 = train[train.target == 0]
text = ' '.join(train_0['question_text'].tolist())

question_word = text.split(' ')

all_question = ' '.join(question_word)

words = all_question.split()



# words wrong datatype

counts = Counter(words)

vocab = sorted(counts, key=counts.get, reverse=True)

vocab_to_int = {word: ii for ii, word in enumerate(vocab, 1)}



questions_ints = []

for questions in question_word:

    questions_ints.append([vocab_to_int[word] for word in questions.split()])
counts.most_common(20)
text = ' '.join(train_1['question_text'].tolist())

question_word = text.split(' ')

all_question = ' '.join(question_word)

words = all_question.split()



# words wrong datatype

counts = Counter(words)

vocab = sorted(counts, key=counts.get, reverse=True)

vocab_to_int = {word: ii for ii, word in enumerate(vocab, 1)}



questions_ints = []

for questions in question_word:

    questions_ints.append([vocab_to_int[word] for word in questions.split()])
counts.most_common(20)
from sklearn.feature_extraction.text import CountVectorizer

from sklearn.model_selection import train_test_split



from sklearn.feature_selection import SelectKBest

from sklearn.feature_selection import chi2



X_train, X_test, y_train, y_test = train_test_split(train["question_text"], train['target'], test_size=0.33

                                    ,random_state=53)



# Initialize a CountVectorizer object: count_vectorizer

count_vectorizer = CountVectorizer(stop_words="english")



# Transform the training data using only the 'text' column values: count_train 

count_train = count_vectorizer.fit_transform(X_train)



y_train = np.asarray(y_train.values)



ch2 = SelectKBest(chi2, k = 300)



X_new = ch2.fit_transform(count_train, y_train)



# Transform the test data using only the 'text' column values: count_test 

count_test = count_vectorizer.transform(X_test)



X_test_new = ch2.transform(X=count_test)
from sklearn.feature_extraction.text import TfidfVectorizer



# Initialize a TfidfVectorizer object: tfidf_vectorizer

tfidf_vectorizer = TfidfVectorizer(stop_words="english", max_df=0.7)



# Transform the training data: tfidf_train 

tfidf_train = tfidf_vectorizer.fit_transform(X_train)



# Transform the test data: tfidf_test 

tfidf_test = tfidf_vectorizer.transform(X_test)
from sklearn.ensemble import RandomForestClassifier

from sklearn import metrics



clf = RandomForestClassifier()

# Fit the classifier to the training data

clf.fit(X_new, y_train)



# Create the predicted tags: pred

pred = clf.predict(X_test_new)



# Calculate the accuracy score: score

score = metrics.accuracy_score(y_test, pred)

print('Accuracy is:',score)

f1 = metrics.f1_score(y_test, pred)

print('F score is:',f1)
sns.heatmap(metrics.confusion_matrix(pred,y_test),annot=True,fmt='2.0f')
test = pd.read_csv('../input/quora-insincere-questions-classification/test.csv')
test = test.drop('qid',axis=1)



test['question_text'] = test.question_text.apply(lambda x: x.lower())

test['question_text'] = test.question_text.apply(lambda x: ''.join([c for c in x if c not in punctuation]))
test['question_text'] = test.question_text.apply(lambda x: word_tokenize(x))



test['question_text'] = test.question_text.apply(lambda x: [w for w in x if w not in stop_words])



test['question_text'] = test.question_text.apply(lambda x: [w for w in x if w not in dup_words])
test['question_text'] = test.question_text.apply(lambda x: ' '.join(x))
# Transform the training data using only the 'text' column values: count_train 

count = count_vectorizer.transform(test.question_text)



X = ch2.transform(count)
y_pred = clf.predict(X)
submission = pd.read_csv('../input/quora-insincere-questions-classification/sample_submission.csv')
submission['prediction'] = y_pred
submission.to_csv('submission.csv')