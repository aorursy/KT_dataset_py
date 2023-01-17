import seaborn as sns

import matplotlib.pyplot as plt

sns.set_style('whitegrid')



import sklearn

from sklearn import preprocessing

from sklearn.model_selection import train_test_split

from sklearn.utils import resample

import string





from sklearn.dummy import DummyClassifier

from sklearn.naive_bayes import MultinomialNB

from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.model_selection import GridSearchCV

from sklearn.svm import SVC

from sklearn.linear_model import SGDClassifier

import xgboost as xgb

from sklearn.pipeline import Pipeline

from sklearn.feature_extraction.text import TfidfTransformer

from sklearn.feature_extraction.text import CountVectorizer

from sklearn import metrics 

from sklearn.metrics import classification_report

from sklearn.metrics import accuracy_score



import nltk

from nltk import TreebankWordTokenizer

from nltk.tokenize.treebank import TreebankWordDetokenizer

from nltk import SnowballStemmer





from nltk.stem import WordNetLemmatizer

from wordcloud import WordCloud

from nltk.corpus import stopwords

from nltk.tokenize import word_tokenize

from nltk import SnowballStemmer

from sklearn.metrics import classification_report

from sklearn.metrics import accuracy_score

from sklearn.metrics import log_loss





import pandas as pd

import numpy as np
train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')

# saves the index of the train and test data set to be able to seperate them accordingly after cleaning

ntrain = train.shape[0]

ntest = test.shape[0]



# merges the two dataframes to create one for data cleaning

all_data = pd.concat((train, test), sort=False).reset_index(drop=True)





# drops the id column as it is not necessary for our predictions

all_data.drop("id", axis=1, inplace=True)

all_data.head()
grouped_wordclouds = all_data.groupby('type').sum()



grouped_wordclouds

wordcloud = WordCloud(max_font_size=50, max_words=100, background_color="white").generate(grouped_wordclouds['posts'][11])

plt.figure()

plt.imshow(wordcloud, interpolation="bilinear")

plt.axis("off")

plt.show()
wordcloud = WordCloud(max_font_size=50, max_words=100, background_color="white").generate(grouped_wordclouds['posts'][8])

plt.figure()

plt.imshow(wordcloud, interpolation="bilinear")

plt.axis("off")

plt.show()
plt.figure(figsize=(10,4))

fig = all_data.type.value_counts().plot(kind='bar')

fig.set_title('Personality Type Frequency')

fig.set_xlabel('Type')

fig.set_ylabel('Total Posts')
#all_data = []

#for i, row in all_mbti.iterrows():

#    for post in row['posts'].split('|||'):

#        all_data.append([row['type'], post])

#all_data = pd.DataFrame(all_data, columns=['type', 'posts'])
print ("There are {} words in the combination of all posts.".format(len(grouped_wordclouds['posts'].sum())))
# remove url's

pattern_url = r'http[s]?://(?:[A-Za-z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9A-Fa-f][0-9A-Fa-f]))+'

subs_url = r'url-web'

all_data['posts'] = all_data['posts'].replace(to_replace=pattern_url, value=subs_url, regex=True)



# make lower case

all_data['posts'] = all_data['posts'].str.lower()

# remove punctuation and numbers

def remove_punctuation_numbers(post):

    """Removes punctuation and numbers from the list



    Parameters:

    post(object): The string who's punctuation and numbers will be removed



    Returns:

    post(post): returns the post without the punctuation and numbers

    """

    punc_numbers = string.punctuation + '0123456789'

    return ''.join([l for l in post if l not in punc_numbers])





all_data['posts'] = all_data['posts'].apply(remove_punctuation_numbers)

tokeniser = TreebankWordTokenizer()

all_data['tokens'] = all_data['posts'].apply(tokeniser.tokenize)

stemmer = SnowballStemmer('english')





def train_stemmer(words, stemmer):

    """Transforms to the root word in the list, that is, and

    removes common word endings from English words



    Parameters:

    words(array): The list which is to be stemmed

    stemmer(object): converts the words to stems



    Returns:

    words(words, stemmer): returns a list of stemmed words

    """

    return [stemmer.stem(word) for word in words]





all_data['stem'] = all_data['tokens'].apply(train_stemmer, args=(stemmer, ))

lemmatizer = WordNetLemmatizer()





def train_lemma(words, lemmatizer):

    """Returns the given forms of word as identified by the word's dictionary form



    Parameters:

    words(array): The list which is to be stemmed

    lemmatizer(object): converts the words to lemmas



    Returns:

    words(words, lemmatizer): returns a list of lemmatized words



    """

    return [lemmatizer.lemmatize(word) for word in words]

all_data['lemma'] = all_data['tokens'].apply(train_lemma, args=(lemmatizer, ))

# Instantiate the detokenizer

detokenizer = TreebankWordDetokenizer()



# detokenize the lemmatized column

all_data['detoken'] = all_data['lemma'].apply(lambda x: detokenizer.detokenize(x))

all_data.head()
# Undo the concatenation we did earlier

train = all_data[:ntrain]

test = all_data[ntrain:]

# Separate input features and target

X = train.detoken

y = train.type



# setting up testing and training sets

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)



# concatenate our training data back together

X = pd.concat([X_train, y_train], axis=1)



# separate minority and majority classes

ESTP = X[X.type == 'ESTP']

ESTJ = X[X.type == 'ESTJ']

ESFP = X[X.type == 'ESFP']

ESFJ = X[X.type == 'ESFJ']

ENTP = X[X.type == 'ENTP']

ENTJ = X[X.type == 'ENTJ']

ENFP = X[X.type == 'ENFP']

ENFJ = X[X.type == 'ENFJ']

INFP = X[X.type == 'INFP']  # majority class



# upsample minority

mbti_upsampledESTP = resample(ESTP,

                              replace=True,  # sample with replacement

                              n_samples=len(INFP),  # match number in majority class

                              random_state=27)  # reproducible results





mbti_upsampledESFP = resample(ESFP,

                              replace=True,  # sample with replacement

                              n_samples=len(INFP),  # match number in majority class

                              random_state=27)  # reproducible results

mbti_upsampledESFJ = resample(ESFJ,

                              replace=True,  # sample with replacement

                              n_samples=len(INFP),  # match number in majority class

                              random_state=27)  # reproducible results



mbti_upsampledESTJ = resample(ESTJ,

                              replace=True,  # sample with replacement

                              n_samples=len(INFP),  # match number in majority class

                              random_state=27)  # reproducible results



mbti_upsampledENFJ = resample(ENFJ,

                              replace=True,  # sample with replacement

                              n_samples=len(INFP),  # match number in majority class

                              random_state=27)  # reproducible results



mbti_upsampledENTJ = resample(ENTJ,

                              replace=True,  # sample with replacement

                              n_samples=len(INFP),  # match number in majority class

                              random_state=27)  # reproducible results



mbti_upsampledENTP = resample(ENTP,

                              replace=True,  # sample with replacement

                              n_samples=len(INFP),  # match number in majority class

                              random_state=27)  # reproducible results



mbti_upsampledENFP = resample(ENFP,

                              replace=True,  # sample with replacement

                              n_samples=len(INFP),  # match number in majority class

                              random_state=27)  # reproducible results



# combine majority and upsampled minority

upsampled_E = pd.concat([INFP, mbti_upsampledESTP,mbti_upsampledESTJ,mbti_upsampledESFP,mbti_upsampledESFJ,mbti_upsampledENFJ,mbti_upsampledENTJ,mbti_upsampledENTP,mbti_upsampledENFP])





# check new class counts

upsampled_E.type.value_counts()

upsampled_E.head()
# separate minority and majority classes

ISTP = X[X.type == 'ISTP']

ISTJ = X[X.type == 'ISTJ']

ISFP = X[X.type == 'ISFP']

ISFJ = X[X.type == 'ISFJ']

INTP = X[X.type == 'INTP']

INTJ = X[X.type == 'INTJ']

INFJ = X[X.type == 'INFJ']

INFP = X[X.type == 'INFP']



# upsample minority

mbti_upsampledISTP = resample(ISTP,

                              replace=True,  # sample with replacement

                              n_samples=len(INFP),  # match number in majority class

                              random_state=27)  # reproducible results





mbti_upsampledISFP = resample(ISFP,

                              replace=True,  # sample with replacement

                              n_samples=len(INFP),  # match number in majority class

                              random_state=27)  # reproducible results

mbti_upsampledISFJ = resample(ISFJ,

                              replace=True,  # sample with replacement

                              n_samples=len(INFP),  # match number in majority class

                              random_state=27)  # reproducible results



mbti_upsampledISTJ = resample(ISTJ,

                              replace=True,  # sample with replacement

                              n_samples=len(INFP),  # match number in majority class

                              random_state=27)  # reproducible results



mbti_upsampledINFJ = resample(INFJ,

                              replace=True,  # sample with replacement

                              n_samples=len(INFP),  # match number in majority class

                              random_state=27)  # reproducible results



mbti_upsampledINTJ = resample(INTJ,

                              replace=True,  # sample with replacement

                              n_samples=len(INFP),  # match number in majority class

                              random_state=27)  # reproducible results



mbti_upsampledINTP = resample(INTP,

                              replace=True,  # sample with replacement

                              n_samples=len(INFP),  # match number in majority class

                              random_state=27)  # reproducible results





# upsampled minority

upsampled_I = pd.concat([mbti_upsampledISTP, mbti_upsampledISTJ, mbti_upsampledISFP, mbti_upsampledISFJ, mbti_upsampledINFJ, mbti_upsampledINTJ, mbti_upsampledINTP])



# check new class counts

upsampled_I.type.value_counts()

# concatenate the upsampled data

upsampled_data = pd.concat([upsampled_E,upsampled_I])



upsampled_data.head()
fig3 = upsampled_data['type'].value_counts().plot(kind = 'bar')

fig3.set_title('Personality Type Frequency')

fig3.set_xlabel('Type')

fig3.set_ylabel('Total Posts')

plt.show()
X_train = upsampled_data.detoken

y_train = upsampled_data.type

#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

clf = DummyClassifier(strategy='most_frequent').fit(X_train, y_train)

y_pred = clf.predict(X_test)

print(' Our baseline accuracy is %s' % accuracy_score(y_pred, y_test))

nb = Pipeline([('vect', CountVectorizer(stop_words='english', ngram_range=(1, 2))),

               ('tfidf', TfidfTransformer(sublinear_tf=True, norm='l2')),

               ('clf', MultinomialNB())])



nb.fit(X_train, y_train)

y_predNB = nb.predict(X_test)

print('accuracy %s' % accuracy_score(y_predNB, y_test))

# calculate the loss

y_probsNB = nb.predict_proba(X_test)

print("The log loss error for our model is: ", log_loss(y_test, y_probsNB))
logreg = Pipeline([('vect', CountVectorizer(stop_words='english', ngram_range=(1, 2))),

                   ('tfidf', TfidfTransformer(norm='l2')),

                   ('clf', LogisticRegression(n_jobs=1, C=1e5))])



logreg.fit(X_train, y_train)

y_predLogRegTrain = logreg.predict(X_test)

y_predLogReg = logreg.predict(test.posts)

print('accuracy %s' % accuracy_score(y_predLogRegTrain, y_test))

# calculate the loss 

y_probslogreg = logreg.predict_proba(X_test)

print("The log loss error for our model is: ", log_loss(y_test, y_probslogreg))
print(classification_report(y_test, y_predLogRegTrain))
DecisionTree = Pipeline([('vect', CountVectorizer(stop_words='english', ngram_range=(1, 2))),

                         ('tfidf', TfidfTransformer()),

                         ('clf', DecisionTreeClassifier())])



clf = DecisionTree.fit(X_train, y_train)

y_predTree = clf.predict(X_test)

print('accuracy %s' % accuracy_score(y_predTree, y_test))

# calculate the loss

y_probsTree = clf.predict_proba(X_test)

print("The log loss error for our model is: ", log_loss(y_test, y_probsTree))
print(classification_report(y_test, y_predTree))





clf = Pipeline([('vect', CountVectorizer(stop_words='english')),

                         ('tfidf', TfidfTransformer()),

                         ('clf', SVC(kernel='polynomial'))])

# polynomial Kernel



#Train the model using the training sets

clf.fit(X_train, y_train)



y_predSVC = clf.predict(X_test)



print('accuracy %s' % accuracy_score(y_predSVC, y_test))

sample = pd.read_csv('../input/random_example.csv')
sample.head()


submission = pd.DataFrame(data = sample['id'], columns= ['id'])

submission['Type'] = y_predLogReg
submission['mind'] = submission['Type'].apply(lambda x: x[0] == 'E').astype('int')

submission['energy'] = submission['Type'].apply(lambda x: x[1] == 'N').astype('int')

submission['nature'] = submission['Type'].apply(lambda x: x[2] == 'T').astype('int')

submission['tactics'] = submission['Type'].apply(lambda x: x[3] == 'J').astype('int')

submission = submission.drop(['Type'],axis=1)
submission.head()
submission.to_csv('submit_2.csv', index=False)