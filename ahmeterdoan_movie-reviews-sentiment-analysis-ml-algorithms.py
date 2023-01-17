import pandas as pd  

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from nltk import ngrams

from bs4 import BeautifulSoup

from nltk.tokenize import WordPunctTokenizer

from nltk.corpus import stopwords 

import string

from collections import Counter

from sklearn.model_selection import train_test_split

import re

import nltk

from nltk.stem.porter import PorterStemmer

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC

from sklearn.naive_bayes import MultinomialNB

from sklearn.tree import DecisionTreeClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import AdaBoostClassifier

from sklearn.ensemble import BaggingClassifier

from sklearn.ensemble import ExtraTreesClassifier

from sklearn.metrics import accuracy_score, confusion_matrix
train = pd.read_csv('../input/movie-review/train.tsv', sep="\t")

test = pd.read_csv('../input/movie-review/test.tsv', sep="\t")
train
test
#for counting ham and spam



train.Sentiment.value_counts()
train.groupby('Sentiment').describe()
tok = WordPunctTokenizer()



pat1 = r'@[A-Za-z0-9_]+'

pat2 = r'https?://[^ ]+'

combined_pat = r'|'.join((pat1, pat2))

www_pat = r'www.[^ ]+'

negations_dic = {"isn't":"is not", "aren't":"are not", "wasn't":"was not", "weren't":"were not",

                "haven't":"have not","hasn't":"has not","hadn't":"had not","won't":"will not",

                "wouldn't":"would not", "don't":"do not", "doesn't":"does not","didn't":"did not",

                "can't":"can not","couldn't":"could not","shouldn't":"should not","mightn't":"might not",

                "mustn't":"must not"}

neg_pattern = re.compile(r'\b(' + '|'.join(negations_dic.keys()) + r')\b')



def tweet_cleaner_updated(text):

    soup = BeautifulSoup(text, 'lxml')

    souped = soup.get_text()

    try:

        bom_removed = souped.decode("utf-8-sig").replace(u"\ufffd", "?")

    except:

        bom_removed = souped

    stripped = re.sub(combined_pat, '', bom_removed)

    stripped = re.sub(www_pat, '', stripped)

    lower_case = stripped.lower()

    neg_handled = neg_pattern.sub(lambda x: negations_dic[x.group()], lower_case)

    letters_only = re.sub("[^a-zA-Z]", " ", neg_handled)

    # During the letters_only process two lines above, it has created unnecessay white spaces,

    # I will tokenize and join together to remove unneccessary white spaces

    words = [x for x  in tok.tokenize(letters_only) if len(x) > 1]

    return (" ".join(words)).strip()
%%time

print ("Cleaning the tweets...\n")

clean_data = []

for row in train.Phrase:

    clean_data.append(tweet_cleaner_updated(row))
len(clean_data)
clean_df = pd.DataFrame(clean_data,columns=['Phrase'])

clean_df['PhraseId'] = train.PhraseId

clean_df['SentenceId'] = train.SentenceId

clean_df['Sentiment'] = train.Sentiment

clean_df.to_csv('cleaned_train_data.csv',encoding='utf-8')
csv = 'cleaned_train_data.csv'

cleaned_train_data = pd.read_csv(csv,index_col=0)

cleaned_train_data
cleaned_train_data[cleaned_train_data.isnull().any(axis=1)]
cleaned_train_data.isnull().any(axis=0)
cleaned_train_data.dropna(inplace=True)

cleaned_train_data.reset_index(drop=True,inplace=True)

cleaned_train_data.info()
cleaned_train_data.shape
%%time

print ("Cleaning the tweets...\n")

clean_data = []

for row in test.Phrase:

    clean_data.append(tweet_cleaner_updated(row))
test.head()
clean_df = pd.DataFrame(clean_data, columns=['Phrase'])

clean_df['PhraseId'] = test.PhraseId

clean_df['SentenceId'] = test.SentenceId



clean_df.to_csv('cleaned_test_data.csv',encoding='utf-8')
csv = 'cleaned_test_data.csv'

cleaned_test_data = pd.read_csv(csv,index_col=0)

cleaned_test_data
cleaned_test_data[cleaned_test_data.isnull().any(axis=1)]
cleaned_test_data.dropna(inplace=True)

cleaned_test_data.reset_index(drop=True,inplace=True)

cleaned_test_data.info()
#plotting graph for distribution



sns.countplot(x = 'Sentiment', data = cleaned_train_data)

cleaned_train_data.loc[:,'Sentiment'].value_counts()

plt.title('Distribution of Train Data')
# plotting graph by length.



neg = cleaned_train_data[cleaned_train_data['Sentiment']== 0]['Phrase'].str.len()

sns.distplot(neg, label='Negative')

som_neg = cleaned_train_data[cleaned_train_data['Sentiment']== 1]['Phrase'].str.len()

sns.distplot(som_neg, label='Somewhat Neg')

neat = cleaned_train_data[cleaned_train_data['Sentiment']== 2]['Phrase'].str.len()

sns.distplot(neat, label='Neatral')

som_pos = cleaned_train_data[cleaned_train_data['Sentiment']== 3]['Phrase'].str.len()

sns.distplot(som_pos, label='Somewhat Pos')

pos = cleaned_train_data[cleaned_train_data['Sentiment']== 4]['Phrase'].str.len()

sns.distplot(pos, label='Positive')

plt.title('Distribution by Length')

plt.legend()
#plotting graph for non-digits.



neg = cleaned_train_data[cleaned_train_data['Sentiment'] == 0]['Phrase'].str.replace(r'\w+', '').str.len()

sns.distplot(neg, label='Negative')

som_neg = cleaned_train_data[cleaned_train_data['Sentiment'] == 1]['Phrase'].str.replace(r'\w+', '').str.len()

sns.distplot(som_neg, label='Somewhat Neg')

neutral = cleaned_train_data[cleaned_train_data['Sentiment'] == 2]['Phrase'].str.replace(r'\w+', '').str.len()

sns.distplot(neutral, label='Neutral')

som_pos = cleaned_train_data[cleaned_train_data['Sentiment'] == 3]['Phrase'].str.replace(r'\w+', '').str.len()

sns.distplot(som_pos, label='Somewhat Pos')

pos = cleaned_train_data[cleaned_train_data['Sentiment'] == 4]['Phrase'].str.replace(r'\w+', '').str.len()

sns.distplot(pos, label='Positive')

plt.title('Distribution of Non-Digits')

plt.legend()
from collections import Counter

#for counting frequently occurence of pos and neg.



count1 = Counter(" ".join(cleaned_train_data[cleaned_train_data['Sentiment']== 0]["Phrase"]).split()).most_common(30)

data1 = pd.DataFrame.from_dict(count1)

data1 = data1.rename(columns={0: "words of neg", 1 : "count"})



count2 = Counter(" ".join(cleaned_train_data[cleaned_train_data['Sentiment']== 1]["Phrase"]).split()).most_common(30)

data2 = pd.DataFrame.from_dict(count2)

data2 = data2.rename(columns={0: "words of somewhat_neg", 1 : "count_"})



count3 = Counter(" ".join(cleaned_train_data[cleaned_train_data['Sentiment']== 2]["Phrase"]).split()).most_common(30)

data3 = pd.DataFrame.from_dict(count3)

data3 = data3.rename(columns={0: "words of neutral", 1 : "count_"})



count4 = Counter(" ".join(cleaned_train_data[cleaned_train_data['Sentiment']== 3]["Phrase"]).split()).most_common(30)

data4 = pd.DataFrame.from_dict(count4)

data4 = data4.rename(columns={0: "words of somewhat_pos", 1 : "count_"})



count5 = Counter(" ".join(cleaned_train_data[cleaned_train_data['Sentiment']== 4]["Phrase"]).split()).most_common(30)

data5 = pd.DataFrame.from_dict(count5)

data5 = data5.rename(columns={0: "words of pos", 1 : "count_"})
data1.plot.bar(legend = False, color = 'purple',figsize = (20,15))

y_neg = np.arange(len(data1["words of neg"]))

plt.xticks(y_neg, data1["words of neg"])

plt.title('Top 30 words of neg')

plt.xlabel('words')

plt.ylabel('number')

plt.show()
data2.plot.bar(legend = False, color = 'purple',figsize = (20,15))

y_somewhat_neg = np.arange(len(data2["words of somewhat_neg"]))

plt.xticks(y_somewhat_neg, data2["words of somewhat_neg"])

plt.title('Top 30 words of somewhat_neg')

plt.xlabel('words')

plt.ylabel('number')

plt.show()
data3.plot.bar(legend = False, color = 'purple',figsize = (20,15))

y_neutral = np.arange(len(data3["words of neutral"]))

plt.xticks(y_neutral, data3["words of neutral"])

plt.title('Top 30 words of neutral')

plt.xlabel('words')

plt.ylabel('number')

plt.show()
data4.plot.bar(legend = False, color = 'purple',figsize = (20,15))

y_somewhat_pos = np.arange(len(data4["words of somewhat_pos"]))

plt.xticks(y_somewhat_pos, data4["words of somewhat_pos"])

plt.title('Top 30 words of somewhat_pos')

plt.xlabel('words')

plt.ylabel('number')

plt.show()
data5.plot.bar(legend = False, color = 'purple',figsize = (20,15))

y_pos = np.arange(len(data5["words of pos"]))

plt.xticks(y_pos, data5["words of pos"])

plt.title('Top 30 words of pos')

plt.xlabel('words')

plt.ylabel('number')

plt.show()
all_string = []

for t in cleaned_train_data.Phrase:

    all_string.append(t)

all_string = pd.Series(all_string).str.cat(sep=' ')
from wordcloud import WordCloud



wordcloud = WordCloud(width=1600, height=800,max_font_size=200,colormap='magma').generate(all_string)

plt.figure(figsize=(12,10))

plt.imshow(wordcloud, interpolation="bilinear")

plt.axis("off")

plt.show()
all_string = []

for t in cleaned_test_data.Phrase:

    all_string.append(t)

all_string = pd.Series(all_string).str.cat(sep=' ')
wordcloud = WordCloud(width=1600, height=800,max_font_size=200,colormap='magma').generate(all_string)

plt.figure(figsize=(12,10))

plt.imshow(wordcloud, interpolation="bilinear")

plt.axis("off")

plt.show()
stopwords_english = stopwords.words('english')
# clean words, i.e. remove stopwords and punctuation

def clean_words(words, stopwords_english):

    words_clean = []

#     print(type(words))

    for word in words:

        word = word.lower()

        if word not in stopwords_english and word not in string.punctuation:

            words_clean.append(word)    

    return words_clean 
# feature extractor function for unigram

def bag_of_words(words):    

    words_dictionary = dict([word, True] for word in words)    

    return words_dictionary
# feature extractor function for ngrams (bigram)

def bag_of_ngrams(words, n=2):

    words_ng = []

    for item in iter(ngrams(words, n)):

        words_ng.append(item)

    words_dictionary = dict([word, True] for word in words_ng)    

    return words_dictionary
from nltk.tokenize import word_tokenize

text = "It was a very good movie."

words = word_tokenize(text.lower())

words
bag_of_ngrams(words)
words_clean = clean_words(words, stopwords_english)

words_clean
important_words = ['above', 'below', 'off', 'over', 'under', 'more', 'most', 'such', 'no', 'nor', 'not', 'only', 'so', 'than', 'too', 'very', 'just', 'but']
stopwords_english_for_bigrams = set(stopwords_english) - set(important_words)
words_clean_for_bigrams = clean_words(words, stopwords_english_for_bigrams)

words_clean_for_bigrams
unigram_features = bag_of_words(words_clean)

unigram_features
bigram_features = bag_of_ngrams(words_clean_for_bigrams)

bigram_features
# combine both unigram and bigram features

all_features = unigram_features.copy()

all_features.update(bigram_features)

all_features
# let's define a new function that extracts all features

# i.e. that extracts both unigram and bigrams features

def bag_of_all_words(words, n=2):

    words_clean = clean_words(words, stopwords_english)

    words_clean_for_bigrams = clean_words(words, stopwords_english_for_bigrams)

 

    unigram_features = bag_of_words(words_clean)

    bigram_features = bag_of_ngrams(words_clean_for_bigrams)

 

    all_features = unigram_features.copy()

    all_features.update(bigram_features)

 

    return all_features
bag_of_all_words(words)
neg_reviews = [row for row in cleaned_train_data[cleaned_train_data.Sentiment == 0].Phrase.str.split()]

neg_reviews[0]
somewhat_neg_reviews = [row for row in cleaned_train_data[cleaned_train_data.Sentiment == 1].Phrase.str.split()]

somewhat_neg_reviews[0]
neutral_reviews = [row for row in cleaned_train_data[cleaned_train_data.Sentiment == 2].Phrase.str.split()]

neutral_reviews[0]
somewhat_pos_reviews = [row for row in cleaned_train_data[cleaned_train_data.Sentiment == 3].Phrase.str.split()]

somewhat_pos_reviews[0]
pos_reviews = [row for row in cleaned_train_data[cleaned_train_data.Sentiment == 4].Phrase.str.split()]

pos_reviews[0]
neg_reviews_set = []

for words in neg_reviews:

#     print(words)

    neg_reviews_set.append((bag_of_all_words(words), '0'))

neg_reviews_set[0]
somewhat_neg_reviews_set = []

for words in somewhat_neg_reviews:

#     print(words)

    somewhat_neg_reviews_set.append((bag_of_all_words(words), '1'))

somewhat_neg_reviews_set[0]
neutral_reviews_set = []

for words in neutral_reviews:

#     print(words)

    neutral_reviews_set.append((bag_of_all_words(words), '2'))

neutral_reviews_set[0]
somewhat_pos_reviews_set = []

for words in somewhat_pos_reviews:

#     print(words)

    somewhat_pos_reviews_set.append((bag_of_all_words(words), '3'))

somewhat_pos_reviews_set[0]
pos_reviews_set = []

for words in pos_reviews:

#     print(words)

    pos_reviews_set.append((bag_of_all_words(words), '4'))

pos_reviews_set[0]
print (len(neg_reviews_set), len(somewhat_neg_reviews_set), len(neutral_reviews_set), len(somewhat_pos_reviews_set), len(pos_reviews_set)) 

print (len(neg_reviews_set)+len(somewhat_neg_reviews_set)+len(neutral_reviews_set)+len(somewhat_pos_reviews_set)+len(pos_reviews_set)) 



# radomize pos_reviews_set and neg_reviews_set

# doing so will output different accuracy result everytime we run the program

from random import shuffle 

shuffle(neg_reviews_set)

shuffle(somewhat_neg_reviews_set)

shuffle(neutral_reviews_set)

shuffle(somewhat_pos_reviews_set)

shuffle(pos_reviews_set)



test_set = neg_reviews_set[:int((0.2*len(neg_reviews_set)))] + somewhat_neg_reviews_set[:int((0.2*len(somewhat_neg_reviews_set)))] + neutral_reviews_set[:int((0.2*len(neutral_reviews_set)))] + somewhat_pos_reviews_set[:int((0.2*len(somewhat_pos_reviews_set)))] + pos_reviews_set[:int((0.2*len(pos_reviews_set)))]

train_set = neg_reviews_set[int((0.2*len(neg_reviews_set))):] + somewhat_neg_reviews_set[int((0.2*len(somewhat_neg_reviews_set))):] + neutral_reviews_set[int((0.2*len(neutral_reviews_set))):] + somewhat_pos_reviews_set[int((0.2*len(somewhat_pos_reviews_set))):] + pos_reviews_set[int((0.2*len(pos_reviews_set))):]

 

print(len(test_set),  len(train_set))
%%time

from nltk import classify

from nltk import NaiveBayesClassifier

 

classifier = NaiveBayesClassifier.train(train_set)

 

accuracy = classify.accuracy(classifier, test_set)

print(accuracy)

 

print (classifier.show_most_informative_features(10)) 
X_train, X_test, y_train, y_test = train_test_split(cleaned_train_data['Phrase'], cleaned_train_data['Sentiment'], test_size = 0.2, random_state = 37)

print ("X_train: ", len(X_train))

print("X_test: ", len(X_test))

print("y_train: ", len(y_train))

print("y_test: ", len(y_test))
cv = CountVectorizer(max_features = 1500)

cv.fit(X_train)
X_train_cv = cv.transform(X_train)

X_train_cv
X_test_cv = cv.transform(X_test)

X_test_cv
%%time

mnb = MultinomialNB(alpha = 0.5)

MNB_model = mnb.fit(X_train_cv,y_train)



y_mnb = mnb.predict(X_test_cv)
print('Naive Bayes Accuracy: ', accuracy_score(y_mnb , y_test))

print('Naive Bayes confusion_matrix: ', confusion_matrix(y_mnb, y_test))

MNB_accuracy_train  = round(MNB_model.score(X_test_cv,y_test) * 100, 2)

MNB_accuracy_train
%%time

svc = SVC(kernel='sigmoid', gamma=1.0)

SVC_model = svc.fit(X_train_cv,y_train)



y_svc = svc.predict(X_test_cv)
print('SVM Accuracy: ', accuracy_score(y_svc , y_test))

print('SVM confusion_matrix: ', confusion_matrix(y_svc, y_test))

SVC_accuracy_train  = round(SVC_model.score(X_test_cv,y_test) * 100, 2)

SVC_accuracy_train 
%%time

knc = KNeighborsClassifier(n_neighbors=100)

KNNC_model = knc.fit(X_train_cv,y_train)



y_knc = knc.predict(X_test_cv)
print('KNeighbors Accuracy_score: ',accuracy_score(y_test,y_knc))

print('KNeighbors confusion_matrix: ', confusion_matrix(y_test, y_knc))

KNNC_accuracy_train  = round(KNNC_model.score(X_test_cv,y_test) * 100, 2)

KNNC_accuracy_train 
%%time

dtc = DecisionTreeClassifier()

DTC_model = dtc.fit(X_train_cv,y_train)



y_dtc = dtc.predict(X_test_cv)
print('Decision Tree Accuracy: ',accuracy_score(y_test,y_dtc))

print('Decision Tree confusion_matrix: ', confusion_matrix(y_dtc, y_test)) 

DTC_accuracy_train  = round(DTC_model.score(X_test_cv,y_test) * 100, 2)

DTC_accuracy_train 
%%time

etc = ExtraTreesClassifier(n_estimators=37, random_state=252)

ETC_model = etc.fit(X_train_cv,y_train)



y_etc = etc.predict(X_test_cv)
print('Extra Tree Accuracy_score: ',accuracy_score(y_test,y_etc))

print('Extra Tree confusion_matrix: ', confusion_matrix(y_etc, y_test))

ETC_accuracy_train  = round(ETC_model.score(X_test_cv,y_test) * 100, 2)

ETC_accuracy_train 
%%time

rfc = RandomForestClassifier(n_estimators=37, random_state=252)

RFC_model = rfc.fit(X_train_cv,y_train)



y_rfc = rfc.predict(X_test_cv)
print('Random Forest Accuracy_score: ',accuracy_score(y_test,y_rfc))

print('Random Forest confusion_matrix: ', confusion_matrix(y_rfc, y_test)) 

RFC_accuracy_train  = round(RFC_model.score(X_test_cv,y_test) * 100, 2)

RFC_accuracy_train 
%%time

abc = AdaBoostClassifier(n_estimators=37, random_state=252)

ABC_model = abc.fit(X_train_cv,y_train)



y_abc = abc.predict(X_test_cv)
print('AdaBoost Accuracy_score: ',accuracy_score(y_test,y_abc))

print('AdaBoost confusion_matrix: ', confusion_matrix(y_abc, y_test))

ABC_accuracy_train  = round(ABC_model.score(X_test_cv,y_test) * 100, 2)

ABC_accuracy_train 
%%time

bc = BaggingClassifier(n_estimators=9, random_state=252)

BC_model = bc.fit(X_train_cv,y_train)



y_bc = bc.predict(X_test_cv)
print('Bagging Accuracy_score: ',accuracy_score(y_test,y_bc))

print('Bagging confusion_matrix: ', confusion_matrix(y_bc, y_test))

BC_accuracy_train  = round(BC_model.score(X_test_cv,y_test) * 100, 2)

BC_accuracy_train 
%%time

log_model = LogisticRegression()

LRC_model = log_model.fit(X_train_cv,y_train)



y_lr = log_model.predict(X_test_cv)
print('LR Accuracy_score: ',accuracy_score(y_test,y_bc))

print('LR confusion_matrix: ', confusion_matrix(y_bc, y_test)) 

LRC_accuracy_train  = round(LRC_model.score(X_test_cv,y_test) * 100, 2)

LRC_accuracy_train
# coefficeints of the trained model

print('Coefficient of model :', log_model.coef_)

print('-'*45)

# intercept of the model

print('Intercept of model',log_model.intercept_)
from sklearn.ensemble import GradientBoostingClassifier
%%time

gbm_model = GradientBoostingClassifier()

GBM_model = gbm_model.fit(X_train_cv,y_train)



y_gbm = gbm_model.predict(X_test_cv)
print('GBM Accuracy_score: ',accuracy_score(y_test,y_gbm))

print('GBM confusion_matrix: ', confusion_matrix(y_gbm, y_test))

GBM_accuracy_train  = round(GBM_model.score(X_test_cv,y_test) * 100, 2)

GBM_accuracy_train


models = pd.DataFrame({

    'Model': ['MultinominalNB','SVM','KNN','Decision Tree','Extra Classification','Random Forest','AdaBost','Bagging Classification','Logistic Regression','GBM'],

    'Accuray Score': [MNB_accuracy_train,SVC_accuracy_train,KNNC_accuracy_train,DTC_accuracy_train,ETC_accuracy_train,RFC_accuracy_train,ABC_accuracy_train,BC_accuracy_train,LRC_accuracy_train,GBM_accuracy_train]

})

models.sort_values(by='Accuray Score', ascending=False)
X_train_cv2 = cv.transform(cleaned_test_data['Phrase'])

X_train_cv2
Phrase_Id = cleaned_test_data['PhraseId'].values
len(Phrase_Id)
y_rfc_test_predict = RFC_model.predict(X_train_cv2)
y_rfc_test_predict.shape
final_file = pd.DataFrame({'PhraseId':Phrase_Id,'Sentiment':y_rfc_test_predict})
final_file
#make the predictions with trained model and submit the predictions.



final_file.to_csv('Submission.csv',index=False)
sub = pd.read_csv('Submission.csv')
sub