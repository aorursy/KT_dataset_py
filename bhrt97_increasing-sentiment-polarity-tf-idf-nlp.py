import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
from bs4 import BeautifulSoup
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
import re
import seaborn as sns
import scipy
import matplotlib.pyplot as plt
import scikitplot as skplt
from wordcloud import WordCloud
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
df = pd.read_csv('/kaggle/input/steam-recommendation-nlp-dataset/train.csv')
game_df = pd.read_csv('/kaggle/input/steam-recommendation-nlp-dataset/game_overview.csv')
testing_df=  pd.read_csv('/kaggle/input/steam-recommendation-nlp-dataset/test.csv')
def rep(text):
    text = re.sub('♥♥♥♥',"worst bad horrible game",text)
    return text

df['user_review']=df.user_review.apply(rep)
testing_df['user_review']=testing_df.user_review.apply(rep)
def low(text):
    return text.lower()

df['user_review']=df.user_review.apply(low)
testing_df['user_review']=testing_df.user_review.apply(low)
def asc(text):
    text = re.sub('[^a-zA-Z]'," ",text)
    return text

df['user_review']=df.user_review.apply(asc)
testing_df['user_review']=testing_df.user_review.apply(asc)

testing_df.head(5)
def punctuation_removal(messy_str):
    clean_list = [char for char in messy_str if char not in string.punctuation]
    clean_str = ''.join(clean_list)
    return clean_str

df['user_review']=df.user_review.apply(punctuation_removal)
testing_df['user_review']=testing_df.user_review.apply(punctuation_removal)

df.head(4)
result_reviews = df['user_review'] 
testing_reviews = testing_df['user_review'] 

from textblob import TextBlob


from tqdm import tqdm
train_sentiments = []
for review in tqdm(result_reviews):
    blob = TextBlob(review)
    train_sentiments += [blob.sentiment.polarity]
    
from tqdm import tqdm
testing_sentiments = []
for review in tqdm(testing_reviews):
    blob = TextBlob(review)
    testing_sentiments += [blob.sentiment.polarity]
df['sentiment'] = train_sentiments
testing_df['sentiment'] =testing_sentiments
df=df.dropna()
np.corrcoef(df["user_suggestion"], df['year'])
np.corrcoef(df["user_suggestion"], df["sentiment"])
# def num(x):
#     if x >= 0:
#         return 1
#     elif x < 0:
#         return -1
# df['sentiment'] = df.sentiment.apply(num)
# testing_df['sentiment'] =testing_df.sentiment.apply(num)
# np.corrcoef(df["user_suggestion"], df["sentiment"])
def stopp(text):
    text = text.split()
    text = [i for i in text if not i in set(stopwords.words('english'))]
    text = ' '.join(text)
    return text
stopp("hello my name is HERO")
def tag(text):
    text = text[1:-1]
    return text

game_df.tags=game_df.tags.apply(tag)
game_df['tags']=game_df.tags.apply(low)
game_df['tags']=game_df.tags.apply(asc)
game_df['tags']=game_df.tags.apply(punctuation_removal)
game_df['tags']=game_df.tags.apply(stopp)

game_df.head(2)
vectorizer = CountVectorizer(tokenizer = lambda x: x.split())
tag_dtm = vectorizer.fit_transform(game_df['tags'])
print("Number of data points :", tag_dtm.shape[0])
print("Number of unique tags :", tag_dtm.shape[1])
print(type(tag_dtm))
# print(type(testing_tag_dtm))
# tag_vector= tag_dtm.toarray()
# tag_vector =  pd.DataFrame(tag_vector)
# tag_vector.head()
#'get_feature_name()' gives us the vocabulary.
tags = vectorizer.get_feature_names()
#Lets look at the tags we have.
print("Some of the tags we have :", tags[:30])
freqs = tag_dtm.sum(axis=0).A1
result = dict(zip(tags, freqs))
result
tags_df = pd.DataFrame(result.items(),columns=['Tags','Counts'])
tags_df.head()
tags_df.Counts.mean()
tags_df.Counts
tag_df_sorted = tags_df.sort_values(['Counts'], ascending=False)
tag_counts = tag_df_sorted['Counts'].values
tags_df[tags_df.Counts==110]
tag_counts
plt.plot(tag_counts)
plt.title("Distribution of number of times tag appeared overview")
plt.grid()
plt.xlabel("Tag number")
plt.ylabel("Number of times tag appeared")
plt.show()
# Store tags greater than 10K in one list
lst_tags_gt_10k = tags_df[tags_df.Counts>5].Tags
#Print the length of the list
print ('{} Tags are used more than 5 times'.format(len(lst_tags_gt_10k)))
# Store tags greater than 100K in one list
lst_tags_gt_100k = tags_df[tags_df.Counts>15].Tags
#Print the length of the list.
print ('{} Tags are used more than 15 times'.format(len(lst_tags_gt_100k)))
tag_df_sorted
i=np.arange(30)
tag_df_sorted.head(30).plot(kind='bar',figsize=(18, 10))
plt.title('Frequency of top 20 tags')
# plt.figure(figsize=(10,10))
plt.xticks(i, tag_df_sorted['Tags'])
plt.xlabel('Tags')
plt.ylabel('Counts')
# plt.set_size_inches(10,10)
plt.show()
big_df = pd.merge(df, game_df,on='title', how='left')
testing_df = pd.merge(testing_df, game_df,on='title', how='left')
testing_df.head(2)
vectorizer = CountVectorizer(tokenizer = lambda x: x.split())
# tag_dtm = vectorizer.fit_transform(game_df['tags'])
training_tag_dtm = vectorizer.fit_transform(big_df['tags'])
training_tag_vector= training_tag_dtm.toarray()
training_tag_vector =  pd.DataFrame(training_tag_vector)
training_tag_vector.shape
testing_tag_dtm = vectorizer.transform(testing_df['tags'])
testing_tag_vector= testing_tag_dtm.toarray()
testing_tag_vector =  pd.DataFrame(testing_tag_vector)
testing_tag_vector.shape
big_df.sample(10)
big_df['user_suggestion'].value_counts()
reviews_count = big_df.groupby(['title'])['user_review'].count().sort_values(ascending=False)

reviews_count = reviews_count.reset_index()

sns.set(style="darkgrid")
plt.figure(figsize=(15,20))
sns.barplot(y=reviews_count['title'], x=reviews_count['user_review'], data=reviews_count,
            label="Total", color="r")

reviews_count_pos = big_df.groupby(['title', 'user_suggestion'])['user_review'].count().sort_values(ascending=False)
reviews_count_pos = reviews_count_pos.reset_index()
reviews_count_pos = reviews_count_pos[reviews_count_pos['user_suggestion'] == 1]
sns.barplot(y=reviews_count_pos['title'], x=reviews_count_pos['user_review'], data=reviews_count_pos,
            label="Total", color="g")
year_count = big_df.groupby(['year'])['user_suggestion'].count().sort_values(ascending=False)

year_count = year_count.reset_index()
year_count
print(big_df[big_df['year']==2011].user_suggestion.value_counts())
print(big_df[big_df['year']==2012].user_suggestion.value_counts())
print(big_df[big_df['year']==2013].user_suggestion.value_counts())
print(big_df[big_df['year']==2014].user_suggestion.value_counts())
print(big_df[big_df['year']==2015].user_suggestion.value_counts())
print(big_df[big_df['year']==2016].user_suggestion.value_counts())
print(big_df[big_df['year']==2017].user_suggestion.value_counts())
print(big_df[big_df['year']==2018].user_suggestion.value_counts())
big_df.head(4)

testing_df.shape
training_gammes=big_df.title.unique()
testing_gammes=testing_df.title.unique()
print(len(training_gammes))
print(len(testing_gammes))
for i in testing_gammes:
    if i not in training_gammes:
        print(i)
training_dev=big_df.developer.unique()
testing_dev=testing_df.developer.unique()
print(len(training_dev))
print(len(testing_dev))
for i in testing_dev:
    if i  in training_dev:
        print(i)
training_dev=big_df.publisher.unique()
testing_dev=testing_df.publisher.unique()
print(len(training_dev))
print(len(testing_dev))
for i in testing_dev:
    if i  in training_dev:
        print(i)
reviews_count = big_df.groupby(['publisher'])['user_review'].count().sort_values(ascending=False)

reviews_count = reviews_count.reset_index()

sns.set(style="darkgrid")
plt.figure(figsize=(15,20))
sns.barplot(y=reviews_count['publisher'], x=reviews_count['user_review'], data=reviews_count,
            label="Total", color="r")

reviews_count_pos = big_df.groupby(['publisher', 'user_suggestion'])['user_review'].count().sort_values(ascending=False)
reviews_count_pos = reviews_count_pos.reset_index()
reviews_count_pos = reviews_count_pos[reviews_count_pos['user_suggestion'] == 1]
sns.barplot(y=reviews_count_pos['publisher'], x=reviews_count_pos['user_review'], data=reviews_count_pos,
            label="Total", color="g")

# Valve  -- 50% > +ve
# Wargaming Group Limited 50% > -ve
# Perfect World Entertainment 50% > +ve
# Hi-Rez Studios 50% > +ve
# Daybreak Game Company 50% > +ve

Title_pos_df = big_df[big_df.sentiment > 0]
Title_neg_df = big_df[big_df.sentiment < 0]

Title_pos_df.sample(3)
Title_neg_df.sample(3)
pos_words =[]
neg_words = []
# neu_words = []

for text in Title_pos_df.user_review:
    pos_words.append(text) 
pos_words = ' '.join(pos_words)
# pos_words[:40]


for text in Title_neg_df.user_review:
    neg_words.append(text)
neg_words = ' '.join(neg_words)
# neg_words[:400]
wordcloud = WordCloud().generate(pos_words)

wordcloud = WordCloud(background_color="white",max_words=len(pos_words),\
                      max_font_size=40, relative_scaling=.5, colormap='summer').generate(pos_words)
plt.figure(figsize=(13,13))
plt.imshow(wordcloud)
plt.axis("off")
plt.show()
wordcloud = WordCloud().generate(neg_words)

wordcloud = WordCloud(background_color="white",max_words=len(neg_words),\
                      max_font_size=40, relative_scaling=.5, colormap='gist_heat').generate(neg_words)
plt.figure(figsize=(13,13))
plt.imshow(wordcloud)
plt.axis("off")
plt.show()
big_df.head(4)

from nltk.tokenize import sent_tokenize, word_tokenize, RegexpTokenizer
stop = stopwords.words('english')
stop.append("i'm")
stop.remove("not")


stop_words = []

for item in stop: 
    new_item = punctuation_removal(item)
    stop_words.append(new_item) 
# print(stop_words[::12])


def stopwords_removal(messy_str):
    messy_str = word_tokenize(messy_str)
    return [word.lower() for word in messy_str 
            if word.lower() not in stop_words ]

big_df['user_review'] = big_df['user_review'].apply(stopwords_removal)
testing_df['user_review'] = testing_df['user_review'].apply(stopwords_removal)
def drop_numbers(list_text):
    list_text_new = []
    for i in list_text:
        if not re.search('\d', i):
            list_text_new.append(i)
    return ' '.join(list_text_new)

big_df['user_review'] = big_df['user_review'].apply(drop_numbers)
testing_df['user_review'] = testing_df['user_review'].apply(drop_numbers)
porter = PorterStemmer()
big_df['user_review'] = big_df['user_review'].apply(lambda x: x.split())
testing_df['user_review'] = testing_df['user_review'].apply(lambda x: x.split())


def stem_update(text_list):
    text_list_new = []
    for word in text_list:
        word = porter.stem(word)
        text_list_new.append(word) 
    return text_list_new


big_df['user_review'] = big_df['user_review'].apply(stem_update)
testing_df['user_review'] = testing_df['user_review'].apply(stem_update)

big_df['user_review'] = big_df['user_review'].apply(lambda x: ' '.join(x))
testing_df['user_review'] = testing_df['user_review'].apply(lambda x: ' '.join(x))

training_text = big_df['user_review']
testing_text = testing_df['user_review'] 
training_text = pd.DataFrame(training_text)
testing_text = pd.DataFrame(testing_text)
def text_vectorizing_process(sentence_string):
    return [word for word in sentence_string.split()]

vector_obj = CountVectorizer(text_vectorizing_process,max_features=2000,ngram_range=(1, 3))
vector_obj.fit(training_text['user_review'])
Title_text= vector_obj.transform(training_text['user_review'])

Title_text_testing= vector_obj.transform(testing_text['user_review'])
print('Shape of Sparse Matrix', Title_text_testing.shape)
print('Shape of Sparse Matrix', Title_text.shape)
print('Amount of Non-Zero occurences:', Title_text.nnz)
Title_tfidf_transformer = TfidfTransformer().fit(Title_text)
Title_messages_tfidf = Title_tfidf_transformer.transform(Title_text)

Title_messages_tfidf_testing = Title_tfidf_transformer.transform(Title_text_testing)
Title_messages_tfidf_testing = Title_messages_tfidf_testing.toarray()
Title_messages_tfidf_testing = pd.DataFrame(Title_messages_tfidf_testing)
print(Title_messages_tfidf_testing.shape)
Title_messages_tfidf_testing.head()

Title_messages_tfidf = Title_messages_tfidf.toarray()
Title_messages_tfidf = pd.DataFrame(Title_messages_tfidf)
print(Title_messages_tfidf.shape)
Title_messages_tfidf.head()

# Title_df_all = pd.merge(Title_messages_tfidf, big_df['user_suggestion'],
#                   left_index=True, right_index=True )
# Title_df_all.head()
# print(type(testing_tag_dtm))
print(type(Title_messages_tfidf_testing))
# testing_tag_dtm = pd.DataFrame(testing_tag_dtm)
# testing_tag_dtm
# testing_tag_dtm = pd.DataFrame(testing_tag_dtm)

# testing_review_tag = pd.merge(Title_messages_tfidf_testing,testing_tag_dtm ,
#                   left_index=True, right_index=True )
# testing_review_tag.head()
# print(type(tag_dtm))
# print(type(Title_df_all))

Title_messages_tfidf.shape
tag_vector.shape
review_tag = pd.merge(Title_messages_tfidf,tag_vector ,
                  left_index=True, right_index=True )
review_tag.head()
review_tag.shape
X = review_tag.values
y = big_df['user_suggestion'].values

# X.head()
from sklearn.model_selection import train_test_split as split

X_train, X_test, y_train, y_test = split(X,y, test_size=0.01, random_state=12)
X_train.shape, X_test.shape, y_train.shape, y_test.shape
from sklearn.preprocessing import StandardScaler, MinMaxScaler

Title_scaler = MinMaxScaler()
X_train_scaled = Title_scaler.fit_transform(X_train)
X_test_scaled = Title_scaler.transform(X_test)
from sklearn.metrics import accuracy_score
clf = LogisticRegression(max_iter=400,C=0.4,tol=1e-5,solver='liblinear',class_weight={1:1,0:1},multi_class='ovr',
                        verbose=0,warm_start=True,
                        )#dual=True,

clf.fit(X_train_scaled, y_train)
y_pred = clf.predict(X_test_scaled)

print('LogisticRegression accuracy_score {0:0.4f}'. format(accuracy_score(y_test, y_pred)))
clf.score(X_train_scaled, y_train)
# from sklearn import svm
# clf = svm.SVC(kernel='linear')
# clf.fit(X_train_scaled, y_train)
# y_pred = clf.predict(X_test_scaled)

# print('LogisticRegression accuracy_score {0:0.4f}'. format(accuracy_score(y_test.values, y_pred)))
# clf.score(X_train_scaled, y_train)
# from xgboost import XGBClassifier

# clf= XGBClassifier(learning_rate=0.04, n_estimators=406, max_depth=5,
#                         min_child_weight=4, 
#                          seed=27)
# clf.fit(X_train_scaled, y_train)
# y_pred = clf.predict(X_test_scaled)

# print('LogisticRegression accuracy_score {0:0.4f}'. format(accuracy_score(y_test.values, y_pred)))
# clf.score(X_train_scaled, y_train)
# from sklearn import linear_model
# clf = linear_model.Lasso(alpha=1,eps=1e-4)

# clf.fit(X_train_scaled, y_train)
# y_pred = clf.predict(X_test_scaled)

# preds = []
# for x in y_pred:
#     preds.append(int(x.round()))
    
# print('LogisticRegression accuracy_score {0:0.4f}'. format(accuracy_score(y_test.values, preds)))
# clf.score(X_train_scaled, y_train)















X = Title_df_all.drop(['user_suggestion'], axis=1)
y = Title_df_all.user_suggestion

X.head()

from sklearn.model_selection import train_test_split as split

X_train, X_test, y_train, y_test = split(X,y, test_size=0.3, random_state=111)
X_train.shape, X_test.shape, y_train.shape, y_test.shape
from sklearn.preprocessing import StandardScaler, MinMaxScaler

Title_scaler = MinMaxScaler()
X_train_scaled = Title_scaler.fit_transform(X_train)
X_test_scaled = Title_scaler.transform(X_test)
from sklearn.metrics import accuracy_score
clf = LogisticRegression(max_iter=400,C=0.4,tol=1e-5,solver='liblinear',class_weight={1:1,0:1},multi_class='ovr',
                        verbose=0,warm_start=True,
                       )
clf.fit(X_train_scaled, y_train)
y_pred = clf.predict(X_test_scaled)

print('LogisticRegression accuracy_score {0:0.4f}'. format(accuracy_score(y_test.values, y_pred)))
clf.score(X_train_scaled, y_train)




vectorizer = CountVectorizer(tokenizer = lambda x: x.split())
tag_dtm = vectorizer.fit_transform(big_df['tags'])
print("Number of data points :", tag_dtm.shape[0])
print("Number of unique tags :", tag_dtm.shape[1])
tag_dtm
big_df.head(1)
tag_dtm = tag_dtm.toarray()
tag_dtm = pd.DataFrame(tag_dtm)
tag_dtm
Tag_df_all = pd.merge(tag_dtm, big_df['user_suggestion'],
                  left_index=True, right_index=True )
Tag_df_all.head()
X = Tag_df_all.drop(['user_suggestion'], axis=1)
y = Tag_df_all.user_suggestion

X.head()
from sklearn.model_selection import train_test_split as split

X_train, X_test, y_train, y_test = split(X,y, test_size=0.3, random_state=111)
X_train.shape, X_test.shape, y_train.shape, y_test.shape
from sklearn.preprocessing import StandardScaler, MinMaxScaler

Title_scaler = MinMaxScaler()
X_train_scaled = Title_scaler.fit_transform(X_train)
X_test_scaled = Title_scaler.transform(X_test)
from sklearn.metrics import accuracy_score
clf = LogisticRegression(max_iter=200, )
clf.fit(X_train_scaled, y_train)
y_pred = clf.predict(X_test_scaled)

print('LogisticRegression accuracy_score {0:0.4f}'. format(accuracy_score(y_test.values, y_pred)))
clf.score(X_train_scaled, y_train)
