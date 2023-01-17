import warnings

import gc

warnings.filterwarnings("ignore")

import pandas as pd

import sqlite3

import csv

import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np

from wordcloud import WordCloud

import re

import os

from sqlalchemy import create_engine # database connection

import datetime as dt

from nltk.corpus import stopwords

from nltk.tokenize import word_tokenize

from nltk.stem.snowball import SnowballStemmer

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.multiclass import OneVsRestClassifier

from sklearn.linear_model import SGDClassifier

from sklearn import metrics

from sklearn.metrics import f1_score,precision_score,recall_score

from sklearn import svm

from sklearn.linear_model import LogisticRegression

from skmultilearn.adapt import mlknn

from skmultilearn.problem_transform import ClassifierChain

from skmultilearn.problem_transform import BinaryRelevance

from skmultilearn.problem_transform import LabelPowerset

from sklearn.naive_bayes import GaussianNB

from datetime import datetime

import pandasql
df = pd.read_csv("/kaggle/input/facebook-recruiting-iii-keyword-extraction/Train.zip",nrows=10000)

df.head()
# df = df_main[:50000]

df.info()

df[df.duplicated()] # printing duplicated rows


#delete when no longer needed

# del df_main

#collect residual garbage

# gc.collect()


df_no_dup = df.drop_duplicates()
df_no_dup.head()

# we can observe that there are duplicates
# number of times each question appeared in our database

# df_no_dup.cnt_dup.value_counts()

# deleting the rows which have empty tags for further processing, there were only 6 such rows

df_no_dup = df_no_dup.dropna()

df_no_dup.isnull().sum()
start = datetime.now()

df_no_dup["tag_count"] = df_no_dup["Tags"].apply(lambda text: len(text.split()))

# adding a new feature number of tags per question

print("Time taken to run this cell :", datetime.now() - start)

df_no_dup.head()
# distribution of number of tags per question

df_no_dup.tag_count.value_counts()

# df.info()

# df_no_dup.info()
#Creating a new database with no duplicates

# os.remove('train.db')

# if not os.path.isfile('train_no_dup.db'):

#     disk_dup = create_engine("sqlite:///train_no_dup.db")

#     no_dup = pd.DataFrame(df_no_dup, columns=['Title', 'Body', 'Tags'])

#     no_dup.to_sql('no_dup_train',disk_dup)
# Importing & Initializing the "CountVectorizer" object, which 

#is scikit-learn's bag of words tool.



#by default 'split()' will tokenize each tag using space.

vectorizer = CountVectorizer(tokenizer = lambda x: x.split())

# fit_transform() does two functions: First, it fits the model

# and learns the vocabulary; second, it transforms our training data

# into feature vectors. The input to fit_transform should be a list of strings.

tag_dtm = vectorizer.fit_transform(df_no_dup['Tags'])
print("Number of data points :", tag_dtm.shape[0])

print("Number of unique tags :", tag_dtm.shape[1])
#'get_feature_name()' gives us the vocabulary.

tags = vectorizer.get_feature_names()

#Lets look at the tags we have.

print("Some of the tags we have :", tags[:10])
# https://stackoverflow.com/questions/15115765/how-to-access-sparse-matrix-elements

#Lets now store the document term matrix in a dictionary.

freqs = tag_dtm.sum(axis=0).A1

result = dict(zip(tags, freqs))
#Saving this dictionary to csv files.

if not os.path.isfile('tag_counts_dict_dtm.csv'):

    with open('tag_counts_dict_dtm.csv', 'w') as csv_file:

        writer = csv.writer(csv_file)

        for key, value in result.items():

            writer.writerow([key, value])

tag_df = pd.read_csv("tag_counts_dict_dtm.csv", names=['Tags', 'Counts'])

tag_df.head()
tag_df_sorted = tag_df.sort_values(['Counts'], ascending=False)

tag_counts = tag_df_sorted['Counts'].values
plt.plot(tag_counts)

plt.title("Distribution of number of times tag appeared questions")

plt.grid()

plt.xlabel("Tag number")

plt.ylabel("Number of times tag appeared")

plt.show()
plt.plot(tag_counts[0:10000])

plt.title('first 10k tags: Distribution of number of times tag appeared questions')

plt.grid()

plt.xlabel("Tag number")

plt.ylabel("Number of times tag appeared")

plt.show()

print(len(tag_counts[0:10000:25]), tag_counts[0:10000:25])
plt.plot(tag_counts[0:1000])

plt.title('first 1k tags: Distribution of number of times tag appeared questions')

plt.grid()

plt.xlabel("Tag number")

plt.ylabel("Number of times tag appeared")

plt.show()

print(len(tag_counts[0:1000:5]), tag_counts[0:1000:5])
plt.plot(tag_counts[0:500])

plt.title('first 500 tags: Distribution of number of times tag appeared questions')

plt.grid()

plt.xlabel("Tag number")

plt.ylabel("Number of times tag appeared")

plt.show()

print(len(tag_counts[0:500:5]), tag_counts[0:500:5])
plt.plot(tag_counts[0:100], c='b')

plt.scatter(x=list(range(0,100,5)), y=tag_counts[0:100:5], c='orange', label="quantiles with 0.05 intervals")

# quantiles with 0.25 difference

plt.scatter(x=list(range(0,100,25)), y=tag_counts[0:100:25], c='m', label = "quantiles with 0.25 intervals")



for x,y in zip(list(range(0,100,25)), tag_counts[0:100:25]):

    plt.annotate(s="({} , {})".format(x,y), xy=(x,y), xytext=(x-0.05, y+500))



plt.title('first 100 tags: Distribution of number of times tag appeared questions')

plt.grid()

plt.xlabel("Tag number")

plt.ylabel("Number of times tag appeared")

plt.legend()

plt.show()

print(len(tag_counts[0:100:5]), tag_counts[0:100:5])
# Store tags greater than 10K in one list

lst_tags_gt_10k = tag_df[tag_df.Counts>10000].Tags

#Print the length of the list

print ('{} Tags are used more than 10000 times'.format(len(lst_tags_gt_10k)))

# Store tags greater than 100K in one list

lst_tags_gt_100k = tag_df[tag_df.Counts>100000].Tags

#Print the length of the list.

print ('{} Tags are used more than 100000 times'.format(len(lst_tags_gt_100k)))
#Storing the count of tag in each question in list 'tag_count'

tag_quest_count = tag_dtm.sum(axis=1).tolist()

#Converting list of lists into single list, we will get [[3], [4], [2], [2], [3]] and we are converting this to [3, 4, 2, 2, 3]

tag_quest_count=[int(j) for i in tag_quest_count for j in i]

print ('We have total {} datapoints.'.format(len(tag_quest_count)))



print(tag_quest_count[:5])
print( "Maximum number of tags per question: %d"%max(tag_quest_count))

print( "Minimum number of tags per question: %d"%min(tag_quest_count))

print( "Avg. number of tags per question: %f"% ((sum(tag_quest_count)*1.0)/len(tag_quest_count)))
sns.countplot(tag_quest_count, palette='gist_rainbow')

plt.title("Number of tags in the questions ")

plt.xlabel("Number of Tags")

plt.ylabel("Number of questions")

plt.show()
# Ploting word cloud

start = datetime.now()



# Lets first convert the 'result' dictionary to 'list of tuples'

tup = dict(result.items())

#Initializing WordCloud using frequencies of tags.

wordcloud = WordCloud(    background_color='black',

                          width=1600,

                          height=800,

                    ).generate_from_frequencies(tup)



fig = plt.figure(figsize=(30,20))

plt.imshow(wordcloud)

plt.axis('off')

plt.tight_layout(pad=0)

fig.savefig("tag.png")

plt.show()

print("Time taken to run this cell :", datetime.now() - start)
i=np.arange(30)

tag_df_sorted.head(30).plot(kind='bar')

plt.title('Frequency of top 20 tags')

plt.xticks(i, tag_df_sorted['Tags'])

plt.xlabel('Tags')

plt.ylabel('Counts')

plt.show()
def striphtml(data):

    cleanr = re.compile('<.*?>')

    cleantext = re.sub(cleanr, ' ', str(data))

    return cleantext

stop_words = set(stopwords.words('english'))

stemmer = SnowballStemmer("english")
#http://www.bernzilla.com/2008/05/13/selecting-a-random-row-from-an-sqlite-table/

# with more weights to title

start = datetime.now()

qus_list=[]

qus_with_code = 0

len_before_preprocessing = 0 

len_after_preprocessing = 0 

for index,row in df.iterrows():

    title, body, tags = row["Title"], row["Body"], row["Tags"]

    if '<code>' in body:

        qus_with_code+=1

    len_before_preprocessing+=len(title) + len(body)

    body=re.sub('<code>(.*?)</code>', '', body, flags=re.MULTILINE|re.DOTALL)

    body = re.sub('<.*?>', ' ', str(body.encode('utf-8')))

    title=title.encode('utf-8')

    question=str(title)+" "+str(title)+" "+str(title)+" "+ body

    question=re.sub(r'[^A-Za-z]+',' ',question)

    words=word_tokenize(str(question.lower()))

    question=' '.join(str(stemmer.stem(j)) for j in words if j not in stop_words and (len(j)!=1 or j=='c'))

    qus_list.append(question)

    len_after_preprocessing += len(question)

df["question_with_more_wt_title"] = qus_list

avg_len_before_preprocessing=(len_before_preprocessing*1.0)/df.shape[0]

avg_len_after_preprocessing=(len_after_preprocessing*1.0)/df.shape[0]

print( "Avg. length of questions(Title+Body) before preprocessing: ", avg_len_before_preprocessing)

print( "Avg. length of questions(Title+Body) after preprocessing: ", avg_len_after_preprocessing)

print ("% of questions containing code: ", (qus_with_code*100.0)/df.shape[0])

print("Time taken to run this cell :", datetime.now() - start)


df_no_dup["question"] = qus_list

avg_len_before_preprocessing=(len_before_preprocessing*1.0)/df_no_dup.shape[0]

avg_len_after_preprocessing=(len_after_preprocessing*1.0)/df_no_dup.shape[0]

print( "Avg. length of questions(Title+Body) before preprocessing: ", avg_len_before_preprocessing)

print( "Avg. length of questions(Title+Body) after preprocessing: ", avg_len_after_preprocessing)

print ("% of questions containing code: ", (qus_with_code*100.0)/df_no_dup.shape[0])
preprocessed_data = df_no_dup[['question','Tags']]

preprocessed_data.head()


# import pickle

# with open('preprocessed.pkl', 'wb') as f:

#     pickle.dump(preprocessed_data, f)

# with open('preprocessed.pkl', 'rb') as f:

#     preprocessed_data = pickle.load(f)
print("number of data points in sample :", preprocessed_data.shape[0])

print("number of dimensions :", preprocessed_data.shape[1])
# binary='true' will give a binary vectorizer

vectorizer = CountVectorizer(tokenizer = lambda x: x.split(), binary='true')

multilabel_y = vectorizer.fit_transform(preprocessed_data['Tags'])
def tags_to_choose(n):

    t = multilabel_y.sum(axis=0).tolist()[0]

    sorted_tags_i = sorted(range(len(t)), key=lambda i: t[i], reverse=True)

    multilabel_yn=multilabel_y[:,sorted_tags_i[:n]]

    return multilabel_yn



def questions_explained_fn(n):

    multilabel_yn = tags_to_choose(n)

    x= multilabel_yn.sum(axis=1)

    return (np.count_nonzero(x==0))
questions_explained = []

total_tags=multilabel_y.shape[1]

total_qs=preprocessed_data.shape[0]

for i in range(500, total_tags, 100):

    questions_explained.append(np.round(((total_qs-questions_explained_fn(i))/total_qs)*100,3))
fig, ax = plt.subplots()

ax.plot(questions_explained)

xlabel = list(500+np.array(range(-50,450,50))*50)

ax.set_xticklabels(xlabel)

plt.xlabel("Number of tags")

plt.ylabel("Number Questions coverd partially")

plt.grid()

plt.show()

# you can choose any number of tags based on your computing power, minimun is 50(it covers 90% of the tags)

print("with ",5500,"tags we are covering ",questions_explained[50],"% of questions")
multilabel_yx = tags_to_choose(5500)

print("number of questions that are not covered :", questions_explained_fn(5500),"out of ", total_qs)
print("Number of tags in sample :", multilabel_y.shape[1])

print("number of tags taken :", multilabel_yx.shape[1],"(",(multilabel_yx.shape[1]/multilabel_y.shape[1])*100,"%)")
total_size=preprocessed_data.shape[0]

train_size=int(0.80*total_size)



x_train=preprocessed_data.head(train_size)

x_test=preprocessed_data.tail(total_size - train_size)



y_train = multilabel_yx[0:train_size,:]

y_test = multilabel_yx[train_size:total_size,:]
print("Number of data points in train data :", y_train.shape)

print("Number of data points in test data :", y_test.shape)
start = datetime.now()

vectorizer = TfidfVectorizer(min_df=0.00009, max_features=200000, smooth_idf=True, norm="l2", \

                             tokenizer = lambda x: x.split(), sublinear_tf=False, ngram_range=(1,4))

x_train_multilabel = vectorizer.fit_transform(x_train['question'])

x_test_multilabel = vectorizer.transform(x_test['question'])

print("Time taken to run this cell :", datetime.now() - start)
print("Dimensions of train data X:",x_train_multilabel.shape, "Y :",y_train.shape)

print("Dimensions of test data X:",x_test_multilabel.shape,"Y:",y_test.shape)
# https://www.analyticsvidhya.com/blog/2017/08/introduction-to-multi-label-classification/

#https://stats.stackexchange.com/questions/117796/scikit-multi-label-classification

# classifier = LabelPowerset(GaussianNB())

"""

from skmultilearn.adapt import MLkNN

classifier = MLkNN(k=21)



# train

classifier.fit(x_train_multilabel, y_train)



# predict

predictions = classifier.predict(x_test_multilabel)

print(accuracy_score(y_test,predictions))

print(metrics.f1_score(y_test, predictions, average = 'macro'))

print(metrics.f1_score(y_test, predictions, average = 'micro'))

print(metrics.hamming_loss(y_test,predictions))



"""

# we are getting memory error because the multilearn package 

# is trying to convert the data into dense matrix

# ---------------------------------------------------------------------------

#MemoryError                               Traceback (most recent call last)

#<ipython-input-170-f0e7c7f3e0be> in <module>()

#----> classifier.fit(x_train_multilabel, y_train)
# this will be taking so much time try not to run it, download the lr_with_equal_weight.pkl file and use to predict

# This takes about 6-7 hours to run.

classifier = OneVsRestClassifier(SGDClassifier(loss='hinge', alpha=0.00001, penalty='l1'), n_jobs=-1)

classifier.fit(x_train_multilabel, y_train)

predictions = classifier.predict(x_test_multilabel)



print("accuracy :",metrics.accuracy_score(y_test,predictions))

print("macro f1 score :",metrics.f1_score(y_test, predictions, average = 'macro'))

print("micro f1 scoore :",metrics.f1_score(y_test, predictions, average = 'micro'))

print("hamming loss :",metrics.hamming_loss(y_test,predictions))

print("Precision recall report :\n",metrics.classification_report(y_test, predictions))

df_test = pd.read_csv('/kaggle/input/facebook-recruiting-iii-keyword-extraction/Test.zip', nrows=10000)

df_test.head()

# x_test_multilabel

# predictions
#http://www.bernzilla.com/2008/05/13/selecting-a-random-row-from-an-sqlite-table/

# with more weights to title

start = datetime.now()

qus_list=[]

qus_with_code = 0

len_before_preprocessing = 0 

len_after_preprocessing = 0 

for index,row in df_test.iterrows():

    title, body = row["Title"], row["Body"]#, row["Tags"]

    if '<code>' in body:

        qus_with_code+=1

    len_before_preprocessing+=len(title) + len(body)

    body=re.sub('<code>(.*?)</code>', '', body, flags=re.MULTILINE|re.DOTALL)

    body = re.sub('<.*?>', ' ', str(body.encode('utf-8')))

    title=title.encode('utf-8')

    question=str(title)+" "+str(title)+" "+str(title)+" "+ body

    question=re.sub(r'[^A-Za-z]+',' ',question)

    words=word_tokenize(str(question.lower()))

    question=' '.join(str(stemmer.stem(j)) for j in words if j not in stop_words and (len(j)!=1 or j=='c'))

    qus_list.append(question)

    len_after_preprocessing += len(question)

df_test["question_with_more_wt_title"] = qus_list

avg_len_before_preprocessing=(len_before_preprocessing*1.0)/df_test.shape[0]

avg_len_after_preprocessing=(len_after_preprocessing*1.0)/df_test.shape[0]

print( "Avg. length of questions(Title+Body) before preprocessing: ", avg_len_before_preprocessing)

print( "Avg. length of questions(Title+Body) after preprocessing: ", avg_len_after_preprocessing)

print ("% of questions containing code: ", (qus_with_code*100.0)/df_test.shape[0])

print("Time taken to run this cell :", datetime.now() - start)
# df_test[['question_with_more_wt_title','Id']]

df_test.head()
preprocessed_data = df_test[["question_with_more_wt_title",'Id']]

print("Shape of preprocessed data :", preprocessed_data.shape)
preprocessed_data.head()
print("number of data points in sample :", preprocessed_data.shape[0])

print("number of dimensions :", preprocessed_data.shape[1])
# vectorizer = TfidfVectorizer(min_df=0.00009, max_features=200000, smooth_idf=True, norm="l2", \

#                              tokenizer = lambda x: x.split(), sublinear_tf=False, ngram_range=(1,4))

# X_train_multilabel = vectorizer.fit_transform(X_train['question_with_more_wt_title'])

X_test_multilabel = vectorizer.transform(preprocessed_data['question_with_more_wt_title'])
y_pred = classifier.predict(X_test_multilabel)
Y_pred = classifier.predict(x_test_multilabel)

Y_pred