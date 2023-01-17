import warnings

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
#Creating db file from csv



if not os.path.isfile('train.db'):

    start = datetime.now()

    disk_engine = create_engine('sqlite:///train.db')

    start = dt.datetime.now()

    chunksize = 180000

    j = 0

    index_start = 1

    for df in pd.read_csv('Train.csv', names=['Id', 'Title', 'Body', 'Tags'], chunksize=chunksize, iterator=True, encoding='utf-8', ):

        df.index += index_start

        j+=1

        print('{} rows'.format(j*chunksize))

        df.to_sql('data', disk_engine, if_exists='append')

        index_start = df.index[-1] + 1

    print("Time taken to run this cell :", datetime.now() - start)
if os.path.isfile('train.db'):

    start = datetime.now()

    con = sqlite3.connect('train.db')

    num_rows = pd.read_sql_query("""SELECT count(*) FROM data""", con)

    print("Number of rows in the database :","\n",num_rows['count(*)'].values[0])

    con.close()

    print("Time taken to count the number of rows :", datetime.now() - start)

else:

    print("Please download the train.db file from drive or run the above cell to genarate train.db file")
if os.path.isfile('train.db'):

    start = datetime.now()

    con = sqlite3.connect('train.db')

    df_no_dup = pd.read_sql_query('SELECT Title, Body, Tags, COUNT(*) as cnt_dup FROM data GROUP BY Title, Body, Tags', con)

    con.close()

    print("Time taken to run this cell :", datetime.now() - start)

else:

    print("Please download the train.db file from drive or run the first to genarate train.db file")
df_no_dup.head()

# we can observe that there are duplicates
print("number of duplicate questions :", num_rows['count(*)'].values[0]- df_no_dup.shape[0], "(",(1-((df_no_dup.shape[0])/(num_rows['count(*)'].values[0])))*100,"% )")
# number of times each question appeared in our database

df_no_dup.cnt_dup.value_counts()
start = datetime.now()

df_no_dup["tag_count"] = df_no_dup["Tags"].apply(lambda text: len(text.split(" ")))

# adding a new feature number of tags per question

print("Time taken to run this cell :", datetime.now() - start)

df_no_dup.head()
# distribution of number of tags per question

df_no_dup.tag_count.value_counts()
#Creating a new database with no duplicates

if not os.path.isfile('train_no_dup.db'):

    disk_dup = create_engine("sqlite:///train_no_dup.db")

    no_dup = pd.DataFrame(df_no_dup, columns=['Title', 'Body', 'Tags'])

    no_dup.to_sql('no_dup_train',disk_dup)
#This method seems more appropriate to work with this much data.

if os.path.isfile('train_no_dup.db'):

    start = datetime.now()

    con = sqlite3.connect('train_no_dup.db')

    tag_data = pd.read_sql_query("""SELECT Tags FROM no_dup_train""", con)

    con.close()



    # Let's now drop unwanted column.

    tag_data.drop(tag_data.index[0], inplace=True)

    #Printing first 5 columns from our data frame

    tag_data.head()

    print("Time taken to run this cell :", datetime.now() - start)

else:

    print("Please download the train.db file from drive or run the above cells to genarate train.db file")
# Importing & Initializing the "CountVectorizer" object, which is scikit-learn's bag of words tool.



#by default 'split()' will tokenize each tag using space.

vectorizer = CountVectorizer(tokenizer = lambda x: x.split())

# fit_transform() does two functions: First, it fits the model

# and learns the vocabulary; second, it transforms our training data

# into feature vectors. The input to fit_transform should be a list of strings.

tag_dtm = vectorizer.fit_transform(tag_data['Tags'])
print("Number of data points :", tag_dtm.shape[0])

print("Number of unique tags :", tag_dtm.shape[1])
#'get_feature_name()' gives us the vocabulary.

tags = vectorizer.get_feature_names()

#Lets look at the tags we have.

print("Some of the tags we have :", tags[:10])
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

#Converting each value in the 'tag_quest_count' to integer.

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
def create_connection(db_file):

    """ create a database connection to the SQLite database

        specified by db_file

    :param db_file: database file

    :return: Connection object or None

    """

    try:

        conn = sqlite3.connect(db_file)

        return conn

    except Error as e:

        print(e)

 

    return None



def create_table(conn, create_table_sql):

    """ create a table from the create_table_sql statement

    :param conn: Connection object

    :param create_table_sql: a CREATE TABLE statement

    :return:

    """

    try:

        c = conn.cursor()

        c.execute(create_table_sql)

    except Error as e:

        print(e)

        

def checkTableExists(dbcon):

    cursr = dbcon.cursor()

    str = "select name from sqlite_master where type='table'"

    table_names = cursr.execute(str)

    print("Tables in the databse:")

    tables =table_names.fetchall() 

    print(tables[0][0])

    return(len(tables))



def create_database_table(database, query):

    conn = create_connection(database)

    if conn is not None:

        create_table(conn, query)

        checkTableExists(conn)

    else:

        print("Error! cannot create the database connection.")

    conn.close()



sql_create_table = """CREATE TABLE IF NOT EXISTS QuestionsProcessed (question text NOT NULL, code text, tags text, words_pre integer, words_post integer, is_code integer);"""

create_database_table("Processed.db", sql_create_table)
start = datetime.now()

read_db = 'train_no_dup.db'

write_db = 'Processed.db'

if os.path.isfile(read_db):

    conn_r = create_connection(read_db)

    if conn_r is not None:

        reader =conn_r.cursor()

        reader.execute("SELECT Title, Body, Tags From no_dup_train ORDER BY RANDOM() LIMIT 1000000;")



if os.path.isfile(write_db):

    conn_w = create_connection(write_db)

    if conn_w is not None:

        tables = checkTableExists(conn_w)

        writer =conn_w.cursor()

        if tables != 0:

            writer.execute("DELETE FROM QuestionsProcessed WHERE 1")

            print("Cleared All the rows")

print("Time taken to run this cell :", datetime.now() - start)
start = datetime.now()

preprocessed_data_list=[]

reader.fetchone()

questions_with_code=0

len_pre=0

len_post=0

questions_proccesed = 0

for row in reader:



    is_code = 0



    title, question, tags = row[0], row[1], row[2]



    if '<code>' in question:

        questions_with_code+=1

        is_code = 1

    x = len(question)+len(title)

    len_pre+=x



    code = str(re.findall(r'<code>(.*?)</code>', question, flags=re.DOTALL))



    question=re.sub('<code>(.*?)</code>', '', question, flags=re.MULTILINE|re.DOTALL)

    question=striphtml(question.encode('utf-8'))



    title=title.encode('utf-8')



    question=str(title)+" "+str(question)

    question=re.sub(r'[^A-Za-z]+',' ',question)

    words=word_tokenize(str(question.lower()))



    #Removing all single letter and and stopwords from question exceptt for the letter 'c'

    question=' '.join(str(stemmer.stem(j)) for j in words if j not in stop_words and (len(j)!=1 or j=='c'))



    len_post+=len(question)

    tup = (question,code,tags,x,len(question),is_code)

    questions_proccesed += 1

    writer.execute("insert into QuestionsProcessed(question,code,tags,words_pre,words_post,is_code) values (?,?,?,?,?,?)",tup)

    if (questions_proccesed%100000==0):

        print("number of questions completed=",questions_proccesed)



no_dup_avg_len_pre=(len_pre*1.0)/questions_proccesed

no_dup_avg_len_post=(len_post*1.0)/questions_proccesed



print( "Avg. length of questions(Title+Body) before processing: %d"%no_dup_avg_len_pre)

print( "Avg. length of questions(Title+Body) after processing: %d"%no_dup_avg_len_post)

print ("Percent of questions containing code: %d"%((questions_with_code*100.0)/questions_proccesed))



print("Time taken to run this cell :", datetime.now() - start)
# close the connections, or else you will end up with locks

conn_r.commit()

conn_w.commit()

conn_r.close()

conn_w.close()
if os.path.isfile(write_db):

    conn_r = create_connection(write_db)

    if conn_r is not None:

        reader =conn_r.cursor()

        reader.execute("SELECT question From QuestionsProcessed LIMIT 10")

        print("Questions after preprocessed")

        print('='*100)

        reader.fetchone()

        for row in reader:

            print(row)

            print('-'*100)

conn_r.commit()

conn_r.close()
#Taking 1 Million entries to a dataframe.

write_db = 'Processed.db'

if os.path.isfile(write_db):

    conn_r = create_connection(write_db)

    if conn_r is not None:

        preprocessed_data = pd.read_sql_query("""SELECT question, Tags FROM QuestionsProcessed""", conn_r)

conn_r.commit()

conn_r.close()
preprocessed_data.head()
print("number of data points in sample :", preprocessed_data.shape[0])

print("number of dimensions :", preprocessed_data.shape[1])
# binary='true' will give a binary vectorizer

vectorizer = CountVectorizer(tokenizer = lambda x: x.split(), binary='true')

multilabel_y = vectorizer.fit_transform(preprocessed_data['tags'])
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

# choose any number of tags based on the computing power, minimun is 50(it covers 90% of the tags)

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

                             tokenizer = lambda x: x.split(), sublinear_tf=False, ngram_range=(1,3))

x_train_multilabel = vectorizer.fit_transform(x_train['question'])

x_test_multilabel = vectorizer.transform(x_test['question'])

print("Time taken to run this cell :", datetime.now() - start)
print("Dimensions of train data X:",x_train_multilabel.shape, "Y :",y_train.shape)

print("Dimensions of test data X:",x_test_multilabel.shape,"Y:",y_test.shape)
# This takes about 6-7 hours to run.

classifier = OneVsRestClassifier(SGDClassifier(loss='log', alpha=0.00001, penalty='l1'), n_jobs=-1)

classifier.fit(x_train_multilabel, y_train)

predictions = classifier.predict(x_test_multilabel)



print("accuracy :",metrics.accuracy_score(y_test,predictions))

print("macro f1 score :",metrics.f1_score(y_test, predictions, average = 'macro'))

print("micro f1 scoore :",metrics.f1_score(y_test, predictions, average = 'micro'))

print("hamming loss :",metrics.hamming_loss(y_test,predictions))

print("Precision recall report :\n",metrics.classification_report(y_test, predictions))

from sklearn.externals import joblib

joblib.dump(classifier, 'lr_with_equal_weight.pkl') 
sql_create_table = """CREATE TABLE IF NOT EXISTS QuestionsProcessed (question text NOT NULL, code text, tags text, words_pre integer, words_post integer, is_code integer);"""

create_database_table("Titlemoreweight.db", sql_create_table)
read_db = 'train_no_dup.db'

write_db = 'Titlemoreweight.db'

train_datasize = 400000

if os.path.isfile(read_db):

    conn_r = create_connection(read_db)

    if conn_r is not None:

        reader =conn_r.cursor()

        # for selecting first 0.5M rows

        reader.execute("SELECT Title, Body, Tags From no_dup_train LIMIT 500001;")

        # for selecting random points

        #reader.execute("SELECT Title, Body, Tags From no_dup_train ORDER BY RANDOM() LIMIT 500001;")



if os.path.isfile(write_db):

    conn_w = create_connection(write_db)

    if conn_w is not None:

        tables = checkTableExists(conn_w)

        writer =conn_w.cursor()

        if tables != 0:

            writer.execute("DELETE FROM QuestionsProcessed WHERE 1")

            print("Cleared All the rows")
start = datetime.now()

preprocessed_data_list=[]

reader.fetchone()

questions_with_code=0

len_pre=0

len_post=0

questions_proccesed = 0

for row in reader:

    

    is_code = 0

    

    title, question, tags = row[0], row[1], str(row[2])

    

    if '<code>' in question:

        questions_with_code+=1

        is_code = 1

    x = len(question)+len(title)

    len_pre+=x

    

    code = str(re.findall(r'<code>(.*?)</code>', question, flags=re.DOTALL))

    

    question=re.sub('<code>(.*?)</code>', '', question, flags=re.MULTILINE|re.DOTALL)

    question=striphtml(question.encode('utf-8'))

    

    title=title.encode('utf-8')

    

    # adding title three time to the data to increase its weight

    # add tags string to the training data

    

    question=str(title)+" "+str(title)+" "+str(title)+" "+question

    

#     if questions_proccesed<=train_datasize:

#         question=str(title)+" "+str(title)+" "+str(title)+" "+question+" "+str(tags)

#     else:

#         question=str(title)+" "+str(title)+" "+str(title)+" "+question



    question=re.sub(r'[^A-Za-z0-9#+.\-]+',' ',question)

    words=word_tokenize(str(question.lower()))

    

    #Removing all single letter and and stopwords from question exceptt for the letter 'c'

    question=' '.join(str(stemmer.stem(j)) for j in words if j not in stop_words and (len(j)!=1 or j=='c'))

    

    len_post+=len(question)

    tup = (question,code,tags,x,len(question),is_code)

    questions_proccesed += 1

    writer.execute("insert into QuestionsProcessed(question,code,tags,words_pre,words_post,is_code) values (?,?,?,?,?,?)",tup)

    if (questions_proccesed%100000==0):

        print("number of questions completed=",questions_proccesed)



no_dup_avg_len_pre=(len_pre*1.0)/questions_proccesed

no_dup_avg_len_post=(len_post*1.0)/questions_proccesed



print( "Avg. length of questions(Title+Body) before processing: %d"%no_dup_avg_len_pre)

print( "Avg. length of questions(Title+Body) after processing: %d"%no_dup_avg_len_post)

print ("Percent of questions containing code: %d"%((questions_with_code*100.0)/questions_proccesed))



print("Time taken to run this cell :", datetime.now() - start)
# closing the connections

conn_r.commit()

conn_w.commit()

conn_r.close()

conn_w.close()
if os.path.isfile(write_db):

    conn_r = create_connection(write_db)

    if conn_r is not None:

        reader =conn_r.cursor()

        reader.execute("SELECT question From QuestionsProcessed LIMIT 10")

        print("Questions after preprocessed")

        print('='*100)

        reader.fetchone()

        for row in reader:

            print(row)

            print('-'*100)

conn_r.commit()

conn_r.close()
#Taking 0.5 Million entries to a dataframe.

write_db = 'Titlemoreweight.db'

if os.path.isfile(write_db):

    conn_r = create_connection(write_db)

    if conn_r is not None:

        preprocessed_data = pd.read_sql_query("""SELECT question, Tags FROM QuestionsProcessed""", conn_r)

conn_r.commit()

conn_r.close()
preprocessed_data.head()
print("number of data points in sample :", preprocessed_data.shape[0])

print("number of dimensions :", preprocessed_data.shape[1])
vectorizer = CountVectorizer(tokenizer = lambda x: x.split(), binary='true')

multilabel_y = vectorizer.fit_transform(preprocessed_data['tags'])
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

# you can choose any number of tags based on your computing power, minimun is 500(it covers 90% of the tags)

print("with ",5500,"tags we are covering ",questions_explained[50],"% of questions")

print("with ",500,"tags we are covering ",questions_explained[0],"% of questions")
# we will be taking 500 tags

multilabel_yx = tags_to_choose(500)

print("number of questions that are not covered :", questions_explained_fn(500),"out of ", total_qs)
x_train=preprocessed_data.head(train_datasize)

x_test=preprocessed_data.tail(preprocessed_data.shape[0] - 400000)



y_train = multilabel_yx[0:train_datasize,:]

y_test = multilabel_yx[train_datasize:preprocessed_data.shape[0],:]
print("Number of data points in train data :", y_train.shape)

print("Number of data points in test data :", y_test.shape)
start = datetime.now()

vectorizer = TfidfVectorizer(min_df=0.00009, max_features=200000, smooth_idf=True, norm="l2", \

                             tokenizer = lambda x: x.split(), sublinear_tf=False, ngram_range=(1,3))

x_train_multilabel = vectorizer.fit_transform(x_train['question'])

x_test_multilabel = vectorizer.transform(x_test['question'])

print("Time taken to run this cell :", datetime.now() - start)
print("Dimensions of train data X:",x_train_multilabel.shape, "Y :",y_train.shape)

print("Dimensions of test data X:",x_test_multilabel.shape,"Y:",y_test.shape)
start = datetime.now()

classifier = OneVsRestClassifier(SGDClassifier(loss='log', alpha=0.00001, penalty='l1'), n_jobs=-1)

classifier.fit(x_train_multilabel, y_train)

predictions = classifier.predict (x_test_multilabel)





print("Accuracy :",metrics.accuracy_score(y_test, predictions))

print("Hamming loss ",metrics.hamming_loss(y_test,predictions))





precision = precision_score(y_test, predictions, average='micro')

recall = recall_score(y_test, predictions, average='micro')

f1 = f1_score(y_test, predictions, average='micro')

 

print("Micro-average quality numbers")

print("Precision: {:.4f}, Recall: {:.4f}, F1-measure: {:.4f}".format(precision, recall, f1))



precision = precision_score(y_test, predictions, average='macro')

recall = recall_score(y_test, predictions, average='macro')

f1 = f1_score(y_test, predictions, average='macro')

 

print("Macro-average quality numbers")

print("Precision: {:.4f}, Recall: {:.4f}, F1-measure: {:.4f}".format(precision, recall, f1))



print (metrics.classification_report(y_test, predictions))

print("Time taken to run this cell :", datetime.now() - start)
joblib.dump(classifier, 'lr_with_more_title_weight.pkl') 
start = datetime.now()

classifier_2 = OneVsRestClassifier(LogisticRegression(penalty='l1'), n_jobs=-1)

classifier_2.fit(x_train_multilabel, y_train)

predictions_2 = classifier_2.predict(x_test_multilabel)

print("Accuracy :",metrics.accuracy_score(y_test, predictions_2))

print("Hamming loss ",metrics.hamming_loss(y_test,predictions_2))





precision = precision_score(y_test, predictions_2, average='micro')

recall = recall_score(y_test, predictions_2, average='micro')

f1 = f1_score(y_test, predictions_2, average='micro')

 

print("Micro-average quality numbers")

print("Precision: {:.4f}, Recall: {:.4f}, F1-measure: {:.4f}".format(precision, recall, f1))



precision = precision_score(y_test, predictions_2, average='macro')

recall = recall_score(y_test, predictions_2, average='macro')

f1 = f1_score(y_test, predictions_2, average='macro')

 

print("Macro-average quality numbers")

print("Precision: {:.4f}, Recall: {:.4f}, F1-measure: {:.4f}".format(precision, recall, f1))



print (metrics.classification_report(y_test, predictions_2))

print("Time taken to run this cell :", datetime.now() - start)