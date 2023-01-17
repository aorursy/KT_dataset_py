# load required packages



# for creating dataframes from csv datasets

import pandas as pd



# for regular expressions

import re



# for stripping stop words

from nltk.corpus import stopwords



# for TF-IDF

from textblob import TextBlob as tb



# for jaccard score

from sklearn.metrics import jaccard_similarity_score



# for removing HTML tags from text body

from html.parser import HTMLParser



# for counting

import collections



# for scientific computing

import numpy as np

import math



# for plotting graphs

import matplotlib.pyplot as plt

import matplotlib.cm as cm

# magic function

%matplotlib inline



# kaggle - data set files are available in the "../input/" directory

dataset_dir = "../input/"

dataset_dir_questions = "Questions.csv"

dataset_dir_answers = "Answers.csv"

dataset_dir_tags = "Tags.csv"



# for offline run

# dataset_dir = "/home/su/Downloads/stacksample"



# list the files in the dataset directory

from subprocess import check_output

print(check_output(["ls", dataset_dir]).decode("utf8"))



cachedStopWords = stopwords.words("english")



class MLStripper(HTMLParser):

    def __init__(self):

        self.reset()

        self.strict = False

        self.convert_charrefs= True

        self.fed = []

    def handle_data(self, d):

        self.fed.append(d)

    def get_data(self):

        return ''.join(self.fed)



def strip_tags(html):

    s = MLStripper()

    s.feed(html)

    return s.get_data()
# tf(word, blob) computes "term frequency" which is the number of times 

# a word appears in a document blob,normalized by dividing by 

# the total number of words in blob. 

# We use TextBlob for breaking up the text into words and getting the word counts.

def tf(word, blob):

    return blob.words.count(word) / len(blob.words)



# n_containing(word, bloblist) returns the number of documents containing word.

# A generator expression is passed to the sum() function.

def n_containing(word, bloblist):

    return sum(1 for blob in bloblist if word in blob.words)



# idf(word, bloblist) computes "inverse document frequency" which measures how common 

# a word is among all documents in bloblist. 

# The more common a word is, the lower its idf. 

# We take the ratio of the total number of documents 

# to the number of documents containing word, then take the log of that. 

# Add 1 to the divisor to prevent division by zero.

def idf(word, bloblist):

    return math.log(len(bloblist) / (1 + n_containing(word, bloblist)))



# tfidf(word, blob, bloblist) computes the TF-IDF score. 

# It is simply the product of tf and idf.

def tfidf(word, blob, bloblist):

    return tf(word, blob) * idf(word, bloblist)
def normalize(str):

    return ' '.join([word for word in re.sub(r'[^\w]', ' ', strip_tags(str)).lower().split() if word not in cachedStopWords and len(word) > 1 and not word.isdigit()])

# Read CSV



# Original Dimensionality - (rows, columns)



# (1264216, 7) 

# Columns (Id, OwnerUserId, CreationDate, ClosedDate, Score, Title, Body)

# frame every 1000th question (resource restraints)

questions_df = pd.read_csv(dataset_dir+dataset_dir_questions, encoding='latin1').iloc[::10000, :]



# (2014516, 6)

# Columns (Id, OwnerUserId, CreationDate, ParentId, Score, Body)

# frame every 1000th answer (resource restraints)

answers_df = pd.read_csv(dataset_dir+dataset_dir_answers, encoding='latin1').iloc[::1000, :]



# (3750994, 2)

# Columns (Id, Tag)

# frame every 1000th tag (resource restraints)

tags_df = pd.read_csv(dataset_dir+dataset_dir_tags, encoding='latin1').iloc[::1000, :]
# Calculate dimensionality

# questions_df.shape 

# answers_df.shape 

# tags_df.shape 



# Sample dataframe - uncomment to view

questions_df.head(10)

# answers_df.head(10)

# tags_df.head(10) 
# Calculate dimensionality

# questions_df.shape 

# answers_df.shape 

# tags_df.shape 



# Sample dataframe - uncomment to view

questions_df.head(10).loc[:, 'Title':'Body']

# answers_df.head(10).loc[:, 'Body':]

# tags_df.head(10) 
# Normalize question body and title

for index, row in questions_df.iterrows():

    questions_df.at[index, 'Body']= normalize(row[6])

    questions_df.at[index, 'Title']= normalize(row[5])



# Normalize answer body

for index, row in answers_df.iterrows():

    answers_df.at[index, 'Body']= normalize(row[5]) 
# Calculate dimensionality

# questions_df.shape 

# answers_df.shape 

# tags_df.shape 



# Sample dataframe - uncomment to view

questions_df.head(10).loc[:, 'Title':'Body']

# answers_df.head(10).loc[:, 'Body':]

# tags_df.head(10) 
tfidf_dict={}

qID_dict={}

bloblist=[]

idlist=[]



for index, row in questions_df.iterrows():

    # also append title to text body

    bloblist.append(tb(row[6]+" "+row[5]))

    idlist.append(row[0])



for i, blob in enumerate(bloblist):

    if i < 5:

        print("Top words in question ID {}".format(idlist[i]))

    scores = {word: tfidf(word, blob, bloblist) for word in blob.words}

    sorted_words = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    for word, score in sorted_words[:5]:

        if i < 5:

            print("\tWord: {}, TF-IDF: {}".format(word, round(score, 5)))

            

        # word dict    

        if word in tfidf_dict:

            tfidf_dict[word].append([idlist[i],round(score, 5)])

        else:

            tfidf_dict[word] = [[idlist[i],round(score, 5)]]

        

        # qID dict

        if idlist[i] in qID_dict:

            qID_dict[idlist[i]].append(word)

        else:

            lst=[]

            lst.append(word)

            qID_dict[idlist[i]]=lst

i = 1

for k, v in tfidf_dict.items():

    print(k, v)

    if i == 10:

        break

    i+=1

    

# i = 1

# for k, v in qID_dict.items():

#     print(k, v)

#     if i == 10:

#         break

#     i+=1
def predict_duplicate(query):

    

    def top_words(text):

        counts = collections.Counter(text.split())

        return [elem for elem, _ in counts.most_common(5)]



    termList=top_words(query)

    

    for k, v in qID_dict.items():

        if jaccard_similarity_score(termList, qID_dict[k]) >= 0.75:

            print("Duplicate Question. Question exists with qID: " + k)

            return

    

print("Not a Duplicate Question.")

                
inputQ_title="What is the most efficient way to deep clone an object in JavaScript?"

inputQ_body="""What is the most efficient way to clone a JavaScript object? 

I've seen obj = eval(uneval(o)); being used, 

but that's non-standard and only supported by Firefox.

I've done things like obj = JSON.parse(JSON.stringify(o)); but question the efficiency. 

I've also seen recursive copying functions with various flaws. 

I'm surprised no canonical solution exists."""

inputQ_tags="javascript, json, object"



# normalize

normalized_query=normalize(inputQ_title + " " + inputQ_body+ " " + inputQ_tags)



# predict whether duplicate question

predict_duplicate(normalized_query)
tags_tally = collections.Counter(tags_df['Tag'])



# x = tag name, y = tag frequency

x, y = zip(*tags_tally.most_common(10))



colormap = plt.cm.gist_ncar #nipy_spectral, Set1,Paired  

colors = [colormap(i) for i in np.linspace(0, 0.8,50)]   



area = [i/3 for i in list(y)]   # 0 to 15 point radiuses

plt.figure(figsize=(8,8))

plt.ylabel("Frequency")

plt.title("Top 10 most common tags")

for i in range(len(y)):

        plt.plot(i,y[i], marker='v', linestyle='',ms=area[i],label=x[i])



plt.legend(numpoints=1)

plt.show()
ans_per_question = collections.Counter(answers_df['ParentId'])

answerid,noAnswers= zip(*ans_per_question.most_common())



N=50

plt.bar(range(N), noAnswers[:N], align='center', alpha=0.7)

#plt.xticks(y_pos, objects)



plt.ylabel('Number of Answers per Question')

plt.xlabel('Question Id')

plt.title('Distribution of Answers per question ')

plt.text(10,1.5,"Average answers per question: "+str(math.floor((np.mean(noAnswers)))))



plt.show()
# -*- coding: utf-8 -*-

"""

Created on Fri Nov  3 19:57:17 2017



@author: RudradeepGuha

"""



from sklearn.naive_bayes import GaussianNB

import numpy as np

import pandas as pd



data = questions_df



X = np.zeros((12643, 2), dtype=int)

Y = np.zeros((12643, 1), dtype=int)

t = data.Title

counter = 0



# For all titles, we count the number of characters and add that to X and depending on the length

# classify them as 0(less likely to be upvoted) or 1(more likely to be upvoted) 

for i in t:

    f1 = len(i) - i.count(" ")

    f2 = data.loc[data['Title'] == i, 'OwnerUserId'].iloc[0]

    X[counter] = np.array([f1, f2])

    score = data.loc[data['Title'] == i, 'Score'].iloc[0]

    if score < 20:

        Y[counter] = 0

    else:

        Y[counter] = 1



print(X)

print(Y)



model = GaussianNB()



model.fit(X, Y)



print(model.predict_proba(np.array([[180, 345768]])))