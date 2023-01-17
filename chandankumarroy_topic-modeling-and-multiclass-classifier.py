# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

# plt.style.use('g')

import seaborn as sns





# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

        





# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train_df = pd.read_csv("../input/janatahack-independence-day-2020-ml-hackathon/train.csv")

train_df.head()
train_df.shape
test_df = pd.read_csv("../input/janatahack-independence-day-2020-ml-hackathon/test.csv")

test_df.head()
test_df.shape
sample_df = pd.read_csv("../input/janatahack-independence-day-2020-ml-hackathon/sample_submission_UVKGLZE.csv")

sample_df.head()
test_id = test_df.ID.values

test_id
plt.style.use('ggplot')
categories = list(train_df.columns.values)[3:]

values = list(train_df.iloc[:,3:].sum(axis =0).values)



sns.set(font_scale =1)

fig = plt.figure(figsize =(10,8))



ax = sns.barplot(categories,values)

ax.set_xlabel("categories")

ax.set_ylabel("number of topics and abstracts")

ax.set_title("total topics in each category")

plt.xticks(rotation=90)





rects = ax.patches

labels = values

for rect, label in zip(rects, labels):

    height = rect.get_height()

    ax.text(rect.get_x() + rect.get_width()/2, height + 5, label, ha='center', va='bottom', fontsize=18)

    

plt.show()
from wordcloud import WordCloud, STOPWORDS
subset = train_df[train_df['Computer Science'] == True]

subset = subset.TITLE.values
stopwords = set(STOPWORDS)

cse_wc = WordCloud(

            background_color='black',

            max_words=3000,

            stopwords= stopwords,

            width =600,height = 400).generate(" ".join(subset))



fig  = plt.figure(figsize=(14,8))

plt.imshow(cse_wc)

plt.axis('off')

plt.show()
import nltk

from nltk.corpus import stopwords

from nltk.stem.snowball import SnowballStemmer

import re

import warnings

import sys



data_train = train_df.copy()

data_test = test_df.copy()





data_train['TITLE'] = data_train[['TITLE','ABSTRACT']].apply(lambda x: " ".join(x),axis =1)

data_test['TITLE'] = data_test[['TITLE','ABSTRACT']].apply(lambda x: " ".join(x),axis =1)



del data_train['ABSTRACT']

del data_test['ABSTRACT']





if not sys.warnoptions:

    warnings.simplefilter("ignore")

        

        
data_train.columns.values
def cleanPunctuation(sentence):

    cl_sentence  = re.sub(r'[?|!|\'|"|#|\_]',r'',sentence)

    cl_sentence = re.sub(r'[.|,|)|(|\|/]',r'',cl_sentence)

    cl_sentence = re.sub(r'\\',r'',cl_sentence)

    cl_sentence = cl_sentence.strip()

    cl_sentence = cl_sentence.replace("\n"," ")

    return cl_sentence



def keepAlphabet(sentence):

    alpha_sent = ""

    for word in sentence.split():

        alpha_word = re.sub('[^a-z A-z]+',' ',word)

        alpha_sent +=alpha_word

        alpha_sent+=" "

    alpha_sent =alpha_sent.strip()

    

    return alpha_sent
data_train['TITLE'] = data_train['TITLE'].str.lower()

data_train['TITLE'] = data_train['TITLE'].str.lower()



data_train['TITLE'] = data_train['TITLE'].apply(cleanPunctuation)

data_test['TITLE'] = data_test['TITLE'].apply(cleanPunctuation)



data_train['TITLE'] = data_train['TITLE'].apply(keepAlphabet)

data_test['TITLE'] = data_test['TITLE'].apply(keepAlphabet)



print("Train data \n",data_train['TITLE'].values[0],"\n")



print("Test data \n",data_test['TITLE'].values[0])



stopwords = set(stopwords.words('english'))

# print(stopwords)

re_stop_words = re.compile(r"\b("+"|".join(stopwords) + ")\\W", re.I)

def removeStopWords(sentence):

    global re_stop_words

    return re_stop_words.sub(" ", sentence)



data_train['TITLE'] = data_train['TITLE'].apply(removeStopWords)

data_test['TITLE'] = data_test['TITLE'].apply(removeStopWords)



print("Train data \n",data_train['TITLE'].values[0])



print("Test data \n",data_test['TITLE'].values[0])



# data_train['TITLE'].values[:10]
from nltk.stem import wordnet

from nltk.stem import WordNetLemmatizer

word_len  = WordNetLemmatizer()
data_new_train = data_train.copy()

data_new_test = data_test.copy()



def stemming(sentence):

    stemSentence =[]

    

    for word in sentence.split():

        stem  = word_len.lemmatize(word)

        stemSentence.append(stem)

        



    return " ".join(stemSentence)



data_new_train['TITLE'] = data_new_train['TITLE'].apply(stemming)

data_new_test['TITLE'] = data_new_test['TITLE'].apply(stemming)



print("Train data \n",data_train['TITLE'].values[0])



print("Test data \n",data_test['TITLE'].values[0])



# data_new['TITLE'].head().values[:10]
data2_train = data_new_train.copy()
from sklearn.model_selection import train_test_split

train,dev = train_test_split(data2_train,random_state =4,test_size = 0.20,shuffle = True)
train.shape
from sklearn.feature_extraction.text import TfidfVectorizer 
col = data2_train.columns.values

col
categories = col[2:]

print(categories)
train_text = train['TITLE']

dev_text = dev['TITLE']

test_text = data_new_test['TITLE']
train_text
vectorizor = TfidfVectorizer(strip_accents='unicode',analyzer='word',ngram_range=(1,2),norm = 'l2')

vectorizor.fit(train_text)

vectorizor.fit(test_text)

vectorizor.fit(dev_text)



x_train= vectorizor.transform(train_text)

x_dev= vectorizor.transform(dev_text)

x_test = vectorizor.transform(test_text)



y_train = train.drop(labels=['ID','TITLE'],axis = 1)

y_dev = dev.drop(labels=['ID','TITLE'],axis = 1)



print(x_train.shape)

print(x_test.shape)

print(x_dev.shape)

print(y_dev.shape)



# x_train.toarray()

# x_dev = x_dev.toarray()

# x_test = x_test.toarray()


from sklearn.linear_model import LogisticRegression

from sklearn.pipeline import Pipeline

from sklearn.metrics import f1_score,accuracy_score

from sklearn.multiclass import OneVsRestClassifier

from sklearn.naive_bayes import MultinomialNB
# categories = [data]

y_train.shape
logReg_pipeline = Pipeline([('oneVrest',OneVsRestClassifier(LogisticRegression(solver ='sag'),n_jobs = -1)),])

pred_arr = []

final_pred = []

for cat in categories:

    print(".....processing.....{} category...".format(cat))

#     print(cat)

    logReg_pipeline.fit(x_train,train[cat])

    pred = logReg_pipeline.predict(x_dev)

    f_pred = logReg_pipeline.predict(x_test)

    pred_arr.append(pred)

    final_pred.append(f_pred)

    

    print("f1 score = {}".format(accuracy_score(dev[cat],pred)))

    print()
y_pred = np.array(pred_arr).T

y_pred.shape

y_dev.shape

print("f1 score for oneVRest classifier = ",f1_score(y_dev,y_pred,average ='micro'))
lst = np.array(final_pred).T

print(lst)
del train_df

del data_new_train

del data_train

del test_df

del data_test



from skmultilearn.problem_transform import LabelPowerset

classifier = LabelPowerset(LogisticRegression(solver= 'sag',n_jobs = -1,class_weight ='balanced'))
# classifier.fit(x_train, y_train)

# pred_lp = classifier.predict(x_dev)

# lst = classifier.predict(x_test)

# print("f1 score =",f1_score(y_dev,pred_lp,average ="micro"))

# lst =lst.toarray()
ID = np.asanyarray(test_id)

df_submit = pd.DataFrame(lst,columns =categories)

df_submit['ID'] = ID

df_submit.head()
col = df_submit.columns.tolist()

col_n = col[-1:]



for ele in range(len(col)-1):

    col_n.append(col[ele])

    

df_submit = df_submit[col_n]

df_submit.head()
compression_opts = dict(method='zip',

                        archive_name='submission4.csv')  

df_submit.to_csv('out13.zip', index=False,

          compression=compression_opts)