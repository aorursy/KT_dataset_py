import pandas as pd

import numpy as np

from IPython.display import HTML

import seaborn as sns

import base64

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn import metrics

import re

from imblearn.over_sampling import SMOTE

import nltk

from nltk.sentiment.vader import SentimentIntensityAnalyzer

from sklearn.model_selection import GridSearchCV

from sklearn import preprocessing

from nltk.stem import PorterStemmer, WordNetLemmatizer

from nltk.corpus import stopwords

from sklearn.feature_extraction.text import TfidfTransformer

from sklearn.feature_extraction.text import CountVectorizer



#import warnings

#warnings.filterwarnings("ignore")
data = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')



print(data.shape)

print(test.shape)
cnt_srs = data['type'].value_counts()



plt.figure(figsize=(12,4))

sns.barplot(cnt_srs.index, cnt_srs.values, alpha=0.8)

plt.ylabel('Number of Occurrences', fontsize=12)

plt.xlabel('Types', fontsize=12)

plt.show()
data['mind'] = data['type'].apply(lambda x: x[0] == 'E').astype('int')

data['energy'] = data['type'].apply(lambda x: x[1] == 'N').astype('int')

data['nature'] = data['type'].apply(lambda x: x[2] == 'T').astype('int')

data['tactics'] = data['type'].apply(lambda x: x[3] == 'J').astype('int')



response = data[['mind','energy','nature','tactics']]



predictors = data.posts



print(predictors.shape)

print(response.shape)
df1 = data['posts']

df2 = test['posts']

frames = [df1,df2]

result = pd.concat(frames)

data = pd.DataFrame(result)

data.columns = ['post']

data.shape
def avg_word(sentence):

  words = sentence.split()

  return (sum(len(word) for word in words)/len(words))
stemmer = PorterStemmer()

lemmatiser = WordNetLemmatizer()

cachedStopWords = set(stopwords.words('english'))



def pre_process_data(data, remove_stop_words=True):

    avg_words = []

    stopwords = []

    list_posts = []

    len_data = len(data)

    i=0

    

    for row in data.iterrows():           

        posts = row[1].post

        temp = re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', posts)

        temp = re.sub("[^a-zA-Z]", " ", temp)

        temp = re.sub(' +', ' ', temp).lower()

        

        avg_words.append(avg_word(temp))

        

        if remove_stop_words:

            temp = " ".join([lemmatiser.lemmatize(w) for w in temp.split(' ') if w not in cachedStopWords])

        else:

            temp = " ".join([lemmatiser.lemmatize(w) for w in temp.split(' ')])

        

        list_posts.append(temp)



    list_posts = np.array(list_posts)

    return list_posts,avg_words



list_posts,avg_words = pre_process_data(data, remove_stop_words=True)

len(list_posts)
i=0

E_count=[]

I_count=[]

S_count=[]

N_count=[]

F_count=[]

T_count=[]

P_count=[]

J_count=[]



while (i<len(list_posts)):

    a=re.findall(r'\bentj|\bintp|\bistp|\bisfj|\besfj|\bisfp|\binfp|\benfp|\bistj|\bestj|\besfp|\bentp|\bestp|\benfj|\binfj|\bintj',list_posts[i].lower())



    count_i=0

    count_e=0

    count_s=0

    count_n=0

    count_f=0

    count_t=0

    count_p=0

    count_j=0



    for d,j in enumerate(a):

        if j[0]=='e':

            count_e+=1

        if j[0]=='i':

            count_i+=1

        if j[1]=='s':

            count_s+=1

        if j[1]=='n':

            count_n+=1

        if j[2]=='f':

            count_f+=1

        if j[2]=='t':

            count_t+=1

        if j[3]=='p':

            count_p+=1

        if j[3]=='j':

            count_j+=1



    E_count.append(count_e)

    I_count.append(count_i)

    S_count.append(count_s)

    N_count.append(count_n)

    F_count.append(count_f)

    T_count.append(count_t)

    P_count.append(count_p)

    J_count.append(count_j)

    

    i=i+1
analyser = SentimentIntensityAnalyzer()

    

def sentiment_analyzer_scores(sentence):

    return analyser.polarity_scores(sentence)



def Analyser(list_in):

    len_data = len(list_in)

    i=0

    neg = []

    neu = []

    pos = []

    compound = []

    for j in range(0,len_data):

        a = sentiment_analyzer_scores(list_in[j])

        b = list(a.values())

        neg.append(b[0])

        neu.append(b[1])

        pos.append(b[2])

        compound.append(b[3])

    return neg,neu,pos,compound



neg,neu,pos,compound = Analyser(list_posts)
cntizer = CountVectorizer(analyzer="word", max_features=5000, tokenizer=None,preprocessor=None,stop_words=None,ngram_range=(1,2)) 



tfizer = TfidfTransformer()



X_cnt = cntizer.fit_transform(list_posts)

X_tfidf =  tfizer.fit_transform(X_cnt).toarray()

print(X_tfidf.shape)
#Only Positive and Negative

data = pd.DataFrame(X_tfidf)

data['neg'] = pd.DataFrame({'neg':neg})

#data['neu'] = pd.DataFrame({'neu':neu})

data['pos'] = pd.DataFrame({'pos':pos})

#data['compound'] = pd.DataFrame({'compound':compound})



data['E_count'] = pd.DataFrame({'E_count':E_count})

data['I_count'] = pd.DataFrame({'I_count':I_count})

data['S_count'] = pd.DataFrame({'S_count':S_count})

data['N_count'] = pd.DataFrame({'N_count':N_count})

data['F_count'] = pd.DataFrame({'F_count':F_count})

data['T_count'] = pd.DataFrame({'T_count':T_count})

data['P_count'] = pd.DataFrame({'P_count':P_count})

data['J_count'] = pd.DataFrame({'J_count':J_count})

data['avg_words'] = pd.DataFrame({'avg_words':avg_words})



min_max_scaler = preprocessing.MinMaxScaler()

data[['E_count','I_count','S_count','N_count','F_count','T_count','P_count','J_count','avg_words']] = min_max_scaler.fit_transform(data[['E_count','I_count','S_count','N_count','F_count','T_count','P_count','J_count','avg_words']])



data.shape
list_posts_train = data.iloc[0:6506]

list_posts_test = data.iloc[6506:]

y_test = pd.DataFrame(list_posts_test)



print(response.shape)

print(list_posts_train.shape)

print(y_test.shape)
reverse_dic = {}

for key in cntizer.vocabulary_:

    reverse_dic[cntizer.vocabulary_[key]] = key

    

top_50 = np.asarray(np.argsort(np.sum(X_cnt, axis=0))[0,-50:][0, ::-1]).flatten()

Top50 = [reverse_dic[v] for v in top_50]



word_list = cntizer.get_feature_names()

count_list = X_cnt.toarray().sum(axis=0)

count_list = sorted(count_list,reverse = True)

count_list = count_list[0:50]





plt.figure(figsize=(12,8))

plt.xticks(rotation=90)

plt.bar(Top50,count_list)
list1 = ['mind','energy','nature','tactics']

for columns in list1:

    cnt_srs = response[columns].value_counts()

    plt.figure(figsize=(5,3))

    sns.barplot(cnt_srs.index, cnt_srs.values, alpha=0.8)

    plt.ylabel('Number of Occurrences', fontsize=12)

    plt.xlabel('Types', fontsize=12)

    plt.show()
list1 = ['mind','energy','nature','tactics']

results = pd.DataFrame()

results1 = pd.DataFrame()

for columns in list1:

    

    X =list_posts_train

    y = response[columns]

    sm = SMOTE(random_state=27, ratio=1.0)

    X_train, y_train = sm.fit_sample(X, y)

    

    logreg = LogisticRegression(random_state=42, solver='lbfgs')

    logreg.fit(X_train, y_train)    

    y_pred_test = logreg.predict_proba(y_test)

    results[columns] = y_pred_test[:,1]

    

    y_pred_test1 = logreg.predict(y_test)

    results1[columns] = y_pred_test1
results = results.reset_index()

results.columns = ['id','mind','energy','nature','tactics']

results.id = results.id + 1

results = results.set_index('id')

results.to_csv('results.csv')
results1 = results1.reset_index()

results1.columns = ['id','mind','energy','nature','tactics']

results1.id = results1.id + 1

results1 = results1.set_index('id')

results1.to_csv('results1.csv')