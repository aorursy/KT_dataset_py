#import required packages

#basics

import pandas as pd 

import numpy as np



# visualization

import matplotlib.pyplot as plt

import matplotlib.gridspec as gridspec 

import seaborn as sns

from wordcloud import WordCloud ,STOPWORDS

import missingno as msno



import matplotlib.pyplot as plt

from matplotlib_venn import venn2

from matplotlib_venn import venn3



#text cleaning

import re

import string



## nlp

from nltk.tokenize import sent_tokenize, word_tokenize

from nltk.corpus import stopwords



from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, HashingVectorizer



train = pd.read_csv("../input/jigsaw-toxic-comment-classification-challenge/train.csv.zip")

test = pd.read_csv("../input/jigsaw-toxic-comment-classification-challenge/test.csv.zip")
train.shape
#missing value computation

def cal_missing_val(df):

    data_dict = {}

    for col in df.columns:

        data_dict[col] = (df[col].isnull().sum()/df.shape[0])*100

    return pd.DataFrame.from_dict(data_dict, orient='index', columns=['MissingValueInPercentage'])
cal_missing_val(train)
cal_missing_val(test)
rowsums=train.iloc[:,2:].sum(axis=1)

train['clean']=(rowsums==0)*1
xxx=train.iloc[:,2:9].sum()

#plot

plt.figure(figsize=(8,4))

ax= sns.barplot(xxx.index, xxx.values, alpha=0.8)

plt.title("# per class")

plt.ylabel('# of Occurrences', fontsize=12)

plt.xlabel('Type ', fontsize=12)

#adding the text labels

rects = ax.patches

labels = xxx.values

for rect, label in zip(rects, labels):

    height = rect.get_height()

    ax.text(rect.get_x() + rect.get_width()/2, height + 5, label, ha='center', va='bottom')

plt.show()
df1 = pd.DataFrame(train.loc[:,"toxic"].value_counts())

df2 = pd.DataFrame(train.loc[:,"severe_toxic"].value_counts())

df3 = pd.DataFrame(train.loc[:,"obscene"].value_counts())

df4 = pd.DataFrame(train.loc[:,"threat"].value_counts())

df5 = pd.DataFrame(train.loc[:,"insult"].value_counts())

df6 = pd.DataFrame(train.loc[:,"identity_hate"].value_counts())

df_train_distribution = pd.concat([df1,df2,df3,df4,df5,df6], axis = 1)

df_train_distribution
percentage_of_1 = []

for i in range(len(df_train_distribution.columns)):

    a = round(df_train_distribution.iloc[1,i]/(df_train_distribution.iloc[0,i]+df_train_distribution.iloc[1,i])*100,2)

    percentage_of_1.append(a)



percentage_of_1
train_msno = train.replace(1,float("NaN"))

# checking overlaps among 6 columns

msno.matrix(train_msno)
#the larger percentage, larger proportion explained by other factors 



# severe_toxic can be 100% explained by toxic 

print(1-(train[train.severe_toxic == 1].severe_toxic - train[train.severe_toxic == 1].toxic).sum()) 
# identity_hate

number_identity_hate = len(train[train.identity_hate == 1].identity_hate)

print(1-(train[train.identity_hate == 1].identity_hate - train[train.identity_hate == 1].insult).sum()/number_identity_hate) #83% explained by insult

print(1-(train[train.identity_hate == 1].identity_hate - train[train.identity_hate == 1].toxic).sum()/number_identity_hate) #93% expalined by toxic

print(1-(train[train.identity_hate == 1].identity_hate - train[train.identity_hate == 1].severe_toxic).sum()/number_identity_hate) #22% expalined by severe_toxic

print(1-(train[train.identity_hate == 1].identity_hate - train[train.identity_hate == 1].obscene).sum()/number_identity_hate) #73% expalined by obscene
# threat

number_threat = len(train[train.threat == 1].threat)

print(1-(train[train.threat == 1].threat - train[train.threat == 1].insult).sum()/number_threat) #64% explained by insult

print(1-(train[train.threat == 1].threat - train[train.threat == 1].toxic).sum()/number_threat) #94% expalined by toxic

print(1-(train[train.threat == 1].threat - train[train.threat == 1].severe_toxic).sum()/number_threat) #24% expalined by severe_toxic

print(1-(train[train.threat == 1].threat - train[train.threat == 1].obscene).sum()/number_threat) #63% expalined by obscene
train["number_tags"] = train.toxic+train.severe_toxic+train.obscene+train.threat+train.insult+train.identity_hate

x = train.number_tags.value_counts().index

y = train.number_tags.value_counts().values



plt.figure(figsize=(8,4))

ax= sns.barplot(x, y, alpha=0.8)

plt.title("# number of 1s")

plt.ylabel('# of Occurrences', fontsize=12)

plt.xlabel('Type ', fontsize=12)

ax.legend(x)

#adding the text labels

rects = ax.patches

labels = y

for rect, label in zip(rects, labels):

    height = rect.get_height()

    ax.text(rect.get_x() + rect.get_width()/2, height + 5, label, ha='center', va='bottom')

plt.show()
train_col_name = train.columns.drop(["id","comment_text","clean"]).tolist()
f, ax = plt.subplots(figsize=(9, 6))

f.suptitle('Correlation matrix for categories')

sns.heatmap(train[train_col_name].corr(), annot=True, linewidths=.5, ax=ax)
t = train[(train['toxic'] == 1) & (train['insult'] == 0) & (train['obscene'] == 0)].shape[0]

i = train[(train['toxic'] == 0) & (train['insult'] == 1) & (train['obscene'] == 0)].shape[0]

o = train[(train['toxic'] == 0) & (train['insult'] == 0) & (train['obscene'] == 1)].shape[0]



t_i = train[(train['toxic'] == 1) & (train['insult'] == 1) & (train['obscene'] == 0)].shape[0]

t_o = train[(train['toxic'] == 1) & (train['insult'] == 0) & (train['obscene'] == 1)].shape[0]

i_o = train[(train['toxic'] == 0) & (train['insult'] == 1) & (train['obscene'] == 1)].shape[0]



t_i_o = train[(train['toxic'] == 1) & (train['insult'] == 1) & (train['obscene'] == 1)].shape[0]





# Make the diagram

plt.figure(figsize=(8, 8))

plt.title("Venn diagram for 'toxic', 'insult' and 'obscene'")

venn3(subsets = (t, i, t_i, o, t_o, i_o, t_i_o), 

      set_labels=('toxic', 'insult', 'obscene'))

plt.show()
from wordcloud import WordCloud ,STOPWORDS

stopword=set(STOPWORDS)



def get_word_cloud(column):

    subset=train[train[column]==1]

    text=subset.comment_text.values

    wc= WordCloud(background_color="black",max_words=2000,stopwords=stopword)

    wc.generate(" ".join(text))

    plt.figure(figsize=(20,10))

    plt.axis("off")

    plt.title("Words frequented in Clean Comments", fontsize=20)

    plt.imshow(wc.recolor(colormap= 'viridis' , random_state=17), alpha=0.98)

    plt.show()
get_word_cloud('clean')   #wordcloud for clean comments
get_word_cloud('toxic')
get_word_cloud('obscene')
get_word_cloud('severe_toxic')
train['ip']  = [re.findall('[0-9]+(?:\.[0-9]+){3}', i)for i in train.comment_text.tolist()]

train['ip'] = train['ip'].astype(str).str.replace('[','').str.replace(']','').str.replace("'",'')

test['ip']  = [re.findall('[0-9]+(?:\.[0-9]+){3}', i)for i in test.comment_text.tolist()]

test['ip'] = test['ip'].astype(str).str.replace('[','').str.replace(']','').str.replace("'",'')
train.head()
train[train['ip'] != ''].count()

10083/train.shape[0]*100
test[test['ip'] != ''].count()

759/test.shape[0]*100
# create a columns 'is_ip'

train['is_ip'] = 0

train.loc[train['ip'] != '', 'is_ip'] =1
sub_ip_df = train[['toxic','severe_toxic','obscene','threat','insult','identity_hate','clean','is_ip','ip']]

same_ip = sub_ip_df.groupby('ip').sum()
same_ip
# take a look at the guy with ip 199.101.61.190

ip_199 = pd.DataFrame(train.loc[train.ip == '199.101.61.190', ['toxic','severe_toxic','obscene','threat','insult','identity_hate','clean','comment_text']])

ip_199.head(10)
# toxic

ip_199.iloc[5,7]
def ip_by_category(df):

    lables = train.columns[2:9]

    lst = []

    for i in lables:

        ips = df.loc[(df[i]==1) & (df['is_ip'] ==1), 'is_ip'].sum()

        totals = df[lables].sum()

        lst.append(ips)

        combined = list(zip(lables,totals,lst))

        res = pd.DataFrame(combined, columns=['category','totals','total_ips'])

        res['pctage'] = round(res.total_ips / res.totals*100,2)

    return res

ip_by_category(train)
#get raw comments for each category, can be used for more feature engineering

def get_raw_comment(category, df):

    text = df.loc[df[category] ==1,'comment_text']

    return text



#get cleaned comments for each category, can be used for count_vectorizer

def get_cleaned_comment(category, df):

    text = df.loc[df[category] ==1,'clean_corpus']

    return text
def clean(comment):

    """

    This function receives comments and returns clean word-list

    """

    #Convert to lower case , so that Hi and hi are the same

    comment=comment.lower()

    #remove \n

    comment=re.sub("\\n","",comment)

    # remove leaky elements like ip,user

    comment=re.sub("\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}","",comment)

    #removing usernames

    comment=re.sub("\[\[.*\]","",comment)



    return comment
train['clean_corpus']=train['comment_text'].apply(lambda x :clean(x))

test['clean_corpus']=test['comment_text'].apply(lambda x :clean(x))
def word_freq(category, df):

    corpus = get_cleaned_comment(category, df)

    

    '''in the countvectorizer, get rid of english stop words, use lower case (default), use binary so 

    when count >1 it will be 1 ''' 

    vectorizer = CountVectorizer(stop_words = 'english', binary = True)

    x = vectorizer.fit_transform(corpus)

    word_freq = pd.DataFrame(x.toarray(), columns = vectorizer.get_feature_names())

    return word_freq
id_word_freq = word_freq('identity_hate',train)

id_word_freq.head()
# run the total of words in each category

id_word_freq.sum().sum()
to_word_freq = word_freq('toxic',train)

to_word_freq.head()
to_word_freq.sum().sum()
se_to_word_freq = word_freq('severe_toxic',train)

se_to_word_freq.head()
se_to_word_freq.sum().sum()
in_word_freq = word_freq('insult',train)

in_word_freq.head()
in_word_freq.sum().sum()
ob_word_freq = word_freq('obscene',train)

ob_word_freq.head()
ob_word_freq.sum().sum()
th_word_freq = word_freq('threat',train)

th_word_freq
th_word_freq.sum().sum()
def top_freq_word(category,df):

    freq = word_freq(category, df)

    top_freq = pd.DataFrame(freq.sum(), columns = ['Freq']).sort_values(by = 'Freq', ascending = False)

 

    return top_freq.head(50)
id_f = top_freq_word('identity_hate',train)
id_f.head()
to_f = top_freq_word('toxic',train)
to_f.head()
st_f = top_freq_word('severe_toxic',train)
st_f.head()
o_f = top_freq_word('obscene',train)
o_f.head()
th_f = top_freq_word('threat',train)
th_f.head()
in_f = top_freq_word('insult',train)
in_f.head()
from itertools import combinations 

def top_combinations(category, df):

    freq_df = word_freq(category, df)

    comb = combinations(top_freq_word(category, df).index,2)

    c = list(comb)

    #print(c)

    for a,b in c: 

        freq_df[str(a)+'_'+str(b)] = 0  

        freq_df[str(a)+'_'+str(b)] = [1 if i >= 1 else 0 for i in (freq_df[a]*freq_df[b])]

    return freq_df
top_combined_id = top_combinations('identity_hate',train)

top_combined_id.head()
top_combined_toxic = top_combinations('toxic',train)

top_combined_toxic.head()
top_combined_severe_toxic = top_combinations('severe_toxic',train)

top_combined_severe_toxic.head()
top_combined_obscene = top_combinations('obscene',train)

top_combined_obscene.head()
top_combined_threat = top_combinations('threat',train)

top_combined_threat.head()
top_combined_insult = top_combinations('insult',train)

top_combined_insult.head()
# see the difference of toxic ~ obscene

set(to_f.index) - (set(to_f.index)&set(o_f.index))
# see the difference of obscene ~ toxic

set(o_f.index) - (set(to_f.index)&set(o_f.index))
# see the difference of insult ~ toxic

set(in_f.index) - (set(to_f.index)&set(in_f.index))
# see the difference of toxic ~ insult

set(to_f.index) - (set(to_f.index)&set(in_f.index))
# see the difference of obscene ~ insult

set(o_f.index) - (set(o_f.index)&set(in_f.index))
# see the difference of insult ~ obscene

set(in_f.index) - (set(o_f.index)&set(in_f.index))
# see the difference of identity ~ threate

set(id_f.index) - (set(id_f.index)&set(th_f.index))
# see the difference of threate ~ identity

set(th_f.index) - (set(id_f.index)&set(th_f.index))
### a lot of racist word, consider to use the racist list from 

#https://en.wikipedia.org/wiki/List_of_ethnic_slurs_by_ethnicity to generate more engineered features

id_text = get_raw_comment('identity_hate',train)

id_text
# toxic comments are more used by regular sware words

tox_text = get_raw_comment('toxic',train)

tox_text
# server toxic is long and has lot of duplicated copy paste. Consider to calculate duplicated words count

st_text = get_raw_comment('severe_toxic',train)

st_text
# genitals related words

ob_text = get_raw_comment('obscene',train)

ob_text
tr_text = get_raw_comment('threat',train)

tr_text
#http://www.insult.wiki/wiki/Insult_List

ins_text =get_raw_comment('insult',train)

ins_text
clean_text = get_raw_comment('clean',train)

clean_text
from nltk.corpus import stopwords

stopWords = set(stopwords.words('english'))
sample = train[0:100]
#lemmatization

from nltk.corpus import wordnet



from nltk.stem import WordNetLemmatizer

wordnet_lemmatizer = WordNetLemmatizer()



# sentence_words = nltk.word_tokenize(sample.comment_text)

# # [j for j in sample['comment_text']]

# [wordnet_lemmatizer.lemmatize(word) for word in sentence_words]





def lemmaSentence(sentence):

    token_words=word_tokenize(sentence)

    token_words

    lemma_sentence=[]

    for word in token_words:

        lemma_sentence.append(wordnet_lemmatizer.lemmatize(word))

        lemma_sentence.append(" ")

    return "".join(lemma_sentence)



x=[lemmaSentence(i) for i in sample['comment_text']]

print(x)
# porterstemmer

from nltk.tokenize import sent_tokenize, word_tokenize

from nltk.stem import PorterStemmer

porter = PorterStemmer()

def stemSentence(sentence):

    token_words=word_tokenize(sentence)

    token_words

    stem_sentence=[]

    for word in token_words:

        stem_sentence.append(porter.stem(word))

        stem_sentence.append(" ")

    return "".join(stem_sentence)



x=[stemSentence(i) for i in sample['comment_text']]

print(x)
import nltk

from nltk.stem.snowball import SnowballStemmer



stemmer = SnowballStemmer(language="english")

stems = [stemmer.stem(t) for t in sample['comment_text']]  

stems
import emoji
corpus = pd.DataFrame([

     'This is the first document 🥰😇😆.',

     'This document is the 😇second document.',

     'And this is the third one. 💝',

     'Is this the first document, ☯️ you\'ll ?',

],columns = ['a'])
corpus
corpus['decoded'] = corpus['a'].apply(lambda x: emoji.demojize(x))
corpus