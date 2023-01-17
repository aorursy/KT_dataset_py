import pandas as pd #Analysis 

import matplotlib.pyplot as plt #Visulization

import seaborn as sns #Visulization

import numpy as np #Analysis 

from scipy.stats import norm #Analysis 

from sklearn.preprocessing import StandardScaler #Analysis 

from scipy import stats #Analysis 

import warnings 

warnings.filterwarnings('ignore')

%matplotlib inline

import gc

from textblob import TextBlob

from nltk import word_tokenize, pos_tag, ne_chunk

import os

import string

color = sns.color_palette()

import spacy

import re

from nltk.tokenize import RegexpTokenizer

%matplotlib inline



from plotly import tools

import plotly.offline as py

py.init_notebook_mode(connected=True)

import plotly.graph_objs as go



from sklearn import model_selection, preprocessing, metrics, ensemble, naive_bayes, linear_model

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

from sklearn.decomposition import TruncatedSVD

import lightgbm as lgb

import os

import nltk

from nltk.tokenize import word_tokenize

from textblob import TextBlob, Word

from nltk.stem import WordNetLemmatizer 

from nltk.corpus import wordnet

from nltk.corpus import stopwords

from wordcloud import WordCloud, STOPWORDS

from collections import defaultdict

from nltk import FreqDist

pd.options.mode.chained_assignment = None

pd.options.display.max_columns = 999

import copy

from sklearn.metrics import confusion_matrix

import seaborn as sn
df_train = pd.read_csv("../input/kuc-hackathon-winter-2018/drugsComTrain_raw.csv", parse_dates=["date"])

df_test = pd.read_csv("../input/kuc-hackathon-winter-2018/drugsComTest_raw.csv", parse_dates=["date"])
print("Train shape :" ,df_train.shape)

print("Test shape :", df_test.shape)
df_all = pd.concat([df_train,df_test])

df_all.info()
df_all = df_all.dropna(axis=0)
print("unique values count of all : " ,len(set(df_all['uniqueID'].values)))

print("length of all : " ,df_all.shape[0])
df_all=df_all.drop(['uniqueID'], axis=1)

df_all=df_all.drop(['date'], axis=1)
condition_dn = df_all.groupby(['condition'])['drugName'].nunique().sort_values(ascending=False)

condition_dn[0:20].plot(kind="bar", figsize = (14,6), fontsize = 10,color="green")

plt.xlabel("", fontsize = 20)

plt.ylabel("", fontsize = 20)

plt.title("Top20 : The number of drugs per condition.", fontsize = 20)
print('data before:')

print(df_all.shape)

df_all=df_all[~df_all.condition.str.contains("</span>", na=False)]

df_all=df_all[df_all['condition']!='Not Listed / Othe']

print('data after:')

print(df_all.shape)
condition_dn = df_all.groupby(['condition'])['drugName'].nunique().sort_values(ascending=False)

condition_dn[0:20].plot(kind="bar", figsize = (14,6), fontsize = 10,color="green")

plt.xlabel("", fontsize = 20)

plt.ylabel("", fontsize = 20)

plt.title("Top20 : The number of drugs per condition.", fontsize = 20)
# condition_dn = df_all.groupby(['condition'])['drugName'].nunique().sort_values(ascending=False)



# condition_dn[condition_dn.shape[0]-20:condition_dn.shape[0]].plot(kind="bar", figsize = (14,6), fontsize = 10,color="green")

# plt.xlabel("", fontsize = 20)

# plt.ylabel("", fontsize = 20)

# plt.title("Bottom20 : The number of drugs per condition.", fontsize = 20)
# condition_dn_all = df_all.groupby(['condition'])['drugName'].nunique().sort_values(ascending=False)

# condition_dn_df_all=pd.DataFrame(condition_dn_all).reset_index()

# condition_dn_df_one_to_one_all = condition_dn_df_all[condition_dn_df_all['drugName']==1].reset_index()

# condition_dn_df_40_all = condition_dn_df_all[condition_dn_df_all['drugName']<=10].reset_index()

# condition_dn_df_other_all = condition_dn_df_all[condition_dn_df_all['drugName']>10].reset_index()
# all_list = set(df_all.index)

# condition_list = []

# for i,j in enumerate(df_all['condition']):

#     for c in list(condition_dn_df_40_all['condition']):

#         if j == c:

#             condition_list.append(i)

            

# new_idx = all_list.difference(set(condition_list))

# df_all = df_all.iloc[list(new_idx)].reset_index()

# del df_all['index']
drug_using_count=df_all.groupby('condition')['drugName'].count().sort_values(ascending=False)

drug_using_count[0:10].plot(kind="bar", figsize = (14,6), fontsize = 10,color="green")

plt.xlabel("", fontsize = 20)

plt.ylabel("", fontsize = 20)

plt.title("Top10 : The number of reviews per condition.", fontsize = 20)
rating = df_all['rating'].value_counts().reset_index(name='count').sort_values(by=['index'],ascending=True)

rating

rating.plot(kind="bar",x='index',y='count', figsize = (14,6), fontsize = 10,color="green", legend=None)

plt.xlabel("", fontsize = 20)

plt.ylabel("", fontsize = 20)

plt.xticks(rotation=0)

plt.title("Number of Reviews for Different Ratings", fontsize = 20)
df_all=df_all[df_all['condition'].isin(['Acne','Pain','Anxiety'])]
medicineList=set(df_all['drugName'])

medicineList=[med.lower() for med in medicineList]

medicineList_copy=copy.deepcopy(medicineList)

for i in medicineList_copy:

    if(len(i.split(' '))>1):

        for j in i.split(' '):

            if len(j)>2:

                medicineList.append(j)

medicineList.append('propanolol')
# drug_using_count=df_all.groupby('drugName')['condition'].count().sort_values(ascending=False)

# print(drug_using_count)

# drug_using_count_df=pd.DataFrame(drug_using_count).reset_index()

# drug_using_count_df_100_all = drug_using_count_df[drug_using_count_df['condition']<=1500].reset_index()

# drug_using_count_df_100_all
# all_list = set(df_all.index)

# condition_list = []

# for i,j in enumerate(df_all['drugName']):

#     for c in list(drug_using_count_df_100_all['drugName']):

#         if j == c:

#             condition_list.append(i)

            

# new_idx = all_list.difference(set(condition_list))

# df_all = df_all.iloc[list(new_idx)].reset_index()

# del df_all['index']
rating = df_all['rating'].value_counts().reset_index(name='count').sort_values(by=['index'],ascending=True)

rating

rating.plot(kind="bar",x='index',y='count', figsize = (14,6), fontsize = 10,color="green", legend=None)

plt.xlabel("", fontsize = 20)

plt.ylabel("", fontsize = 20)

plt.xticks(rotation=0)

plt.title("Number of Reviews for Different Ratings", fontsize = 20)
import spacy

nlp = spacy.load("en_core_web_sm")

def get_wordnet_pos(word):

    """Map POS tag to first character lemmatize() accepts"""

    #tag = nltk.pos_tag([word])[0][1][0].upper()

    doc = nlp(word)

    tag=doc[0].tag_[0]    

    tag_dict = {"J": wordnet.ADJ,

                "N": wordnet.NOUN,

                "V": wordnet.VERB,

                "R": wordnet.ADV}

    if tag=="R":

        return tag_dict.get(tag, wordnet.ADJ)

    elif tag=="J":

        return tag_dict.get(tag, wordnet.ADJ)

    elif tag=="V":

        return tag_dict.get(tag, wordnet.VERB)

    else:

        return tag_dict.get(tag, wordnet.NOUN)





# 1. Init Lemmatizer

lemmatizer = WordNetLemmatizer()



# # 2. Lemmatize Single Word with the appropriate POS tag

# word = 'feet'

# print(lemmatizer.lemmatize(word, get_wordnet_pos(word)))



# 3. Lemmatize a Sentence with the appropriate POS tag

# sentence = "The striped aren't been hanging on their feet for best"

# print([lemmatizer.lemmatize(w, get_wordnet_pos(w)) for w in nltk.word_tokenize(sentence)])
def lemmatize_with_postag(sentence):

    sent = TextBlob(sentence)

    tag_dict = {"J": 'a', 

                "N": 'n', 

                "V": 'v', 

                "R": 'r'}

    words_and_tags = [(w, tag_dict.get(pos[0], 'n')) for w, pos in sent.tags]    

    lemmatized_list = [wd.lemmatize(tag) for wd, tag in words_and_tags]

    return lemmatized_list

#lemmatize_with_postag("The striped 123 aren't been hanging on their feet for best")
# from nltk.collocations import *

# bigram_measures = nltk.collocations.BigramAssocMeasures()

# trigram_measures = nltk.collocations.TrigramAssocMeasures()

# text="Pristiq 12-23 1324 12mg is awesome so far.Don&#039;t reject it bcs it is nearly identical to Effexor. Effexor caused extreme sweating.I would stand in front of a forceful fan while dressing, and still be drenched!My doc told me I might not have that with pristiq and I don&#039;t! I am just over 6 weeks, and it took a full 4 weeks to kick in.The first couple of days I felt great, calm and happy, and got hopeful, but then slid back into the dark place.But I kept going and at 4 wks I suddenly one day woke up and started planning fun outings for my teen, after months of sofa lounging.  100mg, F, 56, 135lbs.Tried prozac (allergic), paxil (weight gain), effex (sweating), lexapro (out of dark place, but flat).I hear it&#039;s hell to stop, but I&#039;m not going to stop. Why wld I?"

# tokens = pre_process(text)

# finder = BigramCollocationFinder.from_words(tokens)

# scored = finder.score_ngrams(bigram_measures.raw_freq)

# scored

#sorted(bigram for bigram, score in scored)  # doctest: +NORMALIZE_WHITESPACE
word_map={'weight gain':'gain weight','depression anxiety':'anxiety depression','start':'begin',

          'lose lb':'lose weight','feel like':'feel','sever pain':'severe pain','remarkable':"good",

          'seem help':'help','work well':'help','awful':'bad','childhood':'child','previously':'previous',

          'affect':'effect','suffers':'suffer','well work':'work well','pain severe':'severe pain',

          'miserable':'bad','terrible':'bad','great':'good','prescribed':'prescribe','doc':'doctor',

          'felt':'feel','tremendously':'lot','could not':'not','get severe':'get bad','found':'find',

          'quickly':'fast','rise':'increase','med':'medication','badly':'bad','bear':'bearable',

          'battling':'battle','medicationicine':'medication','drug':'medication',

          'cautiously':'cautious','burning':'burn','begin':'start','bad':'bad','well':'good','medicationication':'medication',

          'absorbs':'absorb','absorbed':'absorb','long term':'long time','issue':'problem','medicine':'medication'}
nlp = spacy.load("en_core_web_sm",disable=['ner', 'parser'])

tokenizer = RegexpTokenizer(r'\w+')

nltk.download('stopwords')

porter=nltk.PorterStemmer()

additional_dic=['soooo','month','year','day','week','say','etc','usually','bcs','one','two','much','something','anything','doctortor','039','ive','absolutely','actually',"also","always","almost","already","still","would",'ago']

additional_dic_step=["story","news","old","home","due","thats","try","ing","seem","haha","age","could","even","usually","since","another","really","theyre","nearly","finally","mg","hour","week","month","year","years","day","yr","able","although","able","around","..."]

stoplist = stopwords.words('english')+additional_dic

stoplist+=additional_dic_step

not_stop = ["aren't","couldn't","didn't","doesn't","don't","hadn't","hasn't","haven't","isn't","mightn't","mustn't","needn't","no","nor","not","shan't","shouldn't","wasn't","weren't","wouldn't"]

to_be_change=["n't","no","nor","didnt","havent","cant","shouldnot","wasnt","wasnot","werent","dont","werenot","wouldnt","wouldnot","couldnt"]

for i in not_stop:

    stoplist.remove(i)



def pre_process(text,n_gram=1):

    text=text.replace('&#039;', '')

    text=text.replace('&quot;', '')

    text=text.replace('&amp;', '')

    text=text.lower()

    for med in medicineList:

        text=text.replace(med,"")

    for i in word_map:

        text=text.replace(i,word_map[i])

    text=re.sub("\d+[A-Za-z]{2}","",text)

    text=re.sub("\d+[-\/]\d+","",text)

    text=re.sub("\d+[x]","",text)

    text=re.sub(r'\d+', '', text)

    #tokens=word_tokenize(text)

    tokens=tokenizer.tokenize(text)

    tokens=[t for t in tokens if t not in stoplist and t not in string.punctuation]    

    #tokens=[nltk.WordNetLemmatizer().lemmatize(t,pos='n' or 'v' or 's' or 'r' or 'a') for t in tokens]

    #tokens=[t for t in tokens if t not in string.digits]

    tokens=['not' if t in to_be_change else t for t in tokens]

    #tokens=[nltk.WordNetLemmatizer().lemmatize(t,pos='n' or 'v' or 'r' or 'a') for t in tokens]

    #tokens=[porter.stem(t) for t in tokens]  

    #tokens=[lemmatizer.lemmatize(t, get_wordnet_pos(t)) for t in tokens]

    tokens=[t for t in tokens if len(t)>=3]

    text=' '.join(tokens)

    doc = nlp(text)

    tokens = [token.lemma_ for token in doc if token.lemma_ != '-PRON-']

    #result=[" ".join(tokens)]

    #ngrams = zip(*[tokens[i:] for i in range(n_gram)])

    return tokens
punctuations = string.punctuation

def pre_process_2(text):

    text=text.replace('&#039;', '')

    text=text.replace('&quot;', '')

    text=text.replace('&amp;', '')

    text=text.lower()

    for med in medicineList:

        text=text.replace(med,"")

    for i in word_map:

        text=text.replace(i,word_map[i])

    text=re.sub("\d+[A-Za-z]{2}","",text)

    text=re.sub("\d+[-\/]\d+","",text)

    text=re.sub("\d+[x]","",text)

    text=re.sub(r'\d+', '', text)

    tokenizer = RegexpTokenizer(r'\w+')

    tokens=tokenizer.tokenize(text)

    text=" ".join(tokens)

    doc = nlp(text)

    tokens = [token.lemma_.strip() for token in doc if token.lemma_ != '-PRON-']

    tokens=[t for t in tokens if t not in stoplist and t not in punctuations]

    tokens=['not' if t in to_be_change else t for t in tokens]

    tokens=[t for t in tokens if len(t)>=3]

    return tokens

# text="Pristiq 12-23 1324 12mg wouldn't is awesome so far.Don&#039;t reject it bcs it is nearly identical to Effexor. Effexor caused extreme sweating.I would stand in front of a forceful fan while dressing, and still be drenched!My doc told me I might not have that with pristiq and I don&#039;t! I am just over 6 weeks, and it took a full 4 weeks to kick in.The first couple of days I felt great, calm and happy, and got hopeful, but then slid back into the dark place.But I kept going and at 4 wks I suddenly one day woke up and started planning fun outings for my teen, after months of sofa lounging.  100mg, F, 56, 135lbs.Tried prozac (allergic), paxil (weight gain), effex (sweating), lexapro (out of dark place, but flat).I hear it&#039;s hell to stop, but I&#039;m not going to stop. Why wld I?"

# pre_process_2(text)
# doc = nlp('days')

# tokens = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct]

# tokens
# text="Pristiq 12-23 1324 wouldnt 12mg is awesome so far.Don&#039;t went it bcs it is nearly identical to Effexor. Effexor caused extreme sweating.I would stand in front of a forceful fan while dressing, and still be drenched!My doc told me I might not have that with pristiq and I don&#039;t! I am just over 6 weeks, and it took a full 4 weeks to kick in.The first couple of days I felt great, calm and happy, and got hopeful, but then slid back into the dark place.But I kept going and at 4 wks I suddenly one day woke up and started planning fun outings for my teen, after months of sofa lounging.  100mg, F, 56, 135lbs.Tried prozac (allergic), paxil (weight gain), effex (sweating), lexapro (out of dark place, but flat).I hear it&#039;s hell to stop, but I&#039;m not going to stop. Why wld I?"

# pre_process(text)
def pre_ngram(tokens,n_gram=2):

    ngrams = zip(*[tokens[i:] for i in range(n_gram)])

    return [" ".join(ngram) for ngram in ngrams]
df_all.shape
toks=df_all['review'].apply(pre_process)

toks_2gram=toks.apply(pre_ngram,n_gram=2)

toks_3gram=toks.apply(pre_ngram,n_gram=3)
#https://www.kaggle.com/sudalairajkumar/simple-exploration-notebook-qiqc kernel 



# Thanks : https://www.kaggle.com/aashita/word-clouds-of-various-shapes ##

def plot_wordcloud(text, mask=None, max_words=200, max_font_size=100, figure_size=(24.0,16.0), 

                   title = None, title_size=40, image_color=False):

    stopwords = set(STOPWORDS)

    more_stopwords = {'one', 'br', 'Po', 'th', 'sayi', 'fo', 'Unknown'}

    stopwords = stopwords.union(more_stopwords)



    wordcloud = WordCloud(background_color='white',

                    stopwords = stopwords,

                    max_words = max_words,

                    max_font_size = max_font_size, 

                    random_state = 42,

                    width=800, 

                    height=400,

                    mask = mask)

    wordcloud.generate(str(text))

    

    plt.figure(figsize=figure_size)

    if image_color:

        image_colors = ImageColorGenerator(mask);

        plt.imshow(wordcloud.recolor(color_func=image_colors), interpolation="bilinear");

        plt.title(title, fontdict={'size': title_size,  

                                  'verticalalignment': 'bottom'})

    else:

        plt.imshow(wordcloud);

        plt.title(title, fontdict={'size': title_size, 'color': 'black', 

                                  'verticalalignment': 'bottom'})

    plt.axis('off');

    plt.tight_layout()  
# plot_wordcloud(df_all["review"], title="Word Cloud of review")

text_tokens=[c for l in toks for c in l]

text_tokens_frequency=FreqDist(text_tokens)

text=" ".join(text_tokens)

plot_wordcloud(text, title="Word Cloud of review")
df_all_9_10 = df_all[df_all["rating"]>8]

df_all_4_8 = df_all[(df_all["rating"]>=4) & (df_all["rating"]<9)]

df_all_1_3 = df_all[df_all["rating"]<4]
toks_9_10=df_all_9_10['review'].apply(pre_process)

toks_4_8=df_all_4_8['review'].apply(pre_process)

toks_1_3=df_all_1_3['review'].apply(pre_process)
text_tokens_9_10=[c for l in toks_9_10 for c in l]

text_tokens_4_8=[c for l in toks_4_8 for c in l]

text_tokens_1_3=[c for l in toks_1_3 for c in l]

text_tokens_frequency_9_10=FreqDist(text_tokens_9_10)

text_tokens_frequency_4_8=FreqDist(text_tokens_4_8)

text_tokens_frequency_1_3=FreqDist(text_tokens_1_3)
toks_1_3_toCombine=' '.join(text_tokens_1_3)

toks_4_8_toCombine=' '.join(text_tokens_4_8)

toks_9_10_toCombine=' '.join(text_tokens_9_10)
from nltk.collocations import *

bigram_measures = nltk.collocations.BigramAssocMeasures()

trigram_measures = nltk.collocations.TrigramAssocMeasures()

#text="Pristiq 12-23 1324 12mg is awesome so far.Don&#039;t reject it bcs it is nearly identical to Effexor. Effexor caused extreme sweating.I would stand in front of a forceful fan while dressing, and still be drenched!My doc told me I might not have that with pristiq and I don&#039;t! I am just over 6 weeks, and it took a full 4 weeks to kick in.The first couple of days I felt great, calm and happy, and got hopeful, but then slid back into the dark place.But I kept going and at 4 wks I suddenly one day woke up and started planning fun outings for my teen, after months of sofa lounging.  100mg, F, 56, 135lbs.Tried prozac (allergic), paxil (weight gain), effex (sweating), lexapro (out of dark place, but flat).I hear it&#039;s hell to stop, but I&#039;m not going to stop. Why wld I?"

tokens = pre_process(toks_1_3_toCombine)

finder = BigramCollocationFinder.from_words(tokens)

scored = finder.score_ngrams(bigram_measures.raw_freq)

scored

#sorted(bigram for bigram, score in scored)  # doctest: +NORMALIZE_WHITESPACE
toks_2gram_1_3=toks_1_3.apply(pre_ngram,n_gram=2)

toks_2gram_4_8=toks_4_8.apply(pre_ngram,n_gram=2)

toks_2gram_9_10=toks_9_10.apply(pre_ngram,n_gram=2)
text_tokens_2gram_1_3=[c for l in toks_2gram_1_3 for c in l]

text_tokens_2gram_4_8=[c for l in toks_2gram_4_8 for c in l]

text_tokens_2gram_9_10=[c for l in toks_2gram_9_10 for c in l]

text_tokens_frequency_2gram_9_10=FreqDist(text_tokens_2gram_9_10)

text_tokens_frequency_2gram_4_8=FreqDist(text_tokens_2gram_4_8)

text_tokens_frequency_2gram_1_3=FreqDist(text_tokens_2gram_1_3)
toks_3gram_1_3=toks_1_3.apply(pre_ngram,n_gram=3)

toks_3gram_4_8=toks_4_8.apply(pre_ngram,n_gram=3)

toks_3gram_9_10=toks_9_10.apply(pre_ngram,n_gram=3)
text_tokens_3gram_1_3=[c for l in toks_3gram_1_3 for c in l]

text_tokens_3gram_4_8=[c for l in toks_3gram_4_8 for c in l]

text_tokens_3gram_9_10=[c for l in toks_3gram_9_10 for c in l]

text_tokens_frequency_3gram_9_10=FreqDist(text_tokens_3gram_9_10)

text_tokens_frequency_3gram_4_8=FreqDist(text_tokens_3gram_4_8)

text_tokens_frequency_3gram_1_3=FreqDist(text_tokens_3gram_1_3)
## custom function for horizontal bar chart ##

def horizontal_bar_chart(df, color):

    trace = go.Bar(

        y=df["word"].values[::-1],

        x=df["wordcount"].values[::-1],

        showlegend=False,

        orientation = 'h',

        marker=dict(

            color=color,

        ),

    )

    return trace
fd_sorted = pd.DataFrame(sorted(text_tokens_frequency.items(), key=lambda x: x[1])[::-1])

fd_sorted.columns = ["word", "wordcount"]

trace = horizontal_bar_chart(fd_sorted.head(50), '#82C8F3')

# Creating two subplots

fig = tools.make_subplots(rows=1, cols=1, vertical_spacing=0.04,

                          subplot_titles=["Frequent words of rating 1 to 10"

                                          ])

fig.append_trace(trace, 1, 1)

fig['layout'].update(height=1200, width=900, paper_bgcolor='rgb(233,233,233)', title="Word Count Plots")

py.iplot(fig, filename='word-plots')
text_9_10=" ".join(text_tokens_9_10)

plot_wordcloud(text, title="Word Cloud of review: Positive Sentiment")

text_4_8=" ".join(text_tokens_4_8)

plot_wordcloud(text, title="Word Cloud of review: Neutral Sentiment")

text_1_3=" ".join(text_tokens_1_3)

plot_wordcloud(text, title="Word Cloud of review: Negative Sentiment")
fd_sorted1 = pd.DataFrame(sorted(text_tokens_frequency_1_3.items(), key=lambda x: x[1])[::-1])

fd_sorted1.columns = ["word", "wordcount"]

trace1 = horizontal_bar_chart(fd_sorted1.head(15), '#82C8F3')



fd_sorted2 = pd.DataFrame(sorted(text_tokens_frequency_4_8.items(), key=lambda x: x[1])[::-1])

fd_sorted2.columns = ["word", "wordcount"]

trace2 = horizontal_bar_chart(fd_sorted2.head(15), '#82C8F3')



fd_sorted3 = pd.DataFrame(sorted(text_tokens_frequency_9_10.items(), key=lambda x: x[1])[::-1])

fd_sorted3.columns = ["word", "wordcount"]

trace3 = horizontal_bar_chart(fd_sorted3.head(15), '#82C8F3')
fig = tools.make_subplots(rows=1, cols=3, vertical_spacing=0.03,horizontal_spacing=0.08,

                          subplot_titles=["Frequent words of rating 1 to 3","Frequent words of rating 4 to 8","Frequent words of rating 9 to 10"

                                          ])

fig.append_trace(trace1, 1, 1)

fig.append_trace(trace2, 1, 2)

fig.append_trace(trace3, 1, 3)

fig['layout'].update(height=800, width=1300, paper_bgcolor='rgb(233,233,233)', title="Word Count Plots",font=dict(size=16))

py.iplot(fig, filename='word-plots')

#['and','the','039','for','was','have','had','but','not','with','this','been','about']
fd_sorted1_2gram = pd.DataFrame(sorted(text_tokens_frequency_2gram_1_3.items(), key=lambda x: x[1])[::-1])

fd_sorted1_2gram.columns = ["word", "wordcount"]

trace1_2gram = horizontal_bar_chart(fd_sorted1_2gram.head(15), '#82C8F3')



fd_sorted2_2gram = pd.DataFrame(sorted(text_tokens_frequency_2gram_4_8.items(), key=lambda x: x[1])[::-1])

fd_sorted2_2gram.columns = ["word", "wordcount"]

trace2_2gram = horizontal_bar_chart(fd_sorted2_2gram.head(15), '#82C8F3')



fd_sorted3_2gram = pd.DataFrame(sorted(text_tokens_frequency_2gram_9_10.items(), key=lambda x: x[1])[::-1])

fd_sorted3_2gram.columns = ["word", "wordcount"]

trace3_2gram = horizontal_bar_chart(fd_sorted3_2gram.head(15), '#82C8F3')
fig = tools.make_subplots(rows=1, cols=3, vertical_spacing=0.03,horizontal_spacing=0.18,

                          subplot_titles=["Frequent words of rating 1 to 3","Frequent words of rating 4 to 8","Frequent words of rating 9 to 10"

                                          ])

fig.append_trace(trace1_2gram, 1, 1)

fig.append_trace(trace2_2gram, 1, 2)

fig.append_trace(trace3_2gram, 1, 3)

fig['layout'].update(height=800, width=1300, paper_bgcolor='rgb(233,233,233)', title="Word Count Plots",font=dict(size=16))

py.iplot(fig, filename='word-plots')

#['and','the','039','for','was','have','had','but','not','with','this','been','about']
fd_sorted1_3gram = pd.DataFrame(sorted(text_tokens_frequency_3gram_1_3.items(), key=lambda x: x[1])[::-1])

fd_sorted1_3gram.columns = ["word", "wordcount"]

trace1_3gram = horizontal_bar_chart(fd_sorted1_3gram.head(15), '#82C8F3')



fd_sorted2_3gram = pd.DataFrame(sorted(text_tokens_frequency_3gram_4_8.items(), key=lambda x: x[1])[::-1])

fd_sorted2_3gram.columns = ["word", "wordcount"]

trace2_3gram = horizontal_bar_chart(fd_sorted2_3gram.head(15), '#82C8F3')



fd_sorted3_3gram = pd.DataFrame(sorted(text_tokens_frequency_3gram_9_10.items(), key=lambda x: x[1])[::-1])

fd_sorted3_3gram.columns = ["word", "wordcount"]

trace3_3gram = horizontal_bar_chart(fd_sorted3_3gram.head(15), '#82C8F3')
fig = tools.make_subplots(rows=1, cols=3, vertical_spacing=0.03,horizontal_spacing=0.18,

                          subplot_titles=["Frequent words of rating 1 to 3","Frequent words of rating 4 to 8","Frequent words of rating 9 to 10"

                                          ])

fig.append_trace(trace1_3gram, 1, 1)

fig.append_trace(trace2_3gram, 1, 2)

fig.append_trace(trace3_3gram, 1, 3)

fig['layout'].update(height=800, width=1300, paper_bgcolor='rgb(233,233,233)', title="Word Count Plots",font=dict(size=16))

py.iplot(fig, filename='word-plots')
" ".join(toks.values[0])
doc = nlp(" ".join(toks.values[1]))

from spacy.symbols import nsubj, VERB

from spacy import displacy

displacy.render(doc, style='dep')
#!pip install sense2vec==1.0.0a0
# import sense2vec

# #from sense2vec import Sense2VecComponent

# s2v = sense2vec.load('../input/reddit-vectors-for-sense2vec-spacy/reddit_vectors-1.1.0/reddit_vectors-1.1.0/')

# s2v = sense2vec.Sense2VecComponent('../input/reddit-vectors-for-sense2vec-spacy/reddit_vectors-1.1.0/reddit_vectors-1.1.0/')

# spacy_tok.add_pipe(s2v)

# doc = spacy_tok(u"dessert.")

# freq = doc[0]._.s2v_freq

# vector = doc[0]._.s2v_vec

# most_similar = doc[0]._.s2v_most_similar(5)

# most_similar,freq
# index_list=[]

# for idx, val in enumerate(toks.values):

#     if len(val)==0 or val is None:

#         index_list.append(idx)

# print(index_list)

# from textblob import TextBlob

# from nltk import word_tokenize, pos_tag, ne_chunk

# for text in toks.values[0:10]:

#     result = TextBlob(" ".join(text))

#         #print(result.tags)

#         #{<JJ><NN>?<NN>}

#         #{<NN><NN>?<JJ>}

#         #{<VBP><JJS>}

#     reg_exp=r"""

#         NP: {<JJ><NN>?<NN>}

#             {<NN><NN>?<JJ>}

#     """

#     rp = nltk.RegexpParser(reg_exp)

#     result = rp.parse(result.tags)

#     print(result)

# print([i for i in result.subtrees()])

# empty_list = []

# filter_list=['try','range','story','box','weekend']

# for subtree in result.subtrees():

#     if subtree.label() == 'NP':

#         subtree = str(subtree.flatten())

#         tree = nltk.Tree.fromstring(subtree, read_leaf=lambda x: x.split("/")[0])   

#         result=[checkIfMatch(element,tree.leaves()) for element in filter_list]

#         if True not in result:

#             if(TextBlob(" ".join(tree.leaves())).sentences[0].sentiment.polarity<0):

#                  empty_list.append(tree.leaves())            

#         print(tree.leaves())

# empty_list

# text='severe muscle stiffness'

# result = TextBlob(text)

# result.sentences[0].sentiment.polarity
def checkIfMatch(elem,match):

    if elem in match:

        return True;

    else :

        return False;





reg_exp=r"""

    NP: {<JJ><NN>?<NN>}

        {<NN><NN>?<JJ>}

        {<VBP><JJS>}

"""

rp = nltk.RegexpParser(reg_exp)

phase_list = []

phase_df_array=[]

filter_list=['try','range','story','box','weekend','thing','school','bit','time','football','way','count','side','prescribe','turkey']

for text in toks.values:

    if text is not None and len(text)>0:

        phase_df_array_sub=[]

        result=rp.parse(TextBlob(" ".join(text)).tags)    

        for subtree in result.subtrees():

            if subtree.label() == 'NP':

                subtree = str(subtree.flatten())

                tree = nltk.Tree.fromstring(subtree, read_leaf=lambda x: x.split("/")[0])   

                result=[checkIfMatch(element,tree.leaves()) for element in filter_list]

                if True not in result:

                    if(TextBlob(" ".join(tree.leaves())).sentences[0].sentiment.polarity<0):

                        phase_list.append(' '.join(tree.leaves()))   

                        phase_df_array_sub.append(' '.join(tree.leaves()))   

        phase_df_array.append(phase_df_array_sub)

    else:

        phase_df_array.append([])
df_all['sentiment'] = df_all["rating"].apply(lambda x: 0 if x <4 else ( 2 if x>8 else 1))

df_all['cleanText']=[" ".join(t) for t in toks]

df_all['side_effect']=phase_df_array
import multiprocessing

from time import time

from gensim.models import Word2Vec

cores = multiprocessing.cpu_count()

w2v_model = Word2Vec(min_count=10,

                     window=4,

                     size=300,

                     sample=6e-5, 

                     alpha=0.03, 

                     min_alpha=0.0007, 

                     negative=20,

                     workers=cores-1)

t = time()



w2v_model.build_vocab(toks, progress_per=10000)



print('Time to build vocab: {} mins'.format(round((time() - t) / 60, 2)))

t = time()



w2v_model.train(toks, total_examples=w2v_model.corpus_count, epochs=30, report_delay=1)



print('Time to train the model: {} mins'.format(round((time() - t) / 60, 2)))

w2v_model.init_sims(replace=True)

len(w2v_model.wv.vocab)
w2v_model.wv.most_similar(positive=["anxiety"])
w2v_model.wv.most_similar(positive=["face"])
import numpy as np



class DocSim(object):

    def __init__(self, w2v_model , stopwords=[]):

        self.w2v_model = w2v_model

        self.stopwords = stopwords



    def vectorize(self, doc):

        """Identify the vector values for each word in the given document"""

        doc = doc.lower()

        words = [w for w in doc.split(" ") if w not in self.stopwords]

        word_vecs = []

        for word in words:

            try:

                vec = self.w2v_model[word]

                word_vecs.append(vec)

            except KeyError:

                # Ignore, if the word doesn't exist in the vocabulary

                pass



        # Assuming that document vector is the mean of all the word vectors

        # PS: There are other & better ways to do it.

        vector = np.mean(word_vecs, axis=0)

        return vector





    def _cosine_sim(self, vecA, vecB):

        """Find the cosine similarity distance between two vectors."""

        csim = np.dot(vecA, vecB) / (np.linalg.norm(vecA) * np.linalg.norm(vecB))

        if np.isnan(np.sum(csim)):

            return 0

        return csim



    def calculate_similarity(self, source_doc, target_docs=[], threshold=0):

        """Calculates & returns similarity scores between given source document & all

        the target documents."""

        if isinstance(target_docs, str):

            target_docs = [target_docs]



        source_vec = self.vectorize(source_doc)

        results = []

        for doc in target_docs:

            target_vec = self.vectorize(doc)

            sim_score = self._cosine_sim(source_vec, target_vec)

            if sim_score > threshold:

                results.append({

                    'score' : sim_score,

                    'doc' : doc

                })

            # Sort results by score in desc order

            results.sort(key=lambda k : k['score'] , reverse=True)



        return results
#document similarity

# from gensim.models.keyedvectors import KeyedVectors

# model_path = './data/GoogleNews-vectors-negative300.bin'

# w2v_model = KeyedVectors.load_word2vec_format(model_path, binary=True)

ds = DocSim(w2v_model)

source_doc = 'how to delete an invoice'

target_docs = ['delete a invoice', 'how do i remove an invoice', 'purge an invoice']



# This will return 3 target docs with similarity score

sim_scores = ds.calculate_similarity(df_all['cleanText'].values[0],df_all['cleanText'].values[6])



print(sim_scores)
#df_all.groupby('drugName')['side_effect'].count()

# df_valid=df_all[df_all.astype(str)['side_effect'] != '[]']

# df_drug_count_all=df_all.groupby('drugName')['side_effect'].count().reset_index(name='count')

# df_drug_count_valid=df_valid.groupby('drugName')['side_effect'].count().reset_index(name='count')

# df_drug_count_bind=pd.merge(df_drug_count_all,df_drug_count_valid,how='left',on=['drugName'])

# df_drug_count_bind['percentage']=df_drug_count_bind['count_y']/df_drug_count_bind['count_x']

# df_drug_count_bind.percentage.fillna(value=0, inplace=True)

# df_drug_count_bind
# DrugNameList=set(df_all['drugName'].values)

# DrugNameList

# df_all_Pain=df_all[df_all['condition']=='Pain']

# df_all_Insomnia=df_all[df_all['condition']=='Insomnia']

# df_all_Anxiety=df_all[df_all['condition']=='Anxiety']

pd.set_option('display.max_rows', 5000)
def sideEffectAnalysis(condition):

    df_all_Pain=df_all[df_all['condition']==condition]

    df_valid=df_all_Pain[df_all_Pain.astype(str)['side_effect'] != '[]']

    df_drug_count_all=df_all_Pain.groupby('drugName')['side_effect'].count().reset_index(name='count')

    df_drug_count_valid=df_valid.groupby('drugName')['side_effect'].count().reset_index(name='count')

    df_drug_count_bind=pd.merge(df_drug_count_all,df_drug_count_valid,how='left',on=['drugName'])

    df_drug_count_bind['percentage']=df_drug_count_bind['count_y']/df_drug_count_bind['count_x']

    df_drug_count_bind.percentage.fillna(value=0, inplace=True)

    df_drug_count_bind=df_drug_count_bind.sort_values('percentage',ascending=False)

    ax=df_drug_count_bind[0:20].plot(kind="bar", figsize = (14,6), fontsize = 10,color="#31b0c1",x='drugName',y='percentage', legend=None)

    plt.xlabel("", fontsize = 20)

    plt.ylabel("", fontsize = 20)

    vals = ax.get_yticks()

    ax.set_yticklabels(['{:,.2%}'.format(x) for x in vals])

    plt.title("Percentage of Side Effect Occurrence for "+condition, fontsize = 20)

    

    df_all_Pain_group=df_all_Pain.groupby('drugName')['side_effect'].apply(sum).reset_index(name='side_effect')

    df_all_Pain_noSide=df_all_Pain_group[df_all_Pain_group.astype(str)['side_effect'] == '[]']['drugName']

    print('----------Medicine with no side effects mentioned: '+condition+'----------')

    print(df_all_Pain_noSide.values)

    side_effect_list_pain=df_all_Pain_group[df_all_Pain_group.astype(str)['side_effect'] != '[]']['side_effect'].values

    #print(side_effect_list_pain)

    side_effect_list_str_pain=[c for l in side_effect_list_pain for c in l]

    print('----------Top Frequent Side Effect for: '+condition+'----------')

    side_effect_list_str_pain_frequency=FreqDist(side_effect_list_str_pain).most_common(20)

    print(side_effect_list_str_pain_frequency)

    side_effect_list_str_pain_text=" ".join(side_effect_list_str_pain)

    plot_wordcloud(side_effect_list_str_pain_text, title="Word Cloud of side effect: "+condition)

    df_pain_with_side_effect=df_all_Pain_group[df_all_Pain_group.astype(str)['side_effect'] != '[]']

    df_pain_with_side_effect['count']=df_pain_with_side_effect['side_effect'].str.len()

    df_pain_with_side_effect=df_pain_with_side_effect.sort_values('count',ascending=False)

    df_pain_with_side_effect[0:20].plot(kind="bar", figsize = (14,6), fontsize = 10,color="#31b0c1",x='drugName',y='count')

    plt.xlabel("", fontsize = 20)

    plt.ylabel("", fontsize = 20)

    plt.title("Count of side effect reviews for "+condition, fontsize = 20)
sideEffectAnalysis('Pain')
sideEffectAnalysis('Acne')
sideEffectAnalysis('Anxiety')
import multiprocessing

from time import time

from gensim.models import Word2Vec

from nltk.cluster import KMeansClusterer

import nltk

def w2v_clustering(condition,cluster):

    NUM_CLUSTERS=cluster

    df_all_cluster=df_all[df_all['condition']==condition]

    df_all_cluster_group=df_all_cluster.groupby('drugName')['side_effect'].apply(sum).reset_index(name='side_effect')

    side_effect_list=df_all_cluster_group[df_all_cluster_group.astype(str)['side_effect'] != '[]']['side_effect'].values



    cores = multiprocessing.cpu_count()

    w2v_model = Word2Vec(min_count=5,

                         window=4,

                         size=300,

                         sample=6e-5, 

                         alpha=0.03, 

                         min_alpha=0.0007, 

                         negative=20,

                         workers=cores-1)

    t = time()



    w2v_model.build_vocab(side_effect_list, progress_per=10000)



    print('Time to build vocab: {} mins'.format(round((time() - t) / 60, 2)))

    t = time()



    w2v_model.train(side_effect_list, total_examples=w2v_model.corpus_count, epochs=30, report_delay=1)



    print('Time to train the model: {} mins'.format(round((time() - t) / 60, 2)))

    w2v_model.init_sims(replace=True)

#     print(w2v_model.wv.vocab)

    kclusterer = KMeansClusterer(NUM_CLUSTERS, distance=nltk.cluster.util.cosine_distance, repeats=25)

    assigned_clusters = kclusterer.cluster(w2v_model[w2v_model.wv.vocab], assign_clusters=True)

#     print (assigned_clusters)

#     print('As:', assigned_clusters)

#     print('Means:', kclusterer.means())

    vocabulary=[i for i in w2v_model.wv.vocab.keys()]

    data = {'Vocabulary':vocabulary, 'Cluster':assigned_clusters} 

    data_cluster=pd.DataFrame(data)

    df=data_cluster.groupby('Cluster')['Vocabulary'].count()

    index_list=['type'+str(i+1) for i in range(NUM_CLUSTERS)]

    df = pd.DataFrame({'count': df.values},

                            index=index_list)

    df['count'].plot.pie( figsize=(5, 5))

    for i in range(NUM_CLUSTERS):

        print('Cluster'+str(i+1)+':')

        print(data_cluster[data_cluster['Cluster']==i]['Vocabulary'].values.tolist())
#w2v_model.wv.most_similar(positive=["dry mouth"])
w2v_clustering('Pain',3)

w2v_clustering('Anxiety',3)

w2v_clustering('Anxiety',2)
w2v_clustering('Acne',3)
from sklearn.decomposition import TruncatedSVD

from sklearn.cluster import KMeans

from sklearn import metrics

import matplotlib.pyplot as plt

def print_terms(cm, num,svd,vectorizer):

    original_space_centroids = svd.inverse_transform(cm.cluster_centers_)

    order_centroids = original_space_centroids.argsort()[:, ::-1]

    terms = vectorizer.get_feature_names()

    for i in range(num):

        print("Cluster %d:" % i, end='')

        for ind in order_centroids[i, :50]:

            print(' %s' % terms[ind], end='')

        print()

def cluster_side_effect(condition,n_cluster=3):

    index_list=['type'+str(i+1) for i in range(n_cluster)]

    df_all_cluster=df_all[df_all['condition']==condition]

    df_all_cluster_group=df_all_cluster.groupby('drugName')['side_effect'].apply(sum).reset_index(name='side_effect')

    side_effect_list=df_all_cluster_group[df_all_cluster_group.astype(str)['side_effect'] != '[]']['side_effect'].values

    #print(side_effect_list)

    side_effect_list_str=[c for l in side_effect_list for c in l]

    #print(side_effect_list)

    side_effect_list_str=set(side_effect_list_str)

    #print(side_effect_list)

    #print(side_effect_list_str)

    vectorizer=TfidfVectorizer(use_idf=True)

    X=vectorizer.fit_transform(side_effect_list_str)

    svd = TruncatedSVD(n_components=500,n_iter=200)

    svd.fit(X)

    print('total explained: ')

    print(svd.explained_variance_ratio_.sum()) 

    X_transformed=svd.fit_transform(X)

    km=KMeans(n_clusters=n_cluster,init='k-means++',max_iter=5000,n_init=1).fit(X_transformed)

    print_terms(km,n_cluster,svd,vectorizer)

    df = pd.DataFrame({'data': km.labels_}).reset_index()

    df=df.groupby('data')['index'].count()

    df = pd.DataFrame({'count': df.values},

                        index=index_list)

    print(df['count'])

    df['count'].plot.pie( figsize=(5, 5))
# def cluster_drug(condition,n_cluster=3):

#     index_list=['type'+str(i+1) for i in range(n_cluster)]

#     df_all_cluster=df_all[df_all['condition']==condition]

#     df_all_cluster_group=df_all_cluster.groupby('drugName')['cleanText'].apply(sum).reset_index(name='cleanText')

#     side_effect_list=df_all_cluster_group['cleanText'].values

#     #print(side_effect_list)

#     side_effect_list_str=side_effect_list

# #     print(side_effect_list_str)

#     vectorizer=TfidfVectorizer(use_idf=True)

#     X=vectorizer.fit_transform(side_effect_list_str)

#     svd = TruncatedSVD(n_components=200,n_iter=100)

#     svd.fit(X)

#     print('total explained: ')

#     print(svd.explained_variance_ratio_.sum()) 

#     X_transformed=svd.fit_transform(X)

#     km=KMeans(n_clusters=n_cluster,init='k-means++',max_iter=5000,n_init=1).fit(X_transformed)

#     print_terms(km,n_cluster,svd,vectorizer)

#     df = pd.DataFrame({'data': km.labels_}).reset_index()

#     df=df.groupby('data')['index'].count()

#     df = pd.DataFrame({'count': df.values},

#                         index=index_list)

#     print(df['count'])

#     df['count'].plot.pie( figsize=(5, 5))
#     df_all_cluster=df_all[df_all['condition']=='Pain']

#     df_all_cluster_group=df_all_cluster.groupby('drugName')['side_effect'].apply(sum).reset_index(name='side_effect')

#     side_effect_list=df_all_cluster_group[df_all_cluster_group.astype(str)['side_effect'] != '[]']['side_effect'].values

#     #print(side_effect_list)

#     side_effect_list_str=[c for l in side_effect_list for c in l]

#     #print(side_effect_list_str)

#     vectorizer=TfidfVectorizer(use_idf=True)

#     X=vectorizer.fit_transform(side_effect_list_str)

#     svd = TruncatedSVD(n_components=500,n_iter=100)

#     svd.fit(X)

#     print('total explained: ')

#     print(svd.explained_variance_ratio_.sum()) 

#     X_transformed=svd.fit_transform(X)

#     km=KMeans(n_clusters=2,init='k-means++',max_iter=5000,n_init=1).fit(X_transformed)

#     print_terms(km,2)

    

#     df = pd.DataFrame({'data': km.labels_}).reset_index()

#     df=df.groupby('data')['index'].count()

#     df = pd.DataFrame({'count': df.values},

#                     index=['Type1','Type2'])

#     df['count'].plot.pie( figsize=(5, 5))

# #     df_all_cluster_count=df_all_cluster.groupby('cluster_type')['drugName'].count().sort_values(ascending=False)
#cluster_drug('Pain',3)
#cluster_drug('Anxiety',3)
#cluster_drug('Acne',3)
cluster_side_effect('Pain',3)

cluster_side_effect('Anxiety',3)
cluster_side_effect('Acne',3)
from sklearn.model_selection import train_test_split

from keras.utils import to_categorical

from sklearn.utils import resample

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.feature_extraction.text import TfidfTransformer

from sklearn.pipeline import Pipeline

from scipy import sparse

from tensorflow.python.keras.models import Sequential

from tensorflow.python.keras.layers import Dense, Bidirectional, LSTM, BatchNormalization, Dropout

from tensorflow.python.keras.preprocessing.sequence import pad_sequences

import numpy as np

import keras

from scipy import sparse

from keras.models import Sequential

from keras.layers import Dense

import random

from tensorflow.python.keras.models import Sequential

from tensorflow.python.keras.layers import Dense, Bidirectional, LSTM, BatchNormalization, Dropout

from tensorflow.python.keras.preprocessing.sequence import pad_sequences

import numpy as np

import keras

from keras.models import Sequential

from keras.layers import Dense

import random

from numpy import array

from numpy import vstack

from keras import optimizers
type1 = df_all[df_all.sentiment==0]

type2 = df_all[df_all.sentiment==1]

type3 = df_all[df_all.sentiment==2]

type1_upsampled = resample(type1,

                          replace=True, # sample with replacement

                          n_samples=len(type3), # match number in majority class

                          random_state=27) # reproducible results

type2_upsampled = resample(type2,

                          replace=True, # sample with replacement

                          n_samples=len(type3), # match number in majority class

                          random_state=27) # reproducible results

type3_upsampled = resample(type3,

                          replace=True, # sample with replacement

                          n_samples=len(type3), # match number in majority class

                          random_state=27) # reproducible results

upsampled = pd.concat([type1_upsampled, type2_upsampled,type3_upsampled])

type1_upsampled.shape,type2_upsampled.shape,type3_upsampled.shape



vectorizer = CountVectorizer(analyzer = 'word', 

                             tokenizer = None,

                             preprocessor = None, 

                             stop_words = None, 

                             min_df = 2, 

                             ngram_range=(2,2),

                             max_features = 10000

                            )

transformer=TfidfTransformer()

pipeline = Pipeline([

    ('vect', vectorizer)

])

pipeline_1 = Pipeline([

    ('vect', vectorizer),

    ('trans', transformer)

])

%time features = pipeline.fit_transform(upsampled['cleanText'])

num_feats =  to_categorical(upsampled['sentiment'])



data = sparse.hstack((features, num_feats))

df_train, df_test = train_test_split(data, test_size=0.2, random_state=42) 
# 1. Dataset

# y_train = df_train['sentiment']

# y_test = df_test['sentiment']



y_train = df_train[:,10000:10003]

y_test = df_test[:,10000:10003]

x_train = df_train[:,0:10000]

x_test = df_test[:,0:10000]

solution = y_test.copy()



# 2. Model Structure

model = keras.models.Sequential()



model.add(keras.layers.Dense(400, input_shape=(10000,)))

model.add(keras.layers.BatchNormalization())

model.add(keras.layers.Activation('relu'))

model.add(keras.layers.Dropout(0.5))



model.add(keras.layers.Dense(200))

model.add(keras.layers.BatchNormalization())

model.add(keras.layers.Activation('relu'))

model.add(keras.layers.Dropout(0.5))



# model.add(keras.layers.Dense(100, activation='softmax'))

model.add(keras.layers.Dense(3, activation='softmax'))



#sgd = optimizers.SGD(lr=0.0001, decay=1e-6, momentum=0.9, nesterov=True)#10

adam=optimizers.Adam(learning_rate=0.0001, beta_1=0.9, beta_2=0.999, amsgrad=False)#8

# 3. Model compile

model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['categorical_accuracy'])

# 4. Train model

hist = model.fit(x=x_train, y=y_train, epochs=20,batch_size=512,validation_data=(x_test,y_test))

#hist = model.fit(x=train_data_features_1, y=y_train, epochs=4,batch_size=512)

# 5. Traing process

%matplotlib inline

import matplotlib.pyplot as plt



fig, loss_ax = plt.subplots()



acc_ax = loss_ax.twinx()



loss_ax.set_ylim([0.0, 2.0])

acc_ax.set_ylim([0.0, 2.0])



loss_ax.plot(hist.history['loss'], 'y', label='train loss')

acc_ax.plot(hist.history['val_loss'], 'b', label='validation loss')



loss_ax.set_xlabel('epoch')

loss_ax.set_ylabel('loss')

acc_ax.set_ylabel('val_loss')



loss_ax.legend(loc='upper left')

acc_ax.legend(loc='lower left')



plt.show()



# 6. Evaluation

loss_and_metrics = model.evaluate(x_test, y_test, batch_size=512)

print('loss_and_metrics : ' + str(loss_and_metrics))



sub_preds_deep = model.predict(x_test,batch_size=512)

resultList=[]

for result in sub_preds_deep:

    resultList.append(np.argmax(result)) 

cm=confusion_matrix(y_pred=resultList, y_true=np.argmax(y_test,axis=1))

plt.figure(figsize = (10,7))

df_cm = pd.DataFrame(cm, index = ["Negative","Neutral","Positive"],

                  columns =  ["Negative","Neutral","Positive"])

sn.heatmap(df_cm, annot=True,fmt='g',cmap="Blues")
vectorizer = CountVectorizer(analyzer = 'word', 

                             tokenizer = None,

                             preprocessor = None, 

                             stop_words = None, 

                             min_df = 2, 

                             ngram_range=(2,2),

                             max_features = 10000

                            )

transformer=TfidfTransformer()

# pipeline = Pipeline([

#     ('vect', vectorizer)

# ])

pipeline = Pipeline([

    ('vect', vectorizer),

    ('trans', transformer)

])

#%time features = pipeline.fit_transform(df_all['cleanText'])

%time features = pipeline.fit_transform(df_all['cleanText'])

df_all['feature']=list(features.toarray())

# num_feats =  df_all['sentiment']

# data = sparse.hstack((features, num_feats))

df_train, df_test = train_test_split(df_all, test_size=0.2, random_state=42) 

type1 = df_train[df_train.sentiment==0]

type2 = df_train[df_train.sentiment==1]

type3 = df_train[df_train.sentiment==2]

type1_upsampled = resample(type1,

                          replace=True, # sample with replacement

                          n_samples=len(type3), # match number in majority class

                          random_state=27) # reproducible results

type2_upsampled = resample(type2,

                          replace=True, # sample with replacement

                          n_samples=len(type3), # match number in majority class

                          random_state=27) # reproducible results

type3_upsampled = resample(type3,

                          replace=True, # sample with replacement

                          n_samples=len(type3), # match number in majority class

                          random_state=27) # reproducible results

upsampled = pd.concat([type1_upsampled, type2_upsampled,type3_upsampled])

type1 = df_test[df_test.sentiment==0]

type2 = df_test[df_test.sentiment==1]

type3 = df_test[df_test.sentiment==2]

type1_upsampled = resample(type1,

                          replace=True, # sample with replacement

                          n_samples=len(type3), # match number in majority class

                          random_state=27) # reproducible results

type2_upsampled = resample(type2,

                          replace=True, # sample with replacement

                          n_samples=len(type3), # match number in majority class

                          random_state=27) # reproducible results

type3_upsampled = resample(type3,

                          replace=True, # sample with replacement

                          n_samples=len(type3), # match number in majority class

                          random_state=27) # reproducible results

upsampled_validate = pd.concat([type1_upsampled, type2_upsampled,type3_upsampled])
# 1. Dataset

# y_train = df_train['sentiment']

# y_test = df_test['sentiment']



y_train = to_categorical(upsampled['sentiment'])

y_test = to_categorical(df_test['sentiment'])

x_train =vstack(upsampled['feature'])

x_test = vstack(df_test['feature'])

y_validate = to_categorical(upsampled_validate['sentiment'])

x_validate =vstack(upsampled_validate['feature'])

solution = y_test.copy()



# 2. Model Structure

model = keras.models.Sequential()



model.add(keras.layers.Dense(300, input_shape=(10000,)))

model.add(keras.layers.BatchNormalization())

model.add(keras.layers.Activation('relu'))

model.add(keras.layers.Dropout(0.5))



model.add(keras.layers.Dense(200))

model.add(keras.layers.BatchNormalization())

model.add(keras.layers.Activation('relu'))

model.add(keras.layers.Dropout(0.5))



# model.add(keras.layers.Dense(100, activation='softmax'))

model.add(keras.layers.Dense(3, activation='softmax'))



sgd = optimizers.SGD(lr=0.0001, decay=1e-6, momentum=0.9, nesterov=True)#10

adam=optimizers.Adam(learning_rate=0.0001, beta_1=0.9, beta_2=0.999, amsgrad=False)#8

# 3. Model compile

model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['categorical_accuracy'])
# vectorizer.get_feature_names()
# from tensorflow.python.keras.models import Sequential

# from tensorflow.python.keras.layers import Dense, Bidirectional, LSTM, BatchNormalization, Dropout

# from tensorflow.python.keras.preprocessing.sequence import pad_sequences

# import numpy as np

# import keras

# from keras.models import Sequential

# from keras.layers import Dense

# import random



# # 1. Dataset

# # y_train = df_train['sentiment']

# # y_test = df_test['sentiment']



# y_train = df_train[:,10000:10003]

# y_test = df_test[:,10000:10003]

# # x_train = df_train[:,0:10000]

# x_test = df_test[:,0:10000]

# solution = y_test.copy()



# # 2. Model Structure

# model = keras.models.Sequential()



# model.add(keras.layers.Dense(400, input_shape=(10000,)))

# model.add(keras.layers.BatchNormalization())

# model.add(keras.layers.Activation('relu'))

# model.add(keras.layers.Dropout(0.5))



# model.add(keras.layers.Dense(200))

# model.add(keras.layers.BatchNormalization())

# model.add(keras.layers.Activation('relu'))

# model.add(keras.layers.Dropout(0.5))



# # model.add(keras.layers.Dense(100, activation='softmax'))

# model.add(keras.layers.Dense(3, activation='softmax'))

# from keras import optimizers

# #sgd = optimizers.SGD(lr=0.0001, decay=1e-6, momentum=0.9, nesterov=True)#10

# #adam=optimizers.Adam(learning_rate=0.0001, beta_1=0.9, beta_2=0.999, amsgrad=False)#8

# # 3. Model compile

# model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['categorical_accuracy'])
print(model.summary())
# 4. Train model

hist = model.fit(x=x_train, y=y_train, epochs=15,batch_size=512,validation_data=(x_test,y_test))

#hist = model.fit(x=train_data_features_1, y=y_train, epochs=4,batch_size=512)

# 5. Traing process

%matplotlib inline

import matplotlib.pyplot as plt



fig, loss_ax = plt.subplots()



acc_ax = loss_ax.twinx()



loss_ax.set_ylim([0.0, 1.0])

acc_ax.set_ylim([0.0, 1.0])



loss_ax.plot(hist.history['categorical_accuracy'], 'y', label='train accuracy')

acc_ax.plot(hist.history['val_categorical_accuracy'], 'b', label='validation accuracy')



loss_ax.set_xlabel('epoch')

loss_ax.set_ylabel('accuracy')

acc_ax.set_ylabel('val_accuracy')



loss_ax.legend(loc='upper left')

acc_ax.legend(loc='lower left')



plt.show()



# 6. Evaluation

loss_and_metrics = model.evaluate(x_test, y_test, batch_size=512)

print('loss_and_metrics : ' + str(loss_and_metrics))



sub_preds_deep = model.predict(x_test,batch_size=512)

resultList=[]

for result in sub_preds_deep:

    resultList.append(np.argmax(result)) 

cm=confusion_matrix(y_pred=resultList, y_true=np.argmax(y_test,axis=1))

plt.figure(figsize = (10,7))

df_cm = pd.DataFrame(cm, index = ["Negative","Neutral","Positive"],

                  columns =  ["Negative","Neutral","Positive"])

sn.heatmap(df_cm, annot=True,fmt='g',cmap="Blues")