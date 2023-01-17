import numpy as np
import scipy.stats as stats
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
%config InlineBackend.figure_format = 'retina'
rtrain = pd.read_csv('../input/train.csv')
# quick look at dataset
rtrain.head(7)
# splitting dataset into X and y subsets
yraw = rtrain.iloc[:, 2:]
Xraw = rtrain.iloc[:,:2]
# adding 2 columns to target variables for EDA 
yraw['is_dirty'] = yraw.apply(max, axis=1)
yraw['sum'] = yraw.apply(sum, axis=1)
# checking for class imbalance (clean vs dirty)
plt.figure(figsize=(6, 8))
ax = sns.countplot(yraw['is_dirty'])
ax.set_xticklabels(['Total clean ', 'Total dirty '],fontsize=12)
sns.despine(bottom=True)
ax.set_xlabel('')
ax.text(-0.10 , 146000, str(round(yraw['is_dirty'].value_counts()[0]/len(yraw)*100,1))+'%',fontsize=13)
ax.text(0.90 , 19000, str(round(yraw['is_dirty'].value_counts()[1]/len(yraw)*100,1))+'%',fontsize=13)
plt.show()
# checking distribution of dirty comments
plt.figure(figsize=(10, 6))
sns.barplot(x=yraw.iloc[:, :6].apply(sum).sort_values(ascending=False), y=yraw.iloc[:, :6].columns, palette='RdYlBu')
plt.show()
# checking distribution of sum of labels (i.e. how many labels a 'dirty' comment has )
sns.barplot(y = yraw['sum'].value_counts()[1:], x=yraw['sum'].value_counts()[1:].index, palette='coolwarm')
sns.despine()
# just curious to see examples of comments with all 6 labels
for comment in Xraw[yraw['sum'] == 6]['comment_text'].head():
    print(comment)
    print('===')
sns.clustermap(yraw.corr())
# used heatmap to construct quick and dirty venn diagram to check if there are any relationship bwt labels
plt.figure(figsize=(15, 6))
heat = yraw[yraw['is_dirty'] != 0].loc[:,['toxic','insult','obscene','severe_toxic','identity_hate','threat']]
heat.sort_values(by=heat.columns.tolist(),inplace=True)
sns.heatmap(heat, yticklabels=False,cbar=False,cmap='viridis')
plt.show()
# examples of the rare dirty comments that are NOT toxic
for comment in Xraw[(yraw['toxic'] == 0) & (yraw['is_dirty'] == 1)]['comment_text'].head():
    print(comment)
    print('===')
# target = 'toxic'
# test = yraw.iloc[:, :6].melt(id_vars=target)
# sns.factorplot(x='value', data = test[test['value'] == 1], kind='count', hue=target, col='variable', row='value')
Xraw.rename(columns={'comment_text' : 'text'}, inplace=True)
Xraw['char_count'] = Xraw['text'].apply(len)
plt.figure(figsize=(10, 4))
plt.hist(Xraw[yraw['is_dirty']==0]['char_count'], bins=50, label='clean')
plt.hist(Xraw[yraw['is_dirty']==1]['char_count'], bins=50, label='dirty')

plt.axvline(Xraw[yraw['is_dirty']==0]['char_count'].mean(),c='black', lw=1, ls='--')
plt.axvline(Xraw[yraw['is_dirty']==1]['char_count'].mean(),c='red', lw=1, ls='--')
plt.title('Character count')
sns.despine()
plt.show()
Xraw['char_count'].mean(), Xraw['char_count'].std()
(Xraw[yraw['is_dirty']==0]['char_count']).mean(), \
(Xraw[yraw['is_dirty']==1]['char_count']).mean()
plt.figure(figsize=(15, 6))
plt.hist(np.log(Xraw[yraw['is_dirty']==0]['char_count']), bins=50, label='clean')
plt.hist(np.log(Xraw[yraw['is_dirty']==1]['char_count']), bins=50, label='dirty')

plt.axvline(np.log(Xraw[yraw['is_dirty']==0]['char_count']).mean(),c='black', lw=2, ls='--')
plt.axvline(np.log(Xraw[yraw['is_dirty']==1]['char_count']).mean(),c='red', lw=2, ls='--')

plt.title('Char count (log) / total no. of toxic classes')
plt.legend()
plt.show()
Xraw['log_char_count'] = np.log(Xraw['char_count'])
# plt.hist(np.log(Xraw[yraw['is_dirty']==1]['char_count']), bins=50, label='clean', color='orange')
# plt.axvline(np.log(Xraw[yraw['is_dirty']==1]['char_count']).mean(),c='red', lw=2, ls='--')

# plt.show()
Xraw.hist(column='log_char_count', by=yraw['is_dirty'], bins=50,figsize=(15,6),)
plt.show()
Xraw[(yraw['is_dirty'] ==0) & (Xraw['char_count'] == 5000)]['text']
# one guy who managed to hit 5000 characters without any spam *_*
print(Xraw.loc[25702,'text'])
Xraw[(yraw['is_dirty']==1) & (Xraw['char_count'] == 5000)].head()
word_count = Xraw['text'].apply(lambda x: x.replace('\n',' ').split())
Xraw['word_count'] = word_count.str.len()
Xraw['unique_word_count'] = word_count.apply(lambda x: len(set(x)))
Xraw['unique_ratio'] = Xraw['unique_word_count']/Xraw['word_count']
Xraw.hist(column='unique_ratio', by=yraw['is_dirty'], bins=50,figsize=(15,6),)
plt.show()
sns.barplot(x=yraw['sum'], y=Xraw['unique_ratio'], palette='coolwarm')
plt.show()
yraw[Xraw['unique_ratio'] < 0.2]['is_dirty'].value_counts()/(Xraw['unique_ratio'] < 0.2).sum()
sns.barplot(yraw['sum'], y=Xraw['char_count'])
plt.show()
# stock-take of the features we currently have
Xraw.head()
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, ENGLISH_STOP_WORDS as esw
from nltk.corpus import stopwords
import spacy
import en_core_web_sm
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import sent_tokenize, word_tokenize
import re
# tokenise by sentence first (referenced from vader documentation)
Xraw['text'] = Xraw['text'].apply(lambda x : x.replace('\n', ' '))
Xsent = Xraw['text'].apply(lambda x: sent_tokenize(x))
vader = SentimentIntensityAnalyzer()
# extracting the average sentence polarity scores in each comment
sentiment = []
for row in Xsent:
    sente_senti = [vader.polarity_scores(sentence) for sentence in row]
    neg = np.mean([senti['neg'] for senti in sente_senti])
    neu = np.mean([senti['neu'] for senti in sente_senti])
    pos = np.mean([senti['pos'] for senti in sente_senti])
    compound = np.mean([senti['compound'] for senti in sente_senti])
    sentiment.append([neg, neu, pos, compound])
sentiment = pd.DataFrame(sentiment, columns=['neg', 'neu', 'pos', 'compound'])
sentiment.head()
Xraw = Xraw.join(sentiment)
# negative sentiment as a feature
quant_features = yraw.join(Xraw['neg'])
quant_plot = pd.melt(quant_features, id_vars='neg')
g = sns.FacetGrid(quant_plot, col='variable', col_wrap=4, sharex=False, sharey=False)
g = g.map(sns.barplot, 'value','neg', palette= 'coolwarm')
g = sns.FacetGrid(quant_plot[quant_plot['value'] == 1], col='variable', col_wrap=4, sharex=False, sharey=False)
g = g.map(sns.violinplot, 'value','neg', color='lightgrey')
#toxic vs severe toxic : mean of neg sentiment and dist diff
# importing some publicly available profanity libraries
from urllib import request
url1 = 'https://raw.githubusercontent.com/RobertJGabriel/Google-profanity-words/master/list.txt'
txt1 = request.urlopen(url1).readlines()
url2 = 'https://raw.githubusercontent.com/areebbeigh/profanityfilter/master/profanityfilter/data/badwords.txt'
txt2 = request.urlopen(url2).readlines()
google_profanity = [line.decode("utf-8").replace('\n','') for line in txt1]
areeb_profanity = [line.decode("utf-8").replace('\n','').lower() for line in txt2]
profanities = list(set(google_profanity + areeb_profanity))
# some basic text pre-processing
def lower_case(word_array):
    word_array = word_array.str.lower()
    return word_array

stop = stopwords.words('english')
def remove_stopwords(word_array):
    word_array = word_array.apply(lambda x: " ".join(x for x in x.split() if x not in stop))
    return word_array
    
def remove_punc(word_array):
    word_array = word_array.str.replace('[^\w\s]' ,'')
    word_array = word_array.str.replace('\n','')
    word_array = word_array.str.replace('  ' ,' ')
    return word_array

st = PorterStemmer()
def stem_words(word_array):
    word_array = word_array.apply(lambda x: " ".join([st.stem(word) for word in x.split()]))
    return word_array

def tokenize(word_array):
    word_array = word_array.apply(word_tokenize)
    return word_array

wnl = WordNetLemmatizer()
def lemm_words(word_array):
    word_array = word_array.apply(lambda x: " ".join([wnl.lemmatize(word) for word in x.split()]))
    return word_array

def remove_numbers(word_array):
    word_array = word_array.apply(lambda x : re.sub(r'\d+', '', x))
    return word_array
def clean_pipeline(word_array):
    word_array = lower_case(word_array)
    word_array = remove_stopwords(word_array)
    #word_array = remove_punc(word_array)
    word_array = lemm_words(word_array)
    #word_array = remove_numbers(word_array)
    #word_array = stem_words(word_array)
    word_array = tokenize(word_array)
    return word_array
Xword = clean_pipeline(Xraw['text'])
profanity_count = []
for comment in Xword:
    profanity_count.append(len([x for x in comment if x in profanities]))
Xraw.shape
Xraw['profanity_count'] = profanity_count
# no. of profanities as a feature
quant_features = yraw.join(Xraw['profanity_count'])
quant_plot = pd.melt(quant_features, id_vars='profanity_count')
g = sns.FacetGrid(quant_plot, col='variable', col_wrap=4, sharex=False, sharey=False)
g = g.map(sns.barplot, 'value','profanity_count', palette= 'coolwarm')
import string
string.punctuation
punc = [len(re.findall('[!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~]', comment)) for comment in Xraw['text']]
Xraw['punc'] = punc
# no of punc as a feature
quant_features = yraw.join(Xraw['punc'])
quant_plot = pd.melt(quant_features, id_vars='punc')
g = sns.FacetGrid(quant_plot, col='variable', col_wrap=4, sharex=False, sharey=False)
g = g.map(sns.barplot, 'value','punc', palette= 'coolwarm')
rtrain[Xraw['punc'] > 1000]
Xraw.loc[67761, 'text']
caps = [len(re.findall('[A-Z]', comment)) for comment in Xraw['text']]
Xraw['caps'] = caps
# no of capital letters as a feature
quant_features = yraw.join(Xraw['caps'])
quant_plot = pd.melt(quant_features, id_vars='caps')
g = sns.FacetGrid(quant_plot, col='variable', col_wrap=4, sharex=False, sharey=False)
g = g.map(sns.barplot, 'value','caps', palette= 'coolwarm')
def clean_pipeline(word_array):
    word_array = lower_case(word_array)
    word_array = remove_stopwords(word_array)
    word_array = remove_punc(word_array)
    word_array = lemm_words(word_array)
    word_array = remove_numbers(word_array)
    #word_array = stem_words(word_array)
    #word_array = tokenize(word_array)
    return word_array
yraw.columns[:6]
temp_list = [[],[],[],[],[],[]]
for i, column in enumerate(yraw.columns[:6]):
    temp_list[i] = Xraw[yraw[column] == 1]['text'].head(4000)
toxic_only = clean_pipeline(temp_list[0])
severe_toxic_only = clean_pipeline(temp_list[1])
obscene_only = clean_pipeline(temp_list[2])
threat_only = clean_pipeline(temp_list[3])
insult_only = clean_pipeline(temp_list[4])
identity_hate_only = clean_pipeline(temp_list[5])
tfv1 = TfidfVectorizer(min_df=10,  max_features=10000, 
            strip_accents='unicode', analyzer='word',ngram_range=(1,2),
            use_idf=1,smooth_idf=1,sublinear_tf=1, lowercase=False)
unigram = tfv1.fit_transform(toxic_only)
features = np.array(tfv1.get_feature_names())
temp_df = pd.DataFrame(unigram.todense(), columns=features)
toxic_only_tfidf = temp_df.sum().sort_values()
unigram = tfv1.fit_transform(severe_toxic_only)
features = np.array(tfv1.get_feature_names())
temp_df = pd.DataFrame(unigram.todense(), columns=features)
severe_toxic_only_tfidf = temp_df.sum().sort_values()
unigram = tfv1.fit_transform(obscene_only)
features = np.array(tfv1.get_feature_names())
temp_df = pd.DataFrame(unigram.todense(), columns=features)
obscene_only_tfidf = temp_df.sum().sort_values()
unigram = tfv1.fit_transform(threat_only)
features = np.array(tfv1.get_feature_names())
temp_df = pd.DataFrame(unigram.todense(), columns=features)
threat_only_tfidf = temp_df.sum().sort_values()
unigram = tfv1.fit_transform(insult_only)
features = np.array(tfv1.get_feature_names())
temp_df = pd.DataFrame(unigram.todense(), columns=features)
insult_only_tfidf = temp_df.sum().sort_values()
tfv1.fit(identity_hate_only)
identity_hate_only_unigram =  tfv1.transform(identity_hate_only)
features = np.array(tfv1.get_feature_names())
identity_hate_tfidf = pd.DataFrame(identity_hate_only_unigram.todense(), columns=features)
identity_hate_only_tfidf = identity_hate_tfidf.sum().sort_values()
toxic_only_tfidf.tail(9).plot(kind='barh',color='salmon')
plt.title('toxic')
plt.show()
severe_toxic_only_tfidf.tail(9).plot(kind='barh',color='darkred')
plt.title('severe toxic')
plt.show()
obscene_only_tfidf.tail(9).plot(kind='barh',color='turquoise')
plt.title('obscene')
plt.show()
plt.figure(figsize=(7,3))
threat_only_tfidf.tail(7).plot(kind='barh',color='salmon')
plt.title('threat')
plt.show()
insult_only_tfidf.tail(9).plot(kind='barh',color='gold')
plt.title('insult')
plt.show()
plt.figure(figsize=(7,3))
identity_hate_only_tfidf.tail(7).plot(kind='barh',color='gold')
plt.title('identity hate')
plt.show()
tfv2 = TfidfVectorizer(min_df=10,  max_features=30000, 
            strip_accents='unicode', analyzer='word',ngram_range=(2,2),
            use_idf=1,smooth_idf=1,sublinear_tf=1)
unigram = tfv2.fit_transform(toxic_only)
features = np.array(tfv2.get_feature_names())
temp_df = pd.DataFrame(unigram.todense(), columns=features)
toxic_only_tfidf2 = temp_df.sum().sort_values()
unigram = tfv2.fit_transform(severe_toxic_only)
features = np.array(tfv2.get_feature_names())
temp_df = pd.DataFrame(unigram.todense(), columns=features)
severe_toxic_only_tfidf2 = temp_df.sum().sort_values()
unigram = tfv2.fit_transform(obscene_only)
features = np.array(tfv2.get_feature_names())
temp_df = pd.DataFrame(unigram.todense(), columns=features)
obscene_only_tfidf2 = temp_df.sum().sort_values()
unigram = tfv2.fit_transform(threat_only)
features = np.array(tfv2.get_feature_names())
temp_df = pd.DataFrame(unigram.todense(), columns=features)
threat_only_tfidf2 = temp_df.sum().sort_values()
unigram = tfv2.fit_transform(insult_only)
features = np.array(tfv2.get_feature_names())
temp_df = pd.DataFrame(unigram.todense(), columns=features)
insult_only_tfidf2 = temp_df.sum().sort_values()
unigram = tfv2.fit_transform(identity_hate_only)
features = np.array(tfv2.get_feature_names())
temp_df = pd.DataFrame(unigram.todense(), columns=features)
identity_hate_only_tfidf2 = temp_df.sum().sort_values()
toxic_only_tfidf2.tail(9).plot(kind='barh',color='salmon')
plt.title('toxic')
plt.show()
severe_toxic_only_tfidf2.tail(9).plot(kind='barh',color='gold')
plt.title('severe toxic')
plt.show()
obscene_only_tfidf2.tail(9).plot(kind='barh',color='turquoise')
plt.title('obscene')
plt.show()
threat_only_tfidf2.tail(9).plot(kind='barh',color='darkgreen')
plt.title('threat')
plt.show()
insult_only_tfidf2.tail(9).plot(kind='barh',color='darkred')
plt.title('insult')
plt.show()
identity_hate_only_tfidf2.tail(9).plot(kind='barh',color='darkblue')
plt.title('identity hate')
plt.show()