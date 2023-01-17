# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib as mlp
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="darkgrid")
from datetime import datetime
from random import random
import re
import string
import nltk
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import cross_val_score,cross_val_predict,StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator,TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from collections import Counter
import tensorflow_hub as hub
from nltk.tokenize import TweetTokenizer
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_auc_score,roc_curve,auc
from sklearn.ensemble import RandomForestClassifier,ExtraTreesClassifier,VotingClassifier
from sklearn.svm import SVC

!wget --quiet https://raw.githubusercontent.com/tensorflow/models/master/official/nlp/bert/tokenization.py
import tokenization
nltk.download('stopwords')
nltk.download('wordnet')

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
MLP_XKCD_COLOR = mlp.colors.XKCD_COLORS
MLP_BASE_COLOR = mlp.colors.BASE_COLORS
MLP_CNAMES = mlp.colors.cnames
MLP_CSS4 = mlp.colors.CSS4_COLORS
MLP_HEX = mlp.colors.hexColorPattern
MLP_TABLEAU = mlp.colors.TABLEAU_COLORS
print('I like COLORS :>')
def random_color_generator(color_type=None):
    if color_type is None:
        colors = sorted(MLP_CNAMES.items(), key=lambda x: random())
    else:
        colors = sorted(color_type.items(), key=lambda x: random())
    return dict(colors)
def timer(start_time=None):
    if not start_time:
        start_time = datetime.now()
        return start_time
    elif start_time:
        thour, temp_sec = divmod((datetime.now() - start_time).total_seconds(), 3600)
        tmin, tsec = divmod(temp_sec, 60)
        print('\n Time taken: %i hours %i minutes and %s seconds.' % (thour, tmin, round(tsec, 2)))
path = '/kaggle/input/nlp-getting-started/'
train = pd.read_csv(path+'train.csv')
test = pd.read_csv(path+'test.csv')
train.head(2)
print(train.shape)
print('Description -- \n',train.describe())
print('Information -- \n',train.info())
print('Checking for Null values -- \n',train.isnull().sum())
print(test.shape)
print('Description -- \n',test.describe())
print('Information -- \n',test.info())
print('Checking for Null values -- \n',test.isnull().sum())
def plot_feature_distribution(df):
    df1 = pd.DataFrame(df.count()).reset_index()
    df1.columns = ['features','Total Count']
    df1['missing_values'] = df.isnull().sum().values
    return df1
fig, axes = plt.subplots(ncols=2, figsize=(17, 4), dpi=100)
feat_train = plot_feature_distribution(train)
feat_test = plot_feature_distribution(test)
colors=random_color_generator()
feat_train.set_index('features').plot.bar(stacked=True,color=colors,ax=axes[0])
colors=random_color_generator()
feat_test.set_index('features').plot.bar(stacked=True,color=colors,ax=axes[1])
axes[0].set_title('Training Set', fontsize=13)
axes[1].set_title('Test Set', fontsize=13)
plt.show()
feat_train.style.background_gradient(cmap='Reds',subset=['missing_values'])
missing_feat = list(feat_train[feat_train['missing_values'] != 0]['features'].values)
def fill_na(df):
    for feature in missing_feat:
        df[feature].fillna('Unknown',inplace=True)
    return df    
train = fill_na(train)
test = fill_na(test)
print('Checking for Null Values in Training Set -- \n',train.isnull().sum())
print('Checking for Null Values in Testing Set -- \n',test.isnull().sum())
fig, axes = plt.subplots(ncols=2, figsize=(17, 4), dpi=100)
colors=random_color_generator()
train.target.value_counts().plot(kind='bar',color=colors,ax=axes[0])
train.target.value_counts().plot(kind='pie',labels=['Non Disaster','Disaster'])
axes[0].set_title('Target Distribution in Training Set', fontsize=13)
axes[1].set_title('Target Count in Training Set', fontsize=13)
axes[0].set_xticklabels(['Not Disaster (4342)', 'Disaster (3271)'])
plt.show()
start_time = timer(None)
colors=random_color_generator(MLP_CSS4)
train.groupby('keyword')['target'].value_counts().unstack().plot.barh(stacked=True,color=colors,figsize=(8,72))
plt.tick_params(axis='x', labelsize=15)
plt.tick_params(axis='y', labelsize=12)
plt.legend(loc=1)
plt.title('Target Distribution in Keywords')
plt.show()
timer(start_time)
STOPWORDS = set(stopwords.words('english'))
PUNCTUATIONS = string.punctuation
print(PUNCTUATIONS)
def generate_meta_features(df):
    df['word_count'] = df['text'].apply(lambda x : len(str(x).split()))
    df['char_count'] = df['text'].apply(lambda x : len(x))
    df['mean_word_length'] = df['text'].apply(lambda x: np.mean([len(w) for w in str(x).split()]))
    df['@count'] = df['text'].apply(lambda x: len([c for c in str(x) if c == '@']))
    df['#count'] = df['text'].apply(lambda x: len([c for c in str(x) if c == '#']))
    df['stop_count'] = df['text'].apply(lambda x: len([s for s in str(x).split() if s in STOPWORDS]))
    df['punc_count'] = df['text'].apply(lambda x: len([p for p in str(x) if p in PUNCTUATIONS]))
    df['url_count'] = df['text'].apply(lambda x: len([u for u in str(x).split() if 'https' in u or 'http' in u or 'www' in u]))
    df['unique_words'] = df['text'].apply(lambda x: len(set(str(x).split())))
    df['punc_word_ratio'] = df['punc_count']/df['word_count']
    df['stop_word_ratio'] = df['stop_count']/df['word_count']
    df['punc_stop_word_ratio'] = (df['punc_count'] + df['stop_count']) /df['word_count']
    df['unique_word_ratio'] = df['unique_words'] / df['word_count']
    return df
meta_train = train.copy()
meta_train = generate_meta_features(meta_train)
meta_train.head(2)
meta_test = test.copy()
meta_test = generate_meta_features(meta_test)
meta_test.head(2)
def plot_feature_ratio(df):
    fig,axes = plt.subplots(ncols=3, figsize=(17, 4), dpi=100)
    colors=random_color_generator()
    df.groupby('target')['punc_word_ratio'].mean().plot(kind='bar',color=colors,ax=axes[0])
    colors=random_color_generator()
    df.groupby('target')['stop_word_ratio'].mean().plot(kind='bar',color=colors,ax=axes[1])
    colors=random_color_generator()
    df.groupby('target')['unique_word_ratio'].mean().plot(kind='bar',color=colors,ax=axes[2])
    axes[0].set_title('Punctuation vs Total Words Ratio in Training Set', fontsize=13)
    axes[1].set_title('Stopwords vs Total Words Ratio in Training Set', fontsize=13)
    axes[2].set_title('Unique Words vs Total Words Ratio in Training Set', fontsize=13)
    axes[0].set_xticklabels(['Not Disaster', 'Disaster'])
    axes[1].set_xticklabels(['Not Disaster', 'Disaster'])
    axes[2].set_xticklabels(['Not Disaster', 'Disaster'])
    plt.show()
plot_feature_ratio(meta_train)
start_time = timer(None)
meta_columns = set(train.columns.values) ^ set(meta_train.columns.values)
fig,axes = plt.subplots(ncols=2,nrows=len(meta_columns),figsize=(30, 50), dpi=100)
for i,feature in enumerate(meta_columns): 
    sns.distplot(meta_train[feature],label='Train',color='blue',ax=axes[i][0],rug=True,kde_kws={'bw':0.1})
    sns.distplot(meta_test[feature],label='Test',color='green',ax=axes[i][0],rug=True,kde_kws={'bw':0.1})
    axes[i][0].set_title(feature+' Distribution in Testing ang Training Set')
    sns.distplot(meta_train[meta_train['target']==0][feature],label='Not Disaster',color='red',ax=axes[i][1],rug=True,kde_kws={'bw':0.1})
    sns.distplot(meta_train[meta_train['target']==1][feature],label='Disaster',color='green',ax=axes[i][1],rug=True,kde_kws={'bw':0.1})
    axes[i][1].set_title(feature+' Target Distribution')
    axes[i][0].set_xlabel('')
    axes[i][1].set_xlabel('')
    axes[i][0].legend(loc='best')
    axes[i][1].legend(loc='best')
plt.show()
timer(start_time)
%time
fasttext_emb = np.load('../input/pickled-crawl300d2m-for-kernel-competitions/crawl-300d-2M.pkl', allow_pickle=True)
glove_emb = np.load('../input/pickled-glove840b300d-for-10sec-loading/glove.840B.300d.pkl',allow_pickle=True)
def build_vocab(df):
    tweets = TweetTokenizer().tokenize(" ".join(df))
    vocab = Counter(tweets)
    return vocab

def checkEmbeddingCoverage(df,embeddings):
    vocab = build_vocab(df)
    covered ={}
    nCovered = 0
    notCovered={}
    nNotCovered=0
    for word in vocab:
        try:
            covered[word] = embeddings[word]
            nCovered += vocab[word]
        except KeyError:
            notCovered[word] = vocab[word]
            nNotCovered += vocab[word]
    vocab_coverage = len(covered)/ len(vocab)
    text_coverage = nCovered/ (nCovered+nNotCovered)
    return nCovered,nNotCovered,covered,notCovered,vocab_coverage,text_coverage
print('| ------------- Word Embedding Summary Before Cleaning ------------- |')
embeddings = [glove_emb,fasttext_emb]
for dataset_name,dataset in zip(['meta_train','meta_test'],[meta_train,meta_test]):
    for emb_name,emb in zip(['Glove Embeddings','FastText Embeddings'],embeddings):
        print('<---------',str(dataset_name),'|----| ',emb_name,' ------------------>')
        nCovered,nNotCovered,covered,notCovered,vocab_coverage,text_coverage = checkEmbeddingCoverage(dataset['text'],emb)
        print('Total No. of Words Covered in ',emb_name,' embeddings -> ',nCovered)
        print('Total No. of Words NOT Covered in ',emb_name,' embeddings -> ',nNotCovered)
        print('Total Vocabulary Covered -> ',round(vocab_coverage * 100,2),' %')
        print('Total Text Covered -> ',round(text_coverage * 100,2),' %')
def clean_keyword(keyword):
    keyword = ''.join([p for p in str(keyword) if p not in PUNCTUATIONS])
    keyword = re.sub(r'[\d-]',r' ',keyword)
    keyword = re.sub(r'\s+',r' ',keyword)
    return keyword
#Example
clean_keyword('burning%20buildings') 
def clean_text(text):
    text = text.lower()
    text = ' '.join([word for word in str(text).split() if word not in STOPWORDS])
    text = re.sub(r'https?://\S+|www\.\S+',r'',text)
    text = ''.join([p for p in str(text) if p not in PUNCTUATIONS])
    text = re.sub(r'[\d-]',r"",text)
    text = re.sub(r'\s+',r" ",text)
    #Special Characters
    text = re.sub(r'\x89','',text)
    text = re.sub(r'ûªs','us',text)
    text = re.sub(r'ûª','you',text)
    text = re.sub(r'ûò','',text)
    text = re.sub(r'\x9d','',text)
    text = re.sub(r'prebreak','pre break',text)
    text = re.sub(r'ûªt','',text)
    text = re.sub(r'ûó','',text)
    text = re.sub(r'typhoondevastated','typhoon devastated',text)
    text = re.sub(r'bestnaijamade','',text)
    text = re.sub(r'soudelor','Typhoon',text)
    text = re.sub(r'gbbo','',text)
    text = re.sub(r'funtenna','software',text)
    text = re.sub(r'ûªve','you have',text)
    text = re.sub(r'bayelsa','state',text) 
    text = re.sub(r'marians','music artist',text)
    text = re.sub(r'udhampur','indian city',text)
    text = re.sub(r'ûïwhen','',text)
    text = re.sub(r'sensorsenso','sensor',text)
    text = re.sub(r'arianagrande','music artist',text)
    text = re.sub(r'selfimage','self image',text)
    text = re.sub(r'irandeal','Iran deal',text)
    text = re.sub(r'trfc','Football Club',text)
    return text
meta_train['cleaned_text'] = meta_train.text.apply(lambda x : clean_text(x))
meta_train['cleaned_keyword'] = meta_train.keyword.apply(lambda k : clean_keyword(k))
meta_test['cleaned_text'] = meta_test.text.apply(lambda x : clean_text(x))
meta_test['cleaned_keyword'] = meta_test.keyword.apply(lambda k : clean_keyword(k))
meta_train.head(2)
print('| ------------- Word Embedding Summary After Cleaning ------------- |')
embeddings = [glove_emb,fasttext_emb]
for dataset_name,dataset in zip(['meta_train','meta_test'],[meta_train,meta_test]):
    for emb_name,emb in zip(['Glove Embeddings','FastText Embeddings'],embeddings):
        print('<---------',str(dataset_name),'|----| ',emb_name,' ------------------>')
        nCovered,nNotCovered,covered,notCovered,vocab_coverage,text_coverage = checkEmbeddingCoverage(dataset['cleaned_text'],emb)
        print('Total No. of Words Covered in ',emb_name,' embeddings -> ',nCovered)
        print('Total No. of Words NOT Covered in ',emb_name,' embeddings -> ',nNotCovered)
        print('Total Vocabulary Covered -> ',round(vocab_coverage * 100,2),' %')
        print('Total Text Covered -> ',round(text_coverage * 100,2),' %')
# import operator
# dict(sorted(notCovered.items(), key=operator.itemgetter(1),reverse=True))
import gc
del glove_emb, fasttext_emb,nCovered,nNotCovered,covered,notCovered,vocab_coverage,text_coverage
gc.collect()
colors = random_color_generator()
meta_train['cleaned_keyword'].value_counts()[:20].sort_values(ascending=True).plot.barh(color=colors,figsize=(10,10))
plt.tick_params(axis='x', labelsize=15)
plt.tick_params(axis='y', labelsize=12)
plt.legend(loc='best')
plt.title('Top 20 Most Comman Keywords')
plt.show()
from wordcloud import WordCloud
colors = random_color_generator()
def show_word_cloud(data,title=None):
    word_cloud = WordCloud(
        background_color = list(colors.keys())[1],
        max_words =100,
        width=800,
        height=400,
        stopwords=STOPWORDS,
        max_font_size = 40, 
        scale = 3,
        random_state = 42 ).generate(data)
    fig = plt.figure(1, figsize = (20, 20))
    plt.axis('off')
    if title: 
        fig.suptitle(title, fontsize = 20)
        fig.subplots_adjust(top = 2.3)

    plt.imshow(word_cloud)
    plt.show()
#Most Comman words in Disaster and Non Disaster Tweets
disaster = " ".join(meta_train[meta_train.target==1]['cleaned_text'])
show_word_cloud(disaster,'TOP 100 DISASTER Words')
#Most Comman words in Disaster and Non Disaster Tweets
notdisaster = " ".join(meta_train[meta_train.target==0]['cleaned_text'])
show_word_cloud(notdisaster,'TOP 100 NON-DISASTER Words')
current_palette = sns.color_palette()
sns.palplot(current_palette)
def get_top_n_words(corpus,n_grams=None):
    vec = CountVectorizer(ngram_range=(n_grams,n_grams)).fit(corpus)
    bag_of_words = vec.transform(corpus)
    
    sum_of_words = bag_of_words.sum(axis=0)
    word_freq = [(word, sum_of_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    word_freq = sorted(word_freq, key = lambda x: x[1], reverse=True)
    return word_freq[:20]
fig,axes = plt.subplots(ncols=2,figsize=(12,7),dpi=100)
top_n_bigrams_d = get_top_n_words(meta_train[meta_train.target==1]['cleaned_text'],2)
top_n_bigrams_nd = get_top_n_words(meta_train[meta_train.target==0]['cleaned_text'],2)
x_d,y_d = map(list,zip(*top_n_bigrams_d))
axes[0].set_title('Top 20 Disaster Bi-grams')
sns.barplot(x=y_d,y=x_d,ax=axes[0])
x_nd,y_nd = map(list,zip(*top_n_bigrams_nd))
axes[1].set_title('Top 20 Non-Disaster Bi-grams')
sns.barplot(x=y_nd,y=x_nd,ax=axes[1],palette='cubehelix')
plt.show()
fig,axes = plt.subplots(ncols=2,figsize=(10,7),dpi=100)
top_n_trigrams_d = get_top_n_words(meta_train[meta_train.target==1]['cleaned_text'],3)
top_n_trigrams_nd = get_top_n_words(meta_train[meta_train.target==0]['cleaned_text'],3)
x_d,y_d = map(list,zip(*top_n_trigrams_d))
axes[0].set_title('Top 20 Disaster Tri-grams')
sns.barplot(x=y_d,y=x_d,ax=axes[0],palette=sns.color_palette("hls", 5))
x_nd,y_nd = map(list,zip(*top_n_trigrams_nd))
axes[1].set_title('Top 20 Non-Disaster Tri-grams')
sns.barplot(x=y_nd,y=x_nd,ax=axes[1],palette=sns.color_palette("Set2", 8))
plt.show()
meta_train.head(2)
#X = meta_train['location']+' '+ meta_train['cleaned_keyword']+' '+ meta_train['cleaned_text']
X =  meta_train['cleaned_text']
X.head()
y=meta_train['target']
y.head()
#test = meta_test['location']+' '+ meta_test['cleaned_keyword']+' '+ meta_test['cleaned_text']
test =  meta_test['cleaned_text']
test.head()
def bert_encode(texts, tokenizer, max_len=512):
    all_tokens = []
    all_masks = []
    all_segments = []
    
    for text in texts:
        text = tokenizer.tokenize(text)
            
        text = text[:max_len-2]
        input_sequence = ["[CLS]"] + text + ["[SEP]"]
        pad_len = max_len - len(input_sequence)
        
        tokens = tokenizer.convert_tokens_to_ids(input_sequence)
        tokens += [0] * pad_len
        pad_masks = [1] * len(input_sequence) + [0] * pad_len
        segment_ids = [0] * max_len
        
        all_tokens.append(tokens)
        all_masks.append(pad_masks)
        all_segments.append(segment_ids)
    
    return np.array(all_tokens), np.array(all_masks), np.array(all_segments)
%%time
module_url = "https://tfhub.dev/tensorflow/bert_en_uncased_L-24_H-1024_A-16/1"
bert_layer = hub.KerasLayer(module_url, trainable=True)
vocab_file = bert_layer.resolved_object.vocab_file.asset_path.numpy()
do_lower_case = bert_layer.resolved_object.do_lower_case.numpy()
tokenizer = tokenization.FullTokenizer(vocab_file, do_lower_case)
train_input = bert_encode(X.values, tokenizer, max_len=160)
test_input = bert_encode(test.values, tokenizer, max_len=160)
train_labels = y.values
train_input_1 = np.array(train_input)
test_input_1 = np.array(test_input)
train_input_1 = train_input_1.reshape(train_input_1.shape[1],train_input_1.shape[0]*train_input_1.shape[2])
test_input_1 = test_input_1.reshape(test_input_1.shape[1],test_input_1.shape[0]*test_input_1.shape[2])
x_train,x_test,y_train,y_test = train_test_split(train_input_1,train_labels,test_size=0.2,random_state=42)
print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)
rf_clf = RandomForestClassifier(n_estimators=1000, random_state=42)
extra_trees_clf = ExtraTreesClassifier(n_estimators=1000, random_state=42)
mlp_clf = MLPClassifier(random_state=42)
nb_clf = MultinomialNB()
svc_clf = SVC(gamma='scale',probability=True , random_state=42)
start_time = timer(None)
y_pred_track=[]
estimators = [rf_clf,extra_trees_clf,mlp_clf,nb_clf,svc_clf]
print('-------------Train ROC-AUC Scores -------------')
for estimator in estimators:
    y_pred = cross_val_predict(estimator,x_train,y_train,cv=3,method='predict_proba')
    y_pred_track.append(y_pred[:,1])
    print(estimator.__class__.__name__,'-->',roc_auc_score(y_train,y_pred[:,1]))
timer(start_time)
start_time = timer(None)
y_pred_track=[]
estimators = [rf_clf,extra_trees_clf,mlp_clf,nb_clf,svc_clf]
print('------------- ROC-AUC Scores -------------')
for estimator in estimators:
    estimator.fit(x_train,y_train)
    y_pred = estimator.predict_proba(x_test)[:,1]
    y_pred_track.append(y_pred)
    print(estimator.__class__.__name__,'-->',roc_auc_score(y_test,y_pred))
timer(start_time)
plt.figure(figsize=(10,8))
plt.title('Reciever Operating Characteristics Curve')
for y_pred,estimator in zip(y_pred_track,estimators):
    colors=random_color_generator()
    frp,trp, threshold = roc_curve(y_test,y_pred)
    roc_auc_ = auc(frp,trp)
    plt.plot(frp,trp,'r',label = '%s AUC = %0.3f' %(estimator.__class__.__name__,roc_auc_),color=list(colors.keys())[1])
plt.legend(loc='lower right')
plt.plot([0,1],[0,1],'b--')
plt.ylabel('True positive rate')
plt.xlabel('False positive rate')
plt.show()
import tensorflow as tf
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint
def build_model(bert_layer, max_len=512):
    input_word_ids = Input(shape=(max_len,), dtype=tf.int32, name="input_word_ids")
    input_mask = Input(shape=(max_len,), dtype=tf.int32, name="input_mask")
    segment_ids = Input(shape=(max_len,), dtype=tf.int32, name="segment_ids")

    _, sequence_output = bert_layer([input_word_ids, input_mask, segment_ids])
    clf_output = sequence_output[:, 0, :]
    out = Dense(1, activation='sigmoid')(clf_output)
    
    model = Model(inputs=[input_word_ids, input_mask, segment_ids], outputs=out)
    model.compile(Adam(lr=0.0001), loss='binary_crossentropy', metrics=['accuracy'])
    
    return model
model = build_model(bert_layer, max_len=16)
model.summary()
from sklearn.model_selection import StratifiedKFold
K = 2
SEED=42
skf = StratifiedKFold(n_splits=K, random_state=SEED, shuffle=True)
for train_idx,test_idx in skf.split(X,y):
    #print('Train Indexes \n',train_idx,'\n Validation Indexes \n',test_idx)
    train_input = bert_encode(X.iloc[train_idx].values, tokenizer, max_len=128)
    val_input = bert_encode(X.iloc[test_idx].values, tokenizer, max_len=128)
    training_batch = train_input
    label_batch = y.iloc[train_idx]
    val_batch = val_input
    val_label_batch = y.iloc[test_idx]
    train_history = model.fit(training_batch, label_batch, validation_data=(val_batch,val_label_batch), epochs=3, batch_size=32)
# train_history = model.fit(
#     train_input, train_labels,
#     validation_split=0.2,
#     epochs=3,
#     batch_size=16
# )

# model.save('model.h5')
# test_pred = model.predict(test_input)