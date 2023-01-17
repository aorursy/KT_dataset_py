# Importing all the required Packages

import os

import pickle

import re

import string

from tqdm import tqdm

from datetime import datetime

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from scipy import sparse

from gensim.models import Word2Vec

import nltk

from nltk.tokenize import word_tokenize

from nltk.stem.snowball import SnowballStemmer

from nltk.stem import WordNetLemmatizer 

from nltk.corpus import stopwords

from textblob import TextBlob

from wordcloud import WordCloud

import plotly.offline as py

import plotly.graph_objs as go

from plotly.offline import init_notebook_mode, iplot

import plotly.figure_factory as ff

init_notebook_mode(connected=True)

from sklearn.preprocessing import MinMaxScaler

from sklearn.manifold import TSNE

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.feature_extraction.text import TfidfVectorizer

import warnings

warnings.filterwarnings('ignore')
# Reading the text data present in the directories. Each review is present as text file.

if not (os.path.isfile('../input/end-to-end-text-processing-for-beginners/train.csv' and 

                       '../input/end-to-end-text-processing-for-beginners/test.csv')):

    path = '../input/imdb-movie-reviews-dataset/aclimdb/aclImdb/'

    train_text = []

    train_label = []

    test_text = []

    test_label = []

    train_data_path_pos = os.path.join(path,'train/pos/')

    train_data_path_neg = os.path.join(path,'train/neg/')



    for data in ['train','test']:

        for label in ['pos','neg']:

            for file in sorted(os.listdir(os.path.join(path,data,label))):

                if file.endswith('.txt'):

                    with open(os.path.join(path,data,label,file)) as file_data:

                        if data=='train':

                            train_text.append(file_data.read())

                            train_label.append( 1 if label== 'pos' else 0)

                        else :

                            test_text.append(file_data.read())

                            test_label.append( 1 if label== 'pos' else 0)



    train_df = pd.DataFrame({'Review': train_text, 'Label': train_label})

    test_df = pd.DataFrame({'Review': test_text, 'Label': test_label})

    train_df = train_df.sample(frac=1).reset_index(drop=True)

    test_df = test_df.sample(frac=1).reset_index(drop=True)

    

    train_df.to_csv('train.csv')

    test_df.to_csv('test.csv')

    

else:

    train_df = pd.read_csv('../input/end-to-end-text-processing-for-beginners/train.csv',index_col=0)

    test_df = pd.read_csv('../input/end-to-end-text-processing-for-beginners/test.csv',index_col=0)

    

print('The shape of train data:',train_df.shape)

print('The shape of test data:', test_df.shape)    
print('The first 5 rows of data:')

train_df.head()
# Basic info

train_df.info()
# Removing the duplicate rows

train_df_nodup = train_df.drop_duplicates(keep='first',inplace=False)

test_df_nodup = test_df.drop_duplicates(keep='first',inplace=False)

print('No of duplicate train rows that are dropped:',len(train_df)-len(train_df_nodup))

print('No of duplicate test rows that are dropped:',len(test_df)-len(test_df_nodup))
# Defining Functions for Polarity, Subjectivity and Parts of Speech counts

stop_words = stopwords.words('english')



def get_polarity(text):

    try:

        tb = TextBlob(str(text))

        polarity = tb.sentiment.polarity

    except:    

        polarity = 0.0

    return polarity    



def get_subjectivity(text):

    try:

        tb = TextBlob(str(text))

        subjectivity = tb.sentiment.subjectivity

    except:    

        subjectivity = 0.0

    return subjectivity





pos_dict = {

    'noun' : ['NN','NNS','NNP','NNPS'],

    'pronoun' : ['PRP','PRP$'],

    'adjective' : ['JJ','JJR','JJS'],

    'adverb' : ['RB','RBR','RBS','WRB'],

    'verb' : ['VB','VBD','VBG','VBN','VBP','VBZ']

           }





def get_pos(text,pos):

    count = 0

    try:

        tokens = nltk.word_tokenize(text)

        for tag in nltk.pos_tag(tokens):

            if tag[1] in pos_dict[pos]:

                count += 1

        return count   

    except:

        pass
# Function for extracting features

def extract_text_features(df):

    df['Word_Count'] = df['Review'].apply(lambda text : len(str(text).split()))

    df['Length'] = df['Review'].apply(len)

    df['Word_Density'] = df['Length']/df['Word_Count']

    df['Stop_Word_Count'] = df['Review'].apply(lambda text: len([word for word in str(text).split() 

                                                                     if word in stop_words]))

    df['Upper_Case_Word_Count'] = df['Review'].apply(lambda text: len([word for word in str(text).split()

                                                                    if word.isupper()]))

    df['Title_Case_Word_Count'] = df['Review'].apply(lambda text: len([word for word in str(text).split()

                                                                    if word.istitle()]))

    df['Numeric_Cnt'] = df['Review'].apply(lambda text:len([word for word in str(text).split()

                                                                    if word.isdigit()]))

    df['Punctuation_Cnt'] = df['Review'].apply(lambda text:len([word for word in str(text).split()

                                                                    if word in string.punctuation]))

    df['Noun_Cnt'] = df['Review'].apply(lambda text: get_pos(str(text),'noun'))

    df['Pronoun_Cnt'] = df['Review'].apply(lambda text: get_pos(str(text),'pronoun'))

    df['Adjective_Cnt'] = df['Review'].apply(lambda text: get_pos(str(text),'adjective'))

    df['Adverb_Cnt'] = df['Review'].apply(lambda text: get_pos(str(text),'adverb'))

    df['Verb_Cnt'] = df['Review'].apply(lambda text: get_pos(str(text),'verb'))

    df['Polarity'] = df['Review'].apply(get_polarity)

    df['Subjectivity'] = df['Review'].apply(get_subjectivity)



    return df
if not (os.path.isfile('../input/end-to-end-text-processing-for-beginners/train_feat.csv' 

                       and '../input/end-to-end-text-processing-for-beginners/test_feat.csv')):

    train_feat = extract_text_features(train_df_nodup)

    test_feat = extract_text_features(test_df_nodup)

    train_feat.to_csv('train_feat.csv')

    test_feat.to_csv('test_feat.csv')

else:    

    train_feat = pd.read_csv('../input/end-to-end-text-processing-for-beginners/train_feat.csv',index_col=0)

    test_feat = pd.read_csv('../input/end-to-end-text-processing-for-beginners/test_feat.csv',index_col=0)
train_feat.head()
pol_1 = train_feat[train_feat['Label']==1]['Polarity'].values

pol_0 = train_feat[train_feat['Label']==0]['Polarity'].values



pol_data=[pol_1,pol_0]

group_labels=['Positive','Negative']

colors = ['#A56CC1', '#63F5EF']





fig = ff.create_distplot(pol_data, group_labels, colors=colors,

                         bin_size=.05, show_rug=False)



fig.layout.margin.update({'t':50, 'l':100})

fig['layout'].update(title='Polarity Distplot')

fig.layout.template = 'plotly_dark'



py.iplot(fig, filename='Polarity')
print('The average polarity of all Positive reviews:',round(np.mean(train_feat[train_feat['Label']==1]['Polarity'].values),2))

print('The average polarity of all Negative reviews:',round(np.mean(train_feat[train_feat['Label']==0]['Polarity'].values),2))
pol_1 = train_feat[train_feat['Label']==1]['Subjectivity'].values

pol_0 = train_feat[train_feat['Label']==0]['Subjectivity'].values



pol_data=[pol_1,pol_0]

group_labels=['Positive','Negative']

colors = ['#A56CC1', '#63F5EF']



fig = ff.create_distplot(pol_data, group_labels, colors=colors,

                         bin_size=.05, show_rug=False)



fig['layout'].update(title='Subjectivity Distplot')

fig.layout.template = 'plotly_dark'



py.iplot(fig, filename='Subjectivity Distplot')
pol_1 = train_feat[train_feat['Label']==1]['Word_Count'].values

pol_0 = train_feat[train_feat['Label']==0]['Word_Count'].values



pol_data=[pol_1,pol_0]

group_labels=['Positive','Negative']

colors = ['#A56CC1', '#63F5EF']



fig = ff.create_distplot(pol_data, group_labels, colors=colors,

                         bin_size=50, show_rug=False)



fig['layout'].update(title='Word Count Distplot')

fig.layout.template = 'plotly_dark'



py.iplot(fig, filename='Word Count Distplot')
print('The average word count of all Positive reviews:',round(np.mean(train_feat[train_feat['Label']==1]['Word_Count'].values),2))

print('The average word count of all Negative reviews:',round(np.mean(train_feat[train_feat['Label']==0]['Word_Count'].values),2))
# Preparing the data for applying to T-SNE

scaler = MinMaxScaler()

col_filter = [col for col in train_feat.columns if col not in ['Review', 'Label'] ]

train_viz = scaler.fit_transform(train_feat[col_filter][:10000])

y_label = train_feat['Label'][:10000].values
# Using TSNE for Dimentionality reduction for 15 Features to 2 dimentions

tsne2d = TSNE(n_components=2,init='random',random_state=101,n_iter=1000,verbose=2,angle=0.5).fit_transform(train_viz)
#2D Plot

trace1 = go.Scatter(

    x=tsne2d[:,0],

    y=tsne2d[:,1],

    mode='markers',

    marker=dict(

        sizemode='diameter',

        color = y_label,

        colorscale = 'Viridis',

        colorbar = dict(title = 'Label'),

        line=dict(color='rgb(255, 255, 255)'),

        opacity=0.75

    )

)



data=[trace1]

layout=dict(height=600, width=600, title='2d embedding with engineered features')

fig=dict(data=data, layout=layout)

py.iplot(fig, filename='2d embedding with engineered features')
# Using TSNE for Dimentionality reduction for 15 Features to 3 dimentions

tsne3d = TSNE(n_components=3,init='random',random_state=101,n_iter=1000,verbose=2,angle=0.5).fit_transform(train_viz)
#3D plot

trace1 = go.Scatter3d(

    x=tsne3d[:,0],

    y=tsne3d[:,1],

    z=tsne3d[:,2],

    mode='markers',

    marker=dict(

        sizemode='diameter',

        color = y_label,

        colorscale = 'Viridis',

        colorbar = dict(title = 'Label'),

        line=dict(color='rgb(255, 255, 255)'),

        opacity=0.75

    )

)



data=[trace1]

layout=dict(height=600, width=600, title='3d embedding with engineered features')

fig=dict(data=data, layout=layout)

py.iplot(fig, filename='3DBubble')
# Defining function to preprocess the text

lemmatizer = WordNetLemmatizer()

stop_words.append('br')



def preprocess(text):

    text = str(text.encode('utf-8')).lower()

    text = re.sub(r'[^A-Za-z1-9]+',' ',text)

    text_tokens = word_tokenize(text)

    

    text_process = []

    for word in text_tokens:

        if word not in stop_words and len(word)>1:

            text_process.append(str(lemmatizer.lemmatize(word)))

    text = ' '.join(text_process)       

            

    return text
if not (os.path.isfile('../input/end-to-end-text-processing-for-beginners/train_feat_clean.csv' 

                       and '../input/end-to-end-text-processing-for-beginners/test_feat_clean.csv')):

    train_feat['Review_Clean'] = train_feat['Review'].apply(preprocess)

    test_feat['Review_Clean'] = test_feat['Review'].apply(preprocess)

    train_feat.to_csv('train_feat_clean.csv')

    test_feat.to_csv('test_feat_clean.csv')

else:    

    train_feat = pd.read_csv('../input/end-to-end-text-processing-for-beginners/train_feat_clean.csv',index_col=0)

    test_feat = pd.read_csv('../input/end-to-end-text-processing-for-beginners/test_feat_clean.csv',index_col=0)
train_feat.head()
print('Review before Text preprocessing:\n',train_feat['Review'][1])

print('\nReview after Text preprocessing:\n',train_feat['Review_Clean'][1])
# Preparing the data in the format accepted by WordCloud

positive_reviews = ' '.join(train_feat[train_feat['Label']==1]['Review_Clean'])

negative_reviews = ' '.join(train_feat[train_feat['Label']==0]['Review_Clean'])
# Positive reviews word cloud

start = datetime.now()

wordcloud = WordCloud(    background_color='black',

                          width=1600,

                          height=800,

                    ).generate(positive_reviews)



fig = plt.figure(figsize=(30,20))

plt.imshow(wordcloud)

plt.axis('off')

plt.tight_layout(pad=0)

print('Word Cloud generated with Positive reviews.')

fig.savefig("positive_wc.png")

plt.show()

print("Time taken to run this cell :", datetime.now() - start)
# Negative reviews word cloud

start = datetime.now()

wordcloud = WordCloud(    background_color='black',

                          width=1600,

                          height=800,

                    ).generate(negative_reviews)



fig = plt.figure(figsize=(30,20))

plt.imshow(wordcloud)

plt.axis('off')

plt.tight_layout(pad=0)

print('Word Cloud generated with Negative reviews.')

fig.savefig("negative_wc.png")

plt.show()

print("Time taken to run this cell :", datetime.now() - start)
# Extracting 15 Engineered features as a seperate Dataframe

feat_15_cols = [col for col in train_feat.columns if col not in ['Review','Label','Review_Clean']]

train_feat_15 = train_feat[feat_15_cols]

test_feat_15 = test_feat[feat_15_cols]



train_feat_15.reset_index(drop=True, inplace=True)

test_feat_15.reset_index(drop=True, inplace=True)
train_feat_15.head()
# Applying BOW text featurization on the feature 'Review_Clean'

bow_vect = CountVectorizer(min_df=5, analyzer='word', dtype=np.float32)

                             

bow_train = bow_vect.fit_transform(train_feat['Review_Clean'].values)

bow_test = bow_vect.transform(test_feat['Review_Clean'].values)



print("The shape of train BOW vectorizer ",bow_train.get_shape())

print("the number of unique words ", bow_train.get_shape()[1])
# Combining Review clean text featurization with 15 engineered features so as to get final dataset.

final_bow_train = sparse.hstack((bow_train, train_feat_15)).tocsr()

final_bow_test = sparse.hstack((bow_test, test_feat_15)).tocsr()



print("The shape of final train BOW dataset:",final_bow_train.get_shape())

print("The shape of final test BOW dataset:",final_bow_test.get_shape())
bow = pd.DataFrame(

    {

      'BOW': list(bow_train.sum(axis=0).A1)

                    

    }, index=bow_vect.get_feature_names())



print('The important features based on BOW values:')

bow.sort_values('BOW',ascending=False).head(10)
bow_vect_ngrams = CountVectorizer(min_df=5, analyzer='word', ngram_range=(1,3), dtype=np.float32)

                             

bow_train_ngrams = bow_vect_ngrams.fit_transform(train_feat['Review_Clean'].values)

bow_test_ngrams = bow_vect_ngrams.transform(test_feat['Review_Clean'].values)



print("The shape of train BOW vectorizer with ngrams ",bow_train_ngrams.get_shape())

print("the number of unique words ", bow_train_ngrams.get_shape()[1])
# Combining Review clean text featurization with 15 engineered features so as to get final dataset.

final_bow_train_ngrams = sparse.hstack((bow_train_ngrams, train_feat_15)).tocsr()

final_bow_test_ngrams = sparse.hstack((bow_test_ngrams, test_feat_15)).tocsr()



print("The shape of final train BOW with ngrams dataset:",final_bow_train_ngrams.get_shape())

print("The shape of final test BOW with ngrams dataset:",final_bow_test_ngrams.get_shape())
tfidf_vect = TfidfVectorizer(min_df=5, analyzer='word',dtype=np.float32) 

tfidf_train = tfidf_vect.fit_transform(train_feat['Review_Clean'].values)

tfidf_test = tfidf_vect.transform(test_feat['Review_Clean'].values)



print("The shape of train Tf-Idf vectorizer ",tfidf_train.get_shape())

print("the number of unique words ", tfidf_train.get_shape()[1])
# Combining Review clean text featurization with 15 engineered features so as to get final dataset.

final_tfidf_train = sparse.hstack((tfidf_train, train_feat_15)).tocsr()

final_tfidf_test = sparse.hstack((tfidf_test, test_feat_15)).tocsr()



print("The shape of final train Tf-Idf dataset:",final_tfidf_train.get_shape())

print("The shape of final test Tf-Idf dataset:",final_tfidf_test.get_shape())
tfidf = pd.DataFrame(

    {

      'Tf-Idf': list(tfidf_train.sum(axis=0).A1)

                    

    }, index=tfidf_vect.get_feature_names())



print('The important features based on TF-IDF values:')

tfidf.sort_values('Tf-Idf',ascending=False).head(10)
tfidf_vect_ngrams = TfidfVectorizer(min_df=5, analyzer='word',ngram_range=(1,3),dtype=np.float32) 

tfidf_train_ngrams = tfidf_vect_ngrams.fit_transform(train_feat['Review_Clean'].values)

tfidf_test_ngrams = tfidf_vect_ngrams.transform(test_feat['Review_Clean'].values)



print("The shape of train Tf-Idf vectorizer with ngrams ",tfidf_train_ngrams.get_shape())

print("the number of unique words ", tfidf_train_ngrams.get_shape()[1])
# Combining Review clean text featurization with 15 engineered features so as to get final dataset.

final_tfidf_train_ngrams = sparse.hstack((tfidf_train_ngrams, train_feat_15)).tocsr()

final_tfidf_test_ngrams = sparse.hstack((tfidf_test_ngrams, test_feat_15)).tocsr()



print("The shape of final train Tf-Idf with ngrams dataset:",final_tfidf_train_ngrams.get_shape())

print("The shape of final test Tf-Idf with ngrams dataset:",final_tfidf_test_ngrams.get_shape())
# Preparing the data in the format required for Word2Vec

list_of_sentence_train=[]

for sentence in train_feat['Review_Clean']:

    list_of_sentence_train.append(sentence.split())

    

list_of_sentence_test=[]

for sentence in test_feat['Review_Clean']:

    list_of_sentence_test.append(sentence.split())
# Training the Word2Vec model on the own corpus

w2v_model=Word2Vec(list_of_sentence_train, size=50, min_count=5, workers=4)

w2v_words = list(w2v_model.wv.vocab)



with open('w2v_model.pkl', 'wb') as f:

    pickle.dump(w2v_model, f)
# Word2Vec representation for the word 'pleasure'

w2v_model.wv['pleasure']
print('Few words trained by Word2Vec:',w2v_words[:10])

print('The no of words trained by Word2Vec:',len(w2v_words))
# compute average word2vec for each review.

sent_vectors_train = []; 

for sent in tqdm(list_of_sentence_train): 

    sent_vec = np.zeros(50) 

    cnt_words =0; 

    for word in sent: 

        if word in w2v_words:

            vec = w2v_model.wv[word]

            sent_vec += vec

            cnt_words += 1

    if cnt_words != 0:

        sent_vec /= cnt_words

    sent_vectors_train.append(sent_vec)

print(len(sent_vectors_train))

print(len(sent_vectors_train[0]))
sent_vectors_test = [];

for sent in tqdm(list_of_sentence_test): 

    sent_vec = np.zeros(50)

    cnt_words =0; 

    for word in sent:

        if word in w2v_words:

            vec = w2v_model.wv[word]

            sent_vec += vec

            cnt_words += 1

    if cnt_words != 0:

        sent_vec /= cnt_words

    sent_vectors_test.append(sent_vec)

print(len(sent_vectors_test))

print(len(sent_vectors_test[0]))
# Combining Review clean text featurization with 15 engineered features so as to get final dataset.

avg_w2v_train = pd.DataFrame(sent_vectors_train)

avg_w2v_test = pd.DataFrame(sent_vectors_test)



final_avg_w2v_train = pd.concat([avg_w2v_train,train_feat_15], axis=1)

final_avg_w2v_test = pd.concat([avg_w2v_test,test_feat_15], axis=1)



print('The shape of final train avg word2vec dataset:',final_avg_w2v_train.shape)

print('The shape of final test avg word2vec dataset:',final_avg_w2v_test.shape)
final_avg_w2v_train.head()
model = TfidfVectorizer(min_df=5, analyzer='word',dtype=np.float32) 

tfidf_matrix = model.fit_transform(train_feat['Review_Clean'].values)

dictionary = dict(zip(model.get_feature_names(), list(model.idf_)))
# TF-IDF weighted Word2Vec

tfidf_feat = model.get_feature_names()

tfidf_sent_vectors_train = [];

row=0;

for sent in tqdm(list_of_sentence_train):

    sent_vec = np.zeros(50)

    weight_sum =0; 

    for word in sent: 

        if word in w2v_words and word in tfidf_feat:

            vec = w2v_model.wv[word]

            tf_idf = dictionary[word]*(sent.count(word)/len(sent))

            sent_vec += (vec * tf_idf)

            weight_sum += tf_idf

    if weight_sum != 0:

        sent_vec /= weight_sum

    tfidf_sent_vectors_train.append(sent_vec)

    row += 1
tfidf_feat = model.get_feature_names()

tfidf_sent_vectors_test = []; 

row=0;

for sent in tqdm(list_of_sentence_test):

    sent_vec = np.zeros(50) 

    weight_sum =0; 

    for word in sent:

        if word in w2v_words and word in tfidf_feat:

            vec = w2v_model.wv[word]

            tf_idf = dictionary[word]*(sent.count(word)/len(sent))

            sent_vec += (vec * tf_idf)

            weight_sum += tf_idf

    if weight_sum != 0:

        sent_vec /= weight_sum

    tfidf_sent_vectors_test.append(sent_vec)

    row += 1
# Combining Review clean text featurization with 15 engineered features so as to get final dataset.

tfidf_w2v_train = pd.DataFrame(tfidf_sent_vectors_train)

tfidf_w2v_test = pd.DataFrame(tfidf_sent_vectors_test)



final_tfidf_w2v_train = pd.concat([tfidf_w2v_train,train_feat_15], axis=1)

final_tfidf_w2v_test = pd.concat([tfidf_w2v_test,test_feat_15], axis=1)



print('The shape of final train avg word2vec dataset:',final_tfidf_w2v_train.shape)

print('The shape of final test avg word2vec dataset:',final_tfidf_w2v_test.shape)