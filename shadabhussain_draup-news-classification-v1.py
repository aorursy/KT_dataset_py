# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
! pip install transformers==3.2.0 --user

! pip install tensorflow==2.2 --user
import warnings

warnings.filterwarnings("ignore")

import numpy as np

import pandas as pd

import string as s

import matplotlib.pyplot as plt

import seaborn as sns

import transformers

from wordcloud import WordCloud

import re

import nltk

from nltk.corpus import stopwords

%matplotlib inline

from sklearn.feature_extraction.text  import TfidfVectorizer

from sklearn.naive_bayes import MultinomialNB, GaussianNB

from sklearn.metrics  import f1_score, accuracy_score, confusion_matrix, recall_score, precision_score

from sklearn.tree import DecisionTreeClassifier

from sklearn.linear_model import SGDClassifier

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import LabelEncoder

import sklearn

from lightgbm import LGBMClassifier

import tensorflow as tf

from keras.utils import to_categorical
#bert large uncased pretrained tokenizer

tokenizer = transformers.BertTokenizer.from_pretrained('bert-large-uncased')
# Loading category list

category_list_df = pd.read_csv("/kaggle/input/draup-news-classification/categories.csv")

category_list_df
# Loading category mapping list

category_mapping_df = pd.read_excel("/kaggle/input/draup-news-classification/category_mapping.xlsx")

category_mapping_df.head()
# Loading news data

news_details_df = pd.read_excel("/kaggle/input/draup-news-classification/news_details.xlsx")

news_details_df = news_details_df.drop_duplicates( keep='first')

news_details_df.head()
# Mapping news_id with category name

category_df = pd.merge(category_mapping_df, category_list_df, left_on='category_id', right_on="id").drop('id', axis=1)

category_df = category_df.drop_duplicates( keep='first')

category_df.head()
# Mapping news_id with category_id

final_df = news_details_df.merge(category_df, on='news_id')

final_df.head()
# Printing Sample Snippet

final_df.iloc[3][1]
# Printing Sample Title

final_df.iloc[3][2]
# Printing Sample News Description

final_df.iloc[3][3]
# Checking for missing snippets/titles/descriptions

final_df.info()
# Replacing NAs with empty string

final_df = final_df.fillna('')

final_df.head()
# Converting each of snippet, title, and news description into lower case.

final_df['snippet'] = final_df['snippet'].apply(lambda snippet: str(snippet).lower())

final_df['title'] = final_df['title'].apply(lambda title: str(title).lower())

final_df['news_description'] = final_df['news_description'].apply(lambda news_description: str(news_description).lower())
#calculating the length of snippets, titles and descriptions

final_df['snippet_len'] = final_df['snippet'].apply(lambda x: len(str(x).split()))

final_df['title_len'] = final_df['title'].apply(lambda x: len(str(x).split()))

final_df['news_description_len'] = final_df['news_description'].apply(lambda x: len(str(x).split()))
final_df.describe()
def fx(x):

    if len(x['news_description'])==0:

        return x['title'] + " " + x['snippet']

    else:

        return x['title'] + " " + x['news_description']   

final_df['text']=final_df.apply(lambda x : fx(x),axis=1)
final_df.head()
def tokenization(text):

    lst=text.split()

    return lst
def remove_new_lines(lst):

    new_lst=[]

    for i in lst:

        i=i.replace(r'\n', ' ').replace(r'\r', ' ').replace(r'\u', ' ')

        new_lst.append(i.strip())

    return new_lst
def remove_punctuations(lst):

    new_lst=[]

    for i in lst:

        for  j in s.punctuation:

            i=i.replace(j,' ')

        new_lst.append(i.strip())

    return new_lst
def remove_numbers(lst):

    nodig_lst=[]

    new_lst=[]

    for i in  lst:

        for j in  s.digits:

            i=i.replace(j,' ')

        nodig_lst.append(i.strip())

    for i in  nodig_lst:

        if  i!='':

            new_lst.append(i.strip())

    return new_lst
def remove_stopwords(lst):

    stop=stopwords.words('english')

    new_lst=[]

    for i in lst:

        if i not in stop:

            new_lst.append(i.strip())

    return new_lst
lemmatizer=nltk.stem.WordNetLemmatizer()

def lemmatization(lst):

    new_lst=[]

    for i in lst:

        i=lemmatizer.lemmatize(i)

        new_lst.append(i.strip())

    return new_lst
def remove_urls(text):

    return re.sub(r'http\S+', ' ', text)
def split_words(text):

    return ' '.join(text).split()
def remove_single_chars(lst):

    new_lst=[]

    for i in lst:

        if len(i)>1:

            new_lst.append(i.strip())

    return new_lst
# Cleaning Text

def denoise_text(text):

    text = remove_urls(text)

    text = tokenization(text)

    text = remove_new_lines(text)

    text = remove_punctuations(text)

    text = remove_numbers(text)

    text = remove_stopwords(text)

    text = split_words(text)

    text = remove_single_chars(text)

    text = lemmatization(text)

    return text



final_df['text'] = final_df['text'].apply(lambda x: denoise_text(x))
ax = final_df.groupby('category')['news_id'].count().plot(kind='barh', figsize=(10,6), fontsize=13)

ax.set_alpha(0.8)

ax.set_title("Count of Each News Category", fontsize=18)

ax.set_ylabel("Category", fontsize=15)

ax.set_xlabel("Count", fontsize=15)

for p in ax.patches:

    ax.annotate(str(p.get_width()), (p.get_x() + p.get_width(), p.get_y()), xytext=(5, 10), textcoords='offset points')

plt.show()
layoff = final_df[final_df.category=='Layoff']['text'].apply(lambda x:' '.join(x))

plt.figure(figsize = (15,20))

wordcloud = WordCloud(min_font_size = 3,  max_words = 2500 , width = 1200 , height = 800).generate(" ".join(layoff))

plt.imshow(wordcloud,interpolation = 'bilinear')

plt.show()
mergers_acquisitions = final_df[final_df.category=='Mergers and Acquisitions']['text'].apply(lambda x: ' '.join(x))

plt.figure(figsize = (15,20))

wordcloud = WordCloud(min_font_size = 3,  max_words = 2500 , width = 1200 , height = 800).generate(" ".join(mergers_acquisitions))

plt.imshow(wordcloud,interpolation = 'bilinear')

plt.show()
mass_hiring = final_df[final_df.category=='Mass Hiring']['text'].apply(lambda x: ' '.join(x))

plt.figure(figsize = (15,20))

wordcloud = WordCloud(min_font_size = 3,  max_words = 2500 , width = 1200 , height = 800).generate(" ".join(mass_hiring))

plt.imshow(wordcloud,interpolation = 'bilinear')

plt.show()
executive_movement = final_df[final_df.category=='Executive Movement']['text'].apply(lambda x: ' '.join(x))

plt.figure(figsize = (15,20))

wordcloud = WordCloud(min_font_size = 3,  max_words = 2500 , width = 1200 , height = 800).generate(" ".join(executive_movement))

plt.imshow(wordcloud,interpolation = 'bilinear')

plt.show()
setup_expansion = final_df[final_df.category=='Centre Setup and Expansion']['text'].apply(lambda x: ' '.join(x))

plt.figure(figsize = (15,20))

wordcloud = WordCloud(min_font_size = 3,  max_words = 2500 , width = 1200 , height = 800).generate(" ".join(setup_expansion))

plt.imshow(wordcloud,interpolation = 'bilinear')

plt.show()
deals = final_df[final_df.category=='Deals']['text'].apply(lambda x: ' '.join(x))

plt.figure(figsize = (15,20))

wordcloud = WordCloud(min_font_size = 3,  max_words = 2500 , width = 1200 , height = 800).generate(" ".join(deals))

plt.imshow(wordcloud,interpolation = 'bilinear')

plt.show()
partnerships = final_df[final_df.category=='Partnerships']['text'].apply(lambda x: ' '.join(x))

plt.figure(figsize = (15,20))

wordcloud = WordCloud(min_font_size = 3,  max_words = 2500 , width = 1200 , height = 800).generate(" ".join(partnerships))

plt.imshow(wordcloud,interpolation = 'bilinear')

plt.show()
final_df['text_char_len'] = final_df['text'].apply(lambda x: len(' '.join(x)))
fig, ax=plt.subplots(4,2,figsize=(16,16))

count = 1

for i in ax:

    for j in i:

        if count==8:

            j.axis('off')

            break

        text_len=final_df[final_df['category_id']==count]['text_char_len'] 

        j.hist(text_len, bins=20)

        j.set_title(final_df[final_df['category_id']==count]['category'].values[0])

        count+=1
final_df['text_word_len'] = final_df['text'].str.len() 
fig, ax=plt.subplots(4,2,figsize=(16,16))

count = 1

for i in ax:

    for j in i:

        if count==8:

            j.axis('off')

            break

        text_len=final_df[final_df['category_id']==count]['text_word_len'] 

        j.hist(text_len, bins=20)

        j.set_title(final_df[final_df['category_id']==count]['category'].values[0])

        count+=1
final_df.text_word_len
# KDE plot for words and characters length

# fig, ax=plt.subplots(1,2,figsize=(16,6))

# sns.kdeplot(data=final_df, x='text_word_len', hue='category', ax=ax[0])

# sns.kdeplot(data=final_df, x='text_char_len', hue='category', ax=ax[1], multiple='stack')

# plt.show()
final_df['text_avg_word_len'] = final_df['text'].apply(lambda x : np.mean([len(i) for i in x]))
fig, ax=plt.subplots(4,2,figsize=(16,16), constrained_layout=True)

count = 1

for i in ax:

    for j in i:

        if count==8:

            j.axis('off')

            break

        word=final_df[final_df['category_id']==count]['text_avg_word_len']

        sns.distplot(word, ax=j)

        j.set_title(final_df[final_df['category_id']==count]['category'].values[0])

        count+=1

        j.set_xlabel('')
# Word Corpus

def get_corpus(text):

    words = []

    for i in text:

        for j in i:

            words.append(j.strip())

    return words

corpus = get_corpus(final_df.text)

corpus[:5]
# Most common words

from collections import Counter

counter = Counter(corpus)

most_common = counter.most_common(10)

most_common = dict(most_common)

most_common
from sklearn.feature_extraction.text import CountVectorizer

def get_top_text_ngrams(corpus, n, g):

    vec = CountVectorizer(ngram_range=(g, g)).fit(corpus)

    bag_of_words = vec.transform(corpus)

    sum_words = bag_of_words.sum(axis=0) 

    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]

    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)

    return words_freq[:n]
plt.figure(figsize = (16,9))

most_common_uni = get_top_text_ngrams(final_df.text.apply(lambda x: ' '.join(x)),10,1)

most_common_uni = dict(most_common_uni)

sns.barplot(x=list(most_common_uni.values()), y=list(most_common_uni.keys()))

plt.show()
plt.figure(figsize = (16,9))

most_common_bi = get_top_text_ngrams(final_df.text.apply(lambda x: ' '.join(x)),10,2)

most_common_bi = dict(most_common_bi)

sns.barplot(x=list(most_common_bi.values()),y=list(most_common_bi.keys()))

plt.show()
plt.figure(figsize = (16,9))

most_common_tri = get_top_text_ngrams(final_df.text.apply(lambda x: ' '.join(x)),10,3)

most_common_tri = dict(most_common_tri)

sns.barplot(x=list(most_common_tri.values()),y=list(most_common_tri.keys()))

plt.show()
#label encoding the categories. After this each category would be mapped to an integer.

encoder = LabelEncoder()

final_df['categoryEncoded'] = encoder.fit_transform(final_df['category'])
X_train, X_test, y_train, y_test = train_test_split(final_df['text'], final_df['categoryEncoded'], random_state = 43, test_size = 0.2)
def regular_encode(texts, tokenizer, maxlen=512):

    enc_di = tokenizer.batch_encode_plus(

        texts, 

        return_attention_mask=False, 

        return_token_type_ids=False,

        pad_to_max_length=True,

        max_length=maxlen

    )

    

    return np.array(enc_di['input_ids'])
# Tokenizing the news descriptions and converting the categories into one hot vectors using tf.keras.utils.to_categorical

Xtrain_encoded = regular_encode(X_train.apply(lambda x: ' '.join(x)).astype('str'), tokenizer, maxlen=256)

Xtest_encoded = regular_encode(X_test.apply(lambda x: ' '.join(x)).astype('str'), tokenizer, maxlen=256)

ytrain_encoded = to_categorical(y_train, dtype='int32')

ytest_encoded = to_categorical(y_test, dtype='int32')
# Detect hardware, return appropriate distribution strategy

try:

    # TPU detection. No parameters necessary if TPU_NAME environment variable is

    # set: this is always the case on Kaggle.

    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()

    print('Running on TPU ', tpu.master())

except ValueError:

    tpu = None



if tpu:

    tf.config.experimental_connect_to_cluster(tpu)

    tf.tpu.experimental.initialize_tpu_system(tpu)

    strategy = tf.distribute.experimental.TPUStrategy(tpu)

else:

    # Default distribution strategy in Tensorflow. Works on CPU and single GPU.

    strategy = tf.distribute.get_strategy()



print("REPLICAS: ", strategy.num_replicas_in_sync)
train_x=X_train.apply(lambda x: ''.join(i+' ' for i in x))

test_x=X_test.apply(lambda x: ''.join(i+' '  for i in x))
tfidf=TfidfVectorizer(max_features=10000,min_df=6)

train_1=tfidf.fit_transform(train_x)

test_1=tfidf.transform(test_x)

print("No. of features extracted:", len(tfidf.get_feature_names()))

print(tfidf.get_feature_names()[:20])



train_arr=train_1.toarray()

test_arr=test_1.toarray()
NB_MN=MultinomialNB()

NB_MN.fit(train_arr,y_train)

pred=NB_MN.predict(test_arr)



print("first 20 actual labels")

print(y_test.tolist()[:20])

print("first 20 predicted labels")

print(pred.tolist()[:20])
def eval_model(y,y_pred):

    print("Recall score of the model:", round(recall_score(y_test, pred, average='weighted'), 3))

    print("Precision score of the model:", round(precision_score(y_test, pred, average='weighted'), 3))

    print("F1 score of the model:", round(f1_score(y,y_pred,average='micro'), 3))

    print("Accuracy of the model:", round(accuracy_score(y,y_pred),3))

    print("Accuracy of the model in percentage:", round(accuracy_score(y,y_pred)*100,3),"%")
def confusion_mat(color):

    cof=confusion_matrix(y_test, pred)

    cof=pd.DataFrame(cof, index=[i for i in range(1,8)], columns=[i for i in range(1,8)])

    sns.set(font_scale=1.5)

    plt.figure(figsize=(8,8));



    sns.heatmap(cof, cmap=color,linewidths=1, annot=True,square=True, fmt='d', cbar=False,xticklabels=list(encoder.classes_),yticklabels=list(encoder.classes_));

    plt.xlabel("Predicted Classes");

    plt.ylabel("Actual Classes");

    
eval_model(y_test,pred)

    

a=round(accuracy_score(y_test,pred)*100,3)
confusion_mat('YlGnBu')
DT=DecisionTreeClassifier()

DT.fit(train_arr,y_train)

pred=DT.predict(test_arr)



print("first 20 actual labels")

print(y_test.tolist()[:20])

print("first 20 predicted labels")

print(pred.tolist()[:20])
eval_model(y_test,pred)

b=round(accuracy_score(y_test,pred)*100,3)
confusion_mat('Blues')
NB=GaussianNB()

NB.fit(train_arr,y_train)

pred=NB.predict(test_arr)
eval_model(y_test,pred)

    

c=round(accuracy_score(y_test,pred)*100,3)
confusion_mat('Greens')
SGD=SGDClassifier()

SGD.fit(train_arr,y_train)

pred=SGD.predict(test_arr)
eval_model(y_test,pred)

    

d=round(accuracy_score(y_test,pred)*100,3)
confusion_mat('Reds')
lgbm=LGBMClassifier()

lgbm.fit(train_arr,y_train)

pred=lgbm.predict(test_arr)
eval_model(y_test,pred)



e=round(accuracy_score(y_test,pred)*100,3)
confusion_mat('YlOrBr')
def build_model(transformer, loss='categorical_crossentropy', max_len=512):

    input_word_ids = tf.keras.layers.Input(shape=(max_len,), dtype=tf.int32, name="input_word_ids")

    sequence_output = transformer(input_word_ids)[0]

    cls_token = sequence_output[:, 0, :]

    #adding dropout layer

    x = tf.keras.layers.Dropout(0.3)(cls_token)

    #using a dense layer of 7 neurons as the number of unique categories is 7. 

    out = tf.keras.layers.Dense(7, activation='softmax')(x)

    model = tf.keras.Model(inputs=input_word_ids, outputs=out)

    #using categorical crossentropy as the loss as it is a multi-class classification problem

    model.compile(tf.keras.optimizers.Adam(lr=3e-5), loss=loss, metrics=['accuracy'])

    return model
#building the model on tpu

with strategy.scope():

    transformer_layer = transformers.TFAutoModel.from_pretrained('bert-large-uncased')

    model = build_model(transformer_layer, max_len=256)

model.summary()
#creating the training and testing dataset.

BATCH_SIZE = 32*strategy.num_replicas_in_sync

AUTO = tf.data.experimental.AUTOTUNE 

train_dataset = (tf.data.Dataset.from_tensor_slices((Xtrain_encoded, ytrain_encoded)).repeat().shuffle(2048).batch(BATCH_SIZE)

                 .prefetch(AUTO))

test_dataset = (tf.data.Dataset.from_tensor_slices(Xtest_encoded).batch(BATCH_SIZE))
#training for 10 epochs

n_steps = Xtrain_encoded.shape[0] // BATCH_SIZE

train_history = model.fit(

    train_dataset,

    steps_per_epoch=n_steps,

    epochs=10

)
#making predictions

preds = model.predict(test_dataset,verbose = 1)

#converting the one hot vector output to a linear numpy array.

pred = np.argmax(preds, axis = 1)
#extracting the classes from the label encoder

encoded_classes = encoder.classes_

#mapping the encoded output to actual categories

predicted_category = [encoded_classes[x] for x in pred]

true_category = [encoded_classes[x] for x in y_test]
result_df = pd.DataFrame({'description':X_test,'true_category':true_category, 'predicted_category':predicted_category})

result_df.head()
eval_model(y_test,pred)



f=round(accuracy_score(y_test,pred)*100,3)
confusion_mat('YlOrBr')
sns.set()

fig = plt.figure(figsize=(10,6))

ax = fig.add_axes([0,0,1,1])

Models = ['MultinomialNB', 'DecisionTree', 'GaussianNB', 'SGD','LGBM', 'BERT']

Accuracy=[a,b,c,d,e,f]

ax.bar(Models,Accuracy);

for i in ax.patches:

    ax.text(i.get_x()+.1, i.get_height()-5.5, str(round(i.get_height(),2))+'%', fontsize=13, color='white')

plt.title('Comparison of Different Classification Models');

plt.ylabel('Accuracy');

plt.xlabel('Classification Models');



plt.show();