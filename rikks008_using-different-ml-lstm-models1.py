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
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import classification_report,confusion_matrix
%matplotlib inline
from numpy import array
from keras.preprocessing.text import one_hot
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers.core import Activation, Dropout, Dense
from keras.layers import Flatten, LSTM,Bidirectional,SpatialDropout1D
from keras.layers import GlobalMaxPooling1D
from keras.models import Model
from keras.layers.embeddings import Embedding
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras.layers import Input
from keras.layers.merge import Concatenate
from keras.layers import Embedding,Flatten,Dense,Dropout
import keras
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from numpy import asarray,zeros
from wordcloud import WordCloud

import warnings
warnings.filterwarnings('ignore')

data = pd.read_json("/kaggle/input/news-category-dataset/News_Category_Dataset_v2.json",lines=True)
data.head()
data.info()
data.isna().apply(pd.value_counts) #missing value check
data.category.nunique() # number of unique categories
data.category.unique()
data.category = data.category.map(lambda x: "WORLDPOST" if x == "THE WORLDPOST" else x)
data.category.nunique() # number of unique categories
# Plotting top 10 category
group_count = data['category'].value_counts()
sns.barplot(group_count.index[:10], group_count.values[:10], alpha=0.8)
plt.title('Top 10 category ')
plt.ylabel('Counts', fontsize=12)
plt.xlabel('Category groups', fontsize=12,)
plt.xticks(rotation=45)
plt.show()
# Plotting top 10 authors
group_count = data['authors'].value_counts()
sns.barplot(group_count.index[:10], group_count.values[:10], alpha=0.8)
plt.title('Top 10 authors')
plt.ylabel('Count', fontsize=12)
plt.xlabel('Author types', fontsize=12,)
plt.xticks(rotation='vertical')
plt.show()
#Plotting word cloud
from wordcloud import WordCloud
desc = " ".join(str(des) for des in data['headline'])

wc_desc = WordCloud(background_color='white', max_words=200, width=400, height=400,random_state=10).generate(desc)
plt.figure(figsize=(10,10))
plt.imshow(wc_desc)
plt.title("Word cloud for Headline column")
#Plotting word cloud for short description
from wordcloud import WordCloud
desc = " ".join(str(des) for des in data['short_description'])

wc_desc = WordCloud(background_color='white', max_words=200, width=400, height=400,random_state=10).generate(desc)
plt.figure(figsize=(10,10))
plt.imshow(wc_desc)
plt.title("Word cloud for short description column")
import unicodedata
import nltk
import spacy
from nltk.tokenize.toktok import ToktokTokenizer
stopword_list = nltk.corpus.stopwords.words('english')
tokenizer = ToktokTokenizer()
nlp = spacy.load('en_core_web_sm', parse = False, tag=False, entity=False)
#from contractions import CONTRACTION_MAP
import nltk
nltk.download('words')
words = set(nltk.corpus.words.words())
stopword_list = nltk.corpus.stopwords.words('english')

import re
# Remove any emails 
def remove_emails(text):
    text = re.sub(r'\b[^\s]+@[^\s]+[.][^\s]+\b', ' ', text)
    return text

def remove_hyperlink(text):
    text=re.sub(r'(http|https)://[^\s]*',' ',text)
    return text

# Removing Digits
def remove_digits(text):
    #text= re.sub(r"\b\d+\b", "", text)
    text= re.sub(r"(\s\d+)", " ", text)
    return text
    

# Removing Special Characters
def remove_special_characters(text):
    text = re.sub('[^a-zA-Z\s]', ' ', text)
    return text


# removing accented charactors
def remove_accented_chars(text):
    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')
    return text

 # Removing Stopwords
def remove_stopwords(text,is_lower_case):
    tokens = tokenizer.tokenize(text)
    tokens = [token.strip() for token in tokens]

    filtered_tokens = [token for token in tokens if token.lower() not in stopword_list]
    filtered_text = ' '.join(filtered_tokens)   
    return filtered_text

# Lemmetization
def lemmatize_text(text):
    text = nlp(text)
    text = ' '.join([word.lemma_ if word.lemma_ != '-PRON-' else word.text for word in text])
    return text
# Combine all the functions and creating a preprocessing pipeline
# # Text preprocessing
def text_preprocessing(corpus,isRemoveEmail,isRemoveDigits,isRemoveHyperLink, 
                     isRemoveSpecialCharac,isRemoveAccentChar,
                       text_lower_case,text_lemmatization, stopword_removal):
    
    normalized_corpus = []
    
    for doc in corpus:
        
        if text_lower_case:
            doc = doc.lower()
        
        if isRemoveEmail:
            doc = remove_emails(doc)
        
        if isRemoveHyperLink:
            doc=remove_hyperlink(doc)
             
        if isRemoveAccentChar:
            doc = remove_accented_chars(doc)
            
        if isRemoveDigits:
            doc = remove_digits(doc)
        
        # remove extra newlines
        doc = re.sub(r'[\r|\n|\r\n]+', ' ',doc)
        # insert spaces between special characters to isolate them    
        special_char_pattern = re.compile(r'([{.(-)!}])')
        doc = special_char_pattern.sub(" \\1 ", doc)
        
        if text_lemmatization:
            doc = lemmatize_text(doc)
        
        if isRemoveSpecialCharac:
            doc = remove_special_characters(doc)
        # remove extra whitespace
        doc = re.sub(' +', ' ', doc)
        
        if stopword_removal:
            doc = remove_stopwords(doc,is_lower_case=text_lower_case)
                
        normalized_corpus.append(doc)
        
    return normalized_corpus

EMAIL_FLAG=True
DIGIT_FLAG=True
HYPER_LINK_FLAG=True
ALL_SPEC_CHAR_FLAG=True
ACCENT_CHAR_FLAG=True
LOWER_CASE_FLAG=True
LEMMETIZE_FLAG=False
STOPWORD_FLAG=True

clean_headline= text_preprocessing(data['headline'],EMAIL_FLAG,DIGIT_FLAG,HYPER_LINK_FLAG,
                   ALL_SPEC_CHAR_FLAG,ACCENT_CHAR_FLAG,
                  LOWER_CASE_FLAG,LEMMETIZE_FLAG,STOPWORD_FLAG)
clean_short_Desc = text_preprocessing(data['short_description'],EMAIL_FLAG,DIGIT_FLAG,HYPER_LINK_FLAG,
                   ALL_SPEC_CHAR_FLAG,ACCENT_CHAR_FLAG,
                  LOWER_CASE_FLAG,LEMMETIZE_FLAG,STOPWORD_FLAG)
data['clean_headline']=clean_headline
data['clean_short_Desc'] = clean_short_Desc
#Description
from wordcloud import WordCloud
import matplotlib.pyplot as plt
plt.figure(figsize=(15,15))

for index, i in enumerate(data['category'].unique()):
  s = str(i)
  i = str(data[data['category']==s].Merged_data)
  i = WordCloud(background_color='white', max_words=200, width=400, height=400,random_state=10).generate(i)
  c = index+1
  plt.subplot(10,5,c)
  plt.imshow(i)
  plt.title(s)

data['MergedColumn'] = data[data.columns[6:8]].apply(
    lambda x: ' '.join(x.astype(str)),
    axis=1
)
pd.set_option('display.max_colwidth', -1)
data['MergedColumn'][0:10]
final_df = data.copy()
del data
final_df.drop(columns=['headline', 'authors', 'link', 'short_description', 'date',
                   'clean_headline', 'clean_short_Desc'],axis=1,inplace=True)
final_df.columns
final_df.to_csv('final_data.csv',index=False)
#Plotting word cloud
from wordcloud import WordCloud
desc = " ".join(str(des) for des in final_df['MergedColumn'])

wc_desc = WordCloud(background_color='white', max_words=200, width=400, height=400,random_state=10).generate(desc)
plt.figure(figsize=(10,10))
plt.imshow(wc_desc)
plt.title("Word cloud for final data after cleaning")
### Count Vectorizer
cv = CountVectorizer(max_features=2500)
X = cv.fit_transform(final_df['MergedColumn']).toarray()

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split( X,final_df['category'], test_size=0.2, random_state=42)

print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)
del final_df


from sklearn.naive_bayes import MultinomialNB
nb = MultinomialNB()
nb.fit(X_train,y_train)

predictions = nb.predict(X_test)

print('Training accuracy',nb.score(X_train,y_train))

from sklearn.metrics import accuracy_score
print("Testing accuracy " ,accuracy_score(predictions,y_test))
confusion_matrix(y_test,predictions)
classification_report(y_test,predictions)
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(X_train,y_train)
print("Training accuracy",lr.score(X_train,y_train))

predictions = lr.predict(X_test)

from sklearn.metrics import accuracy_score
print("Testing accuracy",accuracy_score(predictions,y_test))
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=5,weights='distance')
knn.fit(X_train,y_train)
print("Training accuracy",knn.score(X_train,y_train))

predictions = knn.predict(X_test)

from sklearn.metrics import accuracy_score
print("Testing accuracy",accuracy_score(predictions,y_test))
from sklearn.svm import SVC
svc= SVC(C=1.0,kernel='linear',degree=3,gamma='auto')
svc.fit(X_train,y_train)

print("Training accuracy",svc.score(X_train,y_train))

predictions = svc.predict(X_test)

from sklearn.metrics import accuracy_score
print("Testing accuracy",accuracy_score(predictions,y_test))
import matplotlib.pyplot as plt

text_length = []

for text in range(len(final_df['MergedColumn'])):
    try:
        text_length.append(len(final_df['MergedColumn'][text]))

    except Exception as e:
        pass

print("Maximum length of  Data", np.max(text_length))
print("Minimum length of  Data", np.min(text_length))
print("Median length of  Data", np.median(text_length))
print("Average length of  Data",np.mean(text_length))
print("Standard Deviation of  Data",np.std(text_length))
plt.boxplot(text_length)
plt.show()
# Setting the maximum length of the sentence as 200
max_features = 2000
maxlen = 200
embedding_size = 200
tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(final_df['MergedColumn'])
X = tokenizer.texts_to_sequences(final_df['MergedColumn'])
X = pad_sequences(X, maxlen = maxlen)
y = np.asarray(final_df['category'])
y = pd.get_dummies(final_df['category']).values

print(X.shape)
print(y.shape)

del final_df

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(X, y, test_size=0.2, random_state=42)
embeddings_dictionary = dict()

#glove_file = open('/content/drive/My Drive/glove.6B.200d.txt', encoding="utf8")
glove_file = open('/kaggle/input/glove200dtxt/glove.6B.200d.txt', encoding="utf8")
for line in glove_file:
    records = line.split()
    word = records[0]
    vector_dimensions = asarray(records[1:], dtype='float32')
    embeddings_dictionary[word] = vector_dimensions

glove_file.close()
num_words = len(tokenizer.word_index) + 1
embedding_matrix = zeros((num_words, 200))
for word, index in tokenizer.word_index.items():
    embedding_vector = embeddings_dictionary.get(word)
    if embedding_vector is not None:
        embedding_matrix[index] = embedding_vector


print(len(embeddings_dictionary.values()))
print("Num words",num_words)
print("matrix size ",embedding_matrix.shape)
print("embeddings size ",embedding_size)
print("Max len",maxlen)
model = Sequential()
model.add(Embedding(num_words, embedding_size, weights = [embedding_matrix]))
model.add(SpatialDropout1D(0.2))
#model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2,return_sequences=True))
model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(40, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())
history= model.fit(x_train,y_train,validation_data=(x_test,y_test),epochs=10,batch_size=64)
loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
print('Accuracy: %f' % (accuracy*100))
plt.title('Accuracy')
plt.plot(history.history['accuracy'], label='train')
plt.plot(history.history['val_accuracy'], label='test')
plt.legend()
plt.show();
plt.title('Loss')
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='test')
plt.legend()
plt.show();
