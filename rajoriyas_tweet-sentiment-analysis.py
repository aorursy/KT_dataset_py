# Text Normalization
"""
Created on Thu May 14 13:38:19 2020

@author: rahul
"""
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.tag import pos_tag
from nltk.corpus import wordnet
import string
import pkg_resources

def replaceElongated(word):
    """ Replaces an elongated word with its basic form, unless the word exists in the lexicon """

    repeat_regexp = re.compile(r'(\w*)(\w)\2(\w*)')
    repl = r'\1\2\3'
    if wordnet.synsets(word):
        return word
    repl_word = repeat_regexp.sub(repl, word)
    if repl_word != word:      
        return replaceElongated(repl_word)
    else:       
        return repl_word
    
def load_dict_contractions():    
    return {
        "cant":"can not",
        "dont":"do not",
        "wont":"will not",
        "ain't":"is not",
        "amn't":"am not",
        "aren't":"are not",
        "can't":"cannot",
        "'cause":"because",
        "couldn't":"could not",
        "couldn't've":"could not have",
        "could've":"could have",
        "daren't":"dare not",
        "daresn't":"dare not",
        "dasn't":"dare not",
        "didn't":"did not",
        "doesn't":"does not",
        "don't":"do not",
        "e'er":"ever",
        "em":"them",
        "everyone's":"everyone is",
        "finna":"fixing to",
        "gimme":"give me",
        "gonna":"going to",
        "gon't":"go not",
        "gotta":"got to",
        "hadn't":"had not",
        "hasn't":"has not",
        "haven't":"have not",
        "he'd":"he would",
        "he'll":"he will",
        "he's":"he is",
        "he've":"he have",
        "how'd":"how would",
        "how'll":"how will",
        "how're":"how are",
        "how's":"how is",
        "I'd":"I would",
        "I'll":"I will",
        "I'm":"I am",
        "I'm'a":"I am about to",
        "I'm'o":"I am going to",
        "isn't":"is not",
        "it'd":"it would",
        "it'll":"it will",
        "it's":"it is",
        "I've":"I have",
        "kinda":"kind of",
        "let's":"let us",
        "mayn't":"may not",
        "may've":"may have",
        "mightn't":"might not",
        "might've":"might have",
        "mustn't":"must not",
        "mustn't've":"must not have",
        "must've":"must have",
        "needn't":"need not",
        "ne'er":"never",
        "o'":"of",
        "o'er":"over",
        "ol'":"old",
        "oughtn't":"ought not",
        "shalln't":"shall not",
        "shan't":"shall not",
        "she'd":"she would",
        "she'll":"she will",
        "she's":"she is",
        "shouldn't":"should not",
        "shouldn't've":"should not have",
        "should've":"should have",
        "somebody's":"somebody is",
        "someone's":"someone is",
        "something's":"something is",
        "that'd":"that would",
        "that'll":"that will",
        "that're":"that are",
        "that's":"that is",
        "there'd":"there would",
        "there'll":"there will",
        "there're":"there are",
        "there's":"there is",
        "these're":"these are",
        "they'd":"they would",
        "they'll":"they will",
        "they're":"they are",
        "they've":"they have",
        "this's":"this is",
        "those're":"those are",
        "'tis":"it is",
        "'twas":"it was",
        "wanna":"want to",
        "wasn't":"was not",
        "we'd":"we would",
        "we'd've":"we would have",
        "we'll":"we will",
        "we're":"we are",
        "weren't":"were not",
        "we've":"we have",
        "what'd":"what did",
        "what'll":"what will",
        "what're":"what are",
        "what's":"what is",
        "what've":"what have",
        "when's":"when is",
        "where'd":"where did",
        "where're":"where are",
        "where's":"where is",
        "where've":"where have",
        "which's":"which is",
        "who'd":"who would",
        "who'd've":"who would have",
        "who'll":"who will",
        "who're":"who are",
        "who's":"who is",
        "who've":"who have",
        "why'd":"why did",
        "why're":"why are",
        "why's":"why is",
        "won't":"will not",
        "wouldn't":"would not",
        "would've":"would have",
        "y'all":"you all",
        "you'd":"you would",
        "you'll":"you will",
        "you're":"you are",
        "you've":"you have",
        "Whatcha":"What are you",
        "luv":"love",
        "sux":"sucks",
        "couldn't":"could not",
        "wouldn't":"would not",
        "shouldn't":"should not",
        "im":"i am"
        }

single_word = list(string.ascii_lowercase)

lemmatizer = WordNetLemmatizer()
ps = PorterStemmer()
stop_words = set(stopwords.words('english'))-set(['not', 'no'])

def normalization(text):
    text = str(text).lower()
    
    # Unicodes
    text = re.sub(r'(\\u[0-9A-Fa-f]+)',r'', text)       
    text = re.sub(r'[^\x00-\x7f]',r'',text)
    
    # URL
    text = re.sub('((www\.[^\s]+)|(https?://[^\s]+))',' ',text)
    text = re.sub(r'#([^\s]+)', r'\1', text)
    
    # User Tag
    text = re.sub('@[^\s]+',' ',text)
    
    # Hash Tag
    text = re.sub(r'#([^\s]+)', r' ', text)
    
    # Number
    text = ''.join([i for i in text if not i.isdigit()])      
    
    
    # Punctuation
    #text = ' '.join([char for char in text if char not in string.punctuation])
    for sym in string.punctuation:
        text = text.replace(sym, " ")
    
    # Elongated Words
    for word in text.split():
        text = text.replace(word, replaceElongated(word))
    
    # Contraction
    CONTRACTIONS = load_dict_contractions()
    text = text.replace("’","'")
    words = text.split()
    reformed = [CONTRACTIONS[word] if word in CONTRACTIONS else word for word in words]
    text = " ".join(reformed)
    
    """
    # Lemmatization 
    for word in text.split():
        text = text.replace(word, lemmatizer.lemmatize(word))
        
    # Stemming
    for word in text.split():
        text = text.replace(word, ps.stem(word))
    
    # Correct Spell
    max_edit_distance_lookup = 2
    suggestions = sym_spell.lookup_compound(text, max_edit_distance_lookup)
    for suggestion in suggestions:
        text = ''.join(suggestion.term)    
    
    # Stop words
    text = " ".join([word for word in text.split() if not word in stop_words])
    
    # Remove Extras
    # Source: https://www.geeksforgeeks.org/part-speech-tagging-stop-words-using-nltk-python/
    # Source: https://stackoverflow.com/questions/39634222/is-there-a-way-to-remove-proper-nouns-from-a-sentence-using-python
    tagged_text = pos_tag(text.split())
    edited_text = [word for word, tag in tagged_text if tag != 'NN' and tag != 'NNS' and tag != 'NNP' and tag != 'NNPS' and tag != 'PRP' and tag != 'PRP$']
    text = ' '.join(edited_text)
    """
    # Single Character   
    text = ' '.join( [w for w in text.split() if len(w)>1 and w != 'a' and w != 'i'])
         
    return text
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from collections import Counter
df_train=pd.read_csv('../input/tweet-sentiment-extraction/train.csv')
df_train_2=pd.read_csv('../input/complete-tweet-sentiment-extraction-data/tweet_dataset.csv')
df_test=pd.read_csv('../input/tweet-sentiment-extraction/test.csv')
#df_train.drop(df_train[df_train.sentiment=='neutral'].index, axis=0, inplace=True)
#df_train_2.drop(df_train[df_train_2.new_sentiment=='neutral'].index, axis=0, inplace=True)
#df_test.drop(df_test[df_test.sentiment=='neutral'].index, axis=0, inplace=True)
from sklearn.preprocessing import LabelEncoder
encoder=LabelEncoder()
df_train_2.info()
df_train_2.drop(columns=['sentiment', 'author', 'old_text', 'aux_id'], inplace=True)
df_train_2=df_train_2.rename(columns={'new_sentiment':'sentiment'})
df_train_2.info()
df_train.info()
df_train=df_train[['textID', 'text', 'selected_text', 'sentiment']]
df_train_2.info()

df_train=df_train.append(df_train_2)
df_train['sentiment'].replace('', np.nan, inplace=True)
df_train.dropna(subset=['sentiment'], inplace=True)
df_train.shape
print(df_train['sentiment'].unique())
df_train['encoded_sentiment']=encoder.fit_transform(df_train['sentiment'])
df_train=pd.get_dummies(df_train, columns=['sentiment'])
df_train=df_train[['textID', 'text', 'encoded_sentiment', 'sentiment_negative', 'sentiment_neutral', 'sentiment_positive', 'selected_text']]
#df_train=df_train[['textID', 'text', 'encoded_sentiment', 'sentiment_negative', 'sentiment_positive', 'selected_text']]

df_train.head(10)
df_test['encoded_sentiment']=encoder.fit_transform(df_test['sentiment'])
df_test=pd.get_dummies(df_test, columns=['sentiment'])
df_test=df_test[['textID', 'text', 'encoded_sentiment', 'sentiment_negative', 'sentiment_neutral', 'sentiment_positive']]
#df_test=df_test[['textID', 'text', 'encoded_sentiment', 'sentiment_negative', 'sentiment_positive']]

df_test.head(10)
df_train['text'] = df_train['text'].apply(normalization)
df_train['text'].replace('', np.nan, inplace=True)
df_train.dropna(subset=['text'], inplace=True)

df_train.head(10)
df_test['text'] = df_test['text'].apply(normalization)
df_test['text'].replace('', np.nan, inplace=True)
df_test.dropna(subset=['text'], inplace=True)

df_test.head(10)
#pos = len(df['encoded_sentiment'][df.encoded_sentiment == 2])
pos = len(df_train['encoded_sentiment'][df_train.encoded_sentiment == 2])
neu = len(df_train['encoded_sentiment'][df_train.encoded_sentiment == 1])
neg = len(df_train['encoded_sentiment'][df_train.encoded_sentiment == 0])
def word_count(sentence):
    return len(str(sentence).split())
df_train['word_count'] = df_train['text'].apply(word_count)

#pos_sen_len = df['word_count'][df.encoded_sentiment == 2]
pos_sen_len = df_train['word_count'][df_train.encoded_sentiment == 2]
neu_sen_len = df_train['word_count'][df_train.encoded_sentiment == 1]
neg_sen_len = df_train['word_count'][df_train.encoded_sentiment == 0]
plt.figure(figsize=(12,6))
plt.xlim(0, 35, 5)
plt.xlabel('word count')
plt.ylabel('frequency')
plt.hist([pos_sen_len, neu_sen_len, neg_sen_len], color=['r', 'g', 'b'], alpha=0.5, label=['positive', 'neutral', 'negative'])
#plt.hist([pos_sen_len, neg_sen_len], color=['r', 'b'], alpha=0.5, label=['positive', 'negative'])
plt.legend(loc='upper right')
all_words = []
for line in list(df_train['text']):
    words = str(line).split()
    for word in words:
        all_words.append(word.lower())     
print(Counter(all_words).most_common(10))
df_train.info()
#X=(df_train.iloc[:, 1].values).astype('U')
y=(df_train.iloc[:, 3:6].values).astype('int32')
df_test.info()
#X_test=(df_test.iloc[:, 1].values).astype('U')
y_test=(df_test.iloc[:, 3:6].values).astype('int32')
"""
from sklearn.utils import shuffle
X, y = shuffle(X, y, random_state=0)
"""
# Glove Word Vocab
from tqdm import tqdm
def read_glove_vecs(glove_file):
    with open(glove_file, 'r') as f:
        words=set()
        word_to_vec_map=dict()
        for line in tqdm(f):
            line=line.strip().split()
            curr_word=''.join(line[:-300])
            words.add(curr_word)
            word_to_vec_map[curr_word]=np.array(line[-300:], dtype=np.float32)
            
        i=1
        words_to_index=dict()
        index_to_words=dict()
        for w in sorted(words):
            words_to_index[w]=i
            index_to_words[i]=w
            i+=1
    return words_to_index, index_to_words, word_to_vec_map
word_to_index, index_to_word, word_to_vec_map=read_glove_vecs('../input/glove840b/glove.840B.300d.txt')

print(len(word_to_index))
print(list(word_to_index.items())[:5])
print(list(word_to_vec_map.items())[:2])
np.random.seed(0)
from keras import regularizers, callbacks
from keras.models import Model, Sequential
from keras.layers import Dense, Input, Dropout, LSTM, Activation, Bidirectional, SpatialDropout1D, GlobalMaxPooling1D, BatchNormalization, PReLU
from keras.layers.embeddings import Embedding
from keras.optimizers import RMSprop, Adam, Adamax, SGD
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.losses import BinaryCrossentropy
from keras.metrics import Precision, Recall, CategoricalAccuracy
np.random.seed(1)
"""
avoided_word = list()

def sentences_to_indices(data, word_to_index, max_len):
    m=data.shape[0]
    indices=np.zeros((m, max_len))
    for i in range(m):
        sentence_words=data[i].lower().split()
        j=0
        for w in sentence_words:
            if w in word_to_index:
                indices[i, j]=word_to_index[w]
            else:
                avoided_word.append(w)
            j+=1
    return indices
"""
from keras.preprocessing import text, sequence
tk=text.Tokenizer(num_words=200000)

df_train.text=df_train.text.astype(str)
df_test.text=df_test.text.astype(str)
tk.fit_on_texts(list(df_train.text.values)+list(df_test.text.values))
X_text_indices=tk.texts_to_sequences(df_train.text.values)
X_test_text_indices=tk.texts_to_sequences(df_test.text.values)

maxlen=-1
for text in X_text_indices:
    if len(text)>maxlen:
        maxlen=len(text)
for text in X_test_text_indices:
    if len(text)>maxlen:
        maxlen=len(text)       
print(maxlen)

X_text_indices=sequence.pad_sequences(X_text_indices, maxlen=maxlen)
print(X_text_indices.shape)

X_test_text_indices=sequence.pad_sequences(X_test_text_indices, maxlen=maxlen)
print(X_test_text_indices.shape)
word_index=tk.word_index
print(len(word_index))
embedding_matrix=np.zeros((len(word_index)+1, 300))
for word, i in tqdm(word_index.items()):
    embedding_vector=word_to_vec_map.get(word)
    if embedding_vector is not None:
        embedding_matrix[i]=embedding_vector
"""
def pretrained_embedding_layer(word_to_vec_map, word_to_index):
    vocab_len=len(word_to_index)+1
    emb_dim=word_to_vec_map['cucumber'].shape[0]
    emb_matrix=np.zeros((vocab_len, emb_dim))
    for word, idx in word_to_index.items():
        emb_matrix[idx, :]=word_to_vec_map[word]
        #emb_matrix[idx, :]=np.pad(word_to_vec_map[word], (0, (word_to_vec_map['cucumber'].shape[0]-word_to_vec_map[word].shape[0])), 'constant')
    embedding_layer=Embedding(input_dim=vocab_len, output_dim=emb_dim, trainable=False)
    embedding_layer.build((None,))
    embedding_layer.set_weights([emb_matrix])
    return embedding_layer

embedding_layer = pretrained_embedding_layer(word_to_vec_map, word_to_index)
print("weights[0][1][3] =", embedding_layer.get_weights()[0][1][3])
"""
"""
maxLen=-1
for text in X:
    if len(text.split())>maxLen:
        maxLen=len(text.split())
"""
"""
def Sentiment_Extraction(input_shape, word_to_vec_map, word_to_index):    
    sentence_indices=Input(shape=input_shape, dtype='int32')
    embedding_layer=pretrained_embedding_layer(word_to_vec_map, word_to_index)
    embeddings=embedding_layer(sentence_indices)
    #embeddings=SpatialDropout1D(0.5)(embeddings) 
    
    X=LSTM(units=300, return_sequences=False, recurrent_dropout=0.2, dropout=0.2)(embeddings)
    X=BatchNormalization()(X)
    
    X=Dense(200)(X)
    X=PReLU()(X)
    X=Dropout(0.2)(X)
    X=BatchNormalization()(X)

    X=Dense(200)(X)
    X=PReLU()(X)
    X=Dropout(0.2)(X)
    X=BatchNormalization()(X)
        
    #X=LSTM(units=128, return_sequences=False)(X)
    #X=Dropout(0.5)(X)
    X=Dense(units=2)(X)
    X=Activation(activation='softmax')(X)
        
    model=Model(inputs=sentence_indices, outputs=X)
    return model
"""
#model=Sentiment_Extraction((maxLen,), word_to_vec_map, word_to_index)
#model.summary()
model=Sequential()
model.add(Embedding(len(word_index)+1, 300, weights=[embedding_matrix], input_length=maxlen, trainable=False))

model.add(LSTM(units=300, return_sequences=False, recurrent_dropout=0.2, dropout=0.2))
model.add(BatchNormalization())

model.add(Dense(200))
model.add(PReLU())
model.add(Dropout(0.2))
model.add(BatchNormalization())

model.add(Dense(200))
model.add(PReLU())
model.add(Dropout(0.2))
model.add(BatchNormalization())

model.add(Dense(200))
model.add(PReLU())
model.add(Dropout(0.2))
model.add(BatchNormalization())

model.add(Dense(200))
model.add(PReLU())
model.add(Dropout(0.2))
model.add(BatchNormalization())

model.add(Dense(units=3))
model.add(Activation(activation='softmax'))
model.summary()
model.compile(loss='categorical_crossentropy' , optimizer='adam', metrics=['accuracy'])
#model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
#model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy', 'precision', 'recall'])
#model.compile(loss='categorical_crossentropy' , optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])
#X_text_indices = sentences_to_indices(X, word_to_index, maxLen)
from sklearn.model_selection import train_test_split
X_train_text_indices, X_dev_text_indices, y_train, y_dev = train_test_split(X_text_indices, y, test_size=0.1, random_state=42)  
#callbacks = [EarlyStopping(monitor='val_loss', patience=3),] #ReduceLROnPlateau(monitor='val_loss', factor=0.01, patience=1),]
#history = model.fit(X_text_indices, y, epochs=50, batch_size=32, validation_split=0.25, callbacks=callbacks ,shuffle=False)
#history = model.fit(X_text_indices, y, epochs=50, batch_size=32, validation_split=0.2, shuffle=True, verbose=1)
history = model.fit(X_train_text_indices, y_train, epochs=50, batch_size=128, validation_data=(X_dev_text_indices, y_dev), shuffle=True)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model train vs validation loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper right')
plt.show()
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model train vs validation accuracy')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper right')
plt.show()
model.save('tweet_sentiment_extraction.h5')
from keras.models import load_model
model = load_model('tweet_sentiment_extraction.h5')
para = model.evaluate(X_dev_text_indices, y_dev)
print()
print("Test loss :", para[0], 'Test accuracy :', para[1])
y_dev_pred = model.predict(X_dev_text_indices)

for i in range(len(y_dev_pred)):
    y_dev_pred[i] = np.argmax(y_dev_pred[i])

y_dev_pred = y_dev_pred[:, 0]
print(y_dev_pred)

for i in range(len(y_dev)):
    y_dev[i] = np.argmax(y_dev[i])
y_dev = y_dev[:, 0]
print(y_dev)
from sklearn.metrics import confusion_matrix
cm_dev = confusion_matrix(y_dev, y_dev_pred)
print(cm_dev)

total=sum(sum(cm_dev))

accuracy = (cm_dev[0,0]+cm_dev[1,1]+cm_dev[2,2])/total
print('Accuracy:', accuracy)

sensitivity = cm_dev[0,0]/(cm_dev[0,0]+cm_dev[1,0])
print('Sensitivity : ', sensitivity )

specificity = cm_dev[1,1]/(cm_dev[1,1]+cm_dev[0,1])
print('Specificity : ', specificity)
"""
accuracy = (cm_dev[0,0]+cm_dev[1,1])/total
print('Accuracy:', accuracy)
"""
y_test_pred = model.predict(X_test_text_indices)

for i in range(len(y_test_pred)):
    y_test_pred[i] = np.argmax(y_test_pred[i])
y_test_pred = y_test_pred[:, 0]

for i in range(len(y_test)):
    y_test[i] = np.argmax(y_test[i])
y_test = y_test[:, 0]
from sklearn.metrics import confusion_matrix
cm_test = confusion_matrix(y_test, y_test_pred)
print(cm_test)

total=np.sum(cm_test)

accuracy = (cm_test[0,0]+cm_test[1,1]+cm_test[2,2])/total
print('Accuracy:', accuracy)

sensitivity = cm_test[0,0]/(cm_test[0,0]+cm_test[1,0])
print('Sensitivity : ', sensitivity )

specificity = cm_test[1,1]/(cm_test[1,1]+cm_test[0,1])
print('Specificity : ', specificity)
"""
accuracy = (cm_test[0,0]+cm_test[1,1])/total
print('Accuracy:', accuracy)
"""
from keras import backend as K
def jaccard_distance(y_true, y_pred, smooth=100):
    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    sum_ = K.sum(K.abs(y_true) + K.abs(y_pred), axis=-1)
    jac = (intersection + smooth) / (sum_ - intersection + smooth)
    return (1 - jac) * smooth, intersection, sum_, jac
score, intersection, sum_, jac = jaccard_distance(y_true=y_test.astype('int32'), y_pred=y_test_pred.astype('int32'))
print(jac)