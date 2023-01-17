import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from gensim.models import Word2Vec
from keras.models import Sequential
from keras.callbacks import EarlyStopping
from keras import optimizers
from keras.layers import Embedding, Conv1D, MaxPooling1D, Dense
from keras.layers import Dense, Activation, Dropout, Flatten,Input
from keras import regularizers
import datetime
from sklearn.metrics import confusion_matrix
datasets_dir =""
vnrows=None
#vnrows=2000
datasets_dir ="../input/"
df = pd.read_csv(datasets_dir + "fakenews_preprocessed_35k.csv",nrows=vnrows, encoding='utf-8')
#  cleanup structure dataset
print (df.columns)
#drop Unamed column
df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
# verify true/fake quantity
print (df.groupby(['label'])['label'].count())
# shuffle
df = df.sample(frac=1).reset_index(drop=True)
# another peek at shuffled data before proceeding
df[['label','target_text']].head(5)
# define the shape of the X array by  mean + standard deviation (std) of articles word count
# max_seq_len = np.round(df['doc_length'].mean() + df['doc_length'].std()).astype(int)
 
# same applied on full prepared  dataset  returns 458, therefore considering the slice saved separately
# just a formula to determine a reasonable number of features
max_seq_len = 458 
# split data for training and test
vtest_size = 0.33
vtarget_text='target_text'
X_train, X_test, y_train, y_test = train_test_split(
                                   df[vtarget_text].to_list(), 
                                   df['label'].values,
                                   test_size=vtest_size, 
                                   random_state=42)
print(len(X_train))
print(len(X_test))
print(len(y_train))
print(len(y_test))
# maximum number of words to map, based on word frequency 
MAX_NB_WORDS = round(int(df.doc_length.sum()/max_seq_len))
# instantiate Tokenizer
tokenizer = Tokenizer(num_words=MAX_NB_WORDS, lower=True, char_level=False)
# Tokenaizer fitting
tokenizer.fit_on_texts(X_train)
word_index = tokenizer.word_index
print("dictionary size: ", len(word_index))
# prepare input data for the model, first to sequences then to pad sequences
word_seq_train = tokenizer.texts_to_sequences(X_train)
word_seq_train = sequence.pad_sequences(word_seq_train, maxlen=max_seq_len)
# a bidemensiona list
word_seq_train.shape
# building a gensim dictionary based on content
gensim_news_desc = []
chunk_data = X_train
for record in range(0,len(chunk_data)):
    news_desc_list = []
    for tok in chunk_data[record].split():
        #print ("Tok for Gensim model:" + str(tok))
        news_desc_list.append(str(tok))
    gensim_news_desc.append(news_desc_list)
# word to vector neural network / it takes a few minutes
vsize=max_seq_len 
vmin_count=4 # default 5
gensim_model = Word2Vec(gensim_news_desc, min_count=vmin_count, size = vsize, sg=1)
print(gensim_model)
# as output a list with gensim word vocabulary that tracks unique words, builds a huffman tree(frequent words are closer to the root), discards extremely rare words
words = list(gensim_model.wv.vocab)
# example list items of the corpus words
words[0:3]
# training params
batch_size = 1024
num_epochs = 10
#model parameters
num_filters = 128
embed_dim = max_seq_len # 200 default
weight_decay = 1e-4
class_weight = {0: 1,1:1}

print('preparing embedding matrix...')
gensim_words_not_found = []
gensim_nb_words = len(gensim_model.wv.vocab)
print("gensim_nb_words : ",gensim_nb_words)
gensim_embedding_matrix = np.zeros((gensim_nb_words, embed_dim))

for word, i in word_index.items():
    #print(word)
    if i >= gensim_nb_words:
        continue
    if word in gensim_model.wv.vocab :
        embedding_vector = gensim_model.wv[word]
        if (embedding_vector is not None) and len(embedding_vector) > 0:
            gensim_embedding_matrix[i] = embedding_vector
    else :
        gensim_words_not_found.append(word)
        
print (gensim_embedding_matrix.shape)
model = Sequential()
# embedding gensim word2vec model
model.add(Embedding(gensim_nb_words, embed_dim, weights=[gensim_embedding_matrix], input_length=max_seq_len))
model.add(Conv1D(num_filters, 5, activation='relu', padding='same'))
model.add(MaxPooling1D(2))
model.add(Dense(64, activation='relu', kernel_regularizer=regularizers.l2(weight_decay)))
model.add(Dropout(0.5)) # set to 6 in Py-reake
model.add(Dense(32, activation='relu', kernel_regularizer=regularizers.l2(weight_decay)))
model.add(Dropout(0.5))  # set to 6 in Py-reake
model.add(Flatten())
model.add(Dense(1, activation='sigmoid')) 
adam = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
# compile
model.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy'])
# define callbacks
early_stopping = EarlyStopping(monitor='val_loss', min_delta=0.01, patience=4, verbose=1)
callbacks_list = [early_stopping]
#  model fitting 
hist = model.fit(word_seq_train, y_train, batch_size=batch_size, epochs=num_epochs, callbacks=callbacks_list, validation_split=0.1, shuffle=True, verbose=2,class_weight=class_weight)
import matplotlib.pyplot as plt
# visuals model loss and accuracy
plt.plot(hist.history['accuracy'])
plt.plot(hist.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summary
model.summary()
# tokenizing and sequencing testing input
word_seq_test = tokenizer.texts_to_sequences(X_test)
word_seq_test = sequence.pad_sequences(word_seq_test, maxlen=max_seq_len)
predictions = model.predict(word_seq_test)
pred_labels = predictions.round()
# 1 testing and model verification
predictions = model.predict(word_seq_test)
pred_labels = predictions.round()
unique, counts = np.unique(y_test, return_counts=True)
cm = confusion_matrix(y_test, pred_labels, labels=[1,0])

# totals, articles, wrong predictions per group
cm
# function to evaluate predictions in percentage
def print_prediction_detail_eval(correct_answers, pred):
    i = 0
    tot_correct = 0
    tot_wrong = 0
    for a in correct_answers:
        if a == pred[i]:
            #print ("prediction n." + str(i) + "is correct !")
            tot_correct = tot_correct + 1
        else:
            #print ("prediction n." + str(i) + "is wrong !")
            tot_wrong = tot_wrong + 1
        i = i + 1
    percentile_correct   = (tot_correct/len(correct_answers) )*100
    percentile_correct = str(round(percentile_correct,3))
    print ("Correct answers in percentage: " + percentile_correct)
    print ("Total correct prediction: " + str(tot_correct)  + " on "  + str(len(correct_answers)) + " in total") 
# model predictions more readable
print_prediction_detail_eval(y_test, pred_labels )
datasets_dir ="../input/"
vnrows=None
df_small = pd.read_csv(datasets_dir + "fakenews_preprocessed_4k.csv",nrows=vnrows, encoding='utf-8')
print (df_small.groupby(['label'])['label'].count())
# 2 additional testing new data (sample) 3000  score: 98.9
df2 = df_small[:3000]

print (df2.groupby(['label'])['label'].count())
X_test2 = df2.target_text.to_list()
y_test2 = df2.label.to_list()
word_seq_test2 = tokenizer.texts_to_sequences(X_test2)
word_seq_test2 = sequence.pad_sequences(word_seq_test2, maxlen=max_seq_len)

predictions2 = model.predict(word_seq_test2)
pred_labels2 = predictions2.round()
print_prediction_detail_eval(y_test2, pred_labels2)
# 3  additional testing new data 2000 articles 20% fake score:98.9
vnrows=None
df3 = pd.read_csv(datasets_dir + "fakenews_preprocessed_4k.csv",nrows=vnrows, encoding='utf-8')
df3 = df3.sample(frac=1)

df_fake = df3[(df3['label'] == 1) ][:400]
df_true = df3[(df3['label'] == 0) ][:2000] 
df3=pd.concat([df_fake,df_true]).reset_index(drop=True)
print (df3.groupby(['label'])['label'].count())

X_test3 = df3.target_text.to_list()
y_test3 = df3.label.to_list()
word_seq_test3 = tokenizer.texts_to_sequences(X_test3)
word_seq_test3 = sequence.pad_sequences(word_seq_test3, maxlen=max_seq_len)

predictions3 = model.predict(word_seq_test3)
pred_labels3 = predictions3.round()
print_prediction_detail_eval(y_test2, pred_labels2)
""" eventually save and load a model  """
# save best model to compete with best one
today = datetime.date.today()
vdate = today.strftime("%Y%m%d")    
model_dir ='/kaggle/working/'
filename = 'gensim_features_nb_20054_458_' + vdate  + '.sav'
#model.save(model_dir + filename)

# can reload a saved  model
from keras.models import load_model
model_dir ='/kaggle/working/'
filename='gensim_features_nb_20054_458_20200516.sav'
#model = load_model(model_dir + filename)