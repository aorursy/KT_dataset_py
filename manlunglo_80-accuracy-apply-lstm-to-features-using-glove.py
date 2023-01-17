# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.

import seaborn as sns

from nltk.corpus import stopwords

import re

import string

import pandas_profiling

import random





from keras.preprocessing.text import Tokenizer

from keras.preprocessing.sequence import pad_sequences

data = pd.read_csv('../input/nlp-getting-started/train.csv')

data_eval = pd.read_csv('../input/nlp-getting-started/test.csv')



data.head()
x = data.target.value_counts()

sns.barplot(x.index, x)
pandas_profiling.ProfileReport(data)
embedding_dict={}

with open('/kaggle/input/glove6b/glove.6B.50d.txt','r') as f:

    for line in f:

        values=line.split()

        word=values[0]

        vectors=np.asarray(values[1:],'float32')

        embedding_dict[word]=vectors

f.close()



glove_embedding_dim = list(embedding_dict.values())[0].shape[0]
print('glove_embedding_dim:',glove_embedding_dim)
def remove_URL(text):

    url = re.compile(r'https?://\S+|www\.\S+')

    return url.sub(r'',text)



def remove_html(text):

    html=re.compile(r'<.*?>')

    return html.sub(r'',text)



def remove_emoji(text):

    emoji_pattern = re.compile("["

                           u"\U0001F600-\U0001F64F"  # emoticons

                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs

                           u"\U0001F680-\U0001F6FF"  # transport & map symbols

                           u"\U0001F1E0-\U0001F1FF"  # flags (iOS)

                           u"\U00002702-\U000027B0"

                           u"\U000024C2-\U0001F251"

                           "]+", flags=re.UNICODE)

    return emoji_pattern.sub(r'', text)



def remove_punct(text):

    table=str.maketrans('','',string.punctuation)

    return text.translate(table)



def remove_tag(text):

    url = re.compile(r'(?<=\@)(.*?)(?= )')

    return url.sub(r'',text)



def remove_space(text):

    return text.replace(r'%20',' ')
def cleanse_data(data_in):

    data = data_in.copy()

    data['text'] = data['text'].apply(lambda x:remove_URL(x))

    data['text'] = data['text'].apply(lambda x:remove_html(x))

    data['text'] = data['text'].apply(lambda x:remove_emoji(x))

    data['text'] = data['text'].apply(lambda x:remove_tag(x))

    data['text'] = data['text'].apply(lambda x:remove_punct(x))

    

    data['keyword'].fillna('Nothing', inplace=True)

    data['keyword'] = data['keyword'].apply(lambda x:remove_space(x))

    data['keyword'] = data['keyword'].apply(lambda x:remove_punct(x))

    

    data['location'].fillna('Nothing', inplace=True)

    data['location'] = data['location'].apply(lambda x:remove_punct(x))

    return data

    
data_cleansed = cleanse_data(data)
def get_train_data(max_len, column_name):

    

    X = data_cleansed[column_name]

    tokenizer = Tokenizer()

    tokenizer.fit_on_texts(X)

    vocab_size = len(tokenizer.word_index)+1



    # Text Embeddings

    X = tokenizer.texts_to_sequences(X)

    X = pad_sequences(X, maxlen=max_len, truncating='post', padding='post')



    # Construct embedding matrix

    embedding_matrix = np.zeros((vocab_size,glove_embedding_dim))



    for word, index in tokenizer.word_index.items():

        embedding_vector = embedding_dict.get(word)

        if embedding_vector is not None:

            embedding_matrix[index] = embedding_vector



    return X, embedding_matrix, vocab_size, tokenizer
max_len = dict()

X = dict()

embedding_matrix = dict()

vocab_size = dict()

tokenizer = dict()



max_len["text"] = 50

max_len["keyword"] = 3

max_len["location"] = 3



for column_name in ["text", "keyword", "location"]:

    X[column_name], embedding_matrix[column_name], vocab_size[column_name], tokenizer[column_name] = get_train_data(max_len[column_name], column_name)

    

y = np.array(data['target'])
from keras.models import Sequential, Model

from keras.layers import Dense, Activation, Embedding, Flatten

from keras.layers import Dense, Dropout, Embedding, LSTM, GRU, Bidirectional, Input, Concatenate, concatenate

from keras.wrappers.scikit_learn import KerasClassifier

from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import KFold # import KFold
input_text = Input(shape=[max_len['text']])

input_keyword = Input(shape=[max_len['keyword']])

input_location = Input(shape=[max_len['location']])



def get_model(dropout_size = 0.3, GRU_size=64, neuron=256, return_seq=True):



    # Embedding layer for Column Text

    embedding_layer_text = Embedding(

                                input_dim=vocab_size['text'], 

                                output_dim=glove_embedding_dim, 

                                input_length=max_len['text'],

                                weights=[embedding_matrix['text']],

                                trainable= False)(input_text)

    if return_seq:

        left = Bidirectional(GRU(GRU_size,return_sequences=True))(embedding_layer_text)

        left = Flatten()(left)

    else:

        left = Bidirectional(GRU(GRU_size,return_sequences=False))(embedding_layer_text)

        

    # Embedding layer for Column Keyword

    embedding_layer_keyword = Embedding(input_dim=vocab_size['keyword'], 

                                output_dim=glove_embedding_dim, 

                                input_length=max_len['keyword'],

                                weights=[embedding_matrix['keyword']],

                                trainable= False)(input_keyword)

    right = GRU(8,return_sequences=False)(embedding_layer_keyword)



    # Embedding layer for Column Location

    embedding_layer_location = Embedding(input_dim=vocab_size['location'], 

                                output_dim=glove_embedding_dim, 

                                input_length=max_len['location'],

                                weights=[embedding_matrix['location']],

                                trainable= False)(input_location)

    mid = GRU(8,return_sequences=False)(embedding_layer_location)



    # Concatenate 3 embedding layers

    output = concatenate([left, right, mid])

    output = Dropout(dropout_size)(output)

    

    while neuron>=8:

        output = Dense(neuron, activation='relu')(output)

        neuron = int(neuron/2)

    

    output = Dense(1, activation='sigmoid')(output)



    model = Model(inputs=[input_text, input_keyword, input_location], outputs=[output])

    model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])



    #model.summary()

    

    return model
def plot_acc(display=0):

    x = [str(x) for x,val_accuracy,accuracy in result]

    val_accuracy = [val_accuracy for x,val_accuracy,accuracy in result]

    accuracy = [accuracy for x,val_accuracy,accuracy in result]



    plot_accuracy = pd.DataFrame(data={'combination':x, 'val_accuracy':val_accuracy, 'accuracy':accuracy})

    plot_accuracy2 = pd.melt(plot_accuracy, id_vars=['combination'], value_vars=['val_accuracy','accuracy'], var_name='acc')



    if display:

        sns.set_color_codes("colorblind")

        chart = sns.barplot(x="combination", y="value", hue="acc", data=plot_accuracy2)

        chart.set_xticklabels(chart.get_xticklabels(), rotation=90)

        chart.set(ylim=(0.79, 0.83))

    #chart.figure.savefig("output.png",dpi=chart.figure.dpi)

    

    plot_accuracy = pd.DataFrame(data={'combination':x, 'val_accuracy':val_accuracy, 'accuracy':accuracy})

    df_accuracy = plot_accuracy.groupby('combination').agg({'val_accuracy': ['mean', 'std'],

                                                        'accuracy': ['mean', 'std']}).sort_values(by=('val_accuracy','mean'), ascending =False)

    print('============ Plot Accuracy ===================')

    print(df_accuracy)

    df_accuracy.to_csv('df_accuracy.csv', index=True)

    print('==============================================')
list_batch_size = [2,4,8,16,32,64,128,256]

list_dropout_size = [0.1,0.2,0.3,0.4,0.5,0.6,0.7]

list_GRU_size = [4,8,16,32,64,128,256]

list_neuron = [8,16,32,64,128,256,512,1024]

list_return_seq = [0, 1]
# result = []

# test_run = 100



# for i in range(test_run):

#     batch_size = random.choice(list_batch_size)

#     epochs = 8

#     dropout_size = random.choice(list_dropout_size)

#     GRU_size = random.choice(list_GRU_size)

#     neuron = random.choice(list_neuron)

#     return_seq = random.choice(list_return_seq)

    

#     print('-----------------------------------------------------------------')

#     print('[current config]:',[batch_size,epochs,dropout_size,GRU_size,neuron,return_seq])

#     #val_accuracy = []

    

#     for j in range(5):

#         print('[current iteration]:',j+1)

#         final_model = get_model(dropout_size, GRU_size, neuron, return_seq)

#         history = final_model.fit(

#                             [X['text'],

#                              X['keyword'], 

#                              X['location']], 

#                             y, 

#                             batch_size=batch_size, 

#                             epochs=epochs, 

#                             verbose=1,

#                             validation_split=0.1

#                             )



#         for ep in range(epochs):

#             result.append([

#                             [batch_size,

#                             ep+1,

#                             dropout_size,

#                             GRU_size,

#                             neuron,

#                             return_seq],

#                             history.history.get('val_accuracy')[ep],

#                             history.history.get('accuracy')[ep]

#                           ])

    

#     plot_accuracy = plot_acc()
batch_size = 32

epochs = 6

dropout_size = 0.2

GRU_size = 8

neuron = 512

return_seq = 0



final_model = get_model(dropout_size, GRU_size, neuron, return_seq)

history = final_model.fit(

                        [X['text'],

                         X['keyword'], 

                         X['location']], 

                        y, 

                        batch_size=batch_size, 

                        epochs=epochs, 

                        verbose=1,

                        validation_split=0.1

                        )
data_eval_cleansed = cleanse_data(data_eval)
X_eval_Text = data_eval_cleansed['text']

X_eval_Text = tokenizer['text'].texts_to_sequences(X_eval_Text)

X_eval_Text = pad_sequences(X_eval_Text, maxlen=max_len['text'], truncating='post', padding='post')



X_eval_Keyword = data_eval_cleansed['keyword']

X_eval_Keyword = tokenizer['keyword'].texts_to_sequences(X_eval_Keyword)

X_eval_Keyword = pad_sequences(X_eval_Keyword, maxlen=max_len['keyword'], truncating='post', padding='post')



X_eval_Location = data_eval_cleansed['location']

X_eval_Location = tokenizer['location'].texts_to_sequences(X_eval_Location)

X_eval_Location = pad_sequences(X_eval_Location, maxlen=max_len['location'], truncating='post', padding='post')
y_eval = final_model.predict([X_eval_Text, X_eval_Keyword, X_eval_Location])

y_eval=np.round(y_eval).astype(int).reshape(X_eval_Text.shape[0])
output = pd.DataFrame({'id': data_eval_cleansed.id, 'target': y_eval})

output.to_csv('my_submission_20200130c.csv', index=False)

print("Your submission was successfully saved!")