import gc

import re

import operator 



import numpy as np

import pandas as pd



from gensim.models import KeyedVectors



from sklearn import model_selection



from keras.preprocessing.text import Tokenizer

from keras.preprocessing.sequence import pad_sequences

from keras.layers import Embedding, Input, Dense, CuDNNGRU,concatenate, Bidirectional, SpatialDropout1D, Conv1D, GlobalAveragePooling1D, GlobalMaxPooling1D

from keras.optimizers import RMSprop, Adam

from keras.models import Model

from keras.callbacks import EarlyStopping



import seaborn as sns
train = pd.read_csv("../input/twitter-my-personality/mypersonality.csv",encoding = "ISO-8859-1")



print("Train shape : ",train.shape)



train.head()

train = train.loc[:, [ 'STATUS', 'sEXT', 'sNEU', 'sAGR', 'sCON', 'sOPN']]



train.rename(columns = {'STATUS': 'comment_text'}, inplace=True)

for col in ['sEXT', 'sNEU', 'sAGR', 'sCON', 'sOPN']:

    train[col] = train[col] - train[col].min()

    train[col] = train[col]/train[col].max()
# Only 13GB of ram available, we gotta be careful !



df = train

del(train)

gc.collect()
gc.collect()
def build_vocab(texts):

    sentences = texts.apply(lambda x: x.split()).values

    vocab = {}

    for sentence in sentences:

        for word in sentence:

            try:

                vocab[word] += 1

            except KeyError:

                vocab[word] = 1

    return vocab
def check_coverage(vocab, embeddings_index):

    known_words = {}

    unknown_words = {}

    nb_known_words = 0

    nb_unknown_words = 0

    for word in vocab.keys():

        try:

            known_words[word] = embeddings_index[word]

            nb_known_words += vocab[word]

        except:

            unknown_words[word] = vocab[word]

            nb_unknown_words += vocab[word]

            pass



    print('Found embeddings for {:.3%} of vocab'.format(len(known_words) / len(vocab)))

    print('Found embeddings for  {:.3%} of all text'.format(nb_known_words / (nb_known_words + nb_unknown_words)))

    unknown_words = sorted(unknown_words.items(), key=operator.itemgetter(1))[::-1]



    return unknown_words
df['comment_text'] = df['comment_text'].apply(lambda x: x.lower())

gc.collect()
gc.collect()
contraction_mapping = {':)': 'happy', ':(': 'sad', "ain't": "is not", "aren't": "are not","can't": "cannot", "'cause": "because", "could've": "could have", "couldn't": "could not", "didn't": "did not",  "doesn't": "does not", "don't": "do not", "hadn't": "had not", "hasn't": "has not", "haven't": "have not", "he'd": "he would","he'll": "he will", "he's": "he is", "how'd": "how did", "how'd'y": "how do you", "how'll": "how will", "how's": "how is",  "I'd": "I would", "I'd've": "I would have", "I'll": "I will", "I'll've": "I will have","I'm": "I am", "I've": "I have", "i'd": "i would", "i'd've": "i would have", "i'll": "i will",  "i'll've": "i will have","i'm": "i am", "i've": "i have", "isn't": "is not", "it'd": "it would", "it'd've": "it would have", "it'll": "it will", "it'll've": "it will have","it's": "it is", "let's": "let us", "ma'am": "madam", "mayn't": "may not", "might've": "might have","mightn't": "might not","mightn't've": "might not have", "must've": "must have", "mustn't": "must not", "mustn't've": "must not have", "needn't": "need not", "needn't've": "need not have","o'clock": "of the clock", "oughtn't": "ought not", "oughtn't've": "ought not have", "shan't": "shall not", "sha'n't": "shall not", "shan't've": "shall not have", "she'd": "she would", "she'd've": "she would have", "she'll": "she will", "she'll've": "she will have", "she's": "she is", "should've": "should have", "shouldn't": "should not", "shouldn't've": "should not have", "so've": "so have","so's": "so as", "this's": "this is","that'd": "that would", "that'd've": "that would have", "that's": "that is", "there'd": "there would", "there'd've": "there would have", "there's": "there is", "here's": "here is","they'd": "they would", "they'd've": "they would have", "they'll": "they will", "they'll've": "they will have", "they're": "they are", "they've": "they have", "to've": "to have", "wasn't": "was not", "we'd": "we would", "we'd've": "we would have", "we'll": "we will", "we'll've": "we will have", "we're": "we are", "we've": "we have", "weren't": "were not", "what'll": "what will", "what'll've": "what will have", "what're": "what are",  "what's": "what is", "what've": "what have", "when's": "when is", "when've": "when have", "where'd": "where did", "where's": "where is", "where've": "where have", "who'll": "who will", "who'll've": "who will have", "who's": "who is", "who've": "who have", "why's": "why is", "why've": "why have", "will've": "will have", "won't": "will not", "won't've": "will not have", "would've": "would have", "wouldn't": "would not", "wouldn't've": "would not have", "y'all": "you all", "y'all'd": "you all would","y'all'd've": "you all would have","y'all're": "you all are","y'all've": "you all have","you'd": "you would", "you'd've": "you would have", "you'll": "you will", "you'll've": "you will have", "you're": "you are", "you've": "you have" }
def clean_contractions(text, mapping):

    specials = ["’", "‘", "´", "`"]

    for s in specials:

        text = text.replace(s, "'")

    text = ' '.join([mapping[t] if t in mapping else t for t in text.split(" ")])

    return text
df['comment_text'] = df['comment_text'].apply(lambda x: clean_contractions(x, contraction_mapping))
punct = "/-'?!.,#$%\'()*+-/:;<=>@[\\]^_`{|}~" + '""“”’' + '∞θ÷α•à−β∅³π‘₹´°£€\×™√²—–&'
punct_mapping = {"_":" ", "`":" "}
def clean_special_chars(text, punct, mapping):

    for p in mapping:

        text = text.replace(p, mapping[p])    

    for p in punct:

        text = text.replace(p, f' {p} ')     

    return text
df['comment_text'] = df['comment_text'].apply(lambda x: clean_special_chars(x, punct, punct_mapping))
swear_words = ['4r5e, 5h1t, 5hit, ass-fucker, assfucker, assfukka, asswhole, a_s_s, b!tch, b17ch, blow job, boiolas, bollok, boooobs, booooobs, booooooobs, bunny fucker, buttmuch, c0cksucker, carpet muncher, cl1t, cockface, cockmunch, cockmuncher, cocksuka, cocksukka, cokmuncher, coksucka, cunillingus, cuntlick, cuntlicker, cuntlicking, cyalis, cyberfuc, cyberfuck, cyberfucked, cyberfucker, cyberfuckers, cyberfucking, dirsa, dlck, dog-fucker, donkeyribber, ejaculatings, ejakulate, f u c k, f u c k e r, f4nny, faggitt, faggs, fannyflaps, fannyfucker, fanyy, fingerfucker, fingerfuckers, fingerfucks, fistfuck, fistfucked, fistfucker, fistfuckers, fistfucking, fistfuckings, fistfucks, fuckingshitmotherfucker, fuckwhit, fudge packer, fudgepacker, fukwhit, fukwit, fux0r, f_u_c_k, god-dam, kawk, knobead, knobed, knobend, knobjocky, knobjokey, kondum, kondums, kummer, kumming, kums, kunilingus, l3itch, m0f0, m0fo, m45terbate, ma5terb8, ma5terbate, master-bate, masterb8, masterbat3, masterbations, mof0, mothafuck, mothafuckaz, mothafucked, mothafucking, mothafuckings, mothafucks, mother fucker, motherfucked, motherfuckings, motherfuckka, motherfucks, muthafecker, muthafuckker, n1gga, n1gger, nigg3r, nigg4h, nob jokey, nobjocky, nobjokey, penisfucker, phuked, phuking, phukked, phukking, phuks, phuq, pigfucker, pimpis, pissflaps, rimjaw, s hit, scroat, sh!t, shitdick, shitfull, shitings, shittings, s_h_i_t, t1tt1e5, t1tties, teez, tittie5, tittiefucker, tittywank, tw4t, twathead, twunter, v14gra, v1gra, w00se, whoar']   

replace_with_fuck = []



for swear in swear_words:

   # if swear[1:(len(swear)-1)] not in embeddings_index:

        replace_with_fuck.append(swear)

        

replace_with_fuck = '|'.join(replace_with_fuck)

replace_with_fuck

        
def handle_swears(text):

    text = re.sub(replace_with_fuck, ' fuck ', text)

    return text
df['comment_text'] = df['comment_text'].apply(lambda x: handle_swears(x))

gc.collect()
train = df



train.head()
del(df)

gc.collect()
train_df, validate_df = model_selection.train_test_split(train, test_size=0.1)

print('%d train comments, %d validate comments' % (len(train_df), len(validate_df)))
import pickle



embedding_matrix = np.load('../input/jigsaw/embedding_matrix_jigsaw.npy')



# loading

with open('../input/jigsaw/tokenizer_jigsaw.pkl', 'rb') as handle:

    tokenizer = pickle.load(handle)
MAX_NUM_WORDS = 100000

PERSONALITY_COLUMNS = ['sEXT', 'sNEU', 'sAGR', 'sCON', 'sOPN']

TEXT_COLUMN = 'comment_text'



# Create a text tokenizer.

#tokenizer = Tokenizer(num_words=MAX_NUM_WORDS)

#tokenizer.fit_on_texts(train_df[TEXT_COLUMN])



# All comments must be truncated or padded to be the same length.

MAX_SEQUENCE_LENGTH = 280

def pad_text(texts, tokenizer):

    return pad_sequences(tokenizer.texts_to_sequences(texts), maxlen=MAX_SEQUENCE_LENGTH)
gc.collect()
EMBEDDINGS_DIMENSION = 300

#embedding_matrix = np.zeros((len(tokenizer.word_index) + 1,EMBEDDINGS_DIMENSION))
num_words_in_embedding = 0



#for word, i in tokenizer.word_index.items():

#    if word in embeddings_index.vocab:

#        embedding_vector = embeddings_index[word]

#        embedding_matrix[i] = embedding_vector        

#        num_words_in_embedding += 1
train_text = pad_text(train_df[TEXT_COLUMN], tokenizer)

train_labels = train_df.loc[:, PERSONALITY_COLUMNS]

validate_text = pad_text(validate_df[TEXT_COLUMN], tokenizer)

validate_labels = validate_df.loc[:, PERSONALITY_COLUMNS]
gc.collect()
sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')

embedding_layer = Embedding(len(tokenizer.word_index) + 1,

                            EMBEDDINGS_DIMENSION,

                            weights=[embedding_matrix],

                            input_length=MAX_SEQUENCE_LENGTH,

                            trainable=False)

x = embedding_layer(sequence_input)

x = SpatialDropout1D(0.2)(x)

x = Bidirectional(CuDNNGRU(64, return_sequences=True))(x)   

x = Conv1D(64, kernel_size = 2, padding = "valid", kernel_initializer = "he_uniform")(x)



avg_pool1 = GlobalAveragePooling1D()(x)

max_pool1 = GlobalMaxPooling1D()(x)     



x = concatenate([avg_pool1, max_pool1])



x = Dense(128, activation='relu')(x)

x = Dense(64, activation='relu')(x)

preds = Dense(5, activation='linear')(x)





model = Model(sequence_input, preds)

model.summary()

model.compile(loss='mean_absolute_error',

              optimizer=Adam(),

              metrics=['mean_absolute_error'])
BATCH_SIZE = 128

NUM_EPOCHS = 300
from keras import callbacks

file_path = "best_model_mypersonality.hdf5"

rlr = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1,patience=10, min_lr=1.1e-6, mode='auto', verbose=1)

check_point = callbacks.ModelCheckpoint(file_path, monitor = "val_loss", verbose = 1, save_best_only = True, mode = "min")
model.fit(

    train_text,

    train_labels,

    batch_size=BATCH_SIZE,

    epochs=NUM_EPOCHS,

    validation_data=(validate_text, validate_labels),

    callbacks = [EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=30), rlr, check_point])
