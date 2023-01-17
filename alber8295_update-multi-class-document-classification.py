# Imports
# Basic
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os, random, math
from sklearn.model_selection import train_test_split

# DL
import tensorflow as tf
import keras
import keras.backend as K
from keras.preprocessing.sequence import pad_sequences
from keras import Model
from keras.layers import Input, Embedding, Dropout, SpatialDropout1D, GlobalAveragePooling1D
from keras.layers import GlobalMaxPooling1D, Bidirectional, GRU, CuDNNGRU, Activation, Dense
from keras.layers import Dot, Reshape, TimeDistributed, concatenate, BatchNormalization
from keras.optimizers import Adam

# Visualization
import matplotlib.pyplot as plt
from IPython.core.display import display, HTML
import seaborn as sns
sns.set()

# GLOBAL VARIABLES
EMB_SIZE = 300
## Utils functions

def prepare_data():
    dataset = pd.read_csv("../input/emotion.data")
    input_sentences = [text.split(" ") for text in dataset["text"].values.tolist()]
    labels = dataset["emotions"].values.tolist()
        
    # Initialize word2id and label2id dictionaries that will be used to encode words and labels
    word2id = dict()
    label2id = dict()

    max_words = 0 # maximum number of words in a sentence

    # Construction of word2id dict
    for sentence in input_sentences:
        for word in sentence:
            # Add words to word2id dict if not exist
            if word not in word2id:
                word2id[word] = len(word2id)
        # If length of the sentence is greater than max_words, update max_words
        if len(sentence) > max_words:
            max_words = len(sentence) # Number of words is set to the longest text

    
    # Construction of label2id, id2label and id2word dicts
    label2id = {l: i for i, l in enumerate(set(labels))}
    id2label = {v: k for k, v in label2id.items()}
    id2word = {v: k for k, v in word2id.items()}
    
    # Encode input words and labels
    X = [[word2id[word] for word in sentence] for sentence in input_sentences]
    Y = [label2id[label] for label in labels]

    # Apply Padding to X
    X = pad_sequences(X, max_words)

    # Convert Y to numpy array
    Y = keras.utils.to_categorical(Y, num_classes=len(label2id), dtype='float32')

    # Print shapes
    print("Shape of X: {}".format(X.shape))
    print("Shape of Y: {}".format(Y.shape))
    
    X_tra, X_te, Y_tra, Y_te = train_test_split(X,Y, stratify=Y, test_size = 0.3)
    
    return X_tra, X_te, Y_tra, Y_te,word2id,id2word,label2id,id2label

def make_plot(data, metric="loss"):
    # Data for plotting
    data1 = data[0]
    data2 = data[1]
    t = np.arange(1,len(data1)+1,1)
    plt.figure(figsize=(10,5))
    plt.plot(t, data1)
    plt.plot(t, data2)
    plt.xlabel('epoch')
    plt.ylabel(metric)
    plt.title('Train vs Val ' + metric)
    plt.grid()
    plt.legend(['train','val'], ncol=2, loc='upper right');
    plt.savefig("train_val_"+metric+".png", dpi=300)
    plt.show()

def focal_loss(target, input):
    gamma = 2.
    input = tf.cast(input, tf.float32)

    max_val = K.clip(-input, 0, 1)
    loss = input - input * target + max_val + K.log(K.exp(-max_val) + K.exp(-input - max_val))
    invprobs = tf.log_sigmoid(-input * (target * 2.0 - 1.0))
    loss = K.exp(invprobs * gamma) * loss

    return K.mean(K.sum(loss, axis=1))

def rgb_to_hex(rgb):
    return '#%02x%02x%02x' % rgb
    
def attention2color(attention_score):
    r = 255 - int(attention_score * 255)
    color = rgb_to_hex((255, r, r))
    return str(color)

def visualize_attention():
    # Make new model for output predictions and attentions
    model_att = keras.Model(inputs=model.input, \
                            outputs=[model.output, model.get_layer('attention_vec').output])
    idx = np.random.randint(low = 0, high=X_te.shape[0]) # Get a random test
    tokenized_sample = np.trim_zeros(X_te[idx]) # Get the tokenized text
    label_probs, attentions = model_att.predict(X_te[idx:idx+1]) # Perform the prediction

    # Get decoded text and labels
    decoded_text = [id2word[word] for word in tokenized_sample] 
    label_probs = {id2label[_id]: prob for (label, _id), prob in zip(label2id.items(),label_probs[0])}

    # Get word attentions using attenion vector
    token_attention_dic = {}
    max_score = 0.0
    min_score = 0.0
    for token, attention_score in zip(decoded_text, attentions[0][-len(tokenized_sample):]):
        token_attention_dic[token] = math.sqrt(attention_score)

    # Build HTML String to viualize attentions
    html_text = "<hr><p style='font-size: large'><b>Text:  </b>"
    for token, attention in token_attention_dic.items():
        html_text += "<span style='background-color:{};'>{} <span> ".format(attention2color(attention),
                                                                            token)
    html_text += "</p>"
    # Display text enriched with attention scores 
    display(HTML(html_text))

    # PLOT EMOTION SCORES
    emotions = [label for label, _ in label_probs.items()]
    scores = [score for _, score in label_probs.items()]
    plt.figure(figsize=(5,2))
    plt.bar(np.arange(len(emotions)), scores, align='center', alpha=0.5, color=['black', 'red', 'green', 'blue', 'cyan', "purple"])
    plt.xticks(np.arange(len(emotions)), emotions)
    plt.ylabel('Scores')
    plt.show()
X_tra, X_te, Y_tra, Y_te,word2id,id2word,label2id,id2label = prepare_data()
def get_model():
    
    inp = Input(shape=(X_tra.shape[1],))
    x = Embedding(X_tra.shape[1], EMB_SIZE)(inp)
    x = SpatialDropout1D(0.2)(x)
    x = Bidirectional(CuDNNGRU(40, return_sequences=True))(x)
    avg_pool = GlobalAveragePooling1D()(x)
    max_pool = GlobalMaxPooling1D()(x)
    conc = concatenate([avg_pool, max_pool])

    dense = Dense(len(label2id))(conc)
    bn = BatchNormalization()(dense)
    outp = Activation("softmax")(bn)
    
    opt = Adam(1e-3)

    model = Model(inputs=inp, outputs=outp)
    model.compile(loss=focal_loss,
                  optimizer=opt,
                  metrics=['accuracy'])

    # Print model summary
    # model.summary()
    
    return model
model = get_model()
model.summary()
# Train model
hist = model.fit(X_tra, Y_tra, validation_data=(X_te, Y_te), epochs=3, batch_size=32, validation_split=0.3, shuffle=True)
val_loss = hist.history['val_loss'];val_acc = hist.history['val_acc']
loss = hist.history['loss'];acc = hist.history['acc']
make_plot([loss, val_loss]);make_plot([acc,val_acc], metric="acc")
def get_model():
    
    # Define input tensor
    sequence_input = Input(shape=(X_tra.shape[1],), dtype='int32')

    # Word embedding layer
    embedded_inputs = Embedding(len(word2id) + 1,
                                            EMB_SIZE,
                                            input_length=X_tra.shape[1])(sequence_input)

    # Apply dropout to prevent overfitting
    embedded_inputs = SpatialDropout1D(0.2)(embedded_inputs)

    # Apply Bidirectional LSTM over embedded inputs
    lstm_outs = Bidirectional(
        CuDNNGRU(40, return_sequences=True)
    )(embedded_inputs)

    # Apply dropout to LSTM outputs to prevent overfitting
    lstm_outs = Dropout(0.2)(lstm_outs)

    # Attention Mechanism - Generate attention vectors
    attention_vector = TimeDistributed(keras.layers.Dense(1))(lstm_outs)
    attention_vector = Reshape((X_tra.shape[1],))(attention_vector)
    attention_vector = Activation('softmax', name='attention_vec')(attention_vector)
    attention_output = Dot(axes=1)([lstm_outs, attention_vector])

    # Last layer: fully connected with softmax activation
    fc = Dense(EMB_SIZE, activation='relu')(attention_output)
    output = Dense(len(label2id), activation='softmax')(fc)

    # Finally building model
    model = Model(inputs=sequence_input, outputs=output)
    model.compile(loss=focal_loss, metrics=["accuracy"], optimizer='adam')

    # Print model summary
    # model.summary()
    
    return model
model = get_model()
model.summary()
# Train model
hist = model.fit(X_tra, Y_tra, validation_data=(X_te, Y_te), epochs=3, batch_size=32)
val_loss = hist.history['val_loss'];val_acc = hist.history['val_acc']
loss = hist.history['loss'];acc = hist.history['acc']
make_plot([loss, val_loss]);make_plot([acc, val_acc], metric="acc")
visualize_attention()
