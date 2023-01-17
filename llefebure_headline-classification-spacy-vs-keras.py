import json
import keras.layers as layers
import numpy as np
import pandas as pd
import spacy
from gensim.corpora import Dictionary
from keras.models import Model
from keras.preprocessing import sequence
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from spacy.util import minibatch
nlp = spacy.load('en')
data = pd.read_json('../input/News_Category_Dataset.json', lines=True)
categories = data.groupby('category').size().sort_values(ascending=False)
categories
TOP_N_CATEGORIES = 7
data = data[data.category.apply(lambda x: x in categories.index[:TOP_N_CATEGORIES]) &\
            (data.headline.apply(len) > 0)]
data_train, data_test = train_test_split(data, test_size=.1, random_state=31)
# check if `textcat` is already in the pipe, add if not
if 'textcat' not in nlp.pipe_names:
    textcat = nlp.create_pipe('textcat')
    nlp.add_pipe(textcat, last=True)
else:
    textcat = nlp.get_pipe('textcat')

# add labels to the model    
for label in categories.index[:TOP_N_CATEGORIES]:
    textcat.add_label(label)

# preprocess training data
data_train_spacy = list(
    zip(data_train.headline,
        data_train.category.apply(
            lambda cat: {'cats': {c: float(c == cat)
                                  for c in categories.index[:TOP_N_CATEGORIES]}}))
)

# train the model
other_pipes = [pipe for pipe in nlp.pipe_names if pipe != 'textcat']
with nlp.disable_pipes(*other_pipes):
    optimizer = nlp.begin_training()
    for i in range(5):
        print('Epoch %d' % i)
        losses = {}
        batches = minibatch(data_train_spacy, size=128)
        for batch in batches:
            texts, annotations = zip(*batch)
            nlp.update(texts, annotations, sgd=optimizer, drop=0.2,
                       losses=losses)
        with textcat.model.use_params(optimizer.averages):
            docs = [nlp.tokenizer(h) for h in data_test.headline]
            test_pred = np.array(
                [sorted(doc.cats.items(), key=lambda x: -x[1])[0][0]
                 for doc in textcat.pipe(docs)])
            print('Test Acc: %.4f' %
                  (pd.Series(test_pred == data_test.category.values).sum() / data_test.shape[0]))
spacy_y_pred = [sorted(doc.cats.items(), key=lambda x: -x[1])[0][0]
                for doc in nlp.pipe(data_test.headline)]
print(classification_report(data_test.category, spacy_y_pred))
MAX_SEQUENCE_LEN = 20
UNK = 'UNK'
PAD = 'PAD'

def text_to_id_list(text, dictionary):
    return [dictionary.token2id.get(tok, dictionary.token2id.get(UNK))
            for tok in text_to_tokens(text)]

def texts_to_input(texts, dictionary):
    return sequence.pad_sequences(
        list(map(lambda x: text_to_id_list(x, dictionary), texts)), maxlen=MAX_SEQUENCE_LEN,
        padding='post', truncating='post', value=dictionary.token2id.get(PAD))

def text_to_tokens(text):
    return [tok.text.lower() for tok in nlp.tokenizer(text)
            if not (tok.is_punct or tok.is_quote)]

def build_dictionary(texts):
    d = Dictionary(text_to_tokens(t)for t in texts)
    d.filter_extremes(no_below=3, no_above=1)
    d.add_documents([[UNK, PAD]])
    d.compactify()
    return d
dictionary = build_dictionary(data.headline)
x_train = texts_to_input(data_train.headline, dictionary)
x_test = texts_to_input(data_test.headline, dictionary)
lb = LabelBinarizer()
lb.fit(categories.index[:TOP_N_CATEGORIES])
y_train = lb.transform(data_train.category)
y_test = lb.transform(data_test.category)
EMBEDDING_DIM = 50

inp = layers.Input(shape=(MAX_SEQUENCE_LEN,), dtype='float32')
emb = layers.Embedding(len(dictionary), EMBEDDING_DIM, input_length=MAX_SEQUENCE_LEN)(inp)
filters = []
for kernel_size in [2, 3, 4]:
    conv = layers.Conv1D(32, kernel_size, padding='same', activation='relu', strides=1)(emb)
    pooled = layers.MaxPooling1D(pool_size=MAX_SEQUENCE_LEN-kernel_size+1)(conv)
    filters.append(pooled)

stacked = layers.Concatenate()(filters)
flatten = layers.Flatten()(stacked)
drop = layers.Dropout(0.2)(flatten)
out = layers.Dense(7, activation='softmax')(drop)

model = Model(inputs=inp, outputs=out)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()
model.fit(x_train, y_train, epochs=5, batch_size=128, validation_data=(x_test, y_test))
y_test_pred = [lb.classes_[i] for i in np.argmax(model.predict(x_test), axis=1)]
print(classification_report(data_test.category, y_test_pred))