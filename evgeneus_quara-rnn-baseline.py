import os

print(os.listdir("../input"))
import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from tqdm import tqdm



seed = 123
class DataLoader:

    

    def load(self, file_names):

        self.train_df = pd.read_csv(file_names['train'])

        self.test_df = pd.read_csv(file_names['test'])

        ## fill up the missing values

        self.train_df['question_text'].fillna('_na_', inplace=True)

        self.test_df['question_text'].fillna('_na_', inplace=True)

        print('Train shape : ', self.train_df.shape)

        print('Test shape : ', self.test_df.shape)
class Embeddings:

    

    def __init__(self):

        self.embed_len = 300

        self.punctuation = set('!#$%&()*+,.:;<>?@[\\]')

    

    ## load embeddings

    def load(self, embedding_file_name):

        self.embeddings_index = {}

        f = open(embedding_file_name)

        for line in tqdm(f):

            values = line.split(' ');

            word = values[0]

            coef = np.asarray(values[1:], dtype='float32')

            self.embeddings_index[word] = coef

        self.embed_len = len(coef) # length of embeddings

        f.close()

        

    def text_to_vec(self, text, max_text_len=30):

        if text[-1] in self.punctuation:

            text = text[:-1].split() + [text[-1]]

        else:

            text = text.split()

        text = text[:max_text_len]

        empyt_emb = np.zeros(self.embed_len)

        embeds = [self.embeddings_index.get(word, empyt_emb) for word in text]

        embeds+= [empyt_emb] * (max_text_len - len(embeds))

        

        return np.array(embeds)

    

    def sequences_to_vec(self, text_sequences, max_text_len=30):

        vectors = [self.text_to_vec(text, max_text_len) for text in text_sequences]

        

        return np.array(vectors)
# generator for training NNet (used via fit_generator method in keras)

def batch_gen(train_df, batch_size, emb, max_text_len=30):

    import math

    n_batches = math.ceil(len(train_df) / batch_size)

    while True: 

        train_df = train_df.sample(frac=1.)  # Shuffle the data.

        for i in range(n_batches):

            batch_df = train_df.iloc[i*batch_size:(i+1)*batch_size]

            X_ = emb.sequences_to_vec( batch_df["question_text"], max_text_len)

            y_ = batch_df["target"].values

            yield X_, y_

            

# generater to do predication on test data

def batch_gen_test(test_df, emb, batch_size_test=256, max_text_len=30):

    import math

    n_batches = math.ceil(len(test_df) / batch_size)

    for i in range(n_batches):

        batch_df = test_df.iloc[i*batch_size:(i+1)*batch_size]

        X_ = emb.sequences_to_vec( batch_df["question_text"], max_text_len)

        yield X_
import matplotlib.pyplot as plt

plt.style.use('ggplot')



def plot_history(history):

    acc = history.history['acc']

    val_acc = history.history['val_acc']

    loss = history.history['loss']

    val_loss = history.history['val_loss']

    x = range(1, len(acc) + 1)



    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)

    plt.plot(x, acc, 'b', label='Training acc')

    plt.plot(x, val_acc, 'r', label='Validation acc')

    plt.xlabel('Epoch')

    plt.ylabel('Accuracy')

    plt.title('Training and validation accuracy')

    plt.legend()

    

    plt.subplot(1, 2, 2)

    plt.plot(x, loss, 'b', label='Training loss')

    plt.plot(x, val_loss, 'r', label='Validation loss')

    plt.xlabel('Epoch')

    plt.ylabel('Loss')

    plt.title('Training and validation loss')

    plt.legend()
file_names = {'train': '../input/train.csv', 'test': '../input/test.csv'}



dl = DataLoader()

dl.load(file_names)
num_words_in_questions = [len(question.split()) for question in dl.train_df['question_text']]

plt.hist(num_words_in_questions,bins = np.arange(0,103,3))

plt.show()



max_text_len = 40 # according to the histogram below
embedding_file_name = '../input/embeddings/glove.840B.300d/glove.840B.300d.txt'

emb = Embeddings()

emb.load(embedding_file_name)

print('Embedding loaded')
from sklearn.model_selection import train_test_split



val_size = 0.03

train_df, val_df, _, _ = train_test_split(dl.train_df, dl.train_df['target'], 

                                          test_size=val_size, 

                                          stratify=dl.train_df['target'],

                                          shuffle=True, random_state=seed)

val_X = emb.sequences_to_vec(val_df["question_text"], max_text_len)

val_y = val_df['target'].values
from keras.models import Sequential

from keras.layers import CuDNNLSTM, Dense, Bidirectional
model = Sequential()

model.add(Bidirectional(CuDNNLSTM(64, return_sequences=True),

                        input_shape=(max_text_len, 300)))

model.add(Bidirectional(CuDNNLSTM(64)))

model.add(Dense(1, activation="sigmoid"))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
batch_size = 128

mg = batch_gen(train_df, batch_size, emb, max_text_len)

model.fit_generator(mg, epochs=20,

                    steps_per_epoch=1000,

                    validation_data=(val_X, val_y),

                    verbose=True)
plot_history(model.history)
from sklearn import metrics

predicted_val_prob = model.predict(val_X).flatten()

 

best_thr, best_f1 = 0., 0.    

for clf_thr in np.arange(0.1, 0.501, 0.01):

    clf_thr = np.round(clf_thr, 2)

    predicted_val_bin = (np.array(predicted_val_prob) > clf_thr).astype(np.int)

    f1_val = metrics.f1_score(val_y, predicted_val_bin)

    if best_f1 <= f1_val:

        best_f1, best_thr = f1_val, clf_thr

    print('F1 score at threshold {} is {:1.3f}'.format(clf_thr, f1_val))

print('Best classification threshold on validation set is {}, F1 is {:1.3f}'.format(best_thr, best_f1))
# predict probabilities

all_preds_prob = []

batch_size_test = 256

for x in tqdm(batch_gen_test(dl.test_df, emb, batch_size_test, max_text_len)):

    all_preds_prob.extend(model.predict(x).flatten())

    

all_preds_prob_bin = (np.array(all_preds_prob) > best_thr).astype(np.int)
submit_df = pd.DataFrame({"qid": dl.test_df["qid"], "prediction": all_preds_prob_bin})

submit_df.to_csv("submission.csv", index=False)