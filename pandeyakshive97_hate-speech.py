import numpy as np

glove_dir = '../input/glove6b100dtxt/glove.6B.100d.txt'
embedding_index = {}

f = open(glove_dir)

for line in f:

    values = line.split()

    word = values[0]

    coef = np.asarray(values[1:], dtype='float32')

    embedding_index[word] = coef

f.close()

print(len(embedding_index))
data_dir = '../input/hate-speech-dataset/hate_speech.csv'
import pandas as pd

data = pd.read_csv(data_dir)
texts = np.array(data['post'])

labels = np.array(data['label'])
from keras.preprocessing.text import Tokenizer

from keras.preprocessing.sequence import pad_sequences



maxlen = 100

samples = texts.shape[0]

tokenizer = Tokenizer()

tokenizer.fit_on_texts(texts)

sequences = tokenizer.texts_to_sequences(texts)

word_index = tokenizer.word_index

print(sequences)
data = pad_sequences(sequences, maxlen)

print(data.shape)

print(labels.shape)
x_train = data[:7925]

y_train = labels[:7925]



x_val = data[7925:8925]

y_val = labels[7925:8925]



x_test = data[8925:]

y_test = labels[8925:]



print(x_train.shape)

print(x_val.shape)

print(x_test.shape)
embedding_dim = 100

embedding_matrix = np.zeros((1+len(word_index), embedding_dim))



for word, i in word_index.items():

    embedding_vector = embedding_index.get(word)

    if embedding_vector is not None:

        embedding_matrix[i] = embedding_vector
from keras.models import Sequential

from keras.layers import Embedding, Flatten, Dense,CuDNNLSTM



model = Sequential()

model.add(Embedding(1+len(word_index), embedding_dim, input_length=maxlen))

model.add(CuDNNLSTM(maxlen))

model.add(Dense(32, activation='relu'))

model.add(Dense(1, activation='sigmoid'))

model.summary()

model.layers[0].set_weights([embedding_matrix])

model.layers[0].trainable = False



print(x_train.shape)

print(y_train.shape)
model.compile(optimizer='rmsprop',loss='binary_crossentropy',metrics=['acc'])

history = model.fit(x_train,y_train,epochs=10,batch_size=32,validation_data=(x_val,y_val))
def cnfmatrix(y_test,results):

    fp = 0.0

    fn = 0.0

    tp = 0.0

    tn = 0.0

    t = 0.0

    n = 0.0

    results.shape

    for i in range(results.shape[0]):

        if y_test[i]==1 and results[i]==1:

            tp+=1

            t+=1

        elif y_test[i]==1 and results[i]==0:

            fn+=1

            t+=1

        elif y_test[i]==0 and results[i]==1:

            fp+=1

            n+=1

        elif y_test[i]==0 and results[i]==0:

            tn+=1

            n+=1

    print(tp/results.shape[0],fp/results.shape[0])

    print(fn/results.shape[0],tn/results.shape[0])

    Precision  = tp/(tp+fp)

    Recall = tp/(tp+fn)

    print("Precision: ",Precision,"Recall: ",Recall)

    f1score = (2*Precision*Recall)/(Precision+Recall)

    print("f1score: ",f1score)

    print("accuracy: ",(tp+tn)/results.shape[0])

    print("hate_acc: ", (tp)/t)

    print("non_hate_acc: ", (tn)/n)
predictions = model.predict(x_test)
results = []

for prediction in predictions:

    if prediction < 0.5:

        results.append(0)

    else:

        results.append(1)

        

results = np.array(results)
print(results)
cnfmatrix(y_test, results)
from keras.wrappers.scikit_learn import KerasRegressor,KerasClassifier

from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier



def the_model():

    model = Sequential()

    model.add(Embedding(1+len(word_index), embedding_dim, input_length=maxlen))

    model.add(CuDNNLSTM(maxlen))

    model.add(Dense(32, activation='relu'))

    model.add(Dense(1, activation='sigmoid'))

    model.compile(optimizer='rmsprop',loss='binary_crossentropy',metrics=['acc'])

    return model



ann_estimator = KerasClassifier(build_fn= the_model, epochs=20, batch_size=None, verbose=True)



boosted_ann = AdaBoostClassifier(base_estimator=ann_estimator, n_estimators = 5)



boosted_ann.fit(x_train, y_train)
predictions = boosted_ann.predict_proba(x_test)

results  = [int(i[0] <  i[1]) for i in predictions]

cnfmatrix(y_test, np.array(results))