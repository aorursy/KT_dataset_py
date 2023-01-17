import warnings
warnings.filterwarnings("ignore")

import pickle
import numpy as np
import progressbar

from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.preprocessing import MultiLabelBinarizer

from keras.models import Sequential
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import SimpleRNN, GRU, LSTM
from keras.layers.core import Dense, Dropout
from keras.layers.wrappers import TimeDistributed
from keras.layers import Convolution1D, MaxPooling1D
from keras import optimizers
from keras.regularizers import l1_l2
with open('../input/atis.pkl/atis.pkl', 'rb') as f:
    train_set, valid_set, test_set, dicts = pickle.load(f, encoding='latin1')
train_x, train_ne, train_label = train_set
val_x, val_ne, val_label = valid_set
test_x, test_ne, test_label = test_set
w2idx, ne2idx, labels2idx = dicts['words2idx'], dicts['tables2idx'], dicts['labels2idx']

# Create index to word/label dicts
idx2w  = {w2idx[k]:k for k in w2idx}
idx2ne = {ne2idx[k]:k for k in ne2idx}
idx2la = {labels2idx[k]:k for k in labels2idx}
print(train_x[0])
print([idx2w[i] for i in train_x[0]])
print([idx2la[i] for i in train_label[0]])
words_val = [ list(map(lambda x: idx2w[x], w)) for w in val_x]
groundtruth_val = [ list(map(lambda x: idx2la[x], y)) for y in val_label]
words_train = [ list(map(lambda x: idx2w[x], w)) for w in train_x]
groundtruth_train = [ list(map(lambda x: idx2la[x], y)) for y in train_label]
n_classes = len(idx2la)
n_vocab = len(idx2w)
n_examples = len(words_train)
print('#labels: ', n_classes, '\t#distinct words: ', n_vocab, '\t#examples: ', n_examples)
model = Sequential()
model.add(Embedding(n_vocab,100))

## Essas duas camadas são opcionais. Daremos mais detalhes nos próximos laboratórios
#model.add(Convolution1D(64, 5, border_mode='same', activation='relu'))
#model.add(Dropout(0.1))

model.add(GRU(100, return_sequences=True, 
              kernel_regularizer=l1_l2(l1=0.0, l2=0.0), 
              recurrent_regularizer=l1_l2(l1=0.0, l2=0.0)
             ))

model.add(TimeDistributed(Dense(n_classes, activation='softmax')))
#optm = optimizers.SGD(lr=0.01, momentum=0.0, decay=0.0, nesterov=False)
#optm = optimizers.RMSprop(lr=0.01, rho=0.9, epsilon=None, decay=0.0)
optm = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
model.compile(loss='categorical_crossentropy', optimizer=optm)
ml = MultiLabelBinarizer(classes=list(idx2la.values())).fit(idx2la.values())

def conlleval( trues, preds ):
    trues = ml.transform(trues)
    preds = ml.transform(preds)
    return score(trues, preds, beta=1, average='macro' )

# Defina aqui o número de épocas
n_epochs = 12

train_f_scores = []
val_f_scores = []
best_val_f1 = 0
con_dict = {}

for i in range(n_epochs):
    print("\nEpoch {}".format(i))
    
    print("Training =>")
    train_pred_label = []
    avgLoss = 0
    
    bar = progressbar.ProgressBar(max_value=len(train_x))
    for n_batch, sent in bar(enumerate(train_x)):
        label = train_label[n_batch]
        label = np.eye(n_classes)[label][np.newaxis,:]
        sent = sent[np.newaxis,:]
        
        if sent.shape[1] > 1: #some bug in keras
            loss = model.train_on_batch(sent, label)
            avgLoss += loss

        pred = model.predict_on_batch(sent)
        pred = np.argmax(pred,-1)[0]
        train_pred_label.append(pred)

    avgLoss = avgLoss/n_batch
    
    predword_train = [ list(map(lambda x: idx2la[x], y)) for y in train_pred_label]
    con_dict['p'], con_dict['r'], con_dict['f1'], _ = conlleval(groundtruth_train, predword_train)
    train_f_scores.append(con_dict['f1'])
    
    print('Loss = {}, Precision = {}, Recall = {}, F1 = {}'.format(avgLoss, con_dict['p'], con_dict['r'], con_dict['f1']))
    
    print("\n\nValidating =>")
    
    val_pred_label = []
    avgLoss = 0
    
    bar = progressbar.ProgressBar(max_value=len(val_x))
    for n_batch, sent in bar(enumerate(val_x)):
        label = val_label[n_batch]
        label = np.eye(n_classes)[label][np.newaxis,:]
        sent = sent[np.newaxis,:]
        
        if sent.shape[1] > 1: #some bug in keras
            loss = model.test_on_batch(sent, label)
            avgLoss += loss

        pred = model.predict_on_batch(sent)
        pred = np.argmax(pred,-1)[0]
        val_pred_label.append(pred)

    avgLoss = avgLoss/n_batch
    
    predword_val = [ list(map(lambda x: idx2la[x], y)) for y in val_pred_label]
    con_dict['p'], con_dict['r'], con_dict['f1'], _ = conlleval(groundtruth_val, predword_val)
    val_f_scores.append(con_dict['f1'])
    
    print('Loss = {}, Precision = {}, Recall = {}, F1 = {}'.format(avgLoss, con_dict['p'], con_dict['r'], con_dict['f1']))

    if con_dict['f1'] > best_val_f1:
        best_val_f1 = con_dict['f1']
        open('model_architecture.json','w').write(model.to_json())
        model.save_weights('best_model_weights.h5',overwrite=True)
        print("Best validation F1 score = {}".format(best_val_f1))


print("Best epoch to stop = {}".format(val_f_scores.index(max(val_f_scores))))
print("Best validation F1 score = {}".format(best_val_f1))
idx = np.random.randint(len(words_val))
preds = [idx2la[i[0]] for i in np.argmax(model.predict(val_x[idx]), -1)]
for k in zip(words_val[idx],groundtruth_val[idx], preds):
    print('Word: %s\t\tLabel: %s\t\tPred: %s' % k)
