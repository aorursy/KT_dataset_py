!pip install transformers
import tensorflow as tf
import numpy as np
from transformers import *
import tokenizers
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
print(tf.__version__)
vocab_file = '../input/tf-roberta/vocab-roberta-base.json'
merge_file = '../input/tf-roberta/merges-roberta-base.txt'
tokenizer = tokenizers.ByteLevelBPETokenizer(vocab_file, merge_file,lowercase = True)
MAX_LEN = 100
train = pd.read_csv('../input/nlp-getting-started/train.csv').fillna('')
train.sample(25)
ct = train.shape[0]
input_ids = np.ones((ct,MAX_LEN),dtype='int32')
attention_mask = np.zeros((ct,MAX_LEN),dtype='int32')
token_type_ids = np.zeros((ct,MAX_LEN),dtype='int32')
for k in range(ct):
    keyword = train.loc[k ,'keyword']
    keyword = " ".join(keyword.split())
    text1 = train.loc[k ,'text']
    text1 = " " + " ".join(text1.split())
    text1_enc = tokenizer.encode(text1)
    keyword_enc = tokenizer.encode(keyword)
    enc = [0] + text1_enc.ids + [2 ,2] + keyword_enc.ids + [2]
    input_ids[k ,:len(enc)] = enc
    attention_mask[k ,:len(enc)] = 1
    if k <=3:
        print('#################################################')
        print('text : {}'.format(text1))
        print('encoding : {}'.format(text1_enc.ids))
        print('tokens : {}'.format(text1_enc.tokens))
        print('keyword : {}'.format(keyword))
        print('keyword_enc : {}'.format(keyword_enc.ids))
        print('keyword_tokens : {}'.format(keyword_enc.tokens))
test = pd.read_csv('../input/nlp-getting-started/test.csv').fillna('')
ct_t = test.shape[0]
input_ids_t = np.ones((ct_t,MAX_LEN),dtype='int32')
attention_mask_t = np.zeros((ct_t,MAX_LEN),dtype='int32')
token_type_ids_t = np.zeros((ct_t,MAX_LEN),dtype='int32')
for k in range(ct_t):
    keyword = test.loc[k ,'keyword']
    keyword = " ".join(keyword.split())
    text1 = test.loc[k ,'text']
    text1 = " " + " ".join(text1.split())
    text1_enc = tokenizer.encode(text1)
    keyword_enc = tokenizer.encode(keyword)
    enc = [0] + text1_enc.ids + [2 ,2] + keyword_enc.ids + [2]
    input_ids_t[k ,:len(enc)] = enc
    attention_mask_t[k ,:len(enc)] = 1
    if k <=3:
        print('#################################################')
        print('text : {}'.format(text1))
        print('encoding : {}'.format(text1_enc.ids))
        print('tokens : {}'.format(text1_enc.tokens))
        print('keyword : {}'.format(keyword))
        print('keyword_enc : {}'.format(keyword_enc.ids))
        print('keyword_tokens : {}'.format(keyword_enc.tokens))
outputs = []
for k in range(ct):
    sent = train.loc[k ,'target']
    #checking for any labels other than 0 or1
    if sent != 0 and sent != 1:
        print(sent ,k)
    if k<5:
        print('{} is {} type'.format(sent ,type(sent)))
    outputs.append(sent)
outputs = np.asarray(outputs)
outputs = outputs.astype('float32')
outputs = outputs.reshape(-1)
#just to check whether everything is going fine 
print(type(outputs) ,outputs.shape)
zeros = 0
ones = 0
for i in range(ct):
    if outputs[i] == 0:
        zeros+=1
    else:
        ones+=1
print(zeros ,ones)

ids = tf.keras.layers.Input((MAX_LEN,), dtype=tf.int32)
att = tf.keras.layers.Input((MAX_LEN,), dtype=tf.int32)
tok = tf.keras.layers.Input((MAX_LEN,), dtype=tf.int32)
config = RobertaConfig.from_pretrained('../input/tf-roberta/config-roberta-base.json')
bert_model = TFRobertaModel.from_pretrained('../input/tf-roberta/pretrained-roberta-base.h5',config=config)
x = bert_model(ids,attention_mask=att,token_type_ids=tok)
drop1 = tf.keras.layers.Dropout(0.2)(x[0])
layer2 = tf.keras.layers.Conv1D(1 ,kernel_size = 1)(drop1)
layer3 = tf.keras.layers.Flatten()(layer2)
layer4 = tf.keras.layers.Activation('elu')(layer3)
output = tf.keras.layers.Dense(1 ,activation = 'sigmoid')(layer4)
model = tf.keras.Model(inputs = [ids ,att ,tok] ,outputs = [output])
model.summary()
def my_loss(gamma):
    '''defining focal loss with parameter gamma'''
    def focal_loss(y_true ,y_pred):
        y_pred =tf.keras.backend.clip(y_pred ,1e-6 ,1-(1e-6))
        log_yp = tf.keras.backend.log(y_pred)
        log_yp_ = tf.keras.backend.log(1-y_pred)
        loss = ((1-y_pred)**gamma)*y_true*log_yp + (y_pred**gamma)*(1-y_true)*log_yp_
        return -tf.keras.backend.sum(loss)
    return focal_loss
adam = tf.keras.optimizers.Adam(lr = 0.000001)
model.compile(optimizer = adam ,loss = my_loss(1.5) ,metrics = ['acc'])
history = model.fit([input_ids[800:] ,attention_mask[800:] ,token_type_ids[800:]] ,
                    outputs[800:] ,
                   epochs = 45 ,
                   batch_size = 32 ,
                   validation_data = ([input_ids[:200] ,attention_mask[:200] ,token_type_ids[:200]] ,outputs[:200]) ,
                   verbose = 1)
# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model acc')
plt.ylabel('acc')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()
pred = model.predict([input_ids[:800] ,attention_mask[:800] ,token_type_ids[:800]] ,verbose = 1)
y_true = outputs[:800]
print(y_true.shape)
y_pred = np.zeros_like(y_true)
for i in range(pred.shape[0]):
    if pred[i] >= 0.5:
        y_pred[i] = 1
accuracy = accuracy_score(y_true, y_pred)
print('Accuracy: %f' % accuracy)
precision = precision_score(y_true ,y_pred)
print('Precision: %f' % precision)
recall = recall_score(y_true ,y_pred)
print('Recall: %f' % recall)
f1 = f1_score(y_true ,y_pred)
print('F1 score: %f' % f1)
auc = roc_auc_score(y_true ,y_pred)
print('ROC AUC: %f' % auc)
matrix = confusion_matrix(y_true ,y_pred)
print('condusion matrix:{}'.format(matrix))
plt.imshow(matrix ,cmap = 'gray')
test_pred = model.predict([input_ids_t ,attention_mask_t ,token_type_ids_t] ,verbose = 1)
test_pred = test_pred.reshape(-1)
all = []
for i in range(test_pred.shape[0]):
    if test_pred[i] >= 0.5:
        all.append(1)
    else:
        all.append(0)
test['target'] = all
test[['id' ,'target']].to_csv('sample_submission.csv' ,index = False)
test.sample(25)
