!pip install bert-for-tf2
import numpy as np
import pandas as pd
import re
import tensorflow as tf
from tensorflow_core.python.keras.layers import Dense, Input
#from tensorflow_core.python.keras.optimizers import Adam
from tensorflow.keras.optimizers import Adam
from tensorflow_core.python.keras.models import Model
from tensorflow_core.python.keras.callbacks import ModelCheckpoint
import tensorflow_hub as hub
from bert.tokenization import bert_tokenization
import math
import seaborn as sns
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold

def clean_text(text):
    new_text = []
    for each in text.split():
        if each.isalpha():
            new_text.append(each)
         
    cleaned_text = ' '.join(new_text)
    cleaned_text = re.sub(r'https?:\/\/t.co\/[A-Za-z0-9]+','',cleaned_text)
    
    return cleaned_text
def bert_encode(texts, tokenizer, max_len =512):
    all_tokens = []
    all_masks = []
    all_segments = []

    for text in texts:
        text = tokenizer.tokenize(text)
        text = text[:max_len-2]
        input_sequence = ['[CLS]'] + text +['[SEP]']
        pad_len = max_len - len(input_sequence)
        tokens = tokenizer.convert_tokens_to_ids(input_sequence)
        tokens += [0]*pad_len
        pad_masks = [1]*len(input_sequence) + [0]*pad_len
        segment_ids = [0]*max_len
        
        all_tokens.append(tokens)
        all_masks.append(pad_masks)
        all_segments.append(segment_ids)

    return np.array(all_tokens), np.array(all_masks), np.array(all_segments)


def build_bert(bert_layer, max_len =512):
    adam = Adam(lr=6e-6)
    #main_input = Input(shape =(max_len,), dtype ='int32')
    input_word_ids = Input(shape = (max_len,),dtype ='int32')
    input_mask = Input(shape = (max_len,),dtype ='int32')
    segment_ids = Input(shape = (max_len,),dtype ='int32')

    pooled_output, sequence_output = bert_layer([input_word_ids, input_mask, segment_ids])
    clf_output = sequence_output[:,0,:]
    out = Dense(1, activation ='sigmoid')(clf_output)
    
    model = Model(inputs =[input_word_ids, input_mask, segment_ids], outputs =out)
    model.compile(optimizer=adam ,loss = 'binary_crossentropy', metrics =['accuracy'])

    return model


bert_layer = hub.KerasLayer('https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/2', trainable=True)
train_data = pd.read_csv('/kaggle/input/nlp-getting-started/train.csv')
test_data = pd.read_csv('/kaggle/input/nlp-getting-started/test.csv')
vocab_file = bert_layer.resolved_object.vocab_file.asset_path.numpy()
do_lower_case = bert_layer.resolved_object.do_lower_case.numpy()
tokenizer = bert_tokenization.FullTokenizer(vocab_file, do_lower_case)
train_data['text'] = train_data['text'].apply(lambda x: clean_text(x))
test_data['text'] = test_data['text'].apply(lambda x: clean_text(x))
#train_text = list(train_data['text'])
#test_text = list(test_data['text'])
#train_labels = np.array(train_data['target'])

all_models = []
all_loss = []
skf = StratifiedKFold(n_splits = 5, random_state =42, shuffle=True)
for fold,(train_idx,val_idx) in enumerate(skf.split(train_data['text'],train_data['target'])):
    print('Fold:'+str(fold))
    train_input = bert_encode(train_data.loc[train_idx,'text'], tokenizer, max_len=100)
    train_labels = train_data.loc[train_idx,'target']
    valid_input = bert_encode(train_data.loc[val_idx,'text'], tokenizer, max_len=100)
    valid_labels = train_data.loc[val_idx,'target']
    
    model = build_bert(bert_layer, max_len=100)
    model.fit(train_input, train_labels,epochs =3, batch_size = 16)
    valid_loss, valid_acc = model.evaluate(valid_input,valid_labels, batch_size =16)
    all_models.append(model)
    all_loss.append(valid_loss)
    
#train_input = bert_encode(train_text, tokenizer, max_len=100)
#test_input = bert_encode(test_text, tokenizer, max_len=100)
#train_labels = np.array(train_data['target'])
#model = build_bert(bert_layer, max_len=100)
#model.summary()
#model.fit(train_input, train_labels, epochs =3, batch_size = 16, validation_split=0.2)
#results = model.predict(test_input).round().astype('int')
#train_pred = model.predict(train_input).round().astype('int')
#submission_data = pd.read_csv('/kaggle/input/nlp-getting-started/sample_submission.csv')
#submission_data['target'] = results
#submission_data.to_csv('/kaggle/working/submission.csv',index=False, header=True)

test_text = list(test_data['text'])
test_input = bert_encode(test_text, tokenizer, max_len=100)
#results = np.zeros((test_input[0].shape[0],1))
min_loss_index = all_loss.index(min(all_loss))
results = all_models[min_loss_index].predict(test_input)
submission_data = pd.read_csv('/kaggle/input/nlp-getting-started/sample_submission.csv')
submission_data['target'] = results.round().astype('int')
#for index,model in enumerate(all_models):
    
    #results += model.predict(test_input)/len(all_models)
    
#submission_data = pd.read_csv('/kaggle/input/nlp-getting-started/sample_submission.csv')
#submission_data['target'] = np.round(results).astype('int')
submission_data.to_csv('/kaggle/working/submission.csv',index=False, header=True)

def plot_confusion_matrix(y_true,y_pred):
    cm = confusion_matrix(y_true, y_pred, labels = np.unique(y_true))
    cm_sum = np.sum(cm, axis = 1, keepdims=True)
    cm_percent = cm/cm_sum.astype('float')*100
    annot = np.empty_like(cm).astype(str)
    rows, columns = np.shape(cm)
    
    for i in range(rows):
        for j in range(columns):
            c = cm[i,j]
            p = cm_percent[i,j]      
            if c == 0:
                annot[i, j] = ''
            else:
                s = cm_sum[i]
                annot[i, j] = '%.1f%%\n%d/%d' % (p, c, s)
    
    #annot = pd.DataFrame(annot)
    cm = pd.DataFrame(cm, index = np.unique(y_true),columns = np.unique(y_true))
    cm.index.name = 'Actual'
    cm.columns.name = 'Predicted'
    fig,ax = plt.subplots(figsize=(8,8))
    plt.title('Confusion Matrix for Bert model')
    sns.heatmap(cm, annot=annot, ax=ax,fmt='' )


train_text = list(train_data['text'])
train_labels = list(train_data['target'])
all_train_input = bert_encode(train_text, tokenizer, max_len =100)
train_pred = np.zeros((all_train_input[0].shape[0],1))
for model in all_models:
    train_pred += model.predict(all_train_input)/len(all_models)
    
train_pred = np.round(train_pred).astype('int')

plot_confusion_matrix(train_labels,train_pred)
