import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import tensorflow as tf

from tensorflow.keras.layers import Dense, Input, Convolution1D, Flatten

from tensorflow.keras.optimizers import Adam

from tensorflow.keras.models import Model

import tensorflow_hub as hub

#import keras.backend as K

from tensorflow.keras.utils import to_categorical

import re



from shutil import copyfile



# copy our file into the working directory (make sure it has .py suffix)

copyfile(src = "../input/bert-tokenization/bert_tokenization.py", dst = "../working/bert_tokenization.py")



# import all our functions

import bert_tokenization

#import nltk

#nltk.download('stopwords')

#nltk.download('punkt')

#from nltk.corpus import stopwords

#from nltk.stem.porter import PorterStemmer
train_data = pd.read_csv('/kaggle/input/tweet-sentiment-extraction/train.csv')

test_data = pd.read_csv('/kaggle/input/tweet-sentiment-extraction/test.csv')

#print(train_data.head())

print(train_data.isnull().sum())

print(test_data.isnull().sum())

train_data = train_data.dropna()

print(train_data.isnull().sum())

colors = sns.color_palette()

plt.subplot(211)

sentiment_num_1 = train_data['sentiment'].value_counts()

sentiment_num_1.plot(kind='bar',figsize=(10,10),color=colors[:3],rot=0)

plt.title('Sentiment Distribution for Train Data')



plt.subplot(212)

sentiment_num_2 = test_data['sentiment'].value_counts()

sentiment_num_2.plot(kind='bar',figsize=(10,10),color=colors[:3],rot=0)

plt.title('Sentiment Distribution for Test Data')



plt.tight_layout(pad =3)

plt.show()

def bert_encode_train(texts,sel_texts,tokenizer, max_len =512):

    all_tokens = []

    all_masks = []

    all_segments = []

    all_start_tokens = []

    all_end_tokens = []

    

    

    for i in range(len(texts)):

        

        start_idx = texts[i].find(sel_texts[i])

        end_idx = start_idx + len(sel_texts[i])-1

        

        ##divide the full text into three parts and tokenize seperately:

        ##texts before selected texts, selected texts and texts after selected texts

        

        full_text_1 = tokenizer.tokenize(texts[i][:start_idx])

        full_text_2 = tokenizer.tokenize(texts[i][start_idx:end_idx+1])

        full_text_3 = tokenizer.tokenize(texts[i][end_idx+1:])

        

        ##input header and Sep symbol

        input_sequence = ['[CLS]'] + full_text_1+ full_text_2 + full_text_3 +['[SEP]']

        pad_len = max_len - len(input_sequence)

        tokens = tokenizer.convert_tokens_to_ids(input_sequence)

        tokens += [0]*pad_len

        pad_masks = [1]*len(input_sequence) + [0]*pad_len

        segment_ids = [0]*max_len  

        start_tokens = [0]*(len(full_text_1)+1)+[1]+[0]*(max_len-len(full_text_1)-2)

        end_tokens = [0]*(len(full_text_1)+len(full_text_2))+[1]+[0]*(max_len-len(full_text_1)-len(full_text_2)-1)

        

        all_tokens.append(tokens)

        all_masks.append(pad_masks)

        all_segments.append(segment_ids)

        all_start_tokens.append(start_tokens) 

        all_end_tokens.append(end_tokens)

        

    return np.array(all_tokens), np.array(all_masks), np.array(all_segments),np.array(all_start_tokens),np.array(all_end_tokens)

        

    



    

def bert_encode_test(texts, tokenizer, max_len=512):

    all_tokens = []

    all_masks = []

    all_segments = []



    for text in texts:

        text = tokenizer.tokenize(text)

        #text = text[:max_len-2]

        ##input header and Sep symbol

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





#bert_layer = hub.KerasLayer('https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/2', trainable=True)

bert_layer = hub.KerasLayer('../input/berthub', trainable=True)

vocab_file = bert_layer.resolved_object.vocab_file.asset_path.numpy()

do_lower_case = bert_layer.resolved_object.do_lower_case.numpy()

tokenizer = bert_tokenization.FullTokenizer(vocab_file, do_lower_case)



train_texts = train_data[train_data['sentiment']!='neutral']['text']

train_sel_texts = train_data[train_data['sentiment']!='neutral']['selected_text']

train_sentiment = train_data[train_data['sentiment']!='neutral']['sentiment']

full_texts = list(train_texts + train_sentiment)

#print(train_sel_texts.shape[0])

sel_texts = list(train_sel_texts)



#train_data['full_text'] = train_data['sentiment'] + train_data['text']

#full_texts = list(train_data['full_text'])

#sel_texts = list(train_data['selected_text'])

train_input = bert_encode_train(full_texts,sel_texts,tokenizer, max_len =100)[:3]

train_labels = bert_encode_train(full_texts,sel_texts,tokenizer, max_len =100)[3:]



test_texts = test_data[test_data['sentiment']!='neutral']['text']

test_sentiment = test_data[test_data['sentiment']!='neutral']['sentiment']

full_texts_test = list(test_texts + test_sentiment)

#test_data['full_text'] = test_data['sentiment'] + test_data['text']

#full_texts_test = list(test_data['full_text'])



test_input = bert_encode_test(full_texts_test, tokenizer, max_len =100)

#print(all_start_tokens[:5])

def build_bert(bert_layer, max_len =512):

    adam = Adam(lr=5e-6)

    #main_input = Input(shape =(max_len,), dtype ='int32')

    input_word_ids = Input(shape = (max_len,),dtype ='int32')

    input_mask = Input(shape = (max_len,),dtype ='int32')

    segment_ids = Input(shape = (max_len,),dtype ='int32')



    pooled_output, sequence_output = bert_layer([input_word_ids, input_mask, segment_ids])

    clf_output = sequence_output[:,0,:]

    out1 = Dense(100, activation ='softmax')(clf_output)

    out2 = Dense(100, activation ='softmax')(clf_output)

    

    #out = Dense(1, activation ='sigmoid')(clf_output)

    

    model = Model(inputs =[input_word_ids, input_mask, segment_ids], outputs =[out1,out2])

    model.compile(optimizer=adam ,loss = 'categorical_crossentropy', metrics =['accuracy'])

    #model.summary()

    return model

    

model = build_bert(bert_layer,max_len=100)

model.fit(train_input, train_labels, epochs =5, batch_size = 16, validation_split=0.2)
pred_start,pred_end = model.predict(test_input)

results = []

for k in range(test_input[0].shape[0]):

    a = np.argmax(pred_start[k,])

    b = np.argmax(pred_end[k,])

    

    if a>b:

        sel_text = test_data.loc[k,'text']

    else:

        sel_text = ' '.join(tokenizer.convert_ids_to_tokens(test_input[0][k,a:b+1]))

        

    results.append(sel_text)



#google fulltokenizer will generate meaingless punction ##   

results = [re.sub('[##]','',x) for x in results]

    

for k in range(test_data.shape[0]):

    if test_data.loc[k, 'sentiment'] == 'neutral':

        test_data.loc[k, 'selected_text'] = test_data.loc[k, 'text']



test_data.loc[test_data['sentiment']!='neutral','selected_text'] = results



output = pd.read_csv('/kaggle/input/tweet-sentiment-extraction/sample_submission.csv')

output['selected_text'] = test_data['selected_text']

print(output.head(10))

#output.to_csv('submission.csv',index=False,header=True)
output.to_csv('submission.csv',index=False,header=True)
bert_layer = hub.KerasLayer('../input/nlp-getting-started/sample_submission.csv', trainable=True)

vocab_file = bert_layer.resolved_object.vocab_file.asset_path.numpy()

do_lower_case = bert_layer.resolved_object.do_lower_case.numpy()

tokenizer = bert_tokenization.BasicTokenizer(do_lower_case)

a=tokenizer.tokenize('*sigh* Off 2 bed 2 try 2 get these crummy 2 hrs of sleep b4 my horrid 12 hour day..smh. Niterzzz evry1. Don`t let the twitterbugz bite..')

tokenizer2 = bert_tokenization.FullTokenizer(vocab_file,do_lower_case)

b=tokenizer2.tokenize('yï¿½n tï¿½m, sang n?m s? th?y **** m?c Tr?n tinh twitter tr? l?i Lï¿½ Thï¿½ng , Th?ch Sanh nh? lï¿½m quiz mï¿½ c??i ???c cï¿½ng chï¿½a","yï¿½n tï¿½m, sang n?m s? th?y **** m?c Tr?n tinh twitter tr? l?i Lï¿½ Thï¿½ng , Th?ch Sanh nh? lï¿½m quiz mï¿½ c??i ???c cï¿½ng chï¿½a')

print(a,b,len(b))