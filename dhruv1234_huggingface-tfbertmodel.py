!pip install transformers

import transformers
import pandas as pd

train = pd.read_csv('/kaggle/input/nlp-getting-started/train.csv')

test = pd.read_csv('/kaggle/input/nlp-getting-started/test.csv')
import matplotlib.pyplot as plt

plt.title('Train Data')

plt.xlabel('Target Distribution')

plt.ylabel('Samples')

plt.hist(train.target)

plt.show()


# def decontracted(phrase):

#     # specific

#     phrase = re.sub(r"won\'t", "will not", phrase)

#     phrase = re.sub(r"can\'t", "can not", phrase)



#     # general

#     phrase = re.sub(r"n\'t", " not", phrase)

#     phrase = re.sub(r"\'re", " are", phrase)

#     phrase = re.sub(r"\'s", " is", phrase)

#     phrase = re.sub(r"\'d", " would", phrase)

#     phrase = re.sub(r"\'ll", " will", phrase)

#     phrase = re.sub(r"\'t", " not", phrase)

#     phrase = re.sub(r"\'ve", " have", phrase)

#     phrase = re.sub(r"\'m", " am", phrase)

#     return phrase
# import spacy

# import re

# nlp = spacy.load('en')

# def preprocessing(text):

#   text = text.replace('#','')

#   text = decontracted(text)

#   text = re.sub('\S*@\S*\s?','',text)

#   text = re.sub('http[s]?:(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+','',text)



#   token=[]

#   result=''

#   text = re.sub('[^A-z]', ' ',text.lower())

  

#   text = nlp(text)

#   for t in text:

#     if not t.is_stop and len(t)>2:  

#       token.append(t.lemma_)

#   result = ' '.join([i for i in token])



#   return result.strip()
# train.text = train.text.apply(lambda x : preprocessing(x))

# test.text = test.text.apply(lambda x : preprocessing(x))
from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained('bert-large-uncased', do_lower_case=True)
import numpy as np

import tensorflow as tf 
def bert_encode(data,maximum_length) :

  input_ids = []

  attention_masks = []

  



  for i in range(len(data.text)):

      encoded = tokenizer.encode_plus(

        

        data.text[i],

        add_special_tokens=True,

        max_length=maximum_length,

        pad_to_max_length=True,

        

        return_attention_mask=True,

        

      )

      

      input_ids.append(encoded['input_ids'])

      attention_masks.append(encoded['attention_mask'])

  return np.array(input_ids),np.array(attention_masks)
train_input_ids,train_attention_masks = bert_encode(train,60)

test_input_ids,test_attention_masks = bert_encode(test,60)
import tensorflow as tf

from tensorflow.keras.optimizers import Adam

def create_model(bert_model):

  input_ids = tf.keras.Input(shape=(60,),dtype='int32')

  attention_masks = tf.keras.Input(shape=(60,),dtype='int32')

  

  output = bert_model([input_ids,attention_masks])

  output = output[1]

  output = tf.keras.layers.Dense(32,activation='relu')(output)

  output = tf.keras.layers.Dropout(0.2)(output)



  output = tf.keras.layers.Dense(1,activation='sigmoid')(output)

  model = tf.keras.models.Model(inputs = [input_ids,attention_masks],outputs = output)

  model.compile(Adam(lr=6e-6), loss='binary_crossentropy', metrics=['accuracy'])

  return model



from transformers import TFBertModel

bert_model = TFBertModel.from_pretrained('bert-large-uncased')
model = create_model(bert_model)

model.summary()
# filepath="weights-improvement-{epoch:02d}-{val_accuracy:.2f}.hdf5"

# checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')

# callbacks_list = [checkpoint]

# add callbacks = callbacks_list to model.fit
history = model.fit([train_input_ids,train_attention_masks],train.target,validation_split=0.2, epochs=2,batch_size=10)
result = model.predict([test_input_ids,test_attention_masks])

result = np.round(result).astype(int)
result = pd.DataFrame(result)

submission = pd.read_csv('/kaggle/input/nlp-getting-started/sample_submission.csv')

output = pd.DataFrame({'id':submission.id,'target':result[0]})

output.to_csv('submission.csv',index=False)
plt.plot(history.history['accuracy'])

plt.plot(history.history['val_accuracy'])

plt.title('model accuracy')

plt.ylabel('accuracy')

plt.xlabel('epoch')

plt.legend(['train', 'test'], loc='upper left')

plt.show()
plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])

plt.title('model loss')

plt.ylabel('loss')

plt.xlabel('epoch')

plt.legend(['train', 'test'], loc='upper left')

plt.show()