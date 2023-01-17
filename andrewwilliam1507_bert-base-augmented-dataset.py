import tensorflow as tf
import seaborn as sns

from matplotlib import pyplot
try:

    # TPU detection. No parameters necessary if TPU_NAME environment variable is

    # set: this is always the case on Kaggle.

    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()

    print('Running on TPU ', tpu.master())

except ValueError:

    tpu = None



if tpu:

    tf.config.experimental_connect_to_cluster(tpu)

    tf.tpu.experimental.initialize_tpu_system(tpu)

    strategy = tf.distribute.experimental.TPUStrategy(tpu)

else:

    # Default distribution strategy in Tensorflow. Works on CPU and single GPU.

    strategy = tf.distribute.get_strategy()



print("REPLICAS: ", strategy.num_replicas_in_sync)
import pandas as pd 

import numpy as np
df=pd.read_csv('../input/text-augmenter-echo/labeled_negative_review_dataset_single_class_augmented.csv')
#Function to check how many 1s there are in each column of the topics

def check_sum_in_columns(df):

  total=0

  for i in df.columns[1:7]:

    summ=np.sum(df[i])

    total+=summ

 

    print(f"Sum in column {i}: {summ} ")

  total=int(total/6)





check_sum_in_columns(df)
import seaborn as sns
def deEmojify(text):

    regrex_pattern = re.compile(pattern = "["

        u"\U0001F600-\U0001F64F"  # emoticons

        u"\U0001F300-\U0001F5FF"  # symbols & pictographs

        u"\U0001F680-\U0001F6FF"  # transport & map symbols

        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)

                           "]+", flags = re.UNICODE)

    text=regrex_pattern.sub(r'',text)

    text=text.replace('\n',' ')

    text=re.sub(' +', ' ', text)

    

    return text
contractions = { 

"ain't": "am not / are not / is not / has not / have not",

"aren't": "are not / am not",

"can't": "cannot",

"can't've": "cannot have",

"cause": "because",

"could've": "could have",

"couldn't": "could not",

"couldn't've": "could not have",

"didn't": "did not",

"doesn't": "does not",

"don't": "do not",

"hadn't": "had not",

"hadn't've": "had not have",

"hasn't": "has not",

"haven't": "have not",

"he'd": "he had / he would",

"he'd've": "he would have",

"he'll": "he shall / he will",

"he'll've": "he shall have / he will have",

"he's": "he has / he is",

"how'd": "how did",

"how'd'y": "how do you",

"how'll": "how will",

"how's": "how has / how is / how does",

"i'd": "i would",

"i'd've": "i would have",

"i'll": "i will",

"i'll've": "i shall have",

"i'm": "i am",

"i've": "i have",

"isn't": "is not",

"it'd": "it had / it would",

"it'd've": "it would have",

"it'll": "it shall / it will",

"it'll've": "it shall have / it will have",

"it's": "it has / it is",

"let's": "let us",

"ma'am": "madam",

"mayn't": "may not",

"might've": "might have",

"mightn't've": "might not have",

"must've": "must have",

"mustn't": "must not",

"mustn't've": "must not have",

"needn't": "need not",

"needn't've": "need not have",

"o'clock": "of the clock",

"oughtn't": "ought not",

"oughtn't've": "ought not have",

"shan't": "shall not",

"sha'n't": "shall not",

"shan't've": "shall not have",

"she'd": "she had / she would",

"she'd've": "she would have",

"she'll": "she shall / she will",

"she'll've": "she shall have / she will have",

"she's": "she has / she is",

"should've": "should have",

"shouldn't": "should not",

"shouldn't've": "should not have",

"so've": "so have",

"that'd": "that would / that had",

"that'd've": "that would have",

"that's": "that has / that is",

"there'd": "there had / there would",

"there'd've": "there would have",

"there's": "there has / there is",

"they'd": "they had / they would",

"they'd've": "they would have",

"they'll": "they shall / they will",

"they'll've": "they shall have / they will have",

"they're": "they are",

"they've": "they have",

"we're": "we are",

"we've": "we have",

"weren't": "were not",

"what'll": "what shall / what will",

"what'll've": "what shall have / what will have",

"what're": "what are",

"what's": "what has / what is",

"what've": "what have",

"when's": "when has / when is",

"when've": "when have",

"where'd": "where did",

"where's": "where has / where is",

"where've": "where have",

"who'll": "who shall / who will",

"who'll've": "who shall have / who will have"

}
def expand_contractions(entry):

    entry=entry.lower()

    entry=re.sub(r"’","'",entry)

    entry=entry.split(" ")



    for idx,word in enumerate(entry):

        if word in contractions:

            

            entry[idx]=contractions[word]

    return " ".join(entry)
import re

def remove_punctuation(entry):

  entry=re.sub(r"[^\w\s]","",entry)

  return entry
def preprocess(dff):

    df=dff.copy()

    df['review']=df['review'].apply(lambda x: x.lower())

    df['review']=df['review'].apply(deEmojify)

    df['review']=df['review'].apply(expand_contractions)

    df['review']=df['review'].apply(lambda x:" ".join([i for i in x.split(" ") if len(i)>2]))

    df['review']=df['review'].apply(remove_punctuation)

    print (len(df))

    return df
ori_df=pd.read_csv('../input/supertype/labeled_negative_reviews_with_versions_ratings_type.csv')

ori_df
df_preprocessed=preprocess(df)

ori_df_preprocessed=preprocess(ori_df)

#Don't forget this..
import transformers
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('bert-large-uncased')
from tensorflow.keras import Model

from tensorflow.keras.layers import concatenate

from tensorflow.keras.callbacks import EarlyStopping

import tensorflow as tf

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Activation

from tensorflow.keras.layers import Dropout

from tensorflow.keras.layers import Dense

from tensorflow.keras.layers import Flatten

from tensorflow.keras.layers import Input

from tensorflow.keras.models import Model
def find_max_len(sentences):

    max_len=0

    lengthlist=[]

    for sent in sentences:

    # Tokenize the text and add `[CLS]` and `[SEP]` tokens.

        input_ids = tokenizer.encode(sent, add_special_tokens=True)

        #Encode will just return the IDs rather than the ID,tokentype id and attention mask

        lengthlist.append(len(input_ids))

    # Update the maximum sentence length.

        max_len = max(max_len, len(input_ids))

    return max_len,lengthlist
maxx,lengthlist=find_max_len(df['review'].values)

maxx
sns.distplot(lengthlist)
enc_di = tokenizer.batch_encode_plus(

        #we can also put add_special_tokens=True but i think this is automatic

        ['test'], 

        return_attention_mask=False, 

        return_token_type_ids=False,

    )

enc_di #to get a glimpse of what the enc_di looks like
#Most of the words cut off at 100-150 ish, so we truncate when it reaches 150 to make training faster..

def regular_encode(texts, tokenizer,max_len=200):

    

    enc_di = tokenizer.batch_encode_plus(

        #we can also put add_special_tokens=True but i think this is automatic

        texts, 

        return_attention_mask=False, 

        return_token_type_ids=False,

        max_length=max_len,

        pad_to_max_length=True,

        truncation= True

    )

    

    return np.array(enc_di['input_ids'])
from sklearn.model_selection import train_test_split

x_train,x_val,y_train,y_val = train_test_split(df_preprocessed.review.values,df.iloc[:,1:7].values, test_size=0.05,random_state=2020)
#see x_train values

x_train[155]
ori_df
ori_df_preprocessed=preprocess(ori_df)
x_train = regular_encode(x_train, tokenizer)

x_val = regular_encode(ori_df_preprocessed.review.values, tokenizer)

y_val=ori_df.iloc[:,1:7]
x_train #the 101 token is the cls token, 0 is the padding 
from random import sample
def build_model(transformer,max_len=200):

    input_word_ids = Input(shape=(max_len,), dtype=tf.int32, name="input_word_ids")

    sequence_output = transformer(input_word_ids)[0] #returns last tuple, containing last hidden layer

    cls_token = sequence_output[:, 0, :] #extract cls token

    #shape of sequence_output--> I think probably (number of sentences,number of tokens, token depth so 768)



  

    z = tf.keras.layers.Dropout(0.1)(cls_token)

    z=tf.keras.layers.Dense(36,activation='relu')(z)

    z=tf.keras.layers.Dropout(0.1)(z)

    z=tf.keras.layers.Dense(36,activation='tanh')(z)

    out =tf.keras.layers.Dense(6,activation='sigmoid')(z)

    model = Model(inputs=input_word_ids, outputs=out)

    model.compile(tf.keras.optimizers.Adam(lr=1e-6), loss='binary_crossentropy', metrics=['accuracy'])

    print(model.summary())

    return model
with strategy.scope():

    transformer=transformers.TFAutoModel.from_pretrained(

    'bert-large-uncased'

    )

    model=build_model(transformer)
ec=EarlyStopping(monitor='val_loss',patience=10)
history=model.fit(x_train,y_train,epochs=25,verbose=1,batch_size=64,validation_data=(x_val,y_val),use_multiprocessing=True)
# plot loss during training

pyplot.title('Loss (binary_cross_entropy)')

#if u use percentage error its around 25% loss

pyplot.plot(history.history['loss'], label='train')

pyplot.plot(history.history['val_loss'], label='test')

pyplot.legend()

pyplot.show()

#Data is too little, but we can see its learning about the train set..

#Validation also too little..

#Should train the bert model on both positive and negative reviews from both maybe

ori_df=pd.read_csv('../input/supertype/labeled_negative_reviews_with_versions_ratings_type.csv')

print(len(ori_df))

ori_df=preprocess(ori_df)

print(len(ori_df))

all_data=regular_encode(ori_df.review.values,tokenizer)
len(ori_df)
dummy=model.predict(all_data,verbose=1)

dummy_bool=dummy>0.5

dummy_bool=dummy_bool.astype(int)

#dummy_bool contains the predicted labels





#ori_df_topic_vectors is the true label

ori_df_topic_vectors=ori_df.iloc[:,1:7].values

len(dummy_bool)
len(ori_df.iloc[:,1:7].values)
dummy_bool=dummy>0.49

dummy_bool=dummy_bool.astype(int)
def create_labels(vector):

    string=[]

    for row in vector:

        row=row.astype(str)

        string.append("".join(row))

    return string
true_string=create_labels(ori_df_topic_vectors)

pred_string=create_labels(dummy_bool)
df_pred=pd.DataFrame({'review':ori_df.review.values,

                     'true':true_string,

                     'pred':pred_string})

df_pred #For us to see the predictions
percentage=(df_pred['true']==df_pred['pred']).sum()/len(df_pred)

print(f"Percentage of predictions model got right: {round(percentage*100)}%")

#Here I mean how exact are the predictions made by BERT, as compared to the true labels, maybe need a better metric for this

#Maybe can use micro/macro accuracy/ or a different metric



#This is a better result, but we have to keep in mind that this might be because the training examples came from here

from sklearn.metrics import hamming_loss,classification_report



print (classification_report(ori_df_topic_vectors,dummy_bool))

#precision is decent, shows that the model is able to extinguish better to some extent, TP/TP+FN, but recall low

#Also, class 4 constantly underperforms, might remove service functionality (change to app_func?)
df.columns[1:7]

#So 0:stock_logistics, 1:delivery logistics, 2:customer service, etc.



#The model does not do so well in recognizing service_functionality
hamming_loss(ori_df_topic_vectors, dummy_bool) #the hamming loss for this BERT model

#Target--> Hammingloss of 0.05 and below
import h5py

model.save_weights("model.h5")

print("Saved model to disk")

model.load_weights("model.h5")
model.evaluate(regular_encode(ori_df.review.values,tokenizer),ori_df.iloc[:,1:7].values)

#So basically using the BERT model, with only a dropout 0.1, and a 36 node FC, followed by 6 Node Fc gives a train accuracy of 0.7, test accuracy of 0.62

#About 45% of the dataset of labels was predicted correctly.





#csv scraped from apple google, produce bar plot--> topics labeled so the 1s and 0s--> 

#input csv--> output csv (sentences, scores)



#Version 2

#Dropout(0.1)-->Dense(36,activation='relu')-->Dropout(0.1)-->z=tf.keras.layers.Dense(36,activation='tanh')-->out =tf.keras.layers.Dense(6,activation='sigmoid')(z)

#trained for about 120-130 epochs, train accuracy 75-77%, val accuracy capped at 65%, 

#predicted 55% of the original dataset right with 0.5 threshold (if we measure the accuracy in terms of 1s and 0s, rather than sigmoid probability outputs), but 0.49 accuracy.

df_pred.to_csv('bert_base_negative_only.csv')