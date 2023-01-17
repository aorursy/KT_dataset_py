from pathlib import Path

import numpy as np 

import pandas as pd 

import seaborn as sns

from matplotlib import pyplot as plt

%matplotlib inline

import re

import tensorflow as tf

import transformers

from tensorflow.keras.layers import Dense, Input, Dropout

from tensorflow.keras.optimizers import Adam

from tensorflow.keras.models import Model

from tensorflow.keras import regularizers

import warnings



warnings.filterwarnings("ignore")

#from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
# Detect hardware, return appropriate distribution strategy

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
DATA_ROOT = Path("..")/"input"/ "jigsaw-multilingual-toxic-comment-classification/"

df1,df2,df3 = [pd.read_csv(DATA_ROOT / fname, usecols=["comment_text", "toxic"]) for fname in ["jigsaw-toxic-comment-train.csv",

                                                            "jigsaw-unintended-bias-train.csv",

                                                            "validation.csv"

                                                                       ]]







test, sample= [pd.read_csv(DATA_ROOT / fname) for fname in ["test.csv",

                                                            "sample_submission.csv"

                                                            ]]

# #Clean up the comment text

# def clean_text(text):

#     text = text.lower()

    

#     text = re.sub(r"what's", "what is ", text)

#     text = re.sub(r"\'s", " ", text)

#     text = re.sub(r"\'ve", " have ", text)

#     text = re.sub(r"can't", "cannot ", text)

#     text = re.sub(r"n't", " not ", text)

#     text = re.sub(r"i'm", "i am ", text)

#     text = re.sub(r"\'re", " are ", text)

#     text = re.sub(r"\'d", " would ", text)

#     text = re.sub(r"\'ll", " will ", text)

#     text = re.sub(r"\'scuse", " excuse ", text)

#     text = re.sub('\W', ' ', text)

#     text = re.sub('\s+', ' ', text)

#     text = text.strip(' ')

#     return text
# #Clean up the comment text

# def clean_text(g_text):

    

#     # Special characters

#     g_text = re.sub(r"\x89Û_", "", g_text)

#     g_text = re.sub(r"\x89ÛÒ", "", g_text)

#     g_text = re.sub(r"\x89ÛÓ", "", g_text)

#     g_text = re.sub(r"\x89ÛÏWhen", "When", g_text)

#     g_text = re.sub(r"\x89ÛÏ", "", g_text)

#     g_text = re.sub(r"China\x89Ûªs", "China's", g_text)

#     g_text = re.sub(r"let\x89Ûªs", "let's", g_text)

#     g_text = re.sub(r"\x89Û÷", "", g_text)

#     g_text = re.sub(r"\x89Ûª", "", g_text)

#     g_text = re.sub(r"\x89Û\x9d", "", g_text)

#     g_text = re.sub(r"å_", "", g_text)

#     g_text = re.sub(r"\x89Û¢", "", g_text)

#     g_text = re.sub(r"\x89Û¢åÊ", "", g_text)

#     g_text = re.sub(r"fromåÊwounds", "from wounds", g_text)

#     g_text = re.sub(r"åÊ", "", g_text)

#     g_text = re.sub(r"åÈ", "", g_text)

#     g_text = re.sub(r"JapÌ_n", "Japan", g_text)    

#     g_text = re.sub(r"Ì©", "e", g_text)

#     g_text = re.sub(r"å¨", "", g_text)

#     g_text = re.sub(r"SuruÌ¤", "Suruc", g_text)

#     g_text = re.sub(r"åÇ", "", g_text)

#     g_text = re.sub(r"å£3million", "3 million", g_text)

#     g_text = re.sub(r"åÀ", "", g_text)

    

#     # Contractions

#     g_text = re.sub(r"he's", "he is", g_text)

#     g_text = re.sub(r"there's", "there is", g_text)

#     g_text = re.sub(r"We're", "We are", g_text)

#     g_text = re.sub(r"That's", "That is", g_text)

#     g_text = re.sub(r"won't", "will not", g_text)

#     g_text = re.sub(r"they're", "they are", g_text)

#     g_text = re.sub(r"Can't", "Cannot", g_text)

#     g_text = re.sub(r"wasn't", "was not", g_text)

#     g_text = re.sub(r"don\x89Ûªt", "do not", g_text)

#     g_text = re.sub(r"aren't", "are not", g_text)

#     g_text = re.sub(r"isn't", "is not", g_text)

#     g_text = re.sub(r"What's", "What is", g_text)

#     g_text = re.sub(r"haven't", "have not", g_text)

#     g_text = re.sub(r"hasn't", "has not", g_text)

#     g_text = re.sub(r"There's", "There is", g_text)

#     g_text = re.sub(r"He's", "He is", g_text)

#     g_text = re.sub(r"It's", "It is", g_text)

#     g_text = re.sub(r"You're", "You are", g_text)

#     g_text = re.sub(r"I'M", "I am", g_text)

#     g_text = re.sub(r"shouldn't", "should not", g_text)

#     g_text = re.sub(r"wouldn't", "would not", g_text)

#     g_text = re.sub(r"i'm", "I am", g_text)

#     g_text = re.sub(r"I\x89Ûªm", "I am", g_text)

#     g_text = re.sub(r"I'm", "I am", g_text)

#     g_text = re.sub(r"Isn't", "is not", g_text)

#     g_text = re.sub(r"Here's", "Here is", g_text)

#     g_text = re.sub(r"you've", "you have", g_text)

#     g_text = re.sub(r"you\x89Ûªve", "you have", g_text)

#     g_text = re.sub(r"we're", "we are", g_text)

#     g_text = re.sub(r"what's", "what is", g_text)

#     g_text = re.sub(r"couldn't", "could not", g_text)

#     g_text = re.sub(r"we've", "we have", g_text)

#     g_text = re.sub(r"it\x89Ûªs", "it is", g_text)

#     g_text = re.sub(r"doesn\x89Ûªt", "does not", g_text)

#     g_text = re.sub(r"It\x89Ûªs", "It is", g_text)

#     g_text = re.sub(r"Here\x89Ûªs", "Here is", g_text)

#     g_text = re.sub(r"who's", "who is", g_text)

#     g_text = re.sub(r"I\x89Ûªve", "I have", g_text)

#     g_text = re.sub(r"y'all", "you all", g_text)

#     g_text = re.sub(r"can\x89Ûªt", "cannot", g_text)

#     g_text = re.sub(r"would've", "would have", g_text)

#     g_text = re.sub(r"it'll", "it will", g_text)

#     g_text = re.sub(r"we'll", "we will", g_text)

#     g_text = re.sub(r"wouldn\x89Ûªt", "would not", g_text)

#     g_text = re.sub(r"We've", "We have", g_text)

#     g_text = re.sub(r"he'll", "he will", g_text)

#     g_text = re.sub(r"Y'all", "You all", g_text)

#     g_text = re.sub(r"Weren't", "Were not", g_text)

#     g_text = re.sub(r"Didn't", "Did not", g_text)

#     g_text = re.sub(r"they'll", "they will", g_text)

#     g_text = re.sub(r"they'd", "they would", g_text)

#     g_text = re.sub(r"DON'T", "DO NOT", g_text)

#     g_text = re.sub(r"That\x89Ûªs", "That is", g_text)

#     g_text = re.sub(r"they've", "they have", g_text)

#     g_text = re.sub(r"i'd", "I would", g_text)

#     g_text = re.sub(r"should've", "should have", g_text)

#     g_text = re.sub(r"You\x89Ûªre", "You are", g_text)

#     g_text = re.sub(r"where's", "where is", g_text)

#     g_text = re.sub(r"Don\x89Ûªt", "Do not", g_text)

#     g_text = re.sub(r"we'd", "we would", g_text)

#     g_text = re.sub(r"i'll", "I will", g_text)

#     g_text = re.sub(r"weren't", "were not", g_text)

#     g_text = re.sub(r"They're", "They are", g_text)

#     g_text = re.sub(r"Can\x89Ûªt", "Cannot", g_text)

#     g_text = re.sub(r"you\x89Ûªll", "you will", g_text)

#     g_text = re.sub(r"I\x89Ûªd", "I would", g_text)

#     g_text = re.sub(r"let's", "let us", g_text)

#     g_text = re.sub(r"it's", "it is", g_text)

#     g_text = re.sub(r"can't", "cannot", g_text)

#     g_text = re.sub(r"don't", "do not", g_text)

#     g_text = re.sub(r"you're", "you are", g_text)

#     g_text = re.sub(r"i've", "I have", g_text)

#     g_text = re.sub(r"that's", "that is", g_text)

#     g_text = re.sub(r"i'll", "I will", g_text)

#     g_text = re.sub(r"doesn't", "does not", g_text)

#     g_text = re.sub(r"i'd", "I would", g_text)

#     g_text = re.sub(r"didn't", "did not", g_text)

#     g_text = re.sub(r"ain't", "am not", g_text)

#     g_text = re.sub(r"you'll", "you will", g_text)

#     g_text = re.sub(r"I've", "I have", g_text)

#     g_text = re.sub(r"Don't", "do not", g_text)

#     g_text = re.sub(r"I'll", "I will", g_text)

#     g_text = re.sub(r"I'd", "I would", g_text)

#     g_text = re.sub(r"Let's", "Let us", g_text)

#     g_text = re.sub(r"you'd", "You would", g_text)

#     g_text = re.sub(r"It's", "It is", g_text)

#     g_text = re.sub(r"Ain't", "am not", g_text)

#     g_text = re.sub(r"Haven't", "Have not", g_text)

#     g_text = re.sub(r"Could've", "Could have", g_text)

#     g_text = re.sub(r"youve", "you have", g_text)  

#     g_text = re.sub(r"donå«t", "do not", g_text)   

            

#     # Character entity references

#     g_text = re.sub(r"&gt;", ">", g_text)

#     g_text = re.sub(r"&lt;", "<", g_text)

#     g_text = re.sub(r"&amp;", "&", g_text)

    

           

#     # Urls

#     g_text = re.sub(r"https?:\/\/t.co\/[A-Za-z0-9]+", "", g_text)

        

#     # Words with punctuations and special characters

#     punctuations = '@#!?+&*[]-%.:/();$=><|{}^' + "'`"

#     for p in punctuations:

#         g_text = g_text.replace(p, f' {p} ')

        

#     # ... and ..

#     g_text = g_text.replace('...', ' ... ')

#     if '...' not in g_text:

#         g_text = g_text.replace('..', ' ... ')      

    

    

#     g_text = re.sub(r"\'scuse", " excuse ", g_text)

#     g_text = re.sub('\W', ' ', g_text)

#     g_text = re.sub('\s+', ' ', g_text)

#     g_text = g_text.strip(' ')

#     return g_text
df2['toxic'] = df2['toxic'].apply(lambda x: 0 if x<0.5 else 1)
df2 = pd.concat([

    

    df2[['comment_text', 'toxic']].query('toxic==1'),

    df2[['comment_text', 'toxic']].query('toxic==0').sample(n=500000, random_state=0)

])
# Percentage of unlabelled comments

unlabelled_df2_all = df2[(df2['toxic']!=1) ]

print('Percentage of unlabelled comments in df2 ', len(unlabelled_df2_all)/len(df2)*100)
# # clean the comment_text in df1

# df1['comment_text'] = df1['comment_text'].map(lambda com : clean_text(com))

# # clean the comment_text in df2

# df2['comment_text'] = df2['comment_text'].map(lambda com : clean_text(com))
# Character length for the rows in the df1 & df2 data

df1['char_length'] = df1['comment_text'].apply(lambda x: len(str(x)))

df2['char_length'] = df2['comment_text'].apply(lambda x: len(str(x)))
# Character length for the rows in the training data

df1= df1[df1['char_length'] >= 545] 

df2= df2[df2['char_length'] >= 545] 
train = pd.concat([df1, df2], axis=0).reset_index(drop=True)

train = train.sample(frac=1).reset_index(drop=True).head(200000)

train = train.reset_index(drop=True)

valid = df3

valid = valid.reset_index(drop=True)
# train['toxic'] = train['toxic'].apply(lambda x: 0 if x<0.5 else 1)
train.describe()
# Percentage of unlabelled comments

unlabelled_in_all = train[(train['toxic']!=1) ]

print('Percentage of unlabelled comments is ', len(unlabelled_in_all)/len(train)*100)
# # check for any 'null' comment in training data

# no_comment = train[train['comment_text'].isnull()]

# len(no_comment)
test.head()
# # check for any 'null' comment in test data

# no_comment = test[test['content'].isnull()]

# len(no_comment)
# # check for any 'null' comment in validation data

# no_comment = valid[valid['comment_text'].isnull()]

# len(no_comment)
# total rows in train, test valid data

print('Total rows in training dataset is {}'.format(len(train)))

print('Total rows in validation dataset is {}'.format(len(valid)))

print('Total rows in test dataset is {}'.format(len(test)))
# # Character length for the rows in the training data

# train['char_length'] = train['comment_text'].apply(lambda x: len(str(x)))
# histogram plot for text length

sns.set()

train['char_length'].hist()

plt.show()
# top 10 largest values in column char_length 

train.nlargest(10, 'char_length') 
# top 10 smallest values in column char_length 

train.nsmallest(10, 'char_length') 
# # Character length for the rows in the training data

# train= train[train['char_length'] >= 15] 
# # top 10 largest values in column char_length 

# train.nsmallest(10, 'char_length') 
# Percentage of unlabelled comments

unlabelled_in_all = valid[(valid['toxic']!=1) ]

print('Percentage of unlabelled comments in validation is ', len(unlabelled_in_all)/len(valid)*100)
valid.describe()
test['char_length'] = test['content'].apply(lambda x: len(str(x)))
sns.set()

test['char_length'].hist()

plt.show()
# top 10 largest values test dataset  in column char_length 

test.nlargest(10, 'char_length') 
# top 10 smallest values test dataset  in column char_length 

test.nsmallest(10, 'char_length') 
valid['char_length'] = valid['comment_text'].apply(lambda x: len(str(x)))
sns.set()

valid['char_length'].hist()

plt.show()
# top 10 largest values valid dataset  in column char_length 

valid.nlargest(10, 'char_length') 
# top 10 smallest values valid dataset  in column char_length 

valid.nsmallest(10, 'char_length') 
train = train.drop('char_length',axis=1)

valid = valid.drop('char_length',axis=1)

test = test.drop('char_length',axis=1)
# # # Function to encode the text

def encode_fn(texts, tokenizer, maxlen=512):

    encode = tokenizer.batch_encode_plus(

        texts, 

        return_attention_masks=False, 

        return_token_type_ids=False,

        pad_to_max_length=True,

        max_length=maxlen

    )

    

    return np.array(encode['input_ids'])
AUTO = tf.data.experimental.AUTOTUNE

# Configuration

EPOCHS = 3

BATCH_SIZE = 16 * strategy.num_replicas_in_sync

MAX_LEN = 192
# First load the real tokenizer

tokenizer = transformers.AutoTokenizer.from_pretrained('jplu/tf-xlm-roberta-large')

%%time

x_train = encode_fn(train.comment_text.astype(str), tokenizer, maxlen=MAX_LEN)

x_valid = encode_fn(valid.comment_text.astype(str), tokenizer, maxlen=MAX_LEN)

x_test =  encode_fn(test.content.astype(str), tokenizer, maxlen=MAX_LEN)



y_train = train.toxic.values

y_valid = valid.toxic.values
train_dataset = (

    tf.data.Dataset

    .from_tensor_slices((x_train, y_train))

    .repeat()

    .shuffle(2048)

    .batch(BATCH_SIZE)

    .prefetch(AUTO)

)



valid_dataset = (

    tf.data.Dataset

    .from_tensor_slices((x_valid, y_valid))

    .batch(BATCH_SIZE)

    .cache()

    .prefetch(AUTO)

)



test_dataset = (

    tf.data.Dataset

    .from_tensor_slices(x_test)

    .batch(BATCH_SIZE)

)
# Function to build the MODEL

def model_fn(transformer, max_len=512):

    input_word_ids = Input(shape=(max_len,), dtype=tf.int32, name="input_word_ids")

    sequence_output = transformer(input_word_ids)[0]

    

    cls_token = sequence_output[:, 0, :]



    

    

#     layers1 = Dense(512, kernel_regularizer=regularizers.l2(0.0001),activation='elu')(cls_token)

  

#     layers3 = Dense(512, kernel_regularizer=regularizers.l2(0.0001),activation='elu')(layers1)

  

#     layers5 = Dense(512, kernel_regularizer=regularizers.l2(0.0001),activation='elu')(layers3)

  

#     layers7 = Dense(512, kernel_regularizer=regularizers.l2(0.0001),activation='elu')(layers5)

  

    

    

    

#     out = Dense(1, activation='sigmoid')(layers7)



  

    out = Dense(1, activation='sigmoid')(cls_token)

    model = Model(inputs=input_word_ids, outputs=out)

    optimizer = tf.keras.optimizers.Adam(

    learning_rate=0.00001, beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=False,

    name='Adam'

)

    model.compile(optimizer, loss='binary_crossentropy', metrics = [tf.keras.metrics.AUC()] )

    return model

    

%%time

with strategy.scope():

    transformer_layer = (

        transformers.TFAutoModel.from_pretrained("jplu/tf-xlm-roberta-large")   

        

    )

    model = model_fn(transformer_layer, max_len=MAX_LEN)

model.summary()
# n_steps = x_train.shape[0] // BATCH_SIZE

# train_history = model.fit(

#     train_dataset,

#     steps_per_epoch=n_steps,

#     #callbacks=[callback],

#     validation_data=valid_dataset,

#     epochs=EPOCHS

# )
# n_steps = x_valid.shape[0] // BATCH_SIZE

# train_history_2 = model.fit(

#     valid_dataset.repeat(),

#     steps_per_epoch=n_steps,

#     epochs=EPOCHS*2

# )
# sample['toxic'] = model.predict(test_dataset, verbose=1)

# sample.to_csv('submission.csv', index=False)
# from keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler



# file_path = "best_model.hdf5"

# check_point = ModelCheckpoint(file_path, monitor = "val_loss", verbose = 1,

#                               save_best_only = True, mode = "min")

# #ra_val = RocAucEvaluation(validation_data=valid_dataset, interval = 1)

# early_stop = EarlyStopping(monitor = "val_loss", mode = "min", patience = 5)





# n_steps = x_train.shape[0] // BATCH_SIZE



# train_history = model.fit(

#     train_dataset,

#     steps_per_epoch=n_steps,

#     validation_steps=127,

#     #callbacks = [check_point, early_stop],

#     validation_data=valid_dataset,

#     epochs=4

# )
callback_stop = tf.keras.callbacks.EarlyStopping(

    monitor='val_loss', min_delta=0, patience=5, verbose=0, mode='auto',

    baseline=None, restore_best_weights=True

)



n_steps = x_train.shape[0]// BATCH_SIZE

train_history1 = model.fit(

    train_dataset,

    steps_per_epoch=n_steps,

    callbacks = [callback_stop],

    validation_data=valid_dataset,

    epochs=20

)
n_steps = x_valid.shape[0] // BATCH_SIZE

train_history_2 = model.fit(

    valid_dataset.repeat(),

    steps_per_epoch=n_steps,

    callbacks = [callback_stop],

    epochs=EPOCHS*2

)
sample['toxic'] = model.predict(test_dataset, verbose=1)

sample.to_csv('submission.csv', index=False)