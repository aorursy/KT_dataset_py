# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import matplotlib.pyplot as plt

import re

import tensorflow as tf

from keras.preprocessing.sequence import pad_sequences

from keras.preprocessing.text import Tokenizer 

from keras.models import Model

from keras.layers import Input, Dense, Embedding, Dropout, Conv1D, GlobalMaxPooling1D

from keras.callbacks import EarlyStopping, ModelCheckpoint

from keras.utils.vis_utils import plot_model

from sklearn.metrics import roc_auc_score



import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)
!unzip /kaggle/input/jigsaw-toxic-comment-classification-challenge/train.csv.zip

!unzip /kaggle/input/jigsaw-toxic-comment-classification-challenge/test.csv.zip
# Loading Training set

df_train=pd.read_csv('./train.csv')

print('Shape=>',df_train.shape)

df_train.head()
# Loading Test set

df_test=pd.read_csv('./test.csv')

print('Shape=>',df_test.shape)

df_test.head()
for i,v in enumerate(df_train['comment_text'].sample(5).values):

    print('Comment ',i+1,'=>',repr(v))
for i in ['toxic','severe_toxic','obscene','threat','insult','identity_hate']:

    print(df_train[i].value_counts(normalize=True)*100)
fig,axes=plt.subplots(3,2,figsize=(15,15))



for ax,class_name in zip(axes.flatten(),['toxic','severe_toxic','obscene','threat','insult','identity_hate']):

    pd.value_counts(df_train[class_name],sort=True).plot(kind='bar',rot=0,ax=ax)

    ax.set_title('{} Distribution'.format(class_name))

    ax.set_xticks([0,1])

    ax.set_xlabel('Labels')

    ax.set_ylabel('Frequency')



plt.show()
def cleaner(text):

    text=text.lower()

    # keeping only words

    text=re.sub("[^a-z]+"," ",text)

    # removing extra spaces

    text=re.sub("[ ]+"," ",text)

    

    return text
# Clean comments in Training Set

df_train['cleaned']=df_train['comment_text'].apply(cleaner)
df_train['comment_text'][:2].values
df_train['cleaned'][:2].values
# Cleaning comments in Testing Set

df_test['cleaned']=df_test['comment_text'].apply(cleaner)
df_train.head()
df_test.head()
comment_word_count = []



#populate the lists with sentence lengths

for i in df_train['cleaned']:

      comment_word_count.append(len(i.split()))



length_df = pd.DataFrame({'Comment Length':comment_word_count})



length_df.hist(bins = 100, range=(0,500),figsize=(12,8))

plt.show()
tokenizer = Tokenizer(oov_token='OOV')

#creating index for words

tokenizer.fit_on_texts(df_train['cleaned'])
tokenizer.word_index
print('Vocabulary Size=>',len(tokenizer.word_index))
# Converting word sequence to integer sequence

train_seq = tokenizer.texts_to_sequences(df_train['cleaned']) 

test_seq = tokenizer.texts_to_sequences(df_test['cleaned'])
# Padding with zero

train_seq=pad_sequences(train_seq,maxlen=100,padding='post')

test_seq=pad_sequences(test_seq,maxlen=100,padding='post')
vocabulary=len(tokenizer.word_index)+1

print('Vocabulary Size=>',vocabulary)
print('Shape of train_sequence=>',train_seq.shape)

print('Shape of test_sequence=>',test_seq.shape)
y_train=df_train[['toxic','severe_toxic','obscene','threat','insult','identity_hate']].values

print('Shape of Training Labels=>',y_train.shape)
# Installing for Stratified Split

!pip install iterative-stratification
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
msss=MultilabelStratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=0)
for train_index, val_index in msss.split(train_seq, y_train):

    # Creating Train Set

    x_train_split,y_train_split=train_seq[train_index],y_train[train_index]

    # Creating Test Set

    x_valid_split,y_valid_split=train_seq[val_index],y_train[val_index]
print('Shape of Train Split=>',x_train_split.shape,y_train_split.shape)

print('Shape of Validation Split=>',x_valid_split.shape,y_valid_split.shape)
print('Class Distribution of Train Split in Percentage')

for i,v in enumerate(['toxic','severe_toxic','obscene','threat','insult','identity_hate']):

    print(v)

    print(pd.Series(y_train_split[:,i]).value_counts(normalize=True)*100)
print('Class Distribution of Validation Split in Percentage')

for i,v in enumerate(['toxic','severe_toxic','obscene','threat','insult','identity_hate']):

    print(v)

    print(pd.Series(y_valid_split[:,i]).value_counts(normalize=True)*100)
input_1=Input(shape=(100,))

embedding_1=Embedding(vocabulary,100)(input_1)

conv_1=Conv1D(filters=352,kernel_size=7,padding="same")(embedding_1)

dropout_1=Dropout(0.06675)(conv_1)

pool_1=GlobalMaxPooling1D()(dropout_1)



dense=Dense(128,activation='relu')(pool_1)

output=Dense(6,activation='sigmoid')(dense)



model=Model(inputs=[input_1],outputs=output)



model.summary()
plot_model(model, to_file= 'model.png', show_shapes=True)
# Compile Model

model.compile(optimizer='adam',loss='binary_crossentropy',metrics=["accuracy"])
# Callbacks

es=EarlyStopping(monitor='val_loss', mode='min', verbose=1,patience=5,min_delta=1e-5)

mc = ModelCheckpoint("/kaggle/working/model.hdf5", monitor='val_loss', verbose=0, save_best_only=True, mode='min')
model.fit(x_train_split,y_train_split, batch_size=512, epochs=100, verbose=1, validation_data=(x_valid_split,y_valid_split), callbacks=[es,mc])
# In-sample Evaluation

train_pred=model.predict(x_train_split)

print('In-sample Evaluation ROC-AUC Score:\n',roc_auc_score(y_train_split,train_pred))
# Out-of-sample Evaluation

valid_pred=model.predict(x_valid_split)

print('In-sample Evaluation ROC-AUC Score:\n',roc_auc_score(y_valid_split,valid_pred))
final_pred=model.predict(test_seq)
#Dataframe for final probabilties

prob=pd.DataFrame(columns=['id','toxic','severe_toxic','obscene','threat','insult','identity_hate'])

prob['id']=df_test['id']

for index,value in enumerate(['toxic','severe_toxic','obscene','threat','insult','identity_hate']):

    prob[value]=final_pred[:,index]
prob
prob.to_csv('submission-CNN-final.csv',index=False)