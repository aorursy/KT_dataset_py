# Disclaimer: This code was not written by me. I used bits & pieces from different websites & used it for learning purposes.

#Lets import all the librarikes required to build a model

import numpy as np

import pandas as pd

import tensorflow as tf

import re

import nltk

from sklearn.feature_extraction.text import TfidfVectorizer



import matplotlib.pyplot as plt

import pandas as pd

import numpy as np

import seaborn as sns

from sklearn.model_selection import train_test_split



from bs4 import BeautifulSoup

from nltk.corpus import stopwords

from sklearn.feature_extraction.text import CountVectorizer

import tensorflow as tf

print('imported')
#import the training dataset

train_comment_raw=pd.read_csv('../input/train.csv')

train_comment_raw['comment_text']=train_comment_raw['comment_text'].fillna('Missing')
#generating new features

def feature_engg(input):

    train_comment_raw=input

    train_comment_raw['capital_word_count']=train_comment_raw.apply(lambda row:sum(1 for each in row['comment_text'].split() if each.isupper()),axis=1)

    train_comment_raw['capital_char_count']=train_comment_raw.apply(lambda row:sum(1 for each in row['comment_text'] if each.isupper()),axis=1)

    train_comment_raw['comment_length']=train_comment_raw['comment_text'].apply(len)

    train_comment_raw['capital_word_vs_len']=train_comment_raw.apply(lambda row:row['capital_word_count']/row['comment_length'],axis=1)

    train_comment_raw['capital_char_vs_len']=train_comment_raw.apply(lambda row:row['capital_char_count']/row['comment_length'],axis=1)

    return train_comment_raw
train_comment_raw=feature_engg(train_comment_raw)

train_comment_raw.head()

#correlation between generated features

def correlation(label_column,feature_column):

    correlation=[{label_col:train_comment_raw[feature_col].corr(train_comment_raw[label_col])  for label_col in label_column }for feature_col  in  feature_column]

    result=pd.DataFrame(correlation,index=feature_column)

    ax = sns.heatmap(result, vmin=-0.2, vmax=0.2, center=0.0)

    plt.show()

    return result





label_column=['toxic','severe_toxic','obscene','threat','insult','identity_hate']

feature_column=['capital_word_count','capital_char_count','comment_length','capital_char_vs_len','capital_word_vs_len']

result=correlation(label_column,feature_column)

result.head()
train_comment_raw[['toxic','severe_toxic','obscene','threat','insult','identity_hate','capital_word_count','capital_char_count','comment_length','capital_word_vs_len']].hist()

plt.show()
train_comment_raw=train_comment_raw.head(2000)
#cleaning of comments

def clean_comment(row):

    input=row['comment_text']

    cleaned_html=BeautifulSoup(input).get_text()

    cleaned_special_chr=re.sub('[^a-zA-Z]',' ',cleaned_html)

    cleaned_special_chr = re.sub(r'[?|$|.|!]',r'',cleaned_special_chr)

    lower=cleaned_special_chr.lower().split()

    clean_stop_word=[each_char for each_char  in lower if each_char not in set(stopwords.words("english"))]

    return ' '.join(clean_stop_word)





train_comment_raw['comment_text']=train_comment_raw.apply(clean_comment,axis=1)

train_comment_raw.head(3)
#vectorize the Comments using tfid vectorizer

def convert_vectorizer(input):    

    tfidf_vectorizer = TfidfVectorizer(max_features=1000,stop_words='english')

    tfidf = tfidf_vectorizer.fit_transform(input)

    return tfidf,tfidf_vectorizer



# print(train_comment_raw.head(2)['comment_text'])

tfidf,tf_feature_names=convert_vectorizer(list(train_comment_raw['comment_text']))

    
#merge the other feature with vector

vectorized_list=np.array(tfidf.toarray())

value_features=np.array(train_comment_raw[['toxic','severe_toxic','obscene','threat','insult','identity_hate']])

all_new_feature=[]

print(vectorized_list.shape)

for new_f,old_f in zip(value_features,vectorized_list):

    feature_tuple=np.append(new_f,old_f)

    all_new_feature.append(feature_tuple)

all_new_feature=np.array(all_new_feature)

print((all_new_feature).shape)

label=np.array(train_comment_raw[['toxic','severe_toxic','obscene','threat','insult','identity_hate']])



def next_batch(num, data, labels):

    '''

    Return a total of `num` random samples and labels. 

    '''

    idx = np.arange(0 , len(data))

    np.random.shuffle(idx)

    idx = idx[:num]

    data_shuffle = [data[ i] for i in idx]

    labels_shuffle = [labels[ i] for i in idx]



    return np.asarray(data_shuffle), np.asarray(labels_shuffle)
#properties used to support Neural network

input_features=1006

h_layer_1_size=3000

h_layer_2_size=2000

h_layer_3_size=2000

output_features=6

learning_rate=0.3

steps=20

batch_size=200
source_input=tf.placeholder('float',[None,input_features])

print(source_input.shape)

target_input=tf.placeholder('float',[None,output_features])

print(target_input.shape)
#weight and bias

weight={

    'w1':tf.Variable(tf.random_normal([input_features,h_layer_1_size])),

    'w2':tf.Variable(tf.random_normal([h_layer_1_size,h_layer_2_size])),

    'w3':tf.Variable(tf.random_normal([h_layer_2_size,h_layer_3_size])),

    'out_w':tf.Variable(tf.random_normal([h_layer_3_size,output_features]))

}

bias={

    'b1':tf.Variable(tf.random_normal([h_layer_1_size])),

    'b2':tf.Variable(tf.random_normal([h_layer_2_size])),

    'b3':tf.Variable(tf.random_normal([h_layer_3_size])),

    'out_b':tf.Variable(tf.random_normal([output_features]))

}
#building model

def model(input):

    h_layer1=tf.add(tf.matmul(input,weight['w1']),bias['b1'])

    h_layer2=tf.add(tf.matmul(h_layer1,weight['w2']),bias['b2'])

    h_layer3=tf.add(tf.matmul(h_layer2,weight['w3']),bias['b3'])

    h_layer4=tf.add(tf.matmul(h_layer3,weight['out_w']),bias['out_b'])

    return h_layer4



logits=model(source_input)  



#optimizing and reduce error

loss=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits,labels=target_input))

optimizer=tf.train.AdamOptimizer(learning_rate=learning_rate)

train_op=optimizer.minimize(loss)

correct_pred = tf.equal(logits, target_input)

accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

init=tf.global_variables_initializer()


print('start')

train_X,test_X,train_y,test_y=train_test_split(all_new_feature,label,test_size=0.33, random_state=42)

print(train_X.shape,' ',train_y.shape)

with tf.Session() as sess:

    sess.run(init)

    for step in range(1,steps):

        X,y=next_batch(batch_size,train_X,train_y)

        tr_loss,acc=sess.run([loss,accuracy],feed_dict={source_input:X,target_input:y})

        print('loss: ',tr_loss)

        print('correct : ',acc)

    print('Finish!!')