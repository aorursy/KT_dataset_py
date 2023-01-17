#####while this model scores low on accuracy it is still a working version

###############of nueral net using tensorfl

##import necessary packages

#####

import tensorflow as tf 

import numpy as np # linear algebra

import scipy as sp 

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib

import seaborn as sns 

import nltk

from nltk.corpus import stopwords

import wordcloud

from wordcloud import WordCloud, STOPWORDS

from matplotlib import pyplot as plt

##read in the data file an display 

df_spooky_author=pd.read_csv('../input/train.csv')

df_spooky_author



sentence_list=df_spooky_author['text'].values.tolist()

author_list=df_spooky_author['author'].values.tolist()

id_list=df_spooky_author['id'].values.tolist()

sentence_list

author_list





combined_list=[list(author_list) for author_list in zip(author_list, sentence_list)]



tokenized_list=[]



for authors,sentence in combined_list:

    tokenized_list.append([authors,nltk.word_tokenize(sentence)])



tokenized_list



##make individual authorwise list of words after removing stop words.

##This is to prepare the data for wordcloud

stop = set(stopwords.words('english'))    



tokenized_stop_words_list=[]





for author,sentence in tokenized_list:

    tokenized_stop_words_list_temp=[]   

    for word in sentence:

        if not word in stop:

            tokenized_stop_words_list_temp.append(word)

    tokenized_stop_words_list.append([author,tokenized_stop_words_list_temp])        

            

tokenized_stop_words_list



tokenized_words_EAP=[]

tokenized_words_MWS=[]

tokenized_words_HPL=[]



for author,sentence in tokenized_stop_words_list:

    if author=='EAP':

        for words in sentence:

            tokenized_words_EAP.append(words)

    if author=='MWS':

        for words in sentence:

            tokenized_words_MWS.append(words)

    if author=='HPL':

        for words in sentence:

            tokenized_words_HPL.append(words)

            

##word cloud for HPLovenCraft            

plt.figure(figsize=(30,30))

wc = WordCloud(background_color="black", max_words=10000, 

               stopwords=stop, max_font_size= 40)

wc.generate(" ".join(tokenized_words_HPL))

##plt.title("HP Lovecraft (Cthulhu-Squidy)", fontsize=16)

##Uncomment below line and run to see wordcloud for HP Lovencraft

##plt.imshow(wc.recolor( colormap= 'Pastel1_r' , random_state=17), alpha=0.98)

plt.axis('off')







##word cloud for Edgar Allen Poe         

plt.figure(figsize=(30,30))

wc = WordCloud(background_color="black", max_words=10000, 

               stopwords=stop, max_font_size= 40)

wc.generate(" ".join(tokenized_words_EAP))

##plt.title("HP Lovecraft (Cthulhu-Squidy)", fontsize=16)

##Uncomment below line and run to see wordcloud for Edgar Allen Poe

##plt.imshow(wc.recolor( colormap= 'Pastel1_r' , random_state=17), alpha=0.98)

plt.axis('off')







##word cloud for Mary Shelley            

plt.figure(figsize=(30,30))

wc = WordCloud(background_color="black", max_words=10000, 

               stopwords=stop, max_font_size= 40)

wc.generate(" ".join(tokenized_words_HPL))

##plt.title("HP Lovecraft (Cthulhu-Squidy)", fontsize=16)

##Uncomment below line and run to see wordcloud for Mary Shelley 

##plt.imshow(wc.recolor( colormap= 'Pastel1_r' , random_state=17), alpha=0.98)

plt.axis('off')



df_test=pd.DataFrame(tokenized_words_HPL)

df_test[0].value_counts()





from sklearn.feature_extraction.text import CountVectorizer

from sklearn.feature_extraction.text import TfidfTransformer

from sklearn.model_selection import train_test_split

from sklearn import preprocessing  

from sklearn import linear_model

from sklearn import metrics

from sklearn.metrics import accuracy_score



## TF-IDF implementation

count_vect = CountVectorizer()



sentence_train_counts = count_vect.fit_transform(sentence_list)

sentence_train_counts.shape





tfidf_transformer = TfidfTransformer()

sentence_train_tfidf = tfidf_transformer.fit_transform(sentence_train_counts)

sentence_train_tfidf



sentence_train_tfidf.shape

len(sentence_list)



##Label Encoder to encode values

le = preprocessing.LabelEncoder()

le.fit(author_list)



list(le.classes_)



author_list_encoded=le.transform(author_list)

print(author_list_encoded)





####One Hot encoding

#onehot_encoder = preprocessing.OneHotEncoder()

#author_list_oh_encoded = onehot_encoder.fit_transform(author_list_encoded)



#author_list

#author_list_encoded

#print(author_list_oh_encoded.size)



########Setting up tensor flow

learning_rate = 0.05

epochs = 3

batch_size = 2000



# declare the training data placeholders

# There are a total of 19579 examples 

# There are a total of 25068 inputs

x = tf.placeholder(tf.float32, [None, 25068])



# 3 digits for the 3 authors

y = tf.placeholder(tf.float32, [None, 3])



###declare number of nodes and layers

inputs=25068

hidden_layer_1_nodes=100

hidden_layer_2_nodes=100

output_layer=3



####declare the weights and biases for the nueral network



# Weights and biases between input layer and hidden layer 1 

W1 = tf.Variable(tf.random_normal([inputs,hidden_layer_1_nodes], stddev=0.03), name='W1')

b1 = tf.Variable(tf.random_normal([hidden_layer_1_nodes]), name='b1')



# Weights and biases between hidden layer 1 and hidden layer2

W2 = tf.Variable(tf.random_normal([hidden_layer_1_nodes,hidden_layer_2_nodes], stddev=0.03), name='W2')

b2 = tf.Variable(tf.random_normal([hidden_layer_2_nodes]), name='b2')



# Weights and biases between hidden layer 2 and ourput layer

W3 = tf.Variable(tf.random_normal([hidden_layer_2_nodes,output_layer], stddev=0.03), name='W3')

b3 = tf.Variable(tf.random_normal([output_layer]), name='b3')





####Calculate output of hidden layer 1

hidden_out_1 = tf.add(tf.matmul(x, W1), b1)

hidden_out_1 = tf.nn.sigmoid(hidden_out_1)



####Calculate output of hidden layer 2

hidden_out_2 = tf.add(tf.matmul(hidden_out_1, W2), b2)

hidden_out_2 = tf.nn.sigmoid(hidden_out_2)



####Calculate output of output layer

final_output = tf.nn.softmax(tf.add(tf.matmul(hidden_out_2, W3), b3))

final_output=tf.nn.sigmoid(final_output)





#limit the output to 1e-10, 0.9999999 to prevent o from being retutned

final_output_clipped = tf.clip_by_value(final_output, 1e-10, 0.9999999)



cross_entropy = -tf.reduce_mean(tf.reduce_sum(y * tf.log(final_output_clipped)

                         + (1 - y) * tf.log(1 - final_output_clipped), axis=1))





optimiser = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cross_entropy)



# varaibles initialzer

init_op = tf.global_variables_initializer()



# define an accuracy assessment operation

correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(final_output, 1))

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))



X_scale_train,X_scale_test,Y_scale_train,Y_scale_test = train_test_split(sentence_train_tfidf,author_list_encoded,test_size=0.3,random_state=78)



y_onehot_labels_train = tf.one_hot(Y_scale_train, 3)

y_onehot_labels_test = tf.one_hot(Y_scale_test, 3)



X_scale_train_toarray=X_scale_train.toarray()

X_scale_test_toarray=X_scale_test.toarray()





#######prepare the preidction data

df_spooky_author_test=pd.read_csv('../input/test.csv')

df_spooky_author_test



sentence_list_test=df_spooky_author_test['text'].values.tolist()

id_list_test=df_spooky_author_test['id'].values.tolist()





#####Very important use transorm here instread of fit transofrm

#####Training data has been fitted above hence here we need to use only transform

##### If fit transform is used we get dimensional mismatch while making predictions from the trained model

sentence_test_count=count_vect.transform(sentence_list_test)

sentence_test_count.shape



sentence_test_tfidf = tfidf_transformer.transform(sentence_test_count)

sentence_test_tfidf_to_array=sentence_test_tfidf.toarray()



prediction=tf.argmax(y,1)







# start the session

with tf.Session() as sess:

    # initialise the variables

    sess.run(init_op)

    total_batch = 1 ##int(len(Y_scale_train) / batch_size)

    

    

    for epoch in range(epochs):

        print("Epoch:", (epoch + 1))

        avg_cost = 0

        for i in range(total_batch):

            print('Batch:-',i)

            _, c = sess.run([optimiser, cross_entropy], 

                         feed_dict={x: X_scale_train_toarray, y: y_onehot_labels_train.eval()}) 

        avg_cost += c / total_batch

        print("Epoch:", (epoch + 1), "cost =", "{:.3f}".format(avg_cost))

    print(sess.run(accuracy, feed_dict={x: X_scale_test_toarray , y: y_onehot_labels_test.eval()}))        

    ##print(sess.run(y,prediction.eval(feed_dict={x: sentence_test_tfidf_to_array,y:})))

    final_pred=sess.run(final_output, feed_dict={x: sentence_test_tfidf_to_array})   

    

    

####Prepareoutput forsubmission

df_X_prob_predict=pd.DataFrame(final_pred)

df_X_prob_predict.columns=['EAP', 'HPL', 'MWS']



df_X_prob_predict

df_X_ID=pd.DataFrame(id_list_test)

df_X_ID.columns=['id']

df_predict_final=df_X_ID.join(df_X_prob_predict)

df_predict_final



###Export Data to csv for final submission

df_predict_final.to_csv('hravat_spooky_autor_predict.csv',sep=',')




