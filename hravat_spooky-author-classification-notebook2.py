##import necessary packages

#####



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

from nltk.stem import WordNetLemmatizer



## TF-IDF implementation

count_vect = CountVectorizer()

wordnet_lemmatizer = WordNetLemmatizer()



sentence_new=''

sentence_new_list=[]

sentence_list_lemmatized=[]



for sentence in sentence_list:

    for words in nltk.word_tokenize(sentence):

        sentence_new=sentence_new+wordnet_lemmatizer.lemmatize(words)

    sentence_list_lemmatized.append(sentence_new)

    sentence_new=''  

    

 

sentence_train_counts=count_vect.fit_transform(sentence_list_lemmatized)

sentence_train_counts.shape



tfidf_transformer = TfidfTransformer()

sentence_train_tfidf = tfidf_transformer.fit_transform(sentence_train_counts)

sentence_train_tfidf.shape



##Label Encoder to encode values

le = preprocessing.LabelEncoder()

le.fit(author_list)



list(le.classes_)



author_list_encoded=le.transform(author_list)

author_list_encoded



X_scale_train,X_scale_test,Y_scale_train,Y_scale_test = train_test_split(sentence_train_tfidf,author_list_encoded,test_size=0.3,random_state=78)



C_list=[0.0001,0.0003,0.001,0.003,0.01,0.01,0.1,0.3,1,3]

accuracy=0

c_final=0

accuracy_1=0



for c in C_list:

    mul_lr = linear_model.LogisticRegression(penalty='l2',multi_class='multinomial', solver='newton-cg',C=c)

    mul_lr.fit(X_scale_train,Y_scale_train)    

    accuracy_1=accuracy_score(Y_scale_test,mul_lr.predict(X_scale_test)) 

    accuracy_1

    if  accuracy_1 > accuracy:  

        accuracy=accuracy_1                     

        c_final=c

        ##Uncomment below line and run to see how acuuracy improves with change in value of c

        print ('C is',c,'Accuracy is ',accuracy_score(Y_scale_test,mul_lr.predict(X_scale_test)))

    



    

mul_lr = linear_model.LogisticRegression(penalty='l2',multi_class='multinomial', solver='newton-cg',C=c)

mul_lr.fit(X_scale_train,Y_scale_train)    

accuracy_1=accuracy_score(Y_scale_test,mul_lr.predict(X_scale_test))     

accuracy_1





y_predict=mul_lr.predict(X_scale_test)

y_predict_author=le.inverse_transform(y_predict)

##list of predicted authors 

y_predict_author



###Predict Probabilities

mul_lr.predict_proba(X_scale_test)



########Preparing the data for training predictions

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

sentence_test_tfidf.shape



X_prob_predict=mul_lr.predict_proba(sentence_test_tfidf)

df_X_prob_predict=pd.DataFrame(X_prob_predict)

df_X_prob_predict.columns=['EAP', 'HPL', 'MWS']



df_X_prob_predict

df_X_ID=pd.DataFrame(id_list_test)

df_X_ID.columns=['id']

df_predict_final=df_X_ID.join(df_X_prob_predict)

df_predict_final



###Export Data to csv for final submission

df_predict_final.to_csv('hravat_spooky_autor_predict.csv',sep=',')
