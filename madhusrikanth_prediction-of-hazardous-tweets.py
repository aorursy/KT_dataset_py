import pandas as pd
import numpy as np #Importing data libraries

#Desc
#you’re challenged to build a machine learning model that predicts which Tweets are about real disasters(words) and which one’s aren’t. 
import seaborn as sns #Importing data visualization
df_tweet = pd.read_csv('../input/nlp-getting-started/train.csv') #This is data acquistion process
df_tweet
# Tweet spam detection 
#To import nltk
import nltk
nltk.download_shell() #first enter l
                      #Keep on to press enter 
                      #again enter download enter d
                      #then in identifier enter stopwords
                      #q is to quit the shell
                      #after process # the nltk.download....
            
            
            
#read the file
tweet_mess = [line.rstrip() for line in open('../input/nlp-getting-started/train.csv')]#write everything in small before/ and after tht wirte orginal file name
tweet_mess[5] #to get value for particular row  just specify the index                                                                         #to grab message as text of messages in a list form
print(len(tweet_mess)) #lenght of instance created
#To check for nan values in dataset
sns.heatmap(df_tweet.isnull(),yticklabels=False,cbar=False,cmap='viridis')
#Insights: Location has largest number of empty set so we can even drop column if not necessary
           #keyword has next level nan values
df_tweet.drop('location',axis=1,inplace=True) #need to specify the axis as w e r deleting from column ,which is for column axis=1
df_tweet.head() #only to get top 5 values
df_tweet.describe() #max count of id is 7613 , there are not messages which are unique
df_tweet.groupby('target').describe() #if we use groupby we can seperately get values for target 0 and 1
                                      #Count of target 0(which is  not real diasaster)greater then count of target 1(which is about real disaster)
                                      #There is might be max percent for target 1 (they might have tweeted (real disaster words) than the target .)
#Feature enginerring(features are very imp in nlp) Rigth feature or attribute will improve the precision of Machine learning Algorithms result
df_tweet['lenght'] = df_tweet['text'].apply(len) #To find lenght using len inbuilt method and use apply method to apply it to dataframe created
df_tweet.head()
df_tweet['lenght'].plot.hist(bins=50) #to compare lenght of texts , max lenght of texts is of 160 which has largest frequency of occurence in given dataset 
df_tweet['lenght'].describe() #lenght of text(maxlenght) is 170
df_tweet[df_tweet['lenght']==157]['text'].iloc[0] #this is masking 
                                                  #to get entire text speciify the iloc(0) by grabbing text column in dataframe
#Facet Grid
df_tweet.hist(column='lenght',by='target',bins=60,figsize=(12,4)) #here by specifies sepration (as well as there seperated graph x axis)
                                          #we have to specify y
                                          #Even though they might look similar target 0 has higher lenght as compared to target one as we can see x axis values between 120-3
#Insights
#Target 0 MESSAGES TEND TO HAVE MORE CHARACTERS.
#Target 1 MESSAGES TEND TO HAVE LESS AS COMPARED.
#For this dataset lenght is good feature
#algorithms
#BAG OF WORDS
#Convert raw messages into sequence of characters
#Also remove common words such as a ,the..which are the  stopwords
#Use python built in string function\

import string
from nltk.corpus import stopwords #to check for stopwords

stopwords.words('english')
#we r going to define fucntion
def text_process(mess):
    ""
    #remove punctuation
    #remove stopwords
    #return clean text words
    ""
    
    nopunc = [char for char in mess if char not in string.punctuation]
    nopunc = ''.join(nopunc)
    return [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]
df_tweet['text'].head(5).apply(text_process) #First to test to 5 most top values 
                                             #This is clean text without stop words
                                            #Returned as lisrt
#Vectorization
#Convert each token to vectorization which can be learned by machine language algorithm 
#Count words
#Frequent token is weighed less
#Import Count vectorization model
#Sparse matrix
from sklearn.feature_extraction.text import CountVectorizer
#Bag of words
bow = CountVectorizer(analyzer=text_process).fit(df_tweet['text'])#where analyzer is func created above 
                                                                      #Large matrix will be created and takes time
                                                                       #we have to fit this for emtire text in the given dataset
bow
print(len(bow.vocabulary_)) #clenght of words in dataset
#get one message and get count
text4 = df_tweet['text'][3] #prints text 6
text4
bow4 = bow.transform([text4]) #pass text4 as list which will result in sparse matrix
print(bow4) #here in given text all letter appear only once
#to get letter which appeared at index 2334
bow.get_feature_names()[2334] #in [] we have to pass in index of columnn which is repeated or which we want to get result for
print(bow4.nnz)#of non zero occurences
print(bow4.shape) #prints shape
#For enttire text in given dataset
text_bow = bow.transform(df_tweet['text']) #sparse matrix creation
print(text_bow)
#import tfid
from sklearn.feature_extraction.text import TfidfTransformer
#Create an instance
tfid_tranformer = TfidfTransformer().fit(text_bow)#fit sparse matrix
#Only for text 4 (Just an example)
tfid4 = tfid_tranformer.transform(bow4)#here example of bow4 is tranformed to sparse matrix which is transformed to tfid count
                                        #word count trasnformed to inverse frequency
print(tfid4)                            #This is vector values for given text in dataset
#to check document frequency if word university (ex:)
tfid_tranformer.idf_[bow.vocabulary_['ablaze']] #we can specify any word in the text column to find vector value 
#tfid for entire message in dataset
#tfid for entire message in dataset
text_tfid = tfid_tranformer.transform(text_bow) #sparse matrix created for text column  with its corresponding vector values
print(text_tfid)
#as messages are now in vector form which is numerical form we can pass in to machine learning algorithm
from sklearn.naive_bayes import MultinomialNB #naive alogirthm is used(Claasifier)
spam_detect_model = MultinomialNB().fit(text_tfid, df_tweet['target']) #first is created vector data(which is actual tweeted text) second is actual data to which text are compared 0 or 1
spam_detect_model.predict(tfid4)[0] #by specifying index we get 0 or 1 for praticular text
df_tweet['target'][3] #predicted answer is right
#Above model is predicts value for each and every data so now we have to split the data as train and test data to improve accuracy
from sklearn.model_selection import train_test_split
text_train,text_test,target_train,target_test = train_test_split(df_tweet['text'],df_tweet['target'],test_size=0.3,random_state=101)
from sklearn.pipeline import Pipeline # Using Data pipeline the above steps can be combined into single step

word = Pipeline([
    ('bow', CountVectorizer(analyzer=text_process)),  # strings to token integer counts
    ('tfidf', TfidfTransformer()),  # integer counts to weighted TF-IDF scores
    ('classifier', MultinomialNB()),  # train on TF-IDF vectors w/ Naive Bayes classifier
]) #first argument in tuple can be any string which is label specifying the message
word.fit(text_train,target_train) #train data
answer = word.predict(text_test)
print(answer)
from sklearn.metrics import classification_report
print(classification_report(target_test,answer)) #precision is about 80 percent correct, this implies our model performs well
df_tweet 
#df_tweet.drop(['keyword','target','text'],inplace=True,axis=1) #drop unnessary columns
df_tweet['target'] = pd.Series(answer)
#df_tweet.drop('lenght',axis=1,inplace=True) #Is the final set of data which is indicated with its id value
#df_tweet.drop('answer',axis=1,inplace=True)
print(df_tweet.head(3262))
from sklearn.metrics import confusion_matrix

print(confusion_matrix(target_test,answer)) #TN has better value which is 1198 model seems to be better and is predicted to happen

