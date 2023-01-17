#loading all necessary packages

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns

from IPython.display import display


from pylab import *
from wordcloud import WordCloud, STOPWORDS 
import collections


from keras.models import Sequential
from keras.layers import Dense, Input, Dropout, LSTM, Activation, Embedding, Bidirectional
from keras.preprocessing.text import Tokenizer
from keras.initializers import Constant
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.utils import to_categorical

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import confusion_matrix
import sklearn.metrics as mt
import warnings
warnings.filterwarnings('ignore')

#Loading and cleaning the data
review=pd.read_csv('../input/womens-ecommerce-clothing-reviews/Womens Clothing E-Commerce Reviews.csv')

print ('Raw data input:')
display(review.head())  
#print (review.head().to_html())

review=review[['Title','Review Text','Rating','Recommended IND']]

#converting Rating and Recommendation to categorical variables

review['Rating'] = review['Rating'].astype('category')
review['Recommended IND'] = review['Recommended IND'].astype('category')
print ('Checking dataframe data type:',review.dtypes)

print ("Saving subset of the dataset by dropping few features:")
display(review.head())
#Summary of data

print ('Summary of the dataset')
review.info()

review.describe()

# Distribution of rating
print ('Distribution of review text by Rating and Recommendation:')

print (review.groupby(['Recommended IND','Rating'])['Review Text'].count())

print ('\n')
print ('Number of positive and negative recommendations')
print (review.groupby(['Recommended IND'])['Recommended IND'].count())

print ('\n')
print ('Count of different ratings')
print (review.groupby(['Rating'])['Rating'].count())

#barchart showing the distribution of rating and recommendations
f, ax = plt.subplots()
f.set_size_inches(20,10)

#plt.plot(x,y,'b--',label=)


sns.countplot(x="Recommended IND", hue='Rating', data=review)

plt.xlabel("Recommendations",fontsize='large')
plt.ylabel('Count', fontsize='large')
#plt.xlim(50,80)
#plt.ylim(58,62)
plt.rcParams.update({'font.size':12})
matplotlib.rc('xtick', labelsize=20)
matplotlib.rc('ytick', labelsize=20)
plt.title('Distribution of rating and recommendations' )
ax.legend(loc='best',fontsize='large')
#plt.savefig('Distribution of rating and recommendations' +  '.png', dpi=300, bbox_inches='tight')
plt.show()

 


#number of missing titles and reviews

miss_title = review.describe().loc['count','Rating']-review.describe().loc['count','Title']
miss_review = review.describe().loc['count','Rating']-review.describe().loc['count','Review Text']

print ('No. of missing titles:',miss_title)
print ('No. of missing reviews:',miss_review)
# subset of the dataset

recc = review[review['Recommended IND']==1]  #reviews that were recommendation
non_recc=review[review['Recommended IND']==0] #reviews that were not recommendation

surp_recc=review[((review['Recommended IND']==1) & (review['Rating']==1)) | ((review['Recommended IND']==1) & (review['Rating']==2))] 
#reviews that were recommended but with poor rating

surp_non_recc=review[((review['Recommended IND']==0) & (review['Rating']==4)) | ((review['Recommended IND']==0) & (review['Rating']==5))]
#reviews that were not recommended but had high rating
#length of words

def review_length(string):
   
    '''
    Measures the length of the review for each entry
    
    Arguments:
    string -- Input string for each review
    
    Returns:
    word_length -- Number of words in each review
    '''

    word_length = len(str(string).split())
    return word_length


recc.loc[:,'Length'] = recc.loc[:,'Review Text'].apply(review_length)
non_recc.loc[:,'Length'] = non_recc.loc[:,'Review Text'].apply(review_length)


print ('Average length of review in Recommended reviews:',round(recc.loc[:,'Length'].mean(),2))
print ('Average length of review in Non-Recommended reviews:',round(non_recc.loc[:,'Length'].mean(),2))

# Distribution of word lengths of positve and negative recommendations

f, ax = plt.subplots()
f.set_size_inches(20,10)

#plt.plot(x,y,'b--',label=)
sns.distplot( recc["Length"] , color="skyblue", label="Recommended")
sns.distplot( non_recc["Length"] , color="red", label="Non-Recommended")

plt.xlabel("Word Length",fontsize='large')
plt.rcParams.update({'font.size':12})
matplotlib.rc('xtick', labelsize=20)
matplotlib.rc('ytick', labelsize=20)
plt.title('Distribution of word length ' )
ax.legend(loc='best',fontsize='large')
#plt.savefig('Distribution of rating and recommendations' +  '.png', dpi=300, bbox_inches='tight')
plt.show()

#developing word cloud for Recommended reviews

pos_comment_words = '' 
#adding specific words related to clothing review as stop words
custom_words=set(['dress','fit','size','color', 'will','look','wear','fabric','colors','much','ordered','-','it.','got','top','small','really','one','material','shirt','way','even'])
stopwords = set(STOPWORDS)
stopwords=stopwords.union(custom_words)
  
for rev in recc['Review Text']: 
    
    tokens = str(rev).split() 

    for i in range(len(tokens)): 
        tokens[i] = tokens[i].lower() 
      
    pos_comment_words += " ".join(tokens)+" "
  

wordcloud = WordCloud(width = 800, height = 800, 
                background_color ='white', 
                stopwords = stopwords,
                collocations = False,
                min_font_size = 10).generate(pos_comment_words) 
  
# plot the WordCloud image                        
plt.figure(figsize = (8, 8), facecolor = None) 
plt.imshow(wordcloud) 
plt.axis("off") 
plt.tight_layout(pad = 0) 
  
plt.show() 



#word cloud for non recommended reviews

comment_words = '' 
#adding specific words related to clothing review as stop words
custom_words=set(['dress','fit','size','color', 'will','look','wear','fabric','colors','much','ordered','-','it.','got','top','small','really','one','material','shirt','way','even','looks','looked'])
stopwords = set(STOPWORDS)
stopwords=stopwords.union(custom_words) 
  
for rev in non_recc['Review Text']: 
    
    tokens = str(rev).split() 

    for i in range(len(tokens)): 
        tokens[i] = tokens[i].lower() 
      
    comment_words += " ".join(tokens)+" "
  

wordcloud = WordCloud(width = 800, height = 800, 
                background_color ='white', 
                stopwords = stopwords,
                collocations = False,
                min_font_size = 10).generate(comment_words) 
  
# plot the WordCloud image                        
plt.figure(figsize = (8, 8), facecolor = None) 
plt.imshow(wordcloud) 
plt.axis("off") 
plt.tight_layout(pad = 0) 
  
plt.show() 

#visualizing the distribution of words

#counting number of words in negative recommendations
neg=comment_words.split()
neg_counter=collections.Counter(neg)
#counting number of words in positive recommendations
pos=pos_comment_words.split()
pos_counter=collections.Counter(pos)

#removing stopwords from the dictionaries
for word in stopwords:
    try:
        neg_counter.pop(word)
    except:
        pass
    try:
        pos_counter.pop(word)
    except:
        pass

    
    
#merging the positive and negative word dataframes
neg_df=pd.DataFrame(list(neg_counter.items()),columns = ['Words','Neg_Count']) 
pos_df=pd.DataFrame(list(pos_counter.items()),columns = ['Words','Pos_Count']) 

combined_df=pd.merge(pos_df,neg_df,on='Words',how='outer')
combined_df = combined_df.fillna(0)
combined_df['Difference']=combined_df['Pos_Count']-combined_df['Neg_Count']

combined_df = combined_df.sort_values(by=['Difference'], ascending=True)
strong_neg =combined_df.head(20)

strong_neg['Difference']=abs(strong_neg['Difference'])
strong_pos=combined_df.tail(20)

#barchart showing the distribution of more frequently observed words in negative reviews
f, ax = plt.subplots()
f.set_size_inches(20,10)

#plt.plot(x,y,'b--',label=)

sns.barplot(x="Words", y='Difference', data=strong_neg)

plt.xlabel("Higher Count of Words in Negative Review",fontsize='large')
plt.ylabel('Count', fontsize='large')
plt.xticks(rotation=90)
plt.rcParams.update({'font.size':12})
matplotlib.rc('xtick', labelsize=20)
matplotlib.rc('ytick', labelsize=20)
plt.title('Count of words occuring more in Negative Reviews' )
ax.legend(loc='best',fontsize='large')
#plt.savefig('Distribution of rating and recommendations' +  '.png', dpi=300, bbox_inches='tight')
plt.show()



#barchart showing the distribution of more frequently observed words in positive reviews
f, ax = plt.subplots()
f.set_size_inches(20,10)

#plt.plot(x,y,'b--',label=)

sns.barplot(x="Words", y='Difference', data=strong_pos)

plt.xlabel("Higher Count of Words in Positive Review",fontsize='large')
plt.ylabel('Count', fontsize='large')
plt.xticks(rotation=90)
plt.rcParams.update({'font.size':12})
matplotlib.rc('xtick', labelsize=20)
matplotlib.rc('ytick', labelsize=20)
plt.title('Count of words occuring more in Positive Reviews' )
ax.legend(loc='best',fontsize='large')
#plt.savefig('Distribution of rating and recommendations' +  '.png', dpi=300, bbox_inches='tight')
plt.show()
#Word Preprocessing
#creating subset of the dataframe

recc_lstm=review[['Review Text','Rating','Recommended IND']]
recc_lstm=recc_lstm.dropna()  #dropping missing text cases in reviews

#converting the panda series to numpy array
X=recc_lstm['Review Text']
X=np.array(X)
Y=recc_lstm['Recommended IND']
Y=np.array(Y)

#tokenizing the strings
tokenizer=Tokenizer()
tokenizer.fit_on_texts(X)
sequencer=tokenizer.texts_to_sequences(X)

#finding maximum length of a review

maxLen=0
for string in sequencer:
    temp=len(string)
    if temp>maxLen:
        maxLen=temp

print ('Maximum sequence length:',maxLen)
#creating the word to index and index to word vectors
word_to_index=tokenizer.word_index #dictionary that maps words in the reviews to indices
index_to_word=tokenizer.index_word #dictionary that maps indices back to words

#loading the GloVe embeddings
embeddings_dict = {} #dictionary of words and their correspondng GloVe vector representation
indices=0

with open("../input/glove6b50dtxt/glove.6B.50d.txt", 'r', encoding ='utf8') as f:
    for line in f:
        words = line.split()
        word = words[0]
        vector = np.asarray(words[1:], "float32")
        embeddings_dict[word] = vector
        indices+=1

#preparing embedding matrix
vocab_size=len(word_to_index)+1 #to account for out of vocabulary words
embedding_dim=50 #number of dimensions chosen in the GloVe representation
present=0
absent=0
embedding_matrix = np.zeros((vocab_size, embedding_dim))
for word, i in word_to_index.items():
    #embedding_vector = embeddings_dict[word]
    embedding_vector=  embeddings_dict.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector
        present+=1
    else:
        absent+=1

print ("No. of words in the matrix",present)
print ("No.of missing words",absent)
#preprocessing the text to create indices

def sentences_to_indices(sequencer,maxLen):
    
    '''
    Converts the tokenized sequencer to 2D matrix for each text entry.
    Each row in the matrix is one review. Each column indice indicate one word in the text.
    
    Arguments:
    sequencer -- List of list comprisong of text reviews converted to indices.
    maxLen -- Maximum length of the nested list consisting of text converted to indices in the sequencer list
    
    Returns:
    X_indices -- 2D matrix where each row corresponds to each review. Each column indice correspond to a word in the review.
    '''
    
    X_indices=np.zeros((len(sequencer),maxLen))
    for i in range(len(sequencer)):
        j=0
        for n in sequencer[i]:
             X_indices[i,j]= n
             j+=1
    return X_indices

X_indices=sentences_to_indices(sequencer,maxLen)
#dividing into training and test data set
np.random.seed(2)
X_tr,X_test,Y_tr,Y_test=train_test_split(X_indices,Y,test_size=0.1)


#building the Bidirectional LSTM model
early_stopping=EarlyStopping(monitor='val_loss',patience=5)
model_save=ModelCheckpoint('top_model.hdf5',save_best_only=True)
model=Sequential()
model.add(Embedding(
    input_dim=vocab_size,
    output_dim=embedding_dim,
    input_length=maxLen,
    embeddings_initializer=Constant(embedding_matrix),
    trainable=False,
))
model.add(Bidirectional(LSTM(units = 10, return_sequences= True)))
#model.add(Dropout(rate=0.5))
model.add(Bidirectional(LSTM(units = 10, return_sequences= False)))
model.add(Dense(10,activation='relu'))
model.add(Dense(1,activation='sigmoid'))

model.summary()


#compiling and fitting the model

model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
model_his=model.fit(X_tr, Y_tr, epochs = 400, batch_size = 32, validation_split=0.1, shuffle=True,verbose=True,callbacks=[early_stopping,model_save])


plt.figure()
plt.plot(model_his.history['accuracy'])
plt.plot(model_his.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train','Validation'])
plt.show()

plt.figure()
plt.plot(model_his.history['loss'])
plt.plot(model_his.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train','Validation'])
plt.show()
#evaluating the model on the test data set

print ('Loss=',model.evaluate(X_test,Y_test)[0])
print ('Accuracy=',model.evaluate(X_test,Y_test)[1])
# Predicting the class
pred=model.predict_proba(X_test)
pred_class=np.array([0 if i<0.5 else 1 for i in pred]) #using threshold of 0.5
pred_class=pred_class.reshape(pred_class.shape[0],1)
Y_test=Y_test.reshape((Y_test.shape[0],1))
diff_new=Y_test-pred_class


# ROC curve
Y_test=Y_test.reshape((Y_test.shape[0],1))
fpr_lstm,tpr_lstm,_ = roc_curve(Y_test,pred)
roc_auc_lstm = auc(fpr_lstm,tpr_lstm)

f, ax = plt.subplots()
f.set_size_inches(10,5)
plt.plot(fpr_lstm, tpr_lstm, color='darkred', lw=2, label='ROC curve (area = {:0.2f})'.format(roc_auc_lstm))
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
ax.set(xlim=[-0.01, 1.0], ylim=[-0.01, 1.05])
ax.set(xlabel ='False Positive Rate', ylabel = 'True Positive Rate', title = 'Bidirectional LSTM')
plt.legend(loc='lower right', fontsize=13)

#Confusion Matrix

lstm_cm=np.around(confusion_matrix(pred_class,Y_test),0)


plt.figure(figsize=(10,7))
plt.suptitle("Confusion Matrices",fontsize=24)

plt.title("Bidirectional LSTM")
sns.heatmap(lstm_cm, annot = True,fmt='0g',cmap="Blues",cbar=False)

#Classification Metrics


print("Bidirectional LSTM")
print(mt.classification_report(Y_test, pred_class))

#calculating false positive and negative count
print ('False Negative Count:',count_nonzero(diff_new==1))
print ('False Positive Count:',count_nonzero(diff_new==-1))
print ('Correct Prediction Count:',count_nonzero(diff_new==0))
print ('\n')   

#Identifying wrong predictions and storing the index in two lists

false_pos_index=[]
false_neg_index=[]

for i in range(len(diff_new)):
    if diff_new[i][0]>0:
        false_neg_index.append(i)
    if diff_new[i][0]<0: 
        false_pos_index.append(i)

#Randomly shuffling the index of positive and negative reviews
np.random.seed(7)
np.random.shuffle(false_pos_index)        
np.random.shuffle(false_neg_index)   
        

    

#The index were shuffled in the previous cell. We choose the first 10 reviews.
print ('\n')  
for i in range(10):
     index=false_pos_index[i]
     print ('\n')
     wrong_pred_text=[]  #stores the wrong predicted review text
     for w in X_test[index]:
            if w==0.0:
                 pass
            else:
                 wrong_pred_text.append(index_to_word[w])  #converting index to word
     print (str(index) + ':')
     print (' '.join(wrong_pred_text))  #joining the word
    
#The index were shuffled in the previous cell. We choose the first 10 reviews.
print ('\n')  
for i in range(10):
     index=false_neg_index[i]
     print ('\n')
     wrong_pred_text=[]  #stores the wrong predicted review text
     for w in X_test[index]:
            if w==0.0:
                 pass
            else:
                 wrong_pred_text.append(index_to_word[w])  #converting index to word
     print (str(index) + ':')
     print (' '.join(wrong_pred_text))  #joining the word
    
#preprocessing the review text and one-hot encoding of Ratings

def reclassifying (var):
    
    '''
    Reclassifies the ratings to two categories: 1 (Positive and >3) and 0 (Negative and <3)
    
    Arguments:
    var -- Acts as dummy input for the original rating

    Returns:
    New Rating with O and 1
    '''
    
    if var>3:
        return 1  
    if var<3:
        return 0
    
new_recc_lstm=recc_lstm[recc_lstm['Rating']!=3]
new_recc_lstm['New Rating']=new_recc_lstm['Rating'].apply(reclassifying)

new_recc_lstm['New Rating'] = new_recc_lstm['New Rating'].astype('category')
#recc_lstm['New Rating']=recc_lstm['New Rating'].cat.codes
#Y_rating = to_categorical(recc_lstm['New Rating'])
Y_rating=np.array(new_recc_lstm['New Rating'])

#converting the panda series to numpy array. Recalculating X as rating 3 were excluded.
X_rating=new_recc_lstm['Review Text']
X_rating=np.array(X_rating)

#tokenizing the strings
rat_tokenizer=Tokenizer()
rat_tokenizer.fit_on_texts(X_rating)
sequencer_rating=rat_tokenizer.texts_to_sequences(X_rating)

X_rating_indices=sentences_to_indices(sequencer_rating,maxLen)

r_maxLen=0
for string in sequencer_rating:
    temp=len(string)
    if temp>r_maxLen:
        r_maxLen=temp

print ('Maximum sequence length:',r_maxLen)

X_rating_indices=sentences_to_indices(sequencer_rating,r_maxLen)

#dividing into training and test data set
np.random.seed(199)
X_rating_train,X_rating_test,Y_rating_train,Y_rating_test=train_test_split(X_rating_indices,Y_rating,test_size=0.1)

#recreating the word to index vector and the embedding matrix

rat_word_to_index=rat_tokenizer.word_index #dictionary that maps words in the reviews to indices
rat_index_to_word=rat_tokenizer.index_word #dictionary that maps indices back to words

#preparing embedding matrix
rat_vocab_size=len(rat_word_to_index)+1 #to account for out of vocabulary words
rat_embedding_dim=50 #number of dimensions chosen in the GloVe representation
present=0
absent=0
rat_embedding_matrix = np.zeros((rat_vocab_size, rat_embedding_dim))
for word, i in rat_word_to_index.items():
    rat_embedding_vector=  embeddings_dict.get(word)
    if rat_embedding_vector is not None:
        rat_embedding_matrix[i] = rat_embedding_vector
        present+=1
    else:
        absent+=1

print ("No. of words in the matrix",present)
print ("No.of missing words in the matrix",absent)
#building the Bidirectional LSTM model
rat_early_stopping=EarlyStopping(monitor='val_loss',patience=10)
rat_model_save=ModelCheckpoint('rating_model.hdf5',save_best_only=True)
rat_model=Sequential()
rat_model.add(Embedding(
    input_dim=rat_vocab_size,
    output_dim=rat_embedding_dim,
    input_length=r_maxLen,
    #embeddings_initializer=Constant(rat_embedding_matrix),
    weights=[rat_embedding_matrix],
    trainable=False,
))
rat_model.add(Bidirectional(LSTM(units = 10, return_sequences= True)))
#rat_model.add(Dropout(rate=0.5))
rat_model.add(Bidirectional(LSTM(units = 10, return_sequences= False)))
#rat_model.add(Dropout(rate=0.5))
rat_model.add(Dense(10,activation='relu'))
rat_model.add(Dense(5,activation='relu'))
rat_model.add(Dense(1,activation='sigmoid'))

rat_model.summary()
#compiling and saving the model
rat_model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
rat_model_his=rat_model.fit(X_rating_train,Y_rating_train, epochs = 1000, batch_size = 128, validation_split=0.03, shuffle=True,verbose=True,callbacks=[rat_early_stopping,rat_model_save])


plt.figure()
plt.plot(rat_model_his.history['accuracy'])
plt.plot(rat_model_his.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train','Validation'])
plt.show()

plt.figure()
plt.plot(rat_model_his.history['loss'])
plt.plot(rat_model_his.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train','Validation'])
plt.show()
#evaluating the model on the test data set

print ('Loss=',rat_model.evaluate(X_rating_test,Y_rating_test)[0])
print ('Accuracy=',rat_model.evaluate(X_rating_test,Y_rating_test)[1])


# Predicting the class
rat_pred=rat_model.predict_proba(X_rating_test)
rat_pred_class=np.array([0 if i<0.8 else 1 for i in rat_pred]) #using threshold of 0.8
rat_pred_class=rat_pred_class.reshape(rat_pred_class.shape[0],1)
Y_rating_test=Y_rating_test.reshape((Y_rating_test.shape[0],1))
diff_new_rating=Y_rating_test-rat_pred_class

# ROC curve
fpr_rating_lstm,tpr_rating_lstm,_ = roc_curve(Y_rating_test,rat_pred)
roc_auc_lstm_rating = auc(fpr_rating_lstm,tpr_rating_lstm)

f, ax = plt.subplots()
f.set_size_inches(10,5)
plt.plot(fpr_rating_lstm, tpr_rating_lstm, color='darkred', lw=2, label='ROC curve (area = {:0.2f})'.format(roc_auc_lstm_rating))
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
ax.set(xlim=[-0.01, 1.0], ylim=[-0.01, 1.05])
ax.set(xlabel ='False Positive Rate', ylabel = 'True Positive Rate', title = 'Bidirectional LSTM')
plt.legend(loc='lower right', fontsize=13)

#Confusion Matrix

rating_lstm_cm=np.around(confusion_matrix(rat_pred_class,Y_rating_test),0)


plt.figure(figsize=(10,7))
plt.suptitle("Confusion Matrices",fontsize=24)

plt.title("Bidirectional LSTM")
sns.heatmap(rating_lstm_cm, annot = True,fmt='0g',cmap="Blues",cbar=False)

#Classification Metrics


print("Bidirectional LSTM for Rating")
print(mt.classification_report(Y_rating_test, rat_pred_class))

#calculating false positive and negative count
print ('False Negative Count:',count_nonzero(diff_new_rating==1))
print ('False Positive Count:',count_nonzero(diff_new_rating==-1))
print ('Correct Prediction Count:',count_nonzero(diff_new_rating==0))
print ('\n')   

#Identifying wrong predictions and storing the index in two lists

rating_false_pos_index=[]
rating_false_neg_index=[]

for i in range(len(diff_new_rating)):
    if diff_new_rating[i][0]>0:
        rating_false_neg_index.append(i)
    if diff_new_rating[i][0]<0: 
        rating_false_pos_index.append(i)

#Randomly shuffling the index of positive and negative reviews
np.random.seed(8)
np.random.shuffle(rating_false_pos_index)        
np.random.shuffle(rating_false_neg_index) 
#The index were shuffled in the previous cell. We choose the first 10 reviews.
print ('\n')  
for i in range(10):
     rating_index=rating_false_pos_index[i]
     print ('\n')
     rating_wrong_pred_text=[]  #stores the wrong predicted review text
     for w in X_rating_test[rating_index]:
            if w==0.0:
                 pass
            else:
                 rating_wrong_pred_text.append(rat_index_to_word[w])  #converting index to word
     print (str(rating_index) + ':')
     print (' '.join(rating_wrong_pred_text))  #joining the word
#The index were shuffled in the previous cell. We choose the first 10 reviews.
print ('\n')  
for i in range(10):
     rating_index=rating_false_neg_index[i]
     print ('\n')
     rating_wrong_pred_text=[]  #stores the wrong predicted review text
     for w in X_rating_test[rating_index]:
            if w==0.0:
                 pass
            else:
                 rating_wrong_pred_text.append(rat_index_to_word[w])  #converting index to word
     print (str(rating_index) + ':')
     print (' '.join(rating_wrong_pred_text))  #joining the word
    