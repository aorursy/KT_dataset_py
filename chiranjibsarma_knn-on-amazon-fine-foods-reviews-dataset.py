# import packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import string
import sqlite3
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import SnowballStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score,precision_score,recall_score,confusion_matrix,classification_report,f1_score
from sklearn.decomposition import TruncatedSVD
from sklearn import cross_validation
from sklearn.cross_validation import cross_val_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
import gensim
# creating sql connection string
con = sqlite3.connect('../input/database.sqlite')
#Positive Review - Rating above 3
#Negative Review - Rating below 3
#Ignoring Reviews with 3 Rating

filtered_data = pd.read_sql_query('SELECT * from Reviews WHERE Score != 3',con)
filtered_data.head(5)
# mapping ratings above 3 as Positive and below 3 as Negative

actual_scores = filtered_data['Score']
positiveNegative = actual_scores.map(lambda x: 'Positive' if x>3 else 'Negative')
filtered_data['Score'] = positiveNegative
filtered_data.head(5)
# Sorting values according to Time for Time Based Slicing
sorted_values = filtered_data.sort_values('Time',kind = 'quicksort')
final = sorted_values.drop_duplicates(subset= { 'UserId', 'ProfileName', 'Time',  'Text'})
print('Rows dropped : ',filtered_data.size - final.size)
print('Percentage Data remaining after dropping duplicates :',(((final.size * 1.0)/(filtered_data.size * 1.0) * 100.0)))
# Dropping rows where HelpfulnessNumerator < HelpfulnessDenominator
final = final[final.HelpfulnessDenominator >= final.HelpfulnessNumerator]
print('Number of Rows remaining in the Dataset: ',final.size)
# Checking the number of positive and negative reviews
final['Score'].value_counts()
# Data Sampling
final = final.iloc[:70000,:]
print(final.shape)
print(final['Score'].value_counts())
# Function to Remove HTML Tags
def cleanhtml(sentence):
    cleaner = re.compile('<.*?>')
    cleantext = re.sub(cleaner,"",sentence)
    return cleantext
# Function to clean punctuations and special characters

def cleanpunct(sentence):
    cleaned = re.sub(r'[?|!|\'|"|#]',r'',sentence)
    cleaned = re.sub(r'[.|,|)|(|\|/]',r' ',cleaned)
    return  cleaned

# Initialize Stop words and PorterStemmer and Lemmetizer
stop = set(stopwords.words('english'))
sno = SnowballStemmer('english')


print(stop)
print('*' * 100)
print(sno.stem('tasty'))
# Cleaning HTML and non-Alphanumeric characters from the review text
i=0
str1=' '
final_string=[]
all_positive_words=[] # store words from +ve reviews here
all_negative_words=[] # store words from -ve reviews here.
s=''
for sent in final['Text'].values:
    filtered_sentence=[]
    #print(sent);
    sent=cleanhtml(sent) # remove HTMl tags
    for w in sent.split():
        for cleaned_words in cleanpunct(w).split():
            if((cleaned_words.isalpha()) & (len(cleaned_words)>2)):    
                if(cleaned_words.lower() not in stop):
                    s=(sno.stem(cleaned_words.lower())).encode('utf8')
                    filtered_sentence.append(s)
                    if (final['Score'].values)[i] == 'positive': 
                        all_positive_words.append(s) #list of all words used to describe positive reviews
                    if(final['Score'].values)[i] == 'negative':
                        all_negative_words.append(s) #list of all words used to describe negative reviews reviews
                else:
                    continue
            else:
                continue 
    #print(filtered_sentence)
    str1 = b" ".join(filtered_sentence) #final string of cleaned words
    #print("***********************************************************************")
    
    final_string.append(str1)
    i+=1
final['CleanedText']=final_string
final.head(5)

# Function for KNN
def runKNN(X_tr_input,x_cv_input,y_tr_input,y_cv_input,VectorizationType):
    
    cv_scores = []
    algorithm = ['kd_tree','brute']
    for algo in algorithm:
        #print(algo)
        # kd_tree cannot consume Sparse Matrix. Converting Sparse Matrix to Dense using Truncated SVD.
        if algo == 'kd_tree':
            svd = TruncatedSVD()
            X_tr_input1 = svd.fit_transform(X_tr_input)
            x_cv_input1 = svd.fit_transform(x_cv_input)
            X_tr_input = X_tr_input1
            x_cv_input = x_cv_input1
            #print(type(x_cv_input))
        
        for i in range(1,30,2):
        # instantiate learning model (k = 30)
            knn = KNeighborsClassifier(n_neighbors=i,algorithm = algo)

            scores = cross_val_score(knn, X_tr_input, y_tr_input, cv=10, scoring='accuracy')
            cv_scores.append(scores.mean())


        #print(cv_scores)
        #print(max(cv_scores))
        k_optimum = cv_scores.index(max(cv_scores)) +1

        knn = KNeighborsClassifier(n_neighbors=k_optimum,algorithm = algo)
        # fitting the model on crossvalidation train
        knn.fit(X_tr_input, y_tr_input)

        # predict the response on the crossvalidation train
        pred = knn.predict(x_cv_input)

        # evaluate CV accuracy
        #acc = accuracy_score(y_cv_input, pred, normalize=True) * float(100)
        print('Details for ',VectorizationType,'Vectorization:')
        print('*'*100)
        print('Accuracy for',algo,' algorithm with K =',k_optimum,' is ' ,np.round(accuracy_score(y_cv_input, pred)*100))
        print('F1 score for',algo,' algorithm with K =',k_optimum,' is ' , np.round(f1_score(y_cv_input, pred,average= 'macro')*100))
        print('Recall for',algo,' agorithm with K =',k_optimum,' is ' , np.round(recall_score(y_cv_input, pred,average= 'macro')*100))
        print('Precision for',algo,' algorithm with K =',k_optimum,' is ' , np.round(precision_score(y_cv_input, pred,average= 'macro')*100))
        print ('\n clasification report for',algo,' algorithm with K =',k_optimum,' is \n ' , classification_report(y_cv_input,pred))
        print ('\n confussion matrix for',algo,' algorithm with K =',k_optimum,' is \n' ,confusion_matrix(y_cv_input, pred))
# BoW Vectorization

# split the data set into train and test
X_1, X_test, y_1, y_test = cross_validation.train_test_split(final['CleanedText'], final['Score'], random_state = 0,test_size = 0.3)

# split the train data set into cross validation train and cross validation test
X_tr, X_cv, y_tr, y_cv = cross_validation.train_test_split(X_1, y_1, test_size=0.3)



vect = CountVectorizer().fit(final['CleanedText'])
#print(vect.get_feature_names()[::2000])
#print(len(vect.get_feature_names()))
X_tr_vectorized = vect.transform(X_tr)
x_cv_vectorized = vect.transform(X_cv)

runKNN(X_tr_vectorized,x_cv_vectorized,y_tr,y_cv,'Bag of Words')
# Applying TFIDF

vect_tfidf = TfidfVectorizer(min_df = 5).fit(final['CleanedText'])
#print(len(vect.get_feature_names()))

#print(vect_tfidf.get_feature_names()[::2000])
#print(len(vect_tfidf.get_feature_names()))

# Vectorizing the datsets
X_tr_vectorized = vect_tfidf.transform(X_tr)
x_cv_vectorized = vect_tfidf.transform(X_cv)

runKNN(X_tr_vectorized,x_cv_vectorized,y_tr,y_cv,'TF-IDF')
# Train your own Word2Vec model using your own text corpus

i=0
list_of_sent=[]
for sent in final['Text'].values:
    filtered_sentence=[]
    sent=cleanhtml(sent)
    for w in sent.split():
        for cleaned_words in cleanpunct(w).split():
            if(cleaned_words.isalpha()):    
                filtered_sentence.append(cleaned_words.lower())
            else:
                continue 
    list_of_sent.append(filtered_sentence)
    
'''print(final['Text'].values[0])
print("*****************************************************************")
print(list_of_sent[0])'''
w2v_model=gensim.models.Word2Vec(list_of_sent,min_count=5,size=50, workers=4)    
words = list(w2v_model.wv.vocab)
#print(len(words))
# average Word2Vec
# compute average word2vec for each review.
sent_vectors = []; # the avg-w2v for each sentence/review is stored in this list
for sent in list_of_sent: # for each review/sentence
    sent_vec = np.zeros(50) # as word vectors are of zero length
    cnt_words =0; # num of words with a valid vector in the sentence/review
    for word in sent: # for each word in a review/sentence
        try:
            vec = w2v_model.wv[word]
            sent_vec += vec
            cnt_words += 1
        except:
            pass
    sent_vec /= cnt_words
    sent_vectors.append(sent_vec)
#print(len(sent_vectors))
#print(len(sent_vectors[0]))

X_1, X_test, y_1, y_test = cross_validation.train_test_split(sent_vectors, final['Score'], random_state = 0,test_size = 0.3)
#print('X_train first entry: \n\n', X_1[0])
#print('\n\nX_train shape: ', X_1.shape)

# split the train data set into cross validation train and cross validation test
X_tr, X_cv, y_tr, y_cv = cross_validation.train_test_split(X_1, y_1, test_size=0.3)


runKNN(X_tr_vectorized,x_cv_vectorized,y_tr,y_cv,'Average Word2Vec')
# TF-IDF weighted Word2Vec
tfidf_feat = vect_tfidf.get_feature_names() # tfidf words/col-names
# final_tf_idf is the sparse matrix with row= sentence, col=word and cell_val = tfidf

tfidf_sent_vectors = []; # the tfidf-w2v for each sentence/review is stored in this list
row=0;
for sent in list_of_sent: # for each review/sentence
    sent_vec = np.zeros(50) # as word vectors are of zero length
    weight_sum =0; # num of words with a valid vector in the sentence/review
    for word in sent: # for each word in a review/sentence
        try:
            vec = w2v_model.wv[word]
            # obtain the tf_idfidf of a word in a sentence/review
            tfidf = vect_tfidf[row, tfidf_feat.index(word)]
            sent_vec += (vec * tf_idf)
            weight_sum += tf_idf
        except:
            pass
    
    #print(type(sent_vec))
    try:
        sent_vec /= weight_sum
    except:
        pass
    
    tfidf_sent_vectors.append(sent_vec)
    row += 1
    

    
X_1, X_test, y_1, y_test = cross_validation.train_test_split(tfidf_sent_vectors, final['Score'], random_state = 0,test_size = 0.3)
#print('X_train first entry: \n\n', X_1[0])
#print('\n\nX_train shape: ', X_1.shape)

# split the train data set into cross validation train and cross validation test
X_tr, X_cv, y_tr, y_cv = cross_validation.train_test_split(X_1, y_1, test_size=0.3)

runKNN(X_tr_vectorized,x_cv_vectorized,y_tr,y_cv,'TF-IDF weighted Word2Vec')
    

