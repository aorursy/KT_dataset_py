#!pip install -U gensim
import warnings
warnings.filterwarnings('ignore')
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
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV,RandomizedSearchCV
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from scipy.stats import expon
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
final = final.iloc[:5000,:]
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

#import nltk
#nltk.download()
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
                    if (final['Score'].values)[i] == 'Positive': 
                        all_positive_words.append(s) #list of all words used to describe positive reviews
                    if(final['Score'].values)[i] == 'Negative':
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
#Split data into Train and Test Set
X_Train,X_Test,y_train,y_test = train_test_split(final['CleanedText'],final['Score'],random_state = 0,test_size = 0.3)

# Function to run SVC with GridSearchCV and RandomSearchCV
def RunSVC(X_Train,X_Test,y_train,y_test,Search_Type):    
    lb_make = LabelEncoder()
    
    y_train_encoded = lb_make.fit_transform(y_train)
    y_test_encoded = lb_make.fit_transform(y_test)
    
    
    if (Search_Type == 'grid'):
        grid_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],'C': [1, 10, 100, 1000]}]
        model = GridSearchCV(SVC(),grid_parameters,cv = 5,scoring = 'f1')
        model.fit(X_Train,y_train_encoded)
        print(model.best_estimator_)
        print('The Score with '+ Search_Type+ 'search CV is: '+ str(model.score(X_Test, y_test_encoded)))
    elif (Search_Type == 'random'):
        random_parameters = dict(C=[1, 10, 100, 1000],gamma=[1e-3, 1e-4])  
        model = RandomizedSearchCV(SVC(),random_parameters,cv = 5,scoring = 'f1',n_jobs= 1)
        model.fit(X_Train,y_train_encoded)
        print(model.best_estimator_)
        print('The Score with '+ Search_Type+ 'search CV is: ' + str(model.score(X_Test, y_test_encoded)))
# BoW Vectorization

vect = CountVectorizer().fit(X_Train)
X_Train_vectorised = vect.transform(X_Train)
X_Test_vectorised = vect.transform(X_Test)


RunSVC(X_Train_vectorised,X_Test_vectorised,y_train,y_test,'grid')
RunSVC(X_Train_vectorised,X_Test_vectorised,y_train,y_test,'random')


# Applying TFIDF

vect_tfidf = TfidfVectorizer(min_df = 5).fit(X_Train)
X_Train_vectorised = vect_tfidf.transform(X_Train)
X_Test_vectorised = vect_tfidf.transform(X_Test)
RunSVC(X_Train_vectorised,X_Test_vectorised,y_train,y_test,'grid')
RunSVC(X_Train_vectorised,X_Test_vectorised,y_train,y_test,'random')
'''print(final['Text'].values[0])
print("*****************************************************************")
'''
print(list_of_sent[0])

w2v_model=gensim.models.Word2Vec(list_of_sent,min_count=5,size=50, workers=4)    
words = list(w2v_model.wv.vocab)
#print(len(words))


