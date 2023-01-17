%matplotlib inline

import warnings

warnings.filterwarnings("ignore")



import sqlite3

import pandas as pd

import numpy as np

import nltk

import string

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.feature_extraction.text import TfidfTransformer

from sklearn.feature_extraction.text import TfidfVectorizer



from sklearn.feature_extraction.text import CountVectorizer

from sklearn.metrics import confusion_matrix

from sklearn import metrics

from sklearn.metrics import roc_curve, auc

from nltk.stem.porter import PorterStemmer



import re

# Tutorial about Python regular expressions: https://pymotw.com/2/re/

import string

from nltk.corpus import stopwords

from nltk.stem import PorterStemmer

from nltk.stem.wordnet import WordNetLemmatizer



from gensim.models import Word2Vec

from gensim.models import KeyedVectors

import pickle



from tqdm import tqdm

import os
# using the SQLite Table to read data.

con = sqlite3.connect('../input/database.sqlite') 

#filtering only positive and negative reviews i.e. 

# not taking into consideration those reviews with Score=3

filtered_data = pd.read_sql_query(""" SELECT * FROM Reviews WHERE Score != 3 """, con) 





# Give reviews with Score>3 a positive rating, and reviews with a score<3 a negative rating.

def partition(x):

    if x < 3:

        return 0

    return 1



#changing reviews with score less than 3 to be positive and vice-versa

actualScore = filtered_data['Score']

positiveNegative = actualScore.map(partition) 

filtered_data['Score'] = positiveNegative

print("Number of data points in our data", filtered_data.shape)

filtered_data.head(3)
display= pd.read_sql_query("""

SELECT *

FROM Reviews

WHERE Score != 3 AND UserId="AR5J8UI46CURR"

ORDER BY ProductID

""", con)

display.head()
#Sorting data according to ProductId in ascending order

sorted_data=filtered_data.sort_values('ProductId', axis=0, ascending=True, inplace=False, kind='quicksort', na_position='last')
#Deduplication of entries

final=sorted_data.drop_duplicates(subset={"UserId","ProfileName","Time","Text"}, keep='first', inplace=False)

final.shape
#Checking to see how much % of data still remains

(final['Id'].size*1.0)/(filtered_data['Id'].size*1.0)*100
display= pd.read_sql_query("""

SELECT *

FROM Reviews

WHERE Score != 3 AND Id=44737 OR Id=64422

ORDER BY ProductID

""", con)



display.head()
final=final[final.HelpfulnessNumerator<=final.HelpfulnessDenominator]
#Before starting the next phase of preprocessing lets see the number of entries left

print(final.shape)



#How many positive and negative reviews are present in our dataset?

final['Score'].value_counts()
# find sentences containing HTML tags

import re

i=0;

for sent in final['Text'].values:

    if (len(re.findall('<.*?>', sent))):

        print(i)

        print(sent)

        break;

    i += 1;
import nltk

nltk.download('stopwords')

stop = set(stopwords.words('english')) #set of stopwords

sno = nltk.stem.SnowballStemmer('english') #initialising the snowball stemmer



def cleanhtml(sentence): #function to clean the word of any html-tags

    cleanr = re.compile('<.*?>')

    cleantext = re.sub(cleanr, ' ', sentence)

    return cleantext

def cleanpunc(sentence): #function to clean the word of any punctuation or special characters

    cleaned = re.sub(r'[?|!|\'|"|#]',r'',sentence)

    cleaned = re.sub(r'[.|,|)|(|\|/]',r' ',cleaned)

    return  cleaned

print(stop)

print('************************************')

print(sno.stem('tasty'))
#Code for implementing step-by-step the checks mentioned in the pre-processing phase

# this code takes a while to run as it needs to run on 500k sentences.

if not os.path.isfile('final.sqlite'):

    i=0

    str1=' '

    final_string=[]

    all_positive_words=[] # store words from +ve reviews here

    all_negative_words=[] # store words from -ve reviews here.

    s=''

    for sent in tqdm(final['Text'].values):

        filtered_sentence=[]

        #print(sent);

        sent=cleanhtml(sent) # remove HTMl tags

        for w in sent.split():

            for cleaned_words in cleanpunc(w).split():

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



    #############---- storing the data into .sqlite file ------########################

    final['CleanedText']=final_string #adding a column of CleanedText which displays the data after pre-processing of the review 

    final['CleanedText']=final['CleanedText'].str.decode("utf-8")

        # store final table into an SQlLite table for future.

    conn = sqlite3.connect('final.sqlite')

    c=conn.cursor()

    conn.text_factory = str

    final.to_sql('Reviews', conn,  schema=None, if_exists='replace', \

                 index=True, index_label=None, chunksize=None, dtype=None)

    conn.close()

    

    

    with open('positive_words.pkl', 'wb') as f:

        pickle.dump(all_positive_words, f)

    with open('negitive_words.pkl', 'wb') as f:

        pickle.dump(all_negative_words, f)
if os.path.isfile('../input/final.sqlite'):

    conn = sqlite3.connect('../input/final.sqlite')

    final = pd.read_sql_query(""" SELECT * FROM Reviews WHERE Score != 3 """, conn)

    conn.close()

else:

    print("Please the above cell")
print(len(final["CleanedText"]))
#used to sample the Data

data1 = final.sample(n=10000)



#used to sort the data

data=data1.sort_values('Time', axis=0, ascending=True, inplace=False, kind='quicksort', na_position='last')



#used to divide the class labels

labels = data['Score']

data_1 = data.drop(columns=['Score'])



#used to split the data 

X_train1 = data_1.iloc[0:int(0.7*len(data1))]

y_train = labels.iloc[0:int(0.7*len(data1))]

X_test1 = data_1.iloc[int(0.7*len(data1)):len(data1)]

y_test = labels.iloc[int(0.7*len(data1)):len(data1)]

X_cvdata = data_1.iloc[int(0.7*len(X_train1)):]

y_cvdata = labels.iloc[int(0.7*len(y_train)):]

list_of_sent_train=[]

list_of_sent_cv = []

list_of_sent_test = []



for sent in X_train1['CleanedText'].values:

    list_of_sent_train.append(sent.split())

    

for sent in X_cvdata['CleanedText'].values:

    list_of_sent_cv.append(sent.split())

    

for sent in X_test1['CleanedText'].values:

    list_of_sent_test.append(sent.split())

    

w2v_model = Word2Vec(list_of_sent_train,min_count=5,workers=4,size=50)   

w2v_words = list(w2v_model.wv.vocab)

print(len(w2v_words))
model = TfidfVectorizer()

tf_idf_matrix = model.fit_transform(X_train1['CleanedText'].values)

# we are converting a dictionary with word as a key, and the idf as a value

dictionary = dict(zip(model.get_feature_names(), list(model.idf_)))
tfidf_feat = model.get_feature_names() # tfidf words/col-names

# final_tf_idf is the sparse matrix with row= sentence, col=word and cell_val = tfidf



tfidf_sent_train = []; 

row=0;

for sent in tqdm(list_of_sent_train):

    sent_vec = np.zeros(50) 

    weight_sum =0;

    for word in sent: 

        if word in w2v_words:

            vec = w2v_model.wv[word]

            tf_idf = dictionary[word]*(sent.count(word)/len(sent))

            sent_vec += (vec * tf_idf)

            weight_sum += tf_idf

    if weight_sum != 0:

        sent_vec /= weight_sum

    tfidf_sent_train.append(sent_vec)

    row += 1

    



tfidf_sent_test = []; 

row=0;

for sent in tqdm(list_of_sent_test):

    sent_vec = np.zeros(50) 

    weight_sum =0;

    for word in sent: 

        if word in w2v_words:

            vec = w2v_model.wv[word]

            tf_idf = dictionary[word]*(sent.count(word)/len(sent))

            sent_vec += (vec * tf_idf)

            weight_sum += tf_idf

    if weight_sum != 0:

        sent_vec /= weight_sum

    tfidf_sent_test.append(sent_vec)

    row += 1 
from sklearn.calibration import CalibratedClassifierCV

import seaborn as sn

from collections import Counter

from sklearn.model_selection import GridSearchCV

from sklearn.metrics import accuracy_score

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import train_test_split

from sklearn.metrics import roc_auc_score

from sklearn.metrics import confusion_matrix

import seaborn as sns

from sklearn.feature_extraction.text import TfidfTransformer

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.metrics import roc_auc_score

from sklearn import metrics

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import GridSearchCV

from sklearn import svm

from sklearn import linear_model







tunedvalues = [{'alpha':[10**-5, 10**-4, 10**-3, 10**-2, 10**-1, 10**0, 10**1, 10**2, 10**3, 10**4, 10**5]}]



#using GridSearchCV

model = GridSearchCV(linear_model.SGDClassifier(loss='hinge',verbose = 10, penalty='l2',random_state = 42,

                                                class_weight = 'balanced'),tunedvalues, scoring = 'roc_auc', cv=3,return_train_score =True)

model.fit(tfidf_sent_train,y_train)

result=model.cv_results_



#caliberated = CalibratedClassifierCV(model, cv=3,method="sigmoid")

#caliberated.fit(tfidf_sent_train,y_train)

#results = result['mean_test_score']

#predict_w2v_train = caliberated.predict_proba(tfidf_sent_train)

#fpr_train,tpr_train,thresholds_train = roc_curve(y_train,predict_w2v_train[:,1])

#scores_train = auc(fpr_train,tpr_train)

#print(scores_train)





#used to plot the graph

test_score = result['mean_test_score']

train_score = result['mean_train_score']

values =[10**-5, 10**-4, 10**-3, 10**-2, 10**-1, 10**0, 10**1, 10**2, 10**3, 10**4, 10**5]

valu = [np.log(i)for i in values]

print(valu)

plt.title("Best alpha founder")

plt.plot(valu,test_score, label = "CV", marker  ='o', color = 'red')

plt.plot(valu,train_score, label = "TRAIN", marker = 'o', color = 'blue')

plt.grid(color='black', linestyle=':', linewidth=1)

plt.legend()

plt.xlabel('alpha_values')

plt.ylabel('AUC_Scores')

plt.show()