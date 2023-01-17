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
data=pd.read_csv('/kaggle/input/trip-advisor-hotel-reviews/tripadvisor_hotel_reviews.csv')
data.shape
data.head()
data.Rating.value_counts()
data.isnull().sum()
import string

punc=string.punctuation



from nltk.corpus import stopwords

stop_words = set(stopwords.words('english'))



from nltk.stem.porter import PorterStemmer

stemmer = PorterStemmer()



from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()
def pre_processing(row):

    #converting to lowercase

    _row=row.lower()

    #Removing Punctuation

    _row="".join([x for x in _row if x not in punc])

    #Removing stopwords

    _row=" ".join([word for word in str(_row).split() if word not in stop_words])

    #Stemming

    _row = " ".join([stemmer.stem(word) for word in _row.split()])

    #Lemmatization

    _row = " ".join([lemmatizer.lemmatize(word) for word in _row.split()])

    #Split

    _row = _row.split()

    return _row

    
data['text'] = data['Review'].apply(pre_processing)
from sklearn.model_selection import train_test_split



X=data['text']



y=data['Rating']



from gensim.models import Word2Vec

import time

# Skip-gram model (sg = 1)

size = 100

window = 3

min_count = 1

workers = 3

sg = 1



OUTPUT_FOLDER=""



word2vec_model_file = OUTPUT_FOLDER + 'word2vec_' + str(size) + '.model'

start_time = time.time()

stemmed_tokens = pd.Series(data['text']).values

# Train the Word2Vec Model

w2v_model = Word2Vec(stemmed_tokens, min_count = min_count, size = size, workers = workers, window = window, sg = sg)

print("Time taken to train word2vec model: " + str(time.time() - start_time))

w2v_model.save(word2vec_model_file)
import numpy as np



# Load the model from the model file

sg_w2v_model = Word2Vec.load(word2vec_model_file)



# Unique ID of the word

print("Index of the word 'action':")

print(sg_w2v_model.wv.vocab["action"].index)



# Total number of the words 

print(len(sg_w2v_model.wv.vocab))



"""

import numpy as np



# Load the model from the model file

sg_w2v_model = Word2Vec.load(word2vec_model_file)



# Unique ID of the word

print("Index of the word 'action':")

print(sg_w2v_model.wv.vocab["action"].index)



# Total number of the words 

print(len(sg_w2v_model.wv.vocab))



# Print the size of the word2vec vector for one word

print("Length of the vector generated for a word")

print(len(sg_w2v_model['action']))



# Get the mean for the vectors for an example review

print("Print the length after taking average of all word vectors in a sentence:")

print(np.mean([sg_w2v_model[token] for token in top_data_df_small['stemmed_tokens'][0]], axis=0)

"""
# Store the vectors for train data in following file

word2vec_filename = OUTPUT_FOLDER + 'train_review_word2vec.csv'

with open(word2vec_filename, 'w+') as word2vec_file:

    for index, row in enumerate(X.tolist()):

        model_vector = (np.mean([sg_w2v_model[token] for token in row], axis=0)).tolist()

        if index == 0:

            header = ",".join(str(ele) for ele in range(100))

            word2vec_file.write(header)

            word2vec_file.write("\n")

        # Check if the line exists else it is vector of zeros

        if type(model_vector) is list:  

            line1 = ",".join( [str(vector_element) for vector_element in model_vector] )

        else:

            line1 = ",".join([str(0) for i in range(100)])

        word2vec_file.write(line1)

        word2vec_file.write('\n')

X_vectors=pd.read_csv('train_review_word2vec.csv')
#SMOTE Technique

from imblearn.over_sampling import SMOTE

# transform the dataset

oversample = SMOTE()

X, y = oversample.fit_resample(X_vectors, y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1)
from xgboost import XGBClassifier



model = XGBClassifier(max_depth=10,random_state=1,learning_rate=0.05,seed=1)

model.fit(X_train, y_train)
y_pred=model.predict(X_test)
from sklearn.metrics import accuracy_score



accuracy_score(y_pred,y_test)
from sklearn.metrics import classification_report,accuracy_score



print(classification_report(y_pred,y_test))
accuracy_score(y_pred,y_test)
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score



print("Accuracy",accuracy_score(y_pred,y_test))

#print("Precission",precision_score(y_pred,y_test))

#print("Recall",recall_score(y_pred,y_test))

#print("F1 Score",f1_score(y_pred,y_test))
from sklearn.metrics import confusion_matrix

confusion_matrix(y_pred,y_test)
from sklearn.metrics import classification_report



print(classification_report(y_pred,y_test))
