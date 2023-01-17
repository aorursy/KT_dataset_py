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
train = pd.read_csv( '../input/clickbait-dataset/clickbait_data.csv')



train.head()
train.shape
#checking for nan values



train = train.fillna('')



train.shape
import re

from string import punctuation



def process_text1(headline):

    

    result = headline.replace('/','').replace('\n','')

    result = re.sub(r'[0-9]+','number', result)   # we are substituting all kinds of no. with word number

    result = re.sub(r'(\w)(\1{2,})', r'\1', result)  # \w matches one word/non word character

    result = re.sub(r'(?x)\b(?=\w*\d)\w+\s*', '', result)

    

    result = ''.join(word for word in result if word not in punctuation)  # removes all characters such as "!"#$%&'()*+, -./:;<=>?@[\]^_`{|}~"

    result = re.sub(r' +', ' ', result).lower().strip()

    return result



#-------------------------------------------------------OR------------------------------------------------------------------------------------



def process_text2(headline):

    

    result2 = re.sub(r'[0-9]+','number', headline)

    result2 = re.sub('[^a-zA-Z]'," ", result)

    result2 = result2.lower()

    

    # we are splitting the messages into word list on the basis of spaces

    result2 = result2.split()

    return result2
# removing the stopwords

from nltk.corpus import stopwords



stop = stopwords.words("english")



def cnt_stopwords(headline):

    

    result1 = headline.split()

    num1 =  len([word for word in result1 if word in stop])

    

    return num1
# Clickbait headlines also tend to contain more informal writing than non-clickbait headlines. As such, they may contain many more contractions, occurrences of slang, etc



contractions = ['tis', 'aint', 'amnt', 'arent', 'cant', 'couldve', 'couldnt', 'couldntve',

                'didnt', 'doesnt', 'dont', 'gonna', 'gotta', 'hadnt', 'hadntve', 'hasnt',

                'havent', 'hed', 'hednt', 'hedve', 'hell', 'hes', 'hesnt', 'howd', 'howll',

                'hows', 'id', 'idnt', 'idntve', 'idve', 'ill', 'im', 'ive', 'ivent', 'isnt',

                'itd', 'itdnt', 'itdntve', 'itdve', 'itll', 'its', 'itsnt', 'mightnt',

                'mightve', 'mustnt', 'mustntve', 'mustve', 'neednt', 'oclock', 'ol', 'oughtnt',

                'shant', 'shed', 'shednt', 'shedntve', 'shedve', 'shell', 'shes', 'shouldve',

                'shouldnt', 'shouldntve', 'somebodydve', 'somebodydntve', 'somebodys',

                'someoned', 'someonednt', 'someonedntve', 'someonedve', 'someonell', 'someones',

                'somethingd', 'somethingdnt', 'somethingdntve', 'somethingdve', 'somethingll',

                'somethings', 'thatll', 'thats', 'thatd', 'thered', 'therednt', 'theredntve',

                'theredve', 'therere', 'theres', 'theyd', 'theydnt', 'theydntve', 'theydve',

                'theydvent', 'theyll', 'theyontve', 'theyre', 'theyve', 'theyvent', 'wasnt',

                'wed', 'wedve', 'wednt', 'wedntve', 'well', 'wontve', 'were', 'weve', 'werent',

                'whatd', 'whatll', 'whatre', 'whats', 'whatve', 'whens', 'whered', 'wheres',

                'whereve', 'whod', 'whodve', 'wholl', 'whore', 'whos', 'whove', 'whyd', 'whyre',

                'whys', 'wont', 'wontve', 'wouldve', 'wouldnt', 'wouldntve', 'yall', 'yalldve',

                'yalldntve', 'yallll', 'yallont', 'yallllve', 'yallre', 'yallllvent', 'yaint',

                'youd', 'youdve', 'youll', 'youre', 'yourent', 'youve', 'youvent']



def cnt_contract(headline):

    

    result2 = headline.split()

    num2 = len([word for word in result2 if word in contractions])

    return num2
# It is often the case that clickbait headlines are stated in the form of a questions, which begins with question words



question_words = ['who', 'whos', 'whose', 'what', 'whats', 'whatre', 'when', 'whenre', 'whens', 'couldnt',

        'where', 'wheres', 'whered', 'why', 'whys', 'can', 'cant', 'could', 'will', 'would', 'is',

        'isnt', 'should', 'shouldnt', 'you', 'your', 'youre', 'youll', 'youd', 'here', 'heres',

        'how', 'hows', 'howd', 'this', 'are', 'arent', 'which', 'does', 'doesnt']





def question_word(headline):

    

    result3 = headline.lower().split()

    

    if result3[0] in question_words:

        return 1

    else:

        return 0
# Lastly, a function is defined to check the part-of-speech (i.e., noun, verb, adjective, etc.) of each word in a headline. It’s possible that non-clickbait headlines contain noun



def pos_tags(headline):

    

    result4 = headline.split()

    

    non_stop = [word for word in result4 if word not in stopwords.words("english")]

    pos = [part[1] for part in nltk.pos_tag(non_stop)]

    pos = " ".join(pos)

    return pos
import nltk



train['processed_headline']     = train['headline'].apply(process_text1)

train['question'] = train['headline'].apply(question_word)



train['num_words']       = train['headline'].apply(lambda x: len(x.split()))

train['part_speech']     = train['headline'].apply(pos_tags)

train['num_contract']    = train['headline'].apply(cnt_contract)

train['num_stop_words']  = train['headline'].apply(cnt_stopwords)

train['stop_word_ratio'] = train['num_stop_words']/train['num_words']

train['contract_ratio']  = train['num_contract']/train['num_words']
train
train = train.drop(columns = ['num_contract','num_stop_words'])



train
from sklearn.model_selection import train_test_split



df_train,df_test = train_test_split(train, test_size=0.25, random_state=0)
from sklearn.feature_extraction.text import TfidfVectorizer



"""

 1) strip_accents: ‘ascii’ is a fast method that only works on characters that have an direct ASCII mapping. ‘unicode’ is a slightly slower method that works on any characters

 2) analyzer: Whether the feature should be made of word or character n-grams.

 3) token_pattern: regexp selects tokens of 2 or more alphanumeric characters. 

 4) min_df: ignore terms that have a document frequency strictly lower than the given threshold.

 5) use_idf: Enable inverse-document-frequency reweighting.

 6) Smooth idf: weights by adding one to document frequencies, as if an extra document was seen containing every term in the collection exactly once.

 7) sublinear_tf: Apply sublinear tf scaling, i.e. replace tf with 1 + log(tf).



"""



tfidf = TfidfVectorizer(min_df=3, max_features=None, strip_accents='unicode',analyzer='word', token_pattern=r'\w{1,}', ngram_range=(1,5),use_idf=1, smooth_idf=1, sublinear_tf=1)



"""



1) fit_transform -> Learn vocabulary and idf, return document-term matrix,This is equivalent to fit followed by transform.



2) transform -> Transform documents to document-term matrix,Uses the vocabulary and document frequencies (df) learned by fit (or fit_transform)



"""



X_train_headline = tfidf.fit_transform(df_train['processed_headline'])

X_test_headline  = tfidf.transform(df_test['processed_headline'])
from sklearn.feature_extraction.text import CountVectorizer

from sklearn.preprocessing import StandardScaler



"""

The idea behind StandardScaler is that it will transform your data such that its distribution will have a mean value 0 and standard deviation of 1.



mean : u

std. deviation : s

data = [x1,x2....]



processes every element as:

x1 = (x1-u)/s

...



"""



cv = CountVectorizer()



X_train_pos = cv.fit_transform(df_train['part_speech'])

X_test_pos = cv.transform(df_test['part_speech'])







sc = StandardScaler(with_mean = False) # taking the mean on sparse matrix will not work, becoz centering them creates dense matrix which is too large to fit in memory.



X_train_pos_sc = sc.fit_transform(X_train_pos)

X_test_pos_sc = sc.transform(X_test_pos)

# removing those rows which are already used up



X_train_val = df_train.drop( columns = ['headline','clickbait','processed_headline','part_speech']).values

X_test_val = df_test.drop( columns = ['headline','clickbait','processed_headline','part_speech']).values



# scaling all the above values

sc = StandardScaler()

X_train_val_sc = sc.fit_transform(X_train_val)

X_test_val_sc  = sc.transform(X_test_val)
Y_train = df_train['clickbait'].values

Y_test = df_test['clickbait'].values
"""

Lastly, we can combine the new tf-idf vectors with the scaled engineered features and store them as sparse arrays.

This helps to save memory as the tf-idf vectors are extremely large, but are composed mostly of zeros.



sparse matrix



When a sparse matrix is represented with a 2-dimensional array, we waste a lot of space to represent that matrix.

For example, consider a matrix of size 100 X 100 containing only 10 non-zero elements.

In this matrix, only 10 spaces are filled with non-zero values and remaining spaces of the matrix are filled with zero.

That means, totally we allocate 100 X 100 X 2 = 20000 bytes of space to store this integer matrix.



1)Triplet Representation (Array Representation)  -  In this representation, we consider only non-zero values along with their row and column index values. In this representation, the 0th row stores the total number of rows, total number of columns and the total number of non-zero values in the sparse matrix.

                                                    For example, consider a matrix of size 3*4 containing 4 number of non-zero values.

                                                    

                                                    0 0 0 9                                  rows   col    non-zero-values

                                     given matrix:  2 0 6 0       triplet representation:     3      4          4

                                                    0 0 0 3                                   0      3          9

                                                                                              1      0          2

                                                                                              1      2          6                

                                                                                              2      3          3

                                                                                              

 e.g - it 1st row shows total row,col, non-0 val then in 2nd row 9 val is present in 0th row and 3rd col in this way...

2)Linked Representation

"""



from scipy import sparse



X_train = sparse.hstack([X_train_val_sc, X_train_headline, X_train_pos_sc]).tocsr()

X_test  = sparse.hstack([X_test_val_sc, X_test_headline, X_test_pos_sc]).tocsr()



# tocsr() - Returns a copy of this matrix in Compressed Sparse Row format.
from sklearn.model_selection import GridSearchCV

from sklearn.linear_model import LogisticRegression



#Regularization is applying a penalty to increasing the magnitude of parameter values in order to reduce overfitting and C is inverse of regularization



param_grid  = [{'C': np.linspace(90,100,20)}]

grid_cv = GridSearchCV(LogisticRegression(), param_grid, scoring='accuracy', cv=5, verbose=1) # No. of K-folds(cv) = 5

grid_cv.fit(X_train, Y_train)



print(grid_cv.best_params_)

print(grid_cv.best_score_)
from sklearn.metrics import confusion_matrix

from sklearn.metrics import classification_report

from sklearn.metrics import accuracy_score



model = LogisticRegression(penalty='l2', C=93.684210526315795)

model = model.fit(X_train, Y_train)

Y_pred = model.predict(X_test)





# 1. accuracy score

accuracy = accuracy_score(Y_test, Y_pred)

print('accuracy :',accuracy*100,"%")





# 2. Classification report

print(classification_report(Y_test, Y_pred))