import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
df_train = pd.read_csv("../input/q_quora.csv")
print  (df_train.iloc[0,4])

print (df_train.head())
print(df_train.shape)
import numpy as np # linear algebra



import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import matplotlib.pyplot as plt



import seaborn as sns



pal = sns.color_palette()



%matplotlib inline



pd.options.mode.chained_assignment = None  # default='warn'



is_dup = df_train['is_duplicate'].value_counts()



plt.figure(figsize=(8,4))



sns.barplot(is_dup.index, is_dup.values, alpha=0.8, color=pal[1])



plt.ylabel('Number of Occurrences', fontsize=12)



plt.xlabel('Is Duplicate', fontsize=12)



plt.show()
train_qs = pd.Series(df_train['question1'].tolist() + df_train['question2'].tolist()).astype(str)

from wordcloud import WordCloud

cloud = WordCloud(width=1440, height=1080).generate(" ".join(train_qs.astype(str)))

plt.figure(figsize=(20, 15))

plt.imshow(cloud)

plt.axis('off')
from nltk.corpus import stopwords

import re







def question_to_words( raw_question ):



    # 2. Remove non-letters        

    letters_only = re.sub("^a-zA-Z",' ',raw_question)

    

    #

    # 3. Convert to lower case, split into individual words

    words = letters_only.lower().split()                             

    #

    # 4. In Python, searching a set is much faster than searching

    #   a list, so convert the stop words to a set

    stops = set(stopwords.words("english"))                  

    # 

    # 5. Remove stop words

    meaningful_words = [w for w in words if not w in stops]   

    #

    # 6. Join the words back into one string separated by space, 

    # and return the result.

    return( " ".join( meaningful_words ))  
print (df_train.iloc[0,4])

q1=df_train.iloc[0,4]

print ("type of q1 is ", type(q1))

print (question_to_words(q1))
cleanq1s = []

cleanq2s=[]

numqs=df_train.shape[0]

print (numqs)

for i in range( 0, numqs):

    # Call our function for each one, and add the result to the list of

    # clean reviews

    cleanq1s.append(question_to_words(list(df_train['question1'])[i] ) )

    cleanq2s.append(question_to_words(list(df_train['question2'])[i] ) )
print ("Creating the bag of words...\n")

from sklearn.feature_extraction.text import CountVectorizer



# Initialize the "CountVectorizer" object, which is scikit-learn's

# bag of words tool.  

vectorizer = CountVectorizer(analyzer = "word",   \

                             tokenizer = None,    \

                             preprocessor = None, \

                             stop_words = None,   \

                             max_features = 5000) 



# fit_transform() does two functions: First, it fits the model

# and learns the vocabulary; second, it transforms our training data

# into feature vectors. The input to fit_transform should be a list of 

# strings.

train_data_features1 = vectorizer.fit_transform(cleanq1s)

train_data_features2 = vectorizer.fit_transform(cleanq2s)



# Numpy arrays are easy to work with, so convert the result to an 

# array

train_data_features1 = train_data_features1.toarray()

train_data_features2 = train_data_features2.toarray()

print (train_data_features1.shape)
vocab = vectorizer.get_feature_names()

print (vocab[0:100])
print ("Training the random forest...")

from sklearn.ensemble import RandomForestClassifier



# Initialize a Random Forest classifier with 100 trees

forest = RandomForestClassifier(n_estimators = 100) 



# Fit the forest to the training set, using the bag of words as 

# features and the sentiment labels as the response variable

#

# This may take a few minutes to run

train_data_features1

#forest = forest.fit( train_data_features1-train_data_features2, df_train["is_duplicate"] )
#preds=forest.predict(train_data_features1-train_data_features2)
from sklearn.metrics import roc_auc_score
from sklearn.cross_validation import train_test_split





y = df_train['is_duplicate']

X = train_data_features1-train_data_features2



X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3)

print (X_train[0:4])



forest = forest.fit( X_train, y_train )

preds_train=forest.predict(X_train)

preds_test=forest.predict(X_test)

from sklearn.metrics import roc_auc_score

print('Train AUC:', roc_auc_score(y_train, preds_train))

print('Test AUC:', roc_auc_score(y_test, preds_test))

#print (train_data_features1[0:4])
from sklearn.metrics import log_loss

print('log loss train:', log_loss(y_train, preds_train))

print('log loss test:', log_loss(y_test, preds_test))
from sklearn.feature_extraction.text import TfidfVectorizer

tvec = TfidfVectorizer(min_df=.0025, max_df=.1, stop_words="english", ngram_range=(1,2))
# Initialize the "CountVectorizer" object, which is scikit-learn's

# bag of words tool.  

vectorizertd = TfidfVectorizer( stop_words="english",max_features=1000)



# fit_transform() does two functions: First, it fits the model

# and learns the vocabulary; second, it transforms our training data

# into feature vectors. The input to fit_transform should be a list of 

# strings.

train_data_features1td = vectorizertd.fit_transform(cleanq1s)

train_data_features2td = vectorizertd.fit_transform(cleanq2s)



# Numpy arrays are easy to work with, so convert the result to an 

# array

train_data_features1td = train_data_features1td.toarray()

train_data_features2td = train_data_features2td.toarray()

print (train_data_features1td.shape)

print (train_data_features2td.shape)
print ("Training the random forest...")

from sklearn.ensemble import RandomForestClassifier



# Initialize a Random Forest classifier with 100 trees

forest = RandomForestClassifier(n_estimators = 100) 



# Fit the forest to the training set, using the bag of words as 

# features and the sentiment labels as the response variable

#

# This may take a few minutes to run

np.nonzero(train_data_features1td)

train_data_features1td[4041,895]
#from sklearn.cross_validation import train_test_split



y = df_train['is_duplicate']

Xtd1 = train_data_features1td-train_data_features2td





X_traintd,X_testtd,y_traintd,y_testtd = train_test_split(Xtd1,y,test_size=0.3)

print (X_train[0:4])



forest = forest.fit( X_traintd, y_traintd )

preds_traintd=forest.predict(X_traintd)

preds_testtd=forest.predict(X_testtd)

#from sklearn.metrics import roc_auc_score

print('Train AUC:', roc_auc_score(y_traintd, preds_traintd))

print('Test AUC:', roc_auc_score(y_testtd, preds_testtd))

#print (train_data_features1[0:4])

print('log loss train with td_idf:', log_loss(y_traintd, preds_traintd))

print('log loss test with td_idf', log_loss(y_testtd, preds_testtd))
import logging

from sklearn.metrics.pairwise import cosine_similarity as cossim

cos_sim_feat=cossim(train_data_features1td,train_data_features2td)

X_traincos,X_testcos,y_traincos,y_testcos = train_test_split(cos_sim_feat,y,test_size=0.3)

#print (X_train[0:4])



forest = forest.fit( X_traincos, y_traincos )

preds_traincos=forest.predict(X_traincos)

preds_testcos=forest.predict(X_testcos)

#from sklearn.metrics import roc_auc_score

print('Train AUC:', roc_auc_score(y_traincos, preds_traincos))

print('Test AUC:', roc_auc_score(y_testcos, preds_testcos))

#print (train_data_features1[0:4])

print('log loss train with cossine:', log_loss(y_traincos, preds_traincos))

print('log loss test with cossine', log_loss(y_testcos, preds_testcos))
from gensim.models import word2vec