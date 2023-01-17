# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
#make compatible with Python 2 and Python 3
from __future__ import print_function, division, absolute_import 

# Remove warnings
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
%matplotlib inline
#import packages

import bs4 as bs
import nltk
nltk.download('stopwords')

import re
from nltk.tokenize import sent_tokenize # tokenizes sentences
from nltk.stem import PorterStemmer
from nltk.tag import pos_tag
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from nltk.util import ngrams

eng_stopwords = stopwords.words('english')
merge = pd.read_csv("/kaggle/input/live-score/242project_merge_finalversion_5_13.csv")
merge
df_raw = pd.read_csv("/kaggle/input/live-score/Compiled_Yelp_Scraped_Reviews_Komplete_5_13.csv")

# Clean the dataframe by getting rid of non-essential columns 
df = df_raw.drop(columns = ['0','alias', 'address2', 'formatted_address'], axis = 1)

# Print data shape
print("Review data shape is {}".format(df.shape))

#Preview data
df.head()
live_raw = pd.read_csv("/kaggle/input/live-score/Restaurant_Scores_-_LIVES_Standard.csv")
# Use the necessary columns:
necessary_list = ['business_id','business_name','business_address','inspection_id','inspection_date','inspection_score','inspection_type','violation_id','violation_description','risk_category']
live = live_raw.loc[:,necessary_list]
#Drop the missing value for inspection_score:
live = live.dropna()
live.head()
def review_cleaner(reviews, lemmatize=True, stem=True):
    '''
    Clean and preprocess a review.

    1. Remove HTML tags
    2. Use regex to remove all special characters (only keep letters)
    3. Make strings to lower case and tokenize / word split reviews
    4. Remove English stopwords
    5. Rejoin to one string
    '''
    ps = PorterStemmer()
    wnl = WordNetLemmatizer()
    cleaned_reviews=[]
    
    for i, review in enumerate(df['text']):
        # print progress
        if((i+1)%500 == 0):
            print("Done with %d reviews" %(i+1))
            
        #1. Remove HTML tags
        review = bs.BeautifulSoup(review).text

        #2. Use regex to find emojis
        emojis = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)', review)

        #3. Remove punctuation
        review = re.sub("[^a-zA-Z]", " ",review)

        #4. Tokenize into words (all lower case)
        review = review.lower().split()

        #5. Remove stopwords
        eng_stopwords = set(stopwords.words("english"))
            
        clean_review=[]
        for word in review:
            if word not in eng_stopwords:
                if lemmatize is True:
                    word=wnl.lemmatize(word)
                elif stem is True:
                    if word == 'oed':
                        continue
                    word=ps.stem(word)
                clean_review.append(word)

        #6. Join the review to one sentence
        
        review_processed = ' '.join(clean_review+emojis)
        cleaned_reviews.append(review_processed)
    

    return(cleaned_reviews)
original_clean_reviews=review_cleaner(df['text'], lemmatize = True, stem = True)
# for j in range(len(original_clean_reviews)):
#     print(j)
#     #original_clean_reviews[j]
## Add a clean review column
df['clean_review'] = original_clean_reviews
df['clean_review'][0]
df
df.to_csv("After_Clean_Reviews_5_13.csv")
merge.head()
grouped_df = df.groupby(['id'])['clean_review'].apply(lambda x: ' '.join(x)).reset_index()

unique_df = df[['id','name','avg_rating','total_review_count']].drop_duplicates()

new_yelp = pd.merge(grouped_df,unique_df,how='left',on='id')
new_yelp.head(40)
new_yelp = new_yelp.drop(list(range(0,37)))
new_yelp = new_yelp.reset_index()
new_yelp
new_yelp = new_yelp.drop(columns=['index'])
new_merge = merge[['inspection_score','inspection_result','id_yelp']]
new_review = pd.merge(new_yelp,new_merge,how='inner',left_on='id',right_on='id_yelp')
new_review
new_review = new_review.drop(columns=['id_yelp'])
new_review.to_csv('Merged_Cleaned_Reviews_5_13_2155_businesses.csv')
reviews = pd.read_csv("/kaggle/input/live-score/Merged_Cleaned_Reviews_5_13_2155_businesses.csv")
individual_reviews = pd.read_csv("/kaggle/input/live-score/After_Clean_Reviews_5_13.csv")
# # Before Joining
# df.head()
reviews['clean_review_sep'] = reviews['clean_review'].apply(lambda x: x.split())
reviews.head()
threshold = 86
reviews['inspection_result'] = reviews['inspection_score'].apply(lambda x: 1 if x>=threshold else 0)
print("There is {} 0".format(len(reviews[reviews['inspection_result']==0])))
print("There is {} 1".format(len(reviews[reviews['inspection_result']==1])))
label_0 = len(reviews[reviews['inspection_result']==0])
label_1 = len(reviews[reviews['inspection_result']==1])
base_accuracy = label_1/(label_0+label_1)
print("Baseline Accuarcy is {}".format(base_accuracy))
from collections import defaultdict
word_freq = defaultdict(int)
for sent in reviews['clean_review_sep']:
    for i in sent:
        word_freq[i] += 1
len(word_freq)
sorted(word_freq, key=word_freq.get, reverse=True)[:120]
#pip install gensim
sentences = reviews['clean_review_sep']
# Set values for various parameters
num_features = 100    # Word vector dimensionality                      
min_word_count = 40   # ignore all words with total frequency lower than this                       
num_workers = 4       # Number of threads to run in parallel
context = 10          # Context window size                                                                                    


# Initialize and train the model (this will take some time)
from gensim.models import word2vec

print("Training word2vec model... ")
model = word2vec.Word2Vec(sentences,
                          workers=num_workers,
                          size=num_features,
                          min_count=min_word_count,
                          window=context)
print("Training finished!")

# save the model for later use. You can load it later using Word2Vec.load()
model_name = "100features_40minwords_10context"
model.save(model_name)
# Get vocabulary count of the model
vocab_tmp = list(model.wv.vocab)
print('Vocab length:',len(vocab_tmp))
# Get Vocabulary words
vocab_tmp
# Get the Word embedding 
model['dirty']
# Get cosine similarity of words
from sklearn.metrics.pairwise import cosine_similarity

model.similarity('dish', 'plate')
model.similarity('gross', 'dirty')
# model.similarity('rat', 'infested') # etc 
model.most_similar(positive=['good', 'great'], negative=['bad'])
model.most_similar("health")
model.most_similar("awful")
from gensim.models import Word2Vec
# Load the trained modelNumeric Representations of Words
model = Word2Vec.load("100features_40minwords_10context")
model.wv.syn0
model.wv.syn0.shape
model.corpus_count
# Get vocabulary count of the model
vocab_tmp = list(model.wv.vocab)
print('Vocab length:',len(vocab_tmp))
# Get distributional representation of each word
X = model[vocab_tmp]
X.shape
from sklearn import decomposition
# get two principle components of the feature space
pca = decomposition.PCA(n_components=2).fit_transform(X)
pca
pca.shape
# set figure settings
plt.figure(figsize=(20,20))

# save pca values and vocab in dataframe df
df = pd.concat([pd.DataFrame(pca),pd.Series(vocab_tmp)],axis=1)
df.columns = ['x', 'y', 'word']

plt.xlabel("1st principal component", fontsize=14)
plt.ylabel('2nd principal component', fontsize=14)

plt.scatter(x=df['x'], y=df['y'],s=3)
for i, word in enumerate(df['word'][0:100]):
    plt.annotate(word, (df['x'].iloc[i], df['y'].iloc[i]))
plt.title("PCA Embedding", fontsize=18)
plt.show()
## A popular non-linear dimensionality reduction technique that preserves greatly the local 
## and global structure of the data. Essentially tries to reconstruct the subspace in which the 
## data exists

from sklearn import manifold
tsne = manifold.TSNE(n_components=2)
X_tsne = tsne.fit_transform(X)
# set figure settings
plt.figure(figsize=(10,10), dpi=100)

# save pca values and vocab in dataframe df
df2 = pd.concat([pd.DataFrame(X_tsne),pd.Series(vocab_tmp)], axis=1)
df2.columns = ['x', 'y', 'word']

plt.xlabel("1st principal component", fontsize=14)
plt.ylabel('2nd principal component', fontsize=14)

plt.scatter(df2['x'], y=df2['y'], s=3)
for i, word in enumerate(df2['word'][0:100]):
    plt.annotate(word, (df2['x'].iloc[i], df2['y'].iloc[i]))
plt.title("tSNE Embedding", fontsize=18)
plt.show()
def makeFeatureVec(review, model):
    # Function to average all of the word vectors in a given paragraph
    featureVec =[]
    
    # Index2word is a list that contains the names of the words in 
    # the model's vocabulary. Convert it to a set, for speed 
    index2word_set = set(model.wv.index2word)
    
    # Loop over each word in the review and, if it is in the model's
    # vocaublary, add its feature vector to the total
    for n,word in enumerate(review):
        if word in index2word_set: 
            featureVec.append(model[word])
            
    # Average the word vectors for a 
    featureVec = np.mean(featureVec,axis=0)
    return featureVec


def getAvgFeatureVecs(reviews, model):
    # Given a set of reviews (each one a list of words), calculate 
    # the average feature vector for each one 
    
    reviewFeatureVecs = []
    # Loop through the reviews
    for counter,review in enumerate(reviews):
        
        # Print a status message every 5000th review
        if counter%5000. == 0.:
            print("Review %d of %d" % (counter, len(reviews)))

        # Call the function (defined above) that makes average feature vectors
        vector= makeFeatureVec(review, model)
        reviewFeatureVecs.append(vector)
            
    return reviewFeatureVecs
from sklearn.ensemble import RandomForestClassifier
# # CountVectorizer can actucally handle a lot of the preprocessing for us
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import metrics # for confusion matrix, accuracy score etc
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix


np.random.seed(0)


def train_sentiment(cleaned_reviews, y,max_features=1000):
    '''This function will:
    y=reviews["label"]
    
    1. Convert reviews into feature vectors using word2vec.
    2. split data into train and test set.
    3. train a random forest model using train n-gram counts and y (labels)
    4. test the model on your test split
    5. print accuracy of sentiment prediction on test and training data
    6. print confusion matrix on test data results

    To change n-gram type, set value of ngram argument
    To change the number of features you want the countvectorizer to generate, set the value of max_features argument'''

    print("1.Creating Feature vectors using word2vec...\n")

    trainDataVecs = getAvgFeatureVecs(cleaned_reviews, model)
    
   
    print("\n2.Splitting dataset into train and test sets...\n")
    X_train, X_test, y_train, y_test = train_test_split(\
    trainDataVecs, y, random_state=0, test_size=.2)

   
    print("3. Training the random forest classifier...\n")
    
    # Initialize a Random Forest classifier with 75 trees
    forest = RandomForestClassifier(n_estimators = 100) 
    
    # Fit the forest to the training set, word2vecfeatures 
    # and the sentiment labels as the target variable
    forest = forest.fit(X_train, y_train)


    train_predictions = forest.predict(X_train)
    test_predictions = forest.predict(X_test)
    
    train_acc = metrics.accuracy_score(y_train, train_predictions)
    valid_acc = metrics.accuracy_score(y_test, test_predictions)
    print("=================Training Statistics======================\n")
    print("The training accuracy is: ", train_acc)
    print("The validation accuracy is: ", valid_acc)
    print()
    print('CONFUSION MATRIX:')
    print('         Predicted')
    print('          neg pos')
    print(' Actual')
    c=confusion_matrix(y_test, test_predictions)
    print('  neg  ',c[0])
    print('  pos  ',c[1])
train_sentiment(cleaned_reviews = sentences, y=reviews["inspection_result"],max_features=1000)
AvgFeatureVecs = getAvgFeatureVecs(reviews=sentences, model=model)
AvgFeatureVecs_np = np.array(AvgFeatureVecs)
AvgFeatureVecs_np.shape
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler

def rfr_model(X, y):
# Perform Grid-Search
    gsc = GridSearchCV(
        estimator=RandomForestRegressor(),
        param_grid={
            'max_depth': range(3,7),
            'n_estimators': (50, 100, 200),
        },
        cv=5, scoring='neg_mean_squared_error', verbose=5, n_jobs=-1)
    
    grid_result = gsc.fit(X, y)
    best_params = grid_result.best_params_
    
    rfr = RandomForestRegressor(max_depth=best_params["max_depth"], n_estimators=best_params["n_estimators"],                               random_state=False, verbose=False)
# Perform K-Fold CV
    scores = cross_val_score(rfr, X, y, cv=5, scoring='neg_root_mean_squared_error')

    return scores,best_params
scores, best = rfr_model(X=AvgFeatureVecs_np,y=reviews["inspection_score"])
scores
import seaborn as sns
sns.swarmplot(reviews["inspection_score"])
best
from sklearn.ensemble import RandomForestClassifier
# CountVectorizer can actucally handle a lot of the preprocessing for us
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn import metrics # for confusion matrix, accuracy score etc
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_curve

from sklearn.model_selection import KFold, cross_val_score

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
# CountVectorizer can actucally handle a lot of the preprocessing for us
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import metrics # for confusion matrix, accuracy score etc
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
count_vect = CountVectorizer(analyzer="word")
X_count = count_vect.fit_transform(reviews['clean_review'])
X_features = pd.DataFrame(X_count.toarray())
X_features.columns = count_vect.get_feature_names()
print(X_features.shape)
X_features.head()
X_train, X_test, y_train, y_test = train_test_split(X_features, reviews['inspection_result'], test_size=0.2)

print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)
logreg = LogisticRegression()                           # instantiate
logreg.fit(X_train, y_train)                            # fit
Y_pred = logreg.predict(X_test)                         # predict
acc_logreg = logreg.score(X_test, y_test)                # evaluate

print('Logistic Regression labeling accuracy:', str(round(acc_logreg*100,2)),'%')
knn = KNeighborsClassifier(n_neighbors = 3)                  # instantiate
knn.fit(X_train, y_train)                                    # fit
acc_knn = knn.score(X_test, y_test)                          # predict + evaluate

print('K-Nearest Neighbors labeling accuracy:', str(round(acc_knn*100,2)),'%')
rf = RandomForestClassifier(n_estimators=50, max_depth=10, n_jobs=-1) # instantiate
rf_model = rf.fit(X_train, y_train) # fit
accuracy_rf = rf_model.score(X_test, y_test) # predict + evaluate

print('Random Forest Regression labeling accuracy:', str(round(accuracy_rf*100,2)),'%')
sorted(zip(rf_model.feature_importances_, X_train.columns), reverse=True)[0:10]
# Look at importnace of features for random forest

def plot_model_var_imp(model , X , y):
    imp = pd.DataFrame( 
        model.feature_importances_  , 
        columns = [ 'Importance' ] , 
        index = X.columns 
    )
    imp = imp.sort_values( [ 'Importance' ] , ascending = True )
    imp[ : 10 ].plot( kind = 'barh' )
    print ('Training accuracy Random Forest:',model.score( X , y ))

plot_model_var_imp(rf, X_train, y_train)
# XGBoost, same API as scikit-learn
gradboost = xgb.XGBClassifier(n_estimators=1000)             # instantiate
gradboost.fit(X_train, y_train)                              # fit
accuracy_xgboost = gradboost.score(X_test, y_test)           # predict + evalute

print('XGBoost labeling accuracy:', str(round(acc_xgboost*100,2)),'%')
## ALEX'S CHUNK 

X_train, X_test, y_train, y_test = train_test_split(X_features, reviews['label'], test_size=0.2)

rf = RandomForestClassifier(n_estimators=50, max_depth=10, n_jobs=-1)
rf_model = rf.fit(X_train, y_train)

sorted(zip(rf_model.feature_importances_, X_train.columns), reverse=True)[0:10]
from sklearn.metrics import precision_recall_fscore_support as score
y_pred = rf_model.predict(X_test)
precision, recall, fscore, support = score(y_test, y_pred, pos_label=1, average='binary')

print('Precision: {} / Recall: {} / Accuracy: {}'.format(round(precision, 3),
                                                        round(recall, 3),
                                                        round((y_pred==y_test).sum() / len(y_pred),3)))
y_train1=y_train
X_test1=X_test
y_test1 =y_test
X_train1 = X_train

#Import svm model
from sklearn import svm

#Create a svm Classifier
clf = svm.SVC(kernel='rbf', gamma = 'scale') # rbf Kernel

#Train the model using the training sets
clf.fit(X_train1, y_train1)

#Predict the response for test dataset
y_pred = clf.predict(X_test1)

#Import scikit-learn metrics module for accuracy calculation
from sklearn import metrics

# Model Accuracy: how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test1, y_pred))

from sklearn.metrics import confusion_matrix
confusion_matrix(y_test1, y_pred)
from sklearn.ensemble import RandomForestClassifier
# CountVectorizer can actucally handle a lot of the preprocessing for us
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import metrics # for confusion matrix, accuracy score etc
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix


np.random.seed(100)


def train_predict_sentiment(cleaned_reviews, y=df["..."], ngram=1, max_features=1000):
    '''This function will:
    1. split data into train and test set.
    2. get n-gram counts from cleaned reviews 
    3. train a random forest model using train n-gram counts and y (labels)
    4. test the model on your test split
    5. print accuracy of sentiment prediction on test and training data
    6. print confusion matrix on test data results

    To change n-gram type, set value of ngram argument
    To change the number of features you want the countvectorizer to generate, set the value of max_features argument'''

    print("Creating the bag of words model!\n")
    # CountVectorizer is scikit-learn's bag of words tool, here we show more keywords 
    vectorizer = CountVectorizer(ngram_range=(1, ngram), analyzer = "word",
                                 tokenizer = None,
                                 preprocessor = None,
                                 stop_words = None,
                                 max_features = max_features) 
    
    X_train, X_test, y_train, y_test = train_test_split(cleaned_reviews, y, random_state=0, test_size=.2)

    # Then we use fit_transform() to fit the model / learn the vocabulary,
    # then transform the data into feature vectors.
    # The input should be a list of strings. .toarraty() converts to a numpy array
    
    train_bag = vectorizer.fit_transform(X_train).toarray()
    test_bag = vectorizer.transform(X_test).toarray()
    # print('TOP 20 FEATURES ARE: ',(vectorizer.get_feature_names()[:20]))

    print("Training the random forest classifier!\n")
    # Initialize a Random Forest classifier with 75 trees
    forest = RandomForestClassifier(n_estimators = 50) 

    # Fit the forest to the training set, using the bag of words as 
    # features and the sentiment labels as the target variable
    forest = forest.fit(train_bag, y_train)


    train_predictions = forest.predict(train_bag)
    test_predictions = forest.predict(test_bag)
    
    train_acc = metrics.accuracy_score(y_train, train_predictions)
    valid_acc = metrics.accuracy_score(y_test, test_predictions)
    print(" The training accuracy is: ", train_acc, "\n", "The validation accuracy is: ", valid_acc)
    print()
    print('CONFUSION MATRIX:')
    print('         Predicted')
    print('          neg pos')
    print(' Actual')
    c=confusion_matrix(y_test, test_predictions)
    print('  neg  ',c[0])
    print('  pos  ',c[1])

    #Extract feature importnace
    print('\nTOP TEN IMPORTANT FEATURES:')
    importances = forest.feature_importances_
    indices = np.argsort(importances)[::-1]
    top_10 = indices[:20]
    print([vectorizer.get_feature_names()[ind] for ind in top_10])
# Clean the reviews in the training set 'train' using review_cleaner function defined above
# Here we use the original reviews without lemmatizing and stemming
original_clean_reviews = review_cleaner(df['text'],lemmatize=False,stem=False)
train_predict_sentiment(cleaned_reviews = original_clean_reviews, y = df["..."], ngram=1, max_features=1000)
