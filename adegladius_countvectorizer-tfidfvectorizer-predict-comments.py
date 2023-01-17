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
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
import re

import pickle 
#import mglearn
import time


from nltk.tokenize import TweetTokenizer # doesn't split at apostrophes
import nltk
from nltk import Text
from nltk.tokenize import regexp_tokenize
from nltk.tokenize import word_tokenize  
from nltk.tokenize import sent_tokenize 
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer


from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression 
from sklearn.naive_bayes import MultinomialNB
from sklearn.multiclass import OneVsRestClassifier


from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline

txt = ["He is ::having a great Time, at the park time?",
       "She, unlike most women, is a big player on the park's grass.",
       "she can't be going"]
# Initialize a CountVectorizer object: count_vectorizer
count_vec = CountVectorizer(stop_words="english", analyzer='word', 
                            ngram_range=(1, 1), max_df=1.0, min_df=1, max_features=None)

# Transforms the data into a bag of words
count_train = count_vec.fit(txt)
bag_of_words = count_vec.transform(txt)

# Print the first 10 features of the count_vec
print("Every feature:\n{}".format(count_vec.get_feature_names()))
print("\nEvery 3rd feature:\n{}".format(count_vec.get_feature_names()[::3]))
print("Vocabulary size: {}".format(len(count_train.vocabulary_)))
print("Vocabulary content:\n {}".format(count_train.vocabulary_))
count_vec = CountVectorizer(stop_words="english", analyzer='word', 
                            ngram_range=(1, 2), max_df=1.0, min_df=1, max_features=None)

count_train = count_vec.fit(txt)
bag_of_words = count_vec.transform(txt)

print(count_vec.get_feature_names())
count_vec = CountVectorizer(stop_words="english", analyzer='word', 
                            ngram_range=(1, 3), max_df=1.0, min_df=1, max_features=None)

count_train = count_vec.fit(txt)
bag_of_words = count_vec.transform(txt)

print(count_vec.get_feature_names())
count_vec = CountVectorizer(stop_words="english", analyzer='word', 
                            ngram_range=(1, 1), max_df=1.0, min_df=0.6, max_features=None)

count_train = count_vec.fit(txt)
bag_of_words = count_vec.transform(txt)

print(count_vec.get_feature_names())
print("\nOnly 'park' becomes the vocabulary of the document term matrix (dtm) because it appears in 2 out of 3 documents, \
meaning 0.66% of the time.\
      \nThe rest of the words such as 'big' appear only in 1 out of 3 documents, meaning 0.33%. which is why they don't appear")
count_vec = CountVectorizer(stop_words="english", analyzer='word', 
                            ngram_range=(1, 1), max_df=0.50, min_df=1, max_features=None)

count_train = count_vec.fit(txt)
bag_of_words = count_vec.transform(txt)

print(count_vec.get_feature_names())
print("\nOnly 'park' is ignored because it appears in 2 out of 3 documents, meaning 0.66% of the time.")
count_vec = CountVectorizer(stop_words="english", analyzer='word', 
                            ngram_range=(1, 1), max_df=1.0, min_df=1, max_features=4)

count_train = count_vec.fit(txt)
bag_of_words = count_vec.transform(txt)

print(count_vec.get_feature_names())
txt1 = ['His smile was not perfect', 'His smile was not not not not perfect', 'she not sang']
tf = TfidfVectorizer(smooth_idf=False, sublinear_tf=False, norm=None, analyzer='word')
txt_fitted = tf.fit(txt1)
txt_transformed = txt_fitted.transform(txt1)
print ("The text: ", txt1)
tf.vocabulary_
idf = tf.idf_
print(dict(zip(txt_fitted.get_feature_names(), idf)))
print("\nWe see that the tokens 'sang','she' have the most idf weight because \
they are the only tokens that appear in one document only.")
print("\nThe token 'not' appears 6 times but it is also in all documents, so its idf is the lowest")
rr = dict(zip(txt_fitted.get_feature_names(), idf))
token_weight = pd.DataFrame.from_dict(rr, orient='index').reset_index()
token_weight.columns=('token','weight')
token_weight = token_weight.sort_values(by='weight', ascending=False)
token_weight 

sns.barplot(x='token', y='weight', data=token_weight)            
plt.title("Inverse Document Frequency(idf) per token")
fig=plt.gcf()
fig.set_size_inches(10,5)
plt.show()
# get feature names
feature_names = np.array(tf.get_feature_names())
sorted_by_idf = np.argsort(tf.idf_)
print("Features with lowest idf:\n{}".format(
       feature_names[sorted_by_idf[:3]]))
print("\nFeatures with highest idf:\n{}".format(
       feature_names[sorted_by_idf[-3:]]))
print("The token 'not' has  the largest weight in document #2 because it appears 3 times there. But in document #1\
 its weight is 0 because it does not appear there.")
txt_transformed.toarray()
new1 = tf.transform(txt1)

# find maximum value for each of the features over all of dataset:
max_val = new1.max(axis=0).toarray().ravel()

#sort weights from smallest to biggest and extract their indices 
sort_by_tfidf = max_val.argsort()

print("Features with lowest tfidf:\n{}".format(
      feature_names[sort_by_tfidf[:3]]))

print("\nFeatures with highest tfidf: \n{}".format(
      feature_names[sort_by_tfidf[-3:]]))
train = pd.read_csv('/kaggle/input/encoded-train/encoded_train.csv')
holdout = pd.read_csv('/kaggle/input/health-data/Test_health.csv')
sub = pd.read_csv('/kaggle/input/health-data/ss_health.csv')
train = pd.read_csv('/kaggle/input/encoded-train/encoded_train.csv')
holdout = pd.read_csv('/kaggle/input/health-data/Test_health.csv').fillna(' ')
"""Lemmatizing and stemming gives us a lower ROC-AUC score. So we will only clean \\n's, Username, IP and http links"""

start_time=time.time()
# remove '\\n'
train['text'] = train['text'].map(lambda x: re.sub('\\n',' ',str(x)))
    
# remove any text starting with User... 
train['text'] = train['text'].map(lambda x: re.sub("\[\[User.*",'',str(x)))
    
# remove IP addresses or user IDs
train['text'] = train['text'].map(lambda x: re.sub("\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}",'',str(x)))
    
#remove http links in the text
train['text'] = train['text'].map(lambda x: re.sub("(http://.*?\s)|(http://.*)",'',str(x)))

end_time=time.time()
print("total time",end_time-start_time)
# remove '\\n'
holdout['text'] = holdout['text'].map(lambda x: re.sub('\\n',' ',str(x)))
    
# remove any text starting with User... 
holdout['text'] = holdout['text'].map(lambda x: re.sub("\[\[User.*",'',str(x)))
    
# remove IP addresses or user IDs
holdout['text'] = holdout['text'].map(lambda x: re.sub("\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}",'',str(x)))
    
#remove http links in the text
holdout['text'] = holdout['text'].map(lambda x: re.sub("(http://.*?\s)|(http://.*)",'',str(x)))
x = train['text']
y = train.iloc[:, 2:6]  
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size= 0.2, random_state=13)
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)
# Instantiate the vectorizer
word_vectorizer = TfidfVectorizer(
    stop_words='english',
    sublinear_tf=True,
    strip_accents='unicode',
    analyzer='word',
    token_pattern=r'\w{2,}',  #vectorize 2-character words or more
    ngram_range=(1, 1),
    max_features=30000)

# fit and transform on it the training features
word_vectorizer.fit(X_train)
X_train_word_features = word_vectorizer.transform(X_train)

#transform the test features to sparse matrix
test_features = word_vectorizer.transform(X_test)

# transform the holdout text for submission at the end
holdout_text = holdout['text']
holdout_word_features = word_vectorizer.transform(holdout_text)
class_names = ['Depression','Alcohol','Suicide','Drugs']

losses = []
auc = []

for class_name in class_names:
    #call the labels one column at a time so we can run the classifier on them
    train_target = y_train[class_name]
    test_target = y_test[class_name]
    classifier = LogisticRegression(solver='sag', C=10)

    cv_loss = np.mean(cross_val_score(classifier, X_train_word_features, train_target, cv=5, scoring='neg_log_loss'))
    losses.append(cv_loss)
    print('CV Log_loss score for class {} is {}'.format(class_name, cv_loss))

    cv_score = np.mean(cross_val_score(classifier, X_train_word_features, train_target, cv=5, scoring='accuracy'))
    print('CV Accuracy score for class {} is {}'.format(class_name, cv_score))
    
    classifier.fit(X_train_word_features, train_target)
    y_pred = classifier.predict(test_features)
    y_pred_prob = classifier.predict_proba(test_features)[:, 1]
    auc_score = metrics.roc_auc_score(test_target, y_pred_prob)
    auc.append(auc_score)
    print("CV ROC_AUC score {}\n".format(auc_score))
    
    print(confusion_matrix(test_target, y_pred))
    print(classification_report(test_target, y_pred))

print('Total average CV Log_loss score is {}'.format(np.mean(losses)))
print('Total average CV ROC_AUC score is {}'.format(np.mean(auc)))

x = train['text']
y = train.iloc[:, 2:8]  
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size= 0.2, random_state=13)
start_time=time.time()

pipe = make_pipeline(TfidfVectorizer(
                                    stop_words='english',
                                    strip_accents='unicode',
                                    token_pattern=r'\w{1,}', #accept tokens that have 1 or more characters
                                    analyzer='word',
                                    ngram_range=(1, 1),
                                    min_df=5),
                     OneVsRestClassifier(LogisticRegression()))
param_grid = {'tfidfvectorizer__max_features': [10000, 30000],
              'onevsrestclassifier__estimator__solver': ['liblinear', 'sag'],
             } 
grid = GridSearchCV(pipe, param_grid, cv=3, scoring='roc_auc')

grid3 = grid.fit(X_train, y_train)

end_time=time.time()
print("total time",end_time-start_time)
# Save classifier to a file

#save_classif = open("Tfidf_LogR_3.pickle", 'wb')   #wb= write in bytes. 
#pickle.dump(grid3, save_classif)   #use pickle to dump the grid3 we trained, as 'Tfidf_LogR.pickle' in wb format
#save_classifier.close() 
# Retrieve the saved file and uplaod it to an object

# vec = open("Tfidf_LogR_3.pickle", 'rb') # rb= read in bytes
# grid3 = pickle.load(vec)
# vec.close()
print(grid3.best_estimator_.named_steps['onevsrestclassifier'])
print(grid3.best_estimator_.named_steps['tfidfvectorizer'])
grid3.best_params_
grid3.best_score_
predicted_y_test = grid3.predict(X_test)
print("Depression Confusion Matrixs: \n{}".format(confusion_matrix(y_test['Depression'], predicted_y_test[:,0])))
print("\nAlcohol: \n{}".format(confusion_matrix(y_test['Alcohol'], predicted_y_test[:,1])))
print("\nSuicide: \n{}".format(confusion_matrix(y_test['Suicide'], predicted_y_test[:,2])))
print("\nDrugs: \n{}".format(confusion_matrix(y_test['Drugs'], predicted_y_test[:,3])))
#print("\nInsult: \n{}".format(confusion_matrix(y_test['insult'], predicted_y_test[:,4])))
#print("\nIdentity Hate: \n{}".format(confusion_matrix(y_test['identity_hate'], predicted_y_test[:,5])))

print("\nDepression Classification report: \n{}".format(classification_report(y_test['Depression'], predicted_y_test[:,0])))
print("\nAlcohol: \n{}".format(classification_report(y_test['Alcohol'], predicted_y_test[:,1])))
print("\nSuicide: \n{}".format(classification_report(y_test['Suicide'], predicted_y_test[:,2])))
print("\nDrugs: \n{}".format(classification_report(y_test['Drugs'], predicted_y_test[:,3])))
#print("\nInsult: \n{}".format(classification_report(y_test['insult'], predicted_y_test[:,4])))
#print("\nIdentity Hate: \n{}".format(classification_report(y_test['identity_hate'], predicted_y_test[:,5])))

 #grid3.cv_results_
vectorizer = grid3.best_estimator_.named_steps["tfidfvectorizer"]
# transform the training dataset:
X_test_set = vectorizer.transform(X_test)


# find maximum value for each of the features over dataset:
max_value = X_test_set.max(axis=0).toarray().ravel()
sorted_by_tfidf = max_value.argsort()

# get feature names
feature_names = np.array(vectorizer.get_feature_names())

print("Features with lowest tfidf:\n{}".format(
      feature_names[sorted_by_tfidf[:20]]))

print("\nFeatures with highest tfidf: \n{}".format(
      feature_names[sorted_by_tfidf[-20:]]))
sorted_by_idf = np.argsort(vectorizer.idf_)
print("Features with lowest idf:\n{}".format(
       feature_names[sorted_by_idf[:100]]))
# print(y_train.columns)
# print("\n-Columns are ordered as above, which is why coef_[0] refers to toxic and coef_[5] refers to identity hate.")
# print("-The blue bars refer to the label (toxic here) and the red refer to Not toxic")
# mglearn.tools.visualize_coefficients(
#     grid3.best_estimator_.named_steps["onevsrestclassifier"].coef_[0],
#     feature_names, n_top_features=40)
# mglearn.tools.visualize_coefficients(
#     grid3.best_estimator_.named_steps["onevsrestclassifier"].coef_[1],
#     feature_names, n_top_features=40)
# mglearn.tools.visualize_coefficients(
#     grid3.best_estimator_.named_steps["onevsrestclassifier"].coef_[5],
#     feature_names, n_top_features=40)
holdout_comments = holdout['text']
# holdoutComments are automatically transformed throguh the grid3 pipeline before prodicting probabilities
twod = grid3.predict_proba(holdout_comments)
holdout_predictions = {}
holdout_predictions = {'ID': holdout['ID']}  

holdout_predictions['Depression']=twod[:,0]
holdout_predictions['Alcohol']=twod[:,1]
holdout_predictions['Suicide']=twod[:,2]
holdout_predictions['Drugs']=twod[:,3]
#holdout_predictions['insult']=twod[:,4]
#holdout_predictions['identity_hate']=twod[:,5]
    
submission = pd.DataFrame.from_dict(holdout_predictions)
submission = submission[['ID','Depression','Alcohol','Suicide','Drugs']] #rearrange columns
submission.to_csv('submission.csv', index=False)
# calculate lenght of characters in each comment
train['len_character'] = train['text'].apply(lambda x: len(re.findall(r"[\w]", str(x))))
from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import FeatureUnion #unites all arrays into one array
from sklearn.pipeline import Pipeline
x = train[['text','len_character']] #these will be our features
y = train.iloc[:, 2:6]  
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size= 0.2, random_state=13)
# Preprocess the numeric data: get_numeric_data
get_numeric_data = FunctionTransformer(lambda a: a[['len_character']], validate=False)
get_text_data = FunctionTransformer(lambda a: a['text'], validate=False)

print(get_text_data.fit_transform(X_train).shape)
print(get_numeric_data.fit_transform(X_train).shape)
pl = Pipeline([
        ('union', FeatureUnion(                      #unites both text and numeric arrays into one array
            transformer_list = [
                ('numeric_features', Pipeline([
                    ('selector', get_numeric_data)
                ])),
                ('text_features', Pipeline([
                    ('selector', get_text_data),
                    ('vectorizer', TfidfVectorizer(
                                                    stop_words='english',
                                                    strip_accents='unicode',
                                                    token_pattern=r'\w{2,}',
                                                    analyzer='word',
                                                    ngram_range=(1, 1),
                                                    min_df=5))
                ]))
             ]
        )), #right here is where we would put interaction terms preprocessing such as PolynomialFeatures
            #(right here is where we would put a scaler if we needed one)
        ('clf', OneVsRestClassifier(LogisticRegression())) 
    ])
param_grid = {'union__text_features__vectorizer__max_features': [10000, 30000],
              'clf__estimator__C': [0.1, 1]
             } 
grid = GridSearchCV(pl, param_grid, cv=3, scoring='roc_auc')

grid4 = grid.fit(X_train, y_train)
# # Pickle grid4 to your computer
#dill: this is necessary in order for pickle to save grid4 which has a lambda function inside of it.
import dill as pickled

# save_grid4 = open("Tfidf_LogR_4.pickle", 'wb') #wb= write in bytes. 'Tfidf_LogR.pickle' is the name of the file saved
# pickled.dump(grid4, save_grid4) #use pickle to dump the grid1 we trained as 'Tfidf_LogR.pickle' in wb format
# save_grid4.close() 
import dill as pickled
# Retrieve the saved file and uplaod it to an object

# vec4 = open("Tfidf_LogR_4.pickle", 'rb') # rb= read in bytes
# grid4 = pickled.load(vec4)
# vec4.close()
print(grid4.best_score_)
print(grid4.best_params_)
print(grid4.estimator)
pred_y_test = grid4.predict(X_test)

print("Depression Confusion Matrixs: \n{}".format(confusion_matrix(y_test['Depression'], pred_y_test[:,0])))
print("\nAlcohol: \n{}".format(confusion_matrix(y_test['Alcohol'], pred_y_test[:,1])))
print("\nSuicide: \n{}".format(confusion_matrix(y_test['Suicide'], pred_y_test[:,2])))
print("\nDrugs: \n{}".format(confusion_matrix(y_test['Drugs'], pred_y_test[:,3])))
#print("\nInsult: \n{}".format(confusion_matrix(y_test['insult'], pred_y_test[:,4])))
#print("\nIdentity Hate: \n{}".format(confusion_matrix(y_test['identity_hate'], pred_y_test[:,5])))

print("\nDepression Classification report: \n{}".format(classification_report(y_test['Depression'], pred_y_test[:,0])))
print("\nAlcohol: \n{}".format(classification_report(y_test['Alcohol'], pred_y_test[:,1])))
print("\nSuicide: \n{}".format(classification_report(y_test['Suicide'], pred_y_test[:,2])))
print("\nDrugs: \n{}".format(classification_report(y_test['Drugs'], pred_y_test[:,3])))
#print("\nInsult: \n{}".format(classification_report(y_test['insult'], pred_y_test[:,4])))
#print("\nIdentity Hate: \n{}".format(classification_report(y_test['identity_hate'], pred_y_test[:,5])))
