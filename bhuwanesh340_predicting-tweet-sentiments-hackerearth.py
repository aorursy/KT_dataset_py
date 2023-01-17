import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

train = pd.read_csv("../input/predicting-tweet-sentiments/train.csv")
train.head()
train.shape
train.columns
train.isnull().sum()
train.dtypes
train.describe()
train.columns
train.head()
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# DOWNLOAD VADER_LEXICON FOR CALCULATING SENTIMENT INTENSITY
nltk.download('vader_lexicon')

from nltk.sentiment.vader import SentimentIntensityAnalyzer
sid = SentimentIntensityAnalyzer()

nltk.download('wordnet')
def tweet_to_words(raw_tweet):
    letters_only = re.sub("[^a-zA-Z]", " ",raw_tweet) 
    words = letters_only.lower().split()                             
    stops = set(stopwords.words("english"))                  
    meaningful_words = [w for w in words if not w in stops] 
    return( " ".join( meaningful_words ))
#FUNCTION TO FIND POLARITY SCORE
def polar_score(text):
    score = sid.polarity_scores(text)
    x = score['compound']
    return x

#CREATING A COLUMN NAMED CLEAN_TEXT WHICH STORES CLEANED TEXT 
train['clean_text']=train['original_text'].apply(lambda x: tweet_to_words(x))

#CREATING A COLUMN NAMED COMPOUND SCORE WITH SENTIMENT INTENSITY AS ITS VALUE 
train['compound_score'] = train['original_text'].apply(lambda x : polar_score(x))

#CREATING A COLUMN NAMED LENGTH WITH LENGTH OF REVIEWS AS ITS VALUE
train['length'] = train['clean_text'].apply(lambda x: len(x) - x.count(" "))
train.head()
train.shape
# "LANG" varibale has lots of irrelevant values

train = train[train['lang'] == "en"]
train.shape
train.compound_score.plot(kind="hist", color="lime")
train.length.plot(kind="hist", color="teal")
train['retweet_count'].value_counts().plot(kind="bar")
train.sentiment_class.value_counts().plot(kind='bar', color=['purple', 'orange', 'lime'])


test = pd.read_csv('../input/predicting-tweet-sentiments/test.csv')

test.shape
test.head(5)
#CREATING A COLUMN NAMED CLEAN_TEXT WHICH STORES CLEANED TEXT 
test['clean_text']=test['original_text'].apply(lambda x: tweet_to_words(x))

#CREATING A COLUMN NAMED COMPOUND SCORE WITH SENTIMENT INTENSITY AS ITS VALUE 
test['compound_score'] = test['original_text'].apply(lambda x : polar_score(x))

#CREATING A COLUMN NAMED LENGTH WITH LENGTH OF REVIEWS AS ITS VALUE
test['length'] = test['clean_text'].apply(lambda x: len(x) - x.count(" "))
test.head()

# train = 2994
# test  = 1387
common_df = pd.DataFrame(pd.concat([train["clean_text"], test["clean_text"]])).reset_index(drop=True)
common_df = pd.concat([train["clean_text"], test["clean_text"]]).reset_index(drop=True)
common_df.head()
type(common_df)
len(common_df)
common_df[0]
documents = []

from nltk.stem import WordNetLemmatizer

stemmer = WordNetLemmatizer()

for sen in range(0, len(common_df)):
    # Remove all the special characters
    document = re.sub(r'\W', ' ', str(common_df[sen]))
    
    # remove all single characters
    document = re.sub(r'\s+[a-zA-Z]\s+', ' ', document)
    
    # Remove single characters from the start
    document = re.sub(r'\^[a-zA-Z]\s+', ' ', document) 
    
    # Substituting multiple spaces with single space
    document = re.sub(r'\s+', ' ', document, flags=re.I)
    
    # Removing prefixed 'b'
    document = re.sub(r'^b\s+', '', document)
    
    # Converting to Lowercase
    document = document.lower()
    
    # Lemmatization
    document = document.split()

    document = [stemmer.lemmatize(word) for word in document]
    document = ' '.join(document)
    
    documents.append(document)

from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer(max_features=500, min_df=5, max_df=0.7, stop_words=stopwords.words('english'))
X = vectorizer.fit_transform(documents).toarray()
X
from sklearn.feature_extraction.text import TfidfTransformer
tfidfconverter = TfidfTransformer()
X = tfidfconverter.fit_transform(X).toarray()
X
x_df = pd.DataFrame(X)
for col in x_df.columns:
    new_col = "Column_" + str(col)
    #print(new_col)
    x_df.rename(columns={col:new_col}, inplace=True)
x_df.shape
train_x_df = x_df[:2994]
test_x_df = x_df[2994:]
test_x_df.reset_index(drop=True, inplace=True)
train_new = pd.concat([train, train_x_df], axis=1)
test_new = pd.concat([test, test_x_df], axis=1, ignore_index=True)

train_new.columns
train_new.reset_index(drop=True, inplace=True)
test_new.reset_index(drop=True, inplace=True)
x = train_new.drop(['id', 'original_text', 'lang', 'original_author', 'sentiment_class', 'clean_text'], axis=1)
y = train_new['sentiment_class']
x.shape
x.fillna(x.median(), inplace=True)
for col in x.columns:
    x[col] = x[col].apply(lambda x: x.median() if x == np.inf or x == -np.inf else x)
x["length"].dtype
x.columns[1:]
for col in x.columns[1:]:
    x[col] = x[col].apply(lambda x: float("{:.2f}".format(x)) )
for col in x.columns:
    x[col] = x[col].astype(np.float64)
x.reset_index(drop=True, inplace=True)
for col in x.columns:
    x[col] = x[col].apply(lambda x: x.median() if x == "" else x)
x
cols_with_missing = [col for col in x.columns 
                                 if x[col].isnull().any()]

y
y.fillna(0.0, inplace=True)
y.isnull().any()

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 100)
print("X_train: ", X_train.shape) 
print("X_test: ", X_test.shape)
print("y_train: ", y_train.shape) 
print("y_test: ", y_test.shape)
X_train.reset_index(drop=True, inplace=True)
X_test.reset_index(drop=True, inplace=True)
y_train.reset_index(drop=True, inplace=True)
y_test.reset_index(drop=True, inplace=True)
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression

# feature extraction
model = LogisticRegression(solver='lbfgs')
rfe = RFE(model, 3)
fit = rfe.fit(x, y)

print("Num Features: %d" % fit.n_features_)
print("Selected Features: %s" % fit.support_)
print("Feature Ranking: %s" % fit.ranking_)


df_feat = pd.DataFrame(fit.ranking_, x.columns)
df_feat.rename(columns = {0:"Feature_Ranking"}, inplace=True)

df_feat[:15].sort_values(by="Feature_Ranking").plot(kind='bar', figsize=(18,7))

from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_curve, auc, classification_report, confusion_matrix, accuracy_score, roc_auc_score
from sklearn.tree import DecisionTreeClassifier

#making the instance
model= DecisionTreeClassifier(random_state=1234)

#Hyper Parameters Set
param_grid = {'max_features': ['auto', 'sqrt', 'log2'],
          'min_samples_split': [2,3,4,5,6,7,8,9,10,11,12,13,14,15], 
          'min_samples_leaf':[1,2,3,4,5,6,7,8,9,10,11],
          'random_state':[123]}

# Create grid search object
clf = GridSearchCV(model, param_grid=param_grid, n_jobs=-1, cv=10)

# Fit on data
best_clf_dt = clf.fit(X_train, y_train)

#Predict
predictions = best_clf_dt.predict(X_test)

print("*******************ACCURACY***************************************************************")
#Check Prediction Score
print("Accuracy of Decision Trees: ",accuracy_score(y_test, predictions))

print("*******************CLASSIFICATION - REPORT***************************************************************")
print("Confusion matrix \n",confusion_matrix(y_test,predictions))
print(classification_report(y_test,predictions))




# Fit RF on full data

from sklearn.ensemble import RandomForestClassifier

#making the instance
model= RandomForestClassifier(random_state=1234)

#Hyper Parameters Set
param_grid = {'criterion':['gini','entropy'],
          'n_estimators':[10,15,20,25,30],
          'min_samples_leaf':[1,2,3],
          'min_samples_split':[3,4,5,6,7], 
          'random_state':[123],
          'n_jobs':[-1]}

# Create grid search object
clf = GridSearchCV(model, param_grid=param_grid, n_jobs=-1, cv=10)

# Fit on data
best_clf_rf = clf.fit(X_train, y_train)

#Predict
predictions = best_clf_rf.predict(X_test)

#Check Prediction Score
print("Accuracy of Random Forest: ",accuracy_score(y_test, predictions))

#Print Classification Report
print("Confusion matrix \n",confusion_matrix(y_test,predictions))
print(classification_report(y_test,predictions))



# Fit RF on full data

from sklearn.ensemble import RandomForestClassifier

#making the instance
model= RandomForestClassifier(random_state=1234)

#Hyper Parameters Set
param_grid = {'criterion':['gini','entropy'],
          'n_estimators':[10,15,20,25,30],
          'min_samples_leaf':[1,2,3],
          'min_samples_split':[3,4,5,6,7], 
          'random_state':[123],
          'n_jobs':[-1]}

# Create grid search object
clf = GridSearchCV(model, param_grid=param_grid, n_jobs=-1, cv=10)

# Fit on data
best_clf_rf = clf.fit(x, y)


test_new = pd.concat([test, test_x_df], axis=1)
test_new.shape
test_new.dtypes
import re
def check_integer(x):
    try:
        num = int(x) 
        return x
    except: 
        return 0
test_new["retweet_count"] = test_new["retweet_count"].apply(lambda x: check_integer(x))
test_for_prediction = test_new.drop(['id', 'original_text', 'lang', 'original_author', 'clean_text'], axis=1)
for col in test_for_prediction.columns:
    test_for_prediction[col] = test_for_prediction[col].astype(np.float64)
test_for_prediction.fillna(test_for_prediction.median(), inplace=True)

for col in test_for_prediction.columns:
    test_for_prediction[col] = test_for_prediction[col].apply(lambda x: x.median() if x == np.inf or x == -np.inf else x)

for col in test_for_prediction.columns[1:]:
    test_for_prediction[col] = test_for_prediction[col].apply(lambda x: float("{:.2f}".format(x)))

test_for_prediction.reset_index(drop=True, inplace=True)


test_for_prediction.head()
#Predict

prediction_from_dt  = best_clf_dt.predict(test_for_prediction)
df_prediction_from_dt = pd.DataFrame({'id':test_new['id'], 'sentiment_class': prediction_from_dt})
df_prediction_from_dt.to_csv("Final_output_prediction_from_dt.csv")
df_prediction_from_dt.head()

prediction_from_rf  = best_clf_rf.predict(test_for_prediction)
df_prediction_from_rf = pd.DataFrame({'id':test_new['id'], 'sentiment_class': prediction_from_rf})
df_prediction_from_rf.to_csv("Final_output_prediction_from_rf.csv")
df_prediction_from_rf.head()
