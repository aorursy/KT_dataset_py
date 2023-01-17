# import libraries
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import TruncatedSVD
import pickle
import sqlite3
import nltk
nltk.download(['punkt', 'wordnet'])
# load data from database
#kaggle dont support below lines
#engine = create_engine('sqlite:///database.db')
#df = pd.read_sql_table("tbl_disaster_messages", con = engine)

#solution
# Create database
#database = 'database.sqlite'
#conn = sqlite3.connect(database)
conn = sqlite3.connect("../input/database.db")
df = pd.read_sql_query("SELECT * FROM table1", conn)
df.head()
X = df["message"]
Y = df.drop(['message', 'genre', 'id', 'original'], axis = 1)
def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)
    return clean_tokens
pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])
pipeline.get_params()
X_train, X_test, y_train, y_test = train_test_split(X, Y,test_size = 0.2, random_state = 45)

# train classifier
pipeline.fit(X_train, y_train)
def perf_report(model, X_test, y_test):
    '''
    Function to generate classification report on the model
    Input: Model, test set ie X_test & y_test
    Output: Prints the Classification report
    '''
    y_pred = model.predict(X_test)
    for i, col in enumerate(y_test):
        print(col)
        print(classification_report(y_test[col], y_pred[:, i]))
perf_report(pipeline, X_test, y_test)
parameters =  {'tfidf__use_idf': (True, False), 
              'clf__estimator__n_estimators': [50, 100], 
              'clf__estimator__min_samples_split': [2, 4]} 

cv = GridSearchCV(pipeline, param_grid=parameters)
cv
cv.fit(X_train, y_train)

perf_report(cv, X_test, y_test)
#Improve  the pipeline
pipeline2 = Pipeline([
    ('vect', CountVectorizer()),
    ('best', TruncatedSVD()),
    ('tfidf', TfidfTransformer()),
    ('clf', MultiOutputClassifier(AdaBoostClassifier()))
])
pipeline2.get_params()
#Train & predict
pipeline2.fit(X_train, y_train)
perf_report(pipeline2, X_test, y_test)

#Param tunning 
parameters2 = { #'vect__ngram_range': ((1, 1), (1, 2)), 
              #'vect__max_df': (0.5, 1.0), 
              #'vect__max_features': (None, 5000), 
              'tfidf__use_idf': (True, False), 
              'clf__estimator__n_estimators': [50, 100],
              'clf__estimator__learning_rate': [1,2] }
cv2 = GridSearchCV(pipeline2, param_grid=parameters2)
cv2
cv2.fit(X_train, y_train)
perf_report(cv2, X_test, y_test)
with open('disaster_model.pkl', 'wb') as f:
    pickle.dump(cv2, f)
