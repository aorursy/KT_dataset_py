import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from sklearn.model_selection import train_test_split as tts, GridSearchCV
from sklearn.pipeline import Pipeline

from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

import pickle
import warnings
warnings.filterwarnings('ignore')
df = pd.read_csv('spam_sms_collection.csv').drop(['Unnamed: 0'], axis=1)
df.dropna(inplace=True)
df.head()
def create_classification_report(Y_test, Y_pred):
    print('--------Classification Report---------\n')
    accuracy = accuracy_score(Y_test, Y_pred)
    f1 = f1_score(Y_test, Y_pred)
    precision = precision_score(Y_test, Y_pred)
    recall = recall_score(Y_test, Y_pred)
    roc_auc = roc_auc_score(Y_test, Y_pred)
    metrices = [accuracy, f1, precision, recall, roc_auc]
    scores = pd.DataFrame(pd.Series(metrices).values, index=['accuracy', 'f1-score', 'precision', 'recall', 'roc auc score'], columns=['score'])
    print(scores)
    print('\n--------Plotting Confusion Matrix---------')
    sns.heatmap(confusion_matrix(Y_test, Y_pred), annot=True, cmap='RdYlGn_r', annot_kws={'size': 16})
    return scores
X = df['msg'] # Independent Features
Y = df['spam'] # Dependent Features
X_train, X_test, Y_train, Y_test = tts(X, Y, test_size=0.2, random_state=44)
pipe = Pipeline([
    ('vectorizer', CountVectorizer()),
    ('model', MultinomialNB())
])
param_grid = [
    {
        'vectorizer': [CountVectorizer()],
        'vectorizer__max_features': [2500, 3000, 3500, 4000, 4500, 5000, 5500, 6000, 6500],
        'model': [MultinomialNB()]
    },
    {
        'vectorizer': [TfidfVectorizer()],
        'vectorizer__max_features': [2500, 3000, 3500, 4000, 4500, 5000, 5500, 6000, 6500],
        'model': [MultinomialNB()]
    }
]
best_pipeline = GridSearchCV(
    estimator=pipe, 
    param_grid=param_grid, 
    cv=10, 
    scoring='accuracy', 
    n_jobs=1
)
pipeline1 = best_pipeline.fit(X_train, Y_train)
Y_pred = best_pipeline.predict(X_test)
scores1 = create_classification_report(Y_test, Y_pred)
best_pipeline = GridSearchCV(
    estimator=pipe, 
    param_grid=param_grid, 
    cv=10, 
    scoring='f1', 
    n_jobs=1
)
pipeline2 = best_pipeline.fit(X_train, Y_train)
Y_pred = best_pipeline.predict(X_test)
scores2 = create_classification_report(Y_test, Y_pred)
best_pipeline = GridSearchCV(
    estimator=pipe, 
    param_grid=param_grid, 
    cv=10, 
    scoring='precision', 
    n_jobs=1
)
pipeline3 = best_pipeline.fit(X_train, Y_train)
Y_pred = best_pipeline.predict(X_test)
scores3 = create_classification_report(Y_test, Y_pred)
best_pipeline = GridSearchCV(
    estimator=pipe, 
    param_grid=param_grid, 
    cv=10, 
    scoring='roc_auc', 
    n_jobs=1
)
pipeline4 = best_pipeline.fit(X_train, Y_train)
Y_pred = best_pipeline.predict(X_test)
scores4 = create_classification_report(Y_test, Y_pred)
results = pd.concat([scores1, scores2, scores3, scores4], axis=1)
results.columns = ['pipeline1', 'pipeline2', 'pipeline3', 'pipeline4']
results
results.plot(kind='line', linewidth=3)
file = open('best_pipeline.pkl', 'wb')
pickle.dump(pipeline2, file)
