import time

start_time = time.time()
import numpy as np 

import pandas as pd 

import os



from sklearn.model_selection import train_test_split

from sklearn.model_selection import StratifiedKFold, KFold, GridSearchCV

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.svm import SVC

from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
train_df = pd.read_csv("/kaggle/input/nlp-getting-started/train.csv")

test_df = pd.read_csv("/kaggle/input/nlp-getting-started/test.csv")

sub_df = pd.read_csv("/kaggle/input/nlp-getting-started/sample_submission.csv")
train_df.head()
train_df.isna().sum()
X = train_df["text"]

y = train_df["target"]

X_test = test_df["text"]

X.shape, y.shape, X_test.shape
X_for_tf_idf = pd.concat([X, X_test])
tfidf = TfidfVectorizer(

                        stop_words = 'english',

#                       token_pattern=r'(?u)\b\w\w+\b',

                        token_pattern=r'(?u)(\b\w\w+\b|\#|\@)'                        

)





tfidf.fit(X_for_tf_idf)



X = tfidf.transform(X)

X_test = tfidf.transform(X_test)

del X_for_tf_idf
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=42)
parameters = { 

            'gamma': [0.7 , 1. , 'auto', 'scale']

}

model = GridSearchCV(

#                       SVC(kernel='rbf'), 

                        SVC(kernel='sigmoid'), 

                        parameters, cv=4, n_jobs=-1

).fit(X_train, y_train)
y_val_pred = model.predict(X_val)

accuracy_score(y_val, y_val_pred), f1_score(y_val, y_val_pred)
confusion_matrix(y_val, y_val_pred)
y_test_pred = model.predict(X_test)

sub_df["target"] = y_test_pred
train_df_copy = train_df

train_df_copy = train_df_copy.fillna('None')

ag = train_df_copy.groupby('keyword').agg({'text':np.size, 'target':np.mean}).rename(columns={'text':'Count', 'target':'Disaster Probability'})



ag.sort_values('Disaster Probability', ascending=False).head(20)
count = 2

prob_disaster = 0.9

keyword_list_disaster = list(ag[(ag['Count']>count) & (ag['Disaster Probability']>=prob_disaster)].index)

#we print the list of keywords which will be used for prediction correction 

keyword_list_disaster
ids_disaster = test_df['id'][test_df.keyword.isin(keyword_list_disaster)].values

sub_df['target'][sub_df['id'].isin(ids_disaster)] = 1

sub_df.to_csv("submission.csv", index=False)

sub_df.head(20)
print("--- %s seconds ---" % (time.time() - start_time))