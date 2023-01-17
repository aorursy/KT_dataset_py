# # import the modules we'll need

# from IPython.display import HTML

# import pandas as pd

# import numpy as np

# import base64



# # function that takes in a dataframe and creates a text link to  

# # download it (will only work for files < 2MB or so)

# def create_download_link(df, title = "Download CSV file", filename = "data.csv"):  

#     csv = df.to_csv()

#     b64 = base64.b64encode(csv.encode())

#     payload = b64.decode()

#     html = '<a download="{filename}" href="data:text/csv;base64,{payload}" target="_blank">{title}</a>'

#     html = html.format(payload=payload,title=title,filename=filename)

#     return HTML(html)
import os

print(os.listdir("../working"))
import os

print(os.listdir('..'))
import numpy as np

import pandas as pd

import time

from sklearn.feature_extraction.text import TfidfVectorizer, HashingVectorizer

from sklearn.model_selection import GridSearchCV

from sklearn.linear_model import SGDClassifier, LogisticRegression

from sklearn.pipeline import Pipeline

from sklearn.metrics import roc_auc_score, precision_score, recall_score, roc_curve, make_scorer

import matplotlib.pyplot as plt

import scipy.sparse

import pickle
tic = time.time()

X_train_str = pd.read_csv("../input/x_train.txt", delimiter= "\r\t", header = None, names = ['comment'], dtype = str)

X_train_str = np.squeeze(X_train_str.values)

toc = time.time()

print(toc-tic)
tic = time.time()

Y_train_str = pd.read_csv('../input/y_train.csv', header = 0, sep =',')['Probability'].values

toc = time.time()

print(toc-tic)
tic = time.time()

X_test_str = pd.read_csv("../input/x_test.txt", delimiter= "\r\t", header = None, names = ['comment'], dtype = str)

X_test_str = np.squeeze(X_test_str.values)

toc = time.time()

print(toc-tic)
X_train_small = X_train_str[0:int(len(X_train_str)*0.8)].copy()

Y_train_small = Y_train_str[0:int(len(Y_train_str)*0.8)].copy()
text_clf2 = Pipeline([

    ('tfidf', TfidfVectorizer()),

    ('clf', LogisticRegression())

])
text_clf3 = Pipeline([

    ('tfidf', TfidfVectorizer(ngram_range=(1, 2), max_features=800000, max_df=0.8, norm='l2', use_idf=True, sublinear_tf=True)),

    ('clf', LogisticRegression(random_state=23, C=5))

])
tic = time.time()

text_clf3.fit(X_train_small, Y_train_small)

toc = time.time()

print(toc-tic)
predicted_answ = text_clf3.predict_proba(X_test_str)

predicted_answ = predicted_answ[:,1]
predicted_answ = predicted_answ.reshape((len(predicted_answ),1))

ind = np.array(range(1,400001,1), dtype=int)

ind = ind.reshape((len(ind),1))

kaggle_answ = np.hstack((ind, predicted_answ))

kaggle_answ_pd = pd.DataFrame(kaggle_answ, columns=['Id', 'Probability'])

kaggle_answ_pd['Id'] = kaggle_answ_pd['Id'].astype(int)

kaggle_answ_pd
# # create a random sample dataframe

# df = kaggle_answ_pd



# # create a link to download the dataframe

# create_download_link(df)



# # ↓ ↓ ↓  Yay, download link! ↓ ↓ ↓ 
# my_submission = pd.DataFrame({'Id': test.Id, 'SalePrice': predicted_prices})

# # you could use any filename. We choose submission here

kaggle_answ_pd.to_csv('kaggle02.csv', index=False)
# # Function for writing predictions to a file

# def write_to_submission_file(predicted_labels, out_file,

#                              target='Probability', index_label="Id"):

#     predicted_df = pd.DataFrame(predicted_answ,

#                                 index = np.arange(1, predicted_answ.shape[0] + 1),

#                                 columns=[target])

#     predicted_df.to_csv(out_file, index_label=index_label)
# write_to_submission_file(predicted_answ, '___tututu.csv')
# parameters = {

#     'tfidf__use_idf': [True],

#     'tfidf__ngram_range': [(1, 2)],

#     'tfidf__sublinear_tf': [True],

# #     'tfidf__max_df': [0.7, 0.8, 0.9, 1.0],

# #     'tfidf__min_df': [0.],

#     'tfidf__norm': ['l2'],

#     'tfidf__max_features': [170000],

    

# #     'clf__loss': ['squared_loss'],

# #     'clf__penalty': ['l2'],

# #     'clf__alpha': [1e-4],

# #     'clf__max_iter': [5],

#     'clf__random_state': [23],

# #     'clf__tol': [None]

# #     'clf__loss': ['squared_loss'],

#     'clf__C': [5]

# }



# roc_score = make_scorer(roc_auc_score, needs_proba=True)

# gs_clf = GridSearchCV(text_clf2, parameters, cv=2, scoring = roc_score, n_jobs=-1)
# tic = time.time()

# gs_clf = gs_clf.fit(X_train_small, Y_train_small)

# toc = time.time()

# print(toc-tic)
# datafr = pd.DataFrame.from_dict(gs_clf.cv_results_).sort_values('rank_test_score')

# datafr[datafr.columns[0:30]]
# datafr.columns
# gs_clf.best_params_
# predicted_answ = gs_clf.best_estimator_.predict_proba(X_test_str)

# predicted_answ = predicted_answ[:,1]
# # Function for writing predictions to a file

# def write_to_submission_file(predicted_labels, out_file,

#                              target='Probability', index_label="Id"):

#     predicted_df = pd.DataFrame(predicted_answ,

#                                 index = np.arange(1, predicted_answ.shape[0] + 1),

#                                 columns=[target])

#     predicted_df.to_csv(out_file, index_label=index_label)
# write_to_submission_file(predicted_answ, '/content/drive/My Drive/Colab Notebooks/Sentiment_Analisys/submission04.csv')