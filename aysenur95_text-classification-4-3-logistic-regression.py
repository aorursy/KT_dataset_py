import pickle

import pandas as pd

import numpy as np

import string

import gzip

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.model_selection import StratifiedShuffleSplit

from sklearn.linear_model import LogisticRegression

#import utility functions

from text_classification_utility_functions import run_model

#load datasets





with gzip.open("../input/text-classification-3-2-text-representation/y_train.pkl", 'rb') as data:

    y_train = pickle.load(data)

    

    

with gzip.open("../input/text-classification-3-2-text-representation/y_test.pkl", 'rb') as data:

    y_test = pickle.load(data)

    

    

with gzip.open("../input/text-classification-3-1-text-representation/x_train_1hot.pkl", 'rb') as data:

    x_train_1hot = pickle.load(data)



    

with gzip.open("../input/text-classification-3-1-text-representation/x_test_1hot.pkl", 'rb') as data:

    x_test_1hot = pickle.load(data)





with gzip.open("../input/text-classification-3-1-text-representation/x_train_bow.pkl", 'rb') as data:

    x_train_bow = pickle.load(data)

    

    

with gzip.open("../input/text-classification-3-1-text-representation/x_test_bow.pkl", 'rb') as data:

    x_test_bow = pickle.load(data)

    



with gzip.open("../input/text-classification-3-2-text-representation/x_train_tfidf.pkl", 'rb') as data:

    x_train_tfidf = pickle.load(data)

       

        

with gzip.open("../input/text-classification-3-2-text-representation/x_test_tfidf.pkl", 'rb') as data:

    x_test_tfidf = pickle.load(data)
df_name_list=["one_hot", "bow", "tf-idf"]

df_list=[(x_train_1hot, x_test_1hot), (x_train_bow, x_test_bow), (x_train_tfidf, x_test_tfidf)]

res_list=[]

logit = LogisticRegression(random_state=0)



#from other account result: The best parameters are {'C': 1, 'max_iter': 1000, 'penalty': 'l2'} with a score of 0.8358



# cross-validation with 10 splits

cv = StratifiedShuffleSplit(n_splits=5, random_state = 42, test_size=0.2)

params = {

                "penalty":['l2'],

                "C": [1],

                "max_iter": [1000],

             }



for X_train, X_test in df_list:

    best_nb, false_preds, test_acc, train_acc = run_model(logit, params, cv, X_train, X_test, y_train, y_test )

    res_list.append((best_nb, false_preds, test_acc, train_acc))
max_acc=0

max_idx=0



for i, res_tuple in enumerate(res_list):

    if res_tuple[2] > max_acc:

        max_acc=res_tuple[2]

        max_idx=i

print("Best accuracy belongs to:", df_name_list[max_idx], ", acc:", max_acc)



res_final_df=pd.DataFrame(columns=["model_name", "train_acc", "test_acc"])

for i, res_tuple in enumerate(res_list):

    res_final_df=res_final_df.append({"model_name": df_name_list[i], "train_acc": res_tuple[3], "test_acc": res_tuple[2]}, ignore_index=True)

    

res_final_df

#pickle best model

with gzip.open('svm_best_model.pkl', 'wb') as output:

    pickle.dump(res_list[max_idx][0], output, protocol=-1)

    

#pickle results

with gzip.open('svm_results.pkl', 'wb') as output:

    pickle.dump(res_final_df, output, protocol=-1)