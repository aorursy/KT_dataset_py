import pickle

import pandas as pd

import numpy as np

import string

import gzip

import seaborn as sns

import matplotlib.pyplot as plt
#load datasets





with gzip.open("../input/text-classification-4-1-naive-bayes/nb_results.pkl", 'rb') as data:

    nb = pickle.load(data)

    

    

with gzip.open("../input/text-classification-4-2-rf/rf_results.pkl", 'rb') as data:

    rf = pickle.load(data)

    

#here svm_results.pkl is a typo. it is the results of logit

with gzip.open("../input/text-classification-4-3-logistic-regression/svm_results.pkl", 'rb') as data:

    logit = pickle.load(data)



    

with gzip.open("../input/text-classification-4-5-xgboost-classifier/xgb_results.pkl", 'rb') as data:

    xg = pickle.load(data)





with gzip.open("../input/text-classification-4-6-word-embedding-nnvanilla/vanilla_nn_results.pkl", 'rb') as data:

    vanilla_nn = pickle.load(data)

    

    

with gzip.open("../input/text-classification-4-7-cnn-glove-6b-50d/glove_cnn_results.pkl", 'rb') as data:

    cnn = pickle.load(data)

    

#maybe svm if we can see the output:(

#with gzip.open("../input/text-classification-3-2-text-representation/x_train_tfidf.pkl", 'rb') as data:

#    x_train_tfidf = pickle.load(data)

       

        
nb['model_name'] = 'nb_' + nb['model_name'].astype(str)

rf['model_name'] = 'rf_' + rf['model_name'].astype(str)

logit['model_name'] = 'logit_' + logit['model_name'].astype(str)

xg['model_name'] = 'xg_' + xg['model_name'].astype(str)

all_df=pd.concat([nb, rf, logit, xg, vanilla_nn, cnn], axis=0).set_index('model_name')
all_df
#plt.figure({figsize=(30,15)})



all_df.plot.bar()



plt.rcParams["figure.figsize"] = [20, 5]