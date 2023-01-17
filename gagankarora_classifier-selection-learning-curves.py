#Base kernel: https://www.kaggle.com/abhishek/maybe-something-interesting-here

import numpy as np
import pandas as pd
import os
import xgboost as xgb
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from scipy import sparse
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
import scipy as sp
from sklearn import linear_model
from functools import partial
from sklearn import metrics
from collections import Counter
import json
import lightgbm as lgb

import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.model_selection import learning_curve

import random as rand
from sklearn.preprocessing import StandardScaler 

import warnings
warnings.filterwarnings('ignore')
train = pd.read_csv('../input/train/train.csv')
test = pd.read_csv('../input/test/test.csv')
# train[train.AdoptionSpeed==0]
doc_sent_mag = []
doc_sent_score = []
nf_count = 0
for pet in train.PetID.values:
    try:
        with open('../input/train_sentiment/' + pet + '.json', 'r',encoding="utf8") as f:
            sentiment = json.load(f)
        doc_sent_mag.append(sentiment['documentSentiment']['magnitude'])
        doc_sent_score.append(sentiment['documentSentiment']['score'])
    except FileNotFoundError:
        nf_count += 1
        doc_sent_mag.append(-1)
        doc_sent_score.append(-1)
train['doc_sent_mag'] = doc_sent_mag
train['doc_sent_score'] = doc_sent_score
doc_sent_mag = []
doc_sent_score = []
nf_count = 0
for pet in test.PetID.values:
    try:
        with open('../input/test_sentiment/' + pet + '.json', 'r',encoding="utf8") as f:
            sentiment = json.load(f)
        doc_sent_mag.append(sentiment['documentSentiment']['magnitude'])
        doc_sent_score.append(sentiment['documentSentiment']['score'])
    except FileNotFoundError:
        nf_count += 1
        doc_sent_mag.append(-1)
        doc_sent_score.append(-1)
test['doc_sent_mag'] = doc_sent_mag
test['doc_sent_score'] = doc_sent_score
lbl_enc = LabelEncoder()
lbl_enc.fit(train.RescuerID.values.tolist() + test.RescuerID.values.tolist())
train.RescuerID = lbl_enc.transform(train.RescuerID.values)
test.RescuerID = lbl_enc.transform(test.RescuerID.values)
train_desc = train.Description.fillna("none").values
test_desc = test.Description.fillna("none").values

tfv = TfidfVectorizer(min_df=3,  max_features=None,
        strip_accents='unicode', analyzer='word', token_pattern=r'\w{1,}',
        ngram_range=(1, 3), use_idf=1, smooth_idf=1, sublinear_tf=1,
        stop_words = 'english')
    
# Fit TFIDF
tfv.fit(list(train_desc) + list(test_desc))
X =  tfv.transform(train_desc)
X_test = tfv.transform(test_desc)


svd = TruncatedSVD(n_components=100)
svd.fit(X)
X = svd.transform(X)
X_test = svd.transform(X_test)
y = train.AdoptionSpeed
y.value_counts()
train = np.hstack((train.drop(['Name', 'Description', 'PetID', 'AdoptionSpeed'], axis=1).values, X))
test = np.hstack((test.drop(['Name', 'Description', 'PetID'], axis=1).values, X_test))

rand.seed(42)
idx_10Percent = rand.sample(list(range(1,train.shape[0]+1)),int(train.shape[0]/10))
train_10_percent = train[idx_10Percent,:]
y_10_percent     = y.values[idx_10Percent]

scaler = StandardScaler()
train_10_percent = scaler.fit_transform(train_10_percent)

classifiers = {
    'Nearest Neighbors' : KNeighborsClassifier(3),
    'Linear SVM'        :SVC(kernel="linear", C=0.025),
    'RBF SVM'           :SVC(gamma=2, C=1),
    'Gaussian Process'  :GaussianProcessClassifier(1.0 * RBF(1.0),optimizer=None, n_jobs = -1),
    'Decision Tree'     :DecisionTreeClassifier(max_depth=10),
    'Random Forest'     :RandomForestClassifier(max_depth=10),
    'Neural Net'        :MLPClassifier(alpha=1),
    'AdaBoost'          :AdaBoostClassifier(),
    'Naive Bayes'       :GaussianNB(),
    #'QDA'               :QuadraticDiscriminantAnalysis()
}


#ref: https://chrisalbon.com/machine_learning/model_evaluation/plot_the_learning_curve/

plt.figure(figsize=(40,30))
i=1


for key in classifiers:
    
    print("Plotting ",key)
    
    train_sizes, train_scores, test_scores = learning_curve(classifiers[key], 
                                                        train_10_percent, 
                                                        y_10_percent,
                                                        # Number of folds in cross-validation
                                                        cv=7,
                                                        # Evaluation metric
                                                        scoring='accuracy',
                                                        # Use all computer cores
                                                        n_jobs=-1, 
                                                        # 50 different sizes of the training set
                                                        train_sizes=np.linspace(0.10, 1.0, 50))

    # Create means and standard deviations of training set scores
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)

    # Create means and standard deviations of test set scores
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)

    plt.subplot(3,3,i)
    
    
    # Draw lines
    plt.plot(train_sizes, train_mean, '--', color="#111111",  label="Training score")
    plt.plot(train_sizes, test_mean, color="#111111", label="Cross-validation score")

    # Draw bands
    plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, color="#DDDDDD")
    plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, color="#DDDDDD")

    # Create plot
    plt.title("Learning Curve " + key)
    plt.xlabel("Training Set Size",fontsize=30), plt.ylabel("Accuracy Score",fontsize=30), plt.legend(loc="best",fontsize=24)
    plt.tight_layout()
    
    plt.tick_params(axis='both', which='major', labelsize=25)
    plt.tick_params(axis='both', which='minor', labelsize=25)
    
    i=i+1
    
    #plt.show()