# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.preprocessing import OneHotEncoder
import xgboost
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import confusion_matrix
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from scipy.cluster import hierarchy
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
import re
import matplotlib.pyplot as plt
import itertools
import json
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input/"))

plt.rcParams.update({'font.size': 14})

# Any results you write to the current directory are saved as output.
# The following 3 functions have been taken from Ben Hamner's github repository
# https://github.com/benhamner/Metrics
def conf_matrix(rater_a, rater_b, min_rating=None, max_rating=None):
    """
    Returns the confusion matrix between rater's ratings
    """
    assert(len(rater_a) == len(rater_b))
    if min_rating is None:
        min_rating = min(rater_a + rater_b)
    if max_rating is None:
        max_rating = max(rater_a + rater_b)
    num_ratings = int(max_rating - min_rating + 1)
    conf_mat = [[0 for i in range(num_ratings)]
                for j in range(num_ratings)]
    for a, b in zip(rater_a, rater_b):
        conf_mat[a - min_rating][b - min_rating] += 1
    return conf_mat


def histogram(ratings, min_rating=None, max_rating=None):
    """
    Returns the counts of each type of rating that a rater made
    """
    if min_rating is None:
        min_rating = min(ratings)
    if max_rating is None:
        max_rating = max(ratings)
    num_ratings = int(max_rating - min_rating + 1)
    hist_ratings = [0 for x in range(num_ratings)]
    for r in ratings:
        hist_ratings[r - min_rating] += 1
    return hist_ratings


def quadratic_weighted_kappa(y_pred,y):
    """
    !!!modified to be an XGBoost custom metric!!!
    
    Calculates the quadratic weighted kappa
    axquadratic_weighted_kappa calculates the quadratic weighted kappa
    value, which is a measure of inter-rater agreement between two raters
    that provide discrete numeric ratings.  Potential values range from -1
    (representing complete disagreement) to 1 (representing complete
    agreement).  A kappa value of 0 is expected if all agreement is due to
    chance.
    quadratic_weighted_kappa(rater_a, rater_b), where rater_a and rater_b
    each correspond to a list of integer ratings.  These lists must have the
    same length.
    The ratings should be integers, and it is assumed that they contain
    the complete range of possible ratings.
    quadratic_weighted_kappa(X, min_rating, max_rating), where min_rating
    is the minimum possible rating, and max_rating is the maximum possible
    rating
    """
    # convert predicted probabilities to hard predictions
    rater_a = np.argmax(y_pred,axis=1)
    # convert from XGB's DMatrix
    rater_b = y.get_label()
    min_rating=None
    max_rating=None
    rater_a = np.array(rater_a, dtype=int)
    rater_b = np.array(rater_b, dtype=int)
    assert(len(rater_a) == len(rater_b))
    if min_rating is None:
        min_rating = min(min(rater_a), min(rater_b))
    if max_rating is None:
        max_rating = max(max(rater_a), max(rater_b))
    conf_mat = conf_matrix(rater_a, rater_b,
                                min_rating, max_rating)
    num_ratings = len(conf_mat)
    num_scored_items = float(len(rater_a))

    hist_rater_a = histogram(rater_a, min_rating, max_rating)
    hist_rater_b = histogram(rater_b, min_rating, max_rating)

    numerator = 0.0
    denominator = 0.0

    for i in range(num_ratings):
        for j in range(num_ratings):
            expected_count = (hist_rater_a[i] * hist_rater_b[j]
                              / num_scored_items)
            d = pow(i - j, 2.0) / pow(num_ratings - 1, 2.0)
            numerator += d * conf_mat[i][j] / num_scored_items
            denominator += d * expected_count / num_scored_items
    return 'qwk', -1e0*(1.0 - numerator / denominator)

def qwk(y, y_pred):
    """
    Calculates the quadratic weighted kappa
    axquadratic_weighted_kappa calculates the quadratic weighted kappa
    value, which is a measure of inter-rater agreement between two raters
    that provide discrete numeric ratings.  Potential values range from -1
    (representing complete disagreement) to 1 (representing complete
    agreement).  A kappa value of 0 is expected if all agreement is due to
    chance.
    quadratic_weighted_kappa(rater_a, rater_b), where rater_a and rater_b
    each correspond to a list of integer ratings.  These lists must have the
    same length.
    The ratings should be integers, and it is assumed that they contain
    the complete range of possible ratings.
    quadratic_weighted_kappa(X, min_rating, max_rating), where min_rating
    is the minimum possible rating, and max_rating is the maximum possible
    rating
    """
    rater_a = y
    rater_b = y_pred
    min_rating=None
    max_rating=None
    rater_a = np.array(rater_a, dtype=int)
    rater_b = np.array(rater_b, dtype=int)
    assert(len(rater_a) == len(rater_b))
    if min_rating is None:
        min_rating = min(min(rater_a), min(rater_b))
    if max_rating is None:
        max_rating = max(max(rater_a), max(rater_b))
    conf_mat = conf_matrix(rater_a, rater_b,
                                min_rating, max_rating)
    num_ratings = len(conf_mat)
    num_scored_items = float(len(rater_a))

    hist_rater_a = histogram(rater_a, min_rating, max_rating)
    hist_rater_b = histogram(rater_b, min_rating, max_rating)

    numerator = 0.0
    denominator = 0.0

    for i in range(num_ratings):
        for j in range(num_ratings):
            expected_count = (hist_rater_a[i] * hist_rater_b[j]
                              / num_scored_items)
            d = pow(i - j, 2.0) / pow(num_ratings - 1, 2.0)
            numerator += d * conf_mat[i][j] / num_scored_items
            denominator += d * expected_count / num_scored_items

    return (1.0 - numerator / denominator)
# an animal class to collect and organize the data. It takes the animals ID as input and
# whether the animal is in train or test. It returns labels (.get_label() method) and 
# features (.tabular_features() method).

df_breed = pd.read_csv('../input/breed_labels.csv')
breed_names = list(df_breed['BreedName'])
breed_IDs = list(df_breed['BreedID'])

df_color = pd.read_csv('../input/color_labels.csv')
color_names = list(df_color['ColorName'])
color_IDs = list(df_color['ColorID'])

df_state = pd.read_csv('../input/state_labels.csv')
state_names = list(df_state['StateName'])
state_IDs = list(df_state['StateID'])

class animal(object):
    
    def __init__(self,ID,in_train=True):
        self.ID = ID
        self.in_train = in_train
    
    def get_label(self):
        if self.in_train:
            data = self.collect_data()
            return data['AdoptionSpeed'].values[0]
        else:
            print('test data without labels')
            raise ValueError
            
    def collect_data(self):
        if self.in_train:
            data = df_train.loc[df_train['PetID'] == self.ID]
        else:
            data = df_test.loc[df_test['PetID'] == self.ID]
        return data
    
    def tabular_features(self):
        # collect features
        data = self.collect_data()
        
        features = []
        feature_names = []
        
        # features that either have only two categories (plus missing) or 
        # the values are ranked, so it's ok not to one-hot encode
        features.append(data["Type"].values[0])
        feature_names.append('Type')

        if pd.isnull(data['Name'].values[0]):
            features.append(0)
        else:
            features.append(1)
        feature_names.append('Has name')
        
        features.append(data['Age'].values[0])
        feature_names.append('Age')
        
        if data['MaturitySize'].values[0] == 0:
            features.append(np.nan)
        else:
            features.append(data['MaturitySize'].values[0])
        feature_names.append('MaturitySize')

        if data['FurLength'].values[0] == 0:
            features.append(np.nan)
        else: 
            features.append(data['FurLength'].values[0])
        feature_names.append('FurLength')
        
        if data['Vaccinated'].values[0] == 3:
            features.append(np.nan)
        else: 
            features.append(data['Vaccinated'].values[0])
        feature_names.append('Vaccinated')
        
        if data['Dewormed'].values[0] == 3:
            features.append(np.nan)
        else: 
            features.append(data['Dewormed'].values[0])
        feature_names.append('Dewormed')
        
        if data['Sterilized'].values[0] == 3:
            features.append(np.nan)
        else: 
            features.append(data['Sterilized'].values[0])
        feature_names.append('Sterilized')

        if data['Health'].values[0] == 0:
            features.append(np.nan)
        else: 
            features.append(data['Health'].values[0])
        feature_names.append('Health')
        
        features.append(data['Quantity'].values[0])
        feature_names.append('Quantity')
        
        features.append(data['Fee'].values[0])
        feature_names.append('Fee')
        
        features.append(data['VideoAmt'].values[0])
        feature_names.append('VideoAmt')
        
        features.append(data['PhotoAmt'].values[0])
        feature_names.append('PhotoAmt')
        
        # features that need to be one-hot encoded
        # gender, breed, color, state
        if data['Gender'].values[0] == 1:
            features.append(1)
        else:
            features.append(0)
        feature_names.append('Male')
        if data['Gender'].values[0] == 2:
            features.append(1)
        else:
            features.append(0)
        feature_names.append('Female')
        if data['Gender'].values[0] == 3:
            features.append(1)
        else:
            features.append(0)
        feature_names.append('Multiple')
        
        breed = np.zeros(len(breed_IDs))
        if data['Breed1'].values[0] > 0:
            breed[breed_IDs.index(data['Breed1'].values[0])] = 1
        if data['Breed2'].values[0] > 0:
            breed[breed_IDs.index(data['Breed2'].values[0])] = 1
        features.extend(breed)
        feature_names.extend(breed_names)
        
        color = np.zeros(len(color_IDs))
        if data['Color1'].values[0] > 0:
            color[color_IDs.index(data['Color1'].values[0])] = 1
        if data['Color2'].values[0] > 0:
            color[color_IDs.index(data['Color2'].values[0])] = 1
        if data['Color3'].values[0] > 0:
            color[color_IDs.index(data['Color3'].values[0])] = 1
        features.extend(color)
        feature_names.extend(color_names)
        
        state = np.zeros(len(state_IDs))
        state[state_IDs.index(data['State'].values[0])] = 1
        features.extend(state)
        feature_names.extend(state_names)
        
        # features related to the description
        desc = self.description()
        
        if desc == 'no_description':
            features.append(0)
        else:
            features.append(len(desc))
        feature_names.append('NrCharsInDesc')

        if desc == 'no_description':
            features.append(0)
        else:
            features.append(len(desc.split()))
        feature_names.append('NrWordsInDesc')
        
        # add sentiment features
        sent_features, sent_ftr_names = self.sentiment()
        features.extend(sent_features)
        feature_names.extend(sent_ftr_names)
        
        if len(features) != len(feature_names):
            print('the number of features must be the same as the number of feature names')
            raise ValueError
        
        return features,feature_names
    
    def description(self):
        data = self.collect_data()
        if isinstance(data['Description'].values[0], str):
            return data['Description'].values[0]
        else:
            return 'no_description'
    
    def sentiment(self):
        try:
            if self.in_train:
                with open('../input/train_sentiment/'+str(self.ID)+'.json') as f:
                    data = json.load(f)
            else:
                with open('../input/test_sentiment/'+str(self.ID)+'.json') as f:
                    data = json.load(f)
            sentence_level = []
            for i in data['sentences']:
                sentence_level.append([i['sentiment']['score'],i['sentiment']['magnitude']])
            sentence_level = np.array(sentence_level).T
            features = []
            feature_names = []
            features.append(data['documentSentiment']['score'])
            feature_names.append('overall score')
            features.append(data['documentSentiment']['magnitude'])
            feature_names.append('overall magnitude')
            features.append(np.min(sentence_level[0]))
            feature_names.append('min sentence score')
            features.append(np.max(sentence_level[0]))
            feature_names.append('max sentence score')
            features.append(np.std(sentence_level[0]))
            feature_names.append('stdev sentence score')
            features.append(np.min(sentence_level[1]))
            feature_names.append('min sentence magnitude')
            features.append(np.max(sentence_level[1]))
            feature_names.append('max sentence magnitude')
            features.append(np.std(sentence_level[1]))
            feature_names.append('stdev sentence magnitude')
        except FileNotFoundError:
            features = []
            feature_names = []
            features.append(-1)
            feature_names.append('overall score')
            features.append(-1)
            feature_names.append('overall magnitude')
            features.append(-1)
            feature_names.append('min sentence score')
            features.append(-1)
            feature_names.append('max sentence score')
            features.append(-1)
            feature_names.append('stdev sentence score')
            features.append(-1)
            feature_names.append('min sentence magnitude')
            features.append(-1)
            feature_names.append('max sentence magnitude')
            features.append(-1)
            feature_names.append('stdev sentence magnitude')
                    
        return features,feature_names
        
# collect features and labels
df_train = pd.read_csv('../input/train/train.csv')
df_test = pd.read_csv('../input/test/test.csv')
train_IDs = list(df_train['PetID'])
test_IDs = list(df_test['PetID'])

__, feature_names = animal(train_IDs[0]).tabular_features()
feature_names_tab = np.array(feature_names)

class_names = ['same day','1st week','1st month','2nd and 3rd m.','>100 days']

Y = np.zeros(len(train_IDs))
X_tab = np.zeros([len(train_IDs),len(feature_names_tab)])
# collect training data of tabular features
for i in range(len(train_IDs)):
    ftrs, __ = animal(train_IDs[i]).tabular_features()
    X_tab[i] = ftrs
    Y[i] = animal(train_IDs[i]).get_label()
print(np.shape(X_tab))
print(np.unique(Y,return_counts=True))

# collect test data
X_test_tab = np.zeros([len(test_IDs),len(feature_names_tab)])
for i in range(len(test_IDs)):
    ftrs, __ = animal(test_IDs[i],in_train=False).tabular_features()
    X_test_tab[i] = ftrs

print(np.shape(X_test_tab))
# n-gram features actually reduced the LB score from 0.33 to 0.28-0.3 I'm not sure why.
# If you spot a bug in the cell, please let me know.

# # collect the descriptions, n-grams will be calculated in the ML pipeline
# def stemming_tokenizer(str_input):
#     words = re.sub(r"[^A-Za-z]", " ", str_input).lower().split()
#     words = [PorterStemmer().stem(word) for word in words]
#     return words

# corpus_train = []
# for i in range(len(train_IDs)):
#     corpus_train.append(animal(train_IDs[i]).description())
# corpus_train = np.array(corpus_train)
# print(len(corpus_train))

# corpus_test = []
# for i in range(len(test_IDs)):
#     corpus_test.append(animal(test_IDs[i],in_train=False).description())
# corpus_test = np.array(corpus_test)
# print(len(corpus_test))

# # 1- to 3-grams are collected and only those ngrams are kept that appear in at least 100 documents
# vectorizer = CountVectorizer(ngram_range=(1,3),min_df=500,tokenizer=stemming_tokenizer)
# X_text = vectorizer.fit_transform(corpus_train)
# feature_names_text = vectorizer.get_feature_names()
# X_test_text = vectorizer.transform(corpus_test)

# print(np.shape(X_text))
# print(np.shape(X_test_text))
# print(len(feature_names_text))
# print(feature_names_text)
# # concatenate the arrays
# X = np.concatenate((X_tab,X_text.toarray()),axis=1)
# X_test = np.concatenate((X_test_tab,X_test_text.toarray()),axis=1)
# feature_names = np.concatenate((feature_names_tab,feature_names_text))

X = np.copy(X_tab)
X_test = np.copy(X_test_tab)
feature_names = np.copy(feature_names_tab)

print(np.shape(X))
print(np.shape(X_test))
print(len(feature_names))

# compare the train and test distributions
# do some feature selection and remove uninformative features
# if you want info and plots of each figure printed, set detailed_output to True 
detailed_output = False
to_keep = []
for i in range(len(feature_names)):
    if len(np.unique(X[:,i])) < 3:
        # categorical features, no figure
        if detailed_output:
            print(feature_names[i])
        values_train, dist_train = np.unique(X[:,i],return_counts=True)
        values_test, dist_test = np.unique(X_test[:,i],return_counts=True)
        # normalize the counts
        dist_train = 1e0*dist_train/np.sum(dist_train)
        dist_test = 1e0*dist_test/np.sum(dist_test)
        unique_values = np.unique(np.concatenate((values_train,values_test)))
        for v in unique_values:
            indx1 = np.where(values_train == v)[0]
            indx2 = np.where(values_test == v)[0]
            if detailed_output:
                if (len(indx1) == 1) and (len(indx2) == 1):
                    print('   ',v,np.around(dist_train[indx1[0]],4),np.around(dist_test[indx2[0]],4))
                else:
                    if len(indx1) == 0:
                        print('   ',v,0e0,np.around(dist_test[indx2[0]],4))                    
                    elif len(indx2) == 0:
                        print('   ',v,np.around(dist_train[indx1[0]],4),0e0)
        if (np.min(np.concatenate((dist_train,dist_test))) > 0.01)&\
           (np.min(np.concatenate((dist_train,dist_test))) < 1e0):
            to_keep.append(i)
    else:
        to_keep.append(i)
        if detailed_output:
            # categorical features with more than two categories and continuous features
            # prepare a figure to compare distributions
            if len(np.unique(X[:,i])) > 5:
                bins = 100
            else:
                bins = np.arange(np.min(np.unique(X[:,i])),len(np.unique(X[:,i]))+1)-0.5
            n, bins, patches = plt.hist(X[:,i],bins = bins,alpha=0.5,color='b',density=True,label='train')
            plt.hist(X_test[:,i],bins = bins,alpha=0.5,color='r',density=True,label='test')
            if feature_names[i] == 'Fee':
                plt.xlim([0,500])
            if feature_names[i] == 'Age':
                plt.xlim([0,50])
            plt.xlabel('feature value')
            plt.ylabel('pdf')
            plt.legend()
            plt.title(feature_names[i])
            plt.show()

        
X = X[:,to_keep]
X_test = X_test[:,to_keep]
feature_names = feature_names[to_keep]
print(np.shape(X))
print(np.shape(X_test))
print(len(feature_names))
# the ML pipeline

# number of runs to average. a higher number reduces the internal randomness of the model
nr_runs = 3
# number of folds
nr_folds = 5

def train_pred(X_tab,Y,X_test_tab):
    rskf = RepeatedStratifiedKFold(n_splits=nr_folds, n_repeats=nr_runs,
        random_state=0)
    SHAP = np.zeros([len(Y),5,len(feature_names)+1])
    SHAP_test = np.zeros([np.shape(X_test)[0],5,len(feature_names)+1])
    # the SHAP interaction features give a memory error in the kernel :(
    #SHAP_int = np.zeros([len(Y),5,len(feature_names)+1,len(feature_names)+1])
    Y_test = np.zeros([np.shape(X_test_tab)[0],5])
    Y_CV_pred = np.zeros([np.shape(X_tab)[0],5])
    models = []
    i = 0
    for train_index, CV_index in rskf.split(X_tab, Y):
        print(i)
        xgb = xgboost.XGBClassifier(learning_rate=0.1,subsample=0.66, colsample_bytree=0.9,\
                                    random_state=0,objective='multi:softprob',num_class=5,\
                                    n_estimators=10000)
        X_train, X_CV = X[train_index], X[CV_index]
        Y_train, Y_CV = Y[train_index], Y[CV_index]
        eval_set = [(X_CV, Y_CV)]
        xgb.fit(X_train,Y_train,early_stopping_rounds=50,eval_set = eval_set,\
                verbose=False,eval_metric=quadratic_weighted_kappa)
        # collect shap values and predictions for the CV set
        Y_CV_pred[CV_index] = Y_CV_pred[CV_index] + xgb.predict_proba(X_CV,ntree_limit=xgb.best_ntree_limit)
        SHAP[CV_index] = SHAP[CV_index] + xgb.get_booster().predict(xgboost.DMatrix(X_CV),pred_contribs=True,ntree_limit=xgb.best_ntree_limit)
        #SHAP_int[CV_index] = SHAP_int[CV_index] + xgb.get_booster().predict(xgboost.DMatrix(X_CV),pred_interactions=True,ntree_limit=xgb.best_ntree_limit)
        # predict test
        Y_test = Y_test + xgb.predict_proba(X_test,ntree_limit=xgb.best_ntree_limit)
        SHAP_test = SHAP_test + xgb.get_booster().predict(xgboost.DMatrix(X_test),pred_contribs=True,ntree_limit=xgb.best_ntree_limit)
        # save the model
        models.append(xgb)
        i = i + 1
        
    Y_test = Y_test / (nr_runs*nr_folds)
    SHAP_test = SHAP_test / (nr_runs*nr_folds)
    Y_CV_pred = Y_CV_pred / nr_runs
    SHAP = SHAP / (nr_runs)

    return Y_test,SHAP_test,Y_CV_pred,SHAP,models
    
    
Y_test, SHAP_test, Y_CV_pred, SHAP, models = train_pred(X_tab,Y,X_test_tab)    

Y_CV = np.argmax(Y_CV_pred,axis=1)
print('local CV score: ',qwk(Y,Y_CV))

# prepare the submission file
submission = pd.DataFrame({'PetID': test_IDs, 'AdoptionSpeed': np.argmax(Y_test,axis=1)})
print(submission.head())
submission.to_csv('submission.csv', index=False)
# confusion matrix
# https://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        #print("Normalized confusion matrix")
    #else:
        #print('Confusion matrix, without normalization')
    #print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap,vmin=0.0)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()

Y_CV = np.argmax(Y_CV_pred,axis=1)
    
# Compute confusion matrix
cnf_matrix = confusion_matrix(Y, Y_CV)
np.set_printoptions(precision=2)

# Plot confusion matrix
plt.figure(figsize=(8,6))
plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=False,
                      title='Confusion matrix')
plt.show()

# Plot normalized confusion matrix
plt.figure(figsize=(8,6))
plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
                      title='Normalize confusion matrix')
plt.show()
# study the SHAP values and extract actionable insights
def plot_SHAP(indcs = 'all',title=''):
    if indcs == 'all':
        indcs = np.arange(len(Y))
        
    # global feature importances
    sorted_indcs = np.argsort(np.mean(np.abs(SHAP_final[indcs,:-1]),axis=0))[::-1]

    print(feature_names[sorted_indcs[:10]])
    
    plt.figure(figsize=(8,6))

    for i in range(10):
        plt.scatter(SHAP_final[indcs,sorted_indcs[i]],np.zeros(len(Y[indcs]))+(10-1-i)+np.random.normal(0e0,0.08,size=len(Y[indcs])),\
                    s=10,c=np.ravel(X[indcs,sorted_indcs[i]]),cmap='plasma')
    plt.axvline(0e0,linestyle='dotted',color='k')
    plt.yticks(range(10)[::-1],feature_names[sorted_indcs[:10]])
    plt.xlabel("feature's contribution")
    plt.colorbar(ticks=[],label='feature value (from low to high)')
    plt.title(title)
    plt.tight_layout()
    plt.show()
    
    return

def plot_ftr_vs_shap(ftr_name,class_indx='all'):
    indx_ftr = np.where(feature_names == ftr_name)[0][0]
    if class_indx=='all':
        indcs = np.arange(len(Y_CV))
    else:
        indcs = np.where(Y_CV == class_indx)[0]
    plt.scatter(np.ravel(X[indcs,indx_ftr]),SHAP_final[indcs,indx_ftr])
    plt.xlabel(ftr_name)
    plt.ylabel('SHAP value')
    plt.show()
    return

# collect the shap values of the most likely class
SHAP_final = np.zeros([len(Y),len(feature_names)+1])
for i in range(len(Y)):
    SHAP_final[i] = SHAP[i,Y_CV[i]]

plot_SHAP(title='all')

plot_SHAP(indcs = np.where(Y_CV == 4)[0],title='predicted class: '+class_names[4])

# closer look at quantity, photoamt, and fee vs. shap value
plot_ftr_vs_shap('Quantity',4)
plot_ftr_vs_shap('Sterilized',4)
plot_ftr_vs_shap('PhotoAmt',4)
plot_ftr_vs_shap('NrCharsInDesc',4)
plot_ftr_vs_shap('Fee',4)
indx = np.argmax(Y_test[:,4])
print('animal ID:',test_IDs[indx])
print('class probabilities: ',Y_test[indx])
ftrs, __ = animal(test_IDs[indx],in_train=False).tabular_features()
ftrs = np.array(ftrs)
sorted_ftrs = np.argsort(np.abs(SHAP_test[indx,4,:-1]))[::-1]
for f in sorted_ftrs[:5]:
    print(feature_names[f],':',ftrs[f], ", feature's contribution to the prediction: ",np.around(SHAP_test[indx,4,f],2))

ftrs_mod = np.copy(ftrs)
prob_class4 = np.zeros(31)
for i in range(0,31):
    ftrs_mod[sorted_ftrs[0]] = i
    probs = np.zeros(5)
    for xgb in models:
        probs = probs + xgb.predict_proba(ftrs_mod[np.newaxis,:],validate_features=False)[0]
    probs = probs / (nr_runs*nr_folds)
    prob_class4[i] = probs[4]
print(probs)

prob_desc_class4 = np.zeros(31)
for i in range(0,31):
    ftrs_mod[sorted_ftrs[0]] = i
    ftrs_mod[sorted_ftrs[3]] = 1000
    probs = np.zeros(5)
    for xgb in models:
        probs = probs + xgb.predict_proba(ftrs_mod[np.newaxis,:],validate_features=False)[0]
    probs = probs / (nr_runs*nr_folds)
    prob_desc_class4[i] = probs[4]
print(probs)

plt.plot(range(0,31),prob_class4,label='short description')
plt.plot(range(0,31),prob_desc_class4,label='long description')
plt.xlabel('PhotoAmt')
plt.ylabel("probability, '>100 days'")
plt.title('Sterilization status unknown')
plt.legend()
plt.ylim([0.25,0.85])
plt.grid()
plt.tight_layout()
plt.show()

ftrs_mod = np.copy(ftrs)
prob_class4 = np.zeros(31)
for i in range(0,31):
    ftrs_mod[sorted_ftrs[4]] = 1    
    ftrs_mod[sorted_ftrs[0]] = i
    probs = np.zeros(5)
    for xgb in models:
        probs = probs + xgb.predict_proba(ftrs_mod[np.newaxis,:],validate_features=False)[0]
    probs = probs / (nr_runs*nr_folds)
    prob_class4[i] = probs[4]
print(probs)
prob_desc_class4 = np.zeros(31)
for i in range(0,31):
    ftrs_mod[sorted_ftrs[4]] = 1    
    ftrs_mod[sorted_ftrs[0]] = i
    ftrs_mod[sorted_ftrs[3]] = 1000
    probs = np.zeros(5)
    for xgb in models:
        probs = probs + xgb.predict_proba(ftrs_mod[np.newaxis,:],validate_features=False)[0]
    probs = probs / (nr_runs*nr_folds)
    prob_desc_class4[i] = probs[4]
print(probs)

plt.plot(range(0,31),prob_class4,label='short description')
plt.plot(range(0,31),prob_desc_class4,label='long description')
plt.xlabel('PhotoAmt')
plt.ylabel("probability, '>100 days'")
plt.title('Sterilized')
plt.legend()
plt.ylim([0.25,0.85])
plt.grid()
plt.tight_layout()
plt.show()


ftrs_mod = np.copy(ftrs)
prob_class4 = np.zeros(31)
for i in range(0,31):
    ftrs_mod[sorted_ftrs[0]] = i
    ftrs_mod[sorted_ftrs[4]] = 2    
    probs = np.zeros(5)
    for xgb in models:
        probs = probs + xgb.predict_proba(ftrs_mod[np.newaxis,:],validate_features=False)[0]
    probs = probs / (nr_runs*nr_folds)
    prob_class4[i] = probs[4]
print(probs)
prob_desc_class4 = np.zeros(31)
for i in range(0,31):
    ftrs_mod[sorted_ftrs[4]] = 2    
    ftrs_mod[sorted_ftrs[0]] = i
    ftrs_mod[sorted_ftrs[3]] = 1000
    probs = np.zeros(5)
    for xgb in models:
        probs = probs + xgb.predict_proba(ftrs_mod[np.newaxis,:],validate_features=False)[0]
    probs = probs / (nr_runs*nr_folds)
    prob_desc_class4[i] = probs[4]
print(probs)

plt.plot(range(0,31),prob_class4,label='short description')
plt.plot(range(0,31),prob_desc_class4,label='long description')
plt.xlabel('PhotoAmt')
plt.ylabel("probability, '>100 days'")
plt.title('Not sterilized')
plt.legend()
plt.ylim([0.25,0.85])
plt.grid()
plt.tight_layout()
plt.show()


