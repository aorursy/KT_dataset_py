# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
from sklearn import model_selection, preprocessing, linear_model, naive_bayes, metrics, svm

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

from sklearn import decomposition, ensemble



import pandas, xgboost, numpy, textblob, string

from keras.preprocessing import text, sequence

from keras import layers, models, optimizers
import warnings

warnings.filterwarnings('ignore')
import pandas as pd

import seaborn as sns

trainDF = pd.read_csv('../input/train_lyrics_1000.csv', usecols=range(7))

validDF =pd.read_csv('../input/valid_lyrics_200.csv')

print("Validation data with lyrics and labels",validDF.head())

#print("Trainning Data with lyrics,title,singer,genre and mood label")

# trainDF['lyrics'] = lyrics

# trainDF['mood'] = mood

trainDF.head()
train_x =trainDF['lyrics']

train_y =trainDF['mood']

valid_x =validDF['lyrics']

valid_y =validDF['mood']
# label encode the target variable 

encoder = preprocessing.LabelEncoder()

train_y = encoder.fit_transform(train_y)

valid_y = encoder.fit_transform(valid_y)
import nltk

import string

import re



porter_stemmer = nltk.stem.porter.PorterStemmer()



def porter_tokenizer(text, stemmer=porter_stemmer):

    """

    A Porter-Stemmer-Tokenizer hybrid to splits sentences into words (tokens) 

    and applies the porter stemming algorithm to each of the obtained token. 

    Tokens that are only consisting of punctuation characters are removed as well.

    Only tokens that consist of more than one letter are being kept.

    

    Parameters

    ----------

        

    text : `str`. 

      A sentence that is to split into words.

        

    Returns

    ----------

    

    no_punct : `str`. 

      A list of tokens after stemming and removing Sentence punctuation patterns.

    

    """

    lower_txt = text.lower()

    tokens = nltk.wordpunct_tokenize(lower_txt)

    stems = [porter_stemmer.stem(t) for t in tokens]

    no_punct = [s for s in stems if re.match('^[a-zA-Z]+$', s) is not None]

    return no_punct

# Commented out to prevent overwriting files:

#

# stp = nltk.corpus.stopwords.words('english')

# with open('./stopwords_eng.txt', 'w') as outfile:

#    outfile.write('\n'.join(stp))

    

    

with open('../input/stopwords_eng.txt', 'r') as infile:

    stop_words = infile.read().splitlines()

print('stop words %s ...' %stop_words[:5])
# create a count vectorizer object 

#analyzer='word',

                      

                      

                       

                     

count_vect = CountVectorizer(analyzer='word',preprocessor=lambda text: re.sub('[^a-zA-Z]', ' ', text.lower()),stop_words=stop_words,

            tokenizer=porter_tokenizer,ngram_range=(1,1))

count_vect.fit(trainDF['lyrics'])



# transform the training and validation data using count vectorizer object

xtrain_count =  count_vect.transform(train_x)

xvalid_count =  count_vect.transform(valid_x)
# word level tf-idf

tfidf_vect = TfidfVectorizer(analyzer='word', preprocessor=lambda text: re.sub('[^a-zA-Z]', ' ', text.lower()), max_features=5000)

tfidf_vect.fit(trainDF['lyrics'])

xtrain_tfidf =  tfidf_vect.transform(train_x)

xvalid_tfidf =  tfidf_vect.transform(valid_x)



# ngram level tf-idf 

tfidf_vect_ngram = TfidfVectorizer(analyzer='word', preprocessor=lambda text: re.sub('[^a-zA-Z]', ' ', text.lower()), ngram_range=(2,2), max_features=5000)

tfidf_vect_ngram.fit(trainDF['lyrics'])

xtrain_tfidf_ngram =  tfidf_vect_ngram.transform(train_x)

xvalid_tfidf_ngram =  tfidf_vect_ngram.transform(valid_x)



# characters level tf-idf

tfidf_vect_ngram_chars = TfidfVectorizer(analyzer='char', preprocessor=lambda text: re.sub('[^a-zA-Z]', ' ', text.lower()), ngram_range=(1,1), max_features=5000)

tfidf_vect_ngram_chars.fit(trainDF['lyrics'])

xtrain_tfidf_ngram_chars =  tfidf_vect_ngram_chars.transform(train_x) 

xvalid_tfidf_ngram_chars =  tfidf_vect_ngram_chars.transform(valid_x) 
#sklearn

from sklearn.model_selection import train_test_split, cross_val_score

from sklearn.metrics import mean_squared_error,confusion_matrix, precision_score, recall_score, auc,roc_curve,f1_score

from sklearn import ensemble, linear_model, neighbors, svm, tree, neural_network

from sklearn.pipeline import make_pipeline

from sklearn.linear_model import Ridge

from sklearn.preprocessing import PolynomialFeatures

from sklearn import svm,model_selection, tree, linear_model, neighbors, naive_bayes, ensemble, discriminant_analysis, gaussian_process



#load package

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

#from math import sqrt

import seaborn as sns



import warnings

warnings.filterwarnings('ignore')
MLA = [

    #Ensemble Methods

    ensemble.AdaBoostClassifier(),

    ensemble.BaggingClassifier(),

    ensemble.ExtraTreesClassifier(),

    ensemble.GradientBoostingClassifier(),

    ensemble.RandomForestClassifier(),



    #Gaussian Processes

    gaussian_process.GaussianProcessClassifier(),

    

    #GLM

    linear_model.LogisticRegressionCV(),

    linear_model.PassiveAggressiveClassifier(),

    linear_model. RidgeClassifierCV(),

    linear_model.SGDClassifier(),

    linear_model.Perceptron(),

    

    #Navies Bayes

    naive_bayes.BernoulliNB(),

    naive_bayes.GaussianNB(),

    

    #Nearest Neighbor

    neighbors.KNeighborsClassifier(),

    

    #SVM

    svm.SVC(probability=True),

    svm.NuSVC(probability=True),

    svm.LinearSVC(),

    

    #Trees    

    tree.DecisionTreeClassifier(),

   #tree.ExtraTreeClassifier(),

    

    ]
MLA_columns = []

MLA_compare = pd.DataFrame(columns = MLA_columns)





row_index = 0

for alg in MLA:

    

    

    predicted = alg.fit(xtrain_tfidf.toarray(), train_y).predict(xvalid_tfidf.toarray())

    fp, tp, th = roc_curve(valid_y, predicted)

    MLA_name = alg.__class__.__name__

    MLA_compare.loc[row_index,'MLA Name'] = MLA_name

    MLA_compare.loc[row_index, 'MLA Train Accuracy'] = round(alg.score(xtrain_tfidf.toarray(), train_y), 4)

    MLA_compare.loc[row_index, 'MLA Test Accuracy'] = round(alg.score(xvalid_tfidf.toarray(), valid_y), 4)

    MLA_compare.loc[row_index, 'MLA Precission'] = precision_score(valid_y, predicted)

    MLA_compare.loc[row_index, 'MLA Recall'] = recall_score(valid_y, predicted)

    MLA_compare.loc[row_index, 'MLA AUC'] = auc(fp, tp)

    MLA_compare.loc[row_index, 'MLA F1 Score'] = f1_score(valid_y, predicted)











    row_index+=1

    

MLA_compare.sort_values(by = ['MLA Test Accuracy'], ascending = False, inplace = True)    

MLA_compare
index = 1

for alg in MLA:

    

    

    predicted = alg.fit(xtrain_tfidf.toarray(), train_y).predict(xvalid_tfidf.toarray())

    fp, tp, th = roc_curve(valid_y, predicted)

    roc_auc_mla = auc(fp, tp)

    MLA_name = alg.__class__.__name__

    plt.plot(fp, tp, lw=2, alpha=0.3, label='ROC %s (AUC = %0.2f)'  % (MLA_name, roc_auc_mla))

   

    index+=1



plt.title('ROC Curve comparison')

plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

plt.plot([0,1],[0,1],'r--')

plt.xlim([0,1])

plt.ylim([0,1])

plt.ylabel('True Positive Rate')

plt.xlabel('False Positive Rate')    

plt.show()
MLA_columns = []

MLA_compare = pd.DataFrame(columns = MLA_columns)





row_index = 0

for alg in MLA:

    

    

    predicted = alg.fit(xtrain_count.toarray(), train_y).predict(xvalid_count.toarray())

    fp, tp, th = roc_curve(valid_y, predicted)

    MLA_name = alg.__class__.__name__

    MLA_compare.loc[row_index,'MLA Name'] = MLA_name

    MLA_compare.loc[row_index, 'MLA Train Accuracy'] = round(alg.score(xtrain_count.toarray(), train_y), 4)

    MLA_compare.loc[row_index, 'MLA Test Accuracy'] = round(alg.score(xvalid_count.toarray(), valid_y), 4)

    MLA_compare.loc[row_index, 'MLA Precission'] = precision_score(valid_y, predicted)

    MLA_compare.loc[row_index, 'MLA Recall'] = recall_score(valid_y, predicted)

    MLA_compare.loc[row_index, 'MLA AUC'] = auc(fp, tp)

    MLA_compare.loc[row_index, 'MLA F1 Score'] = f1_score(valid_y, predicted)











    row_index+=1

    

MLA_compare.sort_values(by = ['MLA Test Accuracy'], ascending = False, inplace = True)    

MLA_compare
index = 1

for alg in MLA:

    

    

    predicted = alg.fit(xtrain_count.toarray(), train_y).predict(xvalid_count.toarray())

    fp, tp, th = roc_curve(valid_y, predicted)

    roc_auc_mla = auc(fp, tp)

    MLA_name = alg.__class__.__name__

    plt.plot(fp, tp, lw=2, alpha=0.3, label='ROC %s (AUC = %0.2f)'  % (MLA_name, roc_auc_mla))

   

    index+=1



plt.title('ROC Curve comparison')

plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

plt.plot([0,1],[0,1],'r--')

plt.xlim([0,1])

plt.ylim([0,1])

plt.ylabel('True Positive Rate')

plt.xlabel('False Positive Rate')    

plt.show()
MLA_columns = []

MLA_compare = pd.DataFrame(columns = MLA_columns)





row_index = 0

for alg in MLA:

    

    

    predicted = alg.fit(xtrain_tfidf_ngram.toarray(), train_y).predict(xvalid_tfidf_ngram.toarray())

    fp, tp, th = roc_curve(valid_y, predicted)

    MLA_name = alg.__class__.__name__

    MLA_compare.loc[row_index,'MLA Name'] = MLA_name

    MLA_compare.loc[row_index, 'MLA Train Accuracy'] = round(alg.score(xtrain_tfidf_ngram.toarray(), train_y), 4)

    MLA_compare.loc[row_index, 'MLA Test Accuracy'] = round(alg.score(xvalid_tfidf_ngram.toarray(), valid_y), 4)

    MLA_compare.loc[row_index, 'MLA Precission'] = precision_score(valid_y, predicted)

    MLA_compare.loc[row_index, 'MLA Recall'] = recall_score(valid_y, predicted)

    MLA_compare.loc[row_index, 'MLA AUC'] = auc(fp, tp)

    MLA_compare.loc[row_index, 'MLA F1 Score'] = f1_score(valid_y, predicted)











    row_index+=1

    

MLA_compare.sort_values(by = ['MLA Test Accuracy'], ascending = False, inplace = True)    

MLA_compare
index = 1

for alg in MLA:

    

    

    predicted = alg.fit(xtrain_tfidf_ngram.toarray(), train_y).predict(xvalid_tfidf_ngram.toarray())

    fp, tp, th = roc_curve(valid_y, predicted)

    roc_auc_mla = auc(fp, tp)

    MLA_name = alg.__class__.__name__

    plt.plot(fp, tp, lw=2, alpha=0.3, label='ROC %s (AUC = %0.2f)'  % (MLA_name, roc_auc_mla))

   

    index+=1



plt.title('ROC Curve comparison')

plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

plt.plot([0,1],[0,1],'r--')

plt.xlim([0,1])

plt.ylim([0,1])

plt.ylabel('True Positive Rate')

plt.xlabel('False Positive Rate')    

plt.show()
MLA_columns = []

MLA_compare = pd.DataFrame(columns = MLA_columns)





row_index = 0

for alg in MLA:

    

    

    predicted = alg.fit(xtrain_tfidf_ngram_chars.toarray(), train_y).predict(xvalid_tfidf_ngram_chars.toarray())

    fp, tp, th = roc_curve(valid_y, predicted)

    MLA_name = alg.__class__.__name__

    MLA_compare.loc[row_index,'MLA Name'] = MLA_name

    MLA_compare.loc[row_index, 'MLA Train Accuracy'] = round(alg.score(xtrain_tfidf_ngram_chars.toarray(), train_y), 4)

    MLA_compare.loc[row_index, 'MLA Test Accuracy'] = round(alg.score(xvalid_tfidf_ngram_chars.toarray(), valid_y), 4)

    MLA_compare.loc[row_index, 'MLA Precission'] = precision_score(valid_y, predicted)

    MLA_compare.loc[row_index, 'MLA Recall'] = recall_score(valid_y, predicted)

    MLA_compare.loc[row_index, 'MLA AUC'] = auc(fp, tp)

    MLA_compare.loc[row_index, 'MLA F1 Score'] = f1_score(valid_y, predicted)











    row_index+=1

    

MLA_compare.sort_values(by = ['MLA Test Accuracy'], ascending = False, inplace = True)    

MLA_compare
index = 1

for alg in MLA:

    

    

    predicted = alg.fit(xtrain_tfidf_ngram_chars.toarray(), train_y).predict(xvalid_tfidf_ngram_chars.toarray())

    fp, tp, th = roc_curve(valid_y, predicted)

    roc_auc_mla = auc(fp, tp)

    MLA_name = alg.__class__.__name__

    plt.plot(fp, tp, lw=2, alpha=0.3, label='ROC %s (AUC = %0.2f)'  % (MLA_name, roc_auc_mla))

   

    index+=1



plt.title('ROC Curve comparison')

plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

plt.plot([0,1],[0,1],'r--')

plt.xlim([0,1])

plt.ylim([0,1])

plt.ylabel('True Positive Rate')

plt.xlabel('False Positive Rate')    

plt.show()
def train_model(classifier, feature_vector_train, label, feature_vector_valid, is_neural_net=False):

    # fit the training dataset on the classifier

    classifier.fit(feature_vector_train, label)

    

    # predict the labels on validation dataset

    predictions = classifier.predict(feature_vector_valid)

    

    if is_neural_net:

        predictions = predictions.argmax(axis=-1)

    

    return metrics.accuracy_score(predictions, valid_y)
# Naive Bayes on Count Vectors

accuracy = train_model(naive_bayes.MultinomialNB(), xtrain_count, train_y, xvalid_count)

accuracy1 = train_model(naive_bayes.BernoulliNB(), xtrain_count, train_y, xvalid_count)

print ("MultinomialNB, Count Vectors: ", accuracy)

print ("BernoulliNB, Count Vectors: ", accuracy1)



# Naive Bayes on Word Level TF IDF Vectors

accuracy = train_model(naive_bayes.MultinomialNB(), xtrain_tfidf, train_y, xvalid_tfidf)

accuracy1 = train_model(naive_bayes.BernoulliNB(), xtrain_tfidf, train_y, xvalid_tfidf)

print ("MultinomialNB, WordLevel TF-IDF: ", accuracy)

print ("BernoulliNB, WordLevel TF-IDF: ", accuracy1)



# Naive Bayes on Ngram Level TF IDF Vectors

accuracy = train_model(naive_bayes.MultinomialNB(), xtrain_tfidf_ngram, train_y, xvalid_tfidf_ngram)

accuracy1 = train_model(naive_bayes.BernoulliNB(), xtrain_tfidf_ngram, train_y, xvalid_tfidf_ngram)

print ("MulitnomialNB, N-Gram Vectors: ", accuracy)

print ("BernoulliNB, N-Gram Vectors: ", accuracy1)



# Naive Bayes on Character Level TF IDF Vectors

accuracy = train_model(naive_bayes.MultinomialNB(), xtrain_tfidf_ngram_chars, train_y, xvalid_tfidf_ngram_chars)

accuracy1 = train_model(naive_bayes.BernoulliNB(), xtrain_tfidf_ngram_chars, train_y, xvalid_tfidf_ngram_chars)

print ("MultinomialNB, CharLevel Vectors: ", accuracy)

print ("BernoulliNB, CharLevel Vectors: ", accuracy1)
accuracy = train_model(linear_model.LogisticRegression(), xtrain_count, train_y, xvalid_count)

print ("LR, Count Vectors: ", accuracy)



# Linear Classifier on Word Level TF IDF Vectors

accuracy = train_model(linear_model.LogisticRegression(), xtrain_tfidf, train_y, xvalid_tfidf)

print ("LR, WordLevel TF-IDF: ", accuracy)



# Linear Classifier on Ngram Level TF IDF Vectors

accuracy = train_model(linear_model.LogisticRegression(), xtrain_tfidf_ngram, train_y, xvalid_tfidf_ngram)

print ("LR, N-Gram Vectors: ", accuracy)



# Linear Classifier on Character Level TF IDF Vectors

accuracy = train_model(linear_model.LogisticRegression(), xtrain_tfidf_ngram_chars, train_y, xvalid_tfidf_ngram_chars)

print ("LR, CharLevel Vectors: ", accuracy)
accuracy = train_model(svm.SVC(), xtrain_tfidf, train_y, xvalid_tfidf)

print ("SVM, WordLevel TF-IDF: ", accuracy)



accuracy = train_model(svm.SVC(), xtrain_count, train_y, xvalid_count)

print ("SVM, Count Vectors: ", accuracy)



# SVM on Ngram Level TF IDF Vectors

accuracy = train_model(svm.SVC(), xtrain_tfidf_ngram, train_y, xvalid_tfidf_ngram)

print ("SVM, N-Gram Vectors: ", accuracy)
accuracy = train_model(ensemble.RandomForestClassifier(), xtrain_count, train_y, xvalid_count)

print ("RF, Count Vectors: ", accuracy)



# RF on Word Level TF IDF Vectors

accuracy = train_model(ensemble.RandomForestClassifier(), xtrain_tfidf, train_y, xvalid_tfidf)

print ("RF, WordLevel TF-IDF: ", accuracy)
# Extereme Gradient Boosting on Count Vectors

accuracy = train_model(xgboost.XGBClassifier(), xtrain_count.tocsc(), train_y, xvalid_count.tocsc())

print ("Xgb, Count Vectors: ", accuracy)



# Extereme Gradient Boosting on Word Level TF IDF Vectors

accuracy = train_model(xgboost.XGBClassifier(), xtrain_tfidf.tocsc(), train_y, xvalid_tfidf.tocsc())

print ("Xgb, WordLevel TF-IDF: ", accuracy)



# Extereme Gradient Boosting on Character Level TF IDF Vectors

accuracy = train_model(xgboost.XGBClassifier(), xtrain_tfidf_ngram_chars.tocsc(), train_y, xvalid_tfidf_ngram_chars.tocsc())

print ("Xgb, CharLevel Vectors: ", accuracy)
def create_model_architecture(input_size):

    # create input layer 

    input_layer = layers.Input((input_size, ), sparse=True)

    

    # create hidden layer

    hidden_layer = layers.Dense(1000, activation="relu")(input_layer)

    

    # create output layer

    output_layer = layers.Dense(1, activation="sigmoid")(hidden_layer)



    classifier = models.Model(inputs = input_layer, outputs = output_layer)

    classifier.compile(optimizer=optimizers.Adam(), loss='binary_crossentropy')

    return classifier 



classifier = create_model_architecture(xtrain_tfidf_ngram.shape[1])

accuracy = train_model(classifier, xtrain_tfidf_ngram, train_y, xvalid_tfidf_ngram, is_neural_net=True)

print ("NN, Ngram Level TF IDF Vectors",  accuracy)
