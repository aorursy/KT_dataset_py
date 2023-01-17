# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
sub=pd.read_csv('/kaggle/input/msk-redefining-cancer-treatment/stage2_sample_submission.csv')

sub.head()
import matplotlib.pyplot as plt

import re

import time

import warnings

from nltk.corpus import stopwords

from sklearn.decomposition import TruncatedSVD

from sklearn.preprocessing import normalize

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.feature_extraction.text import TfidfTransformer

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.manifold import TSNE

import seaborn as sns

from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import confusion_matrix

from sklearn.metrics.classification import accuracy_score, log_loss

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.linear_model import SGDClassifier

from imblearn.over_sampling import SMOTE

from collections import Counter

from scipy.sparse import hstack

from sklearn.multiclass import OneVsRestClassifier

from sklearn.svm import SVC

from sklearn.model_selection import train_test_split

#from sklearn.cross_validation import StratifiedKFold 

from collections import Counter, defaultdict

from sklearn.calibration import CalibratedClassifierCV

from sklearn.naive_bayes import MultinomialNB

from sklearn.naive_bayes import GaussianNB

from sklearn.model_selection import train_test_split

from sklearn.model_selection import GridSearchCV

import math

from sklearn.metrics import normalized_mutual_info_score

from sklearn.ensemble import RandomForestClassifier

warnings.filterwarnings("ignore")



from mlxtend.classifier import StackingClassifier



from sklearn import model_selection

from sklearn.linear_model import LogisticRegression

#lets see the data

import pandas as pd

data=pd.read_csv("/kaggle/input/msk-redefining-cancer-treatment/training_variants")

print('Number of data points : ', data.shape[0])

print('Number of features : ', data.shape[1])

print('Features : ', data.columns.values)

data.head()
# note the seprator in this file

data_text =pd.read_csv("/kaggle/input/msk-redefining-cancer-treatment/training_text",sep="\|\|",engine="python",names=["ID","TEXT"],skiprows=1)

print('Number of data points : ', data_text.shape[0])

print('Number of features : ', data_text.shape[1])

print('Features : ', data_text.columns.values)

data_text.head()
# loading stop words from nltk library

stop_words = set(stopwords.words('english'))





def nlp_preprocessing(total_text, index, column):

    if type(total_text) is not int:

        string = ""

        # replace every special char with space

        total_text = re.sub('[^a-zA-Z0-9\n]', ' ', total_text)

        # replace multiple spaces with single space

        total_text = re.sub('\s+',' ', total_text)

        # converting all the chars into lower-case.

        total_text = total_text.lower()

        

        for word in total_text.split():

        # if the word is a not a stop word then retain that word from the data

            if not word in stop_words:

                string += word + " "

        

        data_text[column][index] = string
#text processing stage.

start_time = time.clock()

for index, row in data_text.iterrows():

    if type(row['TEXT']) is str:

        nlp_preprocessing(row['TEXT'], index, 'TEXT')

    else:

        print("there is no text description for id:",index)

print('Time took for preprocessing the text :',time.clock() - start_time, "seconds")
#merging both gene_variations and text data based on ID

result = pd.merge(data, data_text,on='ID', how='left')

result.head()
result[result.isnull().any(axis=1)]
result.loc[result['TEXT'].isnull(),'TEXT'] = result['Gene'] +' '+result['Variation']
result[result['ID']==1109]
y_true = result['Class'].values

result.Gene      = result.Gene.str.replace('\s+', '_')

result.Variation = result.Variation.str.replace('\s+', '_')



# split the data into test and train by maintaining same distribution of output varaible 'y_true' [stratify=y_true]

X_train, test_df, y_train, y_test = train_test_split(result, y_true, stratify=y_true, test_size=0.2)

# split the train data into train and cross validation by maintaining same distribution of output varaible 'y_train' [stratify=y_train]

train_df, cv_df, y_train, y_cv = train_test_split(X_train, y_train, stratify=y_train, test_size=0.2)
print('Number of data points in train data:', train_df.shape[0])

print('Number of data points in test data:', test_df.shape[0])

print('Number of data points in cross validation data:', cv_df.shape[0])
# This function plots the confusion matrices given y_i, y_i_hat.

def plot_confusion_matrix(test_y, predict_y):

    C = confusion_matrix(test_y, predict_y)

    # C = 9,9 matrix, each cell (i,j) represents number of points of class i are predicted class j

    

    A =(((C.T)/(C.sum(axis=1))).T)

    #divid each element of the confusion matrix with the sum of elements in that column

    

    # C = [[1, 2],

    #     [3, 4]]

    # C.T = [[1, 3],

    #        [2, 4]]

    # C.sum(axis = 1)  axis=0 corresonds to columns and axis=1 corresponds to rows in two diamensional array

    # C.sum(axix =1) = [[3, 7]]

    # ((C.T)/(C.sum(axis=1))) = [[1/3, 3/7]

    #                           [2/3, 4/7]]



    # ((C.T)/(C.sum(axis=1))).T = [[1/3, 2/3]

    #                           [3/7, 4/7]]

    # sum of row elements = 1

    

    B =(C/C.sum(axis=0))

    #divid each element of the confusion matrix with the sum of elements in that row

    # C = [[1, 2],

    #     [3, 4]]

    # C.sum(axis = 0)  axis=0 corresonds to columns and axis=1 corresponds to rows in two diamensional array

    # C.sum(axix =0) = [[4, 6]]

    # (C/C.sum(axis=0)) = [[1/4, 2/6],

    #                      [3/4, 4/6]] 

    

    labels = [1,2,3,4,5,6,7,8,9]

    # representing A in heatmap format

    print("-"*20, "Confusion matrix", "-"*20)

    plt.figure(figsize=(20,7))

    sns.heatmap(C, annot=True, cmap="YlGnBu", fmt=".3f", xticklabels=labels, yticklabels=labels)

    plt.xlabel('Predicted Class')

    plt.ylabel('Original Class')

    plt.show()



    print("-"*20, "Precision matrix (Columm Sum=1)", "-"*20)

    plt.figure(figsize=(20,7))

    sns.heatmap(B, annot=True, cmap="YlGnBu", fmt=".3f", xticklabels=labels, yticklabels=labels)

    plt.xlabel('Predicted Class')

    plt.ylabel('Original Class')

    plt.show()

    

    # representing B in heatmap format

    print("-"*20, "Recall matrix (Row sum=1)", "-"*20)

    plt.figure(figsize=(20,7))

    sns.heatmap(A, annot=True, cmap="YlGnBu", fmt=".3f", xticklabels=labels, yticklabels=labels)

    plt.xlabel('Predicted Class')

    plt.ylabel('Original Class')

    plt.show()
# we need to generate 9 numbers and the sum of numbers should be 1

# one solution is to genarate 9 numbers and divide each of the numbers by their sum

# ref: https://stackoverflow.com/a/18662466/4084039

test_data_len = test_df.shape[0]

cv_data_len = cv_df.shape[0]



# we create a output array that has exactly same size as the CV data

cv_predicted_y = np.zeros((cv_data_len,9))

for i in range(cv_data_len):

    rand_probs = np.random.rand(1,9)

    cv_predicted_y[i] = ((rand_probs/sum(sum(rand_probs)))[0])

print("Log loss on Cross Validation Data using Random Model",log_loss(y_cv,cv_predicted_y, eps=1e-15))





# Test-Set error.

#we create a output array that has exactly same as the test data

test_predicted_y = np.zeros((test_data_len,9))

for i in range(test_data_len):

    rand_probs = np.random.rand(1,9)

    test_predicted_y[i] = ((rand_probs/sum(sum(rand_probs)))[0])

print("Log loss on Test Data using Random Model",log_loss(y_test,test_predicted_y, eps=1e-15))



predicted_y =np.argmax(test_predicted_y, axis=1)

plot_confusion_matrix(y_test, predicted_y+1)
# one-hot encoding of Gene feature.

gene_vectorizer = CountVectorizer()

train_gene_feature_onehotCoding = gene_vectorizer.fit_transform(train_df['Gene'])

test_gene_feature_onehotCoding = gene_vectorizer.transform(test_df['Gene'])

cv_gene_feature_onehotCoding = gene_vectorizer.transform(cv_df['Gene'])
# one-hot encoding of variation feature.

variation_vectorizer = CountVectorizer()

train_variation_feature_onehotCoding = variation_vectorizer.fit_transform(train_df['Variation'])

test_variation_feature_onehotCoding = variation_vectorizer.transform(test_df['Variation'])

cv_variation_feature_onehotCoding = variation_vectorizer.transform(cv_df['Variation'])
# building a Vectorizer with all the words that occured minimum 3 times in train data

from sklearn.feature_selection import SelectKBest

text_vectorizer = TfidfVectorizer(ngram_range=(1,4),min_df=3,max_features=1000)

train_text_feature_onehotCoding = text_vectorizer.fit_transform(train_df['TEXT'])

#KBest = SelectKBest(k=2000)

#train_text_feature_onehotCoding=KBest.fit_transform(train_text_feature_onehotCoding,train_df['Class'])

# getting all the feature names (words)

#train_text_features= text_vectorizer.get_feature_names()



# train_text_feature_onehotCoding.sum(axis=0).A1 will sum every row and returns (1*number of features) vector

#train_text_fea_counts = train_text_feature_onehotCoding.sum(axis=0).A1



# zip(list(text_features),text_fea_counts) will zip a word with its number of times it occured

#text_fea_dict = dict(zip(list(train_text_features),train_text_fea_counts))





#print("Total number of unique words in train data :", len(train_text_features))
# don't forget to normalize every feature

train_text_feature_onehotCoding = normalize(train_text_feature_onehotCoding, axis=0)



# we use the same vectorizer that was trained on train data

test_text_feature_onehotCoding = text_vectorizer.transform(test_df['TEXT'])

#test_text_feature_onehotCoding=KBest.transform(test_text_feature_onehotCoding)

# don't forget to normalize every feature

test_text_feature_onehotCoding = normalize(test_text_feature_onehotCoding, axis=0)



# we use the same vectorizer that was trained on train data

cv_text_feature_onehotCoding = text_vectorizer.transform(cv_df['TEXT'])

#cv_text_feature_onehotCoding=KBest.transform(cv_text_feature_onehotCoding)

# don't forget to normalize every feature

cv_text_feature_onehotCoding = normalize(cv_text_feature_onehotCoding, axis=0)
train_gene_var_onehotCoding = hstack((train_gene_feature_onehotCoding,train_variation_feature_onehotCoding))

test_gene_var_onehotCoding = hstack((test_gene_feature_onehotCoding,test_variation_feature_onehotCoding))

cv_gene_var_onehotCoding = hstack((cv_gene_feature_onehotCoding,cv_variation_feature_onehotCoding))



train_x_onehotCoding = hstack((train_gene_var_onehotCoding, train_text_feature_onehotCoding)).tocsr()

train_y = np.array(list(train_df['Class']))



test_x_onehotCoding = hstack((test_gene_var_onehotCoding, test_text_feature_onehotCoding)).tocsr()

test_y = np.array(list(test_df['Class']))



cv_x_onehotCoding = hstack((cv_gene_var_onehotCoding, cv_text_feature_onehotCoding)).tocsr()

cv_y = np.array(list(cv_df['Class']))
alpha = [10 ** x for x in range(-6, 3)]

cv_log_error_array = []

for i in alpha:

    print("for alpha =", i)

    clf = SGDClassifier(class_weight='balanced', alpha=i, penalty='l2', loss='log', random_state=42)

    clf.fit(train_x_onehotCoding, train_y)

    sig_clf = CalibratedClassifierCV(clf, method="sigmoid")

    sig_clf.fit(train_x_onehotCoding, train_y)

    sig_clf_probs = sig_clf.predict_proba(cv_x_onehotCoding)

    cv_log_error_array.append(log_loss(cv_y, sig_clf_probs, labels=clf.classes_, eps=1e-15))

    # to avoid rounding error while multiplying probabilites we use log-probability estimates

    print("Log Loss :",log_loss(cv_y, sig_clf_probs)) 



fig, ax = plt.subplots()

ax.plot(alpha, cv_log_error_array,c='g')

for i, txt in enumerate(np.round(cv_log_error_array,3)):

    ax.annotate((alpha[i],str(txt)), (alpha[i],cv_log_error_array[i]))

plt.grid()

plt.title("Cross Validation Error for each alpha")

plt.xlabel("Alpha i's")

plt.ylabel("Error measure")

plt.show()





best_alpha = np.argmin(cv_log_error_array)

clf = SGDClassifier(class_weight='balanced', alpha=alpha[best_alpha], penalty='l2', loss='log', random_state=42)

clf.fit(train_x_onehotCoding, train_y)

sig_clf = CalibratedClassifierCV(clf, method="sigmoid")

sig_clf.fit(train_x_onehotCoding, train_y)



predict_y = sig_clf.predict_proba(train_x_onehotCoding)

print('For values of best alpha = ', alpha[best_alpha], "The train log loss is:",log_loss(y_train, predict_y, labels=clf.classes_, eps=1e-15))

predict_y = sig_clf.predict_proba(cv_x_onehotCoding)

print('For values of best alpha = ', alpha[best_alpha], "The cross validation log loss is:",log_loss(y_cv, predict_y, labels=clf.classes_, eps=1e-15))

predict_y = sig_clf.predict_proba(test_x_onehotCoding)

print('For values of best alpha = ', alpha[best_alpha], "The test log loss is:",log_loss(y_test, predict_y, labels=clf.classes_, eps=1e-15))
def predict_and_plot_confusion_matrix(train_x, train_y,test_x, test_y, clf):

    clf.fit(train_x, train_y)

    sig_clf = CalibratedClassifierCV(clf, method="sigmoid")

    sig_clf.fit(train_x, train_y)

    pred_y = sig_clf.predict(test_x)



    # for calculating log_loss we willl provide the array of probabilities belongs to each class

    print("Log loss :",log_loss(test_y, sig_clf.predict_proba(test_x)))

    # calculating the number of data points that are misclassified

    print("Number of mis-classified points :", np.count_nonzero((pred_y- test_y))/test_y.shape[0])

    plot_confusion_matrix(test_y, pred_y)