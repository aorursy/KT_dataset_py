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
import matplotlib.pyplot as plt

import re

import time

import warnings

from nltk.corpus import stopwords

from sklearn.decomposition import TruncatedSVD

from sklearn.preprocessing import normalize

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.manifold import TSNE

import seaborn as sns

from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import confusion_matrix

from sklearn.metrics.classification import accuracy_score, log_loss

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.linear_model import SGDClassifier

#from imblearn.over_sampling import SMOTE

from collections import Counter

from scipy.sparse import hstack

#from sklearn.multiclass import OneVsRestClassifier

from sklearn.svm import SVC

#from sklearn.cross_validation import StratifiedKFold

from sklearn.model_selection import StratifiedKFold

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
!pip install patool
!pip install pyunpack
from pyunpack import Archive

Archive('/kaggle/input/msk-redefining-cancer-treatment/stage2_test_variants.csv.7z').extractall('/kaggle/working')
Archive('/kaggle/input/msk-redefining-cancer-treatment/stage2_test_text.csv.7z').extractall('/kaggle/working')
df=pd.read_csv('/kaggle/input/msk-redefining-cancer-treatment/training_variants.zip')

dftest=pd.read_csv('/kaggle/working/stage2_test_variants.csv')
df.head()
dftest.head()
print('Train Number of data points:', df.shape[0])

print('Train Number of features:',df.shape[1])

print('Train Features:',df.columns.values)



print('Test Number of data points:', dftest.shape[0])

print('Test Number of features:',dftest.shape[1])

print('Test Features:',dftest.columns.values)
dftext=pd.read_csv('/kaggle/input/msk-redefining-cancer-treatment/training_text.zip', 

                   sep='\|\|', names=['ID', 'TEXT'], skiprows=1, engine='python')

dftesttext=pd.read_csv('/kaggle/working/stage2_test_text.csv', 

                   sep='\|\|', names=['ID', 'TEXT'], skiprows=1, engine='python')
dftext.head()
print('Number of data points:', dftext.shape[0])
dftesttext.head()
print('Test Number of data points:', dftesttext.shape[0])
# loading stopwords from nltk library



stop_words=set(stopwords.words('english'))



def nlp_preprocessing(total_text, index, column, cat):

    if type(total_text) is not int:

        s1=""

        total_text = re.sub('[^a-zA-Z0-9\n]', ' ', total_text)

        # replace multiple spaces with single space

        total_text = re.sub('\s+',' ', total_text)

        # converting all the chars into lower-case.

        total_text = total_text.lower()

        for word in total_text.split():

            # if the word is a not a stop word then retain that word from the data

            if not word in stop_words:

                s1 += word + " "

        if cat=='train':

            dftext[column][index] = s1

        elif cat=='test':

            dftesttext[column][index]=s1
#text processing stage.

start_time = time.clock()

for index, row in dftext.iterrows():

    if type(row['TEXT']) is str:

        nlp_preprocessing(row['TEXT'], index, 'TEXT', 'train')

    else:

        print("there is no text description for id:",index)

print('Time took for preprocessing the text :',time.clock() - start_time, "seconds")
#text processing stage  -  test data

start_time = time.clock()

for index, row in dftesttext.iterrows():

    if type(row['TEXT']) is str:

        nlp_preprocessing(row['TEXT'], index, 'TEXT', 'test')

    else:

        print("there is no text description for id:",index)

print('Time took for preprocessing the text :',time.clock() - start_time, "seconds")
#merging both gene_variations and text data based on ID

result = pd.merge(df, dftext,on='ID', how='left')

result.head()
result[result.isnull().any(axis=1)]
result.loc[result['TEXT'].isnull(), 'TEXT']= result['Gene']+' '+result['Variation']
result[result.isnull().any(axis=1)]
result['GV']=result['Gene']+' '+result['Variation']
result.head()
#merging both gene_variations and text data based on ID - Test data

resulttest = pd.merge(dftest, dftesttext,on='ID', how='left')

resulttest.head()
resulttest[resulttest.isnull().any(axis=1)]
resulttest.loc[resulttest['TEXT'].isnull(), 'TEXT']= resulttest['Gene']+' '+resulttest['Variation']
resulttest[resulttest.isnull().any(axis=1)]
resulttest['GV']=resulttest['Gene']+' '+resulttest['Variation']
resulttest.head()
y_true = result['Class'].values

result.Gene = result.Gene.str.replace('\s+', '_')

result.Variation = result.Variation.str.replace('\s+', '_')

# split the data into cv and train by maintaining same distribution of output

#varaible 'y_true' [stratify=y_true]

xtrain, xcv, ytrain, ycv = train_test_split(result, y_true, stratify=y_true, test_size=0.3)

print('Number of data points in train data:', xtrain.shape[0])

print('Number of data points in cross validation data:', xcv.shape[0])
## Distribution of classes



train_class_distribution = xtrain['Class'].value_counts().sort_index()

cv_class_distribution = xcv['Class'].value_counts().sort_index()
my_colors = ['r','g','b','k','y','m','c']

train_class_distribution.plot(kind='bar', color=my_colors)

plt.xlabel('Class')

plt.ylabel('Data points per Class')

plt.title('Distribution of class in train data')

plt.grid()

plt.show()
sorted_yi = np.argsort(-train_class_distribution.values)

for i in sorted_yi:

    print('Number of data points in class', i+1, ':',train_class_distribution.values[i], 

      '(', np.round((train_class_distribution.values[i]/xtrain.shape[0]*100), 3), '%)')
## cross-validation data



my_colors = ['r','g','b','k','y','m','c']

cv_class_distribution.plot(kind='bar', color=my_colors)

plt.xlabel('Class')

plt.ylabel('Data points per Class')

plt.title('Distribution of class in cross validation data')

plt.grid()

plt.show()
sorted_yi = np.argsort(-cv_class_distribution.values)

for i in sorted_yi:

    print('Number of data points in class', i+1, ':',cv_class_distribution.values[i], 

          '(', np.round((cv_class_distribution.values[i]/xcv.shape[0]*100), 3),'%)')
## to plot confusion matrix

def plot_confusion_matrix(test_y, predict_y):

    C = confusion_matrix(test_y, predict_y)

    A =(((C.T)/(C.sum(axis=1))).T)

    B =(C/C.sum(axis=0))

    labels = [1,2,3,4,5,6,7,8,9]



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



    print("-"*20, "Recall matrix (Row sum=1)", "-"*20)

    plt.figure(figsize=(20,7))

    sns.heatmap(A, annot=True, cmap="YlGnBu", fmt=".3f", xticklabels=labels, yticklabels=labels)

    plt.xlabel('Predicted Class')

    plt.ylabel('Original Class')

    plt.show()
# TFIDF vectorization of variation feature

variation_vectorizer_tf = TfidfVectorizer()

train_variation_feature_tf=variation_vectorizer_tf.fit_transform(xtrain['Variation'])

cv_variation_feature_tf = variation_vectorizer_tf.transform(xcv['Variation'])
print('train variation feature TFIDF vectorized shape: ', train_variation_feature_tf.shape)

print('cv variation feature TFIDF vectorized shape: ', cv_variation_feature_tf.shape)
## Test 

test_variation_feature_tf=variation_vectorizer_tf.transform(resulttest['Variation'])

print('Test variation feature TFIDF vectorized shape:', test_variation_feature_tf.shape)
## TFIDF vectorization of Gene feature

gene_tf_vectorizer=TfidfVectorizer()

train_gene_feature_tf=gene_tf_vectorizer.fit_transform(xtrain['Gene'])

cv_gene_feature_tf=gene_tf_vectorizer.transform(xcv['Gene'])
print("Shape of train gene feature when tfidf vectorized : ", train_gene_feature_tf.shape)

print("Shape of cv gene feature when tfidf vectorized : ", cv_gene_feature_tf.shape)
# Test

test_gene_feature_tf=gene_tf_vectorizer.transform(resulttest['Gene'])

print("Shape of Test gene feature when tfidf vectorized : ", test_gene_feature_tf.shape)
# building a TFIDFVectorizer for TEXT with all the words that occurred minimum 3 times 

#in train data,

# and max features 6000 and taking 1gram to 5gram



text_vectorizer_tf=TfidfVectorizer(min_df=3, ngram_range=(1,5), max_features=100000)

train_text_feature_tf=text_vectorizer_tf.fit_transform(xtrain['TEXT'])

train_text_features=text_vectorizer_tf.get_feature_names()

print('total number of unique words in train data: ', len(train_text_features))

train_text_fea_counts_tf=train_text_feature_tf.sum(axis=0).A1

text_fea_dict_tf=dict(zip(list(train_text_features), train_text_fea_counts_tf))

cv_text_feature_tf=text_vectorizer_tf.transform(xcv['TEXT'])



#print(train_text_fea_counts_tf)
print(train_text_feature_tf.shape)

print(cv_text_feature_tf.shape)
# Test

test_text_feature_tf=text_vectorizer_tf.transform(resulttest['TEXT'])

print(test_text_feature_tf.shape)
# merging gene, variance and text features



# building train, test and cross validation data sets

# a = [[1, 2], 

#      [3, 4]]

# b = [[4, 5], 

#      [6, 7]]

# hstack(a, b) = [[1, 2, 4, 5],

#                [ 3, 4, 6, 7]]



train_gene_var_tf = hstack((train_gene_feature_tf,train_variation_feature_tf))

cv_gene_var_tf = hstack((cv_gene_feature_tf,cv_variation_feature_tf))



train_x_tf = hstack((train_gene_var_tf, train_text_feature_tf)).tocsr()

train_y = np.array(list(xtrain['Class']))



cv_x_tf = hstack((cv_gene_var_tf, cv_text_feature_tf)).tocsr()

cv_y = np.array(list(xcv['Class']))
print("TFIDF features :")

print("(number of data points * number of features) in train data = ", train_x_tf.shape)

print("(number of data points * number of features) in cross validation data =", cv_x_tf.shape)
## Test

test_gene_var_tf=hstack((test_gene_feature_tf, test_variation_feature_tf))

test_x_tf=hstack((test_gene_var_tf, test_text_feature_tf)).tocsr()

print('Test shape:', test_x_tf.shape)
alpha = [10 ** x for x in range(-6, 3)]

cv_log_error_array = []

for i in alpha:

    print("for alpha =", i)

    clf = SGDClassifier(class_weight='balanced', alpha=i, penalty='l2', loss='log', random_state=42)

    sig_clf = CalibratedClassifierCV(clf, method="sigmoid")

    sig_clf.fit(train_x_tf, train_y)

    sig_clf_probs = sig_clf.predict_proba(cv_x_tf)

    cv_log_error_array.append(log_loss(cv_y, sig_clf_probs, eps=1e-15))

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

clf.fit(train_x_tf, train_y)

sig_clf = CalibratedClassifierCV(clf, method="sigmoid")

sig_clf.fit(train_x_tf, train_y)



predict_y = sig_clf.predict_proba(train_x_tf)

print('For values of best alpha = ', alpha[best_alpha], "The train log loss is:",log_loss(ytrain, predict_y,  eps=1e-15))

predict_y = sig_clf.predict_proba(cv_x_tf)

print('For values of best alpha = ', alpha[best_alpha], "The cross validation log loss is:",log_loss(ycv, predict_y,  eps=1e-15))

res = sig_clf.predict_proba(test_x_tf)
res
ids=np.arange(1, len(res)+1)
line1='class1 class2 class3 class4 class5 class6 class7 class8 class9'

s1=line1.split()

print(s1)
resdf=pd.DataFrame(res, columns=s1)
resdf.head()
resdf.insert(0,'ID',ids,True)
resdf.head()
filename='cancer_treat_logistic.csv'

resdf.to_csv(filename, index=False)

print('Saved file ',filename)