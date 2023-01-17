import pandas as pd

import matplotlib.pyplot as plt

import re

import time

import warnings

import numpy as np

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

from imblearn.over_sampling import SMOTE

from collections import Counter

from scipy.sparse import hstack

from sklearn.multiclass import OneVsRestClassifier

from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold 

from collections import Counter, defaultdict

from sklearn.calibration import CalibratedClassifierCV

from sklearn.naive_bayes import MultinomialNB

from sklearn.naive_bayes import GaussianNB



# from sklearn.model_selection import train_test_split

from sklearn.model_selection import train_test_split

from sklearn.model_selection import GridSearchCV

import math

from sklearn.metrics import normalized_mutual_info_score

from sklearn.ensemble import RandomForestClassifier

warnings.filterwarnings("ignore")



from mlxtend.classifier import StackingClassifier



from sklearn import model_selection

from sklearn.linear_model import LogisticRegression

print("DONE")

import sklearn
import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
data = pd.read_csv('/kaggle/input/msk-redefining-cancer-treatment/training_variants.zip')

# data = pd.read_csv('training/training_variants')

print('Number of data points : ', data.shape[0])

print('Number of features : ', data.shape[1])

print('Features : ', data.columns.values)

data.head()
# note the separator in this file

data_text =pd.read_csv("/kaggle/input/msk-redefining-cancer-treatment/training_text.zip",sep="\|\|",engine="python",names=["ID","TEXT"],skiprows=1)

# data_text =pd.read_csv("training/training_text",sep="\|\|",engine="python",names=["ID","TEXT"],skiprows=1)

print('Number of data points : ', data_text.shape[0])

print('Number of features : ', data_text.shape[1])

print('Features : ', data_text.columns.values)

data_text.head()
import nltk

nltk.download('stopwords')
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
tempresult=result.copy()
tempresult
result.loc[result['TEXT'].isnull(),'TEXT'] = result['Gene'] +' '+result['Variation']
result['TEXT'].isnull()
result[result['ID']==1109]
result.Variation
result.Gene 
result.Variation
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
# it returns a dict, keys as class labels and values as the number of data points in that class

train_class_distribution = train_df['Class'].value_counts().sort_index()

test_class_distribution = test_df['Class'].value_counts().sort_index()

cv_class_distribution = cv_df['Class'].value_counts().sort_index()



my_colors = 'rgbkymc'

train_class_distribution.plot(kind='bar')

plt.xlabel('Class')

plt.ylabel('Data points per Class')

plt.title('Distribution of yi in train data')

plt.grid()

plt.show()



# ref: argsort https://docs.scipy.org/doc/numpy/reference/generated/numpy.argsort.html

# -(train_class_distribution.values): the minus sign will give us in decreasing order

sorted_yi = np.argsort(-train_class_distribution.values)

for i in sorted_yi:

    print('Number of data points in class', i+1, ':',train_class_distribution.values[i], '(', np.round((train_class_distribution.values[i]/train_df.shape[0]*100), 3), '%)')



    

print('-'*80)

my_colors = 'rgbkymc'

test_class_distribution.plot(kind='bar')

plt.xlabel('Class')

plt.ylabel('Data points per Class')

plt.title('Distribution of yi in test data')

plt.grid()

plt.show()



# ref: argsort https://docs.scipy.org/doc/numpy/reference/generated/numpy.argsort.html

# -(train_class_distribution.values): the minus sign will give us in decreasing order

sorted_yi = np.argsort(-test_class_distribution.values)

for i in sorted_yi:

    print('Number of data points in class', i+1, ':',test_class_distribution.values[i], '(', np.round((test_class_distribution.values[i]/test_df.shape[0]*100), 3), '%)')



print('-'*80)

my_colors = 'rgbkymc'

cv_class_distribution.plot(kind='bar')

plt.xlabel('Class')

plt.ylabel('Data points per Class')

plt.title('Distribution of yi in cross validation data')

plt.grid()

plt.show()



# ref: argsort https://docs.scipy.org/doc/numpy/reference/generated/numpy.argsort.html

# -(train_class_distribution.values): the minus sign will give us in decreasing order

sorted_yi = np.argsort(-train_class_distribution.values)

for i in sorted_yi:

    print('Number of data points in class', i+1, ':',cv_class_distribution.values[i], '(', np.round((cv_class_distribution.values[i]/cv_df.shape[0]*100), 3), '%)')

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

#     return logLoss,missClassified
# This function plots the confusion matrices given y_i, y_i_hat.

def plot_confusion_matrix2(test_y, predict_y,logLoss,missClassified):

    C = confusion_matrix(test_y, predict_y)

    # C = 9,9 matrix, each cell (i,j) represents number of points of class i are predicted class j

    

    A =(((C.T)/(C.sum(axis=1))).T)

       

    B =(C/C.sum(axis=0))

    

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

    return logLoss,missClassified
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
# code for response coding with Laplace smoothing.

# alpha : used for laplace smoothing

# feature: ['gene', 'variation']

# df: ['train_df', 'test_df', 'cv_df']

# algorithm

# ----------

# Consider all unique values and the number of occurances of given feature in train data dataframe

# build a vector (1*9) , the first element = (number of times it occured in class1 + 10*alpha / number of time it occurred in total data+90*alpha)

# gv_dict is like a look up table, for every gene it store a (1*9) representation of it

# for a value of feature in df:

# if it is in train data:

# we add the vector that was stored in 'gv_dict' look up table to 'gv_fea'

# if it is not there is train:

# we add [1/9, 1/9, 1/9, 1/9,1/9, 1/9, 1/9, 1/9, 1/9] to 'gv_fea'

# return 'gv_fea'

# ----------------------



# get_gv_fea_dict: Get Gene varaition Feature Dict

def get_gv_fea_dict(alpha, feature, df):

    # value_count: it contains a dict like

    # print(train_df['Gene'].value_counts())

    # output:

    #        {BRCA1      174

    #         TP53       106

    #         EGFR        86

    #         BRCA2       75

    #         PTEN        69

    #         KIT         61

    #         BRAF        60

    #         ERBB2       47

    #         PDGFRA      46

    #         ...}

    # print(train_df['Variation'].value_counts())

    # output:

    # {

    # Truncating_Mutations                     63

    # Deletion                                 43

    # Amplification                            43

    # Fusions                                  22

    # Overexpression                            3

    # E17K                                      3

    # Q61L                                      3

    # S222D                                     2

    # P130S                                     2

    # ...

    # }

    value_count = train_df[feature].value_counts()

    

    # gv_dict : Gene Variation Dict, which contains the probability array for each gene/variation

    gv_dict = dict()

    

    # denominator will contain the number of time that particular feature occured in whole data

    for i, denominator in value_count.items():

        # vec will contain (p(yi==1/Gi) probability of gene/variation belongs to perticular class

        # vec is 9 diamensional vector

        vec = []

        for k in range(1,10):

            # print(train_df.loc[(train_df['Class']==1) & (train_df['Gene']=='BRCA1')])

            #         ID   Gene             Variation  Class  

            # 2470  2470  BRCA1                S1715C      1   

            # 2486  2486  BRCA1                S1841R      1   

            # 2614  2614  BRCA1                   M1R      1   

            # 2432  2432  BRCA1                L1657P      1   

            # 2567  2567  BRCA1                T1685A      1   

            # 2583  2583  BRCA1                E1660G      1   

            # 2634  2634  BRCA1                W1718L      1   

            # cls_cnt.shape[0] will return the number of rows



            cls_cnt = train_df.loc[(train_df['Class']==k) & (train_df[feature]==i)]

            

            # cls_cnt.shape[0](numerator) will contain the number of time that particular feature occured in whole data

            vec.append((cls_cnt.shape[0] + alpha*10)/ (denominator + 90*alpha))



        # we are adding the gene/variation to the dict as key and vec as value

        gv_dict[i]=vec

    return gv_dict



# Get Gene variation feature

def get_gv_feature(alpha, feature, df):

    # print(gv_dict)

    #     {'BRCA1': [0.20075757575757575, 0.03787878787878788, 0.068181818181818177, 0.13636363636363635, 0.25, 0.19318181818181818, 0.03787878787878788, 0.03787878787878788, 0.03787878787878788], 

    #      'TP53': [0.32142857142857145, 0.061224489795918366, 0.061224489795918366, 0.27040816326530615, 0.061224489795918366, 0.066326530612244902, 0.051020408163265307, 0.051020408163265307, 0.056122448979591837], 

    #      'EGFR': [0.056818181818181816, 0.21590909090909091, 0.0625, 0.068181818181818177, 0.068181818181818177, 0.0625, 0.34659090909090912, 0.0625, 0.056818181818181816], 

    #      'BRCA2': [0.13333333333333333, 0.060606060606060608, 0.060606060606060608, 0.078787878787878782, 0.1393939393939394, 0.34545454545454546, 0.060606060606060608, 0.060606060606060608, 0.060606060606060608], 

    #      'PTEN': [0.069182389937106917, 0.062893081761006289, 0.069182389937106917, 0.46540880503144655, 0.075471698113207544, 0.062893081761006289, 0.069182389937106917, 0.062893081761006289, 0.062893081761006289], 

    #      'KIT': [0.066225165562913912, 0.25165562913907286, 0.072847682119205295, 0.072847682119205295, 0.066225165562913912, 0.066225165562913912, 0.27152317880794702, 0.066225165562913912, 0.066225165562913912], 

    #      'BRAF': [0.066666666666666666, 0.17999999999999999, 0.073333333333333334, 0.073333333333333334, 0.093333333333333338, 0.080000000000000002, 0.29999999999999999, 0.066666666666666666, 0.066666666666666666],

    #      ...

    #     }

    gv_dict = get_gv_fea_dict(alpha, feature, df)

    # value_count is similar in get_gv_fea_dict

    value_count = train_df[feature].value_counts()

    

    # gv_fea: Gene_variation feature, it will contain the feature for each feature value in the data

    gv_fea = []

    # for every feature values in the given data frame we will check if it is there in the train data then we will add the feature to gv_fea

    # if not we will add [1/9,1/9,1/9,1/9,1/9,1/9,1/9,1/9,1/9] to gv_fea

    for index, row in df.iterrows():

        if row[feature] in dict(value_count).keys():

            gv_fea.append(gv_dict[row[feature]])

        else:

            gv_fea.append([1/9,1/9,1/9,1/9,1/9,1/9,1/9,1/9,1/9])

#             gv_fea.append([-1,-1,-1,-1,-1,-1,-1,-1,-1])

    return gv_fea
uniGenes=train_df['Gene'].value_counts()
uniGenes
unique_genes = train_df['Gene'].value_counts()

print('Number of Unique Genes :', unique_genes.shape[0])

# the top 10 genes that occured most

print(unique_genes.head(10))
unique_genes.shape
print("Ans: There are", unique_genes.shape[0] ,"different categories of genes in the train data, and they are distibuted as follows",)
s = sum(unique_genes.values);

h = unique_genes.values/s;

plt.plot(h, label="Histrogram of Genes")

plt.xlabel('Index of a Gene')

plt.ylabel('Number of Occurances')

plt.legend()

plt.grid()

plt.show()

c = np.cumsum(h)

plt.plot(c,label='Cumulative distribution of Genes')

plt.grid()

plt.legend()

plt.show()
# #response-coding of the Gene feature

# # alpha is used for laplace smoothing

# alpha = 1

# # train gene feature

# train_gene_feature_responseCoding = np.array(get_gv_feature(alpha, "Gene", train_df))

# # test gene feature

# test_gene_feature_responseCoding = np.array(get_gv_feature(alpha, "Gene", test_df))

# # cross validation gene feature

# cv_gene_feature_responseCoding = np.array(get_gv_feature(alpha, "Gene", cv_df))
# cv_gene_feature_responseCoding[0]
# cv_gene_feature_responseCoding[0].sum()
# print("train_gene_feature_responseCoding is converted feature using respone coding method. The shape of gene feature:", train_gene_feature_responseCoding.shape)
# one-hot encoding of Gene feature.

## gene_vectorizer = CountVectorizer()

gene_vectorizer = TfidfVectorizer(max_features=1000)

train_gene_feature_onehotCoding = gene_vectorizer.fit_transform(train_df['Gene'])

test_gene_feature_onehotCoding = gene_vectorizer.transform(test_df['Gene'])

cv_gene_feature_onehotCoding = gene_vectorizer.transform(cv_df['Gene'])

print('train_gene_feature_onehotCoding',train_gene_feature_onehotCoding.shape)
dd=train_gene_feature_onehotCoding.todense()
dd[0]
dd.shape
train_df['Gene'].head()
gene_vectorizer.get_feature_names()
print("train_gene_feature_onehotCoding is converted feature using one-hot encoding method. The shape of gene feature:", train_gene_feature_onehotCoding.shape)
from sklearn.calibration import CalibratedClassifierCV
alpha = [10 ** x for x in range(-5, 1)] # hyperparam for SGD classifier.



# read more about SGDClassifier() at http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDClassifier.html

# ------------------------------

# default parameters

# SGDClassifier(loss=’hinge’, penalty=’l2’, alpha=0.0001, l1_ratio=0.15, fit_intercept=True, max_iter=None, tol=None, 

# shuffle=True, verbose=0, epsilon=0.1, n_jobs=1, random_state=None, learning_rate=’optimal’, eta0=0.0, power_t=0.5, 

# class_weight=None, warm_start=False, average=False, n_iter=None)



# some of methods

# fit(X, y[, coef_init, intercept_init, …])	Fit linear model with Stochastic Gradient Descent.

# predict(X)	Predict class labels for samples in X.



#-------------------------------

# video link: 

#------------------------------





cv_log_error_array=[]

for i in alpha:

    clf = SGDClassifier(alpha=i, penalty='l2', loss='log', random_state=42)

    clf.fit(train_gene_feature_onehotCoding, y_train)

    sig_clf = CalibratedClassifierCV(clf, method="sigmoid")

    sig_clf.fit(train_gene_feature_onehotCoding, y_train)

    predict_y = sig_clf.predict_proba(cv_gene_feature_onehotCoding)

    cv_log_error_array.append(log_loss(y_cv, predict_y, labels=clf.classes_, eps=1e-15))

    print('For values of alpha = ', i, "The log loss is:",log_loss(y_cv, predict_y, labels=clf.classes_, eps=1e-15))



fig, ax = plt.subplots()

ax.plot(alpha, cv_log_error_array,c='g')

for i, txt in enumerate(np.round(cv_log_error_array,3)):

    ax.annotate((alpha[i],np.round(txt,3)), (alpha[i],cv_log_error_array[i]))

plt.grid()

plt.title("Cross Validation Error for each alpha")

plt.xlabel("Alpha i's")

plt.ylabel("Error measure")

plt.show()





best_alpha = np.argmin(cv_log_error_array)

clf = SGDClassifier(alpha=alpha[best_alpha], penalty='l2', loss='log', random_state=42)

clf.fit(train_gene_feature_onehotCoding, y_train)

sig_clf = CalibratedClassifierCV(clf, method="sigmoid")

sig_clf.fit(train_gene_feature_onehotCoding, y_train)



predict_y = sig_clf.predict_proba(train_gene_feature_onehotCoding)

print('For values of best alpha = ', alpha[best_alpha], "The train log loss is:",log_loss(y_train, predict_y, labels=clf.classes_, eps=1e-15))

predict_y = sig_clf.predict_proba(cv_gene_feature_onehotCoding)

print('For values of best alpha = ', alpha[best_alpha], "The cross validation log loss is:",log_loss(y_cv, predict_y, labels=clf.classes_, eps=1e-15))

predict_y = sig_clf.predict_proba(test_gene_feature_onehotCoding)

print('For values of best alpha = ', alpha[best_alpha], "The test log loss is:",log_loss(y_test, predict_y, labels=clf.classes_, eps=1e-15))

print("Q6. How many data points in Test and CV datasets are covered by the ", unique_genes.shape[0], " genes in train dataset?")



test_coverage=test_df[test_df['Gene'].isin(list(set(train_df['Gene'])))].shape[0]

cv_coverage=cv_df[cv_df['Gene'].isin(list(set(train_df['Gene'])))].shape[0]



print('Ans\n1. In test data',test_coverage, 'out of',test_df.shape[0], ":",(test_coverage/test_df.shape[0])*100)

print('2. In cross validation data',cv_coverage, 'out of ',cv_df.shape[0],":" ,(cv_coverage/cv_df.shape[0])*100)
# one-hot encoding of Gene feature.

gene_vectorizer_onehot = CountVectorizer(ngram_range=(1,2))

# gene_vectorizer = TfidfVectorizer(max_features=1000)

train_gene_feature_onehotCoding_onehot = gene_vectorizer_onehot.fit_transform(train_df['Gene'])

test_gene_feature_onehotCoding_onehot = gene_vectorizer_onehot.transform(test_df['Gene'])

cv_gene_feature_onehotCoding_onehot = gene_vectorizer_onehot.transform(cv_df['Gene'])

print('train_gene_feature_onehotCoding_onehot',train_gene_feature_onehotCoding_onehot.shape)
from sklearn.calibration import CalibratedClassifierCV
alpha = [10 ** x for x in range(-5, 1)] # hyperparam for SGD classifier.

cv_log_error_array=[]

for i in alpha:

    clf = SGDClassifier(alpha=i, penalty='l2', loss='log', random_state=42)

    clf.fit(train_gene_feature_onehotCoding_onehot, y_train)

    sig_clf = CalibratedClassifierCV(clf, method="sigmoid")

    sig_clf.fit(train_gene_feature_onehotCoding_onehot, y_train)

    predict_y = sig_clf.predict_proba(cv_gene_feature_onehotCoding_onehot)

    cv_log_error_array.append(log_loss(y_cv, predict_y, labels=clf.classes_, eps=1e-15))

    print('For values of alpha = ', i, "The log loss is:",log_loss(y_cv, predict_y, labels=clf.classes_, eps=1e-15))



fig, ax = plt.subplots()

ax.plot(alpha, cv_log_error_array,c='g')

for i, txt in enumerate(np.round(cv_log_error_array,3)):

    ax.annotate((alpha[i],np.round(txt,3)), (alpha[i],cv_log_error_array[i]))

plt.grid()

plt.title("Cross Validation Error for each alpha")

plt.xlabel("Alpha i's")

plt.ylabel("Error measure")

plt.show()





best_alpha = np.argmin(cv_log_error_array)

clf = SGDClassifier(alpha=alpha[best_alpha], penalty='l2', loss='log', random_state=42)

clf.fit(train_gene_feature_onehotCoding_onehot, y_train)

sig_clf = CalibratedClassifierCV(clf, method="sigmoid")

sig_clf.fit(train_gene_feature_onehotCoding_onehot, y_train)



predict_y = sig_clf.predict_proba(train_gene_feature_onehotCoding_onehot)

print('For values of best alpha = ', alpha[best_alpha], "The train log loss is:",log_loss(y_train, predict_y, labels=clf.classes_, eps=1e-15))

predict_y = sig_clf.predict_proba(cv_gene_feature_onehotCoding_onehot)

print('For values of best alpha = ', alpha[best_alpha], "The cross validation log loss is:",log_loss(y_cv, predict_y, labels=clf.classes_, eps=1e-15))

predict_y = sig_clf.predict_proba(test_gene_feature_onehotCoding_onehot)

print('For values of best alpha = ', alpha[best_alpha], "The test log loss is:",log_loss(y_test, predict_y, labels=clf.classes_, eps=1e-15))

unique_variations = train_df['Variation'].value_counts()

print('Number of Unique Variations :', unique_variations.shape[0])

# the top 10 variations that occured most

print(unique_variations.head(10))
print("Ans: There are", unique_variations.shape[0] ,"different categories of variations in the train data, and they are distibuted as follows",)
s = sum(unique_variations.values);

h = unique_variations.values/s;

plt.plot(h, label="Histrogram of Variations")

plt.xlabel('Index of a Variation')

plt.ylabel('Number of Occurances')

plt.legend()

plt.grid()

plt.show()
c = np.cumsum(h)

print(c)

plt.plot(c,label='Cumulative distribution of Variations')

plt.grid()

plt.legend()

plt.show()
# # alpha is used for laplace smoothing

# alpha = 1

# # train gene feature

# train_variation_feature_responseCoding = np.array(get_gv_feature(alpha, "Variation", train_df))

# # test gene feature

# test_variation_feature_responseCoding = np.array(get_gv_feature(alpha, "Variation", test_df))

# # cross validation gene feature

# cv_variation_feature_responseCoding = np.array(get_gv_feature(alpha, "Variation", cv_df))
# print("train_variation_feature_responseCoding is a converted feature using the response coding method. The shape of Variation feature:", train_variation_feature_responseCoding.shape)
### one-hot encoding of variation feature.

variation_vectorizer = TfidfVectorizer(max_features=1000)

train_variation_feature_onehotCoding = variation_vectorizer.fit_transform(train_df['Variation'])

test_variation_feature_onehotCoding = variation_vectorizer.transform(test_df['Variation'])

cv_variation_feature_onehotCoding = variation_vectorizer.transform(cv_df['Variation'])
print("train_variation_feature_onehotEncoded is converted feature using the onne-hot encoding method. The shape of Variation feature:", train_variation_feature_onehotCoding.shape)
alpha = [10 ** x for x in range(-5, 1)]



# read more about SGDClassifier() at http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDClassifier.html

# ------------------------------

# default parameters

# SGDClassifier(loss=’hinge’, penalty=’l2’, alpha=0.0001, l1_ratio=0.15, fit_intercept=True, max_iter=None, tol=None, 

# shuffle=True, verbose=0, epsilon=0.1, n_jobs=1, random_state=None, learning_rate=’optimal’, eta0=0.0, power_t=0.5, 

# class_weight=None, warm_start=False, average=False, n_iter=None)



# some of methods

# fit(X, y[, coef_init, intercept_init, …])	Fit linear model with Stochastic Gradient Descent.

# predict(X)	Predict class labels for samples in X.



#-------------------------------

# video link: 

#------------------------------





cv_log_error_array=[]

for i in alpha:

    clf = SGDClassifier(alpha=i, penalty='l2', loss='log', random_state=42)

    clf.fit(train_variation_feature_onehotCoding, y_train)

    

    sig_clf = CalibratedClassifierCV(clf, method="sigmoid")

    sig_clf.fit(train_variation_feature_onehotCoding, y_train)

    predict_y = sig_clf.predict_proba(cv_variation_feature_onehotCoding)

    

    cv_log_error_array.append(log_loss(y_cv, predict_y, labels=clf.classes_, eps=1e-15))

    print('For values of alpha = ', i, "The log loss is:",log_loss(y_cv, predict_y, labels=clf.classes_, eps=1e-15))



fig, ax = plt.subplots()

ax.plot(alpha, cv_log_error_array,c='g')

for i, txt in enumerate(np.round(cv_log_error_array,3)):

    ax.annotate((alpha[i],np.round(txt,3)), (alpha[i],cv_log_error_array[i]))

plt.grid()

plt.title("Cross Validation Error for each alpha")

plt.xlabel("Alpha i's")

plt.ylabel("Error measure")

plt.show()





best_alpha = np.argmin(cv_log_error_array)

clf = SGDClassifier(alpha=alpha[best_alpha], penalty='l2', loss='log', random_state=42)

clf.fit(train_variation_feature_onehotCoding, y_train)

sig_clf = CalibratedClassifierCV(clf, method="sigmoid")

sig_clf.fit(train_variation_feature_onehotCoding, y_train)



predict_y = sig_clf.predict_proba(train_variation_feature_onehotCoding)

print('For values of best alpha = ', alpha[best_alpha], "The train log loss is:",log_loss(y_train, predict_y, labels=clf.classes_, eps=1e-15))

predict_y = sig_clf.predict_proba(cv_variation_feature_onehotCoding)

print('For values of best alpha = ', alpha[best_alpha], "The cross validation log loss is:",log_loss(y_cv, predict_y, labels=clf.classes_, eps=1e-15))

predict_y = sig_clf.predict_proba(test_variation_feature_onehotCoding)

print('For values of best alpha = ', alpha[best_alpha], "The test log loss is:",log_loss(y_test, predict_y, labels=clf.classes_, eps=1e-15))

print("Q12. How many data points are covered by total ", unique_variations.shape[0], " genes in test and cross validation data sets?")

test_coverage=test_df[test_df['Variation'].isin(list(set(train_df['Variation'])))].shape[0]

cv_coverage=cv_df[cv_df['Variation'].isin(list(set(train_df['Variation'])))].shape[0]

print('Ans\n1. In test data',test_coverage, 'out of',test_df.shape[0], ":",(test_coverage/test_df.shape[0])*100)

print('2. In cross validation data',cv_coverage, 'out of ',cv_df.shape[0],":" ,(cv_coverage/cv_df.shape[0])*100)
# one-hot encoding of variation feature.

variation_vectorizer_bigram = CountVectorizer(ngram_range=(1,2))

train_variation_feature_onehotCoding_bigram = variation_vectorizer_bigram.fit_transform(train_df['Variation'])

test_variation_feature_onehotCoding_bigram = variation_vectorizer_bigram.transform(test_df['Variation'])

cv_variation_feature_onehotCoding_bigram = variation_vectorizer_bigram.transform(cv_df['Variation'])
print("train_variation_feature_onehotEncoded is converted feature using the one-hot encoding method (UNI and BIGRAM). The shape of Variation feature:", train_variation_feature_onehotCoding_bigram.shape)
alpha = [10 ** x for x in range(-5, 1)]



# read more about SGDClassifier() at http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDClassifier.html

# ------------------------------

# default parameters

# SGDClassifier(loss=’hinge’, penalty=’l2’, alpha=0.0001, l1_ratio=0.15, fit_intercept=True, max_iter=None, tol=None, 

# shuffle=True, verbose=0, epsilon=0.1, n_jobs=1, random_state=None, learning_rate=’optimal’, eta0=0.0, power_t=0.5, 

# class_weight=None, warm_start=False, average=False, n_iter=None)



# some of methods

# fit(X, y[, coef_init, intercept_init, …])	Fit linear model with Stochastic Gradient Descent.

# predict(X)	Predict class labels for samples in X.



#-------------------------------

# video link: 

#------------------------------





cv_log_error_array=[]

for i in alpha:

    clf = SGDClassifier(alpha=i, penalty='l2', loss='log', random_state=42)

    clf.fit(train_variation_feature_onehotCoding_bigram, y_train)

    

    sig_clf = CalibratedClassifierCV(clf, method="sigmoid")

    sig_clf.fit(train_variation_feature_onehotCoding_bigram, y_train)

    predict_y = sig_clf.predict_proba(cv_variation_feature_onehotCoding_bigram)

    

    cv_log_error_array.append(log_loss(y_cv, predict_y, labels=clf.classes_, eps=1e-15))

    print('For values of alpha = ', i, "The log loss is:",log_loss(y_cv, predict_y, labels=clf.classes_, eps=1e-15))



fig, ax = plt.subplots()

ax.plot(alpha, cv_log_error_array,c='g')

for i, txt in enumerate(np.round(cv_log_error_array,3)):

    ax.annotate((alpha[i],np.round(txt,3)), (alpha[i],cv_log_error_array[i]))

plt.grid()

plt.title("Cross Validation Error for each alpha")

plt.xlabel("Alpha i's")

plt.ylabel("Error measure")

plt.show()





best_alpha = np.argmin(cv_log_error_array)

clf = SGDClassifier(alpha=alpha[best_alpha], penalty='l2', loss='log', random_state=42)

clf.fit(train_variation_feature_onehotCoding_bigram, y_train)

sig_clf = CalibratedClassifierCV(clf, method="sigmoid")

sig_clf.fit(train_variation_feature_onehotCoding_bigram, y_train)



predict_y = sig_clf.predict_proba(train_variation_feature_onehotCoding_bigram)

print('For values of best alpha = ', alpha[best_alpha], "The train log loss is:",log_loss(y_train, predict_y, labels=clf.classes_, eps=1e-15))

predict_y = sig_clf.predict_proba(cv_variation_feature_onehotCoding_bigram)

print('For values of best alpha = ', alpha[best_alpha], "The cross validation log loss is:",log_loss(y_cv, predict_y, labels=clf.classes_, eps=1e-15))

predict_y = sig_clf.predict_proba(test_variation_feature_onehotCoding_bigram)

print('For values of best alpha = ', alpha[best_alpha], "The test log loss is:",log_loss(y_test, predict_y, labels=clf.classes_, eps=1e-15))

# cls_text is a data frame

# for every row in data fram consider the 'TEXT'

# split the words by space

# make a dict with those words

# increment its count whenever we see that word



def extract_dictionary_paddle(cls_text):

    dictionary = defaultdict(int)

    for index, row in cls_text.iterrows():

        for word in row['TEXT'].split():

            dictionary[word] +=1

    return dictionary
# import math

# #https://stackoverflow.com/a/1602964

# def get_text_responsecoding(df):

#     text_feature_responseCoding = np.zeros((df.shape[0],9))

#     for i in range(0,9):

#         row_index = 0

#         for index, row in df.iterrows():

#             sum_prob = 0

#             for word in row['TEXT'].split():

#                 sum_prob += math.log(((dict_list[i].get(word,0)+10 )/(total_dict.get(word,0)+90)))

#             text_feature_responseCoding[row_index][i] = math.exp(sum_prob/len(row['TEXT'].split()))

#             row_index += 1

#     return text_feature_responseCoding
# building a CountVectorizer with all the words that occured minimum 3 times in train data

## text_vectorizer = CountVectorizer(min_df=3)

text_vectorizer = TfidfVectorizer(min_df=3,max_features=1000)

train_text_feature_onehotCoding = text_vectorizer.fit_transform(train_df['TEXT'])

# getting all the feature names (words)

train_text_features= text_vectorizer.get_feature_names()



# train_text_feature_onehotCoding.sum(axis=0).A1 will sum every row and returns (1*number of features) vector

train_text_fea_counts = train_text_feature_onehotCoding.sum(axis=0).A1



# zip(list(text_features),text_fea_counts) will zip a word with its number of times it occured

text_fea_dict = dict(zip(list(train_text_features),train_text_fea_counts))





print("Total number of unique words in train data :", len(train_text_features))
# building a CountVectorizer with all the words that occured minimum 3 times in train data

text_vectorizer_bigram = CountVectorizer(min_df=3,ngram_range=(1,2))

train_text_feature_onehotCoding_bigram = text_vectorizer_bigram.fit_transform(train_df['TEXT'])

# getting all the feature names (words)

train_text_features_bigram= text_vectorizer_bigram.get_feature_names()



# train_text_feature_onehotCoding.sum(axis=0).A1 will sum every row and returns (1*number of features) vector

train_text_fea_counts_bigram = train_text_feature_onehotCoding_bigram.sum(axis=0).A1



# zip(list(text_features),text_fea_counts) will zip a word with its number of times it occured

text_fea_dict_bigram = dict(zip(list(train_text_features_bigram),train_text_fea_counts_bigram))





print("Total number of unique words in train data :", len(train_text_features_bigram))
dict_list = []

# dict_list =[] contains 9 dictoinaries each corresponds to a class

for i in range(1,10):

    cls_text = train_df[train_df['Class']==i]

    # build a word dict based on the words in that class

    dict_list.append(extract_dictionary_paddle(cls_text))

    # append it to dict_list



# dict_list[i] is build on i'th  class text data

# total_dict is buid on whole training text data

total_dict = extract_dictionary_paddle(train_df)





confuse_array = []

for i in train_text_features:

    ratios = []

    max_val = -1

    for j in range(0,9):

        ratios.append((dict_list[j][i]+10 )/(total_dict[i]+90))

    confuse_array.append(ratios)

confuse_array = np.array(confuse_array)
# #response coding of text features

# train_text_feature_responseCoding  = get_text_responsecoding(train_df)

# test_text_feature_responseCoding  = get_text_responsecoding(test_df)

# cv_text_feature_responseCoding  = get_text_responsecoding(cv_df)
# # https://stackoverflow.com/a/16202486

# # we convert each row values such that they sum to 1  

# train_text_feature_responseCoding = (train_text_feature_responseCoding.T/train_text_feature_responseCoding.sum(axis=1)).T

# test_text_feature_responseCoding = (test_text_feature_responseCoding.T/test_text_feature_responseCoding.sum(axis=1)).T

# cv_text_feature_responseCoding = (cv_text_feature_responseCoding.T/cv_text_feature_responseCoding.sum(axis=1)).T
# don't forget to normalize every feature

train_text_feature_onehotCoding = normalize(train_text_feature_onehotCoding, axis=0)



# we use the same vectorizer that was trained on train data

test_text_feature_onehotCoding = text_vectorizer.transform(test_df['TEXT'])

# don't forget to normalize every feature

test_text_feature_onehotCoding = normalize(test_text_feature_onehotCoding, axis=0)



# we use the same vectorizer that was trained on train data

cv_text_feature_onehotCoding = text_vectorizer.transform(cv_df['TEXT'])

# don't forget to normalize every feature

cv_text_feature_onehotCoding = normalize(cv_text_feature_onehotCoding, axis=0)
####
# don't forget to normalize every feature

train_text_feature_onehotCoding_bigram = normalize(train_text_feature_onehotCoding_bigram, axis=0)



# we use the same vectorizer that was trained on train data

test_text_feature_onehotCoding_bigram = text_vectorizer_bigram.transform(test_df['TEXT'])

# don't forget to normalize every feature

test_text_feature_onehotCoding_bigram = normalize(test_text_feature_onehotCoding_bigram, axis=0)



# we use the same vectorizer that was trained on train data

cv_text_feature_onehotCoding_bigram = text_vectorizer_bigram.transform(cv_df['TEXT'])

# don't forget to normalize every feature

cv_text_feature_onehotCoding_bigram = normalize(cv_text_feature_onehotCoding_bigram, axis=0)
#https://stackoverflow.com/a/2258273/4084039

sorted_text_fea_dict = dict(sorted(text_fea_dict.items(), key=lambda x: x[1] , reverse=True))

sorted_text_occur = np.array(list(sorted_text_fea_dict.values()))
# Number of words for a given frequency.

print(Counter(sorted_text_occur))
# Train a Logistic regression+Calibration model using text features whicha re on-hot encoded

alpha = [10 ** x for x in range(-5, 1)]



# read more about SGDClassifier() at http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDClassifier.html

# ------------------------------

# default parameters

# SGDClassifier(loss=’hinge’, penalty=’l2’, alpha=0.0001, l1_ratio=0.15, fit_intercept=True, max_iter=None, tol=None, 

# shuffle=True, verbose=0, epsilon=0.1, n_jobs=1, random_state=None, learning_rate=’optimal’, eta0=0.0, power_t=0.5, 

# class_weight=None, warm_start=False, average=False, n_iter=None)



# some of methods

# fit(X, y[, coef_init, intercept_init, …])	Fit linear model with Stochastic Gradient Descent.

# predict(X)	Predict class labels for samples in X.



#-------------------------------

# video link: 

#------------------------------





cv_log_error_array=[]

for i in alpha:

    clf = SGDClassifier(alpha=i, penalty='l2', loss='log', random_state=42)

    clf.fit(train_text_feature_onehotCoding, y_train)

    

    sig_clf = CalibratedClassifierCV(clf, method="sigmoid")

    sig_clf.fit(train_text_feature_onehotCoding, y_train)

    predict_y = sig_clf.predict_proba(cv_text_feature_onehotCoding)

    cv_log_error_array.append(log_loss(y_cv, predict_y, labels=clf.classes_, eps=1e-15))

    print('For values of alpha = ', i, "The log loss is:",log_loss(y_cv, predict_y, labels=clf.classes_, eps=1e-15))



fig, ax = plt.subplots()

ax.plot(alpha, cv_log_error_array,c='g')

for i, txt in enumerate(np.round(cv_log_error_array,3)):

    ax.annotate((alpha[i],np.round(txt,3)), (alpha[i],cv_log_error_array[i]))

plt.grid()

plt.title("Cross Validation Error for each alpha")

plt.xlabel("Alpha i's")

plt.ylabel("Error measure")

plt.show()





best_alpha = np.argmin(cv_log_error_array)

clf = SGDClassifier(alpha=alpha[best_alpha], penalty='l2', loss='log', random_state=42)

clf.fit(train_text_feature_onehotCoding, y_train)

sig_clf = CalibratedClassifierCV(clf, method="sigmoid")

sig_clf.fit(train_text_feature_onehotCoding, y_train)



predict_y = sig_clf.predict_proba(train_text_feature_onehotCoding)

print('For values of best alpha = ', alpha[best_alpha], "The train log loss is:",log_loss(y_train, predict_y, labels=clf.classes_, eps=1e-15))

predict_y = sig_clf.predict_proba(cv_text_feature_onehotCoding)

print('For values of best alpha = ', alpha[best_alpha], "The cross validation log loss is:",log_loss(y_cv, predict_y, labels=clf.classes_, eps=1e-15))

predict_y = sig_clf.predict_proba(test_text_feature_onehotCoding)

print('For values of best alpha = ', alpha[best_alpha], "The test log loss is:",log_loss(y_test, predict_y, labels=clf.classes_, eps=1e-15))

def get_intersec_text(df):

#     df_text_vec = CountVectorizer(min_df=3)

    df_text_vec = TfidfVectorizer(min_df=3,max_features=1000)

    df_text_fea = df_text_vec.fit_transform(df['TEXT'])

    df_text_features = df_text_vec.get_feature_names()



    df_text_fea_counts = df_text_fea.sum(axis=0).A1

    df_text_fea_dict = dict(zip(list(df_text_features),df_text_fea_counts))

    len1 = len(set(df_text_features))

    len2 = len(set(train_text_features) & set(df_text_features))

    return len1,len2
len1,len2 = get_intersec_text(test_df)

print(np.round((len2/len1)*100, 3), "% of word of test data appeared in train data")

len1,len2 = get_intersec_text(cv_df)

print(np.round((len2/len1)*100, 3), "% of word of Cross Validation appeared in train data")
alpha = [10 ** x for x in range(-5, 1)]



cv_log_error_array=[]

for i in alpha:

    clf = SGDClassifier(alpha=i, penalty='l2', loss='log', random_state=42)

    clf.fit(train_text_feature_onehotCoding_bigram, y_train)

    

    sig_clf = CalibratedClassifierCV(clf, method="sigmoid")

    sig_clf.fit(train_text_feature_onehotCoding_bigram, y_train)

    predict_y = sig_clf.predict_proba(cv_text_feature_onehotCoding_bigram)

    cv_log_error_array.append(log_loss(y_cv, predict_y, labels=clf.classes_, eps=1e-15))

    print('For values of alpha = ', i, "The log loss is:",log_loss(y_cv, predict_y, labels=clf.classes_, eps=1e-15))



fig, ax = plt.subplots()

ax.plot(alpha, cv_log_error_array,c='g')

for i, txt in enumerate(np.round(cv_log_error_array,3)):

    ax.annotate((alpha[i],np.round(txt,3)), (alpha[i],cv_log_error_array[i]))

plt.grid()

plt.title("Cross Validation Error for each alpha")

plt.xlabel("Alpha i's")

plt.ylabel("Error measure")

plt.show()





best_alpha = np.argmin(cv_log_error_array)

clf = SGDClassifier(alpha=alpha[best_alpha], penalty='l2', loss='log', random_state=42)

clf.fit(train_text_feature_onehotCoding_bigram, y_train)

sig_clf = CalibratedClassifierCV(clf, method="sigmoid")

sig_clf.fit(train_text_feature_onehotCoding_bigram, y_train)



predict_y = sig_clf.predict_proba(train_text_feature_onehotCoding_bigram)

print('For values of best alpha = ', alpha[best_alpha], "The train log loss is:",log_loss(y_train, predict_y, labels=clf.classes_, eps=1e-15))

predict_y = sig_clf.predict_proba(cv_text_feature_onehotCoding_bigram)

print('For values of best alpha = ', alpha[best_alpha], "The cross validation log loss is:",log_loss(y_cv, predict_y, labels=clf.classes_, eps=1e-15))

predict_y = sig_clf.predict_proba(test_text_feature_onehotCoding_bigram)

print('For values of best alpha = ', alpha[best_alpha], "The test log loss is:",log_loss(y_test, predict_y, labels=clf.classes_, eps=1e-15))

def get_intersec_text(df):

    df_text_vec_bigram = TfidfVectorizer(min_df=3,ngram_range=(1,2))

    df_text_fea_bigram = df_text_vec_bigram.fit_transform(df['TEXT'])

    df_text_features_bigram = df_text_vec_bigram.get_feature_names()



    df_text_fea_counts_bigram = df_text_fea_bigram.sum(axis=0).A1

    df_text_fea_dict_bigram = dict(zip(list(df_text_features_bigram),df_text_fea_counts_bigram))

    len1 = len(set(df_text_features_bigram))

    len2 = len(set(train_text_features_bigram) & set(df_text_features_bigram))

    return len1,len2
len1,len2 = get_intersec_text(test_df)

print(np.round((len2/len1)*100, 3), "% of word of test data appeared in train data")

len1,len2 = get_intersec_text(cv_df)

print(np.round((len2/len1)*100, 3), "% of word of Cross Validation appeared in train data")
#Data preparation for ML models.



#Misc. functionns for ML models





def predict_and_plot_confusion_matrix(train_x, train_y,test_x, test_y, clf):

    clf.fit(train_x, train_y)

    sig_clf = CalibratedClassifierCV(clf, method="sigmoid")

    sig_clf.fit(train_x, train_y)

    pred_y = sig_clf.predict(test_x)



    # for calculating log_loss we willl provide the array of probabilities belongs to each class

    logLoss=log_loss(test_y, sig_clf.predict_proba(test_x))

    print("Log loss :",logLoss)

    # calculating the number of data points that are misclassified

    missClassified=np.count_nonzero((pred_y- test_y))/test_y.shape[0]

    print("Number of mis-classified points :", missClassified)

#     plot_confusion_matrix2(test_y, pred_y,logLoss,missClassified)

    plot_confusion_matrix(test_y, pred_y)

    return missClassified
def report_log_loss(train_x, train_y, test_x, test_y,  clf):

    clf.fit(train_x, train_y)

    sig_clf = CalibratedClassifierCV(clf, method="sigmoid")

    sig_clf.fit(train_x, train_y)

    sig_clf_probs = sig_clf.predict_proba(test_x)

    return log_loss(test_y, sig_clf_probs, eps=1e-15)
# this function will be used just for naive bayes

# for the given indices, we will print the name of the features

# and we will check whether the feature present in the test point text or not

def get_impfeature_names(indices, text, gene, var, no_features):

#     gene_count_vec = CountVectorizer()

#     var_count_vec = CountVectorizer()

#     text_count_vec = CountVectorizer(min_df=3)

    gene_count_vec = TfidfVectorizer(max_features=1000)

    var_count_vec = TfidfVectorizer(max_features=1000)

    text_count_vec = TfidfVectorizer(min_df=3,max_features=1000)

    

    gene_vec = gene_count_vec.fit(train_df['Gene'])

    var_vec  = var_count_vec.fit(train_df['Variation'])

    text_vec = text_count_vec.fit(train_df['TEXT'])

    

    fea1_len = len(gene_vec.get_feature_names())

    fea2_len = len(var_count_vec.get_feature_names())

    

    word_present = 0

    for i,v in enumerate(indices):

        if (v < fea1_len):

            word = gene_vec.get_feature_names()[v]

            yes_no = True if word == gene else False

            if yes_no:

                word_present += 1

                print(i, "Gene feature [{}] present in test data point [{}]".format(word,yes_no))

        elif (v < fea1_len+fea2_len):

            word = var_vec.get_feature_names()[v-(fea1_len)]

            yes_no = True if word == var else False

            if yes_no:

                word_present += 1

                print(i, "variation feature [{}] present in test data point [{}]".format(word,yes_no))

        else:

            word = text_vec.get_feature_names()[v-(fea1_len+fea2_len)]

            yes_no = True if word in text.split() else False

            if yes_no:

                word_present += 1

                print(i, "Text feature [{}] present in test data point [{}]".format(word,yes_no))



    print("Out of the top ",no_features," features ", word_present, "are present in query point")
# merging gene, variance and text features



# building train, test and cross validation data sets

# a = [[1, 2], 

#      [3, 4]]

# b = [[4, 5], 

#      [6, 7]]

# hstack(a, b) = [[1, 2, 4, 5],

#                [ 3, 4, 6, 7]]



train_gene_var_onehotCoding = hstack((train_gene_feature_onehotCoding,train_variation_feature_onehotCoding))

test_gene_var_onehotCoding = hstack((test_gene_feature_onehotCoding,test_variation_feature_onehotCoding))

cv_gene_var_onehotCoding = hstack((cv_gene_feature_onehotCoding,cv_variation_feature_onehotCoding))



train_x_onehotCoding = hstack((train_gene_var_onehotCoding, train_text_feature_onehotCoding)).tocsr()

train_y = np.array(list(train_df['Class']))



test_x_onehotCoding = hstack((test_gene_var_onehotCoding, test_text_feature_onehotCoding)).tocsr()

test_y = np.array(list(test_df['Class']))



cv_x_onehotCoding = hstack((cv_gene_var_onehotCoding, cv_text_feature_onehotCoding)).tocsr()

cv_y = np.array(list(cv_df['Class']))





# train_gene_var_responseCoding = np.hstack((train_gene_feature_responseCoding,train_variation_feature_responseCoding))

# test_gene_var_responseCoding = np.hstack((test_gene_feature_responseCoding,test_variation_feature_responseCoding))

# cv_gene_var_responseCoding = np.hstack((cv_gene_feature_responseCoding,cv_variation_feature_responseCoding))



# train_x_responseCoding = np.hstack((train_gene_var_responseCoding, train_text_feature_responseCoding))

# test_x_responseCoding = np.hstack((test_gene_var_responseCoding, test_text_feature_responseCoding))

# cv_x_responseCoding = np.hstack((cv_gene_var_responseCoding, cv_text_feature_responseCoding))

## for count vectorizer

train_x_onehotCoding_count = hstack((train_gene_feature_onehotCoding_onehot,train_variation_feature_onehotCoding_bigram, train_text_feature_onehotCoding_bigram)).tocsr()

test_x_onehotCoding_count = hstack((test_gene_feature_onehotCoding_onehot,test_variation_feature_onehotCoding_bigram, test_text_feature_onehotCoding_bigram)).tocsr()

cv_x_onehotCoding_count = hstack((cv_gene_feature_onehotCoding_onehot,cv_variation_feature_onehotCoding_bigram, cv_text_feature_onehotCoding_bigram)).tocsr()
print("One hot encoding features :")

print("(number of data points * number of features) in train data = ", train_x_onehotCoding.shape)

print("(number of data points * number of features) in test data = ", test_x_onehotCoding.shape)

print("(number of data points * number of features) in cross validation data =", cv_x_onehotCoding.shape)
print("One hot encoding COUNT VECTORIZER features :")

print("(number of data points * number of features) in train data = ", train_x_onehotCoding_count.shape)

print("(number of data points * number of features) in test data = ", test_x_onehotCoding_count.shape)

print("(number of data points * number of features) in cross validation data =", cv_x_onehotCoding_count.shape)
# print(" Response encoding features :")

# print("(number of data points * number of features) in train data = ", train_x_responseCoding.shape)

# print("(number of data points * number of features) in test data = ", test_x_responseCoding.shape)

# print("(number of data points * number of features) in cross validation data =", cv_x_responseCoding.shape)
# find more about Multinomial Naive base function here http://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.MultinomialNB.html

# -------------------------

# default paramters

# sklearn.naive_bayes.MultinomialNB(alpha=1.0, fit_prior=True, class_prior=None)



# some of methods of MultinomialNB()

# fit(X, y[, sample_weight])	Fit Naive Bayes classifier according to X, y

# predict(X)	Perform classification on an array of test vectors X.

# predict_log_proba(X)	Return log-probability estimates for the test vector X.

# -----------------------

# video link: https://www.appliedaicourse.com/course/applied-ai-course-online/lessons/naive-bayes-algorithm-1/

# -----------------------





# find more about CalibratedClassifierCV here at http://scikit-learn.org/stable/modules/generated/sklearn.calibration.CalibratedClassifierCV.html

# ----------------------------

# default paramters

# sklearn.calibration.CalibratedClassifierCV(base_estimator=None, method=’sigmoid’, cv=3)

#

# some of the methods of CalibratedClassifierCV()

# fit(X, y[, sample_weight])	Fit the calibrated model

# get_params([deep])	Get parameters for this estimator.

# predict(X)	Predict the target of new samples.

# predict_proba(X)	Posterior probabilities of classification

# ----------------------------

# video link: https://www.appliedaicourse.com/course/applied-ai-course-online/lessons/naive-bayes-algorithm-1/

# -----------------------





alpha = [0.00001, 0.0001, 0.001, 0.1, 1, 10, 100,1000]

cv_log_error_array = []

for i in alpha:

    print("for alpha =", i)

    clf = MultinomialNB(alpha=i)

    clf.fit(train_x_onehotCoding, train_y)

    sig_clf = CalibratedClassifierCV(clf, method="sigmoid")

    sig_clf.fit(train_x_onehotCoding, train_y)

    sig_clf_probs = sig_clf.predict_proba(cv_x_onehotCoding)

    cv_log_error_array.append(log_loss(cv_y, sig_clf_probs, labels=clf.classes_, eps=1e-15))

    # to avoid rounding error while multiplying probabilites we use log-probability estimates

    print("Log Loss :",log_loss(cv_y, sig_clf_probs)) 



fig, ax = plt.subplots()

ax.plot(np.log10(alpha), cv_log_error_array,c='g')

for i, txt in enumerate(np.round(cv_log_error_array,3)):

    ax.annotate((alpha[i],str(txt)), (np.log10(alpha[i]),cv_log_error_array[i]))

plt.grid()

plt.xticks(np.log10(alpha))

plt.title("Cross Validation Error for each alpha")

plt.xlabel("Alpha i's")

plt.ylabel("Error measure")

plt.show()





best_alpha = np.argmin(cv_log_error_array)

clf = MultinomialNB(alpha=alpha[best_alpha])

clf.fit(train_x_onehotCoding, train_y)

sig_clf = CalibratedClassifierCV(clf, method="sigmoid")

sig_clf.fit(train_x_onehotCoding, train_y)





predict_y = sig_clf.predict_proba(train_x_onehotCoding)

nbLoss_train=log_loss(y_train, predict_y, labels=clf.classes_, eps=1e-15)

print('For values of best alpha = ', alpha[best_alpha], "The train log loss is:",nbLoss_train)



predict_y = sig_clf.predict_proba(cv_x_onehotCoding)

nbLoss_cv=log_loss(y_cv, predict_y, labels=clf.classes_, eps=1e-15)

print('For values of best alpha = ', alpha[best_alpha], "The cross validation log loss is:",nbLoss_cv)



predict_y = sig_clf.predict_proba(test_x_onehotCoding)

nbLoss_test=log_loss(y_test, predict_y, labels=clf.classes_, eps=1e-15)

print('For values of best alpha = ', alpha[best_alpha], "The test log loss is:",nbLoss_test)

print(nbLoss_train)

print(nbLoss_cv)

print(nbLoss_test)
## WITHOUT CALIBRATION





alpha = [0.00001, 0.0001, 0.001, 0.1, 1, 10, 100,1000]

cv_log_error_array = []

for i in alpha:

    print("for alpha =", i)

    clf = MultinomialNB(alpha=i)

    clf.fit(train_x_onehotCoding, train_y)

#     sig_clf = CalibratedClassifierCV(clf, method="sigmoid")

#     sig_clf.fit(train_x_onehotCoding, train_y)

#     sig_clf_probs = sig_clf.predict_proba(cv_x_onehotCoding)

    sig_clf_probs = clf.predict_proba(cv_x_onehotCoding)

    cv_log_error_array.append(log_loss(cv_y, sig_clf_probs, labels=clf.classes_, eps=1e-15))

    # to avoid rounding error while multiplying probabilites we use log-probability estimates

    print("Log Loss :",log_loss(cv_y, sig_clf_probs)) 



fig, ax = plt.subplots()

ax.plot(np.log10(alpha), cv_log_error_array,c='g')

for i, txt in enumerate(np.round(cv_log_error_array,3)):

    ax.annotate((alpha[i],str(txt)), (np.log10(alpha[i]),cv_log_error_array[i]))

plt.grid()

plt.xticks(np.log10(alpha))

plt.title("Cross Validation Error for each alpha")

plt.xlabel("Alpha i's")

plt.ylabel("Error measure")

plt.show()





best_alpha = np.argmin(cv_log_error_array)

clf = MultinomialNB(alpha=alpha[best_alpha])

clf.fit(train_x_onehotCoding, train_y)

# sig_clf = CalibratedClassifierCV(clf, method="sigmoid")

# sig_clf.fit(train_x_onehotCoding, train_y)





predict_y = clf.predict_proba(train_x_onehotCoding)

print('For values of best alpha = ', alpha[best_alpha], "The train log loss is:",log_loss(y_train, predict_y, labels=clf.classes_, eps=1e-15))

predict_y = clf.predict_proba(cv_x_onehotCoding)

print('For values of best alpha = ', alpha[best_alpha], "The cross validation log loss is:",log_loss(y_cv, predict_y, labels=clf.classes_, eps=1e-15))

predict_y = clf.predict_proba(test_x_onehotCoding)

print('For values of best alpha = ', alpha[best_alpha], "The test log loss is:",log_loss(y_test, predict_y, labels=clf.classes_, eps=1e-15))

# find more about Multinomial Naive base function here http://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.MultinomialNB.html

# -------------------------

# default paramters

# sklearn.naive_bayes.MultinomialNB(alpha=1.0, fit_prior=True, class_prior=None)



# some of methods of MultinomialNB()

# fit(X, y[, sample_weight])	Fit Naive Bayes classifier according to X, y

# predict(X)	Perform classification on an array of test vectors X.

# predict_log_proba(X)	Return log-probability estimates for the test vector X.

# -----------------------

# video link: https://www.appliedaicourse.com/course/applied-ai-course-online/lessons/naive-bayes-algorithm-1/

# -----------------------





# find more about CalibratedClassifierCV here at http://scikit-learn.org/stable/modules/generated/sklearn.calibration.CalibratedClassifierCV.html

# ----------------------------

# default paramters

# sklearn.calibration.CalibratedClassifierCV(base_estimator=None, method=’sigmoid’, cv=3)

#

# some of the methods of CalibratedClassifierCV()

# fit(X, y[, sample_weight])	Fit the calibrated model

# get_params([deep])	Get parameters for this estimator.

# predict(X)	Predict the target of new samples.

# predict_proba(X)	Posterior probabilities of classification

# ----------------------------



clf = MultinomialNB(alpha=alpha[best_alpha])

clf.fit(train_x_onehotCoding, train_y)

sig_clf = CalibratedClassifierCV(clf, method="sigmoid")

sig_clf.fit(train_x_onehotCoding, train_y)

sig_clf_probs = sig_clf.predict_proba(cv_x_onehotCoding)

# to avoid rounding error while multiplying probabilites we use log-probability estimates

nbloss_test=log_loss(cv_y, sig_clf_probs)

nbmp=np.count_nonzero((sig_clf.predict(cv_x_onehotCoding)- cv_y))/cv_y.shape[0]

print("Log Loss :",nbloss)

print("Number of missclassified point :", nbmp)

plot_confusion_matrix(cv_y, sig_clf.predict(cv_x_onehotCoding.toarray()))
test_point_index = 1

no_feature = 100

predicted_cls = sig_clf.predict(test_x_onehotCoding[test_point_index])

print("Predicted Class :", predicted_cls[0])

print("Predicted Class Probabilities:", np.round(sig_clf.predict_proba(test_x_onehotCoding[test_point_index]),4))

print("Actual Class :", test_y[test_point_index])

indices=np.argsort(abs(-clf.coef_))[predicted_cls-1][:,:no_feature]

print("-"*50)

get_impfeature_names(indices[0], test_df['TEXT'].iloc[test_point_index],test_df['Gene'].iloc[test_point_index],test_df['Variation'].iloc[test_point_index], no_feature)
test_point_index = 100

no_feature = 100

predicted_cls = sig_clf.predict(test_x_onehotCoding[test_point_index])

print("Predicted Class :", predicted_cls[0])

print("Predicted Class Probabilities:", np.round(sig_clf.predict_proba(test_x_onehotCoding[test_point_index]),4))

print("Actual Class :", test_y[test_point_index])

indices = np.argsort(abs(-clf.coef_))[predicted_cls-1][:,:no_feature]

print("-"*50)

get_impfeature_names(indices[0], test_df['TEXT'].iloc[test_point_index],test_df['Gene'].iloc[test_point_index],test_df['Variation'].iloc[test_point_index], no_feature)
# find more about KNeighborsClassifier() here http://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html

# -------------------------

# default parameter

# KNeighborsClassifier(n_neighbors=5, weights=’uniform’, algorithm=’auto’, leaf_size=30, p=2, 

# metric=’minkowski’, metric_params=None, n_jobs=1, **kwargs)



# methods of

# fit(X, y) : Fit the model using X as training data and y as target values

# predict(X):Predict the class labels for the provided data

# predict_proba(X):Return probability estimates for the test data X.

#-------------------------------------

# video link: https://www.appliedaicourse.com/course/applied-ai-course-online/lessons/k-nearest-neighbors-geometric-intuition-with-a-toy-example-1/

#-------------------------------------





# find more about CalibratedClassifierCV here at http://scikit-learn.org/stable/modules/generated/sklearn.calibration.CalibratedClassifierCV.html

# ----------------------------

# default paramters

# sklearn.calibration.CalibratedClassifierCV(base_estimator=None, method=’sigmoid’, cv=3)

#

# some of the methods of CalibratedClassifierCV()

# fit(X, y[, sample_weight])	Fit the calibrated model

# get_params([deep])	Get parameters for this estimator.

# predict(X)	Predict the target of new samples.

# predict_proba(X)	Posterior probabilities of classification

#-------------------------------------

# video link:

#-------------------------------------





alpha = [5, 11, 15, 21, 31, 41, 51, 99]

cv_log_error_array = []

for i in alpha:

    print("for alpha =", i)

    clf = KNeighborsClassifier(n_neighbors=i)

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

clf = KNeighborsClassifier(n_neighbors=alpha[best_alpha])

clf.fit(train_x_onehotCoding, train_y)

sig_clf = CalibratedClassifierCV(clf, method="sigmoid")

sig_clf.fit(train_x_onehotCoding, train_y)



predict_y = sig_clf.predict_proba(train_x_onehotCoding)

knnLoss_train=log_loss(y_train, predict_y, labels=clf.classes_, eps=1e-15)

print('For values of best alpha = ', alpha[best_alpha], "The train log loss is:",knnLoss_train)

predict_y = sig_clf.predict_proba(cv_x_onehotCoding)

knnLoss_cv=log_loss(y_cv, predict_y, labels=clf.classes_, eps=1e-15)

print('For values of best alpha = ', alpha[best_alpha], "The cross validation log loss is:",knnLoss_cv)

predict_y = sig_clf.predict_proba(test_x_onehotCoding)

knnLoss_test=log_loss(y_test, predict_y, labels=clf.classes_, eps=1e-15)

print('For values of best alpha = ', alpha[best_alpha], "The test log loss is:",knnLoss_test)

# find more about KNeighborsClassifier() here http://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html

# -------------------------

# default parameter

# KNeighborsClassifier(n_neighbors=5, weights=’uniform’, algorithm=’auto’, leaf_size=30, p=2, 

# metric=’minkowski’, metric_params=None, n_jobs=1, **kwargs)



# methods of

# fit(X, y) : Fit the model using X as training data and y as target values

# predict(X):Predict the class labels for the provided data

# predict_proba(X):Return probability estimates for the test data X.

#-------------------------------------

# video link: https://www.appliedaicourse.com/course/applied-ai-course-online/lessons/k-nearest-neighbors-geometric-intuition-with-a-toy-example-1/

#-------------------------------------

clf = KNeighborsClassifier(n_neighbors=alpha[best_alpha])

knnmp=predict_and_plot_confusion_matrix(train_x_onehotCoding, train_y, cv_x_onehotCoding, cv_y, clf)
clf = KNeighborsClassifier(n_neighbors=alpha[best_alpha])

clf.fit(train_x_onehotCoding, train_y)

sig_clf = CalibratedClassifierCV(clf, method="sigmoid")

sig_clf.fit(train_x_onehotCoding, train_y)



test_point_index = 1

predicted_cls = sig_clf.predict(test_x_onehotCoding[0].reshape(1,-1))

print("Predicted Class :", predicted_cls[0])

print("Actual Class :", test_y[test_point_index])

neighbors = clf.kneighbors(test_x_onehotCoding[test_point_index].reshape(1, -1), alpha[best_alpha])

print("The ",alpha[best_alpha]," nearest neighbours of the test points belongs to classes",train_y[neighbors[1][0]])

print("Fequency of nearest points :",Counter(train_y[neighbors[1][0]]))
clf = KNeighborsClassifier(n_neighbors=alpha[best_alpha])

clf.fit(train_x_onehotCoding, train_y)

sig_clf = CalibratedClassifierCV(clf, method="sigmoid")

sig_clf.fit(train_x_onehotCoding, train_y)



test_point_index = 100



predicted_cls = sig_clf.predict(test_x_onehotCoding[test_point_index].reshape(1,-1))

print("Predicted Class :", predicted_cls[0])

print("Actual Class :", test_y[test_point_index])

neighbors = clf.kneighbors(test_x_onehotCoding[test_point_index].reshape(1, -1), alpha[best_alpha])

print("the k value for knn is",alpha[best_alpha],"and the nearest neighbours of the test points belongs to classes",train_y[neighbors[1][0]])

print("Fequency of nearest points :",Counter(train_y[neighbors[1][0]]))


# read more about SGDClassifier() at http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDClassifier.html

# ------------------------------

# default parameters

# SGDClassifier(loss=’hinge’, penalty=’l2’, alpha=0.0001, l1_ratio=0.15, fit_intercept=True, max_iter=None, tol=None, 

# shuffle=True, verbose=0, epsilon=0.1, n_jobs=1, random_state=None, learning_rate=’optimal’, eta0=0.0, power_t=0.5, 

# class_weight=None, warm_start=False, average=False, n_iter=None)



# some of methods

# fit(X, y[, coef_init, intercept_init, …])	Fit linear model with Stochastic Gradient Descent.

# predict(X)	Predict class labels for samples in X.



#-------------------------------

# video link: https://www.appliedaicourse.com/course/applied-ai-course-online/lessons/geometric-intuition-1/

#------------------------------





# find more about CalibratedClassifierCV here at http://scikit-learn.org/stable/modules/generated/sklearn.calibration.CalibratedClassifierCV.html

# ----------------------------

# default paramters

# sklearn.calibration.CalibratedClassifierCV(base_estimator=None, method=’sigmoid’, cv=3)

#

# some of the methods of CalibratedClassifierCV()

# fit(X, y[, sample_weight])	Fit the calibrated model

# get_params([deep])	Get parameters for this estimator.

# predict(X)	Predict the target of new samples.

# predict_proba(X)	Posterior probabilities of classification

#-------------------------------------

# video link:

#-------------------------------------



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

lrLossClassBalance_tfidf_train=log_loss(y_train, predict_y, labels=clf.classes_, eps=1e-15)

print('For values of best alpha = ', alpha[best_alpha], "The train log loss is:",lrLossClassBalance_tfidf_train)



predict_y = sig_clf.predict_proba(cv_x_onehotCoding)

lrLossClassBalance_tfidf_cv=log_loss(y_cv, predict_y, labels=clf.classes_, eps=1e-15)

print('For values of best alpha = ', alpha[best_alpha], "The cross validation log loss is:",lrLossClassBalance_tfidf_cv)



predict_y = sig_clf.predict_proba(test_x_onehotCoding)

lrLossClassBalance_tfidf_test=log_loss(y_test, predict_y, labels=clf.classes_, eps=1e-15)

print('For values of best alpha = ', alpha[best_alpha], "The test log loss is:",lrLossClassBalance_tfidf_test)
print("lrLossClassBalance_tfidf_train",lrLossClassBalance_tfidf_train)

print("lrLossClassBalance_tfidf_cv",lrLossClassBalance_tfidf_cv)

print("lrLossClassBalance_tfidf_test",lrLossClassBalance_tfidf_test)
# read more about SGDClassifier() at http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDClassifier.html

# ------------------------------

# default parameters

# SGDClassifier(loss=’hinge’, penalty=’l2’, alpha=0.0001, l1_ratio=0.15, fit_intercept=True, max_iter=None, tol=None, 

# shuffle=True, verbose=0, epsilon=0.1, n_jobs=1, random_state=None, learning_rate=’optimal’, eta0=0.0, power_t=0.5, 

# class_weight=None, warm_start=False, average=False, n_iter=None)



# some of methods

# fit(X, y[, coef_init, intercept_init, …])	Fit linear model with Stochastic Gradient Descent.

# predict(X)	Predict class labels for samples in X.



#-------------------------------

# video link: https://www.appliedaicourse.com/course/applied-ai-course-online/lessons/geometric-intuition-1/

#------------------------------

clf = SGDClassifier(class_weight='balanced', alpha=alpha[best_alpha], penalty='l2', loss='log', random_state=42)

lrClassBalancemp=predict_and_plot_confusion_matrix(train_x_onehotCoding, train_y, cv_x_onehotCoding, cv_y, clf)
def get_imp_feature_names(text, indices, removed_ind = []):

    word_present = 0

    tabulte_list = []

    incresingorder_ind = 0

    for i in indices:

        if i < train_gene_feature_onehotCoding.shape[1]:

            tabulte_list.append([incresingorder_ind, "Gene", "Yes"])

        elif i< 18:

            tabulte_list.append([incresingorder_ind,"Variation", "Yes"])

        if ((i > 17) & (i not in removed_ind)) :

            word = train_text_features[i]

            yes_no = True if word in text.split() else False

            if yes_no:

                word_present += 1

            tabulte_list.append([incresingorder_ind,train_text_features[i], yes_no])

        incresingorder_ind += 1

    print(word_present, "most importent features are present in our query point")

    print("-"*50)

    print("The features that are most importent of the ",predicted_cls[0]," class:")

    print (tabulate(tabulte_list, headers=["Index",'Feature name', 'Present or Not']))
# from tabulate import tabulate

clf = SGDClassifier(class_weight='balanced', alpha=alpha[best_alpha], penalty='l2', loss='log', random_state=42)

clf.fit(train_x_onehotCoding,train_y)

test_point_index = 1

no_feature = 500

predicted_cls = sig_clf.predict(test_x_onehotCoding[test_point_index])

print("Predicted Class :", predicted_cls[0])

print("Predicted Class Probabilities:", np.round(sig_clf.predict_proba(test_x_onehotCoding[test_point_index]),4))

print("Actual Class :", test_y[test_point_index])

indices = np.argsort(abs(-clf.coef_))[predicted_cls-1][:,:no_feature]

print("-"*50)

get_impfeature_names(indices[0], test_df['TEXT'].iloc[test_point_index],test_df['Gene'].iloc[test_point_index],test_df['Variation'].iloc[test_point_index], no_feature)
test_point_index = 100

no_feature = 500

predicted_cls = sig_clf.predict(test_x_onehotCoding[test_point_index])

print("Predicted Class :", predicted_cls[0])

print("Predicted Class Probabilities:", np.round(sig_clf.predict_proba(test_x_onehotCoding[test_point_index]),4))

print("Actual Class :", test_y[test_point_index])

indices = np.argsort(abs(-clf.coef_))[predicted_cls-1][:,:no_feature]

print("-"*50)

get_impfeature_names(indices[0], test_df['TEXT'].iloc[test_point_index],test_df['Gene'].iloc[test_point_index],test_df['Variation'].iloc[test_point_index], no_feature)
# read more about SGDClassifier() at http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDClassifier.html

# ------------------------------

# default parameters

# SGDClassifier(loss=’hinge’, penalty=’l2’, alpha=0.0001, l1_ratio=0.15, fit_intercept=True, max_iter=None, tol=None, 

# shuffle=True, verbose=0, epsilon=0.1, n_jobs=1, random_state=None, learning_rate=’optimal’, eta0=0.0, power_t=0.5, 

# class_weight=None, warm_start=False, average=False, n_iter=None)



# some of methods

# fit(X, y[, coef_init, intercept_init, …])	Fit linear model with Stochastic Gradient Descent.

# predict(X)	Predict class labels for samples in X.



#-------------------------------

# video link: https://www.appliedaicourse.com/course/applied-ai-course-online/lessons/geometric-intuition-1/

#------------------------------







# find more about CalibratedClassifierCV here at http://scikit-learn.org/stable/modules/generated/sklearn.calibration.CalibratedClassifierCV.html

# ----------------------------

# default paramters

# sklearn.calibration.CalibratedClassifierCV(base_estimator=None, method=’sigmoid’, cv=3)

#

# some of the methods of CalibratedClassifierCV()

# fit(X, y[, sample_weight])	Fit the calibrated model

# get_params([deep])	Get parameters for this estimator.

# predict(X)	Predict the target of new samples.

# predict_proba(X)	Posterior probabilities of classification

#-------------------------------------

# video link:

#-------------------------------------



alpha = [10 ** x for x in range(-6, 1)]

cv_log_error_array = []

for i in alpha:

    print("for alpha =", i) 

    clf = SGDClassifier(alpha=i, penalty='l2', loss='log', random_state=42)

    clf.fit(train_x_onehotCoding, train_y)

    sig_clf = CalibratedClassifierCV(clf, method="sigmoid")

    sig_clf.fit(train_x_onehotCoding, train_y)

    sig_clf_probs = sig_clf.predict_proba(cv_x_onehotCoding)

    cv_log_error_array.append(log_loss(cv_y, sig_clf_probs, labels=clf.classes_, eps=1e-15))

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

clf = SGDClassifier(alpha=alpha[best_alpha], penalty='l2', loss='log', random_state=42)

clf.fit(train_x_onehotCoding, train_y)

sig_clf = CalibratedClassifierCV(clf, method="sigmoid")

sig_clf.fit(train_x_onehotCoding, train_y)



predict_y = sig_clf.predict_proba(train_x_onehotCoding)

lrLossWithoutClassBalance_tfidf_train=log_loss(y_train, predict_y, labels=clf.classes_, eps=1e-15)

print('For values of best alpha = ', alpha[best_alpha], "The train log loss is:",lrLossWithoutClassBalance_tfidf_train)



predict_y = sig_clf.predict_proba(cv_x_onehotCoding)

lrLossWithoutClassBalance_tfidf_cv=log_loss(y_cv, predict_y, labels=clf.classes_, eps=1e-15)

print('For values of best alpha = ', alpha[best_alpha], "The cross validation log loss is:",lrLossWithoutClassBalance_tfidf_cv)



predict_y = sig_clf.predict_proba(test_x_onehotCoding)

lrLossWithoutClassBalance_tfidf_test=log_loss(y_test, predict_y, labels=clf.classes_, eps=1e-15)

print('For values of best alpha = ', alpha[best_alpha], "The test log loss is:",lrLossWithoutClassBalance_tfidf_test)
# read more about SGDClassifier() at http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDClassifier.html

# ------------------------------

# default parameters

# SGDClassifier(loss=’hinge’, penalty=’l2’, alpha=0.0001, l1_ratio=0.15, fit_intercept=True, max_iter=None, tol=None, 

# shuffle=True, verbose=0, epsilon=0.1, n_jobs=1, random_state=None, learning_rate=’optimal’, eta0=0.0, power_t=0.5, 

# class_weight=None, warm_start=False, average=False, n_iter=None)



# some of methods

# fit(X, y[, coef_init, intercept_init, …])	Fit linear model with Stochastic Gradient Descent.

# predict(X)	Predict class labels for samples in X.



#-------------------------------

# video link: 

#------------------------------



clf = SGDClassifier(alpha=alpha[best_alpha], penalty='l2', loss='log', random_state=42)

lrWithoutClassBalancemp=predict_and_plot_confusion_matrix(train_x_onehotCoding, train_y, cv_x_onehotCoding, cv_y, clf)
lrWithoutClassBalancemp
clf = SGDClassifier(alpha=alpha[best_alpha], penalty='l2', loss='log', random_state=42)

clf.fit(train_x_onehotCoding,train_y)

test_point_index = 1

no_feature = 500

predicted_cls = sig_clf.predict(test_x_onehotCoding[test_point_index])

print("Predicted Class :", predicted_cls[0])

print("Predicted Class Probabilities:", np.round(sig_clf.predict_proba(test_x_onehotCoding[test_point_index]),4))

print("Actual Class :", test_y[test_point_index])

indices = np.argsort(abs(-clf.coef_))[predicted_cls-1][:,:no_feature]

print("-"*50)

get_impfeature_names(indices[0], test_df['TEXT'].iloc[test_point_index],test_df['Gene'].iloc[test_point_index],test_df['Variation'].iloc[test_point_index], no_feature)
test_point_index = 100

no_feature = 500

predicted_cls = sig_clf.predict(test_x_onehotCoding[test_point_index])

print("Predicted Class :", predicted_cls[0])

print("Predicted Class Probabilities:", np.round(sig_clf.predict_proba(test_x_onehotCoding[test_point_index]),4))

print("Actual Class :", test_y[test_point_index])

indices = np.argsort(abs(-clf.coef_))[predicted_cls-1][:,:no_feature]

print("-"*50)

get_impfeature_names(indices[0], test_df['TEXT'].iloc[test_point_index],test_df['Gene'].iloc[test_point_index],test_df['Variation'].iloc[test_point_index], no_feature)
alpha = [10 ** x for x in range(-6, 3)]

cv_log_error_array = []

for i in alpha:

    print("for alpha =", i)

    clf = SGDClassifier(class_weight='balanced', alpha=i, penalty='l2', loss='log', random_state=42)

    clf.fit(train_x_onehotCoding_count, train_y)

    sig_clf = CalibratedClassifierCV(clf, method="sigmoid")

    sig_clf.fit(train_x_onehotCoding_count, train_y)

    sig_clf_probs = sig_clf.predict_proba(cv_x_onehotCoding_count)

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

clf.fit(train_x_onehotCoding_count, train_y)

sig_clf = CalibratedClassifierCV(clf, method="sigmoid")

sig_clf.fit(train_x_onehotCoding_count, train_y)



predict_y = sig_clf.predict_proba(train_x_onehotCoding_count)

lrLossClassBalance_count_train=log_loss(y_train, predict_y, labels=clf.classes_, eps=1e-15)

print('For values of best alpha = ', alpha[best_alpha], "The train log loss is:",lrLossClassBalance_count_train)



predict_y = sig_clf.predict_proba(cv_x_onehotCoding_count)

lrLossClassBalance_count_cv=log_loss(y_cv, predict_y, labels=clf.classes_, eps=1e-15)

print('For values of best alpha = ', alpha[best_alpha], "The cross validation log loss is:",lrLossClassBalance_count_cv)



predict_y = sig_clf.predict_proba(test_x_onehotCoding_count)

lrLossClassBalance_count_test=log_loss(y_test, predict_y, labels=clf.classes_, eps=1e-15)

print('For values of best alpha = ', alpha[best_alpha], "The test log loss is:",lrLossClassBalance_count_test)
clf = SGDClassifier(class_weight='balanced', alpha=alpha[best_alpha], penalty='l2', loss='log', random_state=42)

lrClassBalanceCountmp=predict_and_plot_confusion_matrix(train_x_onehotCoding_count, train_y, cv_x_onehotCoding_count, cv_y, clf)
def get_imp_feature_names_countVec(text, indices, removed_ind = []):

    word_present = 0

    tabulte_list = []

    incresingorder_ind = 0

    for i in indices:

        if i < train_gene_feature_onehotCoding_onehot.shape[1]:

            tabulte_list.append([incresingorder_ind, "Gene", "Yes"])

        elif i< 18:

            tabulte_list.append([incresingorder_ind,"Variation", "Yes"])

        if ((i > 17) & (i not in removed_ind)) :

            word = train_text_features[i]

            yes_no = True if word in text.split() else False

            if yes_no:

                word_present += 1

            tabulte_list.append([incresingorder_ind,train_text_features[i], yes_no])

        incresingorder_ind += 1

    print(word_present, "most importent features are present in our query point")

    print("-"*50)

    print("The features that are most importent of the ",predicted_cls[0]," class:")

    print (tabulate(tabulte_list, headers=["Index",'Feature name', 'Present or Not']))
# from tabulate import tabulate

clf = SGDClassifier(class_weight='balanced', alpha=alpha[best_alpha], penalty='l2', loss='log', random_state=42)

clf.fit(train_x_onehotCoding_count,train_y)

test_point_index = 1

no_feature = 500

predicted_cls = sig_clf.predict(test_x_onehotCoding_count[test_point_index])

print("Predicted Class :", predicted_cls[0])

print("Predicted Class Probabilities:", np.round(sig_clf.predict_proba(test_x_onehotCoding_count[test_point_index]),4))

print("Actual Class :", test_y[test_point_index])

indices = np.argsort(abs(-clf.coef_))[predicted_cls-1][:,:no_feature]

print("-"*50)

# get_impfeature_names_countVec(indices[0], test_df['TEXT'].iloc[test_point_index],test_df['Gene'].iloc[test_point_index],test_df['Variation'].iloc[test_point_index], no_feature)
test_point_index = 100

no_feature = 500

predicted_cls = sig_clf.predict(test_x_onehotCoding_count[test_point_index])

print("Predicted Class :", predicted_cls[0])

print("Predicted Class Probabilities:", np.round(sig_clf.predict_proba(test_x_onehotCoding_count[test_point_index]),4))

print("Actual Class :", test_y[test_point_index])

indices = np.argsort(abs(-clf.coef_))[predicted_cls-1][:,:no_feature]

print("-"*50)

# get_impfeature_names_countVec(indices[0], test_df['TEXT'].iloc[test_point_index],test_df['Gene'].iloc[test_point_index],test_df['Variation'].iloc[test_point_index], no_feature)
alpha = [10 ** x for x in range(-6, 3)]

cv_log_error_array = []

for i in alpha:

    print("for alpha =", i)

    clf = SGDClassifier( alpha=i, penalty='l2', loss='log', random_state=42)

    clf.fit(train_x_onehotCoding_count, train_y)

    sig_clf = CalibratedClassifierCV(clf, method="sigmoid")

    sig_clf.fit(train_x_onehotCoding_count, train_y)

    sig_clf_probs = sig_clf.predict_proba(cv_x_onehotCoding_count)

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

clf.fit(train_x_onehotCoding_count, train_y)

sig_clf = CalibratedClassifierCV(clf, method="sigmoid")

sig_clf.fit(train_x_onehotCoding_count, train_y)



predict_y = sig_clf.predict_proba(train_x_onehotCoding_count)

lrLossWithoutClassBalance_count_train=log_loss(y_train, predict_y, labels=clf.classes_, eps=1e-15)

print('For values of best alpha = ', alpha[best_alpha], "The train log loss is:",lrLossWithoutClassBalance_count_train)



predict_y = sig_clf.predict_proba(cv_x_onehotCoding_count)

lrLossWithoutClassBalance_count_cv=log_loss(y_cv, predict_y, labels=clf.classes_, eps=1e-15)

print('For values of best alpha = ', alpha[best_alpha], "The cross validation log loss is:",lrLossWithoutClassBalance_count_cv)



predict_y = sig_clf.predict_proba(test_x_onehotCoding_count)

lrLossWithoutClassBalance_count_test=log_loss(y_test, predict_y, labels=clf.classes_, eps=1e-15)

print('For values of best alpha = ', alpha[best_alpha], "The test log loss is:",lrLossWithoutClassBalance_count_test)
clf = SGDClassifier(class_weight='balanced', alpha=alpha[best_alpha], penalty='l2', loss='log', random_state=42)

lrWithoutClassBalanceCountmp=predict_and_plot_confusion_matrix(train_x_onehotCoding_count, train_y, cv_x_onehotCoding_count, cv_y, clf)
# from tabulate import tabulate

clf = SGDClassifier(alpha=alpha[best_alpha], penalty='l2', loss='log', random_state=42)

clf.fit(train_x_onehotCoding_count,train_y)

test_point_index = 1

no_feature = 500

predicted_cls = sig_clf.predict(test_x_onehotCoding_count[test_point_index])

print("Predicted Class :", predicted_cls[0])

print("Predicted Class Probabilities:", np.round(sig_clf.predict_proba(test_x_onehotCoding_count[test_point_index]),4))

print("Actual Class :", test_y[test_point_index])

indices = np.argsort(abs(-clf.coef_))[predicted_cls-1][:,:no_feature]

print("-"*50)

# get_impfeature_names_countVec(indices[0], test_df['TEXT'].iloc[test_point_index],test_df['Gene'].iloc[test_point_index],test_df['Variation'].iloc[test_point_index], no_feature)
test_point_index = 100

no_feature = 500

predicted_cls = sig_clf.predict(test_x_onehotCoding_count[test_point_index])

print("Predicted Class :", predicted_cls[0])

print("Predicted Class Probabilities:", np.round(sig_clf.predict_proba(test_x_onehotCoding_count[test_point_index]),4))

print("Actual Class :", test_y[test_point_index])

indices = np.argsort(abs(-clf.coef_))[predicted_cls-1][:,:no_feature]

print("-"*50)

# get_impfeature_names_countVec(indices[0], test_df['TEXT'].iloc[test_point_index],test_df['Gene'].iloc[test_point_index],test_df['Variation'].iloc[test_point_index], no_feature)
# read more about support vector machines with linear kernals here http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html



# --------------------------------

# default parameters 

# SVC(C=1.0, kernel=’rbf’, degree=3, gamma=’auto’, coef0=0.0, shrinking=True, probability=False, tol=0.001, 

# cache_size=200, class_weight=None, verbose=False, max_iter=-1, decision_function_shape=’ovr’, random_state=None)



# Some of methods of SVM()

# fit(X, y, [sample_weight])	Fit the SVM model according to the given training data.

# predict(X)	Perform classification on samples in X.

# --------------------------------

# video link: https://www.appliedaicourse.com/course/applied-ai-course-online/lessons/mathematical-derivation-copy-8/

# --------------------------------







# find more about CalibratedClassifierCV here at http://scikit-learn.org/stable/modules/generated/sklearn.calibration.CalibratedClassifierCV.html

# ----------------------------

# default paramters

# sklearn.calibration.CalibratedClassifierCV(base_estimator=None, method=’sigmoid’, cv=3)

#

# some of the methods of CalibratedClassifierCV()

# fit(X, y[, sample_weight])	Fit the calibrated model

# get_params([deep])	Get parameters for this estimator.

# predict(X)	Predict the target of new samples.

# predict_proba(X)	Posterior probabilities of classification

#-------------------------------------

# video link:

#-------------------------------------



alpha = [10 ** x for x in range(-5, 3)]

cv_log_error_array = []

for i in alpha:

    print("for C =", i)

#     clf = SVC(C=i,kernel='linear',probability=True, class_weight='balanced')

    clf = SGDClassifier( class_weight='balanced', alpha=i, penalty='l2', loss='hinge', random_state=42)

    clf.fit(train_x_onehotCoding, train_y)

    sig_clf = CalibratedClassifierCV(clf, method="sigmoid")

    sig_clf.fit(train_x_onehotCoding, train_y)

    sig_clf_probs = sig_clf.predict_proba(cv_x_onehotCoding)

    cv_log_error_array.append(log_loss(cv_y, sig_clf_probs, labels=clf.classes_, eps=1e-15))

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

# clf = SVC(C=i,kernel='linear',probability=True, class_weight='balanced')

clf = SGDClassifier(class_weight='balanced', alpha=alpha[best_alpha], penalty='l2', loss='hinge', random_state=42)

clf.fit(train_x_onehotCoding, train_y)

sig_clf = CalibratedClassifierCV(clf, method="sigmoid")

sig_clf.fit(train_x_onehotCoding, train_y)



predict_y = sig_clf.predict_proba(train_x_onehotCoding)

svmLoss_train=log_loss(y_train, predict_y, labels=clf.classes_, eps=1e-15)

print('For values of best alpha = ', alpha[best_alpha], "The train log loss is:",svmLoss_train)



predict_y = sig_clf.predict_proba(cv_x_onehotCoding)

svmLoss_cv=log_loss(y_cv, predict_y, labels=clf.classes_, eps=1e-15)

print('For values of best alpha = ', alpha[best_alpha], "The cross validation log loss is:",svmLoss_cv)



predict_y = sig_clf.predict_proba(test_x_onehotCoding)

svmLoss_test=log_loss(y_test, predict_y, labels=clf.classes_, eps=1e-15)

print('For values of best alpha = ', alpha[best_alpha], "The test log loss is:",svmLoss_test)

# read more about support vector machines with linear kernals here http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html



# --------------------------------

# default parameters 

# SVC(C=1.0, kernel=’rbf’, degree=3, gamma=’auto’, coef0=0.0, shrinking=True, probability=False, tol=0.001, 

# cache_size=200, class_weight=None, verbose=False, max_iter=-1, decision_function_shape=’ovr’, random_state=None)



# Some of methods of SVM()

# fit(X, y, [sample_weight])	Fit the SVM model according to the given training data.

# predict(X)	Perform classification on samples in X.

# --------------------------------

# video link: https://www.appliedaicourse.com/course/applied-ai-course-online/lessons/mathematical-derivation-copy-8/

# --------------------------------





# clf = SVC(C=alpha[best_alpha],kernel='linear',probability=True, class_weight='balanced')

clf = SGDClassifier(alpha=alpha[best_alpha], penalty='l2', loss='hinge', random_state=42,class_weight='balanced')

svmmp=predict_and_plot_confusion_matrix(train_x_onehotCoding, train_y,cv_x_onehotCoding,cv_y, clf)
clf = SGDClassifier(alpha=alpha[best_alpha], penalty='l2', loss='hinge', random_state=42)

clf.fit(train_x_onehotCoding,train_y)

test_point_index = 1

# test_point_index = 100

no_feature = 500

predicted_cls = sig_clf.predict(test_x_onehotCoding[test_point_index])

print("Predicted Class :", predicted_cls[0])

print("Predicted Class Probabilities:", np.round(sig_clf.predict_proba(test_x_onehotCoding[test_point_index]),4))

print("Actual Class :", test_y[test_point_index])

indices = np.argsort(abs(-clf.coef_))[predicted_cls-1][:,:no_feature]

print("-"*50)

get_impfeature_names(indices[0], test_df['TEXT'].iloc[test_point_index],test_df['Gene'].iloc[test_point_index],test_df['Variation'].iloc[test_point_index], no_feature)
test_point_index = 100

no_feature = 500

predicted_cls = sig_clf.predict(test_x_onehotCoding[test_point_index])

print("Predicted Class :", predicted_cls[0])

print("Predicted Class Probabilities:", np.round(sig_clf.predict_proba(test_x_onehotCoding[test_point_index]),4))

print("Actual Class :", test_y[test_point_index])

indices = np.argsort(abs(-clf.coef_))[predicted_cls-1][:,:no_feature]

print("-"*50)

get_impfeature_names(indices[0], test_df['TEXT'].iloc[test_point_index],test_df['Gene'].iloc[test_point_index],test_df['Variation'].iloc[test_point_index], no_feature)
# --------------------------------

# default parameters 

# sklearn.ensemble.RandomForestClassifier(n_estimators=10, criterion=’gini’, max_depth=None, min_samples_split=2, 

# min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features=’auto’, max_leaf_nodes=None, min_impurity_decrease=0.0, 

# min_impurity_split=None, bootstrap=True, oob_score=False, n_jobs=1, random_state=None, verbose=0, warm_start=False, 

# class_weight=None)



# Some of methods of RandomForestClassifier()

# fit(X, y, [sample_weight])	Fit the SVM model according to the given training data.

# predict(X)	Perform classification on samples in X.

# predict_proba (X)	Perform classification on samples in X.



# some of attributes of  RandomForestClassifier()

# feature_importances_ : array of shape = [n_features]

# The feature importances (the higher, the more important the feature).



# --------------------------------

# video link: https://www.appliedaicourse.com/course/applied-ai-course-online/lessons/random-forest-and-their-construction-2/

# --------------------------------





# find more about CalibratedClassifierCV here at http://scikit-learn.org/stable/modules/generated/sklearn.calibration.CalibratedClassifierCV.html

# ----------------------------

# default paramters

# sklearn.calibration.CalibratedClassifierCV(base_estimator=None, method=’sigmoid’, cv=3)

#

# some of the methods of CalibratedClassifierCV()

# fit(X, y[, sample_weight])	Fit the calibrated model

# get_params([deep])	Get parameters for this estimator.

# predict(X)	Predict the target of new samples.

# predict_proba(X)	Posterior probabilities of classification

#-------------------------------------

# video link:

#-------------------------------------



alpha = [100,200,500,1000,2000]

max_depth = [5, 10]

cv_log_error_array = []

for i in alpha:

    for j in max_depth:

        print("for n_estimators =", i,"and max depth = ", j)

        clf = RandomForestClassifier(n_estimators=i, criterion='gini', max_depth=j, random_state=42, n_jobs=-1)

        clf.fit(train_x_onehotCoding, train_y)

        sig_clf = CalibratedClassifierCV(clf, method="sigmoid")

        sig_clf.fit(train_x_onehotCoding, train_y)

        sig_clf_probs = sig_clf.predict_proba(cv_x_onehotCoding)

        cv_log_error_array.append(log_loss(cv_y, sig_clf_probs, labels=clf.classes_, eps=1e-15))

        print("Log Loss :",log_loss(cv_y, sig_clf_probs)) 



'''fig, ax = plt.subplots()

features = np.dot(np.array(alpha)[:,None],np.array(max_depth)[None]).ravel()

ax.plot(features, cv_log_error_array,c='g')

for i, txt in enumerate(np.round(cv_log_error_array,3)):

    ax.annotate((alpha[int(i/2)],max_depth[int(i%2)],str(txt)), (features[i],cv_log_error_array[i]))

plt.grid()

plt.title("Cross Validation Error for each alpha")

plt.xlabel("Alpha i's")

plt.ylabel("Error measure")

plt.show()

'''



best_alpha = np.argmin(cv_log_error_array)

clf = RandomForestClassifier(n_estimators=alpha[int(best_alpha/2)], criterion='gini', max_depth=max_depth[int(best_alpha%2)], random_state=42, n_jobs=-1)

clf.fit(train_x_onehotCoding, train_y)

sig_clf = CalibratedClassifierCV(clf, method="sigmoid")

sig_clf.fit(train_x_onehotCoding, train_y)



predict_y = sig_clf.predict_proba(train_x_onehotCoding)

rfLoss_train=log_loss(y_train, predict_y, labels=clf.classes_, eps=1e-15)

print('For values of best estimator = ', alpha[int(best_alpha/2)], "The train log loss is:",rfLoss_train)

predict_y = sig_clf.predict_proba(cv_x_onehotCoding)

rfLoss_cv=log_loss(y_cv, predict_y, labels=clf.classes_, eps=1e-15)

print('For values of best estimator = ', alpha[int(best_alpha/2)], "The cross validation log loss is:",rfLoss_cv)

predict_y = sig_clf.predict_proba(test_x_onehotCoding)

rfLoss_test=log_loss(y_test, predict_y, labels=clf.classes_, eps=1e-15)

print('For values of best estimator = ', alpha[int(best_alpha/2)], "The test log loss is:",rfLoss_test)
# --------------------------------

# default parameters 

# sklearn.ensemble.RandomForestClassifier(n_estimators=10, criterion=’gini’, max_depth=None, min_samples_split=2, 

# min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features=’auto’, max_leaf_nodes=None, min_impurity_decrease=0.0, 

# min_impurity_split=None, bootstrap=True, oob_score=False, n_jobs=1, random_state=None, verbose=0, warm_start=False, 

# class_weight=None)



# Some of methods of RandomForestClassifier()

# fit(X, y, [sample_weight])	Fit the SVM model according to the given training data.

# predict(X)	Perform classification on samples in X.

# predict_proba (X)	Perform classification on samples in X.



# some of attributes of  RandomForestClassifier()

# feature_importances_ : array of shape = [n_features]

# The feature importances (the higher, the more important the feature).



# --------------------------------

# video link: https://www.appliedaicourse.com/course/applied-ai-course-online/lessons/random-forest-and-their-construction-2/

# --------------------------------



clf = RandomForestClassifier(n_estimators=alpha[int(best_alpha/2)], criterion='gini', max_depth=max_depth[int(best_alpha%2)], random_state=42, n_jobs=-1)

rfmp=predict_and_plot_confusion_matrix(train_x_onehotCoding, train_y,cv_x_onehotCoding,cv_y, clf)
# test_point_index = 10

clf = RandomForestClassifier(n_estimators=alpha[int(best_alpha/2)], criterion='gini', max_depth=max_depth[int(best_alpha%2)], random_state=42, n_jobs=-1)

clf.fit(train_x_onehotCoding, train_y)

sig_clf = CalibratedClassifierCV(clf, method="sigmoid")

sig_clf.fit(train_x_onehotCoding, train_y)



test_point_index = 1

no_feature = 100

predicted_cls = sig_clf.predict(test_x_onehotCoding[test_point_index])

print("Predicted Class :", predicted_cls[0])

print("Predicted Class Probabilities:", np.round(sig_clf.predict_proba(test_x_onehotCoding[test_point_index]),4))

print("Actual Class :", test_y[test_point_index])

indices = np.argsort(-clf.feature_importances_)

print("-"*50)

get_impfeature_names(indices[:no_feature], test_df['TEXT'].iloc[test_point_index],test_df['Gene'].iloc[test_point_index],test_df['Variation'].iloc[test_point_index], no_feature)
test_point_index = 100

no_feature = 100

predicted_cls = sig_clf.predict(test_x_onehotCoding[test_point_index])

print("Predicted Class :", predicted_cls[0])

print("Predicted Class Probabilities:", np.round(sig_clf.predict_proba(test_x_onehotCoding[test_point_index]),4))

print("Actuall Class :", test_y[test_point_index])

indices = np.argsort(-clf.feature_importances_)

print("-"*50)

get_impfeature_names(indices[:no_feature], test_df['TEXT'].iloc[test_point_index],test_df['Gene'].iloc[test_point_index],test_df['Variation'].iloc[test_point_index], no_feature)
# # --------------------------------

# # default parameters 

# # sklearn.ensemble.RandomForestClassifier(n_estimators=10, criterion=’gini’, max_depth=None, min_samples_split=2, 

# # min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features=’auto’, max_leaf_nodes=None, min_impurity_decrease=0.0, 

# # min_impurity_split=None, bootstrap=True, oob_score=False, n_jobs=1, random_state=None, verbose=0, warm_start=False, 

# # class_weight=None)



# # Some of methods of RandomForestClassifier()

# # fit(X, y, [sample_weight])	Fit the SVM model according to the given training data.

# # predict(X)	Perform classification on samples in X.

# # predict_proba (X)	Perform classification on samples in X.



# # some of attributes of  RandomForestClassifier()

# # feature_importances_ : array of shape = [n_features]

# # The feature importances (the higher, the more important the feature).



# # --------------------------------

# # video link: https://www.appliedaicourse.com/course/applied-ai-course-online/lessons/random-forest-and-their-construction-2/

# # --------------------------------





# # find more about CalibratedClassifierCV here at http://scikit-learn.org/stable/modules/generated/sklearn.calibration.CalibratedClassifierCV.html

# # ----------------------------

# # default paramters

# # sklearn.calibration.CalibratedClassifierCV(base_estimator=None, method=’sigmoid’, cv=3)

# #

# # some of the methods of CalibratedClassifierCV()

# # fit(X, y[, sample_weight])	Fit the calibrated model

# # get_params([deep])	Get parameters for this estimator.

# # predict(X)	Predict the target of new samples.

# # predict_proba(X)	Posterior probabilities of classification

# #-------------------------------------

# # video link:

# #-------------------------------------



# alpha = [10,50,100,200,500,1000]

# max_depth = [2,3,5,10]

# cv_log_error_array = []

# for i in alpha:

#     for j in max_depth:

#         print("for n_estimators =", i,"and max depth = ", j)

#         clf = RandomForestClassifier(n_estimators=i, criterion='gini', max_depth=j, random_state=42, n_jobs=-1)

#         clf.fit(train_x_responseCoding, train_y)

#         sig_clf = CalibratedClassifierCV(clf, method="sigmoid")

#         sig_clf.fit(train_x_responseCoding, train_y)

#         sig_clf_probs = sig_clf.predict_proba(cv_x_responseCoding)

#         cv_log_error_array.append(log_loss(cv_y, sig_clf_probs, labels=clf.classes_, eps=1e-15))

#         print("Log Loss :",log_loss(cv_y, sig_clf_probs)) 

# '''

# fig, ax = plt.subplots()

# features = np.dot(np.array(alpha)[:,None],np.array(max_depth)[None]).ravel()

# ax.plot(features, cv_log_error_array,c='g')

# for i, txt in enumerate(np.round(cv_log_error_array,3)):

#     ax.annotate((alpha[int(i/4)],max_depth[int(i%4)],str(txt)), (features[i],cv_log_error_array[i]))

# plt.grid()

# plt.title("Cross Validation Error for each alpha")

# plt.xlabel("Alpha i's")

# plt.ylabel("Error measure")

# plt.show()

# '''



# best_alpha = np.argmin(cv_log_error_array)

# clf = RandomForestClassifier(n_estimators=alpha[int(best_alpha/4)], criterion='gini', max_depth=max_depth[int(best_alpha%4)], random_state=42, n_jobs=-1)

# clf.fit(train_x_responseCoding, train_y)

# sig_clf = CalibratedClassifierCV(clf, method="sigmoid")

# sig_clf.fit(train_x_responseCoding, train_y)



# predict_y = sig_clf.predict_proba(train_x_responseCoding)

# print('For values of best alpha = ', alpha[int(best_alpha/4)], "The train log loss is:",log_loss(y_train, predict_y, labels=clf.classes_, eps=1e-15))

# predict_y = sig_clf.predict_proba(cv_x_responseCoding)

# print('For values of best alpha = ', alpha[int(best_alpha/4)], "The cross validation log loss is:",log_loss(y_cv, predict_y, labels=clf.classes_, eps=1e-15))

# predict_y = sig_clf.predict_proba(test_x_responseCoding)

# print('For values of best alpha = ', alpha[int(best_alpha/4)], "The test log loss is:",log_loss(y_test, predict_y, labels=clf.classes_, eps=1e-15))
# # --------------------------------

# # default parameters 

# # sklearn.ensemble.RandomForestClassifier(n_estimators=10, criterion=’gini’, max_depth=None, min_samples_split=2, 

# # min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features=’auto’, max_leaf_nodes=None, min_impurity_decrease=0.0, 

# # min_impurity_split=None, bootstrap=True, oob_score=False, n_jobs=1, random_state=None, verbose=0, warm_start=False, 

# # class_weight=None)



# # Some of methods of RandomForestClassifier()

# # fit(X, y, [sample_weight])	Fit the SVM model according to the given training data.

# # predict(X)	Perform classification on samples in X.

# # predict_proba (X)	Perform classification on samples in X.



# # some of attributes of  RandomForestClassifier()

# # feature_importances_ : array of shape = [n_features]

# # The feature importances (the higher, the more important the feature).



# # --------------------------------

# # video link: https://www.appliedaicourse.com/course/applied-ai-course-online/lessons/random-forest-and-their-construction-2/

# # --------------------------------



# clf = RandomForestClassifier(max_depth=max_depth[int(best_alpha%4)], n_estimators=alpha[int(best_alpha/4)], criterion='gini', max_features='auto',random_state=42)

# predict_and_plot_confusion_matrix(train_x_responseCoding, train_y,cv_x_responseCoding,cv_y, clf)
# clf = RandomForestClassifier(n_estimators=alpha[int(best_alpha/4)], criterion='gini', max_depth=max_depth[int(best_alpha%4)], random_state=42, n_jobs=-1)

# clf.fit(train_x_responseCoding, train_y)

# sig_clf = CalibratedClassifierCV(clf, method="sigmoid")

# sig_clf.fit(train_x_responseCoding, train_y)





# test_point_index = 1

# no_feature = 27

# predicted_cls = sig_clf.predict(test_x_responseCoding[test_point_index].reshape(1,-1))

# print("Predicted Class :", predicted_cls[0])

# print("Predicted Class Probabilities:", np.round(sig_clf.predict_proba(test_x_responseCoding[test_point_index].reshape(1,-1)),4))

# print("Actual Class :", test_y[test_point_index])

# indices = np.argsort(-clf.feature_importances_)

# print("-"*50)

# for i in indices:

#     if i<9:

#         print("Gene is important feature")

#     elif i<18:

#         print("Variation is important feature")

#     else:

#         print("Text is important feature")
# test_point_index = 100

# predicted_cls = sig_clf.predict(test_x_responseCoding[test_point_index].reshape(1,-1))

# print("Predicted Class :", predicted_cls[0])

# print("Predicted Class Probabilities:", np.round(sig_clf.predict_proba(test_x_responseCoding[test_point_index].reshape(1,-1)),4))

# print("Actual Class :", test_y[test_point_index])

# indices = np.argsort(-clf.feature_importances_)

# print("-"*50)

# for i in indices:

#     if i<9:

#         print("Gene is important feature")

#     elif i<18:

#         print("Variation is important feature")

#     else:

#         print("Text is important feature")
# read more about SGDClassifier() at http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDClassifier.html

# ------------------------------

# default parameters

# SGDClassifier(loss=’hinge’, penalty=’l2’, alpha=0.0001, l1_ratio=0.15, fit_intercept=True, max_iter=None, tol=None, 

# shuffle=True, verbose=0, epsilon=0.1, n_jobs=1, random_state=None, learning_rate=’optimal’, eta0=0.0, power_t=0.5, 

# class_weight=None, warm_start=False, average=False, n_iter=None)



# some of methods

# fit(X, y[, coef_init, intercept_init, …])	Fit linear model with Stochastic Gradient Descent.

# predict(X)	Predict class labels for samples in X.



#-------------------------------

# video link: https://www.appliedaicourse.com/course/applied-ai-course-online/lessons/geometric-intuition-1/

#------------------------------





# read more about support vector machines with linear kernals here http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html

# --------------------------------

# default parameters 

# SVC(C=1.0, kernel=’rbf’, degree=3, gamma=’auto’, coef0=0.0, shrinking=True, probability=False, tol=0.001, 

# cache_size=200, class_weight=None, verbose=False, max_iter=-1, decision_function_shape=’ovr’, random_state=None)



# Some of methods of SVM()

# fit(X, y, [sample_weight])	Fit the SVM model according to the given training data.

# predict(X)	Perform classification on samples in X.

# --------------------------------

# video link: https://www.appliedaicourse.com/course/applied-ai-course-online/lessons/mathematical-derivation-copy-8/

# --------------------------------





# read more about support vector machines with linear kernals here http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html

# --------------------------------

# default parameters 

# sklearn.ensemble.RandomForestClassifier(n_estimators=10, criterion=’gini’, max_depth=None, min_samples_split=2, 

# min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features=’auto’, max_leaf_nodes=None, min_impurity_decrease=0.0, 

# min_impurity_split=None, bootstrap=True, oob_score=False, n_jobs=1, random_state=None, verbose=0, warm_start=False, 

# class_weight=None)



# Some of methods of RandomForestClassifier()

# fit(X, y, [sample_weight])	Fit the SVM model according to the given training data.

# predict(X)	Perform classification on samples in X.

# predict_proba (X)	Perform classification on samples in X.



# some of attributes of  RandomForestClassifier()

# feature_importances_ : array of shape = [n_features]

# The feature importances (the higher, the more important the feature).



# --------------------------------

# video link: https://www.appliedaicourse.com/course/applied-ai-course-online/lessons/random-forest-and-their-construction-2/

# --------------------------------





clf1 = SGDClassifier(alpha=0.001, penalty='l2', loss='log', class_weight='balanced', random_state=0)

clf1.fit(train_x_onehotCoding, train_y)

sig_clf1 = CalibratedClassifierCV(clf1, method="sigmoid")



clf2 = SGDClassifier(alpha=1, penalty='l2', loss='hinge', class_weight='balanced', random_state=0)

clf2.fit(train_x_onehotCoding, train_y)

sig_clf2 = CalibratedClassifierCV(clf2, method="sigmoid")





clf3 = MultinomialNB(alpha=0.001)

clf3.fit(train_x_onehotCoding, train_y)

sig_clf3 = CalibratedClassifierCV(clf3, method="sigmoid")



sig_clf1.fit(train_x_onehotCoding, train_y)

print("Logistic Regression :  Log Loss: %0.2f" % (log_loss(cv_y, sig_clf1.predict_proba(cv_x_onehotCoding))))

sig_clf2.fit(train_x_onehotCoding, train_y)

print("Support vector machines : Log Loss: %0.2f" % (log_loss(cv_y, sig_clf2.predict_proba(cv_x_onehotCoding))))

sig_clf3.fit(train_x_onehotCoding, train_y)

print("Naive Bayes : Log Loss: %0.2f" % (log_loss(cv_y, sig_clf3.predict_proba(cv_x_onehotCoding))))

print("-"*50)

alpha = [0.0001,0.001,0.01,0.1,1,10] 

best_alpha = 999

for i in alpha:

    lr = LogisticRegression(C=i)

    sclf = StackingClassifier(classifiers=[sig_clf1, sig_clf2, sig_clf3], meta_classifier=lr, use_probas=True)

    sclf.fit(train_x_onehotCoding, train_y)

    print("Stacking Classifer : for the value of alpha: %f Log Loss: %0.3f" % (i, log_loss(cv_y, sclf.predict_proba(cv_x_onehotCoding))))

    log_error =log_loss(cv_y, sclf.predict_proba(cv_x_onehotCoding))

    if best_alpha > log_error:

        best_alpha = log_error
lr = LogisticRegression(C=0.1)

sclf = StackingClassifier(classifiers=[sig_clf1, sig_clf2, sig_clf3], meta_classifier=lr, use_probas=True)

sclf.fit(train_x_onehotCoding, train_y)



stack_log_error_train = log_loss(train_y, sclf.predict_proba(train_x_onehotCoding))

print("Log loss (train) on the stacking classifier :",stack_log_error_train)



stack_log_error_cv = log_loss(cv_y, sclf.predict_proba(cv_x_onehotCoding))

print("Log loss (CV) on the stacking classifier :",stack_log_error_cv)



stack_log_error_test = log_loss(test_y, sclf.predict_proba(test_x_onehotCoding))

print("Log loss (test) on the stacking classifier :",stack_log_error_test)



stackmp=np.count_nonzero((sclf.predict(test_x_onehotCoding)- test_y))/test_y.shape[0]

print("Number of missclassified point :", stackmp)

plot_confusion_matrix(test_y=test_y, predict_y=sclf.predict(test_x_onehotCoding))
#Refer:http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.VotingClassifier.html

from sklearn.ensemble import VotingClassifier

vclf = VotingClassifier(estimators=[('lr', sig_clf1), ('svc', sig_clf2), ('rf', sig_clf3)], voting='soft')

vclf.fit(train_x_onehotCoding, train_y)



max_log_error_train=log_loss(train_y, vclf.predict_proba(train_x_onehotCoding))

max_log_error_cv=log_loss(cv_y, vclf.predict_proba(cv_x_onehotCoding))

max_log_error_test=log_loss(test_y, vclf.predict_proba(test_x_onehotCoding))



print("Log loss (train) on the VotingClassifier :",max_log_error_train)

print("Log loss (CV) on the VotingClassifier :",max_log_error_train)

print("Log loss (test) on the VotingClassifier :",max_log_error_train)

maxmp=np.count_nonzero((vclf.predict(test_x_onehotCoding)- test_y))/test_y.shape[0]

print("Number of missclassified point :", maxmp)

plot_confusion_matrix(test_y=test_y, predict_y=vclf.predict(test_x_onehotCoding))
# one-hot encoding of Gene feature.

gene_vectorizer_fe = CountVectorizer()

train_gene_feature_onehotCoding_fe = gene_vectorizer_fe.fit_transform(train_df['Gene'])

test_gene_feature_onehotCoding_fe = gene_vectorizer_fe.transform(test_df['Gene'])

cv_gene_feature_onehotCoding_fe = gene_vectorizer_fe.transform(cv_df['Gene'])

print('train_gene_feature_onehotCoding',train_gene_feature_onehotCoding_fe.shape)
### one-hot encoding of variation feature.

variation_vectorizer_fe = CountVectorizer()

train_variation_feature_onehotCoding_fe = variation_vectorizer_fe.fit_transform(train_df['Variation'])

test_variation_feature_onehotCoding_fe = variation_vectorizer_fe.transform(test_df['Variation'])

cv_variation_feature_onehotCoding_fe = variation_vectorizer_fe.transform(cv_df['Variation'])
# building a CountVectorizer with all the words that occured minimum 3 times in train data

## text_vectorizer = CountVectorizer(min_df=3)

text_vectorizer_fe = TfidfVectorizer(ngram_range=(2,2),min_df=3,max_features=20000)

train_text_feature_onehotCoding_fe = text_vectorizer_fe.fit_transform(train_df['TEXT'])

# getting all the feature names (words)

train_text_features= text_vectorizer_fe.get_feature_names()



# train_text_feature_onehotCoding.sum(axis=0).A1 will sum every row and returns (1*number of features) vector

train_text_fea_counts = train_text_feature_onehotCoding_fe.sum(axis=0).A1



# zip(list(text_features),text_fea_counts) will zip a word with its number of times it occured

text_fea_dict = dict(zip(list(train_text_features),train_text_fea_counts))





print("Total number of unique words in train data :", len(train_text_features))
# don't forget to normalize every feature

train_text_feature_onehotCoding_fe = normalize(train_text_feature_onehotCoding_fe, axis=0)



# we use the same vectorizer that was trained on train data

test_text_feature_onehotCoding_fe = text_vectorizer_fe.transform(test_df['TEXT'])

# don't forget to normalize every feature

test_text_feature_onehotCoding_fe = normalize(test_text_feature_onehotCoding_fe, axis=0)



# we use the same vectorizer that was trained on train data

cv_text_feature_onehotCoding_fe = text_vectorizer_fe.transform(cv_df['TEXT'])

# don't forget to normalize every feature

cv_text_feature_onehotCoding_fe = normalize(cv_text_feature_onehotCoding_fe, axis=0)
## for count vectorizer

train_x_onehotCoding_count_fe = hstack((train_gene_feature_onehotCoding_fe,train_variation_feature_onehotCoding_fe, train_text_feature_onehotCoding_fe)).tocsr()

test_x_onehotCoding_count_fe = hstack((test_gene_feature_onehotCoding_fe,test_variation_feature_onehotCoding_fe, test_text_feature_onehotCoding_fe)).tocsr()

cv_x_onehotCoding_count_fe = hstack((cv_gene_feature_onehotCoding_fe,cv_variation_feature_onehotCoding_fe, cv_text_feature_onehotCoding_fe)).tocsr()
print("One hot encoding features :")

print("(number of data points * number of features) in train data = ", train_x_onehotCoding_count_fe.shape)

print("(number of data points * number of features) in test data = ", test_x_onehotCoding_count_fe.shape)

print("(number of data points * number of features) in cross validation data =", cv_x_onehotCoding_count_fe.shape)
alpha = [10 ** x for x in range(-6, 3)]

cv_log_error_array = []

for i in alpha:

    print("for alpha =", i)

    clf = SGDClassifier(class_weight='balanced', alpha=i, penalty='l2', loss='log', random_state=42)

    clf.fit(train_x_onehotCoding_count_fe, train_y)

    sig_clf = CalibratedClassifierCV(clf, method="sigmoid")

    sig_clf.fit(train_x_onehotCoding_count_fe, train_y)

    sig_clf_probs = sig_clf.predict_proba(cv_x_onehotCoding_count_fe)

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

clf.fit(train_x_onehotCoding_count_fe, train_y)

sig_clf = CalibratedClassifierCV(clf, method="sigmoid")

sig_clf.fit(train_x_onehotCoding_count_fe, train_y)



predict_y = sig_clf.predict_proba(train_x_onehotCoding_count_fe)

lrLossClassBalance_tfidf_train_fe=log_loss(y_train, predict_y, labels=clf.classes_, eps=1e-15)

print('For values of best alpha = ', alpha[best_alpha], "The train log loss is:",lrLossClassBalance_tfidf_train_fe)



predict_y = sig_clf.predict_proba(cv_x_onehotCoding_count_fe)

lrLossClassBalance_tfidf_cv_fe=log_loss(y_cv, predict_y, labels=clf.classes_, eps=1e-15)

print('For values of best alpha = ', alpha[best_alpha], "The cross validation log loss is:",lrLossClassBalance_tfidf_cv_fe)



predict_y = sig_clf.predict_proba(test_x_onehotCoding_count_fe)

lrLossClassBalance_tfidf_test_fe=log_loss(y_test, predict_y, labels=clf.classes_, eps=1e-15)

print('For values of best alpha = ', alpha[best_alpha], "The test log loss is:",lrLossClassBalance_tfidf_test_fe)
print("lrLossClassBalance_tfidf_train",lrLossClassBalance_tfidf_train_fe)

print("lrLossClassBalance_tfidf_cv",lrLossClassBalance_tfidf_cv_fe)

print("lrLossClassBalance_tfidf_test",lrLossClassBalance_tfidf_test_fe)
clf = SGDClassifier(class_weight='balanced', alpha=alpha[best_alpha], penalty='l2', loss='log', random_state=42)

lrClassBalancemp_fe=predict_and_plot_confusion_matrix(train_x_onehotCoding_count_fe, train_y, cv_x_onehotCoding_count_fe, cv_y, clf)
from prettytable import PrettyTable

# Names of models

model=['Naive Bayes ','KNN','Logistic Regression With Class balancing '

       ,'LogisticRegression Without Class balancing','Linear SVM '

       ,'Random Forest Classifier With One hot Encoding'

       ,'Stack Models:LR+NB+SVM','Maximum Voting classifier'

       ,'LR(BALANCED): CountVectorizer Features, including both unigrams and bigrams','LR(UNBALANCED): CountVectorizer Features, including both unigrams and bigrams'

       ,'LR: after feature engineering']

train =[

    nbLoss_train,

    knnLoss_train,

    lrLossClassBalance_tfidf_train,

    lrLossWithoutClassBalance_tfidf_train,

    svmLoss_train,

    rfLoss_train,

    stack_log_error_train,

    max_log_error_train,

    lrLossClassBalance_count_train,

    lrLossWithoutClassBalance_count_train,

    lrLossClassBalance_tfidf_train_fe]

cv=[

    nbLoss_cv,

    knnLoss_cv,

    lrLossClassBalance_tfidf_cv,

    lrLossWithoutClassBalance_tfidf_cv,

    svmLoss_cv,

    rfLoss_cv,

    stack_log_error_cv,

    max_log_error_cv,

    lrLossClassBalance_count_cv,

    lrLossWithoutClassBalance_count_cv,

    lrLossClassBalance_tfidf_cv_fe

]

test = [

    nbLoss_test,

    knnLoss_test,

    lrLossClassBalance_tfidf_test,

    lrLossWithoutClassBalance_tfidf_test,

    svmLoss_test,

    rfLoss_test,

    stack_log_error_test,

    max_log_error_test,

    lrLossClassBalance_count_test,

    lrLossWithoutClassBalance_count_test,

    lrLossClassBalance_tfidf_test_fe,

]



mp=[

    nbmp,

    knnmp,

    lrClassBalancemp,

    lrWithoutClassBalancemp,

    svmmp,

    rfmp,

    stackmp,

    maxmp,

    lrClassBalanceCountmp,

    lrWithoutClassBalanceCountmp,

    lrClassBalancemp_fe,

   ]





train=[round(x,2) for x in train]

cv=[round(x,2) for x in cv]

test=[round(x,2) for x in test]

mp=[round(x,2) for x in mp]

numbering=[1,2,3,4,5,6,7,8,9,10,11]

# Initializing prettytable

ptable = PrettyTable()

# Adding columns

ptable.add_column("S.NO.",numbering)

ptable.add_column("model",model)

ptable.add_column("train",train)

ptable.add_column("cv",cv)

ptable.add_column("test",test)

ptable.add_column("% Misclassified Points",mp)

# Printing the Table

print(ptable)