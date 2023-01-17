# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
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

data = pd.read_csv('../input/msk-redefining-cancer-treatment/training_variants.zip')

print('Number of data points: ',data.shape[0])

print('Number of features: ',data.shape[1])

print('Features: ',data.columns.values)

data.head()
data_text = pd.read_csv('../input/msk-redefining-cancer-treatment/training_text.zip',sep='\|\|',

                        engine='python',

                       names=['ID','TEXT'],skiprows=1)

print('Number of data points: ',data_text.shape[0])

print('Number of features: ',data_text.shape[1])

print('Features: ',data_text.columns.values)

data_text.head()
stop_words = set(stopwords.words('english'))



def nlp_preprocessing(total_text,index,column):

    if type(total_text) is not int:

        string =''

        # replace every special char with space

        total_text = re.sub('[^a-zA-Z0-9\n]', ' ', total_text)

        # replace multiple spaces with single space

        total_text = re.sub('\s+',' ',total_text)

        # All text to lower case

        total_text = total_text.lower()

        

        for word in total_text.split():

            if not word in stop_words:

                string += word + ' '

        data_text[column][index] = string
start_time = time.clock()

for index, row in data_text.iterrows():

    if type(row['TEXT']) is str:

        nlp_preprocessing(row['TEXT'],index,'TEXT')

    else:

        print('there is no text description for id: ',index)

print('Time took for preprocessing the text : ',time.clock()-start_time,'seconds')
result = pd.merge(data,data_text,on='ID',how='left')

result.head()
result[result.isnull().any(axis=1)]
result.loc[result['TEXT'].isnull(),'TEXT'] = result['Gene'] +' '+ result['Variation']
result[result['ID']==1109]
y_true = result['Class'].values

result.Gene = result['Gene'].str.replace('\s+','_')

result.Variation = result['Variation'].str.replace('\s+','_')



X_train,test_df,y_train,y_test = train_test_split(result,y_true,stratify=y_true,test_size=0.2)

train_df,cv_df,y_train,y_cv = train_test_split(X_train,y_train,stratify=y_train,test_size=0.2)
print('Number of data points in train data:', train_df.shape[0])

print('Number of data points in test data:', test_df.shape[0])

print('Number of data points in cross validation data:', cv_df.shape[0])
train_class_distribution = train_df['Class'].value_counts().sort_index()

test_class_distribution = test_df['Class'].value_counts().sort_index()

cv_class_distribution = cv_df['Class'].value_counts().sort_index()



my_colors = 'rgbkymc'

train_class_distribution.plot(kind='bar')

plt.xlabel('Class')

plt.ylabel('Data points per class')

plt.title('Distribution of yi in train Data')

plt.grid()

plt.show()



sorted_yi = np.argsort(-train_class_distribution.values)

for i in sorted_yi:

    print('Number of data points in class', i+1, ':',train_class_distribution.values[i], '(', np.round((train_class_distribution.values[i]/train_df.shape[0]*100), 3), '%)')



    

test_class_distribution.plot(kind='bar')

plt.xlabel('Class')

plt.ylabel('Data points per class')

plt.title('Distribution of yi in test data')

plt.grid()

plt.show()



sorted_yi =np.argsort(-test_class_distribution.values)

for i in sorted_yi:

    print('Number of data points in class',i+1,':',test_class_distribution.values[i],'(',np.round((test_class_distribution.values[i]/test_df.shape[0] *100),3),'%)')

    

cv_class_distribution.plot(kind='bar')

plt.xlabel('Class')

plt.ylabel('Data points per class')

plt.title('Distribution of yi in cv data')

plt.grid()

plt.show()



sorted_yi = np.argsort(-cv_class_distribution.values)

for i in sorted_yi:

    print('Number of data points in class',i+1,':',cv_class_distribution.values[i],'(',np.round((cv_class_distribution.values[i]/cv_df.shape[0] *100),3),'%)')    
def plot_confusion_matrix(test_y,predict_y):

    C = confusion_matrix(test_y,predict_y)

    A = (((C.T)/(C.sum(axis=1))).T)

    B = (C/C.sum(axis=0))

    

    labels =[1,2,3,4,5,6,7,8,9]

    print('-'*20,'Confusion Matrix','-'*20)

    plt.figure(figsize=(20,7))

    sns.heatmap(C,annot=True,cmap="YlGnBu", fmt=".3f", xticklabels=labels, yticklabels=labels)

    plt.xlabel('Predicted Class')

    plt.ylabel('Original Class')

    plt.show()

    

    print('-'*20,'Precision Matrix (column sum=1)','-'*20)

    plt.figure(figsize=(20,7))

    sns.heatmap(A,annot=True,cmap='YlGnBu',fmt='.3f',xticklabels = labels,yticklabels=labels)

    plt.xlabel('Predicted Class')

    plt.ylabel('Original Class')

    plt.show()

    

    print('-'*20,'Recall Matrix (Row sum=1)','-'*20)

    plt.figure(figsize=(20,7))

    sns.heatmap(B,annot=True,cmap="YlGnBu", fmt=".3f", xticklabels=labels, yticklabels=labels)

    plt.xlabel('Predicted Class')

    plt.ylabel('Original Class')

    plt.show()
test_data_len =test_df.shape[0]

cv_data_len = cv_df.shape[0]

# cv set error

cv_predicted_y = np.zeros((cv_data_len,9))

for i in range(cv_data_len):

    rand_probs = np.random.rand(1,9)

    cv_predicted_y[i] = ((rand_probs/sum(sum(rand_probs)))[0])

print('Log loss on Cross valibation data using random model: ', log_loss(y_cv,cv_predicted_y,eps=1e-15))



# Test set error

test_predicted_y = np.zeros((test_data_len,9))

for i in range(test_data_len):

    rand_probs = np.random.rand(1,9)

    test_predicted_y[i] = ((rand_probs/sum(sum(rand_probs)))[0])

print('Log loss on test data using random model: ',log_loss(y_test,test_predicted_y,eps=1e-15))



predicted_y = np.argmax(test_predicted_y,axis=1)

plot_confusion_matrix(y_test,predicted_y)
unique_genes = train_df['Gene'].value_counts()

print('Number of Unique Genes: ',unique_genes.shape[0])

print(unique_genes.head(10))
s = sum(unique_genes.values)

h = unique_genes.values/s

plt.plot(h,label='Histogram of Genes')

plt.xlabel('Index of gene')

plt.ylabel('Number of occurences')

plt.legend()

plt.show()
c = np.cumsum(h)

plt.plot(c)

plt.show()
# code for response coding with laplase smoothing:

# alpha: used for laplase smoothing

# feature: ['gene','Variation']

# df: ['train_df','test_df','cv_df']

# get gene variation feature Dict

def get_gv_fea_dict(alpha,feature,df):

    value_count = train_df[feature].value_counts()

    gv_dict =dict()

    for i,denominator in value_count.items():

        vec =[]

        for k in range(1,10):

            # print(train_df.loc[(train_df['Class']==1) & (train_df['Gene']=='BRCA1')])

            cls_cnt = train_df.loc[(train_df['Class']==k) & (train_df[feature]==i)]

            vec.append((cls_cnt.shape[0]+alpha*10)/(denominator + 90*alpha))

            

        gv_dict[i] = vec

    return gv_dict



def get_gv_feature(alpha,feature,df):

    gv_dict= get_gv_fea_dict(alpha,feature,df)

    value_counts = train_df[feature].value_counts()

    gv_fea =[]

    

    for index,row in df.iterrows():

        if row[feature] in dict(value_counts).keys():

            gv_fea.append(gv_dict[row[feature]])

        else:

            gv_fea.append([1/9,1/9,1/9,1/9,1/9,1/9,1/9,1/9,1/9])

    return gv_fea
# response coding for gene feature

alpha =1

train_gene_feature_responseCoding = np.array(get_gv_feature(alpha,'Gene',train_df))

test_gene_feature_responseCoding = np.array(get_gv_feature(alpha,'Gene',test_df))

cv_gene_feature_responseCoding = np.array(get_gv_feature(alpha,'Gene',cv_df))
print('The shape of gene feature using response coding: ',train_gene_feature_responseCoding.shape)
# One hot encoding of Gene features

gene_vectorizer = CountVectorizer()

train_gene_feature_onehotCoding = gene_vectorizer.fit_transform(train_df['Gene'])

test_gene_feature_onehotCoding = gene_vectorizer.transform(test_df['Gene'])

cv_gene_feature_onehotCoding = gene_vectorizer.transform(cv_df['Gene'])
print('The shape of gene feature using one hot encoding: ',train_gene_feature_onehotCoding.shape)
train_df['Gene'].head()
gene_vectorizer.get_feature_names()
alpha = [10 ** x for x in range(-5,1)] # hyperparameter for SGD Classifier.

cv_log_error_array =[]

for i in alpha:

    clf = SGDClassifier(alpha=i,penalty='l2',loss='log',random_state=42)

    clf.fit(train_gene_feature_onehotCoding,y_train)

    sig_clf = CalibratedClassifierCV(clf,method='sigmoid')

    sig_clf.fit(train_gene_feature_onehotCoding,y_train)

    predict_y = sig_clf.predict_proba(cv_gene_feature_onehotCoding)

    cv_log_error_array.append(log_loss(y_cv,predict_y,labels=clf.classes_,eps=1e-15))

    print('For values of alpha :',i,'The log loss is:',log_loss(y_cv,predict_y,labels=clf.classes_,eps=1e-15))

    

    

fig,ax = plt.subplots()

ax.plot(alpha,cv_log_error_array,c='g')

for i,txt in enumerate(np.round(cv_log_error_array,3)):

    ax.annotate((alpha[i],np.round(txt,3)), (alpha[i],cv_log_error_array[i]))

plt.grid()

plt.title('Cross Validation Error for each alpha')

plt.xlabel('alpha i')

plt.ylabel('Error measure')

plt.show()



best_alpha = np.argmin(cv_log_error_array)

clf = SGDClassifier(alpha=alpha[best_alpha],penalty='l2',loss='log',random_state=42)

clf.fit(train_gene_feature_onehotCoding,y_train)

sig_clf = CalibratedClassifierCV(clf,method='sigmoid')

sig_clf.fit(train_gene_feature_onehotCoding,y_train)



predict_y = sig_clf.predict_proba(train_gene_feature_onehotCoding)

print('For value of best alpha : ',alpha[best_alpha], 'The train log loss is: ',log_loss(y_train,predict_y,labels=clf.classes_,eps=1e-15))

predict_y = sig_clf.predict_proba(cv_gene_feature_onehotCoding)

print('For value of best alpha : ',alpha[best_alpha], 'The cv log loss is: ',log_loss(y_cv,predict_y,labels=clf.classes_,eps=1e-15))

predict_y = sig_clf.predict_proba(test_gene_feature_onehotCoding)

print('For value of best alpha : ',alpha[best_alpha], 'The test log loss is: ',log_loss(y_test,predict_y,labels=clf.classes_,eps=1e-15))
test_coverage =test_df[test_df['Gene'].isin(list(set(train_df['Gene'])))].shape[0]

cv_coverage = cv_df[cv_df['Gene'].isin(list(set(train_df['Gene'])))].shape[0]



print('In test data: ',test_coverage,'out of: ',test_df.shape[0], ':',(test_coverage/test_df.shape[0])*100)

print('In cv data: ',cv_coverage,'out of: ',cv_df.shape[0],':',(cv_coverage/cv_df.shape[0])*100)
unique_variations = train_df['Variation'].value_counts()

print('Number of unique features: ',unique_variations.shape[0])

print(unique_variations.head(10))
s = sum(unique_variations.values)

h = unique_variations.values/s

plt.plot(h,label='Histogram of Variations')

plt.grid()

plt.legend()

plt.show()
c = np.cumsum(h)

plt.plot(c,label='Cumulative distribution of variation')

plt.grid()

plt.legend()

plt.show()

# alpha is used for laplace smoothing

alpha = 1

# train gene feature

train_variation_feature_responseCoding = np.array(get_gv_feature(alpha, "Variation", train_df))

# test gene feature

test_variation_feature_responseCoding = np.array(get_gv_feature(alpha, "Variation", test_df))

# cross validation gene feature

cv_variation_feature_responseCoding = np.array(get_gv_feature(alpha, "Variation", cv_df))
print("train_variation_feature_responseCoding is a converted feature using the response coding method. The shape of Variation feature:", train_variation_feature_responseCoding.shape)
# One hot encoding of variation feature

variation_vectorizer = CountVectorizer()

train_variation_feature_onehotCoding = variation_vectorizer.fit_transform(train_df['Variation'])

test_variation_feature_onehotCoding = variation_vectorizer.transform(test_df['Variation'])

cv_variation_feature_onehotCoding = variation_vectorizer.transform(cv_df['Variation'])
print("train_variation_feature_onehotEncoded is converted feature using the one-hot encoding method. The shape of Variation feature:", train_variation_feature_onehotCoding.shape)
alpha = [10 ** x for x in range(-5,1)]

cv_log_error_array=[]

for i in alpha:

    clf = SGDClassifier(alpha=i,penalty='l2',loss='log',random_state=42)

    clf.fit(train_variation_feature_onehotCoding,y_train)

    

    sig_clf = CalibratedClassifierCV(clf,method='sigmoid')

    sig_clf.fit(train_variation_feature_onehotCoding,y_train)

    predict_y = sig_clf.predict_proba(cv_variation_feature_onehotCoding)

    

    cv_log_error_array.append(log_loss(y_cv,predict_y,labels=clf.classes_,eps=1e-15))

    print('For values of alpha: ',i,'The log loss is: ',log_loss(y_cv, predict_y, labels=clf.classes_, eps=1e-15))

    

fig,ax = plt.subplots()

ax.plot(alpha,cv_log_error_array,c='g')

for i,txt in enumerate(np.round(cv_log_error_array,3)):

    ax.annotate((alpha[i],np.round(txt,3)),(alpha[i],cv_log_error_array[i]))

plt.grid()

plt.title('Cross validate error for each alpha')

plt.xlabel('Alpha i')

plt.ylabel('Error measure')

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

test_coverage = test_df[test_df['Variation'].isin(list(set(train_df['Variation'])))].shape[0]

cv_coverage = cv_df[cv_df['Variation'].isin(list(set(train_df['Variation'])))].shape[0]

print('In test data',test_coverage, 'out of',test_df.shape[0], ":",(test_coverage/test_df.shape[0])*100)

print('In cross validation data',cv_coverage, 'out of ',cv_df.shape[0],":" ,(cv_coverage/cv_df.shape[0])*100)
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
import math

#https://stackoverflow.com/a/1602964

def get_text_responsecoding(df):

    text_feature_responseCoding = np.zeros((df.shape[0],9))

    for i in range(0,9):

        row_index = 0

        for index, row in df.iterrows():

            sum_prob = 0

            for word in row['TEXT'].split():

                sum_prob += math.log(((dict_list[i].get(word,0)+10 )/(total_dict.get(word,0)+90)))

            text_feature_responseCoding[row_index][i] = math.exp(sum_prob/len(row['TEXT'].split()))

            row_index += 1

    return text_feature_responseCoding
# building a CountVectorizer with all the words that occured minimum 3 times in train data

text_vectorizer = CountVectorizer(min_df=3)

train_text_feature_onehotCoding = text_vectorizer.fit_transform(train_df['TEXT'])

# getting all the feature names (words)

train_text_features= text_vectorizer.get_feature_names()



# train_text_feature_onehotCoding.sum(axis=0).A1 will sum every row and returns (1*number of features) vector

train_text_fea_counts = train_text_feature_onehotCoding.sum(axis=0).A1



# zip(list(text_features),text_fea_counts) will zip a word with its number of times it occured

text_fea_dict = dict(zip(list(train_text_features),train_text_fea_counts))





print("Total number of unique words in train data :", len(train_text_features))
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
#response coding of text features

train_text_feature_responseCoding  = get_text_responsecoding(train_df)

test_text_feature_responseCoding  = get_text_responsecoding(test_df)

cv_text_feature_responseCoding  = get_text_responsecoding(cv_df)
# https://stackoverflow.com/a/16202486

# we convert each row values such that they sum to 1  

train_text_feature_responseCoding = (train_text_feature_responseCoding.T/train_text_feature_responseCoding.sum(axis=1)).T

test_text_feature_responseCoding = (test_text_feature_responseCoding.T/test_text_feature_responseCoding.sum(axis=1)).T

cv_text_feature_responseCoding = (cv_text_feature_responseCoding.T/cv_text_feature_responseCoding.sum(axis=1)).T
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
#https://stackoverflow.com/a/2258273/4084039

sorted_text_fea_dict = dict(sorted(text_fea_dict.items(), key=lambda x: x[1] , reverse=True))

sorted_text_occur = np.array(list(sorted_text_fea_dict.values()))
# Number of words for a given frequency.

print(Counter(sorted_text_occur))
# Train a Logistic regression+Calibration model using text features whicha re on-hot encoded

alpha = [10 ** x for x in range(-5, 1)]

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

    df_text_vec = CountVectorizer(min_df=3)

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
#Data preparation for ML models.



#Misc. functionns for ML models





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

    gene_count_vec = CountVectorizer()

    var_count_vec = CountVectorizer()

    text_count_vec = CountVectorizer(min_df=3)

    

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



train_gene_var_onehotCoding = hstack((train_gene_feature_onehotCoding,

                                      train_variation_feature_onehotCoding))

test_gene_var_onehotCoding = hstack((test_gene_feature_onehotCoding,test_variation_feature_onehotCoding))

cv_gene_var_onehotCoding = hstack((cv_gene_feature_onehotCoding,cv_variation_feature_onehotCoding))



train_x_onehotCoding = hstack((train_gene_var_onehotCoding, train_text_feature_onehotCoding)).tocsr()

train_y = np.array(list(train_df['Class']))



test_x_onehotCoding = hstack((test_gene_var_onehotCoding, test_text_feature_onehotCoding)).tocsr()

test_y = np.array(list(test_df['Class']))



cv_x_onehotCoding = hstack((cv_gene_var_onehotCoding, cv_text_feature_onehotCoding)).tocsr()

cv_y = np.array(list(cv_df['Class']))





train_gene_var_responseCoding = np.hstack((train_gene_feature_responseCoding,train_variation_feature_responseCoding))

test_gene_var_responseCoding = np.hstack((test_gene_feature_responseCoding,test_variation_feature_responseCoding))

cv_gene_var_responseCoding = np.hstack((cv_gene_feature_responseCoding,cv_variation_feature_responseCoding))



train_x_responseCoding = np.hstack((train_gene_var_responseCoding, train_text_feature_responseCoding))

test_x_responseCoding = np.hstack((test_gene_var_responseCoding, test_text_feature_responseCoding))

cv_x_responseCoding = np.hstack((cv_gene_var_responseCoding, cv_text_feature_responseCoding))

print("One hot encoding features :")

print("(number of data points * number of features) in train data = ", train_x_onehotCoding.shape)

print("(number of data points * number of features) in test data = ", test_x_onehotCoding.shape)

print("(number of data points * number of features) in cross validation data =", cv_x_onehotCoding.shape)
print(" Response encoding features :")

print("(number of data points * number of features) in train data = ", train_x_responseCoding.shape)

print("(number of data points * number of features) in test data = ", test_x_responseCoding.shape)

print("(number of data points * number of features) in cross validation data =", cv_x_responseCoding.shape)
import pickle

alpha = [0.00001,0.0001,0.001,0.01,0.1,1,10,100,1000]

cv_log_error_array =[]

for i in alpha:

    print('for alpha: ',i)

    clf = MultinomialNB(alpha=i)

    clf.fit(train_x_onehotCoding,train_y)

    sig_clf = CalibratedClassifierCV(clf,method ='sigmoid')

    sig_clf.fit(train_x_onehotCoding,train_y)

    sig_clf_probs = sig_clf.predict_proba(cv_x_onehotCoding)

    cv_log_error_array.append(log_loss(cv_y,sig_clf_probs,labels = clf.classes_,eps=1e-15))

    print('log loss: ',log_loss(cv_y,sig_clf_probs))

    

fig,ax = plt.subplots()

ax.plot(np.log10(alpha),cv_log_error_array,c='g')

for i ,txt in enumerate(np.round(cv_log_error_array,3)):

    ax.annotate((alpha[i],str(txt)),(np.log10(alpha[i]),cv_log_error_array[i]))

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

print('For values of best alpha = ', alpha[best_alpha], "The train log loss is:",log_loss(y_train, predict_y, labels=clf.classes_, eps=1e-15))

predict_y = sig_clf.predict_proba(cv_x_onehotCoding)

print('For values of best alpha = ', alpha[best_alpha], "The cross validation log loss is:",log_loss(y_cv, predict_y, labels=clf.classes_, eps=1e-15))

predict_y = sig_clf.predict_proba(test_x_onehotCoding)

print('For values of best alpha = ', alpha[best_alpha], "The test log loss is:",log_loss(y_test, predict_y, labels=clf.classes_, eps=1e-15))



# Variables that will be used in the end to make comparison table of all models

nb_train = log_loss(y_train, sig_clf.predict_proba(train_x_onehotCoding), labels=clf.classes_, eps=1e-15)

nb_cv = log_loss(y_cv, sig_clf.predict_proba(cv_x_onehotCoding), labels=clf.classes_, eps=1e-15)

nb_test = log_loss(y_test, sig_clf.predict_proba(test_x_onehotCoding), labels=clf.classes_, eps=1e-15)
clf = MultinomialNB(alpha = alpha[best_alpha])

clf.fit(train_x_onehotCoding, train_y)

sig_clf = CalibratedClassifierCV(clf, method="sigmoid")

sig_clf.fit(train_x_onehotCoding, train_y)

sig_clf_probs = sig_clf.predict_proba(cv_x_onehotCoding)

# to avoid rounding error while multiplying probabilites we use log-probability estimates

print("Log Loss :",log_loss(cv_y, sig_clf_probs))

print("Number of missclassified point :", np.count_nonzero((sig_clf.predict(cv_x_onehotCoding)- cv_y))/cv_y.shape[0])

plot_confusion_matrix(cv_y, sig_clf.predict(cv_x_onehotCoding.toarray()))
# Variables that will be used in the end to make comparison table of models

nb_misclassified = (np.count_nonzero((sig_clf.predict(cv_x_onehotCoding)- cv_y))/cv_y.shape[0])*100
test_point_index = 1

no_feature = 100

predicted_cls = sig_clf.predict(test_x_onehotCoding[test_point_index])

print("Predicted Class :", predicted_cls[0])

print("Predicted Class Probabilities:", np.round(sig_clf.predict_proba(test_x_onehotCoding[test_point_index]),4))

print("Actual Class :", test_y[test_point_index])

indices=np.argsort(-1*abs(clf.coef_))[predicted_cls-1][:,:no_feature]

print("-"*50)

get_impfeature_names(indices[0], test_df['TEXT'].iloc[test_point_index],test_df['Gene'].iloc[test_point_index],test_df['Variation'].iloc[test_point_index], no_feature)
alpha = [x for x in range(1,100,4)]

cv_log_error_array = []

for i in alpha:

    print("for alpha =", i)

    clf = KNeighborsClassifier(n_neighbors=i)

    clf.fit(train_x_responseCoding, train_y)

    sig_clf = CalibratedClassifierCV(clf, method="sigmoid")

    sig_clf.fit(train_x_responseCoding, train_y)

    sig_clf_probs = sig_clf.predict_proba(cv_x_responseCoding)

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

clf.fit(train_x_responseCoding, train_y)

sig_clf = CalibratedClassifierCV(clf, method="sigmoid")

sig_clf.fit(train_x_responseCoding, train_y)



predict_y = sig_clf.predict_proba(train_x_responseCoding)

print('For values of best alpha = ', alpha[best_alpha], "The train log loss is:",log_loss(y_train, predict_y, labels=clf.classes_, eps=1e-15))

predict_y = sig_clf.predict_proba(cv_x_responseCoding)

print('For values of best alpha = ', alpha[best_alpha], "The cross validation log loss is:",log_loss(y_cv, predict_y, labels=clf.classes_, eps=1e-15))

predict_y = sig_clf.predict_proba(test_x_responseCoding)

print('For values of best alpha = ', alpha[best_alpha], "The test log loss is:",log_loss(y_test, predict_y, labels=clf.classes_, eps=1e-15))



# Variables that will be used in the end to make comparison table of all models

knn_train = log_loss(y_train, sig_clf.predict_proba(train_x_responseCoding), labels=clf.classes_, eps=1e-15)

knn_cv = log_loss(y_cv, sig_clf.predict_proba(cv_x_responseCoding), labels=clf.classes_, eps=1e-15)

knn_test = log_loss(y_test, sig_clf.predict_proba(test_x_responseCoding), labels=clf.classes_, eps=1e-15)
clf = KNeighborsClassifier(n_neighbors=alpha[best_alpha])

predict_and_plot_confusion_matrix(train_x_responseCoding, train_y, cv_x_responseCoding, cv_y, clf)

# Variables that will be used in the end to make comparison table of models

knn_misclassified = (np.count_nonzero((sig_clf.predict(cv_x_responseCoding)- cv_y))/cv_y.shape[0])*100
## with Class Balancing

alpha = [10 ** x for x in range(-6,3)]

cv_log_error_array =[]

for i in alpha:

    print('for alpha: ',i)

    clf = SGDClassifier(class_weight='balanced',alpha=i,penalty='l2',loss='log',random_state=42)

    clf.fit(train_x_onehotCoding,train_y)

    sig_clf = CalibratedClassifierCV(clf,method='sigmoid')

    sig_clf.fit(train_x_onehotCoding,train_y)

    sig_clf_probs = sig_clf.predict_proba(cv_x_onehotCoding)

    cv_log_error_array.append(log_loss(cv_y,sig_clf_probs,labels=clf.classes_,eps=1e-15))

    print('Log loss: ',log_loss(cv_y,sig_clf_probs))

    

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

print('For values of best alpha : ',alpha[best_alpha],'The train log loss is: ',log_loss(y_train,predict_y,labels=clf.classes_,eps=1e-15))

predict_y = sig_clf.predict_proba(cv_x_onehotCoding)

print('For values of best alpha = ', alpha[best_alpha], "The cross validation log loss is:",log_loss(y_cv, predict_y, labels=clf.classes_, eps=1e-15))

predict_y = sig_clf.predict_proba(test_x_onehotCoding)

print('For values of best alpha = ', alpha[best_alpha], "The test log loss is:",log_loss(y_test, predict_y, labels=clf.classes_, eps=1e-15))



# Variables that will be used in the end to make comparison table of all models

lr_balance_train = log_loss(y_train, sig_clf.predict_proba(train_x_onehotCoding), labels=clf.classes_, eps=1e-15)

lr_balance_cv = log_loss(y_cv, sig_clf.predict_proba(cv_x_onehotCoding), labels=clf.classes_, eps=1e-15)

lr_balance_test = log_loss(y_test, sig_clf.predict_proba(test_x_onehotCoding), labels=clf.classes_, eps=1e-15)
clf = SGDClassifier(class_weight='balanced', alpha=alpha[best_alpha], penalty='l2', loss='log', random_state=42)

predict_and_plot_confusion_matrix(train_x_onehotCoding, train_y, cv_x_onehotCoding, cv_y, clf)



# Variables that will be used in the end to make comparison table of models

lr_balance_misclassified = (np.count_nonzero((sig_clf.predict(cv_x_onehotCoding)- cv_y))/cv_y.shape[0])*100
# from tabulate import tabulate

clf = SGDClassifier(class_weight='balanced', alpha=alpha[best_alpha], 

                    penalty='l2', loss='log', random_state=42)

clf.fit(train_x_onehotCoding,train_y)

test_point_index = 1

no_feature = 500

predicted_cls = sig_clf.predict(test_x_onehotCoding[test_point_index])

print("Predicted Class :", predicted_cls[0])

print("Predicted Class Probabilities:", np.round(sig_clf.predict_proba(test_x_onehotCoding[test_point_index]),4))

print("Actual Class :", test_y[test_point_index])

indices = np.argsort(-1*abs(clf.coef_))[predicted_cls-1][:,:no_feature]

print("-"*50)

get_impfeature_names(indices[0], test_df['TEXT'].iloc[test_point_index],test_df['Gene'].iloc[test_point_index],test_df['Variation'].iloc[test_point_index], no_feature)
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

print('For values of best alpha = ', alpha[best_alpha], "The train log loss is:",log_loss(y_train, predict_y, labels=clf.classes_, eps=1e-15))

predict_y = sig_clf.predict_proba(cv_x_onehotCoding)

print('For values of best alpha = ', alpha[best_alpha], "The cross validation log loss is:",log_loss(y_cv, predict_y, labels=clf.classes_, eps=1e-15))

predict_y = sig_clf.predict_proba(test_x_onehotCoding)

print('For values of best alpha = ', alpha[best_alpha], "The test log loss is:",log_loss(y_test, predict_y, labels=clf.classes_, eps=1e-15))



# Variables that will be used in the end to make comparison table of all models

lr_train = log_loss(y_train, sig_clf.predict_proba(train_x_onehotCoding), labels=clf.classes_, eps=1e-15)

lr_cv = log_loss(y_cv, sig_clf.predict_proba(cv_x_onehotCoding), labels=clf.classes_, eps=1e-15)

lr_test = log_loss(y_test, sig_clf.predict_proba(test_x_onehotCoding), labels=clf.classes_, eps=1e-15)
clf = SGDClassifier(alpha=alpha[best_alpha], penalty='l2', loss='log', random_state=42)

predict_and_plot_confusion_matrix(train_x_onehotCoding, train_y, cv_x_onehotCoding, cv_y, clf)



lr_misclassified = (np.count_nonzero((sig_clf.predict(cv_x_onehotCoding)- cv_y))/cv_y.shape[0])*100
clf = SGDClassifier(alpha=alpha[best_alpha], penalty='l2', loss='log', random_state=42)

clf.fit(train_x_onehotCoding,train_y)

test_point_index = 1

no_feature = 500

predicted_cls = sig_clf.predict(test_x_onehotCoding[test_point_index])

print("Predicted Class :", predicted_cls[0])

print("Predicted Class Probabilities:", np.round(sig_clf.predict_proba(test_x_onehotCoding[test_point_index]),4))

print("Actual Class :", test_y[test_point_index])

indices = np.argsort(-1*abs(clf.coef_))[predicted_cls-1][:,:no_feature]

print("-"*50)

get_impfeature_names(indices[0], test_df['TEXT'].iloc[test_point_index],test_df['Gene'].iloc[test_point_index],test_df['Variation'].iloc[test_point_index], no_feature)
test_point_index = 100

no_feature = 500

predicted_cls = sig_clf.predict(test_x_onehotCoding[test_point_index])

print("Predicted Class :", predicted_cls[0])

print("Predicted Class Probabilities:", np.round(sig_clf.predict_proba(test_x_onehotCoding[test_point_index]),4))

print("Actual Class :", test_y[test_point_index])

indices = np.argsort(-1*abs(clf.coef_))[predicted_cls-1][:,:no_feature]

print("-"*50)

get_impfeature_names(indices[0], test_df['TEXT'].iloc[test_point_index],test_df['Gene'].iloc[test_point_index],test_df['Variation'].iloc[test_point_index], no_feature)
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

print('For values of best alpha = ', alpha[best_alpha], "The train log loss is:",log_loss(y_train, predict_y, labels=clf.classes_, eps=1e-15))

predict_y = sig_clf.predict_proba(cv_x_onehotCoding)

print('For values of best alpha = ', alpha[best_alpha], "The cross validation log loss is:",log_loss(y_cv, predict_y, labels=clf.classes_, eps=1e-15))

predict_y = sig_clf.predict_proba(test_x_onehotCoding)

print('For values of best alpha = ', alpha[best_alpha], "The test log loss is:",log_loss(y_test, predict_y, labels=clf.classes_, eps=1e-15))



# Variables that will be used in the end to make comparison table of all models

svm_train = log_loss(y_train, sig_clf.predict_proba(train_x_onehotCoding), labels=clf.classes_, eps=1e-15)

svm_cv = log_loss(y_cv, sig_clf.predict_proba(cv_x_onehotCoding), labels=clf.classes_, eps=1e-15)

svm_test = log_loss(y_test, sig_clf.predict_proba(test_x_onehotCoding), labels=clf.classes_, eps=1e-15)
clf = SGDClassifier(alpha=alpha[best_alpha], penalty='l2', loss='hinge', random_state=42,class_weight='balanced')

predict_and_plot_confusion_matrix(train_x_onehotCoding, train_y,cv_x_onehotCoding,cv_y, clf)

# Variables that will be used in the end to make comparison table of models

svm_misclassified = (np.count_nonzero((sig_clf.predict(cv_x_onehotCoding)- cv_y))/cv_y.shape[0])*100
alpha =[100,200,500,1000,2000]

max_depth =[5,10]

cv_log_error_array=[]

for i in alpha:

    for j in max_depth:

        print('for n_estimators: ',i, ' and max depth: ',j)

        clf = RandomForestClassifier(n_estimators=i, criterion='gini', max_depth=j, random_state=42, n_jobs=-1)

        clf.fit(train_x_onehotCoding, train_y)

        sig_clf = CalibratedClassifierCV(clf, method="sigmoid")

        sig_clf.fit(train_x_onehotCoding, train_y)

        sig_clf_probs = sig_clf.predict_proba(cv_x_onehotCoding)

        cv_log_error_array.append(log_loss(cv_y, sig_clf_probs, labels=clf.classes_, eps=1e-15))

        print("Log Loss :",log_loss(cv_y, sig_clf_probs))

        

best_alpha = np.argmin(cv_log_error_array)

clf = RandomForestClassifier(n_estimators=alpha[int(best_alpha/2)], criterion='gini', max_depth=max_depth[int(best_alpha%2)], random_state=42, n_jobs=-1)

clf.fit(train_x_onehotCoding, train_y)

sig_clf = CalibratedClassifierCV(clf, method="sigmoid")

sig_clf.fit(train_x_onehotCoding, train_y)



predict_y = sig_clf.predict_proba(train_x_onehotCoding)

print('For values of best estimator = ', alpha[int(best_alpha/2)], "The train log loss is:",log_loss(y_train, predict_y, labels=clf.classes_, eps=1e-15))

predict_y = sig_clf.predict_proba(cv_x_onehotCoding)

print('For values of best estimator = ', alpha[int(best_alpha/2)], "The cross validation log loss is:",log_loss(y_cv, predict_y, labels=clf.classes_, eps=1e-15))

predict_y = sig_clf.predict_proba(test_x_onehotCoding)

print('For values of best estimator = ', alpha[int(best_alpha/2)], "The test log loss is:",log_loss(y_test, predict_y, labels=clf.classes_, eps=1e-15))



# Variables that will be used in the end to make comparison table of all models

rf_train = log_loss(y_train, sig_clf.predict_proba(train_x_onehotCoding), labels=clf.classes_, eps=1e-15)

rf_cv = log_loss(y_cv, sig_clf.predict_proba(cv_x_onehotCoding), labels=clf.classes_, eps=1e-15)

rf_test = log_loss(y_test, sig_clf.predict_proba(test_x_onehotCoding), labels=clf.classes_, eps=1e-15)
clf = RandomForestClassifier(n_estimators=alpha[int(best_alpha/2)], criterion='gini', max_depth=max_depth[int(best_alpha%2)], random_state=42, n_jobs=-1)

predict_and_plot_confusion_matrix(train_x_onehotCoding, train_y,cv_x_onehotCoding,cv_y, clf)

rf_misclassified = (np.count_nonzero((sig_clf.predict(cv_x_onehotCoding)- cv_y))/cv_y.shape[0])*100
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
alpha = [10,50,100,200,500,1000]

max_depth = [2,3,5,10]

cv_log_error_array = []

for i in alpha:

    for j in max_depth:

        print("for n_estimators =", i,"and max depth = ", j)

        clf = RandomForestClassifier(n_estimators=i, criterion='gini', max_depth=j, random_state=42, n_jobs=-1)

        clf.fit(train_x_responseCoding, train_y)

        sig_clf = CalibratedClassifierCV(clf, method="sigmoid")

        sig_clf.fit(train_x_responseCoding, train_y)

        sig_clf_probs = sig_clf.predict_proba(cv_x_responseCoding)

        cv_log_error_array.append(log_loss(cv_y, sig_clf_probs, labels=clf.classes_, eps=1e-15))

        print("Log Loss :",log_loss(cv_y, sig_clf_probs))

best_alpha = np.argmin(cv_log_error_array)

clf = RandomForestClassifier(n_estimators=alpha[int(best_alpha/4)], criterion='gini', max_depth=max_depth[int(best_alpha%4)], random_state=42, n_jobs=-1)

clf.fit(train_x_responseCoding, train_y)

sig_clf = CalibratedClassifierCV(clf, method="sigmoid")

sig_clf.fit(train_x_responseCoding, train_y)



predict_y = sig_clf.predict_proba(train_x_responseCoding)

print('For values of best alpha = ', alpha[int(best_alpha/4)], "The train log loss is:",log_loss(y_train, predict_y, labels=clf.classes_, eps=1e-15))

predict_y = sig_clf.predict_proba(cv_x_responseCoding)

print('For values of best alpha = ', alpha[int(best_alpha/4)], "The cross validation log loss is:",log_loss(y_cv, predict_y, labels=clf.classes_, eps=1e-15))

predict_y = sig_clf.predict_proba(test_x_responseCoding)

print('For values of best alpha = ', alpha[int(best_alpha/4)], "The test log loss is:",log_loss(y_test, predict_y, labels=clf.classes_, eps=1e-15))



# Variables that will be used in the end to make comparison table of all models

rf_response_train = log_loss(y_train, sig_clf.predict_proba(train_x_responseCoding), labels=clf.classes_, eps=1e-15)

rf_response_cv = log_loss(y_cv, sig_clf.predict_proba(cv_x_responseCoding), labels=clf.classes_, eps=1e-15)

rf_response_test = log_loss(y_test, sig_clf.predict_proba(test_x_responseCoding), labels=clf.classes_, eps=1e-15)
clf = RandomForestClassifier(max_depth=max_depth[int(best_alpha%4)], n_estimators=alpha[int(best_alpha/4)], criterion='gini', max_features='auto',random_state=42)

predict_and_plot_confusion_matrix(train_x_responseCoding, train_y,cv_x_responseCoding,cv_y, clf)



# Variables that will be used in the end to make comparison table of models

rf_response_misclassified = (np.count_nonzero((sig_clf.predict(cv_x_responseCoding)- cv_y))/cv_y.shape[0])*100
# read more about SGDClassifier() at http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDClassifier.html

# ------------------------------

# default parameters

# SGDClassifier(loss=’hinge’, penalty=’l2’, alpha=0.0001, l1_ratio=0.15, fit_intercept=True, max_iter=None, tol=None, 

# shuffle=True, verbose=0, epsilon=0.1, n_jobs=1, random_state=None, learning_rate=’optimal’, eta0=0.0, power_t=0.5, 

# class_weight=None, warm_start=False, average=False, n_iter=None)



# some of methods

# fit(X, y[, coef_init, intercept_init, …])	Fit linear model with Stochastic Gradient Descent.

# predict(X)	Predict class labels for samples in X.

# read more about support vector machines with linear kernals here http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html

# --------------------------------

# default parameters 

# SVC(C=1.0, kernel=’rbf’, degree=3, gamma=’auto’, coef0=0.0, shrinking=True, probability=False, tol=0.001, 

# cache_size=200, class_weight=None, verbose=False, max_iter=-1, decision_function_shape=’ovr’, random_state=None)



# Some of methods of SVM()

# fit(X, y, [sample_weight])	Fit the SVM model according to the given training data.

# predict(X)	Perform classification on samples in X.

# --------------------------------

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



log_error = log_loss(train_y, sclf.predict_proba(train_x_onehotCoding))

print("Log loss (train) on the stacking classifier :",log_error)



log_error = log_loss(cv_y, sclf.predict_proba(cv_x_onehotCoding))

print("Log loss (CV) on the stacking classifier :",log_error)



log_error = log_loss(test_y, sclf.predict_proba(test_x_onehotCoding))

print("Log loss (test) on the stacking classifier :",log_error)



print("Number of missclassified point :", np.count_nonzero((sclf.predict(test_x_onehotCoding)- test_y))/test_y.shape[0])

plot_confusion_matrix(test_y=test_y, predict_y=sclf.predict(test_x_onehotCoding))
# Variables that will be used in the end to make comparison table of all models

stack_train = log_error

stack_cv = log_error1

stack_test = log_error2

stack_misclassified = (np.count_nonzero((sclf.predict(test_x_onehotCoding)- test_y))/test_y.shape[0])*100
from prettytable import PrettyTable



# name of models

names = ['Naive Bayes','K-Nearest Neighbour','LR With Class Balancing',\

        'LR Without Class Balancing','Linear SVM',\

        'RF With One hot Encoding','RF With Response Coding']

# Training loss

train_loss = [nb_train,knn_train,lr_balance_train,lr_train,svm_train,rf_train,rf_response_train]



# cv loss

cv_loss = [nb_cv,knn_cv,lr_balance_cv,lr_cv,svm_cv,rf_cv,rf_response_cv]



## Test loss

test_loss = [nb_test,knn_test,lr_balance_test,lr_test,svm_test,rf_test,rf_response_test]



# Percentage Misclassified points

misclassified = [nb_misclassified,knn_misclassified,lr_balance_misclassified,lr_misclassified,svm_misclassified,\

                 rf_misclassified,rf_response_misclassified]



numbering = [1,2,3,4,5,6,7]



# Initializing prettytable

ptable = PrettyTable()

ptable.add_column("S.NO.",numbering)

ptable.add_column("MODEL",names)

ptable.add_column("Train_loss",train_loss)

ptable.add_column("CV_loss",cv_loss)

ptable.add_column("Test_loss",test_loss)

ptable.add_column("Misclassified(%)",misclassified)



# Printing the Table

print(ptable)