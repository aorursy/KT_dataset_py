# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



# Loading all required packages

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

from IPython.display import Image







from sklearn import model_selection

from sklearn.linear_model import LogisticRegression



%matplotlib inline



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
data_variants = pd.read_csv('../input/msk-redefining-cancer-treatment/training_variants')

data_text =pd.read_csv("../input/msk-redefining-cancer-treatment/training_text",sep="\|\|",engine="python",names=["ID","TEXT"],skiprows=1)
stop_words = set(stopwords.words('english'))



def data_text_preprocess(total_text, ind, col):

    # Remove int values from text data as that might not be imp

    if type(total_text) is not int:

        string = ""

        # replacing all special char with space

        total_text = re.sub('[^a-zA-Z0-9\n]', ' ', str(total_text))

        # replacing multiple spaces with single space

        total_text = re.sub('\s+',' ', str(total_text))

        # bring whole text to same lower-case scale.

        total_text = total_text.lower()

        

        for word in total_text.split():

        # if the word is a not a stop word then retain that word from text

            if not word in stop_words:

                string += word + " "

        

        data_text[col][ind] = string
for index, row in data_text.iterrows():

    if type(row['TEXT']) is str:

        data_text_preprocess(row['TEXT'], index, 'TEXT')
all_data = pd.merge(data_variants, data_text,on='ID', how='left')

all_data.head()
plt.figure(figsize=(16,8))



plt.subplot(233)

sns.countplot(y='Class',data=all_data)

plt.gca().xaxis.tick_bottom()

plt.title('Data count by Class')
def report_missing_data(df):

    '''

    IN: Dataframe 

    OUT: Dataframe with reported count of missing values, % missing per column and per total data

    '''

    

    missing_count_per_column = df.isnull().sum()

    missing_count_per_column = missing_count_per_column[missing_count_per_column>0]

    total_count_per_column = df.isnull().count()

    total_cells = np.product(df.shape)

    

    # Percent calculation

    percent_per_columnn = 100*missing_count_per_column/total_count_per_column

    percent_of_total = 100*missing_count_per_column/total_cells

    

    # Creating new dataframe for reporting purposes only

    missing_data = pd.concat([missing_count_per_column,

                              percent_per_columnn,

                              percent_of_total], axis=1, keys=['Total_Missing', 'Percent_per_column','Percent_of_total'])

        

    missing_data = missing_data.dropna()

    missing_data.index.names = ['Feature']

    missing_data.reset_index(inplace=True)



    

    

    return missing_data.sort_values(by ='Total_Missing',ascending=False)



missing_data = report_missing_data(all_data)

missing_data
all_data[all_data.isnull().any(axis=1)]
# Adding Variation column to TEXT

all_data.loc[all_data['TEXT'].isnull(),'TEXT'] = all_data['Gene'] + ' ' + all_data['Variation'] 
Image("../input/cancer-pics/pic11.png",height=800 , width=600)

y_true = all_data['Class'].values

all_data['Gene']      = all_data['Gene'].str.replace('\s+', '_')

all_data['Variation'] = all_data['Variation'].str.replace('\s+', '_')
# Breaking up all data: 80 Train / 20 Test

X_train, test_df, y_train, y_test = train_test_split(all_data, y_true, stratify = y_true, test_size=0.2)
# Breaking up test data: 80 Train / 20 Validation

train_df, cv_df, y_train, y_cv = train_test_split(X_train,y_train,stratify = y_train, test_size=0.2 )
print('Data points in train data:', train_df.shape[0])

print('Data points in test data:', test_df.shape[0])

print('Data points in cross validation data:', cv_df.shape[0])
train_set = []

cv_set = []

test_set = []



train_class_distribution = train_df['Class'].value_counts()

test_class_distribution = test_df['Class'].value_counts()

cv_class_distribution = cv_df['Class'].value_counts()



sorted_train = np.argsort(-train_class_distribution.values)

sorted_test = np.argsort(-test_class_distribution.values)

sorted_cv = np.argsort(-cv_class_distribution.values)



for i in sorted_train:

    train_set.append(np.round((train_class_distribution.values[i]/train_df.shape[0]*100), 3))

for i in sorted_test:

    test_set.append(np.round((test_class_distribution.values[i]/test_df.shape[0]*100),3))

for i in sorted_cv:

    cv_set.append(np.round((cv_class_distribution.values[i]/cv_df.shape[0]*100), 3))



distribution_per_set = pd.DataFrame(

    {

     'Train Set(%)': train_set,

     'CV Set(%)': cv_set,

     'Test Set(%)':test_set

    })



# Plotting Distribution per class 

distribution_per_set.index = distribution_per_set.index + 1

distribution_per_set.plot.bar(figsize=(12,6))

plt.xticks(rotation=0)

plt.title('Distribution of data per set and class')

plt.xlabel('Class')

plt.ylabel('% Of total data')





test_data_len = test_df.shape[0]

cv_data_len = cv_df.shape[0]



# we create a output array that has exactly same size as the CV data

cv_predicted_y = np.zeros((cv_data_len,9))

for i in range(cv_data_len):

    rand_probs = np.random.rand(1,9)

    cv_predicted_y[i] = ((rand_probs/rand_probs.sum())[0])



cv_log_loss = round(log_loss(y_cv,cv_predicted_y, eps=1e-15),2)



print("Log loss on Cross Validation Data using Random Model",cv_log_loss)



# Test-Set error.

#we create a output array that has exactly same as the test data

test_predicted_y = np.zeros((test_data_len,9))

for i in range(test_data_len):

    rand_probs = np.random.rand(1,9)

    test_predicted_y[i] = ((rand_probs/rand_probs.sum())[0])

test_log_loss = round(log_loss(y_test,test_predicted_y, eps=1e-15),2)



print("Log loss on Test Data using Random Model",test_log_loss)
predicted_y =np.argmax(test_predicted_y, axis=1)

# Since class values vary for 0-8. And we have 1-9. Apply n+1 formula to make it 1-9 

predicted_y = predicted_y+1
def plot_matrices(y_test,predicted_y):  



    confusion = confusion_matrix(y_test, predicted_y)

    precision =(confusion/confusion.sum(axis=0))

    recall =(((confusion.T)/(confusion.sum(axis=1))).T)

    

    f,(ax1,ax2,ax3, axcb) = plt.subplots(1,4, 

                gridspec_kw={'width_ratios':[1,1,1,0.08]},figsize=(22,6))

    

    labels = [1,2,3,4,5,6,7,8,9]

    

    g1 = sns.heatmap(confusion,cbar=False,ax=ax1,annot=True, cmap="Blues", fmt=".3f", xticklabels=labels, yticklabels=labels,)

    g1.set_ylabel('Original Class')

    g1.set_xlabel('Predicted Class')

    g1.set_title('Confusion')

    g2 = sns.heatmap(precision,cmap="Blues",cbar=False,ax=ax2, annot=True,fmt=".3f", xticklabels=labels, yticklabels=labels)

    g2.set_ylabel('Original Class')

    g2.set_xlabel('Predicted Class')

    g2.set_yticks(labels)

    g2.set_title('Precision')

    g3 = sns.heatmap(recall,cmap="Blues",ax=ax3, cbar_ax=axcb, annot=True, fmt=".3f", xticklabels=labels, yticklabels=labels)

    g3.set_ylabel('Original Class')

    g3.set_xlabel('Predicted Class')

    g3.set_title('Recall')

    g3.set_yticks(labels)

    

    for ax in [g1,g2,g3]:

        tl = ax.get_xticklabels()

        ax.set_xticklabels(tl, rotation=0)

        tly = ax.get_yticklabels()

        ax.set_yticklabels(tly, rotation=0)

    

    plt.show()



plot_matrices(y_test,predicted_y)
Image("../input/cancer-pics/pic22.png",height=800 , width=600)
def eval_alpha_loss(alpha,train_feat_hotencode,cv_feat_hotencode):

    """

    IN: Hyperparameter Alpha, Train_Feature_onehotencoded, CV_Feature_onehotencoded

    OUT: Hyperparameter Tunning DataFrame 

    """

    cv_log_error_array=[]

    for i in alpha:

        clf = SGDClassifier(alpha=i, penalty='l2', loss='log', random_state=42)

        clf.fit(train_feat_hotencode, y_train)

        sig_clf = CalibratedClassifierCV(clf, method="sigmoid")

    

    # 

        sig_clf.fit(train_feat_hotencode, y_train)

        predict_y = sig_clf.predict_proba(cv_feat_hotencode)

    

        cv_log_error_array.append(log_loss(y_cv, predict_y, labels=clf.classes_, eps=1e-15))



    temp_df = pd.DataFrame(data={'alpha': np.round(alpha,5), 

                                 'cv_log_loss': np.round(cv_log_error_array,5)})

    return temp_df





def eval_all_set(name,best_alpha,

                 train_feat_hotencode,

                 cv_feat_hotencode,

                 test_feat_hotencode):

    '''

    IN: Feature name, Best Alpha, and All 3 OneHotEncoded Sets 

    OUT: Log-Loss Report data frame

    '''

    # Model

    clf = SGDClassifier(alpha=best_alpha, penalty='l2', loss='log', random_state=42)

    clf.fit(train_feat_hotencode, y_train)

    sig_clf = CalibratedClassifierCV(clf, method="sigmoid")

    sig_clf.fit(train_feat_hotencode, y_train)

    

    train_predict_y = sig_clf.predict_proba(train_feat_hotencode)

    train_log_loss = np.round(log_loss(y_train, train_predict_y, labels=clf.classes_, eps=1e-15),3)



    cv_predict_y = sig_clf.predict_proba(cv_feat_hotencode)

    cv_log_loss = np.round(log_loss(y_cv, cv_predict_y, labels=clf.classes_, eps=1e-15),3)

    

    test_predict_y = sig_clf.predict_proba(test_feat_hotencode)

    test_log_loss = np.round(log_loss(y_test, test_predict_y, labels=clf.classes_, eps=1e-15),3)

    

    report_log_loss=[name,

                     best_alpha,

                     train_log_loss,

                     cv_log_loss,

                     test_log_loss]

    

    temp_df = pd.DataFrame([report_log_loss],columns=['Feature','best alpha','train_log_loss','cv_log_loss','test_log_loss' ])   

    return temp_df
# How many unique values ? 

unique_gene = train_df['Gene'].value_counts()

print ('Number of unique Genes:',unique_gene.shape[0])
total_unique_values = sum(unique_gene.values);

percent_per_total = unique_gene.values/total_unique_values;

cumulative = np.cumsum(percent_per_total)

plt.plot(cumulative,label='Cumulative distribution of Genes',)



plt.grid()

plt.axhline(0.75, color='k')

plt.legend()

plt.tight_layout()

plt.show()
# Vectorizing our 'Gene' feature

vectorizer = CountVectorizer()

train_gene_feature_onehotCoding =  vectorizer.fit_transform(train_df['Gene'])

test_gene_feature_onehotCoding  =  vectorizer.transform(test_df['Gene'])

cv_gene_feature_onehotCoding    =  vectorizer.transform(cv_df['Gene'])
# Evaluation Overalap

test_train_coverage=test_df[test_df['Gene'].isin(list(set(train_df['Gene'])))].shape[0]

cv_train_coverage=cv_df[cv_df['Gene'].isin(list(set(train_df['Gene'])))].shape[0]

test_train_overlap = np.round(test_train_coverage*100/test_df.shape[0],1)

cv_train_overlap =  np.round(cv_train_coverage*100/cv_df.shape[0],1)



overlap= pd.DataFrame(data=[[test_train_overlap,cv_train_overlap]],columns=['Test-Train Data Overlap[%]','CV-Train Data Overlap[%]'])



# Evaluating Gene Feature

alpha = [10 ** x for x in range(-5, 1)]

# Tunning Hyper Parameter (Alpha)

tunning_df = eval_alpha_loss(alpha,train_gene_feature_onehotCoding,cv_gene_feature_onehotCoding)

# Selecting Best Alpha

best_alpha = tunning_df.loc[tunning_df['cv_log_loss'] == tunning_df['cv_log_loss'].min(), 'alpha'].item()



# Calculating Log_Loss for all test sets

gene_feat = eval_all_set('Gene',best_alpha,

                         train_gene_feature_onehotCoding,

                         cv_gene_feature_onehotCoding,

                         test_gene_feature_onehotCoding)



# Combining Report

gene_report=pd.concat([gene_feat,overlap],axis=1)

gene_report
# How many unique values 

unique_variation = train_df['Variation'].value_counts()

print ('Number of unique Variation:',unique_variation.shape[0])
total_unique_values = sum(unique_variation.values);

percent_per_total = unique_variation.values/total_unique_values;

cumulative = np.cumsum(percent_per_total)

plt.plot(cumulative,label='Cumulative distribution of Genes',)



plt.grid()

plt.axhline(0.80, color='k')

plt.legend()

plt.tight_layout()

plt.show()
vectorizer = CountVectorizer()



train_variation_feature_onehotCoding =  vectorizer.fit_transform(train_df['Variation'])

test_variation_feature_onehotCoding  =  vectorizer.transform(test_df['Variation'])

cv_variation_feature_onehotCoding    =  vectorizer.transform(cv_df['Variation'])
test_train_coverage=test_df[test_df['Variation'].isin(list(set(train_df['Variation'])))].shape[0]

cv_train_coverage=cv_df[cv_df['Variation'].isin(list(set(train_df['Variation'])))].shape[0]

test_train_overlap = np.round(test_train_coverage*100/test_df.shape[0],1)

cv_train_overlap =  np.round(cv_train_coverage*100/cv_df.shape[0],1)



overlap= pd.DataFrame(data=[[test_train_overlap,cv_train_overlap]],columns=['Test-Train Data Overlap[%]','CV-Train Data Overlap[%]'])



# Evaluating Gene Feature

alpha = [10 ** x for x in range(-5, 1)]

# Tunning Hyper Parameter (Alpha)

tunning_df = eval_alpha_loss(alpha,train_gene_feature_onehotCoding,cv_gene_feature_onehotCoding)

# Selecting Best Alpha

best_alpha = tunning_df.loc[tunning_df['cv_log_loss'] == tunning_df['cv_log_loss'].min(), 'alpha'].item()



# Calculating Log_Loss for all test sets

feat_rep = eval_all_set('Variation',best_alpha,

                         train_variation_feature_onehotCoding,

                         cv_variation_feature_onehotCoding,

                         test_variation_feature_onehotCoding)



# Combining Report

variation_report=pd.concat([feat_rep,overlap],axis=1)

variation_report
# building a CountVectorizer with all the words that occured minimum 3 times in train data

text_vectorizer = CountVectorizer(min_df=3)

train_text_feature_onehotCoding = text_vectorizer.fit_transform(train_df['TEXT'])



# getting all the feature names (words)

train_text_features= text_vectorizer.get_feature_names()

print("Total number of unique words in train data :", len(train_text_features))
# Normalizing One_Hot_Encoding



# we use the same vectorizer that was trained on train data

train_text_feature_onehotCoding = normalize(train_text_feature_onehotCoding, axis=0)



test_text_feature_onehotCoding = text_vectorizer.transform(test_df['TEXT'])

test_text_feature_onehotCoding = normalize(test_text_feature_onehotCoding, axis=0)



cv_text_feature_onehotCoding = text_vectorizer.transform(cv_df['TEXT'])

cv_text_feature_onehotCoding = normalize(cv_text_feature_onehotCoding, axis=0)
# train_text_feature_onehotCoding.sum(axis=0).A1 will sum every row and returns (1*number of features) vector

# .A1 turns(compresses) Matrix into Array

train_text_fea_counts = train_text_feature_onehotCoding.sum(axis=0).A1



# We will use it to check overlaps between data sets

text_fea_dict = dict(zip(list(train_text_features),train_text_fea_counts))



# Sorting dictionary based on the value ( not key ) 

sorted_text_fea_dict = dict(sorted(text_fea_dict.items(), key=lambda x: x[1] , reverse=True))

sorted_text_occur = np.array(list(sorted_text_fea_dict.values()))



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

test_train_overlap =np.round((len2/len1)*100, 1)

len1,len2 = get_intersec_text(cv_df)

cv_train_overlap = np.round((len2/len1)*100, 1)
overlap= pd.DataFrame(data=[[test_train_overlap, cv_train_overlap]],columns=['Test-Train Data Overlap[%]','CV-Train Data Overlap[%]'])



# Evaluating Gene Feature

alpha = [10 ** x for x in range(-5, 1)]

# Tunning Hyper Parameter (Alpha)

tunning_df = eval_alpha_loss(alpha,train_gene_feature_onehotCoding,cv_gene_feature_onehotCoding)

# Selecting Best Alpha

best_alpha = tunning_df.loc[tunning_df['cv_log_loss'] == tunning_df['cv_log_loss'].min(), 'alpha'].item()



# Calculating Log_Loss for all test sets

feat_rep = eval_all_set('TEXT',best_alpha,

                         train_text_feature_onehotCoding,

                         cv_text_feature_onehotCoding,

                         test_text_feature_onehotCoding)



# Combining Report

text_report=pd.concat([feat_rep,overlap],axis=1)
all_features = pd.concat([gene_report,variation_report,text_report],axis=0)

all_features
Image("../input/cancer-pics/pic31.png",height=800 , width=600)
def report_log_loss(train_x, train_y, test_x, test_y,  clf):

    clf.fit(train_x, train_y)

    sig_clf = CalibratedClassifierCV(clf, method="sigmoid")

    sig_clf.fit(train_x, train_y)

    sig_clf_probs = sig_clf.predict_proba(test_x)

    return log_loss(test_y, sig_clf_probs, eps=1e-15)









def eval_alpha_model_loss(alpha,train_feat_hotencode,cv_feat_hotencode):

    """

    IN: Hyperparameter Alpha, Train_Feature_onehotencoded, CV_Feature_onehotencoded

    OUT: Hyperparameter Tunning DataFrame 

    """

    cv_log_error_array=[]

    for i in alpha:

        clf = SGDClassifier(alpha=i, penalty='l2', loss='log', random_state=42)

        clf.fit(train_feat_hotencode, y_train)

        sig_clf = CalibratedClassifierCV(clf, method="sigmoid")

        sig_clf.fit(train_feat_hotencode, y_train)

        predict_y = sig_clf.predict_proba(cv_feat_hotencode)

    

        cv_log_error_array.append(log_loss(y_cv, predict_y, labels=clf.classes_, eps=1e-15))



    temp_df = pd.DataFrame(data={'alpha': np.round(alpha,5), 

                                 'cv_log_loss': np.round(cv_log_error_array,5)})

    return temp_df





def plot_matrices(y_test,predicted_y):  



    confusion = confusion_matrix(y_test, predicted_y)

    precision =(confusion/confusion.sum(axis=0))

    recall =(((confusion.T)/(confusion.sum(axis=1))).T)

    

    f,(ax1,ax2,ax3, axcb) = plt.subplots(1,4, 

                gridspec_kw={'width_ratios':[1,1,1,0.08]},figsize=(22,6))

    

    labels = [1,2,3,4,5,6,7,8,9]

    

    g1 = sns.heatmap(confusion,cbar=False,ax=ax1,annot=True, cmap="Blues", fmt=".3f", xticklabels=labels, yticklabels=labels,)

    g1.set_ylabel('Class')

    g1.set_xlabel('Class')

    g1.set_title('Confusion')

    g2 = sns.heatmap(precision,cmap="Blues",cbar=False,ax=ax2, annot=True,fmt=".3f", xticklabels=labels, yticklabels=labels)

    g2.set_ylabel('Class')

    g2.set_xlabel('Class')

    g2.set_yticks(labels)

    g2.set_title('Precision')

    g3 = sns.heatmap(recall,cmap="Blues",ax=ax3, cbar_ax=axcb, annot=True, fmt=".3f", xticklabels=labels, yticklabels=labels)

    g3.set_ylabel('Class')

    g3.set_xlabel('Class')

    g3.set_title('Recall')

    g3.set_yticks(labels)

    

    for ax in [g1,g2,g3]:

        tl = ax.get_xticklabels()

        ax.set_xticklabels(tl, rotation=0)

        tly = ax.get_yticklabels()

        ax.set_yticklabels(tly, rotation=0)  

    plt.show()   



def predict_and_plot_confusion_matrix(train_x, train_y,test_x, test_y, clf):

    clf.fit(train_x, train_y)

    sig_clf = CalibratedClassifierCV(clf, method="sigmoid")

    sig_clf.fit(train_x, train_y)

    pred_y = sig_clf.predict(test_x)



    # for calculating log_loss we willl provide the array of probabilities belongs to each class

    # calculating the number of data points that are misclassified

    plot_matrices(test_y, pred_y)    



def model_performance(name,clf,best_alpha,

                 train_X_hotencode,

                 cv_X_hotencode,

                 test_X_hotencode):

    '''

    IN: Model name, Classifier, Best Alpha, and All 3 OneHotEncoded Sets 

    OUT: Log-Loss Report data frame

    '''

    # Model

    clf = clf

    clf.fit(train_X_hotencode, train_y)

    sig_clf = CalibratedClassifierCV(clf, method="sigmoid")

    sig_clf.fit(train_X_hotencode, train_y)

    

    train_predict_y = sig_clf.predict_proba(train_X_hotencode)

    train_log_loss = np.round(log_loss(y_train, train_predict_y, labels=clf.classes_, eps=1e-15),3)



    cv_predict_y = sig_clf.predict_proba(cv_X_hotencode)

    cv_log_loss = np.round(log_loss(y_cv, cv_predict_y, labels=clf.classes_, eps=1e-15),3)

    

    test_predict_y = sig_clf.predict_proba(test_X_hotencode)

    test_log_loss = np.round(log_loss(y_test, test_predict_y, labels=clf.classes_, eps=1e-15),3)

    

    pred_y = sig_clf.predict(test_X_hotencode)

    

    miss_class = np.count_nonzero((pred_y- test_y))/test_y.shape[0]

    

    

    report_log_loss=[name,

                     best_alpha,

                     train_log_loss,

                     cv_log_loss,

                     test_log_loss,

                     miss_class]

    

    temp_df = pd.DataFrame([report_log_loss],columns=['Model','best alpha','train_log_loss','cv_log_loss','test_log_loss','Miss_classified(%)' ])   

    return temp_df





train_gene_var_onehotCoding = hstack((train_gene_feature_onehotCoding,train_variation_feature_onehotCoding))

test_gene_var_onehotCoding = hstack((test_gene_feature_onehotCoding,test_variation_feature_onehotCoding))

cv_gene_var_onehotCoding = hstack((cv_gene_feature_onehotCoding,cv_variation_feature_onehotCoding))



train_x_onehotCoding = hstack((train_gene_var_onehotCoding, train_text_feature_onehotCoding)).tocsr()

train_y = np.array(list(train_df['Class']))



test_x_onehotCoding = hstack((test_gene_var_onehotCoding, test_text_feature_onehotCoding)).tocsr()

test_y = np.array(list(test_df['Class']))



cv_x_onehotCoding = hstack((cv_gene_var_onehotCoding, cv_text_feature_onehotCoding)).tocsr()

cv_y = np.array(list(cv_df['Class']))
alpha = [0.00001, 0.0001, 0.001, 0.1, 1, 10, 100,1000]

cv_log_error_array = []

for i in alpha:

    #print("for alpha =", i)

    clf = MultinomialNB(alpha=i)

    clf.fit(train_x_onehotCoding, train_y)

    sig_clf = CalibratedClassifierCV(clf, method="sigmoid")

    sig_clf.fit(train_x_onehotCoding, train_y)

    sig_clf_probs = sig_clf.predict_proba(cv_x_onehotCoding)

    

    # to avoid rounding error while multiplying probabilites we use log-probability estimates

    cv_log_error_array.append(log_loss(cv_y, sig_clf_probs, labels=clf.classes_, eps=1e-15))

    

    

temp_df = pd.DataFrame(data={'alpha': np.round(alpha,5), 'cv_log_error': np.round(cv_log_error_array,5)})

temp_df.sort_values(by ='cv_log_error',ascending=True)
best_alpha = temp_df.loc[temp_df['cv_log_error'] == temp_df['cv_log_error'].min(), 'alpha'].item()



# Model 

clf = MultinomialNB(alpha=best_alpha)



# Calculating Log_Loss for all test sets

NB_report = model_performance('Naive Bayes',

                             clf,

                             best_alpha,

                         train_x_onehotCoding,

                         cv_x_onehotCoding,

                         test_x_onehotCoding)

NB_report
print ('Naive Bayse')

predict_and_plot_confusion_matrix(train_x_onehotCoding, train_y, cv_x_onehotCoding, cv_y, clf)
alpha = [10 ** x for x in range(-6, 3)]

cv_log_error_array = []

for i in alpha:

    clf = SGDClassifier(class_weight='balanced', alpha=i, penalty='l2', loss='log', random_state=42)

    clf.fit(train_x_onehotCoding, train_y)

    sig_clf = CalibratedClassifierCV(clf, method="sigmoid")

    sig_clf.fit(train_x_onehotCoding, train_y)

    sig_clf_probs = sig_clf.predict_proba(cv_x_onehotCoding)

    

    # to avoid rounding error while multiplying probabilites we use log-probability estimates

    cv_log_error_array.append(log_loss(cv_y, sig_clf_probs, labels=clf.classes_, eps=1e-15))

    

    

temp_df = pd.DataFrame(data={'alpha': np.round(alpha,5), 'cv_log_error': np.round(cv_log_error_array,3)})

temp_df.sort_values(by ='cv_log_error',ascending=True)
best_alpha = temp_df.loc[temp_df['cv_log_error'] == temp_df['cv_log_error'].min(), 'alpha'].item()





clf = SGDClassifier(class_weight='balanced', alpha=best_alpha, penalty='l2', loss='log', random_state=42)



LR_report = model_performance('Logistic Regression',

                             clf,

                             best_alpha,

                         train_x_onehotCoding,

                         cv_x_onehotCoding,

                         test_x_onehotCoding)

LR_report
predict_and_plot_confusion_matrix(train_x_onehotCoding, train_y, cv_x_onehotCoding, cv_y, clf)
alpha = [10 ** x for x in range(-6, 3)]

cv_log_error_array = []

for i in alpha:

    clf = SGDClassifier(class_weight=None, alpha=i, penalty='l2', loss='log', random_state=42)

    clf.fit(train_x_onehotCoding, train_y)

    sig_clf = CalibratedClassifierCV(clf, method="sigmoid")

    sig_clf.fit(train_x_onehotCoding, train_y)

    sig_clf_probs = sig_clf.predict_proba(cv_x_onehotCoding)

    

    # to avoid rounding error while multiplying probabilites we use log-probability estimates

    cv_log_error_array.append(log_loss(cv_y, sig_clf_probs, labels=clf.classes_, eps=1e-15))

    

    

temp_df = pd.DataFrame(data={'alpha': np.round(alpha,5), 'cv_log_error': np.round(cv_log_error_array,3)})

temp_df.sort_values(by ='cv_log_error',ascending=True)
best_alpha = temp_df.loc[temp_df['cv_log_error'] == temp_df['cv_log_error'].min(), 'alpha'].item()





clf = SGDClassifier(class_weight=None,alpha=best_alpha, penalty='l2', loss='log', random_state=42, )



LR_NoBal_report = model_performance('Logistic Regression (No Weight Balance)',

                             clf,

                             best_alpha,

                         train_x_onehotCoding,

                         cv_x_onehotCoding,

                         test_x_onehotCoding)

LR_NoBal_report
predict_and_plot_confusion_matrix(train_x_onehotCoding, train_y, cv_x_onehotCoding, cv_y, clf)
alpha = [100,200,500,1000,1500]

max_depth = [5, 10]



cv_log_error_array = []

n_estimators=[]

depth=[]

for i in alpha:

    for j in max_depth:

        clf = RandomForestClassifier(n_estimators=i, criterion='gini', max_depth=j, random_state=42, n_jobs=-1)

        clf.fit(train_x_onehotCoding, train_y)

        sig_clf = CalibratedClassifierCV(clf, method="sigmoid")

        sig_clf.fit(train_x_onehotCoding, train_y)

        sig_clf_probs = sig_clf.predict_proba(cv_x_onehotCoding)

        cv_log_error_array.append(log_loss(cv_y, sig_clf_probs, labels=clf.classes_, eps=1e-15))

        depth.append(j)

        n_estimators.append(i)

        

temp_df = pd.DataFrame(data={'n_estimators': np.round(n_estimators), 'max_depth': np.round(depth),'cv_log_error': np.round(cv_log_error_array,5)})

temp_df.sort_values(by ='cv_log_error',ascending=True)

best_estimator = temp_df.loc[temp_df['cv_log_error'] == temp_df['cv_log_error'].min(), 'n_estimators'].item()

best_depth = temp_df.loc[temp_df['cv_log_error'] == temp_df['cv_log_error'].min(), 'max_depth'].item()

best_alpha ='n_estimators=',str(best_estimator),'max depth=',str(best_depth)

best_alpha
best_alpha ='n_estimators=',str(best_estimator),'max depth=',str(best_depth)

best_alpha





clf = RandomForestClassifier(n_estimators=best_estimator, 

                             max_depth=best_depth,

                             criterion='gini',

                             random_state=42,

                             n_jobs=-1)



RF_report = model_performance('Random Forest',

                             clf,

                             best_alpha,

                         train_x_onehotCoding,

                         cv_x_onehotCoding,

                         test_x_onehotCoding)

RF_report
predict_and_plot_confusion_matrix(train_x_onehotCoding, train_y, cv_x_onehotCoding, cv_y, clf)
#Naive Bayse

clf1 = MultinomialNB(alpha=0.1)

clf1.fit(train_x_onehotCoding, train_y)

sig_clf1 = CalibratedClassifierCV(clf1, method="sigmoid")

#Logistic Regression

clf2 = SGDClassifier(alpha=0.001, class_weight='balanced',  penalty='l2', loss='log', random_state=42)

clf2.fit(train_x_onehotCoding, train_y)

sig_clf2 = CalibratedClassifierCV(clf2, method="sigmoid")

# Random Forest

clf3 = RandomForestClassifier(n_estimators=best_estimator,max_depth=best_depth, criterion='gini',random_state=42,n_jobs=-1)

clf3.fit(train_x_onehotCoding, train_y)

sig_clf3 = CalibratedClassifierCV(clf3, method="sigmoid")



sig_clf1.fit(train_x_onehotCoding, train_y)

sig_clf2.fit(train_x_onehotCoding, train_y)

sig_clf3.fit(train_x_onehotCoding, train_y)



alpha = [0.0001,0.001,0.01,0.1,1,10] 

cv_log_error_array=[]

for i in alpha:

    lr = LogisticRegression(C=i)

    sclf = StackingClassifier(classifiers=[sig_clf1, sig_clf2, sig_clf3], meta_classifier=lr, use_probas=True)

    sclf.fit(train_x_onehotCoding, train_y)

    cv_log_error_array.append(log_loss(cv_y, sclf.predict_proba(cv_x_onehotCoding)))

    

    

    

temp_df = pd.DataFrame(data={'alpha': np.round(alpha,5), 'cv_log_error': np.round(cv_log_error_array,3)})

temp_df.sort_values(by ='cv_log_error',ascending=True)
best_alpha = temp_df.loc[temp_df['cv_log_error'] == temp_df['cv_log_error'].min(), 'alpha'].item()



lr = LogisticRegression(C=best_alpha)

sclf = StackingClassifier(classifiers=[sig_clf1, sig_clf2, sig_clf3], meta_classifier=lr, use_probas=True)

sclf.fit(train_x_onehotCoding, train_y)



train_log_error = log_loss(train_y, sclf.predict_proba(train_x_onehotCoding))

cv_log_error = log_loss(cv_y, sclf.predict_proba(cv_x_onehotCoding))

test_log_error = log_loss(test_y, sclf.predict_proba(test_x_onehotCoding))



miss_class = np.count_nonzero((sclf.predict(test_x_onehotCoding)- test_y))/test_y.shape[0]



name = "Stacked (NB,LG,RF)"





report_log_loss=[name,

                     best_alpha,

                     train_log_error,

                     cv_log_error,

                     test_log_error,

                     miss_class]

    

stacked_report = pd.DataFrame([report_log_loss],columns=['Model','best alpha','train_log_loss','cv_log_loss','test_log_loss','Miss_classified(%)' ])   
all_models = pd.concat([NB_report,

                        LR_report,

                        LR_NoBal_report,

                        RF_report,

                        stacked_report])



all_models = all_models.sort_values(by ='Miss_classified(%)',ascending=True)

print(all_models.to_string())



plt.figure(figsize=(10,6))

sns.barplot(y='Model',x='Miss_classified(%)',data=all_models, alpha=0.7)

plt.title('ML Model Performance')

plt.tight_layout()
submission_text = pd.read_csv('../input/msk-redefining-cancer-treatment/stage2_test_text.csv',sep="\|\|",engine="python",names=["ID","TEXT"],skiprows=1)

submission_var =  pd.read_csv('../input/msk-redefining-cancer-treatment/stage2_test_variants.csv')
submiss_data = pd.merge(submission_var, submission_text,on='ID', how='left')
gene_old_onehotCoding =  vectorizer.fit_transform(train_df['Gene'])

gene_onehotCoding =  vectorizer.transform(submiss_data['Gene'])



variation_old_onehotCoding =  vectorizer.fit_transform(train_df['Variation'])

variation_onehotCoding =  vectorizer.transform(submiss_data['Variation'])



text_old_onehotCoding =  text_vectorizer.fit_transform(train_df['TEXT'])

text_onehotCoding = text_vectorizer.transform(submiss_data['TEXT'])



text_old_onehotCoding = normalize(text_old_onehotCoding, axis=0)

text_onehotCoding = normalize(text_onehotCoding, axis=0)



old_gene_var_onehotCoding = hstack((gene_old_onehotCoding,variation_old_onehotCoding))

old_onehotCoding = hstack((old_gene_var_onehotCoding, text_old_onehotCoding)).tocsr()



sub_test_gene_var_onehotCoding = hstack((gene_onehotCoding,variation_onehotCoding))

sub_test_onehotCoding = hstack((sub_test_gene_var_onehotCoding, text_onehotCoding)).tocsr()



best_alpha = 0.001





clf = SGDClassifier(class_weight='balanced', alpha=best_alpha, penalty='l2', loss='log', random_state=42)



LR_report = model_performance('Logistic Regression',

                             clf,

                             best_alpha,

                         train_x_onehotCoding,

                         cv_x_onehotCoding,

                         test_x_onehotCoding)

LR_report
clf = SGDClassifier(class_weight='balanced', alpha=best_alpha, penalty='l2', loss='log', random_state=42)

clf.fit(old_onehotCoding, train_y)

sig_clf = CalibratedClassifierCV(clf, method="sigmoid")

sig_clf.fit(old_onehotCoding, train_y)



result = sig_clf.predict_proba(sub_test_onehotCoding)
result_df = pd.DataFrame(result,columns=['Class1','Class2','Class3','Class4','Class5',

                                        'Class6','Class7','Class8','Class9'])

result_df['ID'] = submiss_data['ID']
cols = result_df.columns.tolist()

cols = cols[-1:] + cols[:-1]

result_df = result_df[cols]

result_df.to_csv('personalized_med_submission.csv',index=False)
