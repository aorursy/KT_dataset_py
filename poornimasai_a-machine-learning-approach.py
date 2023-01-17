#all imports



###     python package imports

import string

import csv

import re

from datetime import datetime

from nltk.stem import WordNetLemmatizer

from nltk.tokenize import wordpunct_tokenize, word_tokenize

from nltk.corpus import stopwords

from nltk import word_tokenize

from nltk.util import ngrams

import random

import math



import os

import itertools

import scipy.stats as stats

from statistics import *



import numpy as np

import pandas as pd

from datetime import date

from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer

from sklearn.model_selection import train_test_split

from sklearn.model_selection import RepeatedKFold

from sklearn.feature_selection import SelectKBest,chi2,RFE

from sklearn.metrics import precision_recall_fscore_support



from sklearn.linear_model import LogisticRegression

from sklearn.pipeline import Pipeline

from sklearn.svm import LinearSVC,SVC

from sklearn.linear_model import SGDClassifier

from sklearn.linear_model import Perceptron

from sklearn.naive_bayes import BernoulliNB, MultinomialNB, GaussianNB

from sklearn.neighbors import KNeighborsClassifier, NearestCentroid

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score

from sklearn.model_selection import cross_val_predict





from IPython.display import Markdown, display

import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D

import plotly

import plotly.offline as offline

import plotly.figure_factory as ff

offline.init_notebook_mode()

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

import plotly.graph_objs as go

from wordcloud import WordCloud, STOPWORDS

from PIL import Image



%matplotlib inline
def fileread(x,with_header):

    ds= open(x)

    incsv =  csv.reader(ds)

    columns = {}                                        #creating a dictionary with header and column values



    keys = []

    if with_header == 'Y':                              #read headers if the file comes with headers

        headers = next(incsv)

        for h in headers:

            l = h.lower().strip()

            columns[l] = []

            keys = [i.lower().strip() for i in headers]

    else:

        i = 1

        for h in headers:

            columns[str(i)] = []

            keys.append(str(i))

            i = i + 1

    for row in incsv:

        for h,v in zip(keys,row):

                columns[h].append(v)

    #close(incsv)



    return columns


#cleanup functions

def text_clean(txt):

    clean_txt = re.sub("[^A-Za-z0-9.!?@#$%^&*{}/|;':\",<>-_=?+ ]","",txt)

    return clean_txt





#duration split

def dur_in_mins(dur):

    durinmins=[]

    if dur in ["", " ", "00:00", "0:00", "00:0", "0:0"]:

        allmins = 0

        

    else:

        hours_mins = re.split(':', dur)

        for tyme in hours_mins:

            if tyme in [""," "]:

                durinmins.append(0)

            else:

                durinmins.append(int(tyme))

            if len(durinmins) == 3:

                allmins = (durinmins[0] * 60) + durinmins[1] + (durinmins[2] / 60)

            elif len(durinmins) == 2:

                allmins = (durinmins[0] * 60) + durinmins[1]

            elif len(durinmins) == 1:   #assumption that single number is hours

                allmins = (durinmins[0] * 60)

    return allmins





# Removing multiple blanks

def rm_multibl(strin):

    blrmd = re.sub(r'\s\s+',' ',strin)

    return blrmd





# Splitting values separated by multiple spaces

def brk_str(vec):

    vec_brk = re.split('  ', vec)

    veclist = [v for v in vec_brk  if rm_multibl(v) not  in [" ",""]]

    return veclist





            

# Convert datetime

def date_time_conv(datetime_str,fmt):

    datetime_vec = [datetime.strptime(string,fmt) for string in datetime_str]

    return datetime_vec





#lemmatize and remove punctuation

#Optional stopwords removal

def lemmatize(vect,stopind,posind):

    lemmatizer = WordNetLemmatizer()

    #lemma = [lemmatizer.lemmatize(re.sub('['+string.punctuation+']', '', word),pos=posind) for word in wordpunct_tokenize(vect) if word not in string.punctuation]

    lemma = [lemmatizer.lemmatize(re.sub('['+string.punctuation+']', '', word),pos=posind) for word in word_tokenize(vect)]

    lemma = [word for word in lemma  if word not in ['',' ']]

    if stopind == 'Y':

        stop = set(stopwords.words('english'))

        lemma = [i for i in lemma if i not in stop]

    return lemma
def cleanse(filename):

    

    filename['purpose_cl'] = []

    for lines in filename['purpose']:

        filename['purpose_cl'].append(text_clean(lines))

    

    filename['duration_in_mins'] = []

    for val in filename['duration']:

        filename['duration_in_mins'].append(dur_in_mins(val))

    

    

    filename['astronauts'] = []

    for mem in filename['crew']:

        filename['astronauts'].append(brk_str(mem))

    

    

    filename['vehicle_cl'] = []

    for veh in filename['vehicle']:

        filename['vehicle_cl'].append(rm_multibl(veh))

    

def assign_id(filename,colname):    

    filename['id'] = [i+1 for i in range(len(filename[colname]))]
#Easy parameter setting



#Easy parameter setting

testset_size = 0.3          #size of test set for initial classification

corp_ngram = (1,3)          #n gram range for the corpus of features

train_ngram = (1,3)         #n gram range for the training set

new_ngram = (1,3)           #n gram range for the unseen test set

orig_ngram = (1,3)          #n gram range for the original unclassified dataset

random_state = 420 #42 #1929137 #67327328          # random seed

chi2_select =  80  #70 #176  #number of k best features to retain during selection
#######################################  READ AND CLEAN BENCHMARK DATASET ###################################    

# Read, cleanse and transform the raw dataset 

# arguments - Path of file, with header or not

evaraw = fileread("../input/eva-classified/EVA_bm.csv",'Y')                                                 #benchmarked dataset

list(evaraw)

cleanse(evaraw)                                     # argument - file read in previous step

assign_id(evaraw,'purpose_cl')                      # argument - file read in previous step and any column name

evaraw['label'] = evaraw.pop('class')





#######################################  LEMMATIZATION ###################################

# arguments - text to be lemmatized, stop_word removal indicator, part of speech indicator

evaraw['purpose_lem']=[' '.join(lemmatize(val,'N','v')) for val in evaraw['purpose_cl']]



#convert the raw dataset into a dataframe and dataframe into array for further use

evaraw_df = pd.DataFrame(evaraw)

eva_array = evaraw_df.values
# Read corpus

def read_corpus(path,lemtiz,stopword_rm):

    lines = [line.lower() for file in os.listdir(path) for line in open(path + file,encoding='utf-8',errors='ignore')] 

    if lemtiz == 'Y': 

        verbs = [lemmatize(line.lower(),stopword_rm,'v') for line in lines]

        sentences = [' '.join(item) for item in verbs]

        nouns = [lemmatize(line.lower(),stopword_rm,'n') for line in sentences]

        lines = [' '.join(item) for item in nouns]

    return lines









# Bag of words

def bow(vec,lemtiz):

    if lemtiz == 'Y': 

        lemmatizer = WordNetLemmatizer()

        verbs = [lemmatize(line.lower(),'N','v') for line in vec]

        sentences = [' '.join(item) for item in verbs]

        nouns = [lemmatize(line,'N','n') for line in sentences]

        bow = [text for sublist in nouns for text in sublist if (not text.isdigit() and text.strip() not in ['',' '])]

    else:

        word_list = [word.lower() for line in vec for word in line.split()]

        bow = [text for text in word_list  if (not text.isdigit() and text.strip() not in ['',' '])]        

    return bow 









# Read corpus and apply tf-idf

def feat_eng_ext(vector,function='engineer',ngrams=(1,1),df_lowerlim=None,voc=None,hist_print='N',idf_ret='N'):

    

    vectorizer = TfidfVectorizer(ngram_range=ngrams,smooth_idf=True,min_df=df_lowerlim,vocabulary=voc,lowercase=True)

    feat_array = vectorizer.fit_transform(vector).toarray()

    idf = sorted(list(vectorizer.idf_))

    

    if hist_print == 'Y':

        # Plot idf values as histogram to determine freq distribution of words

        plt.hist(idf)    

        plt.show()  

        print('Number of features = ',len(idf),', (Documents, Features) = ',(feat_array.shape))

        print('Min idf = ',min(idf),', Max idf = ',max(idf),', Avg idf = ',mean(idf))

        print('Median idf = ',median(idf),', Mode idf = ',mode(idf),', Std Deviation idf = ',np.std(idf))



    if function.lower() == 'engineer':

        all_features = dict(zip(vectorizer.get_feature_names(), idf))

        if idf_ret == 'Y':

            return all_features,min(idf),max(idf),mean(idf),np.std(idf)

        else:

            return all_features

    else:

        return vectorizer, feat_array
#######################################  FEATURE ENGINEERING ###################################

#engineer features from general corpus (instead of training data, to reduce over-fitting)

# arguments Path of corpus, lemmatization indicator, stop word removal indicator 

#vect_list = read_corpus('./corpus/corpus1/','N','N') ## Original read_corpus reads many files from a folder



vect_list = [line.lower() for line in open('../input/eva-general-corpus/corpus.txt',encoding='utf-8',errors='ignore')] 



# arguments vector name, engineer or extract, ngram range, minimum document frequency, histogram print

corp_features,min_idf,max_idf,mean_idf,sd_idf = feat_eng_ext(vect_list,'engineer',corp_ngram,3,None,'N','Y')

print('Number of features extracted from corpus : ',len(corp_features))
#######################################  FEATURE SELECTION FROM FULL LIST ###################################

#New list to store relevant features

rel_features=[]

for key in corp_features:

        if min_idf+(2*sd_idf) < corp_features[key]  :

                if (not key.isdigit() and key.strip() not in ['',' ']):

                    rel_features.append(key)          

print('Number of relevant features selected from corpus : ',len(rel_features))
#######################################  FEATURE EXTRACTION FROM DATASET ###################################

# arguments vector name, ngram range, minimum document frequency, vocabulary, histogram print, idf return

vectorizer,txt_features = feat_eng_ext(evaraw_df.purpose_cl,'extract',train_ngram,1,rel_features,'N','N')

feature_names = vectorizer.get_feature_names()

features_shape = txt_features.shape
#######################################  TRAIN/TEST SPLIT ###################################

#train_test(evaraw_df,evaraw['id'],0.7)                            # 60/40 train/test

train_data,test_data,train_label,test_label=train_test_split(txt_features,eva_array[:,8],test_size=testset_size,random_state=random_state)
#######################################  FEATURE SELECTION FROM DATASET ###################################

ch2 = SelectKBest(chi2, k=chi2_select)

train_data=ch2.fit_transform(train_data, train_label)

test_data = ch2.transform(test_data)  

feature_names = [feature_names[i] for i in ch2.get_support(indices=True)]

print('Number of features after chi2 feature selection : ',len(feature_names))

print('\n Best',(chi2_select),'Features Selected:')

print('******************************************* \n')

print(feature_names)
vectorizer_kbest,kbest_features = feat_eng_ext(evaraw_df.purpose_cl,'extract',train_ngram,1,feature_names,'N','N')
clf_lbls = []



def fit_predict(clf,train_data,train_label,test_data):

    clf.fit(train_data, train_label)

    return clf.predict(test_data)



def which_clf(classifier):

    if classifier.lower() == 'logistic':

        clf = LogisticRegression()

    elif classifier.lower() == 'gaussiannb':

        clf = GaussianNB()

    elif classifier.lower() == 'bernoullinb':

        clf = BernoulliNB()

    elif classifier.lower() == 'svm':

        clf = LinearSVC()

    elif classifier.lower().replace(' ','') in ['knn','knearestneighbours','knearestneighbors']:

        clf = KNeighborsClassifier(n_neighbors=4) 

    elif classifier.lower().replace(' ','') in ['rocchio','centroid','nearestcentroid']:

        clf = NearestCentroid()

    elif classifier.lower().replace(' ','') == 'randomforest':

        clf = RandomForestClassifier(n_estimators=100) 

    elif classifier.lower() == 'perceptron':

        clf = Perceptron(max_iter=50)

    else:

        clf = 0

    return clf

    

def classify(classifier,train_data,train_label,test_data,test_label):

    clf = which_clf(classifier)

    if clf == 0:

        print(classifier,': No classifier function for this classifier')

        return 0,0

    else:

        predicted=fit_predict(clf,train_data,train_label,test_data)

        acc_score = accuracy_score(test_label,predicted)

        clf_lbls.append(classifier.upper())

        return predicted,acc_score

        

           

    

def all_classifiers(train_data,train_label,test_data,test_label,acc_scr_prt,acc_hist_print):

    

    clf_logreg,acc_logreg=classify('logistic',train_data,train_label,test_data,test_label)

    

    clf_gn_bayes,acc_gn_bayes=classify('GaussianNB',train_data,train_label,test_data,test_label)

    

    clf_bn_bayes,acc_bn_bayes=classify('BernoulliNB',train_data,train_label,test_data,test_label)

       

    clf_lin_svm,acc_lin_svm=classify('SVM',train_data,train_label,test_data,test_label)

    

    clf_knn,acc_knn=classify('KNN',train_data,train_label,test_data,test_label)

    

    clf_centroid,acc_centroid=classify('Rocchio',train_data,train_label,test_data,test_label)

    

    clf_randomf,acc_randomf=classify('Random forest',train_data,train_label,test_data,test_label)

    

    clf_perceptron,acc_perceptron=classify('perceptron',train_data,train_label,test_data,test_label)

    

    accuracy=[acc_logreg,acc_gn_bayes,acc_bn_bayes,acc_lin_svm,acc_knn,acc_centroid,acc_randomf,acc_perceptron]

    

    if acc_scr_prt == 'Y':  

           

        print('\n ********** ACCURACY SCORES ********** \n')

        

        print('From logistic Regression : ',acc_logreg)

        print('From Gaussian Naive Bayes : ',acc_gn_bayes)

        print('From Bernoulli Naive Bayes : ',acc_bn_bayes)

        print('From Linear SVM : ',acc_lin_svm)

        print('From KNN : ',acc_knn)

        print('From Nearest Cetroid(Rocchio) : ',acc_centroid)

        print('From Random forest',acc_randomf)

        print('From Perceptron : ',acc_perceptron)

        

  

    if acc_hist_print == 'Y':

        

        

        #y_pos = np.arange(len(clf_lbls))

        #plt.bar(y_pos, accuracy, align='center', alpha=0.5)

        #plt.xticks(y_pos, clf_lbls,rotation='vertical')

        #plt.ylabel('Classifier Accuracy score')

        #plt.title('Comparison of accuracy scores from different classifiers')

        #plt.show()

        

        plt.rcdefaults()

        plt.style.use('ggplot')

        fig, ax = plt.subplots()

        #print(plt.style.available)

        

        y_pos = np.arange(len(clf_lbls))

        plt.xticks(np.arange(min(accuracy), 1, .05),rotation=45)

        

        ax.barh(y_pos, accuracy,  align='center',alpha=0.8, color='turquoise')

        ax.set_yticks(y_pos)

        ax.set_yticklabels(clf_lbls)

        ax.invert_yaxis()  # labels read top-to-bottom

        ax.set_xlabel('Classifier Accuracy score')

        ax.set_title('Comparison of accuracy scores from different classifiers')

        

        plt.show()
# To run all classifiers without K fold 



all_classifiers(train_data,train_label,test_data,test_label,'Y','Y')
#Run K-fold classification



rkf = RepeatedKFold(n_splits=10, n_repeats=5, random_state=random_state)





def kfold(data,label,clf_list):

    classifier_accuracies =[]

    accuracy_sd = []

    for clf in clf_list:

        acc_list = []

        for train, test in rkf.split(data):

            train_data,test_data,train_label,test_label=data[train],data[test],label[train],label[test]

            classified,acc=classify(clf,train_data,train_label,test_data,test_label)

            acc_list.append(acc)

        classifier_accuracies.append(np.mean(acc_list))

        accuracy_sd.append(np.std(acc_list))

    return classifier_accuracies,accuracy_sd



clf_list=['logistic','GaussianNB','BernoulliNB','SVM','KNN','Rocchio','Random forest','perceptron']

classifier_accuracies,accuracy_sd=kfold(kbest_features,eva_array[:,8],clf_list)

clf_list=[i.upper() for i in clf_list]
display(Markdown("**CLASSIFIER ACCURACIES**"))

#display(Markdown("**********"))

for i in range(len(classifier_accuracies)):

    print(" Classifier: {0:25} Accuracy: {1:20} \t Standard Deviation: {2:20}".format(clf_list[i],classifier_accuracies[i],accuracy_sd[i]))

    



pltx = classifier_accuracies

plt.rcdefaults()

plt.style.use('ggplot')

fig, ax = plt.subplots()





y_pos = np.arange(len(clf_list))

plt.xticks(np.arange(min(pltx), 1, .05),rotation = 45)



ax.barh(y_pos, classifier_accuracies,  align='center',alpha=0.9 ,color='gold') 

ax.set_yticks(y_pos)

ax.set_yticklabels(clf_list)

ax.invert_yaxis()  # labels read top-to-bottom

ax.set_xlabel('Classifier Accuracy score from K fold validation')

ax.set_title('Comparison of mean accuracy scores from different classifiers')



plt.show()
max_acc_clf = clf_list[classifier_accuracies.index(max(classifier_accuracies))]

print('Maximum Accuracy classifier:',max_acc_clf,'\t Accuracy:' ,max(classifier_accuracies)*100,'%')

clf_fn = which_clf(max_acc_clf)
#######################################  READ AND CLEAN NEW DATASET ###################################    

# Read, cleanse and transform the raw dataset 

# arguments - Path of file, with header or not



evanew = fileread('../input/eva-2014-2015/eva2014-2015.csv','Y')                                                 #new dataset

cleanse(evanew)                                     # argument - file read in previous step

assign_id(evanew,'purpose_cl')                      # argument - file read in previous step and any column name

evanew['date_time'] = date_time_conv(evanew['date'],'%m/%d/%Y')

evanew['year'] = [dttm_obj.year for dttm_obj in evanew['date_time']]





#######################################  LEMMATIZATION NEW DATASET ###################################

# arguments - text to be lemmatized, stop_word removal indicator, part of speech indicator

evanew['purpose_lem']=[' '.join(lemmatize(val,'N','v')) for val in evanew['purpose_cl']]



#convert the raw dataset into a dataframe and dataframe into array for further use

evanew_df = pd.DataFrame(evanew)

evanew_array = evanew_df.values
#######################################  FEATURE EXTRACTION FROM NEW DATASET ###################################

# arguments vector name, ngram range, minimum document frequency, vocabulary, histogram print, idf return

vectorizer_new,txt_features_new = feat_eng_ext(evanew_df.purpose_cl,'extract',new_ngram,1,feature_names,'N','N')

feature_names_new = vectorizer_new.get_feature_names()

features_shape_new = txt_features_new.shape

 

new_lbls = np.asarray(['0','0','0','0','0','0','0','0','0','0','0','0','0','0'])

clf_predict_new = fit_predict(clf_fn,kbest_features,eva_array[:,8],txt_features_new)

acc_score = accuracy_score(new_lbls,clf_predict_new)

print('Accuracy of classification on new, unseen test set: ',acc_score * 100,'%')
#######################################  READ AND CLEAN ORIGINAL DATASET ###################################    

# Read, cleanse and transform the raw dataset 

# arguments - Path of file, with header or not

evaorig = fileread('../input/eva-cleaned/Extra-vehicular_Activity__EVA__-_US_and_Russia_cl.csv','Y')     #benchmarked dataset

cleanse(evaorig)                                     # argument - file read in previous step

assign_id(evaorig,'purpose_cl')                      # argument - file read in previous step and any column name

evaorig['date_time'] = date_time_conv(evaorig['date'],'%m/%d/%Y')

evaorig['year'] = [dttm_obj.year for dttm_obj in evaorig['date_time']]

evaorig['counter'] = 1





#######################################  LEMMATIZATION - ORIGINAL DATASET ###################################

# arguments - text to be lemmatized, stop_word removal indicator, part of speech indicator

evaorig['purpose_lem']=[' '.join(lemmatize(val,'N','v')) for val in evaorig['purpose_cl']]



#convert the raw dataset into a dataframe and dataframe into array for further use

evaorig_df = pd.DataFrame(evaorig)

evaorig_array = evaorig_df.values
vectorizer_orig,txt_features_orig = feat_eng_ext(evaorig_df.purpose_cl,'extract',orig_ngram,1,feature_names,'N','N')

feature_names_orig = vectorizer_orig.get_feature_names()

features_shape_orig = txt_features_orig.shape



clf_predict_orig = fit_predict(clf_fn,kbest_features,eva_array[:,8],txt_features_orig)

acc_score = accuracy_score(eva_array[:,8],clf_predict_orig)

precision_recall=precision_recall_fscore_support(eva_array[:,8],clf_predict_orig, average='macro')

print('Accuracy of classification on original, unclassified dataset: ',acc_score * 100,'%')

print('Precision: ',precision_recall[0],' Recall:',precision_recall[1])
evaorig_df['problem_label'] = clf_predict_orig.tolist()
vect_score,feature_score = feat_eng_ext(evaorig_df.purpose_cl,'extract',orig_ngram,1,rel_features,'N','N')

ch2_orig = ch2.transform(feature_score)



feature_df = pd.DataFrame(ch2_orig,columns=[re.sub(' ','_',feature) for feature in feature_names])



final_df = pd.concat([evaorig_df,feature_df],axis=1)

final_df['total_feature_value'] = np.sum(ch2_orig,axis=1).tolist()
numvars = ['cooling', 'due', 'due_to', 'ended', 'failed', 'helmet', 'lost', 'overboard', 'problem', 'vent']

duration = ['duration_in_mins']

counter=['counter']

key_grp1234 = ['country','year','vehicle','problem_label']

key_grp123 = ['country','year','vehicle']

key_grp124 = ['country','year','problem_label']

key_grp134 = ['country','vehicle','problem_label']

key_grp234 = ['year','vehicle','problem_label']

key_grp14 = ['country','problem_label']

key_grp24 = ['year','problem_label']

key_grp34 = ['vehicle','problem_label']

key_grp1 = ['country']

key_grp4 = ['problem_label']

key_grp12 = ['country','year']

key_grp13 = ['country','vehicle']

key_grp23 = ['year','vehicle']



def create_groups(df,groupcols):

    df = df[groupcols]

    df_g = pd.DataFrame({'count' : df.groupby(groupcols).size()}).reset_index()

    return df_g

group1 = create_groups(final_df[final_df['problem_label']=='1'],key_grp12)

group2 = create_groups(final_df[final_df['problem_label']=='1'],key_grp123)

group3 = create_groups(final_df,key_grp124)

#group3['country_year'] = [country + year for (country,year) in group3['country'],group3['year']]

#print(group3)
fig = ff.create_scatterplotmatrix(final_df[['problem_label','lost','due_to','failed','problem','total_feature_value']], diag='histogram', index='problem_label',height=700, width=700)

offline.iplot(fig, filename='Histograms along Diagonal Subplots')
# Group data together

#hist_data = [final_df['lost'],final_df['due_to'], final_df['problem'], final_df['helmet'],final_df['work'],final_df['untethered'],final_df['not']]



group_labels = ['cooling', 'due', 'due_to', 'ended', 'failed', 'helmet', 'lost', 'overboard', 'problem', 'vent']



#group_labels = [re.sub(' ','_',feature) for feature in feature_names]

hist_data = [final_df[i] for i in group_labels]



# Create distplot with custom bin_size

fig = ff.create_distplot(hist_data, group_labels, bin_size=.01, show_hist=False)



# Plot!

offline.iplot(fig, filename='Distplot with Multiple Features')


y1 = group3[(group3.country == 'USA') & (group3.problem_label == '0')]

y2 = group3[(group3.country == 'Russia') & (group3.problem_label == '0')]

y3 = group3[(group3.country == 'USA') & (group3.problem_label == '1')]

y4 = group3[(group3.country == 'Russia') & (group3.problem_label == '1')]



US_missions_without_problems = go.Bar(

    x=y1['year'],

    y=y1['count'],

    name='US missions without problems',

    text='USA missions with no problems encountered',

    textposition = 'auto',

    marker=dict(

        color='rgb(128,255,0)',

        line=dict(

            color='rgb(128,255,0)',

            width=1.5),

        ),

    opacity=0.8

)



Russia_missions_without_problems = go.Bar(

    x=y2['year'],

    y=y2['count'],

    name='Russia missions without problems',

    text='Russian missions with no problems encountered',

    textposition = 'auto',

    marker=dict(

        color='rgb(255,158,0)',

        line=dict(

            color='rgb(255,158,0)',

            width=1.5),

        ),

    opacity=0.6

)



US_missions_with_problems = go.Bar(

    x=y3['year'],

    y=y3['count'],

    name='US missions with problems',

    text='USA missions with problems encountered',

    textposition = 'auto',

    marker=dict(

        color='rgb(51,49,35)',

        line=dict(

            color='rgb(51,49,35)',

            width=1.5),

        ),

    opacity=0.6

)



Russia_missions_with_problems = go.Bar(

    x=y4['year'],

    y=y4['count'],

    name='Russia missions with problems',

    text='Russia missions with problems encountered',

    textposition = 'auto',

    marker=dict(

        color='rgb(255,51,51)',

        line=dict(

            color='rgb(255,51,51)',

            width=1.5),

        ),

    opacity=0.6

)



data = [US_missions_without_problems,Russia_missions_without_problems,US_missions_with_problems,Russia_missions_with_problems]



layout = go.Layout(

    title='Missions by country by year classified on problems encountered',

    xaxis=dict(

        title='Year of mission'

    ),

    yaxis=dict(

        title='Count of Missions'

    )

)



fig = go.Figure(data=data, layout=layout)

offline.iplot(fig)
def select_gram(allgrams,ngrams):

    gram=[]

    once=[]

    for grams in allgrams:

        if ngrams == 2: 

            selected_tup = [tup for tup in grams if (tup[0].lower() in problem_features) or (tup[1].lower() in problem_features)]

            oneset = [tup for tup in grams if tup[0].lower() in problem_features]

        elif ngrams == 3:

            selected_tup = [tup for tup in grams if (tup[0].lower() in problem_features) or (tup[2].lower() in problem_features)]

            oneset = [tup for tup in grams if tup[0].lower() in problem_features]

        gram.append(selected_tup)

        once.append(oneset)

    return gram, once
tokens = [text.split() for text in evaorig_df[evaorig_df.problem_label == '1'].purpose_cl]

bigrams = [list(ngrams(item,2)) for item in tokens]

trigrams = [list(ngrams(item,3)) for item in tokens]



#problem_features= ['able', 'able to', 'accidental', 'almost', 'array', 'bar', 'bay', 'but', 'by', 'caused', 'close', 'co2', 'common', 'cooling', 'cover', 'damage', 'data', 'did', 'did not', 'difficult', 'difficult to', 'due', 'due to', 'early', 'end of', 'ended', 'eye', 'facility', 'failed', 'failed to', 'failure', 'fogging', 'for future', 'four', 'helmet', 'in', 'in suit', 'into', 'latch', 'led', 'left', 'lost', 'main', 'materials', 'mission', 'move', 'not', 'opening', 'out of', 'overboard', 'partially', 'performed', 'prevented', 'prior', 'problem', 'replace', 'restraints', 'retrieve', 'seals', 'servicing', 'start', 'started', 'still', 'stuck', 'suit', 'test', 'that', 'time', 'time to', 'times', 'to', 'tool', 'two', 'untethered', 'vent', 'via', 'vision', 'was', 'work', 'wrong']

top20=['but', 'co2', 'cooling', 'data', 'due', 'due to', 'early', 'ended', 'eye', 'failed', 'helmet', 'in suit', 'latch', 'lost', 'not', 'overboard', 'prevented', 'problem', 'untethered', 'vent']

problem_features = top20 #feature_names     #numvars        

word_bigram,cloud2=select_gram(bigrams,2)

word_trigram,cloud3=select_gram(trigrams,3)

for_cloud = ' '.join(evaorig_df[evaorig_df.problem_label == '1'].purpose_cl)

cloud_text = ' '.join([text for lists in word_bigram for tup in lists for text in list(tup)])
#for_cloud_us = ' '.join(evaorig_df[(evaorig_df.problem_label == '1') & (evaorig_df.country == 'USA')].purpose_cl)

for_cloud = ' '.join(evaorig_df[evaorig_df.problem_label == '1'].purpose_cl)
mask=np.array(Image.open("../input/saturn/saturn.png"))

stopwords = set(STOPWORDS)

wc = WordCloud(background_color='white', mask=mask, stopwords = stopwords,max_font_size=50,collocations = False).generate(cloud_text)
def grey_color_func(word, font_size, position,orientation,random_state=None, **kwargs):

    return("hsl(265,100%%, %d%%)" % np.random.randint(49,51))



plt.figure( figsize=(11,9) )

#plt.imshow(wc, interpolation='bilinear')

plt.imshow(wc.recolor(color_func = grey_color_func),interpolation="bilinear")

plt.axis("off")

plt.show()
x0 = final_df[final_df.country == 'USA'].duration_in_mins

x1 = final_df[final_df.country == 'Russia'].duration_in_mins



trace1 = go.Histogram(

    x=x0,

    name='Histogram of EVA duration in minutes - USA',

    opacity=0.6

)

trace2 = go.Histogram(

    x=x1,

    name='Histogram of EVA duration in minutes - Russia',

    opacity=0.5

)



data = [trace1, trace2]

layout = go.Layout(barmode='overlay')

fig = go.Figure(data=data, layout=layout)



offline.iplot(fig, filename='overlaid histogram')
x0 = final_df[(final_df.country == 'USA') & (final_df.problem_label == '0')].duration_in_mins

x1 = final_df[(final_df.country == 'Russia') & (final_df.problem_label == '0')].duration_in_mins

x2 = final_df[(final_df.country == 'USA') & (final_df.problem_label == '1')].duration_in_mins

x3 = final_df[(final_df.country == 'Russia') & (final_df.problem_label == '1')].duration_in_mins



trace0 = go.Histogram(

    x=x0,

    marker=dict(

        color='#006633',

        line=dict(

            color='#006633',

            width=1.5),

        ),

    name='Duration in minutes (non-problem) USA',

    opacity=0.5

)

trace1 = go.Histogram(

    x=x1,

    marker=dict(

        color='#FF6666',

        line=dict(

            color='#FF6666',

            width=1.5),

        ),

    name='Duration in minutes (non-problem) Russia',

    opacity=0.6

)

trace2 = go.Histogram(

    x=x2,

    marker=dict(

        color='#FF8000',

        line=dict(

            color='#FF8000',

            width=1.5),

        ),

    name='Duration in minutes (problem) USA',

    opacity=0.5

)

trace3 = go.Histogram(

    x=x3,

    marker=dict(

        color='#004C99',

        line=dict(

            color='#004C99',

            width=1.5),

        ),

    name='Duration in minutes (problem) Russia',

    opacity=0.4

)



layout = go.Layout(barmode='overlay',

    title='EVA durations by country by year classified on problems encountered',

    xaxis=dict(

        title='Duration in minutes'

    ),

    yaxis=dict(

        title='Count'

    )

)



data = [trace0, trace1, trace2, trace3]

fig = go.Figure(data=data, layout=layout)



offline.iplot(fig, filename='overlaid histogram')
lost_gram = []

for item in tokens:

    punct_idx=[]

    for i in item:

        try:

            punctp_idx = i.index('.')

        except ValueError:

            punctp_idx = -1

        try:

            punctc_idx = i.index(',')

        except ValueError:

            punctc_idx = -1

        if punctp_idx != -1 or punctc_idx != -1:

            punct_idx.append(item.index(i))

    try:

            lost_idx = item.index('lost')

    except ValueError:

            lost_idx = -1

    punct_idx.sort()

    if lost_idx != -1:

        start = 0

        end = 0

        starts = [i for i in punct_idx if i < lost_idx]

        ends = [i for i in punct_idx if i > lost_idx]

        if len(starts) > 0:

            start = max(starts) + 1

        else:

            start = lost_idx

        if len(ends) > 0: 

            end = min(ends) + 1

        else:

            end = len(item)

        subset_list = item[start:end]

        lost_gram.append(' '.join(subset_list))
x0 = final_df[final_df.country == 'USA' ].duration_in_mins

x1 = final_df[final_df.country == 'Russia'].duration_in_mins

x2 = final_df[(final_df.country == 'USA') & (final_df.problem_label == '1')].duration_in_mins

x3 = final_df[(final_df.country == 'Russia') & (final_df.problem_label == '1')].duration_in_mins



trace0 = go.Box(

    x=x0,

    marker=dict(

        color='#330066',

        line=dict(

            color='#330066',

            width=1.5),

        ),

    name='USA - total',

    opacity=0.6

)

trace1 = go.Box(

    x=x1,

    marker=dict(

        color='#006633',

        line=dict(

            color='#006633',

            width=1.5),

        ),

    name='Russia - total',

    opacity=0.5

)



trace2 = go.Box(

    x=x2,

    marker=dict(

        color='#F9815D',

        line=dict(

            color='#F9815D',

            width=1.5),

        ),

    name='USA (problem)',

    opacity=0.6

)



trace3 = go.Box(

    x=x3,

    marker=dict(

        color='#FFF668',

        line=dict(

            color='#FFF668',

            width=1.5),

        ),

    name='Russia (problem)',

    opacity=0.5

)



layout = go.Layout(

    title='EVA durations by country',

    xaxis=dict(

        title='Duration in minutes'

    ),

    yaxis=dict(

        title='Country'

    )

)



data = [trace0, trace2, trace1, trace3]

fig = go.Figure(data=data, layout=layout)

offline.iplot(fig)
crew_mem_pmissions = []



for i in range(len(final_df)):

     for mem in final_df.astronauts[i]: 

            crew_mem_pmissions.append([final_df.country[i],final_df.year[i],mem,final_df.problem_label[i]])



crew_mem_pmissions_df = pd.DataFrame(crew_mem_pmissions)

crew_mem_pmissions_df.columns = ['country', 'year', 'astronaut', 'problem_label']

crew_mem_pmissions_df = crew_mem_pmissions_df.sort_values('country')



groupc1 = create_groups(crew_mem_pmissions_df[crew_mem_pmissions_df['problem_label']=='1'],['country', 'astronaut'])





y1 = groupc1[groupc1.country == 'USA'  ]

y2 = groupc1[groupc1.country == 'Russia']



trace0 = go.Bar(

    x=y1['astronaut'],

    y=y1['count'],

    name='Astronauts on problematic  US EVAs',

    text='Missions with problems',

    textposition = 'auto',

    marker=dict(

        color='rgb(0,255,128)',

        line=dict(

            color='rgb(0,255,128)',

            width=1.5),

        ),

    opacity=0.8

)



trace1 = go.Bar(

    x=y2['astronaut'],

    y=y2['count'],

    name='Astronauts on problematic Russia EVAs',

    text='Missions with problems',

    textposition = 'auto',

    marker=dict(

        color='rgb(255,102,102)',

        line=dict(

            color='rgb(255,102,102)',

            width=1.5),

        ),

    opacity=0.6

)





data = [trace0, trace1]



layout = go.Layout(

    title='Missions(problem) count by Astronaut',

    yaxis=dict(

        title='Count of Missions'

    )

)



fig = go.Figure(data=data, layout=layout)

offline.iplot(fig)