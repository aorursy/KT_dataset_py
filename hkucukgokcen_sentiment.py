# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import nltk

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
A=[]

with open('../input/amazon_cells_labelled.txt') as infile:

    for lineno, line in enumerate(infile): # enumerate returns the iteration number along with the item.

        A.append(line) 

    A = pd.DataFrame(A)

A['source']='Amazon'



I=[]

with open('../input/imdb_labelled.txt') as infile:

    for lineno, line in enumerate(infile): # enumerate returns the iteration number along with the item.

        I.append(line) 

    I = pd.DataFrame(I)

I['source']='Imdb'



Y=[]

with open('../input/yelp_labelled.txt') as infile:

    for lineno, line in enumerate(infile): # enumerate returns the iteration number along with the item.

        Y.append(line) 

    Y = pd.DataFrame(Y)

Y['source']='Yelp'



df= pd.concat([A,I,Y])



df.rename(columns = {0 : 'Sentences'},inplace = True)



d = list(map(lambda x: int(x[-2]),df['Sentences'] ))

s = list(map(lambda x: x[:-3].strip().lower(),df['Sentences'] ))



data= pd.DataFrame()





data['Sentences'] = s

data['Label'] = d

data['source']= list(df['source'])



del s,d,A,I,Y



data.head(2)
data[['Sentences','Label','source']].groupby(by=['Label','source']).count()
data['raw_lengths'] = data['Sentences'].apply(len)



data.raw_lengths.plot(bins=10,kind = 'hist')



data.hist(column='raw_lengths',by=['source','Label'],bins=10, figsize=(10,8),sharey=True)
data.groupby(by=['source','Label'])[['raw_lengths']].describe()
#check the outliers for raw_length

print(data[data.raw_lengths == 149]['Sentences'].iloc[0])

print(' ')

print(data[data.raw_lengths == 11]['Sentences'].iloc[0])
import numpy

data1 = data.raw_lengths



bins = numpy.linspace(0, max(data1), 10)

digitized = numpy.digitize(data1, bins,right = True)

bin_means = [[i, data1[digitized == i].mean()] for i in range(1, len(bins))]



bin_means = pd.DataFrame(bin_means)



bin_means
data['bin_length'] = digitized



data = pd.merge(left = data, right = bin_means, left_on = 'bin_length', right_on = 0)



data = data.drop(0,axis = 1)



data.rename(columns = {1 : 'bin_length_means'}, inplace = True)



data.head(2)
data.hist(column='bin_length_means',by =['source','Label'],bins=9,figsize = (10,8),sharey=True)
import string



punctuation_signs = string.punctuation



punctuation_signs
import collections as ct

r = []



for i in range(len(data)):

    t = sum(v for c, v in ct.Counter(data['Sentences'][i]).items() if c in punctuation_signs)

    r.append(t)

    

data['length_punctuation'] = r



print(data.head(2))



data.hist(column='length_punctuation',by ='Label',bins=10,figsize = (10,4),sharey=True,sharex=True)
#description for length of punctuation



data['length_punctuation'].describe()
#where length of punctuation in one review is 0

data[(data['length_punctuation'] == 0) & (data['Label'] == 0)]
#case with max length

data[data['length_punctuation']==19]['Sentences']
import string

from nltk.corpus import stopwords

from nltk.stem import WordNetLemmatizer



def text_mining(sentence):



    wo_punctuation = [text for text in sentence if text not in string.punctuation]

    wo_punctuation = ''.join(wo_punctuation)



    wo_punct_stopw = [text for text in wo_punctuation.split() if text not in stopwords.words('english')]

    wo_punct_stopw = ' '.join(wo_punct_stopw)



    token=nltk.word_tokenize(wo_punct_stopw)

    tagged=nltk.pos_tag(token, lang='eng')

    wnl=WordNetLemmatizer()



    final_words=[]

    lemmatized=[]



    for i in range(len(tagged)):

        if tagged[i][1].startswith('NN'):

            newtag='n'

        elif tagged[i][1].startswith('JJ'):

            newtag='a'

        elif tagged[i][1].startswith('V'):

            newtag='v'    

        elif tagged[i][1].startswith('R'):

            newtag='r'

        else:

            newtag=''

        if (newtag!=''):

            k=wnl.lemmatize(tagged[i][0],newtag)

            final_words.append(k)

            lemmatized=' '.join(final_words)



    

    return final_words



data['Final_words'] = data[['Sentences']].apply(text_mining, axis = 1)



data.head()
import string

from nltk.corpus import stopwords

from nltk.stem import WordNetLemmatizer



def lemmatization(sentence):



    wo_punctuation = [text for text in sentence if text not in string.punctuation]

    wo_punctuation = ''.join(wo_punctuation)



    wo_punct_stopw = [text for text in wo_punctuation.split() if text not in stopwords.words('english')]

    wo_punct_stopw = ' '.join(wo_punct_stopw)



    token=nltk.word_tokenize(wo_punct_stopw)

    tagged=nltk.pos_tag(token, lang='eng')

    wnl=WordNetLemmatizer()



    final_words=[]

    lemmatized=[]



    for i in range(len(tagged)):

        if tagged[i][1].startswith('NN'):

            newtag='n'

        elif tagged[i][1].startswith('JJ'):

            newtag='a'

        elif tagged[i][1].startswith('V'):

            newtag='v'    

        elif tagged[i][1].startswith('R'):

            newtag='r'

        else:

            newtag=''

        if (newtag!=''):

            k=wnl.lemmatize(tagged[i][0],newtag)

            final_words.append(k)

            lemmatized=' '.join(final_words)



    

    return lemmatized



data['lemmatized'] = data[['Sentences']].apply(lemmatization, axis = 1)



data.head()
import string

from nltk.corpus import stopwords

from nltk.stem import WordNetLemmatizer

from sklearn.metrics import classification_report

from nltk.corpus import sentiwordnet as swn

from sklearn.metrics import confusion_matrix



def sentiWordNet(sentence):



    wo_punctuation = [text for text in sentence if text not in string.punctuation]

    wo_punctuation = ''.join(wo_punctuation)



    wo_punct_stopw = [text for text in wo_punctuation.split() if text not in stopwords.words('english')]

    wo_punct_stopw = ' '.join(wo_punct_stopw)



    token=nltk.word_tokenize(wo_punct_stopw)

    tagged=nltk.pos_tag(token, lang='eng')

    wnl=WordNetLemmatizer()



    u=[]

    sent_score=[]



    for i in range(len(tagged)):

        if tagged[i][1].startswith('NN'):

            newtag='n'

        elif tagged[i][1].startswith('JJ'):

            newtag='a'

        elif tagged[i][1].startswith('V'):

            newtag='v'    

        elif tagged[i][1].startswith('R'):

            newtag='r'

        else:

            newtag=''

        if (newtag!=''):

            k=(wnl.lemmatize(tagged[i][0],newtag),newtag)

            u.append(k)

            score=0.0

            synsets= list(swn.senti_synsets(k[0]))

            if (len(synsets) > 0) :

                h=[]

                for syn in synsets:

                    score+=syn.pos_score()-syn.neg_score()

                    h.append(score)

                    sent_score.append(max(h))

    if (len(sent_score)==0 or len(sent_score)==1):

        return (float(0.0))

    else:

        return (sum([word_score for word_score in sent_score])/(len(sent_score)))
import string

from nltk.corpus import stopwords

from nltk.stem import WordNetLemmatizer

from sklearn.metrics import classification_report

from nltk.corpus import sentiwordnet as swn

from sklearn.metrics import confusion_matrix



def sentiWordNet(sentence):



    wo_punctuation = [text for text in sentence if text not in string.punctuation]

    wo_punctuation = ''.join(wo_punctuation)



    wo_punct_stopw = [text for text in wo_punctuation.split() if text not in stopwords.words('english')]

    wo_punct_stopw = ' '.join(wo_punct_stopw)



    token=nltk.word_tokenize(wo_punct_stopw)

    tagged=nltk.pos_tag(token, lang='eng')

    wnl=WordNetLemmatizer()



    u=[]

    sent_score=[]



    for i in range(len(tagged)):

        if tagged[i][1].startswith('NN'):

            newtag='n'

        elif tagged[i][1].startswith('JJ'):

            newtag='a'

        elif tagged[i][1].startswith('V'):

            newtag='v'    

        elif tagged[i][1].startswith('R'):

            newtag='r'

        else:

            newtag=''

        if (newtag!=''):

            k=(wnl.lemmatize(tagged[i][0],newtag),newtag)

            u.append(k)

            score=0.0

            synsets= list(swn.senti_synsets(k[0]))

            if (len(synsets) > 0) :

                h=[]

                for syn in synsets:

                    score+=syn.pos_score()-syn.neg_score()

                    h.append(score)

                    sent_score.append(max(h))

    if (len(sent_score)==0 or len(sent_score)==1):

        return (float(0.0))

    else:

        return (sum([word_score for word_score in sent_score])/(len(sent_score)))

    

sentiwordnet_perf = data.copy()



sentiwordnet_perf['sentiWordNet_scr'] = sentiwordnet_perf[['Sentences']].apply(sentiWordNet, axis = 1)





a = []

for i in range(len(sentiwordnet_perf)):

    if list(sentiwordnet_perf['sentiWordNet_scr'])[i] > 0.20:  

        w = 1

    else:

        w = 0

    a.append(w)  

    

sentiwordnet_perf['sentiWordNet_scr_binary'] = a 





print(classification_report(sentiwordnet_perf['Label'], sentiwordnet_perf['sentiWordNet_scr_binary'], digits=4))



print(confusion_matrix(sentiwordnet_perf['Label'], sentiwordnet_perf['sentiWordNet_scr_binary']))



sentiwordnet_perf.head(1)



data['sentiWordNet_scr'] = data[['Sentences']].apply(sentiWordNet, axis = 1)





b = []

for i in range(len(data)):

    if data['sentiWordNet_scr'][i] > 0.50:  

        t = 1

    elif data['sentiWordNet_scr'][i] < -0.02:

        t = 0

    else:

        t = -9

    b.append(t)  

    

data['sentiWordNet_scr_binary'] = b 



data.head()
r = data[(data['sentiWordNet_scr_binary'] != -9) & (data['source'] == 'Amazon')]



len(r)



print(classification_report(r['Label'], r['sentiWordNet_scr_binary'], digits=4))



print(confusion_matrix(r['Label'], r['sentiWordNet_scr_binary']))
import nltk

from nltk.sentiment.vader import SentimentIntensityAnalyzer



def SentimentIntensity(sentence):



    g = []

    index = []

    for i in range(len(sentence)):

        sid = SentimentIntensityAnalyzer()

        result = sid.polarity_scores(sentence[i])

        m = result['compound']

        g.append(m)

    return g
sentiintensity_perf = data.copy()



sentiintensity_perf['sent_intensity_scr'] = SentimentIntensity(list(sentiintensity_perf['Sentences']))



a = []

for i in range(len(sentiintensity_perf)):

    if list(sentiintensity_perf['sent_intensity_scr'])[i] >  0 :  #conservative davrandık asagısı ve indet alan 0

        w = 1

    else:

        w = 0

    a.append(w)  

    

sentiintensity_perf['sent_intensity_scr_binary'] = a 



print(classification_report(sentiintensity_perf['Label'], sentiintensity_perf['sent_intensity_scr_binary'], digits=4))



print(confusion_matrix(sentiintensity_perf['Label'], sentiintensity_perf['sent_intensity_scr_binary']))



sentiintensity_perf.head(2)
data['sent_intensity_scr'] = SentimentIntensity(data['Sentences'])



b = []

for i in range(len(data)):

    if data['sent_intensity_scr'][i] >  0.50:  #conservative davrandık asagısı ve indet alan 0

        t = 1

    elif data['sent_intensity_scr'][i] < -0.10:  #conservative davrandık asagısı ve indet alan 0

        t = 0

    else:

        t = -9

    b.append(t)  

    

data['sent_intensity_scr_binary'] = b 



data.head()
r = data[(data['sent_intensity_scr_binary'] != -9) & (data['source'] == 'Amazon')]



print(len(r))



print(classification_report(r['Label'], r['sent_intensity_scr_binary'], digits=4))



print(confusion_matrix(r['Label'], r['sent_intensity_scr_binary']))
data['index'] = data.index
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.feature_extraction.text import TfidfTransformer

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.naive_bayes import MultinomialNB

from sklearn.svm import LinearSVC

from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import StratifiedKFold

from sklearn.metrics import confusion_matrix

from sklearn.metrics import classification_report

import itertools

from sklearn.metrics import roc_curve, auc



seed = 2

length_of_data = len(data[data['source'] == 'Amazon']['Sentences'])





totalSVM = 0           # Accuracy measure

totalNB = 0

totalLR = 0

totalRF = 0

totalGBT = 0



totalMatSvm = np.zeros((2,2));  # Confusion matrix

totalMatNB = np.zeros((2,2));

totalMatLR = np.zeros((2,2));

totalMatRF = np.zeros((2,2));

totalMatGBT = np.zeros((2,2));





X_train = data[(data['source'] == 'Yelp') | (data['source'] == 'Imdb')]['Sentences']

X_test =  data[data['source'] == 'Amazon']['Sentences']

y_train = data[(data['source'] == 'Yelp') | (data['source'] == 'Imdb')]['Label']

y_test =  data[data['source'] == 'Amazon']['Label']

index_test = data[data['source'] == 'Amazon']['index']



#data manipulation, feature engineering

transformer = CountVectorizer(analyzer = text_mining, stop_words = {'English'})

sparse_matrix_train = transformer.fit_transform(X_train)

sparse_matrix_test = transformer.transform(X_test)



idf_transformer = TfidfTransformer()

tfidf_train = idf_transformer.fit_transform(sparse_matrix_train)

tfidf_test = idf_transformer.transform(sparse_matrix_test)



#model types

SVM = LinearSVC(random_state=seed)

NB = MultinomialNB(alpha=1.0, class_prior=None, fit_prior=True) 

LR = LogisticRegression(random_state=seed, max_iter=500)

RF = RandomForestClassifier(n_estimators = 250, max_depth = 50, criterion = "entropy" ,random_state=seed)

GBT = GradientBoostingClassifier(loss='deviance', learning_rate=0.1, n_estimators=250, subsample=1.0, \

                                 criterion='mse',max_depth=50, random_state=seed)



#lets fit data

SVM.fit(tfidf_train,y_train)

NB.fit(tfidf_train,y_train)

LR.fit(tfidf_train,y_train)

RF.fit(tfidf_train,y_train)

GBT.fit(tfidf_train,y_train)



#predictions

Target = y_test

pred_SVM = SVM.predict(tfidf_test)

pred_NB = NB.predict(tfidf_test)

pred_LR = LR.predict(tfidf_test)

pred_RF = RF.predict(tfidf_test)

pred_GBT = GBT.predict(tfidf_test)





#confusion matrix

totalMatSvm = totalMatSvm + confusion_matrix(y_test,pred_SVM)

totalMatNB = totalMatNB + confusion_matrix(y_test, pred_NB)

totalMatLR = totalMatLR + confusion_matrix(y_test, pred_LR)

totalMatRF = totalMatRF + confusion_matrix(y_test, pred_RF)

totalMatGBT = totalMatGBT + confusion_matrix(y_test, pred_GBT)



totalSVM = totalSVM+sum(y_test==pred_SVM)

totalNB = totalNB+sum(y_test==pred_NB)

totalLR = totalLR+sum(y_test==pred_LR)

totalRF = totalRF+sum(y_test==pred_RF)

totalGBT = totalGBT+sum(y_test==pred_GBT)







fpr_SVM, tpr_SVM, thresholds = roc_curve(Target, pred_SVM)

ROC_SVM = auc(fpr_SVM, tpr_SVM)

fpr_NB, tpr_NB, thresholds = roc_curve(Target, pred_NB)

ROC_NB = auc(fpr_NB, tpr_NB)

fpr_LR, tpr_LR, thresholds = roc_curve(Target, pred_LR)

ROC_LR = auc(fpr_LR, tpr_LR)

fpr_RF, tpr_RF, thresholds = roc_curve(Target, pred_RF)

ROC_RF = auc(fpr_RF, tpr_RF)

fpr_GBT, tpr_GBT, thresholds = roc_curve(Target, pred_GBT)

ROC_GBT = auc(fpr_GBT, tpr_GBT)

    

    

print (totalMatSvm, totalSVM/length_of_data, totalMatNB, totalNB/length_of_data, totalMatLR, totalLR/length_of_data, \

totalMatRF, totalRF/length_of_data,totalMatGBT, totalGBT/length_of_data),           \

print('Support Vector Machines'),print(classification_report(Target, pred_SVM, digits=4)), \

print('Naive Bayes'), print(classification_report(Target, pred_NB, digits=4)),\

print('Logistic Regression'),print(classification_report(Target, pred_LR, digits=4)), \

print('Random Forest'),print(classification_report(Target, pred_RF, digits=4)),    \

print('Gradient Boosting'),print(classification_report(Target, pred_GBT, digits=4)), \

print('ROC_SVM:',ROC_SVM, '  ROC_NB:',ROC_NB, '  ROC_LR:',ROC_LR, '  ROC_RF:',ROC_RF, '  ROC_GBT:',ROC_GBT)
df = pd.DataFrame()



df['index'] = index_test

df['Target'] = Target

df['pred_SVM'] = pred_SVM

df['pred_NB'] = pred_NB

df['pred_LR'] = pred_LR

df['pred_RF'] = pred_RF

df['pred_GBT'] = pred_GBT



df['ensemble_scr'] = df['pred_SVM'] + df['pred_NB'] + df['pred_LR'] + df['pred_RF'] + df['pred_GBT']





j = []

for i in range(len(df)):

    w = 0

    if list(df['ensemble_scr'])[i] > 2:

        w = 1

    j.append(w)

        

df['ensemble_scr_binary'] = j

                      

df.head()                  
full_merge = pd.merge(left = data, right = df, left_on = 'index', right_on = 'index')



print(classification_report(full_merge['Target'], full_merge['ensemble_scr_binary'], digits=4))



print(confusion_matrix(full_merge['Target'], full_merge['ensemble_scr_binary']))





full_merge.head()
prediction_ensemble = np.empty([len(full_merge),1])

prediction_SVM = np.empty([len(full_merge),1])

prediction_NB = np.empty([len(full_merge),1])

prediction_LR = np.empty([len(full_merge),1])

prediction_RF = np.empty([len(full_merge),1])

prediction_GBT = np.empty([len(full_merge),1])





for i in range(len(full_merge)):

    if list(full_merge['sent_intensity_scr_binary'])[i] != -9:

        prediction_ensemble[i] = list(full_merge['sent_intensity_scr_binary'])[i]

        prediction_SVM[i] = list(full_merge['sent_intensity_scr_binary'])[i]

        prediction_NB[i] = list(full_merge['sent_intensity_scr_binary'])[i]

        prediction_LR[i] = list(full_merge['sent_intensity_scr_binary'])[i]

        prediction_RF[i] = list(full_merge['sent_intensity_scr_binary'])[i]

        prediction_GBT[i] = list(full_merge['sent_intensity_scr_binary'])[i]

    else:

        prediction_ensemble[i] = list(full_merge['ensemble_scr_binary'])[i]

        prediction_SVM[i] = list(full_merge['pred_SVM'])[i]

        prediction_NB[i] = list(full_merge['pred_NB'])[i]

        prediction_LR[i] = list(full_merge['pred_LR'])[i]

        prediction_RF[i] = list(full_merge['pred_RF'])[i]

        prediction_GBT[i] = list(full_merge['pred_GBT'])[i]



full_merge['pred_ensemble_combined'] = prediction_ensemble

full_merge['pred_SVM_combined'] = prediction_SVM

full_merge['pred_NB_combined'] = prediction_NB

full_merge['pred_LR_combined'] = prediction_LR

full_merge['pred_RF_combined'] = prediction_RF

full_merge['pred_GBT_combined'] = prediction_GBT



full_merge.head()





print('prediction_ensemble')

print(classification_report(full_merge['Label'], full_merge['pred_ensemble_combined'], digits=4))

print(confusion_matrix(full_merge['Label'], full_merge['pred_ensemble_combined']))

print(' ')

print('prediction_SVM')

print(classification_report(full_merge['Label'], full_merge['pred_SVM_combined'], digits=4))

print(confusion_matrix(full_merge['Label'], full_merge['pred_SVM_combined']))

print(' ')

print('prediction_NB')

print(classification_report(full_merge['Label'], full_merge['pred_NB_combined'], digits=4))

print(confusion_matrix(full_merge['Label'], full_merge['pred_NB_combined']))

print(' ')

print('prediction_LR')

print(classification_report(full_merge['Label'], full_merge['pred_LR_combined'], digits=4))

print(confusion_matrix(full_merge['Label'], full_merge['pred_LR_combined']))

print(' ')

print('prediction_RF')

print(classification_report(full_merge['Label'], full_merge['pred_RF_combined'], digits=4))

print(confusion_matrix(full_merge['Label'], full_merge['pred_RF_combined']))

print(' ')

print('prediction_GBT')

print(classification_report(full_merge['Label'], full_merge['pred_GBT_combined'], digits=4))

print(confusion_matrix(full_merge['Label'], full_merge['pred_GBT_combined']))