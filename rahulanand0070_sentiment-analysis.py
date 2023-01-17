import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

#ignore warning messages

import warnings

warnings.filterwarnings('ignore')



from sklearn.feature_extraction.text import CountVectorizer

from sklearn.model_selection import GridSearchCV

#NB

from sklearn.metrics import f1_score

from sklearn.naive_bayes import MultinomialNB

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score

from sklearn.linear_model import LogisticRegression



import seaborn as sns

from sklearn.metrics import accuracy_score

from sklearn.metrics import confusion_matrix

from sklearn.metrics import precision_score

from sklearn.metrics import f1_score

from sklearn.metrics import recall_score

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.linear_model import SGDClassifier

from wordcloud import WordCloud
df = pd.read_csv("../input/google-play-store-apps/googleplaystore_user_reviews.csv")

df
#total no of Apps before cleaning

print(df.shape)

#Checking null value

print(df.isnull().values.any())
df=df.drop(['Sentiment_Polarity', 'Sentiment_Subjectivity'], axis=1)
#dropping the rows which has null value 

final = df.dropna(how='any',axis=0) 
#printing shape after removing the rows which has null values

print(final.shape)

print(final.isnull().values.any())
Sentiment=final['Sentiment'].value_counts()

print(Sentiment)

plt.figure(figsize=(12,12))

sns.barplot(Sentiment.index, Sentiment.values, alpha=0.8)

plt.title('Content Rating vs No Apps')

plt.ylabel('Apps')

plt.xlabel('Content Rating')

plt.show()
# find sentences containing HTML tags

import re

i=0;

for sent in final['Translated_Review'].values:

    if (len(re.findall('<.*?>', sent))):

        print(i)

        print(sent)

        break;

    i += 1;
import nltk

from nltk.corpus import stopwords

stop = set(stopwords.words('english')) #set of stopwords

sno = nltk.stem.SnowballStemmer('english') #initialising the snowball stemmer



def cleanhtml(sentence): #function to clean the word of any html-tags

    cleanr = re.compile('<.*?>')

    cleantext = re.sub(cleanr, ' ', sentence)

    return cleantext

def cleanpunc(sentence): #function to clean the word of any punctuation or special characters

    cleaned = re.sub(r'[?|!|\'|"|#]',r'',sentence)

    cleaned = re.sub(r'[.|,|)|(|\|/]',r' ',cleaned)

    return  cleaned
#Code for implementing step-by-step the checks mentioned in the pre-processing phase

# this code takes a while to run as it needs to run on 500k sentences.

i=0

str1=' '

final_string=[]

all_positive_words=[] # store words from +ve reviews here

all_negative_words=[] # store words from -ve reviews here.

all_netural_words=[]

s=''

for sent in final['Translated_Review'].values:

    filtered_sentence=[]

    #print(sent);

    sent=cleanhtml(sent) # remove HTMl tags

    for w in sent.split():

        for cleaned_words in cleanpunc(w).split():

            if((cleaned_words.isalpha()) & (len(cleaned_words)>2)):    

                if(cleaned_words.lower() not in stop):

                    s=(sno.stem(cleaned_words.lower())).encode('utf8')

                    filtered_sentence.append(s)

                    if (final['Sentiment'].values)[i] == 'Positive': 

                        all_positive_words.append(s) #list of all words used to describe positive reviews

                    if(final['Sentiment'].values)[i] == 'Negative':

                        all_negative_words.append(s) #list of all words used to describe negative reviews reviews

                    if(final['Sentiment'].values)[i] == 'Negative':

                        all_netural_words.append(s) #list of all words used to describe negative reviews reviews    

                else:

                    continue

            else:

                continue 

    #print(filtered_sentence)

    str1 = b" ".join(filtered_sentence) #final string of cleaned words

    #print("***********************************************************************")

    

    final_string.append(str1)

    i+=1
wordcloud = WordCloud(width = 800, height = 800, 

            background_color ='white').generate(str(all_positive_words))

plt.figure(figsize = (8, 8), facecolor = None) 

plt.imshow(wordcloud) 

plt.axis("off") 

plt.tight_layout(pad = 0) 

plt.show()
wordcloud = WordCloud(width = 800, height = 800, 

            background_color ='white').generate(str(all_negative_words))

plt.figure(figsize = (8, 8), facecolor = None) 

plt.imshow(wordcloud) 

plt.axis("off") 

plt.tight_layout(pad = 0) 

plt.show()
wordcloud = WordCloud(width = 800, height = 800, 

            background_color ='white').generate(str(all_netural_words))

plt.figure(figsize = (8, 8), facecolor = None) 

plt.imshow(wordcloud) 

plt.axis("off") 

plt.tight_layout(pad = 0) 

plt.show()
final.head()
from sklearn.model_selection import train_test_split

x=final["Translated_Review"]

y=final["Sentiment"]

x_tr,x_test,y_tr,y_test=train_test_split(x, y, test_size=0.2,shuffle=False)
print(x_tr.shape,x_test.shape,y_tr.shape,y_test.shape)
#BOW for unigram

bow = CountVectorizer()

x_tr_uni = bow.fit_transform(x_tr)

x_test_uni= bow.transform(x_test)
bi_gram = CountVectorizer(ngram_range=(1,2))

x_tr_bi = bi_gram.fit_transform(x_tr)

x_test_bi = bi_gram.transform(x_test)
tf_idf_vect = TfidfVectorizer(ngram_range=(1,2))

x_tr_tfidf = tf_idf_vect.fit_transform(x_tr)

x_test_tfidf = tf_idf_vect.transform(x_test)
NB = MultinomialNB()

alpha=[1000,500,100,50,10,5,1,0.5,0.1,0.05,0.01,0.005,0.001,0.0005,0.0001]

alpha_value = {'alpha':alpha} #params we need to try on classifier

gsv = GridSearchCV(NB,alpha_value,cv=5,verbose=1,scoring='f1_micro')

gsv.fit(x_tr_uni,y_tr)

print("Best HyperParameter: ",gsv.best_params_)

print(gsv.best_score_)

optimal_alpha=gsv.best_params_['alpha']





x=[]

plt.figure(figsize=(8,8))

for a in gsv.cv_results_['mean_test_score']:

    x.append(a)

plt.xlabel(r"$\alpha$",fontsize=15)

plt.ylabel("F1-Score")

plt.title(r'F1-Score v/s $\alpha$')

plt.plot(alpha,x,linestyle='dashed', marker='x', markerfacecolor='red', markersize=10)

plt.show()
NB = MultinomialNB(1)

NB.fit(x_tr_uni,y_tr)

y_pred = NB.predict(x_test_uni)

print("Accuracy on test set: %0.3f%%"%(accuracy_score(y_test, y_pred)*100))

print("Precision on test set: %0.3f"%(precision_score(y_test, y_pred,average='weighted')))

print("Recall on test set: %0.3f"%(recall_score(y_test, y_pred,average='weighted')))

print("F1-Score on test set: %0.3f"%(f1_score(y_test, y_pred,average='weighted')))

labels = ['-ve','0','+ve']

df_cm = pd.DataFrame(confusion_matrix(y_test, y_pred), range(3),range(3))

sns.set(font_scale=1.4)#for label size

sns.heatmap(df_cm, annot=True,annot_kws={"size": 16}, fmt='g',xticklabels=labels, yticklabels=labels)
NB = MultinomialNB()

alpha=[1000,500,100,50,10,5,1,0.5,0.1,0.05,0.01,0.005,0.001,0.0005,0.0001]

alpha_value = {'alpha':alpha} #params we need to try on classifier

gsv = GridSearchCV(NB,alpha_value,cv=5,verbose=1,scoring='f1_micro')

gsv.fit(x_tr_bi,y_tr)

print("Best HyperParameter: ",gsv.best_params_)

print(gsv.best_score_)

optimal_alpha=gsv.best_params_['alpha']





x=[]

plt.figure(figsize=(8,8))

for a in gsv.cv_results_['mean_test_score']:

    x.append(a)

plt.xlabel(r"$\alpha$",fontsize=15)

plt.ylabel("F1-Score")

plt.title(r'F1-Score v/s $\alpha$')

plt.plot(alpha,x,linestyle='dashed', marker='x', markerfacecolor='red', markersize=10)

plt.show()
NB = MultinomialNB(1)

NB.fit(x_tr_bi,y_tr)

y_pred = NB.predict(x_test_bi)

print("Accuracy on test set: %0.3f%%"%(accuracy_score(y_test, y_pred)*100))

print("Precision on test set: %0.3f"%(precision_score(y_test, y_pred,average='weighted')))

print("Recall on test set: %0.3f"%(recall_score(y_test, y_pred,average='weighted')))

print("F1-Score on test set: %0.3f"%(f1_score(y_test, y_pred,average='weighted')))

labels = ['-ve','0','+ve']

df_cm = pd.DataFrame(confusion_matrix(y_test, y_pred), range(3),range(3))

sns.set(font_scale=1.4)#for label size

sns.heatmap(df_cm, annot=True,annot_kws={"size": 16}, fmt='g',xticklabels=labels, yticklabels=labels)
NB = MultinomialNB()

alpha=[1000,500,100,50,10,5,1,0.5,0.1,0.05,0.01,0.005,0.001,0.0005,0.0001]

alpha_value = {'alpha':alpha} #params we need to try on classifier

gsv = GridSearchCV(NB,alpha_value,cv=5,verbose=1,scoring='f1_micro')

gsv.fit(x_tr_tfidf,y_tr)

print("Best HyperParameter: ",gsv.best_params_)

print(gsv.best_score_)

optimal_alpha=gsv.best_params_['alpha']





x=[]

plt.figure(figsize=(8,8))

for a in gsv.cv_results_['mean_test_score']:

    x.append(a)

plt.xlabel(r"$\alpha$",fontsize=15)

plt.ylabel("F1-Score")

plt.title(r'F1-Score v/s $\alpha$')

plt.plot(alpha,x,linestyle='dashed', marker='x', markerfacecolor='red', markersize=10)

plt.show()
NB = MultinomialNB(0.1)

NB.fit(x_tr_tfidf,y_tr)

y_pred = NB.predict(x_test_tfidf)

print("Accuracy on test set: %0.3f%%"%(accuracy_score(y_test, y_pred)*100))

print("Precision on test set: %0.3f"%(precision_score(y_test, y_pred,average='weighted')))

print("Recall on test set: %0.3f"%(recall_score(y_test, y_pred,average='weighted')))

print("F1-Score on test set: %0.3f"%(f1_score(y_test, y_pred,average='weighted')))

labels = ['-ve','0','+ve']

df_cm = pd.DataFrame(confusion_matrix(y_test, y_pred), range(3),range(3))

sns.set(font_scale=1.4)#for label size

sns.heatmap(df_cm, annot=True,annot_kws={"size": 16}, fmt='g',xticklabels=labels, yticklabels=labels)
LR = LogisticRegression(penalty='l1')

C=[10**-4, 10**-2, 10**0, 10**2, 10**4]

C_value = [{'C': C}]



gsv = GridSearchCV(LR,C_value,cv=5,verbose=1,scoring='f1_micro')

gsv.fit(x_tr_uni,y_tr)

print("Best HyperParameter: ",gsv.best_params_)

print(gsv.best_score_)

optimal_C=gsv.best_params_['C']





x=[]

plt.figure(figsize=(8,8))

for a in gsv.cv_results_['mean_test_score']:

    x.append(a)

plt.xlabel(r"C",fontsize=15)

plt.ylabel("F1-Score")

plt.title(r'F1-Score v/s C')

plt.plot(C,x,linestyle='dashed', marker='x', markerfacecolor='red', markersize=10)

plt.show()
LR= LogisticRegression(penalty='l1',C=1)

LR.fit(x_tr_uni,y_tr)

y_pred =LR.predict(x_test_uni)

print("Accuracy on test set: %0.3f%%"%(accuracy_score(y_test, y_pred)*100))

print("Precision on test set: %0.3f"%(precision_score(y_test, y_pred,average='macro')))

print("Recall on test set: %0.3f"%(recall_score(y_test, y_pred,average='macro')))

print("F1-Score on test set: %0.3f"%(f1_score(y_test, y_pred,average='macro')))

labels = ['-ve','0','+ve']

df_cm = pd.DataFrame(confusion_matrix(y_test, y_pred), range(3),range(3))

sns.set(font_scale=1.4)#for label size

sns.heatmap(df_cm, annot=True,annot_kws={"size": 16}, fmt='g',xticklabels=labels, yticklabels=labels)
LR = LogisticRegression(penalty='l1')

C=[10**-4, 10**-2, 10**0, 10**2, 10**4]

C_value = [{'C': C}]



gsv = GridSearchCV(LR,C_value,cv=5,verbose=1,scoring='f1_micro')

gsv.fit(x_tr_bi,y_tr)

print("Best HyperParameter: ",gsv.best_params_)

print(gsv.best_score_)

optimal_C=gsv.best_params_['C']





x=[]

plt.figure(figsize=(8,8))

for a in gsv.cv_results_['mean_test_score']:

    x.append(a)

plt.xlabel("F1-Score",fontsize=15)

plt.ylabel("F1-Score")

plt.title('F1-Score v/s C')

plt.plot(C,x,linestyle='dashed', marker='x', markerfacecolor='red', markersize=10)

plt.show()
LR= LogisticRegression(penalty='l1',C=100)

LR.fit(x_tr_bi,y_tr)

y_pred =LR.predict(x_test_bi)

print("Accuracy on test set: %0.3f%%"%(accuracy_score(y_test, y_pred)*100))

print("Precision on test set: %0.3f"%(precision_score(y_test, y_pred,average='macro')))

print("Recall on test set: %0.3f"%(recall_score(y_test, y_pred,average='macro')))

print("F1-Score on test set: %0.3f"%(f1_score(y_test, y_pred,average='macro')))

labels = ['-ve','0','+ve']

df_cm = pd.DataFrame(confusion_matrix(y_test, y_pred), range(3),range(3))

sns.set(font_scale=1.4)#for label size

sns.heatmap(df_cm, annot=True,annot_kws={"size": 16}, fmt='g',xticklabels=labels, yticklabels=labels)
LR = LogisticRegression(penalty='l1')

C=[10**-4, 10**-2, 10**0, 10**2, 10**4]

C_value = [{'C': C}]



gsv = GridSearchCV(LR,C_value,cv=5,verbose=1,scoring='f1_micro')

gsv.fit(x_tr_tfidf,y_tr)

print("Best HyperParameter: ",gsv.best_params_)

print(gsv.best_score_)

optimal_C=gsv.best_params_['C']





x=[]

plt.figure(figsize=(8,8))

for a in gsv.cv_results_['mean_test_score']:

    x.append(a)

plt.xlabel("F1-Score",fontsize=15)

plt.ylabel("F1-Score")

plt.title('F1-Score v/s alpha')

plt.plot(C,x,linestyle='dashed', marker='x', markerfacecolor='red', markersize=10)

plt.show()
LR= LogisticRegression(penalty='l1',C=100)

LR.fit(x_tr_tfidf,y_tr)

y_pred =LR.predict(x_test_tfidf)

print("Accuracy on test set: %0.3f%%"%(accuracy_score(y_test, y_pred)*100))

print("Precision on test set: %0.3f"%(precision_score(y_test, y_pred,average='macro')))

print("Recall on test set: %0.3f"%(recall_score(y_test, y_pred,average='macro')))

print("F1-Score on test set: %0.3f"%(f1_score(y_test, y_pred,average='macro')))

labels = ['-ve','0','+ve']

df_cm = pd.DataFrame(confusion_matrix(y_test, y_pred), range(3),range(3))

sns.set(font_scale=1.4)#for label size

sns.heatmap(df_cm, annot=True,annot_kws={"size": 16}, fmt='g',xticklabels=labels, yticklabels=labels)
LR = SGDClassifier(loss = 'hinge', class_weight='balanced', learning_rate='optimal',eta0=0.001, n_jobs = -1)

a_val = [0.0000001,0.0000005,0.000001,0.000005,0.00001,0.00005,0.0001,0.0005,0.001,0.005,0.01,0.05,0.1]

a_value = [{'alpha': a_val}]



gsv = GridSearchCV(LR,a_value,cv=5,verbose=1,scoring='f1_micro')

gsv.fit(x_tr_uni,y_tr)

print("Best HyperParameter: ",gsv.best_params_)

print(gsv.best_score_)

optimal_alpha=gsv.best_params_['alpha']





x=[]

plt.figure(figsize=(8,8))

for a in gsv.cv_results_['mean_test_score']:

    x.append(a)

plt.xlabel("F1-Score",fontsize=15)

plt.ylabel("F1-Score")

plt.title('F1-Score v/s alpha')

plt.plot(a_val,x,linestyle='dashed', marker='x', markerfacecolor='red', markersize=13)

plt.show()
clf = SGDClassifier(loss = 'hinge', alpha = 0.0001, class_weight='balanced', learning_rate='optimal',eta0=0.001, n_jobs = -1) 

clf.fit(x_tr_uni,y_tr)

y_pred = clf.predict(x_test_uni)

print("Accuracy on test set: %0.3f%%"%(accuracy_score(y_test, y_pred)*100))

print("Precision on test set: %0.3f"%(precision_score(y_test, y_pred,average='macro')))

print("Recall on test set: %0.3f"%(recall_score(y_test, y_pred,average='macro')))

print("F1-Score on test set: %0.3f"%(f1_score(y_test, y_pred,average='macro')))

labels = ['-ve','0','+ve']

df_cm = pd.DataFrame(confusion_matrix(y_test, y_pred), range(3),range(3))

sns.set(font_scale=1.4)#for label size

sns.heatmap(df_cm, annot=True,annot_kws={"size": 16},fmt='g',xticklabels=labels, yticklabels=labels)
LR = SGDClassifier(loss = 'hinge', class_weight='balanced', learning_rate='optimal',eta0=0.001, n_jobs = -1)

a_val = [0.0000001,0.0000005,0.000001,0.000005,0.00001,0.00005,0.0001,0.0005,0.001,0.005,0.01,0.05,0.1]

a_value = [{'alpha': a_val}]



gsv = GridSearchCV(LR,a_value,cv=5,verbose=1,scoring='f1_micro')

gsv.fit(x_tr_bi,y_tr)

print("Best HyperParameter: ",gsv.best_params_)

print(gsv.best_score_)

optimal_alpha=gsv.best_params_['alpha']





x=[]

plt.figure(figsize=(8,8))

for a in gsv.cv_results_['mean_test_score']:

    x.append(a)

plt.xlabel("F1-Score",fontsize=15)

plt.ylabel("F1-Score")

plt.title('F1-Score v/s alpha')

plt.plot(a_val,x,linestyle='dashed', marker='x', markerfacecolor='red', markersize=13)

plt.show()
clf = SGDClassifier(loss = 'hinge', alpha = 0.0001, class_weight='balanced', learning_rate='optimal',eta0=0.001, n_jobs = -1) 

clf.fit(x_tr_bi,y_tr)

y_pred = clf.predict(x_test_bi)

print("Accuracy on test set: %0.3f%%"%(accuracy_score(y_test, y_pred)*100))

print("Precision on test set: %0.3f"%(precision_score(y_test, y_pred,average='macro')))

print("Recall on test set: %0.3f"%(recall_score(y_test, y_pred,average='macro')))

print("F1-Score on test set: %0.3f"%(f1_score(y_test, y_pred,average='macro')))

labels = ['-ve','0','+ve']

df_cm = pd.DataFrame(confusion_matrix(y_test, y_pred), range(3),range(3))

sns.set(font_scale=1.4)#for label size

sns.heatmap(df_cm, annot=True,annot_kws={"size": 16},fmt='g',xticklabels=labels, yticklabels=labels)
LR = SGDClassifier(loss = 'hinge', class_weight='balanced', learning_rate='optimal',eta0=0.001, n_jobs = -1)

a_val = [0.0000001,0.0000005,0.000001,0.000005,0.00001,0.00005,0.0001,0.0005,0.001,0.005,0.01,0.05,0.1]

a_value = [{'alpha': a_val}]



gsv = GridSearchCV(LR,a_value,cv=5,verbose=1,scoring='f1_micro')

gsv.fit(x_tr_tfidf,y_tr)

print("Best HyperParameter: ",gsv.best_params_)

print(gsv.best_score_)

optimal_alpha=gsv.best_params_['alpha']





x=[]

plt.figure(figsize=(8,8))

for a in gsv.cv_results_['mean_test_score']:

    x.append(a)

plt.xlabel("F1-Score",fontsize=15)

plt.ylabel("F1-Score")

plt.title('F1-Score v/s alpha')

plt.plot(a_val,x,linestyle='dashed', marker='x', markerfacecolor='red', markersize=13)

plt.show()
clf = SGDClassifier(loss = 'hinge', alpha = 0.00005, class_weight='balanced', learning_rate='optimal',eta0=0.001, n_jobs = -1) 

clf.fit(x_tr_tfidf,y_tr)

y_pred = clf.predict(x_test_tfidf)

print("Accuracy on test set: %0.3f%%"%(accuracy_score(y_test, y_pred)*100))

print("Precision on test set: %0.3f"%(precision_score(y_test, y_pred,average='macro')))

print("Recall on test set: %0.3f"%(recall_score(y_test, y_pred,average='macro')))

print("F1-Score on test set: %0.3f"%(f1_score(y_test, y_pred,average='macro')))

labels = ['-ve','0','+ve']

df_cm = pd.DataFrame(confusion_matrix(y_test, y_pred), range(3),range(3))

sns.set(font_scale=1.4)#for label size

sns.heatmap(df_cm, annot=True,annot_kws={"size": 16},fmt='g',xticklabels=labels, yticklabels=labels)