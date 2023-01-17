import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sqlite3
from tqdm import tqdm
import os
from imblearn.over_sampling import SMOTE
#Loading the dataset
con = sqlite3.connect("../input/amazon-fine-food-reviews/database.sqlite")
filtered = pd.read_sql_query("select * from Reviews where Score != 3",con)
filtered.head()
#Sorting the filtered dataset according to product id
filtered_sort = filtered.sort_values("ProductId", axis = 0, ascending = True)
#Removing duplicate datasets keeping only one of them
filtered = filtered_sort.drop_duplicates(subset = {"UserId","ProfileName","Time","Text"}, keep = 'first', inplace = False)
print("Shape after dropping duplicates = ", filtered.shape)
#Dropping those rows where HelpFulnessNumerator > HelpFullnessDenominator
filtered = filtered[filtered.HelpfulnessNumerator <= filtered.HelpfulnessDenominator]
print("Shape after removing HelpFulnessNumerator > HelpFullnessDenominator rows = ",filtered.shape)
#Changing scores as either "negative" if score<3 or "positive" if score>3
def polarity(score):
    if score < 3:
        return 0
    else:
        return 1

l = filtered['Score']
pos_neg = l.map(polarity)
filtered['Score'] = pos_neg
filtered.head()
#Taking out only Text, Time and Score columns in another dataframe
df = filtered[['Text','Time','Score']]
df.shape
counts = df['Score'].value_counts()
print(counts)
print("Ratio of positive reviews:- ",counts[1]/(counts[0]+counts[1]))
print("Ratio of negative points:- ",counts[0]/(counts[0]+counts[1]))
'''#Text Cleaning
import string
import re
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
#Text cleaning
def cleantext(review):
    cleanr = re.compile('<.*?>')
    text = re.sub(cleanr," ",review)
    text = re.sub('[^a-zA-Z]',' ',text)
    return text
corpus = []
for review in tqdm(df['Text']):
    st = cleantext(review)
    st = st.lower()
    st = st.split()
    sn = SnowballStemmer('english')
    st = [sn.stem(word) for word in st if not word in set(stopwords.words('english'))]
    st = ' '.join(st)
    corpus.append(st)'''

'''df['Text'] = corpus
#Sorting the values according to time for time based splitting
df = df.sort_values('Time',axis=0,ascending=True)
df = df.reset_index(drop=True)
df.head()
#df.to_csv("corpus.csv",index = False)'''
df = pd.read_csv("../input/amazon-fine-food-corpus/corpus.csv")
print(df.head())
df['Score'].value_counts()
df1k = df.loc[0:99999,:]
print("Shape of the new dataframe:- ",df1k.shape)
df1k['Score'].value_counts()
from sklearn.model_selection import TimeSeriesSplit
def timeseries(x,y):
    ts = TimeSeriesSplit(n_splits=4)
    for train_index,test_index in ts.split(x):
        x_train,x_test = x[train_index],x[test_index]
        y_train,y_test = y[train_index],y[test_index]
    return x_train,x_test,y_train,y_test

x_train,x_test,y_train,y_test = timeseries(df1k['Text'].values,df1k['Score'].values)
def imp_feature(model,classifier):
    neg = classifier.feature_log_prob_[0,:].argsort()
    pos = classifier.feature_log_prob_[1,:].argsort()
    top_pos_words = np.take(model.get_feature_names(),pos)
    top_neg_words = np.take(model.get_feature_names(),neg)
    imp_df = pd.DataFrame(columns = ['Pos_Words','Pos_Importance','Neg_Words','Neg_Importance'])
    imp_df['Pos_Words'] = top_pos_words[::-1]
    imp_df['Pos_Importance'] = np.take(classifier.feature_log_prob_[1,:],pos)[::-1]
    imp_df['Neg_Words'] = top_neg_words[::-1]
    imp_df['Neg_Importance'] = np.take(classifier.feature_log_prob_[0,:],neg)[::-1]
    return imp_df
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()
bow_train = cv.fit_transform(x_train)
print("Bag of words train vector shape:- ",bow_train.shape)
%%time
print("Count of 1 before oversampling:- ",sum(y_train==1))
print("Count of 0 before oversampling:- ",sum(y_train==0))
sm = SMOTE(random_state = 0)
bow_train_ov,y_train_ov = sm.fit_sample(bow_train,y_train)
print("Count of 1 after oversampling:- ",sum(y_train_ov==1))
print("Count of 0 after oversampling:- ",sum(y_train_ov==0))
print("Shape of x_train data after oversampling:- ",bow_train_ov.shape)
print("Shape of y_train data after oversampling:- ",y_train_ov.shape)
%%time
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import MultinomialNB
mnb = MultinomialNB()
param_grid = {'alpha':list(np.arange(0.1,10,0.5))}
gs = GridSearchCV(mnb,param_grid,scoring = 'f1',cv = 5,n_jobs = -1)
gs.fit(bow_train_ov,y_train_ov)
print("Best value of alpha:- ",gs.best_params_)
print("Best score:- ",gs.best_score_)
%%time
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
bow_test = cv.transform(x_test)
mnb = MultinomialNB(alpha = 7.1)
mnb.fit(bow_train_ov,y_train_ov)
y_pred_log = mnb.predict_log_proba(bow_test) # predicting log prob. of test data
print("Log probabilities:- ",y_pred_log)
y_pred1 = np.exp(y_pred_log) # Converting log_prob. to normal prob.(0to1)
print("Normal Probabilties:- ",y_pred1)
y_pred = np.zeros(len(y_test))
for i,p in enumerate(y_pred1[:,0]): # Converting the prob. to 0 or 1 values.
    if p > 0.5:
        y_pred[i] = 0
    else:
        y_pred[i] = 1
f1 = f1_score(y_test,y_pred)
print("BOW test f1_Score:- ",(f1*100))
test_acc = accuracy_score(y_test,y_pred)
print("BOW Test accuracy:- ",(test_acc*100))
sns.set_style("whitegrid")
plt.figure(1,figsize = (12,5))
plt.subplot(121)
delta0 = y_test[y_test == 0] - y_pred1[y_test==0,0]
sns.kdeplot(np.array(delta0))
plt.title("Error plot for class 0")
plt.subplot(122)
delta1 = y_test[y_test == 1] - y_pred1[y_test==1,1]
sns.kdeplot(np.array(delta1))
plt.title("Error plot for class 1")
plt.show()
%%time
#Training error and test error
y_train_pred = mnb.predict(bow_train)
bow_train_error = 1 - accuracy_score(y_train,y_train_pred)
print("Training accuracy:- ",(1-bow_train_error)*100)
bow_test_error = 1 - test_acc
print("Test accuracy:- ",test_acc*100)
x = imp_feature(cv,mnb)
print("Feature importance of the words in each class")
x
#Confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)
print(cm)
#Plotting confusion matrix
sns.heatmap(cm, annot = True)
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(sublinear_tf = True)
tfidf_train = tfidf.fit_transform(x_train)
print("Shape of tfidf_train:- ",tfidf_train.shape)
%%time
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import MultinomialNB
mnb = MultinomialNB(class_prior=(1,1))
param_grid = {'alpha':list(np.arange(0.1,5,0.2))}
gs = GridSearchCV(mnb,param_grid,scoring = 'neg_log_loss',cv = 5,n_jobs = -1)
gs.fit(tfidf_train,y_train)
print("Best value of alpha:- ",gs.best_params_)
print("Best score:- ",gs.best_score_)
%%time
tfidf_test = tfidf.transform(x_test)
mnb = MultinomialNB(alpha = 1.5, class_prior=(1,1))
mnb.fit(tfidf_train,y_train)
y_pred_log = mnb.predict_log_proba(tfidf_test) # predicting log prob. of test data
print("Log probabilities:- ",y_pred_log)
y_pred1 = np.exp(y_pred_log) # Converting log_prob to normal prob.(0to1)
print("Normal Probabilties:- ",y_pred1)
y_pred = np.zeros(len(y_test))
for i,p in enumerate(y_pred1[:,0]): # Converting the prob. to 0 or 1 values.
    if p > 0.5:
        y_pred[i] = 0
    else:
        y_pred[i] = 1
f1 = f1_score(y_test,y_pred)
print("Tfidf test f1_Score:- ",(f1*100))
test_acc = accuracy_score(y_test,y_pred)
print("Tfidf Test accuracy:- ",(test_acc*100))
sns.set_style("whitegrid")
plt.figure(1,figsize = (12,5))
plt.subplot(121)
delta0 = y_test[y_test == 0] - y_pred1[y_test==0,0]
sns.kdeplot(np.array(delta0))
plt.title("Error plot for class 0")
plt.subplot(122)
delta1 = y_test[y_test == 1] - y_pred1[y_test==1,1]
sns.kdeplot(np.array(delta1))
plt.title("Error plot for class 1")
plt.show()
%%time
#Training error and test error
y_train_pred = mnb.predict(tfidf_train)
tfidf_train_error = 1 - accuracy_score(y_train,y_train_pred)
print("Training accuracy:- ",(1-tfidf_train_error)*100)
tfidf_test_error = 1 - test_acc
print("Test accuracy:- ",test_acc*100)
x = imp_feature(tfidf,mnb)
print("Feature importance of each class")
x
#Confusion matrix
cm = confusion_matrix(y_test,y_pred)
print(cm)
#Plotting confusion matrix
sns.heatmap(cm,annot= True)