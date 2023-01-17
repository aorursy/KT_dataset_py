import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import re

import time

import warnings

warnings.filterwarnings('ignore')

import numpy as np

from nltk.corpus import stopwords

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.model_selection import train_test_split

from sklearn.model_selection import GridSearchCV

from sklearn.linear_model import SGDClassifier

from sklearn.metrics import f1_score

from sklearn.metrics import accuracy_score

from sklearn.metrics import confusion_matrix

from sklearn.metrics import precision_score

from sklearn.metrics import f1_score

from sklearn.metrics import recall_score

from sklearn.preprocessing import LabelEncoder

from keras.utils import np_utils

from sklearn.ensemble import RandomForestClassifier

from nltk.stem import PorterStemmer

import nltk
data = pd.read_csv("../input/youtubevideodataset/Youtube Video Dataset.csv")

data.head(10)
print(data.shape)

print(data.isnull().values.any())
data=data.dropna()

print(data.isnull().values.any())
Category=data['Category'].value_counts()

print(Category.shape)

print(Category)
# loading stop words from nltk library

stop_words = set(stopwords.words('english'))

#Stemmering the word

sno = nltk.stem.SnowballStemmer('english') #initialising the snowball stemmer



def preprocessing(total_text, index, column):

    if type(total_text) is not int:

        string = ""

        #Removing link

        url_pattern = r'((http|ftp|https):\/\/)?[\w\-_]+(\.[\w\-_]+)+([\w\-\.,@?^=%&amp;:/~\+#]*[\w\-\@?^=%&amp;/~\+#])?'

        total_text = re.sub(url_pattern, ' ', total_text)

        #removing email address

        email_pattern =r'[a-z0-9\.\-+_]+@[a-z0-9\.\-+_]+\.[a-z]+'

        total_text = re.sub(email_pattern, ' ', total_text)

        # replace every special char with space

        total_text = re.sub('[^a-zA-Z0-9\n]', ' ', total_text)

        # replace multiple spaces with single space

        total_text = re.sub('\s+',' ', total_text)

        # converting all the chars into lower-case.

        total_text = total_text.lower()

        

        for word in total_text.split():

        # if the word is a not a stop word then retain that word from the data

            if not word in stop_words:

                word=(sno.stem(word))

                string += word + " "

        

        data[column][index] = string
#text processing stage.

start_time = time.clock()

for index, row in data.iterrows():

    if type(row['Description']) is str:

        preprocessing(row['Description'], index, 'Description')

print('Time took for preprocessing the text :',time.clock() - start_time, "seconds")
data.head(10)
y_true = data['Category'].values

# split the data into test and train by maintaining same distribution of output varaible 'y_true' [stratify=y_true]

X_train, test_df, y_train, y_test = train_test_split(data, y_true, stratify=y_true, test_size=0.2)

train_df, cv_df, y_train, y_cv = train_test_split(X_train, y_train, stratify=y_train, test_size=0.2)
print('Number of data points in train data:', train_df.shape[0])

print('Number of data points in test data:', test_df.shape[0])

print('Number of data points in cv data:', cv_df.shape[0])
X_trainCategory=train_df['Category'].value_counts()

print(X_trainCategory)

print('Distribution of y in train')

plt.figure(figsize=(8,8))

sns.barplot(X_trainCategory.index, X_trainCategory.values, alpha=0.8)

plt.title('Distribution of y in train')

plt.ylabel('data point per class')

plt.xlabel('Class')

plt.xticks(rotation=90)

plt.show()









test_dfCategory=test_df['Category'].value_counts()

print(test_dfCategory)

print('Distribution of y in test')

plt.figure(figsize=(8,8))

sns.barplot(test_dfCategory.index,test_dfCategory.values, alpha=0.8)

plt.title('Distribution of y in test')

plt.ylabel('data point per class')

plt.xlabel('Class')

plt.xticks(rotation=90)

plt.show()



cv_dfCategory=cv_df['Category'].value_counts()

print(cv_dfCategory)

print('Distribution of y in cv')

plt.figure(figsize=(8,8))

sns.barplot(cv_dfCategory.index,cv_dfCategory.values, alpha=0.8)

plt.title('Distribution of y in cv')

plt.ylabel('data point per class')

plt.xlabel('Class')

plt.xticks(rotation=90)

plt.show()

x_tr=train_df['Description']

x_test=test_df['Description']

x_cv=cv_df['Description']
bow = CountVectorizer()

x_tr_uni = bow.fit_transform(x_tr)

x_test_uni= bow.transform(x_test)

x_cv_uni= bow.transform(x_cv)
tf_idf_vect = TfidfVectorizer(ngram_range=(1,2))

x_tr_tfidf = tf_idf_vect.fit_transform(x_tr)

x_test_tfidf = tf_idf_vect.transform(x_test)

x_cv_tfidf = tf_idf_vect.transform(x_cv)
def plotPrecisionRecall(y_test,y_pred):

    C = confusion_matrix(y_test, y_pred)

    A =(((C.T)/(C.sum(axis=1))).T)

    B =(C/C.sum(axis=0))

    labels = ['Art&Music','Food','History','Sci&Tech','Manu','TravelBlog']





    print("-"*20, "Precision matrix", "-"*20)

    plt.figure(figsize=(20,7))

    sns.heatmap(B, annot=True,annot_kws={"size": 16}, fmt='g', xticklabels=labels, yticklabels=labels)

    plt.xlabel('Predicted Class')

    plt.ylabel('Original Class')

    plt.show()



    # representing B in heatmap format

    print("-"*20, "Recall matrix", "-"*20)

    plt.figure(figsize=(20,7))

    sns.heatmap(A, annot=True,annot_kws={"size": 16}, fmt='g', xticklabels=labels, yticklabels=labels)

    plt.xlabel('Predicted Class')

    plt.ylabel('Original Class')

    plt.show()
clf = SGDClassifier(loss = 'hinge', alpha = 0.01, class_weight='balanced', learning_rate='optimal',eta0=0.001, n_jobs = -1) 

clf.fit(x_tr_uni,y_train)

y_pred = clf.predict(x_test_uni)

print("Accuracy on test set: %0.3f%%"%(accuracy_score(y_test, y_pred)*100))

print("Precision on test set: %0.3f"%(precision_score(y_test, y_pred,average='macro')))

print("Recall on test set: %0.3f"%(recall_score(y_test, y_pred,average='macro')))

print("F1-Score on test set: %0.3f"%(f1_score(y_test, y_pred,average='macro')))



print("-"*20, "confusion matrix", "-"*20)

plt.figure(figsize=(12,8))

matrix=confusion_matrix(y_test, y_pred)

df_cm = pd.DataFrame(matrix)

sns.set(font_scale=1.4)#for label size

labels = ['Art&Music','Food','History','Sci&Tech','Manu','TravelBlog']

sns.heatmap(df_cm, annot=True,annot_kws={"size": 16}, fmt='g',xticklabels=labels, yticklabels=labels)

plt.xlabel('Predicted Class')

plt.ylabel('Original Class')

plt.show()

plotPrecisionRecall(y_test,y_pred)
clf = SGDClassifier(loss = 'hinge', alpha =0.0001, class_weight='balanced', learning_rate='optimal',eta0=0.001, n_jobs = -1) 

clf.fit(x_tr_tfidf,y_train)

y_pred = clf.predict(x_test_tfidf)

print("Accuracy on test set: %0.3f%%"%(accuracy_score(y_test, y_pred)*100))

print("Precision on test set: %0.3f"%(precision_score(y_test, y_pred,average='macro')))

print("Recall on test set: %0.3f"%(recall_score(y_test, y_pred,average='macro')))

print("F1-Score on test set: %0.3f"%(f1_score(y_test, y_pred,average='macro')))



print("-"*20, "confusion matrix", "-"*20)

plt.figure(figsize=(12,8))

df_cm = pd.DataFrame(confusion_matrix(y_test, y_pred), range(6),range(6))

sns.set(font_scale=1.4)#for label size

labels = ['Art&Music','Food','History','Sci&Tech','Manu','TravelBlog']

sns.heatmap(df_cm, annot=True,annot_kws={"size": 16}, fmt='g',xticklabels=labels, yticklabels=labels)

plt.xlabel('Predicted Class')

plt.ylabel('Original Class')

plt.show()

plotPrecisionRecall(y_test,y_pred)
y_true = data['Category'].values

# split the data into test and train by maintaining same distribution of output varaible 'y_true' [stratify=y_true]

X_train, test_df, y_train, y_test = train_test_split(data, y_true, stratify=y_true, test_size=0.2)
x_tr=X_train['Description']

x_test=test_df['Description']
bow = CountVectorizer()

x_tr_uni = bow.fit_transform(x_tr)

x_test_uni= bow.transform(x_test)
tf_idf_vect = TfidfVectorizer(ngram_range=(1,2))

x_tr_tfidf = tf_idf_vect.fit_transform(x_tr)

x_test_tfidf = tf_idf_vect.transform(x_test)
RF= RandomForestClassifier(n_estimators=16,max_depth=130)

RF.fit(x_tr_uni,y_train)

y_pred =RF.predict(x_test_uni)

print("Accuracy on test set: %0.3f%%"%(accuracy_score(y_test, y_pred)*100))

print("Precision on test set: %0.3f"%(precision_score(y_test, y_pred,average='macro')))

print("Recall on test set: %0.3f"%(recall_score(y_test, y_pred,average='macro')))

print("F1-Score on test set: %0.3f"%(f1_score(y_test, y_pred,average='macro')))

print("-"*20, "confusion matrix", "-"*20)

plt.figure(figsize=(12,8))

df_cm = pd.DataFrame(confusion_matrix(y_test, y_pred), range(6),range(6))

sns.set(font_scale=1.4)#for label size

labels = ['Art&Music','Food','History','Sci&Tech','Manu','TravelBlog']

sns.heatmap(df_cm, annot=True,annot_kws={"size": 16}, fmt='g',xticklabels=labels, yticklabels=labels)

plt.xlabel('Predicted Class')

plt.ylabel('Original Class')

plt.show()

plotPrecisionRecall(y_test,y_pred)
RF= RandomForestClassifier(n_estimators=20,max_depth=190)

RF.fit(x_tr_tfidf,y_train)

y_pred =RF.predict(x_test_tfidf)

print("Accuracy on test set: %0.3f%%"%(accuracy_score(y_test, y_pred)*100))

print("Precision on test set: %0.3f"%(precision_score(y_test, y_pred,average='macro')))

print("Recall on test set: %0.3f"%(recall_score(y_test, y_pred,average='macro')))

print("F1-Score on test set: %0.3f"%(f1_score(y_test, y_pred,average='macro')))

print("-"*20, "confusion matrix", "-"*20)

plt.figure(figsize=(12,8))

df_cm = pd.DataFrame(confusion_matrix(y_test, y_pred), range(6),range(6))

sns.set(font_scale=1.4)#for label size

labels = ['Art&Music','Food','History','Sci&Tech','Manu','TravelBlog']

sns.heatmap(df_cm, annot=True,annot_kws={"size": 16}, fmt='g',xticklabels=labels, yticklabels=labels)

plt.xlabel('Predicted Class')

plt.ylabel('Original Class')

plt.show()



plotPrecisionRecall(y_test,y_pred)