import numpy as np 

import pandas as pd

from wordcloud import WordCloud, STOPWORDS

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import f1_score,accuracy_score

from sklearn.feature_selection import SelectKBest, chi2

from sklearn.svm import SVC

from sklearn.metrics import confusion_matrix

import seaborn as sns
df=pd.read_csv('/kaggle/input/sms-spam-collection-dataset/spam.csv',encoding='Windows-1252',usecols=['v1','v2'])

df.columns=['label','message']

df.head()
df.info()
df['label'].value_counts().plot.pie(autopct='%1.1f%%')
comments=''

for i in df['message']:

    cmt=str(i)

    tokens=cmt.split()

    

    comments += " ".join(tokens)+" "
stopwords =set(STOPWORDS)



wordcloud = WordCloud(stopwords = stopwords, 

                min_font_size = 8).generate(comments)

plt.figure(figsize = (12, 8), facecolor = None) 

plt.imshow(wordcloud, interpolation='bilinear')

plt.axis("off")

plt.show()
dfspam=df[df['label']=='spam']

dfham=df[df['label']=='ham']
commentsspam=''

for i in dfspam['message']:

    cmt=str(i)

    tokens=cmt.split()

    

    commentsspam += " ".join(tokens)+" "



commentsham=''

for i in dfham['message']:

    cmt=str(i)

    tokens=cmt.split()

    

    commentsham += " ".join(tokens)+" "
stopwords =set(STOPWORDS)



wordcloud = WordCloud(stopwords = stopwords, 

                min_font_size = 8).generate(commentsspam)

plt.figure(figsize = (12, 8), facecolor = None) 

plt.imshow(wordcloud, interpolation='bilinear')

plt.title('Spam')

plt.axis("off")

plt.show()
stopwords =set(STOPWORDS)



wordcloud = WordCloud(stopwords = stopwords, 

                min_font_size = 8).generate(commentsham)

plt.figure(figsize = (12, 8), facecolor = None) 

plt.imshow(wordcloud, interpolation='bilinear')

plt.title('Ham')

plt.axis("off")

plt.show()
X=df.message.values

y=df.label.values
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)

print('Train size: ',X_train.shape,' Test size: ',X_test.shape)
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer

vectorizer=CountVectorizer(min_df = 0.01, max_df=0.9,ngram_range=(1,7),stop_words='english',analyzer='char')

X_train_vec=vectorizer.fit_transform(X_train)

X_test_vec=vectorizer.transform(X_test)

X_train_vec.shape
print(vectorizer.get_feature_names()[:150])
clf=LogisticRegression(max_iter=1000)

clf.fit(X_train_vec,y_train)

y_pred=clf.predict(X_test_vec)

acc=accuracy_score(y_test, y_pred)

acc_hist = {'LogReg Char Full':acc}

acc
confmatrix=confusion_matrix(y_test, y_pred)

sns.heatmap(confmatrix,annot=True)
from sklearn.svm import SVC

clf=SVC(kernel='rbf')

clf.fit(X_train_vec, y_train)

y_pred=clf.predict(X_test_vec)

acc=accuracy_score(y_test,y_pred)

acc_hist['SVM RBF Char Full']=acc

acc
clf=SVC(kernel='linear')

clf.fit(X_train_vec, y_train)

y_pred=clf.predict(X_test_vec)

acc=accuracy_score(y_test,y_pred)

acc_hist['SVM Linear Char Full']=acc

acc


selkb = SelectKBest(chi2, k=2048)

X_train_sel=selkb.fit_transform(X_train_vec, y_train)

X_test_sel=selkb.transform(X_test_vec)

X_train_sel.shape
clf=LogisticRegression(max_iter=1000)

clf.fit(X_train_sel,y_train)

y_pred=clf.predict(X_test_sel)

acc=accuracy_score(y_test, y_pred)

acc_hist['LogReg Char Sel']=acc

acc
clf=SVC(kernel='linear')

clf.fit(X_train_sel, y_train)

y_pred=clf.predict(X_test_sel)

acc=accuracy_score(y_test,y_pred)

acc_hist['SVM Linear Char Sel']=acc

acc
confmatrix=confusion_matrix(y_test, y_pred)

sns.heatmap(confmatrix,annot=True)
clf=SVC(kernel='rbf')

clf.fit(X_train_sel, y_train)

y_pred=clf.predict(X_test_sel)

acc=accuracy_score(y_test, y_pred)

acc_hist['SVM RBF Char Sel']=acc

acc
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer

vectorizer=CountVectorizer(min_df = 0.01, max_df=0.9,ngram_range=(1,7),stop_words='english')

X_train_vec=vectorizer.fit_transform(X_train)

X_test_vec=vectorizer.transform(X_test)

X_train_vec.shape
print(vectorizer.get_feature_names())
clf=LogisticRegression(max_iter=1000)

clf.fit(X_train_vec,y_train)

y_pred=clf.predict(X_test_vec)

acc=accuracy_score(y_test, y_pred)

acc_hist['LogReg Word']=acc

acc
clf=SVC(kernel='rbf')

clf.fit(X_train_sel, y_train)

y_pred=clf.predict(X_test_sel)

acc=accuracy_score(y_test, y_pred)

acc_hist['SVM RBF Word']=acc

acc
clf=SVC(kernel='linear')

clf.fit(X_train_sel, y_train)

y_pred=clf.predict(X_test_sel)

acc=accuracy_score(y_test, y_pred)

acc_hist['SVM Linear Word']=acc

acc
confmatrix=confusion_matrix(y_test, y_pred)

sns.heatmap(confmatrix,annot=True)
pd.Series(acc_hist)