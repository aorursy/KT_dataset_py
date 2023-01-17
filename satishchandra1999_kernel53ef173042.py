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
!pip install wordcloud
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud, STOPWORDS 


DATASET_COLUMNS  = ["sentiment", "ids", "date", "flag", "user", "text"]


data=pd.read_csv('../input/sentiment140/training.1600000.processed.noemoticon.csv',encoding='latin', names=DATASET_COLUMNS ,header=None)
data.head()
dataset=data.iloc[:,5].values
ax = data.groupby('sentiment').count().plot(kind='bar', title='Distribution of data',
                                               legend=False)
ax.set_xticklabels(['Negative','Positive'], rotation=0)

import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
corpus = []
for i in range(0,1600000):
    review = re.sub('[^a-zA-Z]', ' ', dataset[i])
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    all_stopwords=stopwords.words('english')
    all_stopwords.remove('not')
    review = [ps.stem(word) for word in review if not word in set(all_stopwords)]
    review = ' '.join(review)
    corpus.append(review)

data_neg = corpus[:]
plt.figure(figsize = (20,20))
wc = WordCloud(max_words = 1000 , width = 1600 , height = 800,
               collocations=False).generate(" ".join(data_neg))
plt.imshow(wc)
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 420)
x = cv.fit_transform(corpus).toarray()
y = data.iloc[:, 0].values
 
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.2, random_state = 0)

#from sklearn.feature_extraction.text import TfidfVectorizer
#vectoriser = TfidfVectorizer(ngram_range=(1,2), max_features=500)
#vectoriser.fit(x_train)


#x_train = vectoriser.transform(x_train)
#x_test=vectoriser.transform(x_test)

from sklearn.ensemble import RandomForestClassifier 
classifier=RandomForestClassifier(n_estimators=10, criterion='entropy',random_state=42)
classifier.fit(x_train, y_train)

y_pred = classifier.predict_proba(x_test)



from sklearn.linear_model import LogisticRegression 
classifier1=LogisticRegression(random_state=0)
classifier1.fit(x_train, y_train) 
y_pred1=classifier1.predict_proba(x_test)
from sklearn.naive_bayes import MultinomialNB
classifier2=MultinomialNB()
classifier2.fit(x_train, y_train)
y_pred2=classifier2.predict_proba(x_test)
from sklearn.svm import LinearSVC 
classifier3=LinearSVC(random_state=42, max_iter=200)
classifier3.fit(x_train, y_train)
y_pred3=classifier3.decision_function(x_test)
from sklearn.linear_model import  SGDClassifier
classifier4 =  SGDClassifier(random_state=42, max_iter=200)
classifier4.fit(x_train, y_train) 
y_pred4 = classifier4.decision_function(x_test)
from sklearn.tree import DecisionTreeClassifier 
classifier5=DecisionTreeClassifier(criterion='entropy', random_state=0)
classifier5.fit(x_train, y_train)
y_pred5=classifier5.predict_proba(x_test)
from sklearn.metrics import confusion_matrix,plot_confusion_matrix,accuracy_score ,f1_score, precision_score, recall_score,classification_report, plot_roc_curve,roc_auc_score
cm = confusion_matrix(y_test, y_pred)
print(cm)
#plot_confusion_matrix(classifier,x_test,y_test,values_format='d')
#plt.show()
print(accuracy_score(y_test, y_pred))
print(f1_score(y_test, y_pred,average='macro'))
print(precision_score(y_test,y_pred,average='macro'))
print(recall_score(y_test,y_pred,average='macro'))
print(classification_report(y_test, y_pred))

#plot_roc_curve(classifier,x_test,y_test)

print(roc_auc_score(y_test,y_pred,average='macro'))
from sklearn.metrics import roc_curve
print(len(y_pred))

print(y_pred.shape)
#y_pred = y_pred[:,1]
rad_fpr, rad_tpr,_= roc_curve(y_test,y_pred,pos_label=4)
#roc_auc = auc(lr_fpr, lr_tpr)
print(rad_fpr.shape)
print(rad_tpr.shape)
plt.plot(rad_fpr, rad_tpr,  marker='.', label='Random Forest' )

plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
# show the legend

# show the plot


from sklearn.metrics import roc_curve
y_pred1 = y_pred1[:,1]
print(y_pred1.shape)
lr_fpr, lr_tpr,_= roc_curve(y_test,y_pred1,pos_label=4)
#roc_auc = auc(lr_fpr, lr_tpr)
print(lr_fpr.shape)
print(lr_tpr.shape)
plt.plot(lr_fpr, lr_tpr,  marker='.', label='Logistic' )

plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
# show the legend

# show the plot

from sklearn.metrics import roc_curve
print(y_pred2.shape)
y_pred2 = y_pred2[:,1]
nb_fpr, nb_tpr,_= roc_curve(y_test,y_pred2,pos_label=4)
#roc_auc = auc(lr_fpr, lr_tpr)
print(nb_fpr.shape)
print(nb_tpr.shape)
plt.plot(nb_fpr, nb_tpr,  marker='.', label='Naive Bayes' )

plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
# show the legend

# show the plot


print(y_pred3.shape)
#y_pred3 = y_pred3[:,1]
svm_fpr, svm_tpr,_= roc_curve(y_test,y_pred3,pos_label=4)
#roc_auc = auc(lr_fpr, lr_tpr)
print(nb_fpr.shape)
print(nb_tpr.shape)
plt.plot(svm_fpr, svm_tpr,  marker='.', label='SVM' )

plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
# show the legend

# show the plot


from sklearn.metrics import roc_curve
print(y_pred4.shape)
#y_pred4 = y_pred4[:,1]
SGD_fpr, SGD_tpr,_= roc_curve(y_test,y_pred4,pos_label=4)
#roc_auc = auc(lr_fpr, lr_tpr)
print(SGD_fpr.shape)
print(SGD_tpr.shape)
plt.plot(SGD_fpr, SGD_tpr,  marker='.', label='SGDC' )

plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
# show the legend

# show the plot


from sklearn.metrics import roc_curve
print(y_pred5.shape)
y_pred5 = y_pred5[:,1]
dt_fpr, dt_tpr,_= roc_curve(y_test,y_pred5,pos_label=4)
#roc_auc = auc(lr_fpr, lr_tpr)
print(dt_fpr.shape)
print(dt_tpr.shape)
plt.plot(dt_fpr, dt_tpr,  marker='.', label='Decision Tree' )

plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
# show the legend

# show the plot


plt.show()
ns_probs = [0 for _ in range(len(y_test))]

ns_fpr, ns_tpr, _ = roc_curve(y_test, ns_probs,pos_label=4)


plt.plot(ns_fpr, ns_tpr, linestyle='--', label='No Skill')
plt.plot(rad_fpr, rad_tpr,  linestyle='-', label='Random Forest' )
plt.plot(lr_fpr, lr_tpr,  linestyle='--', label='Logistic' )
plt.plot(nb_fpr, nb_tpr,  linestyle='--', label='Naive Bayes' )
plt.plot(svm_fpr, svm_tpr,  linestyle='--', label='SVM' )
plt.plot(SGD_fpr, SGD_tpr,  linestyle='--', label='SGDC' )
plt.plot(dt_fpr, dt_tpr,  linestyle='--', label='Decision Tree' )
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.show()