import nltk
import pandas as pd

Data=pd.read_csv('../input/imdb-review-stemming/IMDB Review Dataset.csv')

Data.head(10)



Data['review']
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


ps=PorterStemmer()
import re 

for i in range(len(Data)):
    #print(Data['review'][i])
    #print(i)
    text=re.sub(r'[^a-zA-Z]',' ',Data['review'][i])
    text=text.lower()
    text=text.split(' ')
    text=[ps.stem(word) for word in text if word not in set(stopwords.words('english'))]
    
    Data['review'][i]=' '.join(text)
    #print(Data['review'][i])
    #break
    
    
Data.to_csv('../input/imdb-review-stemming/PreProcessedStemming.csv')    
# Pre-Proccessed review by removing the following.
#  1. Remove special character
#  2. remove numbers
#  3. Removing Stopwords
#  4. stemming of words

Data=pd.read_csv('../input/imdb-review-stemming/PreProcessedStemming.csv')


Data.head()
Data.shape
corpus=[]
import re


for i in range(0,len(Data)):
    
    review=re.sub('[^a-zA-Z]',' ', Data['review'][i])
    review.lower()
    review=review.split(' ')
    #review=[ps.stem(word) for word in reviews if  not word in set(stopwords.words('english'))]
    
    
    review=' '.join(review)
    corpus.append(review)
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

cv=CountVectorizer(max_features=10000,ngram_range=(1,3))
X=cv.fit_transform(corpus).toarray()
X.shape

corpus=[]

#y=Data['sentiment']

# label encoding of sentiment

# 0 - Positive
# 1 - Negative

from sklearn.preprocessing import LabelEncoder

lb_make = LabelEncoder()
Data['sentiment'] = lb_make.fit_transform(Data['sentiment'])


y=Data['sentiment']

y

## Train Test Split


from sklearn.model_selection import train_test_split

X_train , X_test, y_train , y_test=train_test_split(X,y,test_size=0.3,random_state=2)
# Model Selection Naive Bayes

from sklearn.naive_bayes import MultinomialNB

model=MultinomialNB().fit(X_train,y_train)
y_pred=model.predict(X_test)
from sklearn.metrics import confusion_matrix

confusionM=confusion_matrix(y_test,y_pred)
confusionM
TP=confusionM[0][0]
FP=confusionM[0][1]
FN=confusionM[1][0]
TN=confusionM[1][1]



accuracy=((TP + TN)/(TP+FP+FN+TN)) * 100
print(accuracy)

#Precision tells us how many of the correctly predicted cases actually turned out to be positive.
precision=(TP/(TP +FP) ) * 100
print(precision)

#Recall tells us how many of the actual positive cases we were able to predict correctly with our model.
recall= (TP / (TP + FN)) * 100
print(recall)
#Or we can use direct library avaiable to get accuracy of the prediction.
from sklearn.metrics import accuracy_score

accuracyscore=accuracy_score(y_test,y_pred)
accuracyscore

#apply LogisticRegression classfier
from sklearn.linear_model import LogisticRegression
lg = LogisticRegression(class_weight='balanced').fit(X_train, y_train)
print (lg.coef_)
print('training set score obtained Logistic Regression: {:.2f}'.format(lg.score(X_train, y_train)))
print('test set score obtained Logistic Regression: {:.2f}'.format(lg.score(X_test, y_test)))
y_pred = lg.predict(X_test)
confusion_matrix = confusion_matrix(y_test, y_pred)
print(confusion_matrix)
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt

logit_roc_auc = roc_auc_score(y_test, lg.predict_proba(X_test)[: ,1])
fpr, tpr, thresholds = roc_curve(y_test, lg.predict_proba(X_test)[:,1])
# print(thresholds)
plt.figure()
plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % logit_roc_auc)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.show()
