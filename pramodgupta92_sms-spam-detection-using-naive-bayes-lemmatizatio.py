import nltk
import pandas as pd

#To check the encoding of the CSV in order to avoid issue which reading CSV using Pandas.

import chardet
with open('/kaggle/input/sms-spam-collection-dataset/spam.csv', 'rb') as rawdata:
    result = chardet.detect(rawdata.read(100000))
result
data=pd.read_csv('/kaggle/input/sms-spam-collection-dataset/spam.csv',encoding='Windows-1252')
data.head()

#Remove the unwanted columns from dataset and change the label v1 and v2 to label and message respectively
data=data[['v1','v2']]
data.columns=['label','message']
data
import nltk
import re

from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
corpus=[]
wn=WordNetLemmatizer()


for i in range(0,len(data)):
    
    review=re.sub('[^a-zA-Z]',' ', data['message'][i])
    review.lower()
    reviews=review.split(' ')
    review=[wn.lemmatize(word) for word in reviews if  not word in set(stopwords.words('english'))]
    
    
    review=' '.join(review)
    corpus.append(review)
    
    


#using Bag Of Word

from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer()

X=cv.fit_transform(corpus).toarray()
X.shape


y=pd.get_dummies(data['label'])
y=y.iloc[:,1].values
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