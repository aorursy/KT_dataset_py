#Naive bayes
import pandas as pd
import numpy as np
import sklearn as sk
train_data = pd.read_csv(r'../input/sentiment-analysis/train_2kmZucJ.csv')
print(np.shape(train_data))
pd.set_option("display.max_rows",None)
pd.set_option("display.max_columns", None)
#print(train_data[:3].head(10))
test_data = pd.read_csv(r'../input/sentiment-analysis/test_oJQbWVk.csv')
print(np.shape(test_data))
#print(test_data.head(20))
#train_data['label'] = train_data['Computer Science'].astype(str) + train_data['Physics'].astype(str) + train_data['Mathematics'].astype(str) +train_data['Statistics'].astype(str) +  train_data['Quantitative Biology'].astype(str) +train_data['Quantitative Finance'].astype(str)

#print(train_data['label'])

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(train_data['tweet'], train_data['label'], random_state=1)
#print(X_test)

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(strip_accents='ascii', token_pattern=u'(?ui)\\b\\w*[a-z]+\\w*\\b', lowercase=True, stop_words='english')
X_train_cv = cv.fit_transform(X_train)
X_test_cv = cv.transform(X_test)
#print(X_train_cv)
from sklearn.naive_bayes import MultinomialNB
naive_bayes = MultinomialNB()
naive_bayes.fit(X_train_cv, y_train)
predictions = naive_bayes.predict(X_test_cv)
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
print('Accuracy score: ', accuracy_score(y_test, predictions))
print('Precision score: ', precision_score(y_test, predictions,average='micro'))
print('Recall score: ', recall_score(y_test, predictions,average='micro'))

#cv1 = CountVectorizer(strip_accents='ascii', token_pattern=u'(?ui)\\b\\w*[a-z]+\\w*\\b', lowercase=True, stop_words='english')
z_test_cv = cv.transform(test_data['tweet'])
pred_test = naive_bayes.predict(z_test_cv)
#print(pred_test)
type(pred_test)
df = pd.DataFrame(data =pred_test)
test_data['label'] = df
#print(test_data.head(20))
test_data= test_data.drop(columns=['tweet'])
print(test_data.head(20))
test_data.to_csv(r'D:\Sambita\hackathon-ideathon\Analytics Vidhya\Identifying sentiments\submission1.csv',index=False)

#Logistic regression
import pandas as pd
import sklearn as sk
train = pd.read_csv(r'../input/sentiment-analysis/train_2kmZucJ.csv')
test = pd.read_csv(r'../input/sentiment-analysis/test_oJQbWVk.csv')
pd.set_option("display.max_columns",None)
train.label.value_counts()# gives count of positive and negative
from sklearn.model_selection import train_test_split 
x_train,x_test,y_train,y_test = train_test_split(train['tweet'],train['label'],test_size =0.2 ,random_state=42)
#print(x_test.head(20))
print(x_train.shape)
print(x_test.shape)
#count vectorizer

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(min_df=0,token_pattern=u'(?ui)\\b\\w*[a-z]+\\w*\\b', lowercase=False,stop_words ='english')
vc=cv.fit(x_train)
#print(vc.vocabulary_)
vc_train = cv.fit_transform(x_train).toarray()
vc_test = cv.fit_transform(x_test).toarray()
print(vc_test[230])
'''This vocabulary serves also as an index of each word.
Now, you can take each sentence and get the word occurrences of the words based on the previous vocabulary. '''
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(vc_train[:,:8094],y_train)
pred = model.predict(vc_test)
#print(pred)
from sklearn.metrics import accuracy_score, precision_score,recall_score, f1_score
print('accuracy score is {:4f}'.format(accuracy_score(y_test,pred)))
print('Precision score: ', precision_score(y_test, pred,average='micro'))
print('Recall score: ', recall_score(y_test, pred,average='micro'))
print('f1 score is {:4f}'.format(f1_score(y_test,pred)))
z_test_cv = cv.transform(test['tweet'])
pred1 =model.predict(z_test_cv)
test['label']=pred1
print(test['label'].value_counts())
test= test.drop(columns=['tweet'])
print(test_data.head(20))
test.to_csv(r'D:\Sambita\hackathon-ideathon\Analytics Vidhya\Identifying sentiments\submission2.csv',index=False)
