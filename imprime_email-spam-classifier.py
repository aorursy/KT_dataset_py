import numpy as np

import pandas as pd
data = pd.read_csv('../input/hamspam.csv', names= ['no', 'label', 'news'])

data = data.drop('no', axis=1)
data.head()
data.head()
data.describe()
data.info()
data.describe()
data.groupby('label').describe()

data['length']=data['news'].apply(len)

data.head()
import matplotlib.pyplot as plt

%matplotlib inline
data['length'].plot(bins=50,kind='hist')
data['length'].describe()
data[data['length']==910]['news'].iloc[0]
import string

mess = 'sample Sample i me my myself we our ours Message message message message message!...'

nopunc=[char for char in mess if char not in string.punctuation]

nopunc=''.join(nopunc)

print(nopunc)
from nltk.corpus import stopwords

stopwords.words('english')[0:10]
clean_mess=[word for word in nopunc.split() if word.lower() not in stopwords.words('english')]

clean_mess
def text_process(mess):

    nopunc =[char for char in mess if char not in string.punctuation]

    nopunc=''.join(nopunc)

    return [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]
# Here is the original DataFrame again:



data.head()
data['news'].head(5).apply(text_process)

from sklearn.feature_extraction.text import CountVectorizer



#t's take one text message and get its bag-of-words counts as a vector, putting to use our new bow_transformer:



bow_transformer = CountVectorizer(analyzer=text_process).fit_transform(data['news'])
bow_transformer.shape
#Split the data into 80% training (X_train & y_train) and 20% testing (X_test & y_test) data sets

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(bow_transformer, data['label'], test_size = 0.20, random_state = 0)
#Create and train the Naive Bayes classifier

#The multinomial Naive Bayes classifier is suitable for classification with discrete features (e.g., word counts for text classification)

from sklearn.naive_bayes import MultinomialNB

classifier = MultinomialNB()

classifier.fit(X_train, y_train)
#Print the predictions

print(classifier.predict(X_train))



#Print the actual values

print(y_train.values)
#Evaluate the model on the training data set

from sklearn.metrics import classification_report,confusion_matrix, accuracy_score

pred = classifier.predict(X_train)

print(classification_report(y_train ,pred ))

print('Confusion Matrix: \n',confusion_matrix(y_train,pred))

print()

print('Accuracy: ', accuracy_score(y_train,pred))
#Print the predictions

print('Predicted value: ',classifier.predict(X_test))



#Print Actual Label

print('Actual value: ',y_test.values)
#Evaluate the model on the test data set

from sklearn.metrics import classification_report,confusion_matrix, accuracy_score

pred = classifier.predict(X_test)

print(classification_report(y_test ,pred ))



print('Confusion Matrix: \n', confusion_matrix(y_test,pred))

print()

print('Accuracy: ', accuracy_score(y_test,pred))
from sklearn.feature_extraction.text import CountVectorizer
bow_transformer = CountVectorizer(analyzer=text_process).fit(data['news'])
# Let's take one text message and get its bag-of-words counts as a vector, putting to use our new bow_transformer:

print(len(bow_transformer.vocabulary_))

news4=data['news'][3]

print(news4)

# Now let's see its vector representation:

bow4=bow_transformer.transform([news4])

print(bow4)

print(bow4.shape)



# This means that there are seven unique words in message number 7 (after removing common stop words).

# One of them appear twice, the rest only once. Let's go ahead and check and confirm which ones appear twice:



print(bow_transformer.get_feature_names()[7794])
news_bow = bow_transformer.transform(data['news'])

print('Shape of Sparse Matrix: ',news_bow.shape)

print('Amount of non-zero occurences:',news_bow.nnz)

news_bow
sparsity =(100.0 * news_bow.nnz/(news_bow.shape[0]*news_bow.shape[1]))

print('sparsity:{}'.format(round(sparsity)))
from sklearn.feature_extraction.text import TfidfTransformer

tfidf_transformer=TfidfTransformer().fit(news_bow)

tfidf = tfidf_transformer.transform(bow4)

print(tfidf)
print(tfidf_transformer.idf_[bow_transformer.vocabulary_['u']])

print(tfidf_transformer.idf_[bow_transformer.vocabulary_['university']])
news_tfidf=tfidf_transformer.transform(news_bow)

print(news_tfidf.shape)
from sklearn.model_selection import train_test_split

X_tfidf_train,X_tfidf_test,y_tfidf_train,y_tfidf_test = train_test_split(news_tfidf,data['label'],test_size=0.25)

print(len(X_train),len(X_test),len(y_train),len(y_test))
from sklearn.naive_bayes import MultinomialNB

spam_detect_model = MultinomialNB().fit(X_tfidf_train,y_tfidf_train)

predict = spam_detect_model.predict(X_tfidf_test)

print('predicted:',predict)

print('expected:',data.label.values)
from sklearn.metrics import classification_report,confusion_matrix

print(classification_report(predict,y_tfidf_test))

print(confusion_matrix(predict,y_tfidf_test))
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(data['news'],data['label'],test_size=0.25)

print(len(X_train),len(X_test),len(y_train),len(y_test))
from sklearn.pipeline import Pipeline

pipeline = Pipeline([

   ( 'bow',CountVectorizer(analyzer=text_process)),

    ('tfidf',TfidfTransformer()),

    ('classifier',MultinomialNB()),

])

pipeline.fit(X_train,y_train)
predictions = pipeline.predict(X_test)


print(predictions)
print(classification_report(predictions,y_test))

print(confusion_matrix(y_test,predictions))