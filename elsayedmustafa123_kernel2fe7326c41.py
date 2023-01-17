# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import matplotlib.pyplot as plt

import re

import nltk 

from nltk.corpus import stopwords

from  nltk.stem.porter import PorterStemmer

nltk.download('stopwords')

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.preprocessing import LabelEncoder

from sklearn.tree import DecisionTreeClassifier

from sklearn.linear_model import Lasso

from sklearn.model_selection import GridSearchCV

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split

from keras.models import Sequential

from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D,Dropout

from keras.utils.np_utils import to_categorical

from keras import regularizers

from sklearn import metrics

from sklearn.metrics import classification_report,multilabel_confusion_matrix

Tweets= pd.read_csv("/kaggle/input/twitter-airline-sentiment/Tweets.csv")

Tweets.head()
classes_count=Tweets['airline_sentiment'].value_counts()
x=[1,2,3]

plt.bar(x,classes_count)

plt.xticks(x,['negative','neutral','positive'])

plt.xlabel('Classes')

plt.ylabel('Number of Sentence')

plt.title('Sentences in each class')
Tweets['tw_len'] = Tweets['text'].apply(len)

Tweets.groupby(['tw_len', 'airline_sentiment']).size().unstack().plot(kind='line', stacked=False)
print(max(Tweets['tw_len']))

print(min(Tweets['tw_len']))

print(np.mean(Tweets['tw_len']))
Tweets=Tweets[['text','airline_sentiment']]

Tweets
Tweets['text'][1]
corpus=[]

for i in range(14640):

    review=re.sub('[^a-zA-z]',' ',Tweets['text'][i])

    review=review.lower()

    review=review.split()

    

    # remove stopwords

    review=[word for word in review if not word in set(stopwords.words('english'))]

    

    

    

    #to taken the route of the word

    ps=PorterStemmer()

    review=[ps.stem(word) for word in  review]

    review=' ' .join(review)

    corpus.append(review)
corpus
cv=CountVectorizer(analyzer = 'word',max_features=8000)

X=cv.fit_transform(corpus).toarray()
X.shape
y= Tweets['airline_sentiment']

labelencoder=LabelEncoder()

y=labelencoder.fit_transform(y)



for i in range (4):

    print('class :',Tweets['airline_sentiment'][i] ,'--> label is :',y[i])
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3)


clasifier=RandomForestClassifier(n_estimators=20,criterion='entropy')

clasifier.fit(X_train,y_train)







##evaluate model 

y_pred=clasifier.predict(X_test)



print(clasifier.score(X_train,y_train))



print(clasifier.score(X_test,y_test))



clasifier= DecisionTreeClassifier()

clasifier.fit(X_train,y_train)





#evaluate model 

y_pred=clasifier.predict(X_test)



print(clasifier.score(X_train,y_train))

print(clasifier.score(X_test,y_test))


random_classifier=RandomForestClassifier()



parameters = { 'max_features':np.arange(5,10),'n_estimators':[400],'min_samples_leaf': [10,30,50,100,200,500]}



random_grid = GridSearchCV(random_classifier, parameters, cv = 5)

random_grid.fit(X_train,y_train)







##evaluate model 

y_pred=random_grid.predict(X_test)



print(random_grid.score(X_train,y_train))



print(random_grid.score(X_test,y_test))

RF_ACC=random_grid.score(X_test,y_test)
acc=random_grid.score(X_test,y_test)

precision=metrics.precision_score(y_test, y_pred, average='macro') 

recall=metrics.recall_score(y_test, y_pred, average='micro')

f1_score=metrics.f1_score(y_test, y_pred, average='weighted')  

cm=multilabel_confusion_matrix(y_test, y_pred,labels=[0, 1,2])

print ('confusion matrix ',cm)

print('Accuracy : ',acc)

print('Precision : ',precision)

print('Recall : ',recall)

print('F1_score : ',f1_score)
y=to_categorical(y)
for i in range (4):

    print('class :',Tweets['airline_sentiment'][i] ,'--> label is :',y[i])
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.15)

print(X_train.shape,y_train.shape)

print(X_test.shape,y_test.shape)
validation_size = 2196



X_validate = X_train[-validation_size:]

y_validate = y_train[-validation_size:]

X_train = X_train[:-validation_size]

y_train = y_train[:-validation_size]
print(X_train.shape,y_train.shape)

print(X_validate.shape,y_validate.shape)

print(X_test.shape,y_test.shape)
reg_model = Sequential()

reg_model.add(Dense(64, kernel_regularizer=regularizers.l2(0.03), activation='relu', input_dim=X.shape[1]))

reg_model.add(Dropout(0.5))

reg_model.add(Dense(64, kernel_regularizer=regularizers.l2(0.02), activation='relu'))

reg_model.add(Dropout(0.5))

reg_model.add(Dense(3, activation='softmax'))

reg_model.summary()
reg_model.compile(optimizer='Adam'

                  , loss='categorical_crossentropy'

                  , metrics=['accuracy'])

    

history = reg_model.fit(X_train

                       , y_train

                       , epochs=20

                       , batch_size=64

                       , validation_data=(X_validate, y_validate)

                       , verbose=1)
score=reg_model.evaluate(X_test,y_test,verbose=0)

print('Accuracy : ' ,score[1])

plt.plot(history.history['accuracy'])

plt.plot(history.history['val_accuracy'])

plt.legend(['train','validation'])

plt.title('accuracy')

plt.xlabel('epochs')
plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])

plt.legend(['training','test'])

plt.title('Loss')

plt.xlabel('epochs')
Models=['Machine Leaning Model ','Deep Learning Model ']

accu=[RF_ACC,score[1]]

     

x = [1,2]

plt.bar(x,accu)

plt.xticks(x, Models,rotation=45)

plt.ylabel('Accuracy')

plt.xlabel('Models')

plt.title('Accuracies of Models')