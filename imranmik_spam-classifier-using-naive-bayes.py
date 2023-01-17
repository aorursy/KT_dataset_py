

import pandas as pd

df = pd.read_csv('../input/SMSSpamCollection',sep='\t',header=None, names=['labels','sms_messages'])

df.head()
df['label'] = df.labels.map({'ham':0,'spam':1})

df['label'].head()


documents = ['Hello, how are you!',

             'Win money, win from home.',

             'Call me now.',

             'Hello, Call hello you tomorrow?']



lower_case_documents = []

for i in documents:

    lower_case_documents.append(i.lower())

print(lower_case_documents)


sans_punctuation_documents = []

import string



for i in lower_case_documents:

    sans_punctuation_documents.append(i.translate(str.maketrans('','',string.punctuation)))

    

print(sans_punctuation_documents)


preprocessed_documents = []

for i in sans_punctuation_documents:

    preprocessed_documents.append(i.split(' '))

print(preprocessed_documents)


frequency_list = []

import pprint

from collections import Counter



for i in preprocessed_documents:

    count=Counter(i)

    frequency_list.append(count)

    

pprint.pprint(frequency_list)


documents = ['Hello, how are you!',

                'Win money, win from home.',

                'Call me now.',

                'Hello, Call hello you tomorrow?']
from sklearn.feature_extraction.text import CountVectorizer

count_vector = CountVectorizer()

print(count_vector)


count_vector.fit(documents)

count_vector.get_feature_names()


doc_array = count_vector.transform(documents).toarray()

doc_array


frequency_matrix = pd.DataFrame(doc_array,columns=count_vector.get_feature_names())

frequency_matrix


from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score





X_train, X_test, y_train, y_test = train_test_split(df['sms_messages'], 

                                                    df['label'], 

                                                    random_state=1)



print('Number of rows in the total set: {}'.format(df.shape[0]))

print('Number of rows in the training set: {}'.format(X_train.shape[0]))

print('Number of rows in the test set: {}'.format(X_test.shape[0]))


count_vector = CountVectorizer()



training_data = count_vector.fit_transform(X_train)



testing_data = count_vector.transform(X_test)


from sklearn.naive_bayes import MultinomialNB

naive_bayes = MultinomialNB()

naive_bayes.fit(training_data,y_train)


predictions = naive_bayes.predict(testing_data)

predictions


print('Accuracy score: ', format(accuracy_score(y_test,predictions)))

print('Precision score: ', format(precision_score(y_test,predictions)))

print('Recall score: ', format(recall_score(y_test,predictions)))

print('F1 score: ', format(f1_score(y_test,predictions)))