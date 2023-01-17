import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.model_selection import train_test_split

from sklearn.naive_bayes import MultinomialNB

from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score

df = pd.read_csv('../input/spam.csv', encoding='latin-1')

df = df[['v1', 'v2']]

df.columns = ['label', 'sms_message']

df.head()
df['label'] = df.label.map({'ham':0, 'spam':1})

print(df.shape)

df.head()
X_train, X_test, y_train, y_test = train_test_split(df['sms_message'], 

                                                    df['label'], 

                                                    random_state = 1)

print("Number of rows in the original set: {}".format(df.shape[0]))

print("Number of rows in the training set: {}".format(X_train.shape[0]))

print("Number of rows in the test set: {}".format(X_test.shape[0]))
# Instantiate the CountVectorizer method

count_vector = CountVectorizer()

print(count_vector)
# Fit the training data and then return the matrix

training_data = count_vector.fit_transform(X_train)



# Transform testing data and return the matrix. 

# Note we are not fitting the testing data into the CountVectorizer()

testing_data = count_vector.transform(X_test)
# Create the Naive Bayes classifier and fit with training set

naive_bayes = MultinomialNB()

naive_bayes.fit(training_data, y_train)
predictions = naive_bayes.predict(testing_data)

print('Accuracy score: {}'.format(accuracy_score(y_test, predictions)))

print('Precision score: {}'.format(precision_score(y_test, predictions)))

print('Recall score: {}'.format(recall_score(y_test, predictions)))

print('F1 score: {}'.format(f1_score(y_test, predictions)))