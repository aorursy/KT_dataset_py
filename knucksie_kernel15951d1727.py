import pandas
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multiclass import OneVsRestClassifier

# reading data from provided document into pandas DataFrame
names = ('id', 'sentence', 'author')
dataframe = pandas.read_csv('../input/traincsv/train.csv', names=names)

# creaing tf_idf matrix based on sentences
vectorizer = TfidfVectorizer()
sentences_tf_idf = vectorizer.fit_transform(dataframe['sentence'])

# training a OVR classifier based on a naive Bayes algorithm
model = MultinomialNB()
classifier = OneVsRestClassifier(model).fit(sentences_tf_idf, dataframe['author'])

# trying to predict author
test_dataframe = pandas.read_csv('../input/spooky-author-ident-csv/test.csv', names=('id', 'sentence'))
test_matrix = vectorizer.transform(test_dataframe['sentence'])
predict_probas = classifier.predict_proba(test_matrix)

# writing the result
predicted_df = pandas.DataFrame(predict_probas, columns=('EAP', 'HPL', 'MWS'))
predicted_df.insert(0, 'id', test_dataframe['id'])

predicted_df.to_csv('sample.csv', index=False)