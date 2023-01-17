import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt

data_dir = "../input/"

df = pd.read_csv(data_dir + '/spam.csv', encoding='latin-1')
# observe the data
print(df.head())
df.shape
from sklearn.model_selection import train_test_split

# split into train and test
data_train, data_test, labels_train, labels_test = train_test_split(
    df.v2,
    df.v1, 
    test_size=0.2, 
    random_state=0) 

print (data_train.head())
print (labels_train.head())
from sklearn.feature_extraction.text import CountVectorizer

vect = CountVectorizer()
example = ['I love you, good bad bad', 'you are soo good']

result = vect.fit_transform(example)
print(result)
print (vect.vocabulary_)
print('\n')

result1 = vect.transform(example)
print(result1)
print (vect.vocabulary_)
vectorizer = CountVectorizer()

data_train_count = vectorizer.fit_transform(data_train)
data_test_count  = vectorizer.transform(data_test)
print (data_train_count.shape)
print (data_test_count.shape)
# print (vectorizer.vocabulary_)
# Count the total numbers of unique word
def GetVocabulary(data): 
    vocab_set = set([])
    for document in data:
        words = document.split()
        for word in words:
            vocab_set.add(word) 
    return list(vocab_set)

vocab_list = GetVocabulary(data_train)
print ('Number of all the unique words : ' + str(len(vocab_list)))
# function that convert sentences into word vectors
def Document2Vector(vocab_list, data):
    word_vector = np.zeros(len(vocab_list))
    words = data.split()
    for word in words:
        if word in vocab_list:
            word_vector[vocab_list.index(word)] += 1
    return word_vector

print (data_train[1:2,])
print (data_train.values[2])
train_matrix = []
for document in data_train.values:
    word_vector = Document2Vector(vocab_list, document)
    train_matrix.append(word_vector)

print (len(train_matrix))
word_freq_df = pd.DataFrame({'term': vectorizer.get_feature_names(), 'occurrences':data_train_count.toarray().sum(axis=0)})
word_freq_df['frequency'] = word_freq_df['occurrences'] / np.sum(word_freq_df['occurrences'])
plt.plot(word_freq_df.occurrences)
plt.show()

word_freq_df_sort = word_freq_df.sort_values(by=['occurrences'], ascending=False)
print(word_freq_df_sort.head())
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

model = MultinomialNB()
model.fit(data_train_count, labels_train)
predictions = model.predict(data_test_count)
print(predictions)
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
from sklearn.model_selection import cross_val_score

print (accuracy_score(labels_test, predictions))
print (classification_report(labels_test, predictions))
print (confusion_matrix(labels_test, predictions))

cross_val = cross_val_score(model, data_train_count, labels_train, cv=20, scoring='accuracy')
print (cross_val)
print (np.mean(cross_val))