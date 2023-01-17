import warnings
warnings.filterwarnings("ignore")
import sqlite3
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
import numpy as np
from sklearn import cross_validation
from sklearn.cross_validation import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.cross_validation import cross_val_score
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import confusion_matrix
import seaborn as sb
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
import gensim
from gensim.models import Word2Vec, KeyedVectors
# Connection to the dataset
con = sqlite3.connect('../input/database.sqlite')

# It is given that the table name is 'Reviews'
# Creating pandas dataframe and storing into variable 'dataset' by help of sql query
dataset = pd.read_sql_query("""
SELECT *
FROM Reviews
""", con)

# Getting the shape of actual data: row, column
display(dataset.shape)
# Displaying first 5 data points
display(dataset.head())
# Considering only those reviews which score is either 1,2 or 4,5
# Since, 3 is kind of neutral review, so, we are eliminating it
filtered_data = pd.read_sql_query("""
SELECT *
FROM Reviews
WHERE Score != 3
""", con)
# Getting shape of new dataset
display(filtered_data.shape)
# Changing the scores into 'positive' or 'negative'
# Score greater that 3 is considered as 'positive' and less than 3 is 'negative'
def partition(x):
    if x>3:
        return 'positive'
    return 'negative'

actual_score = filtered_data['Score']
positiveNegative = actual_score.map(partition)
filtered_data['Score'] = positiveNegative
# Sorting data points according to the 'ProductId'
sorted_data = filtered_data.sort_values('ProductId', axis=0, ascending=True, inplace=False, kind='quicksort', na_position='last')

# Eliminating the duplicate data points based on: 'UserId', 'ProfileName', 'Time', 'Summary'
final = sorted_data.drop_duplicates(subset={'UserId', 'ProfileName', 'Time', 'Summary'}, keep='first', inplace=False)

# Eliminating the row where 'HelpfulnessDenominator' is greater than 'HelpfulnessNumerator' as these are the wrong entry
final = final[final['HelpfulnessDenominator'] >= final['HelpfulnessNumerator']]

# Getting shape of final data frame
display(final.shape)
%%time

# Creating the set of stopwords
stop = set(stopwords.words('english'))

# For stemming purpose
snow = nltk.stem.SnowballStemmer('english')

# Defining function to clean html tags
def cleanhtml(sentence):
    cleaner = re.compile('<.*>')
    cleantext = re.sub(cleaner, ' ', sentence)
    return cleantext

# Defining function to remove special symbols
def cleanpunc(sentence):
    cleaned = re.sub(r'[?|.|!|*|@|#|\'|"|,|)|(|\|/]', r'', sentence)
    return cleaned


# Important steps to clean the text data. Please trace it out carefully
i = 0
str1 = ''
all_positive_words = []
all_negative_words = []
final_string = []
s=''
for sent in final['Text'].values:
    filtered_sentence = []
    sent = cleanhtml(sent)
    for w in sent.split():
        for cleaned_words in cleanpunc(w).split():
            if ((cleaned_words.isalpha()) & (len(cleaned_words)>2)):
                if (cleaned_words.lower() not in stop):
                    s = (snow.stem(cleaned_words.lower())).encode('utf-8')
                    filtered_sentence.append(s)
                    if (final['Score'].values)[i] == 'positive':
                        all_positive_words.append(s)
                    if (final['Score'].values)[i] == 'negative':
                        all_negative_words.append(s)
                else:
                    continue
            else:
                continue
    str1 = b" ".join(filtered_sentence)
    final_string.append(str1)
    i += 1
    
# Adding new column into dataframe to store cleaned text
final['CleanedText'] = final_string
final['CleanedText'] = final['CleanedText'].str.decode('utf-8')

# Creating new dataset with cleaned text for future use
conn = sqlite3.connect('final.sqlite')
c = conn.cursor()
conn.text_factory = str
final.to_sql('Reviews', conn, schema=None, if_exists='replace', index=True, index_label=None, chunksize=None, dtype=None)

# Getting shape of new datset
print(final.shape)
# Creating connection to read from database
conn = sqlite3.connect('./final.sqlite')

# Creating data frame for visualization using sql query
final = pd.read_sql_query("""
SELECT *
FROM Reviews
""", conn)
# Displaying first 3 indices
display(final.head(3))
# Sampling positive and negative reviews
positive_points = final[final['Score'] == 'positive'].sample(
    n=10000, random_state=0)
negative_points = final[final['Score'] == 'negative'].sample(
    n=10000, random_state=0)
total_points = pd.concat([positive_points, negative_points])

# Sorting based on time
total_points['Time'] = pd.to_datetime(
    total_points['Time'], origin='unix', unit='s')
total_points = total_points.sort_values('Time')
sample_points = total_points['CleanedText']
labels = total_points['Score']#.map(lambda x: 1 if x == 'positive' else 0).values
# Splitting into train and test
X_train, X_test, Y_train, Y_test = train_test_split(
    sample_points, labels, test_size=0.30, random_state=0)
count_vect = CountVectorizer(ngram_range=(1, 1))
X_train = count_vect.fit_transform(X_train)
X_test = count_vect.transform(X_test)
%%time

neighbors = list(range(20, 80, 4))
cv_score = []
for k in neighbors:
    knn = KNeighborsClassifier(n_neighbors=k, algorithm='brute')
    scores = cross_val_score(knn, X_train, Y_train, cv=10, scoring='accuracy')
    cv_score.append(scores.mean())
MSE = [1 - x for x in cv_score]
optimal_k = neighbors[MSE.index(min(MSE))]
print("_" * 101)
print("Optimal number of neighbors: ", optimal_k)
print("_" * 101)
print("Missclassification error for each k values: ", np.round(MSE, 3))
print("_" * 101)

plt.plot(neighbors, MSE)
plt.title("Number of neighbors and error")
plt.xlabel("Number of neighbors")
plt.ylabel("Missclassification error")
plt.show()
%%time
optimal_model = KNeighborsClassifier(n_neighbors=optimal_k)
optimal_model.fit(X_train, Y_train)
prediction = optimal_model.predict(X_test)

training_accuracy = optimal_model.score(X_train, Y_train)
training_error = 1 - training_accuracy
test_accuracy = accuracy_score(Y_test, prediction)
test_error = 1 - test_accuracy

print("_" * 101)
print("Training Accuracy: ", training_accuracy)
print("Train Error: ", training_error)
print("Test Accuracy: ", test_accuracy)
print("Test Error: ", test_error)
print("_" * 101)
print("_" * 101)
print("Classification Report: \n")
print(classification_report(Y_test, prediction))
print("_" * 101)
conf_matrix = confusion_matrix(Y_test, prediction)
class_label = ['negative', 'positive']
df_conf_matrix = pd.DataFrame(
    conf_matrix, index=class_label, columns=class_label)
sb.heatmap(df_conf_matrix, annot=True, fmt='d')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()
print("_" * 101)
%%time

neighbors = list(range(20, 80, 4))
cv_score = []
for k in neighbors:
    knn = KNeighborsClassifier(n_neighbors=k, algorithm='kd_tree')
    scores = cross_val_score(knn, X_train, Y_train, cv=10, scoring='accuracy')
    cv_score.append(scores.mean())
MSE = [1 - x for x in cv_score]
optimal_k = neighbors[MSE.index(min(MSE))]
print("_" * 101)
print("Optimal number of neighbors: ", optimal_k)
print("_" * 101)
print("Missclassification error for each k values: ", np.round(MSE, 3))
print("_" * 101)

plt.plot(neighbors, MSE)
#for xy in zip(neighbors, np.round(MSE, 3)):
#    plt.annotate("%s %s" %xy, xy=xy, textcoords='data')
plt.title("Number of neighbors and error")
plt.xlabel("Number of neighbors")
plt.ylabel("Missclassification error")
plt.show()
%%time

optimal_model = KNeighborsClassifier(
    n_neighbors=optimal_k, algorithm='kd_tree')
optimal_model.fit(X_train, Y_train)
prediction = optimal_model.predict(X_test)

training_accuracy = optimal_model.score(X_train, Y_train)
training_error = 1 - training_accuracy
test_accuracy = accuracy_score(Y_test, prediction)
test_error = 1 - test_accuracy

print("_" * 101)
print("Training Accuracy: ", training_accuracy)
print("Train Error: ", training_error)
print("Test Accuracy: ", test_accuracy)
print("Test Error: ", test_error)
print("_" * 101)
print("Classification Report: \n")
print(classification_report(Y_test, prediction))
conf_matrix = confusion_matrix(Y_test, prediction)
class_label = ['negative', 'positive']
df_conf_matrix = pd.DataFrame(
    conf_matrix, index=class_label, columns=class_label)
sb.heatmap(df_conf_matrix, annot=True, fmt='d')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()
print("_" * 101)
# Splitting into train and test
X_train, X_test, Y_train, Y_test = train_test_split(
    sample_points, labels, test_size=0.30)
print(X_train.shape, X_test.shape, Y_train.shape, Y_test.shape)
# Initializing tfidf vectorizer
tfidf_vect = TfidfVectorizer(ngram_range=(1, 1))

# Fitting for tfidf vectorization
X_train = tfidf_vect.fit_transform(X_train)
X_test = tfidf_vect.transform(X_test)

print(X_train.shape, X_test.shape, Y_train.shape, Y_test.shape)
%%time

neighbors = list(range(20, 80, 4))
cv_score = []
for k in neighbors:
    knn = KNeighborsClassifier(n_neighbors=k, algorithm='brute')
    scores = cross_val_score(knn, X_train, Y_train, cv=10, scoring='accuracy')
    cv_score.append(scores.mean())
MSE = [1 - x for x in cv_score]
optimal_k = neighbors[MSE.index(min(MSE))]
print("_" * 101)
print("Optimal number of neighbors: ", optimal_k)
print("_" * 101)
print("Missclassification error for each k values: ", np.round(MSE, 3))
print("_" * 101)

plt.plot(neighbors, MSE)
#for xy in zip(neighbors, np.round(MSE, 3)):
#    plt.annotate("%s %s" %xy, xy=xy, textcoords='data')
plt.title("Number of neighbors and error")
plt.xlabel("Number of neighbors")
plt.ylabel("Missclassification error")
plt.show()
%%time

optimal_model = KNeighborsClassifier(
    n_neighbors=optimal_k, algorithm='kd_tree')
optimal_model.fit(X_train, Y_train)
prediction = optimal_model.predict(X_test)

training_accuracy = optimal_model.score(X_train, Y_train)
training_error = 1 - training_accuracy
test_accuracy = accuracy_score(Y_test, prediction)
test_error = 1 - test_accuracy

print("_" * 101)
print("Training Accuracy: ", training_accuracy)
print("Train Error: ", training_error)
print("Test Accuracy: ", test_accuracy)
print("Test Error: ", test_error)
print("_" * 101)
print("_" * 101)
print("Classification Report: \n")
print(classification_report(Y_test, prediction))
print("_" * 101)
conf_matrix = confusion_matrix(Y_test, prediction)
class_label = ['negative', 'positive']
df_conf_matrix = pd.DataFrame(
    conf_matrix, index=class_label, columns=class_label)
sb.heatmap(df_conf_matrix, annot=True, fmt='d')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()
%%time

neighbors = list(range(20, 80, 4))
cv_score = []
for k in neighbors:
    knn = KNeighborsClassifier(n_neighbors=k, algorithm='kd_tree')
    scores = cross_val_score(knn, X_train, Y_train, cv=10, scoring='accuracy')
    cv_score.append(scores.mean())
MSE = [1 - x for x in cv_score]
optimal_k = neighbors[MSE.index(min(MSE))]
print("_" * 101)
print("Optimal number of neighbors: ", optimal_k)
print("_" * 101)
print("Missclassification error for each k values: ", np.round(MSE, 3))
print("_" * 101)

plt.plot(neighbors, MSE)
#for xy in zip(neighbors, np.round(MSE, 3)):
#    plt.annotate("%s %s" %xy, xy=xy, textcoords='data')
plt.title("Number of neighbors and error")
plt.xlabel("Number of neighbors")
plt.ylabel("Missclassification error")
plt.show()
%%time

optimal_model = KNeighborsClassifier(
    n_neighbors=optimal_k, algorithm='kd_tree')
optimal_model.fit(X_train, Y_train)
prediction = optimal_model.predict(X_test)

training_accuracy = optimal_model.score(X_train, Y_train)
training_error = 1 - training_accuracy
test_accuracy = accuracy_score(Y_test, prediction)
test_error = 1 - test_accuracy

print("_" * 101)
print("Training Accuracy: ", training_accuracy)
print("Train Error: ", training_error)
print("Test Accuracy: ", test_accuracy)
print("Test Error: ", test_error)
print("_" * 101)
print("_" * 101)
print("Classification Report: \n")
print(classification_report(Y_test, prediction))
print("_" * 101)
conf_matrix = confusion_matrix(Y_test, prediction)
class_label = ['negative', 'positive']
df_conf_matrix = pd.DataFrame(
    conf_matrix, index=class_label, columns=class_label)
sb.heatmap(df_conf_matrix, annot=True, fmt='d')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()
print("_" * 101)
sample_points = total_points['Text']
#labels = total_points['Score']
X_train, X_test, Y_train, Y_test = train_test_split(
    sample_points, labels, test_size=0.3, random_state=0)
import re

def cleanhtml(sentence):
    cleantext = re.sub('<.*>', '', sentence)
    return cleantext

def cleanpunc(sentence):
    cleaned = re.sub(r'[?|!|\'|#|@|.|,|)|(|\|/]', r'', sentence)
    return cleaned
train_sent_list = []
for sent in X_train:
    train_sentence = []
    sent = cleanhtml(sent)
    for w in sent.split():
        for cleaned_words in cleanpunc(w).split():
            if (cleaned_words.isalpha()):
                train_sentence.append(cleaned_words.lower())
            else:
                continue
    train_sent_list.append(train_sentence)
test_sent_list = []
for sent in X_test:
    train_sentence = []
    sent = cleanhtml(sent)
    for w in sent.split():
        for cleaned_words in cleanpunc(w).split():
            if (cleaned_words.isalpha()):
                train_sentence.append(cleaned_words.lower())
            else:
                continue
    test_sent_list.append(train_sentence)
train_w2v_model = gensim.models.Word2Vec(
    train_sent_list, min_count=5, size=50, workers=4)
train_w2v_words = train_w2v_model[train_w2v_model.wv.vocab]
test_w2v_model = gensim.models.Word2Vec(
    test_sent_list, min_count=5, size=50, workers=4)
test_w2v_words = test_w2v_model[test_w2v_model.wv.vocab]
print(train_w2v_words.shape, test_w2v_words.shape)
import numpy as np
train_vectors = []
for sent in train_sent_list:
    sent_vec = np.zeros(50)
    cnt_words = 0
    for word in sent:
        try:
            vec = train_w2v_model.wv[word]
            sent_vec += vec
            cnt_words += 1
        except:
            pass
    sent_vec /= cnt_words
    train_vectors.append(sent_vec)
train_vectors = np.nan_to_num(train_vectors)
test_vectors = []
for sent in test_sent_list:
    sent_vec = np.zeros(50)
    cnt_words = 0
    for word in sent:
        try:
            vec = test_w2v_model.wv[word]
            sent_vec += vec
            cnt_words += 1
        except:
            pass
    sent_vec /= cnt_words
    test_vectors.append(sent_vec)
test_vectors = np.nan_to_num(test_vectors)
X_train = train_vectors
X_test = test_vectors
%%time

neighbors = list(range(20, 50, 2))
cv_score = []
for k in neighbors:
    knn = KNeighborsClassifier(n_neighbors=k, algorithm='brute')
    scores = cross_val_score(knn, X_train, Y_train, cv=10, scoring='accuracy')
    cv_score.append(scores.mean())
MSE = [1 - x for x in cv_score]
optimal_k = neighbors[MSE.index(min(MSE))]
print("_" * 101)
print("Optimal number of neighbors: ", optimal_k)
print("_" * 101)
print("Missclassification error for each k values: ", np.round(MSE, 3))
print("_" * 101)

plt.plot(neighbors, MSE)
#for xy in zip(neighbors, np.round(MSE, 3)):
#    plt.annotate("%s %s" %xy, xy=xy, textcoords='data')
plt.title("Number of neighbors and error")
plt.xlabel("Number of neighbors")
plt.ylabel("Missclassification error")
plt.show()
%%time

optimal_model = KNeighborsClassifier(
    n_neighbors=optimal_k, algorithm='kd_tree')
optimal_model.fit(X_train, Y_train)
prediction = optimal_model.predict(X_test)

training_accuracy = optimal_model.score(X_train, Y_train)
training_error = 1 - training_accuracy
test_accuracy = accuracy_score(Y_test, prediction)
test_error = 1 - test_accuracy

print("_" * 101)
print("Training Accuracy: ", training_accuracy)
print("Train Error: ", training_error)
print("Test Accuracy: ", test_accuracy)
print("Test Error: ", test_error)
print("_" * 101)
print("_" * 101)
print("Classification Report: \n")
print(classification_report(Y_test, prediction))
print("_" * 101)
conf_matrix = confusion_matrix(Y_test, prediction)
class_label = ['negative', 'positive']
df_conf_matrix = pd.DataFrame(
    conf_matrix, index=class_label, columns=class_label)
sb.heatmap(df_conf_matrix, annot=True, fmt='d')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()
print("_" * 101)
%%time

neighbors = list(range(20, 50, 4))
cv_score = []
for k in neighbors:
    knn = KNeighborsClassifier(n_neighbors=k, algorithm='kd_tree')
    scores = cross_val_score(knn, X_train, Y_train, cv=10, scoring='accuracy')
    cv_score.append(scores.mean())
MSE = [1 - x for x in cv_score]
optimal_k = neighbors[MSE.index(min(MSE))]
print("_" * 101)
print("Optimal number of neighbors: ", optimal_k)
print("_" * 101)
print("Missclassification error for each k values: ", np.round(MSE, 3))
print("_" * 101)

plt.plot(neighbors, MSE)
#for xy in zip(neighbors, np.round(MSE, 3)):
#    plt.annotate("%s %s" %xy, xy=xy, textcoords='data')
plt.title("Number of neighbors and error")
plt.xlabel("Number of neighbors")
plt.ylabel("Missclassification error")
plt.show()
%%time

optimal_model = KNeighborsClassifier(
    n_neighbors=optimal_k, algorithm='kd_tree')
optimal_model.fit(X_train, Y_train)
prediction = optimal_model.predict(X_test)

training_accuracy = optimal_model.score(X_train, Y_train)
training_error = 1 - training_accuracy
test_accuracy = accuracy_score(Y_test, prediction)
test_error = 1 - test_accuracy

print("_" * 101)
print("Training Accuracy: ", training_accuracy)
print("Train Error: ", training_error)
print("Test Accuracy: ", test_accuracy)
print("Test Error: ", test_error)
print("_" * 101)
print("_" * 101)
print("Classification Report: \n")
print(classification_report(Y_test, prediction))
print("_" * 101)
conf_matrix = confusion_matrix(Y_test, prediction)
class_label = ['negative', 'positive']
df_conf_matrix = pd.DataFrame(
    conf_matrix, index=class_label, columns=class_label)
sb.heatmap(df_conf_matrix, annot=True, fmt='d')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()
print("_" * 101)
X_train, X_test, Y_train, Y_test = train_test_split(
    sample_points, labels, test_size=0.3)
print(X_train.shape, X_test.shape, Y_train.shape, Y_test.shape)
tfidf_vect = TfidfVectorizer(ngram_range=(1, 1))
train_tfidf_w2v = tfidf_vect.fit_transform(X_train)
test_tfidf_w2v = tfidf_vect.transform(X_test)
print(train_tfidf_w2v.shape, test_tfidf_w2v.shape)
%%time

tfidf_feat = tfidf_vect.get_feature_names()
train_tfidf_w2v_vectors = []
row = 0
for sent in train_sent_list:
    sent_vec = np.zeros(50)
    weight_sum = 0
    for word in sent:
        if word in train_w2v_words:
            vec = train_w2v_model.wv[word]
            tf_idf = train_tfidf_w2v[row, tfidf_feat.index(word)]
            sent_vec += (vec * tf_idf)
            weight_sum += tf_idf
    if weight_sum != 0:
        sent_vec /= weight_sum
    train_tfidf_w2v_vectors.append(sent_vec)
    row += 1
%%time

tfidf_feat = tfidf_vect.get_feature_names()
test_tfidf_w2v_vectors = []
row = 0
for sent in test_sent_list:
    sent_vec = np.zeros(50)
    weighted_sum = 0
    for word in sent:
        if word in test_w2v_words:
            vec = test_w2v_model[word]
            tf_idf = test_tfidf_w2v[row, tfidf_feat.index(word)]
            sent_vec += (vec * tf_idf)
            weight_sum += tf_idf
    if weight_sum != 0:
        sent_vec /= weight_sum
    test_tfidf_w2v_vectors.append(sent_vec)
    row += 1
X_train = train_tfidf_w2v_vectors
X_test = test_tfidf_w2v_vectors
%%time

neighbors = list(range(1, 50, 2))
cv_score = []
for k in neighbors:
    knn = KNeighborsClassifier(n_neighbors=k, algorithm='brute')
    scores = cross_val_score(knn, X_train, Y_train, cv=10, scoring='accuracy')
    cv_score.append(scores.mean())
MSE = [1 - x for x in cv_score]
optimal_k = neighbors[MSE.index(min(MSE))]
print("_" * 101)
print("Optimal number of neighbors: ", optimal_k)
print("_" * 101)
print("Missclassification error for each k values: ", np.round(MSE, 3))
print("_" * 101)

plt.plot(neighbors, MSE)
#for xy in zip(neighbors, np.round(MSE, 3)):
#    plt.annotate("%s %s" %xy, xy=xy, textcoords='data')
plt.title("Number of neighbors and error")
plt.xlabel("Number of neighbors")
plt.ylabel("Missclassification error")
plt.show()
%%time

optimal_model = KNeighborsClassifier(
    n_neighbors=optimal_k, algorithm='kd_tree')
optimal_model.fit(X_train, Y_train)
prediction = optimal_model.predict(X_test)

training_accuracy = optimal_model.score(X_train, Y_train)
training_error = 1 - training_accuracy
test_accuracy = accuracy_score(Y_test, prediction)
test_error = 1 - test_accuracy

print("_" * 101)
print("Training Accuracy: ", training_accuracy)
print("Train Error: ", training_error)
print("Test Accuracy: ", test_accuracy)
print("Test Error: ", test_error)
print("_" * 101)
print("_" * 101)
print("Classification Report: \n")
print(classification_report(Y_test, prediction))
print("_" * 101)
conf_matrix = confusion_matrix(Y_test, prediction)
class_label = ['negative', 'positive']
df_conf_matrix = pd.DataFrame(
    conf_matrix, index=class_label, columns=class_label)
sb.heatmap(df_conf_matrix, annot=True, fmt='d')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()
print("_" * 101)
%%time

neighbors = list(range(1, 50, 2))
cv_score = []
for k in neighbors:
    knn = KNeighborsClassifier(n_neighbors=k, algorithm='kd_tree')
    scores = cross_val_score(knn, X_train, Y_train, cv=10, scoring='accuracy')
    cv_score.append(scores.mean())
MSE = [1 - x for x in cv_score]
optimal_k = neighbors[MSE.index(min(MSE))]
print("_" * 101)
print("Optimal number of neighbors: ", optimal_k)
print("_" * 101)
print("Missclassification error for each k values: ", np.round(MSE, 3))
print("_" * 101)

plt.plot(neighbors, MSE)
#for xy in zip(neighbors, np.round(MSE, 3)):
#    plt.annotate("%s %s" %xy, xy=xy, textcoords='data')
plt.title("Number of neighbors and error")
plt.xlabel("Number of neighbors")
plt.ylabel("Missclassification error")
plt.show()
%%time

optimal_model = KNeighborsClassifier(
    n_neighbors=optimal_k, algorithm='kd_tree')
optimal_model.fit(X_train, Y_train)
prediction = optimal_model.predict(X_test)

training_accuracy = optimal_model.score(X_train, Y_train)
training_error = 1 - training_accuracy
test_accuracy = accuracy_score(Y_test, prediction)
test_error = 1 - test_accuracy

print("_" * 101)
print("Training Accuracy: ", training_accuracy)
print("Train Error: ", training_error)
print("Test Accuracy: ", test_accuracy)
print("Test Error: ", test_error)
print("_" * 101)
print("_" * 101)
print("Classification Report: \n")
print(classification_report(Y_test, prediction))
print("_" * 101)
conf_matrix = confusion_matrix(Y_test, prediction)
class_label = ['negative', 'positive']
df_conf_matrix = pd.DataFrame(
    conf_matrix, index=class_label, columns=class_label)
sb.heatmap(df_conf_matrix, annot=True, fmt='d')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()
print("_" * 101)