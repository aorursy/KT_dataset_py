import warnings

warnings.filterwarnings('ignore')

import sqlite3

import pandas as pd

import numpy as np

from time import time

import nltk

from nltk.corpus import stopwords

import re
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
positive_points = final[final['Score'] == 'positive'].sample(n=30000, random_state=5)

negative_points = final[final['Score'] == 'negative'].sample(n=30000, random_state=5)

total_points = pd.concat([positive_points, negative_points])



total_points['Time'] = pd.to_datetime(total_points['Time'], origin='unix', unit='s')

total_points = total_points.sort_values("Time")



sample_points = total_points['CleanedText']

label = total_points['Score']
from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(sample_points, label, test_size=0.3, random_state=15)

print(X_train.shape, X_test.shape, Y_train.shape, Y_test.shape)
from sklearn.feature_extraction.text import CountVectorizer

from sklearn.preprocessing import StandardScaler

count_vect = CountVectorizer(ngram_range=(1,1))

std_scaler = StandardScaler(with_mean=False)



X_train = count_vect.fit_transform(X_train)

X_test = count_vect.transform(X_test)



X_train = std_scaler.fit_transform(X_train)

X_test = std_scaler.fit_transform(X_test)



print(X_train.shape, X_test.shape)
%%time

from sklearn.naive_bayes import BernoulliNB

from sklearn.cross_validation import cross_val_score

import matplotlib.pyplot as plt

start_time = time()



alpha_list = list(range(0,50,1))

cv_score = []

for alpha in alpha_list:

    bernauliNB = BernoulliNB(alpha=alpha)

    scores = cross_val_score(bernauliNB, X_train, Y_train, cv=10, scoring="accuracy")

    cv_score.append(scores.mean())

    

MSE = [1-x for x in cv_score]

optimal_alpha = alpha_list[MSE.index(min(MSE))]

print("_"*101)

print("Optimal alpha: ",optimal_alpha)

print("_"*101)



plt.plot(alpha_list, MSE)

plt.title("Alpha vs Misclassification error")

plt.xlabel("value of alpha")

plt.ylabel("MSE")

plt.show()



print("Time consumed: %0.3fs."%(time()-start_time))
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report



optimal_model = BernoulliNB(alpha=optimal_alpha)

optimal_model.fit(X_train, Y_train)

prediction = optimal_model.predict(X_test)



train_accuracy = optimal_model.score(X_train, Y_train)

train_error = 1-train_accuracy

test_accuracy = accuracy_score(Y_test, prediction)

test_error = 1-test_accuracy



print("_"*101)

print("Training accuracy: ", train_accuracy)

print("Training error: ", train_error)

print("Test accuracy: ", test_accuracy)

print("Test error: ", test_error)

print("_"*101)
features = count_vect.get_feature_names()

feature_count = optimal_model.feature_count_

class_count = optimal_model.class_count_

pos_points_prob_sort = optimal_model.feature_log_prob_[1, :].argsort()

neg_points_prob_sort = optimal_model.feature_log_prob_[0, :].argsort()

print(feature_count.shape, pos_points_prob_sort.shape, neg_points_prob_sort.shape)
log_prob = optimal_model.feature_log_prob_

feature_prob = pd.DataFrame(log_prob, columns=features).T

#feature_prob.head()
top_positive = feature_prob[1].sort_values(ascending=False)[:10]

top_negative = feature_prob[0].sort_values(ascending=False)[:10]
from IPython.core.display import HTML



def multi_table(table_list):    

    return HTML(

        '<table><tr style="background-color:white;">' + 

        ''.join(['<td>' + table._repr_html_() + '</td>' for table in table_list]) +

        '</tr></table>'

    )
print("_"*101)

print("Top 10 words with feature importance")

print("_"*101)

print("     positive          negative")

display(multi_table([pd.DataFrame(top_positive), pd.DataFrame(top_negative)]))

print("_"*101)
import seaborn as sb



print("Classificaton Report: \n")

print(classification_report(Y_test, prediction))
class_label = ['negative', 'positive']

conf_matrix = confusion_matrix(Y_test, prediction)

conf_matrix_df = pd.DataFrame(

    conf_matrix, index=class_label, columns=class_label)

sb.heatmap(conf_matrix_df, annot=True, fmt='d')

plt.title("Confusion Matrix")

plt.xlabel("Predicted")

plt.ylabel("Actual")

plt.show()



print("_" * 101)
# Splitting into train and test

X_train, X_test, Y_train, Y_test = train_test_split(

    sample_points, label, test_size=0.30)

print(X_train.shape, X_test.shape, Y_train.shape, Y_test.shape)
from sklearn.feature_extraction.text import TfidfVectorizer



# Initializing tfidf vectorizer

tfidf_vect = TfidfVectorizer(ngram_range=(1, 1))



# Fitting for tfidf vectorization

X_train = tfidf_vect.fit_transform(X_train)

X_test = tfidf_vect.transform(X_test)



# Initializing Standard Scaler

std_scaler = StandardScaler(with_mean=False)



# Standardizing the data

X_train = std_scaler.fit_transform(X_train)

X_test = std_scaler.fit_transform(X_test)



print(X_train.shape, X_test.shape, Y_train.shape, Y_test.shape)
from sklearn.naive_bayes import BernoulliNB

from sklearn.cross_validation import cross_val_score

import matplotlib.pyplot as plt

start_time = time()



alpha_list = list(range(0,50,1))

cv_score = []

for alpha in alpha_list:

    bernauliNB = BernoulliNB(alpha=alpha)

    scores = cross_val_score(bernauliNB, X_train, Y_train, cv=10, scoring="accuracy")

    cv_score.append(scores.mean())

    

MSE = [1-x for x in cv_score]

optimal_alpha = alpha_list[MSE.index(min(MSE))]

print("_"*101)

print("Optimal alpha: ",optimal_alpha)

print("_"*101)



plt.plot(alpha_list, MSE)

plt.title("Alpha vs Misclassification error")

plt.xlabel("value of alpha")

plt.ylabel("MSE")

plt.show()



print("Time consumed: %0.3fs."%(time()-start_time))
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report



optimal_model = BernoulliNB(alpha=optimal_alpha)

optimal_model.fit(X_train, Y_train)

prediction = optimal_model.predict(X_test)



train_accuracy = optimal_model.score(X_train, Y_train)

train_error = 1-train_accuracy

test_accuracy = accuracy_score(Y_test, prediction)

test_error = 1-test_accuracy



print("_"*101)

print("Training accuracy: ", train_accuracy)

print("Training error: ", train_error)

print("Test accuracy: ", test_accuracy)

print("Test error: ", test_error)

print("_"*101)
features = tfidf_vect.get_feature_names()

feature_count = optimal_model.feature_count_



log_prob = optimal_model.feature_log_prob_

feature_prob = pd.DataFrame(log_prob, columns=features).T

#feature_prob.head()





top_positive = feature_prob[1].sort_values(ascending=False)[:10]

top_negative = feature_prob[0].sort_values(ascending=False)[:10]



print("_"*101)

print("Top 10 words with feature importance")

print("_"*101)

print("     positive          negative")

display(multi_table([pd.DataFrame(top_positive), pd.DataFrame(top_negative)]))

print("_"*101)
import seaborn as sb



print("_"*101)

print("Classificaton Report: \n")

print(classification_report(Y_test, prediction))

print("_"*101)




class_label = ['negative', 'positive']

conf_matrix = confusion_matrix(Y_test, prediction)

conf_matrix_df = pd.DataFrame(conf_matrix, index=class_label, columns=class_label)

sb.heatmap(conf_matrix_df, annot=True, fmt='d')

plt.title("Confusion Matrix")

plt.xlabel("Predicted")

plt.ylabel("Actual")

plt.show()



print("_"*101)
sample_points = total_points['Text']

#labels = total_points['Score']

X_train, X_test, Y_train, Y_test = train_test_split(

    sample_points, label, test_size=0.3, random_state=0)
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
import gensim

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
from sklearn.naive_bayes import BernoulliNB

from sklearn.cross_validation import cross_val_score

import matplotlib.pyplot as plt

start_time = time()



alpha_list = list(range(0,5))

cv_score = []

for alpha in alpha_list:

    bernauliNB = BernoulliNB(alpha=alpha)

    scores = cross_val_score(bernauliNB, X_train, Y_train, cv=10, scoring="accuracy")

    cv_score.append(scores.mean())

    

MSE = [1-x for x in cv_score]

optimal_alpha = alpha_list[MSE.index(min(MSE))]

print("_"*101)

print("Optimal alpha: ",optimal_alpha)

print("_"*101)



plt.plot(alpha_list, MSE)

plt.title("Alpha vs Misclassification error")

plt.xlabel("value of alpha")

plt.ylabel("MSE")

plt.show()



print("Time consumed: %0.3fs."%(time()-start_time))
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report



optimal_model = BernoulliNB(alpha=optimal_alpha)

optimal_model.fit(X_train, Y_train)

prediction = optimal_model.predict(X_test)



train_accuracy = optimal_model.score(X_train, Y_train)

train_error = 1-train_accuracy

test_accuracy = accuracy_score(Y_test, prediction)

test_error = 1-test_accuracy



print("_"*101)

print("Training accuracy: ", train_accuracy)

print("Training error: ", train_error)

print("Test accuracy: ", test_accuracy)

print("Test error: ", test_error)

print("_"*101)
import seaborn as sb



print("_"*101)

print("Classificaton Report: \n")

print(classification_report(Y_test, prediction))

print("_"*101)
class_label = ['negative', 'positive']

conf_matrix = confusion_matrix(Y_test, prediction)

conf_matrix_df = pd.DataFrame(conf_matrix, index=class_label, columns=class_label)

sb.heatmap(conf_matrix_df, annot=True, fmt='d')

plt.title("Confusion Matrix")

plt.xlabel("Predicted")

plt.ylabel("Actual")

plt.show()



print("_"*101)
X_train, X_test, Y_train, Y_test = train_test_split(

    sample_points, label, test_size=0.3)

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
from sklearn.naive_bayes import BernoulliNB

from sklearn.cross_validation import cross_val_score

import matplotlib.pyplot as plt

start_time = time()



alpha_list = list(range(0,50,1))

cv_score = []

for alpha in alpha_list:

    bernauliNB = BernoulliNB(alpha=alpha)

    scores = cross_val_score(bernauliNB, X_train, Y_train, cv=10, scoring="accuracy")

    cv_score.append(scores.mean())

    

MSE = [1-x for x in cv_score]

optimal_alpha = alpha_list[MSE.index(min(MSE))]

print("_"*101)

print("Optimal alpha: ",optimal_alpha)

print("_"*101)



plt.plot(alpha_list, MSE)

plt.title("Alpha vs Misclassification error")

plt.xlabel("value of alpha")

plt.ylabel("MSE")

plt.show()



print("Time consumed: %0.3fs."%(time()-start_time))
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report



optimal_model = BernoulliNB(alpha=optimal_alpha)

optimal_model.fit(X_train, Y_train)

prediction = optimal_model.predict(X_test)



train_accuracy = optimal_model.score(X_train, Y_train)

train_error = 1-train_accuracy

test_accuracy = accuracy_score(Y_test, prediction)

test_error = 1-test_accuracy



print("_"*101)

print("Training accuracy: ", train_accuracy)

print("Training error: ", train_error)

print("Test accuracy: ", test_accuracy)

print("Test error: ", test_error)

print("_"*101)
import seaborn as sb



print("_"*101)

print("Classificaton Report: \n")

print(classification_report(Y_test, prediction))

print("_"*101)




class_label = ['negative', 'positive']

conf_matrix = confusion_matrix(Y_test, prediction)

conf_matrix_df = pd.DataFrame(conf_matrix, index=class_label, columns=class_label)

sb.heatmap(conf_matrix_df, annot=True, fmt='d')

plt.title("Confusion Matrix")

plt.xlabel("Predicted")

plt.ylabel("Actual")

plt.show()



print("_"*101)