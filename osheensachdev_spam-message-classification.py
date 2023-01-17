import numpy as np

import pandas as pd

from sklearn.model_selection import train_test_split





def get_data(train_file, test_file = None):

    if test_file == None:

        frame = pd.read_csv(train_file)

        data = frame.values

        np.random.shuffle(data)

        return data

    else:

        train_frame = pd.read_csv(train_file)

        test_frame = pd.read_csv(test_file)



        train_data = train_frame.values

        test_data = test_frame.values

        np.random.shuffle(train_data)

        np.random.shuffle(test_data)



        return train_data, test_data





def get_training_testing_sets(train_file, test_file = None):

    if test_file == None:

        data = get_data(train_file)

        train_data, test_data = train_test_split(data)

    else:



        train_data, test_data = get_data(train_file, test_file)



    X_train = train_data[:, 1:]

    Y_train = train_data[:, :1]

    X_test = test_data[:, 1:]

    Y_test = test_data[:, :1]



    print(X_train.shape, X_test.shape)

    

    return X_train, Y_train, X_test, Y_test









data = get_data('../input/SPAM text message 20170820 - Data.csv')

m = data.shape[0]



print(data[0])







from sklearn.feature_extraction.text import CountVectorizer

from sklearn.preprocessing import LabelEncoder

import string

from nltk.corpus import stopwords

from nltk.stem import WordNetLemmatizer





punctuations = string.punctuation

stopwords = stopwords.words('english')

lemmatizer = WordNetLemmatizer()



for i in range(m):

    data[i][1] = ''.join(j for j in data[i][1] if j not in punctuations)

    data[i][1] = ' '.join(lemmatizer.lemmatize(j.lower()) for j in data[i][1].split() if j not in stopwords)



print(data[0])   





filtered_spam_data = ''

filtered_ham_data = ''

for i in range(data.shape[0]):

    filtered_spam_data +=' '.join(j for j in data[i][1].split() if data[i][0] == 'spam')

    filtered_ham_data +=' '.join(j for j in data[i][1].split() if data[i][0] == 'ham')



 

from wordcloud import WordCloud

from matplotlib import pyplot as plt

%matplotlib inline





x, y = np.ogrid[:300, :300]

mask = (x - 150) ** 2 + (y - 150) ** 2 > 130 ** 2

mask = 255 * mask.astype(int)



wc = WordCloud(max_font_size=40, max_words=200, background_color='white', random_state=1337, mask=mask).generate(filtered_spam_data)

plt.figure(figsize=(10,10))

plt.imshow(wc, interpolation='bilinear')

plt.axis("off")

plt.title("Spam Words", fontsize=20)

plt.show()



wc = WordCloud(max_font_size=40, max_words=200, background_color='white', random_state=1337, mask=mask).generate(filtered_ham_data)

plt.figure(figsize=(10,10))

plt.imshow(wc, interpolation='bilinear')

plt.axis("off")

plt.title("Ham Words", fontsize=20)

plt.show()







from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from sklearn.preprocessing import LabelEncoder







# cv = TfidfVectorizer(ngram_range=(1, 2))

cv = TfidfVectorizer()

X = cv.fit_transform(data[:, 1])

print(X[0])



le = LabelEncoder()

Y = le.fit_transform(data[:, 0])

print(Y[0])





X_test = X[:1500, :]

X_train = X[1500:, :]



Y_test = Y[:1500]

Y_train = Y[1500:]



print('X_train', X_train.shape)

print('X_test', X_test.shape)

print('Y_train', Y_train.shape)

print('Y_test', Y_test.shape)





from sklearn.naive_bayes import BernoulliNB

from sklearn.svm import LinearSVC

from sklearn.tree import DecisionTreeClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score, confusion_matrix

import warnings



with warnings.catch_warnings():

    warnings.simplefilter("ignore")

    results = []

    for clf, name in [(BernoulliNB(),'BernoulliNB'), (LinearSVC(C = 0.1), 'SVC'), (DecisionTreeClassifier(max_depth = 15), 'DecisionTreeClassifier', ), (LogisticRegression(C = 5), 'LogisticRegression') ]:

    #     Y_train.reshape(Y_train.shape[0],)

    #     Y_test.reshape(Y_test.shape[0])

        clf.fit(X_train, Y_train)

        

        predictions = clf.predict(X_train)

        training_accuracy = accuracy_score(predictions, Y_train)



        m_test = X_test.shape[0]

        predictions = clf.predict(X_test)

        accuracy = accuracy_score(predictions, Y_test)

        confusion = confusion_matrix(predictions, Y_test)



        results.append([name, training_accuracy, accuracy, confusion])



for result in results:

    print(result[0], result[1], result[2])

    print(result[3])