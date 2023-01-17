# This kernel is a very simple code to help in getting started with the shark tank pitches and deals dataset.

# I believe much improvement is possible in this and hopefully this code will help you get accustomed to the dataset columns, headers, etc. for writing your own algorithms



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import pandas as pd

data = pd.read_csv('../input/Sharktankpitchesdeals.csv') #text in column 1, classifier in column 2.

import numpy as np

numpy_array = data.as_matrix()

X = numpy_array[:,2] #the description text of the companies

Y = np.asarray(numpy_array[:,3], dtype="|S6") #the decision on the deal 

#Y_train = np.asarray(train['P1'], dtype="|S6")

#Y = np.dtype("|S6")

from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(

 X, Y, test_size=0.4, random_state=42)

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.feature_extraction.text import TfidfTransformer

from sklearn.naive_bayes import MultinomialNB

from sklearn.pipeline import Pipeline

text_clf = Pipeline([('vect', CountVectorizer(stop_words='english')),

 ('tfidf', TfidfTransformer()),

 ('clf', MultinomialNB()),

])

text_clf = text_clf.fit(X_train,Y_train)

predicted = text_clf.predict(X_test)



np.mean(predicted == Y_test)



from sklearn.metrics import accuracy_score





y_true = Y_test

y_pred = predicted



accuracy_score(y_true, y_pred)

print(accuracy_score)



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.