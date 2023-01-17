import pandas as pd

df = pd.read_csv('../input/spam.csv', encoding='latin-1')

df.head()
df.shape
# split into train and test

from sklearn import cross_validation

data_train, data_test, labels_train, labels_test = cross_validation.train_test_split(

    df.v2,

    df.v1, 

    test_size=0.1, 

    random_state=42)



print (data_train[:10])



### text vectorization--go from strings to lists of numbers

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.feature_selection import SelectPercentile, f_classif



vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5)

data_train_transformed = vectorizer.fit_transform(data_train)

data_test_transformed  = vectorizer.transform(data_test)



print (data_train_transformed[:10])





# slim the data for training and testing

selector = SelectPercentile(f_classif, percentile=10)

selector.fit(data_train_transformed, labels_train)

data_train_transformed = selector.transform(data_train_transformed).toarray()

data_test_transformed  = selector.transform(data_test_transformed).toarray()



print (data_train_transformed[:10])
from sklearn.naive_bayes import GaussianNB

from sklearn.metrics import accuracy_score



clf = GaussianNB()

clf.fit(data_train_transformed, labels_train)

predictions = clf.predict(data_test_transformed)



print (accuracy_score(labels_test, predictions))