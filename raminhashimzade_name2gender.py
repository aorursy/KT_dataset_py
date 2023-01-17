import pandas as pd

# load data

#dataset = pd.read_csv("../input/gender_testdata.csv")

dataset = pd.read_excel("../input/gender/gender.xlsx")
dataset["NAME"] = dataset["FIRST_NAME"] + " " + dataset["LAST_NAME"]

dataset.groupby('SEX')['FIRST_NAME'].count()
dataset.head()
# Dropping last 20K rows as we have limitation in memory in this kernel. In real case this line code have to be commented

dataset.drop(dataset.tail(20000).index,inplace=True)
from sklearn.feature_extraction.text import CountVectorizer

cv = CountVectorizer()

X = cv.fit_transform(dataset["NAME"].values.astype('U')).toarray()

y = dataset.iloc[:, 2].values
# split to train and test

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 1)
print ('Train set:', X_train.shape,  y_train.shape)

print ('Test set:', X_test.shape,  y_test.shape)
from sklearn.naive_bayes import GaussianNB

classifier = GaussianNB()

classifier.fit(X_train, y_train)
y_pred= classifier.predict(X_test)
########## Making the Confusion Matrix

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)

print(cm)

print('Accurancy: {:.0f}%'.format(classifier.score(X_test, y_test)*100))