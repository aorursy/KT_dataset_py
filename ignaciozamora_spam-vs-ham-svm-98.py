import pandas as pd



data = pd.read_csv('/kaggle/input/sms-spam-collection-dataset/spam.csv', encoding='latin-1')

y = data.v1

X = data.v2
from sklearn.feature_extraction.text import CountVectorizer



cv = CountVectorizer(binary=True)

cv.fit(X)

X = cv.transform(X)
from sklearn import svm

from sklearn.model_selection import cross_val_score



model = svm.SVC(kernel='linear')

print(cross_val_score(model, X, y, cv=5))