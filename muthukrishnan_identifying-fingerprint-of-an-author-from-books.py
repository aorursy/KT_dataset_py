from collections import Counter
import numpy as np
from os import listdir
import pandas as pd

data = pd.read_csv('../input/famous-authors-books/training_data_books.csv')
data.columns
#my dataset
data
# now that we have the data, lets try to build a classification modal with it
#for my feature set i dont need the author name and book_name so I will remove them
X = data.drop(['author', 'book_name'],axis=1)
X = X.drop(X.columns[0],axis=1)
#my target is the author name, this is what my predictor is supposed to give
y = data['author']

#printing all the unique authors present in my dataset
print(np.unique(y))
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
label_encoder.fit(y.values)
target = label_encoder.transform(y.values)
features = X
from sklearn.linear_model import LogisticRegression

classifier = LogisticRegression()
classifier.fit(features, target)
 
# Predicting the Test set results
y_pred = classifier.predict(features)
from sklearn.metrics import classification_report
print(classification_report(target,y_pred))
classifier.score(features, target)
