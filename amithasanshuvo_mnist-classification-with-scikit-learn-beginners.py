import numpy as np

import pandas as pd


train = pd.read_csv("../input/digit-recognizer/train.csv")

test = pd.read_csv("../input/digit-recognizer/test.csv")
train.shape
test.shape
X_train = (train.iloc[:,1:].values).astype('float32') # all pixel values

y_train = train.iloc[:,0].values.astype('int32') # only labels i.e targets digits

X_test = test.iloc[:,1:].values.astype('float32')

y_test = test.iloc[:,0:].values.astype('int32')
print ("X_train: ", X_train)

print ("y_train: ", y_train)

print("X_test: ", X_test)

print ("y_test: ", y_test)
#X_train = X_train.reshape(X_train.shape[0], 28, 28,1)

#X_train.shape
#X_test = X_test.reshape(X_test.shape[0], 28, 28,1)

#X_test.shape
y_train_5 = (y_train ==5)

y_test_5 = (y_test ==5)



# This means true for all 5's and false for others
# Now we have to pick a classifier and train it.



# Stochastic Gradient Descent (it can handle big datasets every efficiently)



from sklearn.linear_model import SGDClassifier

sgd_clf = SGDClassifier( random_state=42)

sgd_clf.fit(X_train,y_train_5)
# Performance Measure with k fold cross validation with three folds

from sklearn.model_selection import cross_val_score

cross_val_score(sgd_clf, X_train, y_train_5, cv=3, scoring="accuracy")
# A classifier that only images that are in 'not 5' class.



from sklearn.base import BaseEstimator

class Never5Classifier(BaseEstimator):

    def fit(self, X, y=None):

        pass

    def predict(self, X):

        return np.zeros((len(X), 1), dtype=bool)
never_5_clf = Never5Classifier()

cross_val_score(never_5_clf, X_train, y_train_5, cv=3, scoring="accuracy")



# Good accuracy but only 10% images are 5's. So if that image is not a 5, then we are right about 90% time.



from sklearn.model_selection import cross_val_predict

y_train_pred = cross_val_predict(sgd_clf, X_train, y_train_5, cv =3)
from sklearn.metrics import confusion_matrix

confusion_matrix (y_train_5,y_train_pred)


y_train_perfect_predictions = y_train_5  

# pretend we reached perfection

confusion_matrix(y_train_5, y_train_perfect_predictions)
from sklearn.metrics import precision_score, recall_score



precision_score(y_train_5, y_train_pred)
recall_score(y_train_5, y_train_pred)



from sklearn.metrics import f1_score



f1_score(y_train_5, y_train_pred)