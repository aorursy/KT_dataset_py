import pandas as pd

import matplotlib

import matplotlib.pyplot as plt

import numpy as np

%matplotlib inline
mnist_train = pd.read_csv("/kaggle/input/digit-recognizer/train.csv")

mnist_test = pd.read_csv("/kaggle/input/digit-recognizer/test.csv")                          
print("Train: \n",mnist_train.shape)

print("Test: \n",mnist_test.shape)
mnist_train.head()
mnist_train.label.value_counts().plot.bar()
some_digit = mnist_train.drop("label", axis = 1).iloc[40000]

some_digit_image = some_digit.values.reshape(28,28)



plt.imshow(some_digit_image, cmap = matplotlib.cm.binary, interpolation = "nearest")
# The image looks like a 2, let's verify if the label indeed returns a 2

print("Label: ",mnist_train.iloc[40000].label)
from sklearn.model_selection import train_test_split



# X_train, X_test, y_train, y_test = train_test_split(mnist_train.drop("label", axis = 1), mnist_train.label, test_size=0.2, random_state=42)

X_train = mnist_train.drop("label", axis = 1)

y_train = mnist_train.label



y_train_5  = (y_train == 5)

y_test_5 = (y_train == 5)
# Train using Stochastic Gradient Descent (SGD)

from sklearn.linear_model import SGDClassifier



sgd_clf = SGDClassifier(random_state = 42)

sgd_clf.fit(X_train, y_train_5)
# Let's try this classifier on the previous digit example:



sgd_clf.predict([some_digit])
# The classifier correctly predict the digit 2 as not-5, but let's see if it can correctly predict a 5:

a_digit_5 = mnist_train[mnist_train['label'] == 5].drop('label', axis = 1).sample(1, random_state = 123)



sgd_clf.predict(a_digit_5)
from sklearn.model_selection import cross_val_score



cv_scores = cross_val_score(sgd_clf, X_train, y_train_5, cv=3, scoring = "accuracy")
cv_scores.mean()
from sklearn.base import BaseEstimator



class Never5Classifier(BaseEstimator):

    def fit(self, X, y=None):

        pass

    def predict(self, X):

        return np.zeros((len(X), 1), dtype = bool)

    

never_5_clf = Never5Classifier()

cross_val_score(never_5_clf, X_train, y_train_5, cv = 3, scoring = "accuracy").mean()
from sklearn.model_selection import cross_val_predict



y_train_pred = cross_val_predict(sgd_clf, X_train, y_train_5, cv = 3)
from sklearn.metrics import confusion_matrix



confusion_matrix(y_train_5, y_train_pred)
from sklearn.metrics import precision_score, recall_score

print("precision_score: ", precision_score(y_train_5, y_train_pred))

print("recall_score: ", recall_score(y_train_5, y_train_pred))
from sklearn.metrics import f1_score



f1_score(y_train_5, y_train_pred)
y_scores = cross_val_predict(sgd_clf, X_train, y_train_5, cv = 3, method = 'decision_function')



from sklearn.metrics import roc_curve, roc_auc_score



fpr, tpr, thresholds = roc_curve(y_train_5, y_scores)

roc_auc = roc_auc_score(y_train_5, y_scores)

def plot_roc_curve(fpr, tpr, label = None):

    plt.plot(fpr, tpr, label = label)

    plt.plot([0,1], [0,1], 'k--')

    plt.axis([0,1,0,1])

    plt.xlabel("False Positive Rate")

    plt.ylabel("True Positive Rate")

    plt.legend(loc="lower right")



plot_roc_curve(fpr,tpr, label='ROC curve (area = {})'.format(roc_auc))
from sklearn.ensemble import RandomForestClassifier



forest_clf = RandomForestClassifier(random_state = 42)



y_probas_forest = cross_val_predict(forest_clf, X_train, y_train_5, cv = 3, method = 'predict_proba')



y_scores_forest = y_probas_forest[:,1]

fpr_forest, tpr_forest, thresholds_forest = roc_curve(y_train_5, y_scores_forest)

roc_auc_forest = roc_auc_score(y_train_5, y_scores_forest)
plot_roc_curve(fpr,tpr, label='ROC curve (area = {})'.format(roc_auc))

plt.plot(fpr_forest, tpr_forest, "b:", label = "Random Forest (area = {})".format(roc_auc_forest))

plt.legend(loc = 'lower right')
y_probas_forest_binary = cross_val_predict(forest_clf, X_train, y_train_5, cv = 3, method = 'predict')



print("Random Forest F1:", f1_score(y_train_5, y_probas_forest_binary))

print("SGD F1:", f1_score(y_train_5, y_train_pred))
forest_clf.fit(X_train, y_train)

forest_clf.predict(a_digit_5)
forest_clf.predict_proba(a_digit_5)
cross_val_score(forest_clf, X_train, y_train, cv = 3, scoring = "accuracy").mean()
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()



X_train_scaled = scaler.fit_transform(X_train.astype(np.float64))

cross_val_score(forest_clf, X_train_scaled, y_train, cv = 3, scoring = "accuracy").mean()
from sklearn.neighbors import KNeighborsClassifier



KNN_clf = KNeighborsClassifier(n_jobs = -1)

KNN_clf.fit(X_train, y_train)
cross_val_score(KNN_clf, X_train, y_train, cv = 3, scoring = "accuracy").mean()
# from sklearn.model_selection import GridSearchCV



# param_grid = {

#     "n_neighbors" : [3,4,5,6,7,8],

#     "weights" : ['uniform', 'distance']

# }



# KNN_clf = KNeighborsClassifier(n_jobs = -1)



# grid_search = GridSearchCV(KNN_clf, param_grid , cv = 5, scoring = "accuracy")



# grid_search.fit(X_train, y_train)



# grid_search.best_params_
KNN_clf = KNeighborsClassifier(n_jobs = -1, n_neighbors = 4, weights = "distance")

KNN_clf.fit(X_train, y_train)

cross_val_score(KNN_clf, X_train, y_train, cv = 3, scoring = "accuracy").mean()
sample_submission = pd.read_csv("/kaggle/input/digit-recognizer/sample_submission.csv")

test_predictions = KNN_clf.predict(mnist_test)

submission = sample_submission.drop('Label', axis = 1)

submission['Label'] = test_predictions
submission.to_csv("mnist_submission.csv",index=False)