import pandas as pd

import numpy as np



import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline



import warnings

warnings.simplefilter('ignore')



from sklearn import datasets

diabetes = datasets.load_diabetes()

df = pd.DataFrame(data=diabetes.data, columns=diabetes.feature_names)

df.head()
X = df['bmi'].values

Y = diabetes.target



plt.scatter(X, Y);

plt.xlabel('Body mass index (BMI)');

plt.ylabel('Disease progression');
import random

random.seed(0)

idx = random.sample(range(len(df)), 5)

x1, y1 = X[idx], Y[idx]

plt.scatter(x1, y1);
def plot_line(w, b):

    x_values = np.linspace(X.min(), X.max(), 100)

    y_values = w*x_values + b

    plt.plot(x_values, y_values, 'r-')
w = 1300

b = 130

plt.scatter(x1, y1);

plot_line(w, b);
random.seed(12)

idx = random.sample(range(len(df)), 5)

x2, y2 = X[idx], Y[idx]

plt.scatter(x1, y1);

plt.scatter(x2, y2);

plot_line(w, b);
w = 1400

b = 140

plt.scatter(x1, y1);

plt.scatter(x2, y2);

plot_line(w, b);
x = np.concatenate([x1, x2])

y = np.concatenate([y1, y2])

y_pred = w*x + b

error = y - y_pred

pd.DataFrame({'x': x, 'y': y, 'y_pred': y_pred, 

              'error': error})
from sklearn.linear_model import LinearRegression

lin_reg = LinearRegression()
x = x.reshape(-1, 1)

lin_reg.fit(x, y)
w = lin_reg.coef_[0]

b = lin_reg.intercept_

w, b
plt.scatter(x, y);

plot_line(w, b);
X = X.reshape(-1, 1)

lin_reg.fit(X, Y)

w = lin_reg.coef_[0]

b = lin_reg.intercept_

plt.scatter(X, Y);

plot_line(w, b);
lin_reg.score(X, Y)
# Create dataset for classification

from sklearn.datasets import make_classification



X, y = make_classification(n_samples=100, n_features=2, 

                           n_redundant=0, n_informative=2, 

                           n_clusters_per_class=2, 

                           random_state=1)
plt.scatter(X[:, 0], X[:, 1], marker='o', c=y,

            s=25, edgecolor='k');
# Train a classifier

from sklearn.linear_model import LogisticRegression

LR_clf = LogisticRegression()
LR_clf.fit(X, y)
print('Accuracy of Logistic regression classifier: {:.2f}'

     .format(LR_clf.score(X, y)))
def plot_decision_boundary(model, X, y):

    x1, x2 = X[:, 0], X[:, 1]

    x1_min, x1_max = x1.min() - 1, x1.max() + 1

    x2_min, x2_max = x2.min() - 1, x2.max() + 1

    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, 0.1),

                         np.arange(x2_min, x2_max, 0.1))



    Z = model.predict(np.c_[xx1.ravel(), xx2.ravel()]).reshape(xx1.shape)



    plt.contourf(xx1, xx2, Z, alpha=0.4)

    plt.scatter(x1, x2, c=y, marker='o',

                s=25, edgecolor='k');
plot_decision_boundary(LR_clf, X, y)
# Split the dataset into testing and validation

from sklearn.model_selection import train_test_split

X_train, X_valid, y_train, y_valid = train_test_split(X, y, random_state=0)
LR_clf = LogisticRegression().fit(X_train, y_train)
print('Accuracy of Logistic regression classifier on training set: {:.2f}'

     .format(LR_clf.score(X_train, y_train)))

print('Accuracy of Logistic regression classifier on test set: {:.2f}'

     .format(LR_clf.score(X_valid, y_valid)))
# First we define a classifier, we do not need to train it

from sklearn.linear_model import LogisticRegression

clf = LogisticRegression()



# cross_val_score method makes k-folds and train the 

# classifier k times and returns k scores for each run

from sklearn.model_selection import cross_val_score

# accuracy is the default scoring metric

print('Cross-validation (accuracy)', cross_val_score(clf, X, y, cv=5))

# use Area Under ROC as scoring metric

print('Cross-validation (AUC)', cross_val_score(clf, X, y, cv=10, scoring = 'roc_auc'))

# use recall as scoring metric

print('Cross-validation (recall)', cross_val_score(clf, X, y, cv=3, scoring = 'recall'))

# use precision as scoring metric

print('Cross-validation (precision)', cross_val_score(clf, X, y, cv=3, scoring = 'precision'))

# use F1-score as scoring metric

print('Cross-validation (F1-score)', cross_val_score(clf, X, y, cv=3, scoring = 'f1'))
# First we create a dataset for demonstration

from sklearn.datasets import make_classification

X1, y1 = make_classification(

    n_samples=300, n_features=2, 

    n_redundant=0, n_informative=2, 

    n_classes=2, n_clusters_per_class=1, 

    class_sep=1, weights=[0.8, 0.2],

    flip_y=0.05, random_state=0 

)



# We fit the PCA transformer and transform our dataset

from sklearn.decomposition import PCA

pca = PCA(n_components=2)

pca.fit(X1)

X_pca = pca.transform(X1)



# Plotting

import matplotlib.pyplot as plt

%matplotlib inline

plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)

plt.title("Before PCA")

plt.axis("equal")

plt.scatter(X1[:, 0], X1[:, 1], marker='o', c=y1,

            s=25, edgecolor='k');

plt.subplot(1, 2, 2)

plt.title("After PCA")

plt.axis("equal")

plt.scatter(X_pca[:, 0], X_pca[:, 1], marker='o', c=y1,

            s=25, edgecolor='k');

plt.tight_layout()
# Create dataset for classification

from sklearn.datasets import make_classification

X2, y2 = make_classification(

    n_samples=400, n_features=2, 

    n_redundant=0, n_informative=2, 

    n_classes=2, n_clusters_per_class=1, 

    class_sep=1, weights=[0.9, 0.1],

    flip_y=0.15, random_state=0 

)



# Split the dataset into testing and validation

from sklearn.model_selection import train_test_split

X_train, X_valid, y_train, y_valid = train_test_split(X2, y2, random_state=0)



# Train a classifier

from sklearn.linear_model import LogisticRegression

LR_clf = LogisticRegression().fit(X_train, y_train)



# Compute confusin matrix

from sklearn.metrics import confusion_matrix

y_predicted = LR_clf.predict(X_valid)

confusion = confusion_matrix(y_valid, y_predicted)

print('Confusion Matrix\n', confusion)
y_proba = LR_clf.predict_proba(X_valid)

y_proba_list = list(zip(y_valid[0:15], y_predicted[0:15], y_proba[0:15, 1]))

print("(Actual class, Predicted class, probability that an observation belongs to class 1):") 

y_proba_list
from sklearn.metrics import precision_recall_curve



y_scores = LR_clf.decision_function(X_valid)

precision, recall, thresholds = precision_recall_curve(y_valid, y_scores)

closest_zero = np.argmin(np.abs(thresholds))

closest_zero_p = precision[closest_zero]

closest_zero_r = recall[closest_zero]



plt.figure(figsize=(6, 6))

plt.xlim([0.0, 1.01])

plt.ylim([0.0, 1.01])

plt.title('Precision-Recall Curve', fontsize=18)

plt.plot(precision, recall, label='Precision-Recall Curve')

plt.plot(closest_zero_p, closest_zero_r, 'o', markersize = 12, fillstyle = 'none', c='r', mew=3)

plt.xlabel('Precision', fontsize=16)

plt.ylabel('Recall', fontsize=16)

plt.legend(loc='lower left', fontsize=12)

plt.axes().set_aspect('equal')

plt.show()
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

accuracy = accuracy_score(y_valid, y_predicted)

precision = precision_score(y_valid, y_predicted)

recall = recall_score(y_valid, y_predicted)

f1 = f1_score(y_valid, y_predicted)

print('Accuracy:', accuracy)

print('Precision:', precision)

print('Recall:', recall)

print('F1:', f1)
from sklearn.metrics import roc_curve, auc



fpr, tpr, _ = roc_curve(y_valid, y_scores)

roc_auc_lr = auc(fpr, tpr)



plt.figure(figsize=(7, 7))

plt.xlim([-0.01, 1.00])

plt.ylim([-0.01, 1.01])

plt.plot(fpr, tpr, lw=3, label='LogRegr ROC curve (area = {:0.2f})'.format(roc_auc_lr))

plt.xlabel('False Positive Rate', fontsize=16)

plt.ylabel('True Positive Rate', fontsize=16)

plt.title('ROC curve', fontsize=16)

plt.legend(loc='lower right', fontsize=12)

plt.plot([0, 1], [0, 1], color='navy', lw=3, linestyle='--')

plt.axes().set_aspect('equal')

plt.show()