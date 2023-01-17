# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt


data_train_file = "../input/fashionmnist/fashion-mnist_train.csv"
data_test_file = "../input/fashionmnist/fashion-mnist_test.csv"

trainSet = pd.read_csv(data_train_file)
testSet = pd.read_csv(data_test_file)
trainSet.head()
trainLabel = trainSet['label']
testLabel=testSet['label']


trainLabel.head()

X_train, X_test, y_train, y_test = trainSet, testSet, trainLabel, testLabel

print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)
# Import, setup, and a utility for int->string class conversion
import matplotlib.pyplot as plt
%matplotlib inline
class_table = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot"
]

def get_label_cls(label):
    """given an int label range [0,9], return the string description of that label"""
    return class_table[label]

get_label_cls(3)
#Let us view some dress in the trainset
for i in range(5000,5005): 
    sample = np.reshape(trainSet[trainSet.columns[1:]].iloc[i].values/255, (28,28))
    plt.figure()
    plt.title("labeled class {}".format(get_label_cls(trainSet["label"].iloc[i])))
    plt.imshow(sample, 'gray')
#Lets apply PCA to reduce the dimensions of the dataset for daster classification

from sklearn.decomposition import PCA

X_train=X_train.drop(['label'],axis=1)
pca = PCA()
pca.fit(X_train)
cumsum = np.cumsum(pca.explained_variance_ratio_)
d = np.argmax(cumsum >= 0.95) + 1
d
pca = PCA(n_components=0.95)
X_reduced_trained = pca.fit_transform(X_train)
pca.n_components_
X_reduced_trained
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_reduced_trained_scaled = scaler.fit_transform(X_reduced_trained.astype(np.float64))
X_reduced_trained_scaled
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import f1_score

from sklearn.linear_model import SGDClassifier

sgd_clf = SGDClassifier(random_state=42) 
sgd_clf.fit(X_reduced_trained_scaled, y_train)
y_train_predict = sgd_clf.predict(X_reduced_trained_scaled[0].reshape(1, -1))
y_train_predict
y_train[0]
# let us rescale back the X_trained sample to see its contents

X_reduced_rescaled=scaler.inverse_transform(X_reduced_trained_scaled[0]) 
X_recovered = pca.inverse_transform(X_reduced_rescaled)


%matplotlib inline
import matplotlib
import matplotlib.pyplot as plt

def showImage(data):
    some_article = data
    some_article_image = some_article.reshape(28, 28) # Reshaping it to get the 28x28 pixels
    plt.imshow(some_article_image, cmap = matplotlib.cm.binary, interpolation="nearest")
    plt.axis("off")
    plt.show()
showImage(X_recovered)
# Let us predict all instances of training dataset X_train_scaled using the above trained model
y_train_predict = sgd_clf.predict(X_reduced_trained_scaled)
y_train_predict
sgd_accuracy = accuracy_score(y_train, y_train_predict)
sgd_precision = precision_score(y_train, y_train_predict, average='weighted')
sgd_recall = recall_score(y_train, y_train_predict, average='weighted')
sgd_f1_score = f1_score(y_train, y_train_predict, average='weighted')


print("SGD Accuracy: ", sgd_accuracy)
print("SGD Precision: ", sgd_precision)
print("SGD Recall: ", sgd_precision)
print("SGD F1 Score: ", sgd_f1_score)
# Let us use Decision tree and Random Forest Classifier

from sklearn.tree import DecisionTreeClassifier

dec_tree_clf = DecisionTreeClassifier(max_depth=50, random_state=42)
dec_tree_clf.fit(X_reduced_trained, y_train)
y_train_predict = dec_tree_clf.predict(X_reduced_trained[0].reshape(1, -1))
y_train_predict
y_train[0]
# Hence we find that the decision tree classifeir has correctly classified, now lets find all its accuracy scores

y_train_predict = dec_tree_clf.predict(X_reduced_trained)
dec_tree_accuracy = accuracy_score(y_train, y_train_predict)
dec_tree_precision = precision_score(y_train, y_train_predict, average='weighted')
dec_tree_recall = recall_score(y_train, y_train_predict, average='weighted')
dec_tree_f1_score = f1_score(y_train, y_train_predict, average='weighted')

print("Decision Tree Accuracy: ", dec_tree_accuracy)
print("Decision Tree Precision: ", dec_tree_precision)
print("Decision Tree Recall: ", dec_tree_precision)
print("Decision Tree F1 Score: ", dec_tree_f1_score)
from sklearn.ensemble import RandomForestClassifier

rnd_clf = RandomForestClassifier(n_estimators=100, max_depth=50, random_state=42)
rnd_clf.fit(X_reduced_trained, y_train)
y_train_predict = rnd_clf.predict(X_reduced_trained)
rnd_accuracy = accuracy_score(y_train, y_train_predict)
rnd_precision = precision_score(y_train, y_train_predict, average='weighted')
rnd_recall = recall_score(y_train, y_train_predict, average='weighted')
rnd_f1_score = f1_score(y_train, y_train_predict, average='weighted')

print("Random Forest Accuracy: ", rnd_accuracy)
print("Random Forest Precision: ", rnd_precision)
print("Random Forest Recall: ", rnd_precision)
print("Random Forest F1 Score: ", rnd_f1_score)
# We see that RandomForestClassifier is giving much better score for the dataset, but since the value is 1 i think its overfitting
#Let us try XGBoost and than cross validation score for all to see which one is really good


from xgboost import XGBClassifier

xgb_clf = XGBClassifier(n_estimators=20, max_depth=10, random_state=42)
xgb_clf.fit(X_reduced_trained, y_train)
y_train_predict = xgb_clf.predict(X_reduced_trained)
xgb_accuracy = accuracy_score(y_train, y_train_predict)
xgb_precision = precision_score(y_train, y_train_predict, average='weighted')
xgb_recall = recall_score(y_train, y_train_predict, average='weighted')
xgb_f1_score = f1_score(y_train, y_train_predict, average='weighted')

print("Random Forest Accuracy: ", xgb_accuracy)
print("Random Forest Precision: ", xgb_precision)
print("Random Forest Recall: ", xgb_recall)
print("Random Forest F1 Score: ", xgb_f1_score)
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score

def display_scores(scores):
    print("Scores:", scores)
    print("Mean:", scores.mean())
    print("Standard deviation:", scores.std())


sgd_scores = cross_val_score(sgd_clf, X_reduced_trained, y_train, cv=3, scoring="accuracy") 
display_scores(sgd_scores)
sgd_accuracy = sgd_scores.mean()

dec_tree_scores = cross_val_score(dec_tree_clf, X_reduced_trained, y_train, cv=3, scoring="accuracy") 
display_scores(dec_tree_scores)
dec_tree_accuracy = dec_tree_scores.mean()

rnd_scores = cross_val_score(rnd_clf, X_reduced_trained, y_train, cv=3, scoring="accuracy") 
display_scores(rnd_scores)
rnd_accuracy = rnd_scores.mean()

xgb_scores = cross_val_score(xgb_clf, X_reduced_trained, y_train, cv=3, scoring="accuracy") 
display_scores(xgb_scores)
xgb_accuracy = xgb_scores.mean()

# Lets draw an confusion Matrix for RandomForestClassifier

y_train_pred = cross_val_predict(rnd_clf, X_reduced_trained, y_train, cv=3)
conf_mx = confusion_matrix(y_train, y_train_pred)
conf_mx
#Finding out the erro rates by dividing each class by number of images in each class

row_sums = conf_mx.sum(axis=1, keepdims=True)
norm_conf_mx = conf_mx / row_sums
np.fill_diagonal(norm_conf_mx, 0)
plt.matshow(norm_conf_mx, cmap=plt.cm.gray)
plt.show()
# lets try Grid Search on XGboost and RandomForect to see which is performing best with what parametrs

# GridSearchCV

from sklearn.model_selection import GridSearchCV

param_grid = [
    # try (1x3)=3 combinations of hyperparameters
    {'n_estimators': [20], 'max_depth': [8, 10, 12]},
    
]

xgb_clf_grid_search = XGBClassifier(random_state=42)
# train across 3 folds, that's a total of 3x3=9 rounds of training 
grid_search = GridSearchCV(xgb_clf_grid_search, param_grid, cv=3,
                           scoring='neg_mean_squared_error')
grid_search.fit(X_reduced_trained, y_train)