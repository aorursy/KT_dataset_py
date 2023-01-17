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

import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline



diabetes = pd.read_csv("/kaggle/input/pima-indians-diabetes-database/diabetes.csv")

print(diabetes.columns)
diabetes.head(5)
print({"dimension of diabetes data: {}".format(diabetes.shape)})
print(diabetes.groupby('Outcome').size())
import seaborn as sns



sns.countplot(diabetes['Outcome'], label='Count');
diabetes.info()
from sklearn.model_selection import train_test_split



X = diabetes.loc[:, diabetes.columns != 'Outcome']

y = diabetes['Outcome']



X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=diabetes['Outcome'], random_state=66)



from sklearn.neighbors import KNeighborsClassifier



training_accuracy = []

test_accuracy = []



# try neighbors from 1 to 11 (similar to GridSearch)

neighbors_settings = range(1, 11)



for n_neighbors in neighbors_settings:

    # instantiate the model with the right hyperparamenter

    knn = KNeighborsClassifier(n_neighbors=n_neighbors)

    

    # fit model against the data

    knn.fit(X_train, y_train)

    

    # append accuracy

    training_accuracy.append(knn.score(X_train, y_train))

    

    # record test accuracy

    test_accuracy.append(knn.score(X_test, y_test))

    

plt.plot(neighbors_settings, training_accuracy, label="trraining accuracy")

plt.plot(neighbors_settings, test_accuracy, label="test accuracy")

plt.ylabel("Accuracy")

plt.xlabel("n_neighbors")

plt.legend();

# plot suggest a n_neighbors around 9 would be the best



knn = KNeighborsClassifier(n_neighbors=9)

knn.fit(X_train, y_train)



print('Accuracy of the K-NN classifier on training set: {:.2f}'.format(knn.score(X_train, y_train)))

print('Accuracy of the K-NN classifier on test set: {:.2f}'.format(knn.score(X_test, y_test)))
# Lets try logistic regression

from sklearn.linear_model import LogisticRegression



logreg = LogisticRegression().fit(X_train, y_train)

print("Training set score: {:.3f}".format(logreg.score(X_train, y_train)))

print("Tets set  score: {:.3f}".format(logreg.score(X_test, y_test)))
# try a C different to the default one, 1

logreg001 = LogisticRegression(C=0.01).fit(X_train,y_train)



print("Training set accuracy: {:.3f}".format(logreg.score(X_train, y_train)))

print("Test set accuracy: {:.3f}".format(logreg.score(X_test, y_test)))



# !!!Note!!! - Try to scale data as suggested in the warnings
# try a C  different to the default one, 1

logreg100 = LogisticRegression(C=100).fit(X_train,y_train)



print("Training set accuracy: {:.3f}".format(logreg.score(X_train, y_train)))

print("Test set accuracy: {:.3f}".format(logreg.score(X_test, y_test)))



# !!!Note!!! - Try to scale data as suggested in the warnings
diabetes_features = [x for i, x in enumerate(diabetes.columns) if i!=len(diabetes)] # create array with features



plt.figure(figsize=(8,6))

plt.plot(logreg.coef_.T, 'o', label="C=1")

plt.plot(logreg100.coef_.T, 'o', label="C=100")

plt.plot(logreg001.coef_.T, 'o', label="C=0.01")

plt.xticks(range(diabetes.shape[1]), diabetes_features, rotation=90)

plt.hlines(0,0, diabetes.shape[1])

plt.ylim(-5, 5)

plt.xlabel("Feature")

plt.ylabel("Coefficient magnitude")

plt.legend();
# lets try with a DecisionTreeClassifier now

from sklearn.tree import DecisionTreeClassifier



tree = DecisionTreeClassifier(random_state=0)

tree.fit(X_train, y_train)

print("Accuracy on training set: {:.3f}".format(tree.score(X_train, y_train)))

print("Accuracy on test set: {:.3f}".format(tree.score(X_test, y_test)))
# Accuracy on training set means Overfitting, lets try to decrease the depth

tree3 = DecisionTreeClassifier(max_depth=3, random_state=0)

tree3.fit(X_train, y_train)



print("Accuracy on training set: {:.3f}".format(tree3.score(X_train, y_train)))

print("Accuracy on test set: {:.3f}".format(tree3.score(X_test, y_test)))
# Lets have a look at feature importance

print("Feature importances:\n{}".format(tree3.feature_importances_))

def plot_feature_importances_diabetes(model):

    plt.figure(figsize=(8,6))

    n_features = 8

    plt.barh(range(n_features), model.feature_importances_, align='center')

    plt.yticks(np.arange(n_features), diabetes_features)

    plt.xlabel("Feature importance")

    plt.ylabel("Feature")

    plt.ylim(-1, n_features)

    

plot_feature_importances_diabetes(tree3)

# lets give it a shot with a random forest as well

from sklearn.ensemble import RandomForestClassifier



rf = RandomForestClassifier(n_estimators=100, random_state=0)

rf.fit(X_train, y_train)



print("Accuracy on training set: {:.3f}".format(rf.score(X_train, y_train)))

print("Accuracy on test set: {:.3f}".format(rf.score(X_test, y_test)))
# feature importance in random forests

plot_feature_importances_diabetes(rf)
# lets try the gradine boosting now

from sklearn.ensemble import GradientBoostingClassifier



gb = GradientBoostingClassifier(random_state=0)

gb.fit(X_train, y_train)



print("Accuracy on training set: {:.3f}".format(gb.score(X_train, y_train)))

print("Accuracy on test set: {:.3f}".format(gb.score(X_test, y_test)))
# 0.917 feels like overfitting,we should apply pre.prunning by limiting rhe maximum depth or lower the learning rate

gb1 = GradientBoostingClassifier(random_state=0, max_depth=1)

gb1.fit(X_train, y_train)



print("Accuracy on training set: {:.3f}".format(gb1.score(X_train, y_train)))

print("Accuracy on test set: {:.3f}".format(gb1.score(X_test, y_test)))
# lets tune a bit more

gb2 = GradientBoostingClassifier(random_state=0,  learning_rate=0.01)

gb2.fit(X_train, y_train)



print("Accuracy on training set: {:.3f}".format(gb2.score(X_train, y_train)))

print("Accuracy on test set: {:.3f}".format(gb2.score(X_test, y_test)))
# acc is decreasing, we will settle for gb1, lets see feature importance

plot_feature_importances_diabetes(gb1)

# lets try support vector machines now

from sklearn.svm import SVC



svc = SVC()

svc.fit(X_train, y_train)



print("Accuracy on training set: {:.2f}".format(svc.score(X_train, y_train)))

print("Accuracy on test set: {:.2f}".format(svc.score(X_test, y_test)))
# lets try a scaller

from sklearn.preprocessing import MinMaxScaler



scaler = MinMaxScaler()

X_train_scaled = scaler.fit_transform(X_train)

X_test_scaled = scaler.fit_transform(X_test)



svc = SVC()

svc.fit(X_train_scaled, y_train)



print("Accuracy on training set: {:.2f}".format(svc.score(X_train_scaled, y_train)))

print("Accuracy on test set: {:.2f}".format(svc.score(X_test_scaled, y_test)))
# scaling yields better results

# let's try some hyperparameter tuning tweaking C or gamma



svc = SVC(C=1000)

svc.fit(X_train_scaled, y_train)



print("Accuracy on training set: {:.3f}".format(

    svc.score(X_train_scaled, y_train)))

print("Accuracy on test set: {:.3f}".format(svc.score(X_test_scaled, y_test)))
from sklearn.neural_network import MLPClassifier # Multilayer perceptrons (MLP)



mlp = MLPClassifier(random_state=42)

mlp.fit(X_train, y_train)



print("Accuracy on training set: {:.2f}".format(mlp.score(X_train, y_train)))

print("Accuracy on test set: {:.2f}".format(mlp.score(X_test, y_test)))
# deep learning algorithms also expect all input features to vary in a similar way, and ideally to have a mean of 0, and a variance of 1. lets rescale

from sklearn.preprocessing import StandardScaler



scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)

X_test_scaled = scaler.fit_transform(X_test)



mlp = MLPClassifier(random_state=0)

mlp.fit(X_train_scaled, y_train)



print("Accuracy on training set: {:.3f}".format(mlp.score(X_train_scaled, y_train)))

print("Accuracy on test set: {:.3f}".format(mlp.score(X_test_scaled, y_test)))
# not converged, increase number of iterations

mlp1000 = MLPClassifier(max_iter=1000, random_state=0)

mlp1000.fit(X_train_scaled, y_train)



print("Accuracy on training set: {:.3f}".format(mlp1000.score(X_train_scaled, y_train)))

print("Accuracy on test set: {:.3f}".format(mlp1000.score(X_test_scaled, y_test)))
# test set performance decreased, most likely we are overfitting



# try increase the alpha parameter and add stronger regularization to the weights

mlp2 = MLPClassifier(max_iter=1000, alpha=1, random_state=0)

mlp2.fit(X_train_scaled, y_train)



print("Accuracy on training set: {:.3f}".format(mlp2.score(X_train_scaled, y_train)))

print("Accuracy on test set: {:.3f}".format(mlp2.score(X_test_scaled, y_test)))

# model is good but is not an increase in performance

# lets try to see feature importance, plot a heat map of the first layer weights in a nural network



plt.figure(figsize=(20, 5))

plt.imshow(mlp.coefs_[0], interpolation="none", cmap='viridis')

plt.yticks(range(8), diabetes_features)

plt.xlabel("Columns in weight matrix")

plt.ylabel("Input feature")

plt.colorbar();