%matplotlib inline

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.model_selection import train_test_split

from sklearn.metrics import classification_report

from sklearn.metrics import accuracy_score

from sklearn.metrics import confusion_matrix

import pylab as pl

import os
for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
print('Total File sizes')

print('-'*10)

for f in os.listdir('../input/Kannada-MNIST'):

    if 'zip' not in f:

        print(f.ljust(30) + str(round(os.path.getsize('../input/Kannada-MNIST/' + f) / 1000000, 2)) + 'MB')
train = pd.read_csv('../input/Kannada-MNIST/train.csv')

test = pd.read_csv('../input/Kannada-MNIST/test.csv')

submission = pd.read_csv('../input/Kannada-MNIST/sample_submission.csv')

val= pd.read_csv('../input/Kannada-MNIST/Dig-MNIST.csv')
test.head()
test.rename(columns={'id':'label'}, inplace=True)

test.head()
train.head()
print('Train Shape: ', train.shape)

print('Test Shape:',test.shape)

print('Submission Shape: ',submission.shape)

print('Validation Shape: ',val.shape)
train.groupby(by='label').size()
X_train, X_test, y_train, y_test = train_test_split(train.iloc[:, 1:], train.iloc[:, 0], test_size=0.2)
X_train.head()
X_test.head()
# Visualization Reference Kernel https://www.kaggle.com/josephvm/kannada-with-pytorch

# Some quick data visualization 

# First 10 images of each class in the training set



fig, ax = plt.subplots(nrows=10, ncols=10, figsize=(10,10))



# I know these for loops look weird, but this way num_i is only computed once for each class

for i in range(10): # Column by column

    num_i = X_train[y_train == i]

    ax[0][i].set_title(i)

    for j in range(10): # Row by row

        ax[j][i].axis('off')

        ax[j][i].imshow(num_i.iloc[j, :].to_numpy().astype(np.uint8).reshape(28, 28), cmap='gray')
# LogisticRegression

from sklearn.linear_model import LogisticRegression

ModelLR = LogisticRegression(C=5, solver='lbfgs', multi_class='multinomial')

ModelLR.fit(X_train, y_train)



y_predLR = ModelLR.predict(X_test)



# Accuracy score

print('accuracy is',accuracy_score(y_predLR,y_test))



score = accuracy_score(y_predLR,y_test)
cm = confusion_matrix(y_test, y_predLR)

print(cm)
plt.figure(figsize=(9,9))

sns.heatmap(cm, annot=True, fmt=".3f", linewidths=.5, square = True, cmap = 'Blues_r');

plt.ylabel('Actual label');

plt.xlabel('Predicted label');

all_sample_title = 'Accuracy Score: {0}'.format(score)

plt.title(all_sample_title, size = 15);
# Seed for reproducability

seed = 1234

np.random.seed(seed)
from sklearn.tree import DecisionTreeClassifier, export_graphviz



DT = DecisionTreeClassifier(max_depth=10, random_state=seed)

DT.fit(X_train, y_train)
y_predDT = DT.predict(X_test)



# Accuracy score

print('accuracy DT',accuracy_score(y_predDT,y_test))



scoreDT= accuracy_score(y_predDT,y_test)
DTm =confusion_matrix(y_test, y_predDT)

print(DTm)
plt.figure(figsize=(9,9))

sns.heatmap(DTm, annot=True, fmt=".3f", linewidths=.5, square = True, cmap = 'Blues_r');

plt.ylabel('Actual label');

plt.xlabel('Predicted label');

all_sample_title = 'Accuracy Score Desicion Tree: {0}'.format(scoreDT)

plt.title(all_sample_title, size = 15);
from sklearn import svm

from sklearn.decomposition import PCA

pca = PCA(n_components=0.7,whiten=True)

X_train_PCA = pca.fit_transform(X_train)

X_test_PCA = pca.transform(X_test)

sv = svm.SVC(kernel='rbf',C=9)

sv.fit(X_train_PCA , y_train)



y_predsv = sv.predict(X_test_PCA)
print('accuracy is',accuracy_score(y_predsv,y_test))



scoreclf= accuracy_score(y_predsv,y_test)
from xgboost import XGBClassifier

# fit model no training data

model = XGBClassifier()

eval_set = [(X_test,y_test)]

model.fit(X_train, y_train, early_stopping_rounds= 5, eval_set=eval_set, verbose=True)

# make predictions for test data

y_pred = model.predict(X_test)

predictions = [round(value) for value in y_pred]
from sklearn.metrics import accuracy_score

# evaluate predictions

accuracy = accuracy_score(y_test, predictions)

print("Accuracy XGBOOST: %.2f%%" % (accuracy * 100.0))
from sklearn.ensemble import AdaBoostClassifier

Model=AdaBoostClassifier()

Model.fit(X_train, y_train)

y_predAda=Model.predict(X_test)



# Summary of the predictions made by the classifier

print(classification_report(y_test,y_predAda))

print(confusion_matrix(y_pred,y_test))

#Accuracy Score

print('accuracy is ',accuracy_score(y_predAda,y_test))



AdaB = accuracy_score(y_predAda,y_test)
models = pd.DataFrame({

    'Model': ['LogisticRegression','Decision Tree', 'PCA', 'XGBOOST', "AdaBoost classifier"

              ],

    'Score': [score,scoreDT,scoreclf,accuracy,AdaB]})

models.sort_values(by='Score', ascending=False)
plt.subplots(figsize =(10, 5))



sns.barplot(x='Score', y = 'Model', data = models, palette="Set3")



#prettify using pyplot: https://matplotlib.org/api/pyplot_api.html

plt.title('Machine Learning Algorithm Accuracy Score \n')

plt.xlabel('Accuracy Score (%)')

plt.ylabel('Algorithm')
test_x = test.values[:,1:]

test_x = pca.transform(test_x)
preds = sv.predict(test_x)

submission['label'] = preds

submission.to_csv('submission.csv', index=False)
submission.head()