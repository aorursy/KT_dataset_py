import matplotlib.pyplot as plt

from matplotlib import colors

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

data = pd.read_csv('/kaggle/input/sms-spam-collection-dataset/spam.csv', usecols=(0,1),

                   encoding='latin-1', names=["Label","Text"], skiprows = 1)

data.head()
data["Label"].value_counts()
from sklearn.model_selection import train_test_split



# Create a training and testing dataset.

train, test = train_test_split(data,shuffle=True, test_size=0.2, random_state=42)
# Compare ratio of SPAM instances to HAM instances

print('Percentage of Instances which are SPAM:')

print('Train: ',round(100.*len(train.loc[train["Label"]=="spam"])/len(train),2),'%')

print('Test: ',round(100.*len(test.loc[test["Label"]=="spam"])/len(test),2),'%')
from sklearn.feature_extraction.text import CountVectorizer



# First replace the label with a numeric value

mapping = {'spam': 1, 'ham': 0}

train_labels = train.replace({"Label": mapping})

train_labels = train_labels["Label"]

test_labels = test.replace({"Label": mapping})

test_labels = test_labels["Label"]



# Define a method to vectorize the text data, and fit it using the 

# training dataset.

vectorizer = CountVectorizer()

vectorizer.fit(train["Text"])



# Now convert all sms data to vector form

train_text = vectorizer.transform(train["Text"])

test_text = vectorizer.transform(test["Text"])
from sklearn.svm import SVC

from sklearn.model_selection import cross_val_score



linear_classifier = SVC(kernel='linear',random_state = 42)

linear_scores = cross_val_score(linear_classifier,train_text,train_labels,cv=5)



poly_classifier = SVC(kernel='poly',random_state = 42)

poly_scores = cross_val_score(poly_classifier,train_text,train_labels,cv=5)



sigmoid_classifier = SVC(kernel='sigmoid',random_state = 42)

sigmoid_scores = cross_val_score(sigmoid_classifier,train_text,train_labels,cv=5)



rbf_classifier = SVC(kernel='rbf',random_state = 42)

rbf_scores = cross_val_score(rbf_classifier,train_text,train_labels,cv=5)



print('Kernel\tMean Score')

print('Linear: ',round(100*np.mean(linear_scores),2),'%')

print('Polynomial: ',round(100*np.mean(poly_scores),2),'%')

print('Sigmoid: ',round(100*np.mean(sigmoid_scores),2),'%')

print('RBF: ',round(100*np.mean(rbf_scores),2),'%')
from sklearn.model_selection import GridSearchCV



classifier = SVC(random_state = 26)

param_grid = [{'kernel': ['linear','rbf'], 'C':[0.5,0.75,1.0,1.5,2.0], 'gamma': ['auto','scale']}]



grid_search = GridSearchCV(classifier, param_grid, cv=5, scoring="accuracy", return_train_score=True)

grid_search.fit(train_text, train_labels)



curves = grid_search.cv_results_

print(f'Highest Score: ', round(100.*max(curves["mean_test_score"]),2), '%')

print(f'Corresponding Parameters: ', curves["params"][np.argmax(curves["mean_test_score"])])
from sklearn.metrics import roc_curve

from sklearn.metrics import roc_auc_score

from sklearn.model_selection import cross_val_predict



classifier = SVC(random_state=34, kernel='linear', probability=True)

spam_prob = cross_val_predict(classifier, train_text, train_labels, cv=3, method="predict_proba" )

spam_score = spam_prob[:,1] # Probability text is spam

fpr,tpr, thresholds = roc_curve(train_labels,spam_score)



fig = plt.figure(figsize=(6,6))

ax = fig.add_subplot(111)

ax.plot([0,1],[0,1], color='black', ls='dashed', label='Random Baseline')

ax.plot(fpr, tpr, color='mediumvioletred', label='Linear SVM')

ax.set_xlabel('False Positive Rate',fontsize=12); ax.set_ylabel("True Positive Rate",fontsize=12)

ax.set_xlim(0,1); ax.set_ylim(0,1)

plt.legend(frameon=False)

plt.show()



print('AUC Score: ',round(roc_auc_score(train_labels,spam_score),5))
from sklearn.metrics import classification_report



spam_pred = cross_val_predict(classifier, train_text, train_labels, cv=3, method="predict" )

print(classification_report(train_labels,spam_pred))
# Evaluate model on test data

classifier.fit(train_text, train_labels)

predictions = classifier.predict(test_text)

correct = test_labels==predictions



print('Accuracy: ', round(100.*np.sum(correct)/len(correct),2),'%')
print(classification_report(test_labels,predictions))