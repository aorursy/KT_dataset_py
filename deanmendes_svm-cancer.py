# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import seaborn as sns

import matplotlib.pyplot as plt



%matplotlib inline
# csv url of cell samples

csv = '/kaggle/input/cell_samples.csv'

# create dataframe using pandas

Data = pd.read_csv(csv)

Data.head()
# view data types, rows, columns etc

Data.info()
Data['BareNuc'].value_counts().sort_values()
# a question mark is evdident in the feature, first replace it with 'np.nan' then replace it with the mean of the feature

Data['BareNuc'].replace('?', np.nan, inplace = True)



# mean of 'BareNuc'

avg_bare = Data['BareNuc'].astype('float').mean()

# replacing np.nan values the mean

Data['BareNuc'].replace(np.nan, avg_bare, inplace = True)



# convert 'BareNuc' to an 'int' type

Data['BareNuc'] = Data['BareNuc'].astype('int')

Data['BareNuc'].dtype
Data['Class'].value_counts().plot(kind='bar')

plt.xlabel('Class of tumor')

plt.ylabel('Counts')

plt.title('Benign(2) vs Malignant(4)')

plt.show()
# visualize distribution of the classes (malignant and benign) based on Clump thickness and Uniformity

malignant = Data['Class'] == 4

benign = Data['Class'] == 2



ax1 = Data[malignant][0:50].plot(kind='scatter',

                                  x='Clump', y='UnifSize', 

                                  color='r', label='Malignant');

Data[benign][0:50].plot(kind='scatter', 

                         x='Clump', y='UnifSize', 

                         color='blue', label='Benign', ax=ax1);

plt.show()
sns.pairplot(Data, hue='Class')
# distrubution plots to test whether features should be scaled

ax1 = sns.distplot(Data['Clump'], hist=False, label='clump')

sns.distplot(Data['UnifSize'], hist=False, label='unifsize',ax=ax1)

sns.distplot(Data['BlandChrom'], hist=False, label='BlandChrom', ax=ax1)
# features

X = Data.drop(columns=['Class', 'ID']).values

print(X[:5])



# target

Y = Data['Class'].values

print(Y[:5])
# create training and testing sets

from sklearn.model_selection import train_test_split



x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=4)



print(x_train.shape, y_train.shape)

print(x_test.shape, y_test.shape)
from sklearn.preprocessing import StandardScaler



# scale object

scale = StandardScaler()

x_train = scale.fit_transform(x_train)

x_test = scale.transform(x_test)



print(x_train[:1])

print(x_test[:1])
# import libraries to analyze svm

from sklearn import svm

from sklearn.metrics import accuracy_score, f1_score, jaccard_score, precision_score, recall_score



# kernels to be used in the svm model

models = ['linear', 'poly', 'rbf', 'sigmoid']

# scoring of the models

acc_score = []

precision = []

recall = []

f1 = []

jacc = []



for m in models:

    

    # svm object fitted with training sets

    mach = svm.SVC(C=0.5, kernel=m)

    mach.fit(x_train, y_train)

    yhat = mach.predict(x_test)

    

    acc_score.append(accuracy_score(y_test, yhat))

    precision.append(precision_score(y_test, yhat, average='weighted'))

    recall.append(recall_score(y_test, yhat, average='weighted'))

    f1.append(f1_score(y_test, yhat, average='weighted'))

    jacc.append(jaccard_score(y_test, yhat, average='weighted'))

# position of the best score in the array

best_acc = np.array(acc_score).argmax()  # best accuracy score

best_prec = np.array(precision).argmax()  # best precision score

best_rec = np.array(recall).argmax()   # best recall score

best_f1 = np.array(f1).argmax()   # best f1 score

best_jacc = np.array(jacc).argmax()  # best jaccard score
print('Models with the best metrics:\n')

print('Best Accuracy: ', models[best_acc])

print('Best Precision: ', models[best_prec])

print('Best Recall: ', models[best_rec])

print('Best F1: ', models[best_f1])

print('Best Jaccard Score: ', models[best_jacc])
# svm object fitted with training data

lr = svm.SVC(C=0.5, kernel='linear', probability=True)

lr.fit(x_train, y_train)



# estimates

yhat = lr.predict(x_test)
# import metrics

from sklearn.metrics import classification_report, confusion_matrix



# confusion matrix

matrix = confusion_matrix(y_test, yhat)

sns.heatmap(pd.DataFrame(matrix), annot=True, cmap='YlGnBu', fmt='.2g')

plt.xticks((0.5,1.5), ['benign(2)', 'malignant(4)'], rotation=45)

plt.yticks((0.5,1.5), ['benign(2)', 'malignant(4)'], rotation=45)

plt.title('Confusion Matrix')

plt.xlabel('Predicted')

plt.ylabel('Actual')

plt.show()
# true positives and negative. False positive and negatives

TN = matrix[0][0]

FN = matrix[1][0]

TP = matrix[1][1]

FP = matrix[0][1]



sens = TP / (TP + FN)

prec = TP/ (TP + FP)

f1 = 2 * ((sens * prec) / (sens + prec))

print('Sensitivity (Recall): ', sens)

print('Positive Prediction Value (Precision): ', prec)

print('F1 score: ', f1)

print('Specificity: ', TN / (TN+FP))

print('Negative Prediction Value: ', TN/ (TN+FN))
# metrics

acc = accuracy_score(y_test, yhat)

prec = precision_score(y_test, yhat, average='weighted')

rec = recall_score(y_test, yhat, average='weighted')

jacc = jaccard_score(y_test, yhat, average='weighted')



print(acc, prec, rec, f1, jacc)
# classification report

report = classification_report(y_test, yhat)

print(report)
print(

    'The model has a Jaccard Index of 95.802% which is ', 

    'exceptionally high and gives us an indication that the ', 

    'testing set and estimate set are extremely similar.\n')

print(

    'The model also has an accuracy of 97.857% which is exceptional!\n')

print(

    'The model has an f1 score of 97.08% which shows the weighted average ', 

    'of precision and recall scores. It takes into account false positive and ',

    'false negatives into account and is a better metric to use when there is ', 

    'an unneven class distribution')

print(

    'The model has a an exceptional precision of 98.03% which shows ', 

    'the ratio of correctly predicted positive observations to the ', 

    'total predicted positive observations.\n')

print(

    'The model has a recall score of 96.15% which shows the ratio ', 

    'of correctly predicted positive observations to the all ', 

    'observations in actual class - benign(2).\n')