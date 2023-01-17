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

import re

import nltk



nltk.download('stopwords')



from nltk.corpus import stopwords

from nltk.stem.porter import PorterStemmer



# Any results you write to the current directory are saved as output.
dataset = pd.read_csv("/kaggle/input/sms-spam-collection-dataset/spam.csv",encoding='latin-1')
print(dataset.head())

print("Size of the dataset:" , len(dataset))
required_dataset = dataset[['v1', 'v2']]



required_dataset = required_dataset.dropna(how='any',axis=0)



print("Size of the required dataset(After removing other columns and removing Nulls):" , len(required_dataset))



print(required_dataset.head())
corpus = []



for i in range(0, len(required_dataset)):

  detailed_desc = re.sub('[^a-zA-Z]', ' ', str(required_dataset['v2'][i]))

  detailed_desc = detailed_desc.lower()

  detailed_desc = detailed_desc.split()

  ps = PorterStemmer()

  detailed_desc = [ps.stem(word) for word in detailed_desc if not word in set(stopwords.words('english'))]

  detailed_desc = ' '.join(detailed_desc)

  corpus.append(detailed_desc)

print("size of the corpus ", len(corpus))

print("corpus: " , corpus[8])

#Creating the bag of words model

from sklearn.feature_extraction.text import CountVectorizer

cv = CountVectorizer(max_features= 500)

X = cv.fit_transform(corpus).toarray()

y = required_dataset.iloc[:,0].values

#print(X)

print(X[12])

print("Dimension of Matrix :" , X.shape)

print(y)

print("Size of the result list y",len(y))
%%time

#using Logistic regression with simple validation set..

from sklearn.model_selection import train_test_split as tts

x_train, x_test, y_train, y_test = tts(X, y, test_size=0.20, random_state = 0)



print("Creating a testing data set and training dataset using train_test_split lib")





from sklearn.metrics import accuracy_score

from sklearn.linear_model import LogisticRegression

print("Importing the logistic regression and accuracy_score")





# train a logistic regression model on the training set

# instantiate model

lrm_model = LogisticRegression(solver = 'newton-cg', multi_class = 'multinomial')



print(lrm_model)



# fit model

lrm_model.fit(x_train, y_train)



# make class predictions for the testing set

lrm_predictions = lrm_model.predict(x_test)



# calculate accuracy

model_accuracy = accuracy_score(y_test, lrm_predictions)



# calculate accuracy on training data

lrm_predictions_on_training_data = lrm_model.predict(x_train)

model_accuracy_on_training_data = accuracy_score(y_train, lrm_predictions_on_training_data)





#print(lrm_predictions)



print("-----------------------------------------------")

print("Using Logistic Regression:")

print("-----------------------------------------------")

print("Size of the Feature Vector  :", len(X[0]))

print()

print("Size of the Dataset         :",len(X))

print()

print("Accuracy score of the model :",model_accuracy)

print()

print("Accuracy score on training data:",model_accuracy_on_training_data)

print()





#accuracy using confusion matrix

from sklearn.metrics import confusion_matrix

from matplotlib import pyplot as plt

import seaborn as sns

cm = confusion_matrix(y_test,lrm_predictions)

#print(cm)

tmp = 0

tt = 0

for i in range(0 , len(cm)):

  for j in range(0, len(cm)):

    tt += cm[i][j]

    if i==j :

      tmp += cm[i][j]

      #print(cm[i][j])

#print(tmp)

#print(tt)

print("Accuracy score using confusion matrix: ",tmp/tt)



#HeatMap

fig, ax = plt.subplots(figsize=(15,15))

sns.heatmap(cm, annot=True, fmt='d')

#Using Kfold for logistic regression Validation



from sklearn.metrics import accuracy_score

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import KFold

from sklearn.metrics import confusion_matrix

from matplotlib import pyplot as plt



lrm_model = LogisticRegression(solver = 'newton-cg', multi_class = 'multinomial')



kf = KFold(n_splits=10, random_state=None, shuffle=False)



X = np.array(X)

y = np.array(y)

count=0

for train_index, test_index in kf.split(X):

  print(count)

  #print("TRAIN:", train_index, "TEST:", test_index)

  x_train, x_test = X[train_index], X[test_index]

  y_train, y_test = y[train_index], y[test_index]

  # fit model

  # train a logistic regression model on the training set

  # fit model

  lrm_model.fit(x_train, y_train)



  # make class predictions for the testing set

  lrm_predictions = lrm_model.predict(x_test)



  # calculate accuracy

  model_accuracy = accuracy_score(y_test, lrm_predictions)



  # calculate accuracy on training data

  lrm_predictions_on_training_data = lrm_model.predict(x_train)

  model_accuracy_on_training_data = accuracy_score(y_train, lrm_predictions_on_training_data)

  #print(lrm_predictions)

  print("-----------------------------------------------")

  print("Using Logistic Regression:")

  print("-----------------------------------------------")

  print("Size of the Feature Vector  :", len(X[0]))

  print()

  print("Size of the Dataset         :",len(X))

  print()

  print("Accuracy score of the model :",model_accuracy)

  print()

  print("Accuracy score on training data:",model_accuracy_on_training_data)

  print()



  #accuracy using confusion matrix

  cm = confusion_matrix(y_test,lrm_predictions)

  #print(cm)

  tmp = 0

  tt = 0

  for i in range(0 , len(cm)):

    for j in range(0, len(cm)):

      tt += cm[i][j]

      if i==j :

        tmp += cm[i][j]

        #print(cm[i][j])

  #print(tmp)

  #print(tt)

  print("Accuracy score using confusion matrix: ",tmp/tt)

  count+=1
