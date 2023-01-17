# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import warnings  

warnings.filterwarnings("ignore")   # ignore warnings



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))





# Any results you write to the current directory are saved as output.
wine = pd.read_csv('../input/winequality-red.csv')

wine.head()
wine.info()
X = wine.iloc[:,0:11].values

X
y = wine.iloc[:,11].values

y
# Creation of train and test sets from the data.



from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
# Scaling with Standardization



from sklearn.preprocessing import StandardScaler



sc = StandardScaler()

X_train = sc.fit_transform(X_train)

X_test = sc.transform(X_test)
from sklearn.decomposition import PCA



pca = PCA(n_components= 2)  # we will reduce the data set from 11 columns to 2 columns.

X_train2 = pca.fit_transform(X_train) # fit means train, fit_transform means train and apply to a data set.

X_test2 = pca.transform(X_test)  # Only transformation



# X_train2 is a 2 dimensional data set.
# LR before PCA transformation

from sklearn.linear_model import LogisticRegression



# random_state = 0 because the model will be used two times and we want to have same structure.

# Thus, same LR algorithm structure will run.

classifier = LogisticRegression(random_state=0)

classifier.fit(X_train, y_train)
# LR after PCA transformation



classifier2 = LogisticRegression(random_state=0)

classifier2.fit(X_train2, y_train)
# Predictions



# Prediction from the data that is not applied PCA.

y_pred = classifier.predict(X_test) 



# Prediction from the data that is applied PCA.

y_pred2 = classifier2.predict(X_test2) 
# Evaluation



from sklearn.metrics import confusion_matrix



# actual / result without PCA

print('actual / without PCA')

cm = confusion_matrix(y_test, y_pred)

print(cm)

print('          ')



# actual / result with PCA

print('actual / with PCA')

cm2 = confusion_matrix(y_test, y_pred2)

print(cm2)

print('          ')



# after PCA / before PCA

print('without PCA / with PCA')

cm3 = confusion_matrix(y_pred, y_pred2)

print(cm3)



# acuracy is 0.63 without PCA.

# accuracy is 0.56 with PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA



lda = LDA(n_components = 2)

X_train_lda = lda.fit_transform(X_train, y_train)

X_test_lda = lda.transform(X_test)
# LR after LDA transformation



classifier_lda = LogisticRegression(random_state=0)

classifier_lda.fit(X_train_lda, y_train)
# Predictions of LDA data



y_pred_lda = classifier_lda.predict(X_test_lda)  
# Evaluation with Confusion matrix



# original / After LDA 

print('Original & LDA')

cm4 = confusion_matrix(y_pred, y_pred_lda)

print(cm4)

print('          ')



# actual / result without LDA

print('actual / without LDA')

cm5 = confusion_matrix(y_test, y_pred)

print(cm5)

print('          ')



# actual / result with LDA

print('actual / with LDA')

cm6 = confusion_matrix(y_test, y_pred_lda)

print(cm6)



# acuracy is 0.63 without LDA.

# acuracy is 0.62 with LDA.