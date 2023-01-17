# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

import scipy.stats as stats



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
Data = pd.read_csv('../input/married-at-first-sight/mafs.csv')

Data.head()
Data.shape #Dimensions
Data.info() #Information about Data
my_data = Data.copy()
print(my_data.describe(include=['O']))

print(my_data.describe()) #Descriptive Statistics
my_data.drop(['DrPepperSchwartz'], axis=1,inplace = True)

my_data.columns
my_data.head()
my_data.drop(['Name','Occupation'], axis=1,inplace = True)

my_data.columns
from sklearn.preprocessing import LabelEncoder

enc = LabelEncoder()

my_data['Gender'] = enc.fit_transform(my_data['Gender'])

my_data['Status'] = enc.fit_transform(my_data['Status'])

my_data['Decision'] = enc.fit_transform(my_data['Decision'])
my_data.drop(['Location'], axis=1,inplace = True)

my_data.head()
plt.figure(figsize=(14,14))

corr_matrix = my_data.corr().round(2)

sns.heatmap(data=corr_matrix, annot=True)
X = my_data.iloc[:, [3,4,6,7,8,9,10,11,12]].values

y = my_data.iloc[:,5].values
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)

#Support Vector Classifier



from sklearn.svm import SVC 

classifier=SVC(kernel='linear',random_state=0)

#Fitting training data and making predictions on test data

classifier.fit(X_train,y_train)

y_pred=classifier.predict(X_test)
from sklearn.metrics import plot_confusion_matrix

plot_confusion_matrix(classifier, X_test, y_test)
#Random Forest Classifier

from sklearn.ensemble import RandomForestClassifier

classifier=RandomForestClassifier(n_estimators=10,criterion='entropy',random_state=0)



#Fitting training data and making predictions on test data



classifier.fit(X_train,y_train)

y_pred=classifier.predict(X_test)
from sklearn.metrics import plot_confusion_matrix

plot_confusion_matrix(classifier, X_test, y_test)
probs = classifier.predict_proba(X_test)

from sklearn.metrics import roc_auc_score

roc_auc_score(y_test, probs[:,1])
#Importing the basic libraries and components



import keras

from keras.models import Sequential

from keras.layers import Dense

from keras.layers import Dropout

from keras.wrappers.scikit_learn import KerasClassifier



#Building the classifier and adding the layers



classifier = Sequential()

classifier.add(Dense(units=5, kernel_initializer='uniform', activation='relu', input_dim=9))

classifier.add(Dense(units=5, kernel_initializer='uniform', activation='relu'))

classifier.add(Dense(units=1, kernel_initializer='uniform', activation='sigmoid'))
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
#Fitting the data to the training set and making predictions on the test set

classifier.fit(X_train,y_train,batch_size=1, epochs=100) 

y_pred=classifier.predict(X_test)
y_pred #Probabilities
y_pred1 = (y_pred>0.5)

from sklearn.metrics import confusion_matrix

cm=confusion_matrix(y_test,y_pred1)

cm
from sklearn.metrics import roc_auc_score

roc_auc_score(y_test, y_pred)