# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

%matplotlib inline



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
data = pd.read_csv("../input/breast-cancer-wisconsin-data/data.csv")

data.head()
# Dropping some of the unwanted variables:

data.drop('id',axis=1,inplace=True)

data.drop('Unnamed: 32',axis=1,inplace=True)
from sklearn import preprocessing



#Convert "diagnosis" to the target variable: M = 1, B = 0

le = preprocessing.LabelEncoder()

data['diagnosis'] = le.fit_transform(data['diagnosis'])

data.head()
df = pd.DataFrame(preprocessing.scale(data.iloc[:,1:32]))

df.columns = list(data.iloc[:,1:32].columns)

df['diagnosis'] = data['diagnosis']
#percentages values of Malignant and Beningn cancer

perc_tumors = df.diagnosis.value_counts(normalize=True) * 100



#Looking at the percentages with Malignant and Benign cancer

perc_tumors.plot(kind='bar', alpha = 0.5, facecolor = 'g', figsize=(10,5))

plt.title("Diagnosis (M=1 , B=0)", fontsize = '18')

plt.ylabel("Percentage of patients")

plt.xlabel("Type of cancer")

plt.grid(b=True)
#Number of patients with Malignant and Benign cancer

df.diagnosis.value_counts().plot(kind='bar', alpha = 0.5, facecolor = 'r', figsize=(10,5))

plt.title("Diagnosis (M=1 , B=0)", fontsize = '18')

plt.ylabel("Total Number of patients")

plt.xlabel("Type of cancer")

plt.grid(b=True)
import seaborn as sns



data_mean = data[['diagnosis','radius_mean','texture_mean','perimeter_mean','area_mean','smoothness_mean', 'compactness_mean', 'concavity_mean','concave points_mean', 'symmetry_mean', 'fractal_dimension_mean']]

plt.figure(figsize=(14,14))

foo = sns.heatmap(data_mean.corr(), vmax=1, square=True, annot=True)
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict

from sklearn import metrics



predictors = data_mean.columns[2:11]

target = "diagnosis"



X = data_mean.loc[:,predictors]

y = np.ravel(data.loc[:,[target]])



# Split the dataset into training and test set:

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

print ('Samples in Training Set : %i' % (X_train.shape[0]))

print ('Samples in Test Set : %i' % (X_test.shape[0]))
# Importing the model:

from sklearn.neighbors import KNeighborsClassifier



# Initiating the model:

knn = KNeighborsClassifier()



scores = cross_val_score(knn, X_train, y_train, scoring='accuracy' ,cv=10).mean()



print("The mean accuracy with 10 fold cross validation is: %s " % round(scores*100,2), "%")
for i in range(1, 21):

    knn = KNeighborsClassifier(n_neighbors = i)

    score = cross_val_score(knn, X_train, y_train, scoring='accuracy' ,cv=10).mean()

    print("N = " + str(i) + " :: Score = " + str(round(score,2)))
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)



# Importing the model:

from sklearn.ensemble import RandomForestClassifier



# Initiating the model:

rf = RandomForestClassifier()



scores = cross_val_score(rf, X_train, y_train, scoring='accuracy' ,cv=10).mean()



print("The mean accuracy with 10 fold cross validation is: %s " % round(scores*100,2), "%")
for i in range(1, 21):

    rf = RandomForestClassifier(n_estimators = i)

    score = cross_val_score(rf, X_train, y_train, scoring='accuracy' ,cv=10).mean()

    print("N = " + str(i) + " :: Score = " + str(round(score,2)))
# Importing the model:

from sklearn.naive_bayes import GaussianNB



# Initiating the model:

nb = GaussianNB()



scores = cross_val_score(rf, X_train, y_train, scoring='accuracy' ,cv=10).mean()



print("The mean accuracy with 10 fold cross validation is: %s" % round(scores*100,2), "%")
from sklearn.ensemble import RandomForestClassifier



# Initiating the model:

rf = RandomForestClassifier(n_estimators=18)



rf = rf.fit(X_train, y_train)



predicted = rf.predict(X_test)



acc_test = metrics.accuracy_score(y_test, predicted)



print ('The accuracy on test set is %s' % (round(acc_test,2)))
from sklearn.naive_bayes import GaussianNB



# Initiating the model:

nb = GaussianNB()



nb = nb.fit(X_train, y_train)



predicted = nb.predict(X_test)



acc_test = metrics.accuracy_score(y_test, predicted)



print ('The accuracy on test set is %s' % (round(acc_test,2)))