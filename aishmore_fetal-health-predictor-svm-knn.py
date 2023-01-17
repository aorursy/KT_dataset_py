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
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns
#Read data from csv file

df = pd.read_csv('../input/fetal-health-classification/fetal_health.csv')
#Preview data as given

df.head(10)
#Check for and remove any null entries

num_null = df.isnull().sum()

print("Number of null value entries:\n",num_null)
#Check for and remove any duplicate entries

num_duplicates = df.duplicated().sum()

print("Number of duplicate entries:\n", num_duplicates)
#Remove duplicate entries

df = df.drop_duplicates()

print("Number of duplicate entries:\n", df.duplicated().sum())
#Basic data structure (data types and number of entries)

df.info()
#Summary statistics for the data

df.describe()
# Plot histogram of fetal health (target variable)

plt.rcParams['figure.figsize'] = (7,7)

sns.countplot(df['fetal_health'])

ax = plt.gca()

#ax.set_title("Histogram of Classes")
# Heatmap to determine correlation between all features

ax=plt.subplots(figsize=(12,12))

sns.heatmap(df.corr())
from sklearn.model_selection import train_test_split

from sklearn.svm import SVC

from sklearn.preprocessing import StandardScaler

from sklearn.metrics import confusion_matrix

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
#Split data into X and y

X = df.drop('fetal_health', axis=1)

y = df['fetal_health']
#Split data into train and test set

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
#Define and fit Linear SVC

svclassifier = SVC(kernel='linear')

svclassifier.fit(X_train, y_train)
#Make prediction using classifier

y_pred = svclassifier.predict(X_test)
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))
print('Accuracy Score:')

print(accuracy_score(y_test,y_pred))
#Will use libraries imported for previous model along with these

from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import cross_val_score
#Split data into X and y

X = df.drop('fetal_health', axis=1)

y = df['fetal_health']
#Split data into train and test set

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
# Get score for different values of k

k_list = np.arange(1, 50, 2)

mean_scores = []

accuracy_list = []

error_rate = []



for i in k_list:

    knn = KNeighborsClassifier(n_neighbors=i)

    knn.fit(X_train,y_train)

    pred_i = knn.predict(X_test)

    score = cross_val_score(knn,X_train, y_train,cv=10)

    mean_scores.append(np.mean(score))

    error_rate.append(np.mean(pred_i != y_test))
# Plot n values and average accuracy scores

plt.plot(k_list,mean_scores, marker='o')



# Added titles and adjust dimensions

plt.title('Accuracy of Model for Varying Values of K')

plt.xlabel("Values of K")

plt.ylabel("Mean Accuracy Score")

plt.xticks(k_list)

plt.rcParams['figure.figsize'] = (12,12) 

plt.grid(linewidth=2)



plt.show()



# Plot n values and average accuracy scores

plt.plot(k_list,error_rate, color='r', marker = 'o')



# Added titles and adjust dimensions

plt.title('Error Rate for Model for Varying Values of K')

plt.xlabel("Values of K")

plt.ylabel("Error Rate")

plt.xticks(k_list)

plt.rcParams['figure.figsize'] = (12,12) 

plt.grid(linewidth=1.5)



plt.show()
# Fit k-nearest neighbors classifier with training sets for n = 5

knn = KNeighborsClassifier(n_neighbors = 5)

knn.fit(X_train, y_train)



# Run prediction

y_pred = knn.predict(X_test)
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))
print('Accuracy Score:')

print(accuracy_score(y_test,y_pred))