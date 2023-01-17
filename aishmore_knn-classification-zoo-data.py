# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



import warnings

warnings.filterwarnings("ignore")



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# Import additional required libraries

from sklearn.model_selection import train_test_split

from sklearn.neighbors import KNeighborsClassifier

from sklearn import metrics

from sklearn.model_selection import *

from sklearn.metrics import confusion_matrix,classification_report, accuracy_score

from sklearn.model_selection import cross_val_score
# Importing dataset

zoo_df = pd.read_csv('../input/zoo-animal-classification/zoo.csv')

class_df = pd.read_csv('../input/zoo-animal-classification/class.csv')

zoo_df.head()
class_df.head()
# Joining datasets along the class number column present in both datasets

animal_df = zoo_df.merge(class_df,how='left',left_on='class_type',right_on='Class_Number')

animal_df.head()
# Dropping unwanted columns

## I am renaming the dataframe as zoo_df because it is shorter to use

zoo_df = animal_df.drop(['class_type','Animal_Names', 'Number_Of_Animal_Species_In_Class'], axis=1)

zoo_df.head()
zoo_df.isnull().any()
# Get names of columns in zoo_df

zoo_df.info()
zoo_df.describe()
# Set default plot grid

sns.set_style('whitegrid')
# Plot histogram of classes

plt.rcParams['figure.figsize'] = (7,7)

sns.countplot(zoo_df['Class_Type'], palette='YlGnBu')

ax = plt.gca()

ax.set_title("Histogram of Classes")
zoo_df['has_legs'] = np.where(zoo_df['legs']>0,1,0)

zoo_df = zoo_df[['animal_name','hair','feathers','eggs','milk', 'airborne', 'aquatic', 'predator', 'toothed', 'backbone', 'breathes','venomous','fins','legs','has_legs','tail','domestic','catsize','Class_Number','Class_Type']]

zoo_df.head()
zoo_df_temp = zoo_df.drop(['has_legs','Class_Number'], axis=1)

zoo_df_temp = zoo_df_temp.groupby(by='animal_name').mean()

plt.rcParams['figure.figsize'] = (16,10) 

sns.heatmap(zoo_df_temp, cmap="inferno")

ax = plt.gca()

ax.set_title("Features for the Animals")
zoo_df_temp = zoo_df.drop(['has_legs','Class_Number'], axis=1)

zoo_df_temp = zoo_df_temp.groupby(by='Class_Type').mean()

plt.rcParams['figure.figsize'] = (16,10) 

sns.heatmap(zoo_df_temp, annot=True, cmap="inferno")

ax = plt.gca()

ax.set_title("HeatMap of Features for the Classes")
zoo_df_temp = zoo_df.drop(['legs','Class_Number'], axis=1)

zoo_df_temp = zoo_df_temp.groupby(by='animal_name').mean()

plt.rcParams['figure.figsize'] = (16,10) 

sns.heatmap(zoo_df_temp, cmap="inferno")

ax = plt.gca()

ax.set_title("Features for the Animals")
zoo_df_temp = zoo_df.drop(['legs','Class_Number'], axis=1)

zoo_df_temp = zoo_df_temp.groupby(by='Class_Type').mean()

plt.rcParams['figure.figsize'] = (16,10) 

sns.heatmap(zoo_df_temp, annot=True, cmap="inferno")

ax = plt.gca()

ax.set_title("HeatMap of Features for the Classes")
zoo_df.head()
# Select columns to add to X and y sets

features = list(zoo_df.columns.values)

features.remove('has_legs')

features.remove('Class_Type')

features.remove('Class_Number')

features.remove('animal_name')

X = zoo_df[features]

y = zoo_df['Class_Number']
# Split X and y into train and test

X_train, X_test, y_train, y_test = train_test_split(X,y,random_state = 0)
# Fit k-nearest neighbors classifier with training sets for n = 5

knn = KNeighborsClassifier(n_neighbors = 5)

knn.fit(X_train, y_train)
# Run prediction

y_pred = knn.predict(X_test)
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))
plt.rcParams['figure.figsize'] = (9,9) 

_, ax = plt.subplots()

ax.hist(y_test, color = 'm', alpha = 0.5, label = 'actual', bins=7)

ax.hist(y_pred, color = 'c', alpha = 0.5, label = 'prediction', bins=7)

ax.yaxis.set_ticks(np.arange(0,11))

ax.legend(loc = 'best')

plt.show()
# Get score for different values of n

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



print("Mean Scores:")

print(mean_scores)

print("Error Rate:")

print(error_rate)
# Plot n values and average accuracy scores

plt.plot(k_list,mean_scores, marker='o')



# Added titles and adjust dimensions

plt.title('Accuracy of Model for Varying Values of K')

plt.xlabel("Values of K")

plt.ylabel("Mean Accuracy Score")

plt.xticks(k_list)

plt.rcParams['figure.figsize'] = (12,12) 



plt.show()
# Plot n values and average accuracy scores

plt.plot(k_list,error_rate, color='r', marker = 'o')



# Added titles and adjust dimensions

plt.title('Error Rate for Model for Varying Values of K')

plt.xlabel("Values of K")

plt.ylabel("Error Rate")

plt.xticks(k_list)

plt.rcParams['figure.figsize'] = (12,12) 



plt.show()
# Select columns to add to X and y sets

features = list(zoo_df.columns.values)

features.remove('legs')

features.remove('Class_Type')

features.remove('Class_Number')

features.remove('animal_name')

X2 = zoo_df[features]

y2 = zoo_df['Class_Type']
# Split X and y into train and test

X2_train, X2_test, y2_train, y2_test = train_test_split(X2,y2,random_state = 0)
# Fit k-nearest neighbors classifier with training sets for n = 5

knn2 = KNeighborsClassifier(n_neighbors = 5)

knn2.fit(X2_train, y2_train)
# Run prediction

y2_pred = knn2.predict(X2_test)
print(confusion_matrix(y2_test,y2_pred))
print(classification_report(y2_test,y2_pred))
plt.rcParams['figure.figsize'] = (9,9) 

_, ax = plt.subplots()

ax.hist(y2_test, color = 'm', alpha = 0.5, label = 'actual', bins=7)

ax.hist(y2_pred, color = 'c', alpha = 0.5, label = 'prediction', bins=7)

ax.yaxis.set_ticks(np.arange(0,11))

ax.legend(loc = 'best')



plt.show()
# Get score for different values of n

k_list = np.arange(1, 50, 2)

mean_scores2 = []

accuracy_list2 = []

error_rate2 = []



for i in k_list:

    knn2 = KNeighborsClassifier(n_neighbors=i)

    knn2.fit(X2_train,y2_train)

    pred_i = knn2.predict(X2_test)

    score = cross_val_score(knn2,X2_train, y2_train,cv=10)

    mean_scores2.append(np.mean(score))

    error_rate2.append(np.mean(pred_i != y2_test))



print("Mean Scores:")

print(mean_scores)

print("Error Rate:")

print(error_rate)
# Plot n values and average accuracy scores

plt.plot(k_list,mean_scores, color='b',marker='o', label='Model using Number of Legs')

plt.plot(k_list,mean_scores2, color='m',marker='x', label='Model using Presence of Legs')



# Added titles and adjust dimensions

plt.title('Accuracy of Model for Varying Values of K')

plt.xlabel("Values of K")

plt.ylabel("Mean Accuracy Score")

plt.xticks(k_list)

plt.legend()

plt.rcParams['figure.figsize'] = (12,12) 



plt.show()
# Plot n values and average accuracy scores

plt.plot(k_list,error_rate, color='r', marker = 'o', label='Model using Number of Legs')

plt.plot(k_list,error_rate2, color='c', marker = 'x', label='Model using Presence of Legs')



# Added titles and adjust dimensions

plt.title('Error Rate for Model for Varying Values of K')

plt.xlabel("Values of K")

plt.ylabel("Error Rate")

plt.xticks(k_list)

plt.legend()

plt.rcParams['figure.figsize'] = (12,12) 



plt.show()