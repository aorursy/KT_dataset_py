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
import pandas as pd

irisr2 = pd.read_csv("../input/irisr2.csv")
import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt
irisr2.sample(10)
irisr2.isnull().sum()
irisr2.info()
irisr2.isna()
irisr2['SepalLengthCm'].fillna(irisr2['SepalLengthCm'].median(),inplace=True)

irisr2['SepalWidthCm'].fillna(irisr2['SepalWidthCm'].median(),inplace = True)

irisr2['PetalLengthCm'].fillna(irisr2['PetalLengthCm'].median(),inplace = True)

irisr2['PetalWidthCm'].fillna(irisr2['PetalWidthCm'].median(),inplace = True)
# We are checking whether the data has been refined



irisr2.isnull().sum()
### Dealing with categorical data



##Change all the classes to numericals (0 to 2)



###Hint: use **LabelEncoder()**



from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

irisr2['Species'] = le.fit_transform(irisr2['Species'])
irisr2.loc[:,~irisr2.corr()['Species'].between(-0.1,0.1,inclusive=True)].head()
irisr2.loc[:,irisr2.var()>0.1].head()
sns.pairplot(irisr2,hue='Species')
from sklearn.model_selection import train_test_split

from sklearn.neighbors import KNeighborsClassifier

from scipy.stats import zscore







feature_columns = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm','PetalWidthCm']

X = irisr2[feature_columns].values

y = irisr2['Species'].values







from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

y = le.fit_transform(y)
# Split X and y into training validation and testing data set



from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

## Calculating the Z Score befor instatiating the model



X_train_z = zscore(X_train)  



X_test_z = zscore(X_test)
# Instantiating the model 

# when value of K = 3



from sklearn.metrics import confusion_matrix, accuracy_score

from sklearn.model_selection import cross_val_score

classifier = KNeighborsClassifier(n_neighbors=3)



#Fitting the model

classifier.fit(X_train, y_train)



# Predicting the Test set results

y_pred = classifier.predict(X_test)
## Building Confusion matrix



cm = confusion_matrix(y_test, y_pred)

cm
## Calculating the accuracy model







accuracy = accuracy_score(y_test, y_pred)*100

print('Accuracy of our model is equal ' + str(round(accuracy, 2)) + ' %.')
# creating list of K for KNN



k_list = list(range(1,50,2))

# creating list of cv scores

cv_scores = []



# perform 10-fold cross validation



for k in k_list:

    knn = KNeighborsClassifier(n_neighbors=k)

    scores = cross_val_score(knn, X_train, y_train, cv=10, scoring='accuracy')

    cv_scores.append(scores.mean())

## Find optimal value of K



##- Run the KNN with no of neighbours to be 1, 3, 5 ... 19

##- Find the **optimal number of neighbours** from the above list



# changing to misclassification error

MSE = [1 - x for x in cv_scores]



plt.figure()

plt.figure(figsize=(15,10))

plt.title('The optimal number of neighbors', fontsize=20, fontweight='bold')

plt.xlabel('Number of Neighbors K', fontsize=15)

plt.ylabel('Misclassification Error', fontsize=15)

sns.set_style("whitegrid")

plt.plot(k_list, MSE)



plt.show()
## FINDING THE BEST VALUE OF K ( OPTIMAL VALUE OF K)



best_k = k_list[MSE.index(min(MSE))]

print("The optimal number of neighbors is %d." % best_k)
## Plot accuracy



##Plot accuracy score vs k (with k value on X-axis) using matplotlib.
neighbors = np.arange(1, 9)

train_accuracy = np.empty(len(neighbors))

test_accuracy = np.empty(len(neighbors))



# Loop over different values of k

for i, k in enumerate(neighbors):

    # Setup a k-NN Classifier with k neighbors: knn

    knn = KNeighborsClassifier(n_neighbors=k)



    # Fit the classifier to the training data

    knn.fit(X_train, y_train)

    

    #Compute accuracy on the training set

    train_accuracy[i] = knn.score(X_train, y_train)



    #Compute accuracy on the testing set

    test_accuracy[i] = knn.score(X_test, y_test)



# Generate plot

plt.title('k-NN: Varying Number of Neighbors')

plt.plot(neighbors, test_accuracy, label = 'Testing Accuracy')

plt.plot(neighbors, train_accuracy, label = 'Training Accuracy')

plt.legend()

plt.xlabel('Number of Neighbors')

plt.ylabel('Accuracy')

plt.show()