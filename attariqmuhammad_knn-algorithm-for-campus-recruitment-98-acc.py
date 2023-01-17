import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df=pd.read_csv("/kaggle/input/factors-affecting-campus-placement/Placement_Data_Full_Class.csv", index_col="sl_no")

df.head(10)
df.info()
print(df.groupby('status')['gender'].value_counts(normalize=True))

print("\n")

print(df.groupby('status')['ssc_b'].value_counts(normalize=True))

print("\n")

print(df.groupby('status')['hsc_b'].value_counts(normalize=True))

print("\n")

print(df.groupby('status')['hsc_s'].value_counts(normalize=True))

print("\n")

print(df.groupby('status')['degree_t'].value_counts(normalize=True))

print("\n")

print(df.groupby('status')['workex'].value_counts(normalize=True))

print("\n")

print(df.groupby('status')['specialisation'].value_counts(normalize=True))

missing_value=pd.DataFrame({" missing values" : df.isnull().sum() , "persentage missing values" : (df.isnull().sum()/len(df.index)*100)})

missing_value
df = df.fillna(df.mean())

df.describe()
X = df.drop('status', axis = 1)

y_cat = df.status



from sklearn.preprocessing import LabelEncoder

encoder=LabelEncoder()



#placed as 1, not placed as 0

y = encoder.fit_transform(y_cat)

X = pd.get_dummies(X)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.3, random_state=41)
from sklearn.neighbors import KNeighborsClassifier



# Setup arrays to store train and test accuracies

neighbors = np.arange(1, 15)

train_accuracy = np.empty(len(neighbors))

test_accuracy = np.empty(len(neighbors))



# Loop over different values of k

for i, k in enumerate(neighbors):

    # Setup a k-NN Classifier with k neighbors: knn

    knn = KNeighborsClassifier(n_neighbors= k )



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

from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, roc_auc_score

from sklearn.model_selection import cross_val_score



knn = KNeighborsClassifier(n_neighbors= 5 )

knn.fit(X_train, y_train)

knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)



cv_scores = cross_val_score(knn, X, y, cv=5)



print(confusion_matrix(y_test, y_pred))

print(classification_report(y_test, y_pred))



print(cv_scores)

print("\n average 5 fold tree : {}".format(np.mean(cv_scores)))



y_pred_proba=knn.predict_proba(X_test)[:,1]



print("\n ROC AUC Score knn : {}".format(roc_auc_score(y_test, y_pred_proba)))

print("\n accuracy score : {}".format(accuracy_score(y_test,y_pred)))

from sklearn.preprocessing import StandardScaler

scaler=StandardScaler()



scaler.fit(X)

X = scaler.transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.3, random_state=41)

from sklearn.neighbors import KNeighborsClassifier



# Setup arrays to store train and test accuracies

neighbors = np.arange(1, 15)

train_accuracy = np.empty(len(neighbors))

test_accuracy = np.empty(len(neighbors))



# Loop over different values of k

for i, k in enumerate(neighbors):

    # Setup a k-NN Classifier with k neighbors: knn

    knn = KNeighborsClassifier(n_neighbors= k )



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

from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, roc_auc_score

from sklearn.model_selection import cross_val_score



knn = KNeighborsClassifier(n_neighbors= 5 )

knn.fit(X_train, y_train)

knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)



cv_scores = cross_val_score(knn, X, y, cv=5)



print(confusion_matrix(y_test, y_pred))

print(classification_report(y_test, y_pred))



print(cv_scores)

print("\n average 5 fold tree : {}".format(np.mean(cv_scores)))



y_pred_proba=knn.predict_proba(X_test)[:,1]



print("\n ROC AUC Score knn : {}".format(roc_auc_score(y_test, y_pred_proba)))

print("\n accuracy score : {}".format(accuracy_score(y_test,y_pred)))
