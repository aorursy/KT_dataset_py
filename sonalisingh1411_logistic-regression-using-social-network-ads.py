# Importing the libraries

import numpy as np

import matplotlib.pyplot as plt

import pandas as pd



# Importing the dataset

dataset = pd.read_csv('../input/Social_Network_Ads.csv')
#Checking the dataset

dataset.head()
#Check the metadata

dataset.info()
#Chceking the null values

dataset.isnull().sum()
#Check its dimensions

dataset.shape
#Plot UserID vs Purchased............

x1 = dataset.iloc[:, 0].values

y1 = dataset.iloc[:, 4].values

plt.scatter(x1,y1,color='Orange',s=50)

plt.xlabel('UserID')

plt.ylabel('Purchased')

plt.title('UserID vs Purchased')

plt.show()
#Plot Gender vs Purchased............

x1 = dataset.iloc[:, 1].values

y1 = dataset.iloc[:, 4].values

plt.scatter(x1,y1,color='pink',s=50)

plt.xlabel('Gender')

plt.ylabel('Purchased')

plt.title('Gender vs Purchased')

plt.show()
#Plot Age vs Purchased............

x1 = dataset.iloc[:, 2].values

y1 = dataset.iloc[:, 4].values

plt.scatter(x1,y1,color='purple',s=50)

plt.xlabel('Age')

plt.ylabel('Purchased')

plt.title('Age vs Purchased')

plt.show()
#Plot Estimatedsalary vs Purchased............

x1 = dataset.iloc[:, 3].values

y1 = dataset.iloc[:, 4].values

plt.scatter(x1,y1,color='red',s=50)

plt.xlabel('Estimatedsalary')

plt.ylabel('Purchased')

plt.title('Estimatedsalary vs Purchased')

plt.show()
#Headmap:-To see the correlation between them!

import seaborn as sns

plt.figure(figsize=(7,4)) #7 is the size of the width and 4 is parts.... 

sns.heatmap(dataset.corr(),annot=True,cmap='cubehelix_r') 
#Seperating dependent and indepndent values

X = dataset.iloc[:, [2, 3]].values

y = dataset.iloc[:, 4].values

print(X)
# Splitting the dataset into the Training set and Test set

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30, random_state = 0)

# Feature Scaling

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

X_train = sc.fit_transform(X_train)

X_test = sc.transform(X_test)
# Fitting Logistic Regression to the Training set

from sklearn.linear_model import LogisticRegression

classifier = LogisticRegression(random_state = 0)

classifier.fit(X_train, y_train)


# Predicting the Test set results

y_pred = classifier.predict(X_test)

# Making the Confusion Matrix

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)

print(cm)
#Accuray=(TN+TP)/Total+

Accuracy=(74+31)/120

Accuracy
#Error_rate=(FN+FP)/Total

Error_rate=(5+10)/120

Error_rate


# Visualising the Training set results

from matplotlib.colors import ListedColormap

X_set, y_set = X_train, y_train

X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),

                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))

plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),

             alpha = 0.75, cmap = ListedColormap(('brown', 'yellow')))

plt.xlim(X1.min(), X1.max())

plt.ylim(X2.min(), X2.max())

for i, j in enumerate(np.unique(y_set)):

    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],

                c = ListedColormap(('yellow', 'brown'))(i), label = j)

plt.title('Logistic Regression (Training set)')

plt.xlabel('Age')

plt.ylabel('Estimated Salary')

plt.legend()

plt.show()

# Visualising the Test set results

from matplotlib.colors import ListedColormap

X_set, y_set = X_test, y_test

X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),

                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))

plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),

             alpha = 0.75, cmap = ListedColormap(('blue', 'black')))

plt.xlim(X1.min(), X1.max())

plt.ylim(X2.min(), X2.max())

for i, j in enumerate(np.unique(y_set)):

    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],

                c = ListedColormap(('black', 'blue'))(i), label = j)

plt.title('Logistic Regression (Test set)')

plt.xlabel('Age')

plt.ylabel('Estimated Salary')

plt.legend()

plt.show()