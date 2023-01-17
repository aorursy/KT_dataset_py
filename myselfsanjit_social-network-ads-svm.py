#importing the libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
plt.rcParams["figure.figsize"] = [15, 7]
#importing dataset
dataset = pd.read_csv('../input/Social_Network_Ads.csv')
dataset.head()

dataset.info()
# Looking for nulls
print(dataset.isnull().any())
#Checking correlation 
dataset.corr()
sns.pairplot(dataset)
#defining x and y in dataset
X = dataset[['Age', 'EstimatedSalary']]
y = dataset['Purchased']
#splitting dataset into the train and test set 
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state = 0)
#Feature scaling
from sklearn.preprocessing import MinMaxScaler
mms = MinMaxScaler()
X_train = mms.fit_transform(X_train)
X_test = mms.transform(X_test)
#creating classification
from sklearn.svm import SVC
classifier = SVC(kernel="rbf", random_state = 0) 
#using default kernel, other kernels are 'linear','poly','sigmoid','precomputed'
classifier.fit(X_train,y_train)
#Predicting the test set result
y_pred = classifier.predict(X_test)
print(y_pred)
#Making confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)
print(cm)
#ploting the graph for test data set
from matplotlib.colors import ListedColormap
X_set,y_set = X_test, y_test
X1,X2 =np.meshgrid(np.arange(start = X_set[:, 0].min() -1, stop = X_set[:, 0].max() +1, step = 0.01),
    np.arange(start = X_set[:, 1].min() -1, stop = X_set[:, 1].max() +1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape), 
             alpha = 0.50, cmap = ListedColormap(('orange','green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X1.min(), X1.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j,1], c=ListedColormap(('red','black'))(i), 
                label = j)

plt.title('SVM (Test Set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()

#ploting the graph for traning data set
from matplotlib.colors import ListedColormap
X_set,y_set = X_train, y_train
X1,X2 =np.meshgrid(np.arange(start = X_set[:, 0].min() -1, stop = X_set[:, 0].max() +1, step = 0.01),
    np.arange(start = X_set[:, 1].min() -1, stop = X_set[:, 1].max() +1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape), 
             alpha = 0.50, cmap = ListedColormap(('orange','green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X1.min(), X1.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j,1], c=ListedColormap(('red','black'))(i), 
                label = j)

plt.title('SVM (Train Set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()







