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



dataset = pd.read_csv('../input/social-network-ads/Social_Network_Ads.csv') #adding data

#print 1st 5 rows

dataset.head()
#tjis data is albeled data

#independent variable (X) will be Age, Estimated Salary

#dependent variable(y) will be Purcahsed (It will be lable)



#define X,y

X = dataset.iloc[:, [2, 3]].values

y = dataset.iloc[:, 4].values



#print X,y

print(X[:5, :])

print(y[:5])
#import train_test_split from sklearn to train and test our data

from sklearn.model_selection import train_test_split 

#define 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)



#print

print(X_train[:5],'\n', y_train[:5],'\n', X_test[:5],'\n', y_test[:5])
#dataset.info()

#check corelation among attributes of dataset 

dataset.corr()
#normalization and feature scaling

from sklearn.preprocessing import StandardScaler

sc_X = StandardScaler()

X_train = sc_X.fit_transform(X_train)

X_test = sc_X.transform(X_test)
print(X_train[:5], '\n','\n', X_test[:5])
#create object of LogisticRegression class to refer as classifier

from sklearn.linear_model import LogisticRegression



classifier = LogisticRegression(random_state = 0)

classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)



print(X_test[:10], '\n', '\n', y_pred[:10])
#Predicted yes are 8th and 10th and they are the oness with good enough positive values

#so we van predict that as the value incerase there are more chances of ad being clicked and purchased

#check this with y_pred to compare our assumptions

print(y_pred[:10],'\n','\n',y_test[:10])
#probability is 9/10 i.e 90% so, good to go
#we'll use a Confusion Matrix to evaluate exactly how accurate our Logistic Regression model is,

#the more the matching values, more is the accuracy



from sklearn.metrics import confusion_matrix

matrix = confusion_matrix(y_test, y_pred)

print(matrix)
#visualization of data by plotting graphs



from matplotlib.colors import ListedColormap



X_set, y_set = X_train, y_train



X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),

                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))



plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),

             alpha = 0.6, cmap = ListedColormap(('blue', 'red')))



plt.xlim(X1.min(), X1.max())

plt.ylim(X2.min(), X2.max())



for i, j in enumerate(np.unique(y_set)):

    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],

                c = ListedColormap(('blue', 'red'))(i), label = j)

plt.title('Training set')

plt.xlabel('Age')

plt.ylabel('Estimated Salary')

plt.legend()

plt.show()
#we can tell that about 80-90% of the observations have been correctly identified

#correlations between the dependent and independent variables shows that as Age and Estimated Salary increase,

#they will click on the ad

#now plotting graph for test data
# Visualizing the Test set results

from matplotlib.colors import ListedColormap



X_set, y_set = X_test, y_test



X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),

                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))

plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),

             alpha = 0.6, cmap = ListedColormap(('blue', 'red')))



plt.xlim(X1.min(), X1.max())

plt.ylim(X2.min(), X2.max())



for i, j in enumerate(np.unique(y_set)):

    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],

                c = ListedColormap(('blue', 'red'))(i), label = j)

plt.title('Test set')

plt.xlabel('Age')

plt.ylabel('Estimated Salary')

plt.legend()

plt.show()
#coclusion : increase in both Age and Estimated Salary will lead to a higher probability of clicking(purchasing) the advertisement