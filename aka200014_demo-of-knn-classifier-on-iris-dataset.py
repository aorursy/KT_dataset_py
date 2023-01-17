# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns # for data plotting and visualization



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
dataset = pd.read_csv('../input/iris/Iris.csv')
dataset.head(100) # prints the top 100 items from the dataset.
dataset.describe() # prints some details about each column.
sns.pairplot(hue = 'Species', data = dataset) # plot between each pair of column 
features = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm','PetalWidthCm']  # Extracting the required columns for Input features.

X = dataset[features].values  # Input features

y = dataset['Species'].values  # Output to be predicted
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()  

y = le.fit_transform(y)  # Converts the String outputs into numbers. 
pd.DataFrame(y).head() # Prints the converted output y.
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0) # Splitting the dataset into Training and Testing in 80-20 ratio.
from sklearn.neighbors import KNeighborsClassifier  # Importing KNN classifier from Scikit learn

from sklearn.metrics import accuracy_score          # For printing the accuracy

classifier = KNeighborsClassifier(n_neighbors=3)   # Create a classifier with number of neighbours to search for as 3 and rest default.

classifier.fit(X_train, y_train)   

y_pred = classifier.predict(X_test)                # Predict
accuracy = accuracy_score(y_test, y_pred)*100   # Print Accuracy

print('Accuracy of our model is equal ' + str(accuracy) + ' %.')