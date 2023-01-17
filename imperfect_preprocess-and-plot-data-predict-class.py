# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
# Imports



import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline



from sklearn.preprocessing import Imputer

from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import train_test_split
# Define Column Names



column_names= ['party','infants','water','budget','physician','salvador','religious','satellite','aid','missile','immigration','synfuels','education','superfund','crime','duty-free-exports','eaa_rsa']
raw_data = pd.read_csv('../input/house-votes-84.csv', names=column_names, index_col=False)
raw_data.head()
# Replace Strings with Numbers

# With the replace function I get floats. Since these must represent votes I want integers. 

# Instead of converting the replaced values, I discard this method and use another method.



data_ignore = raw_data.copy()



data_ignore.replace('n',int(0), inplace=True)

data_ignore.replace('y',int(1), inplace=True)

data_ignore.replace('?',np.nan, inplace=True)

data_ignore.head()
data = raw_data.copy()



data[data=='n'] = 0

data[data=='y'] = 1

data[data=='?'] = np.nan

data.head()
# Save and Remove 'party'

# This is needed since the imputer can only work with numerical data



labels = data['party']

data.drop(['party'], axis=1, inplace=True)

data.head()
# Instantiate and Fit the Imputer



imp = Imputer(strategy='most_frequent') #defaults: 'missing_values' = 'NaN', axis=0

imp.fit(data)
# Replace NaNs with Imputed Data

# The method 'transform' returns a ndarray that needs to be converted to a DataFrame.

# Remember that we removed the first column before imputing the missing values, 

# therefore we now need to slice the previously defined column names.



data = pd.DataFrame(imp.transform(data), columns=column_names[1:], dtype=int)
data.head()
# In the raw data we had for instance 21 missing values and the most frequest class was 0



raw_data['synfuels'].value_counts()
# As expected these missing values were transformed in 0

data['synfuels'].value_counts()
data.info()
# Seaborn Countplots are very useful for the visualization of binary data



data['party'] = labels

plt.figure(figsize=(10,5))

sns.countplot(x='education', hue='party', data=data, palette='RdBu')

plt.xticks([0,1], ['No','Yes'])

plt.show()
plt.figure(figsize=(10,5))

sns.countplot(x='missile',hue='party',data=data,palette='viridis')

plt.xticks([0,1],['No','Yes'])

plt.show()
plt.figure(figsize=(10,5))

sns.countplot(x='satellite',hue='party',data=data,palette='winter')

plt.xticks([0,1], ['No','Yes'])

plt.show()
# Prepare Classifier Inputs

# We need ndarrays



y = data['party'].values

X = data.drop('party', axis=1).values
# Classification

# Split the data in two groups. Instantiate the classifier and fit it to the training data.

# Then evaluate the model accuracy first on the 

# training data (this is a good sanity check) and finally on the test data.



X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=42,stratify=y)

knn = KNeighborsClassifier(n_neighbors=6)

knn.fit(X_train,y_train)

print(f'Accuracy on the training set: {knn.score(X_train,y_train)}')

print(f'Accuracy on the test set: {knn.score(X_test,y_test)}')