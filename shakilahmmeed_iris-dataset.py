import numpy as np

import pandas as pd

import seaborn as sns
# 1. Get the data

data = pd.read_csv('../input/Iris.csv')
# 2. Explore the data

data.info()

# data.shape

# data.head(10)

# data.describe()

# data.drop('Id', axis=1, inplace=True)

# data.info()
# 3. Clean the data

data.isna().sum()
# 4. Train and test - Prepare the data for ML model

from sklearn.model_selection import train_test_split



train, test = train_test_split(data, random_state=4, test_size=0.2)

# train.shape

# test.shape



train_x = train[['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm']]

test_x = test[['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm']]

train_y = train.Species

test_y = test.Species
# 4. Train and test (Support Vector Machine Algorithm)

from sklearn.svm import SVC

from sklearn.metrics import accuracy_score



# Create

model = SVC(kernel='linear')



# Train

model.fit(train_x, train_y)



# Test

output = model.predict(test_x)

accuracy_score(output, test_y)
# 4. Train and test (Logistic Regression Algorithm)

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score



# Create

model = LogisticRegression()



# Train

model.fit(train_x, train_y)



#Test

output = model.predict(test_x)

accuracy_score(output, test_y)
# 4. Train and test (Decision Tree)

from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import accuracy_score



# Create

model = DecisionTreeClassifier()



# Train

model.fit(train_x, train_y)



#Test

output = model.predict(test_x)

accuracy_score(output, test_y)
# 4. Train and test (K-Nearest Neighbours)

from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import accuracy_score



# Create

model = KNeighborsClassifier(n_neighbors=3)



# Train

model.fit(train_x, train_y)



#Test

output = model.predict(test_x)

accuracy_score(output, test_y)