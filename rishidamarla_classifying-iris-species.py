# Importing all necessary libraries.

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt
df = pd.read_csv('../input/iris/Iris.csv')
df.head()
X = df.iloc[:, :-1]

y = df.iloc[:, -1]
# Visualizing the effects of each feature.

plt.xlabel('Features')

plt.ylabel('Species')



pltX = df.loc[:, 'SepalLengthCm']

pltY = df.loc[:, 'Species']

plt.scatter(pltX, pltY, color = 'blue', Label = 'SepalLengthCm')



pltX = df.loc[:, 'SepalWidthCm']

pltY = df.loc[:, 'Species']

plt.scatter(pltX, pltY, color = 'green', Label = 'SepalWidthCm')



pltX = df.loc[:, 'PetalLengthCm']

pltY = df.loc[:, 'Species']

plt.scatter(pltX, pltY, color = 'red', Label = 'PetalLengthCm')



pltX = df.loc[:, 'PetalWidthCm']

pltY = df.loc[:, 'Species']

plt.scatter(pltX, pltY, color = 'black', Label = 'PetalWidthCm')



plt.legend(loc=4, prop={'size':7.5})

plt.show()
# Splitting the data into training and testing sets.

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
# Implementing Logistic Regression.

from sklearn.linear_model import LogisticRegression

model = LogisticRegression()

model.fit(x_train, y_train)
predictions = model.predict(x_test)

print(predictions)
print(y_test)
# Viewing how well the model performed.

from sklearn.metrics import classification_report

from sklearn.metrics import accuracy_score

print(classification_report(y_test, predictions))
# Viewing the accuracy of the model.

print(accuracy_score(y_test, predictions))