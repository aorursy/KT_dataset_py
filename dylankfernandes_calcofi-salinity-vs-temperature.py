import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import preprocessing, svm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Setting up our datasets
bottle = pd.read_csv('../input/bottle.csv')
bottle = bottle[['Salnty', 'T_degC']]
bottle.columns = ['Sal', 'Temp']

# Limiting amount of entries to speed up regression time
bottle = bottle[:][:500]

print(bottle.head())
# This produces a scatter 
sns.lmplot(x="Sal", y="Temp", data=bottle,
           order=2, ci=None);
# Picturing a residual plot to check for heteroscedasticity 
sns.residplot(bottle['Sal'], bottle['Temp'], order=2, lowess=True)
# Eliminating NaN or missing input numbers
bottle.fillna(method='ffill', inplace=True)
# Set up the training data
X = np.array(bottle['Sal']).reshape(-1, 1)
y = np.array(bottle['Temp']).reshape(-1, 1)

bottle.dropna(inplace=True)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

clf = LinearRegression()
clf.fit(X_train, y_train)
accuracy = clf.score(X_test, y_test)
print(accuracy)
# Make predictions using the new model
y_pred = clf.predict(X_test)
plt.scatter(X_test, y_test, color='b')
plt.plot(X_test, y_pred, color='k')
plt.show()
