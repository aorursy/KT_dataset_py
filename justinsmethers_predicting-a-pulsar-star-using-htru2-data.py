# Import necessary packages

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler

from sklearn.neural_network import MLPClassifier

from sklearn.model_selection import train_test_split

from sklearn.metrics import classification_report, confusion_matrix
df = pd.read_csv('../input/pulsar_stars.csv')

df.info()
plt.figure(figsize=(10, 7))

sns.heatmap(df.corr(), annot=True, cmap=sns.color_palette('coolwarm', 15))

plt.show
# Choos data set. Exclude the target set from the data

X = df.drop('target_class', axis=1)

# Choose target set

y = df['target_class']



# Split data into random training and testing sets

X_train, X_test, y_train, y_test = train_test_split(X, y)



# Create an object of the standard scaler from sklearn

scaler = StandardScaler()

# Fit the standard scalar to the training data

scaler.fit(X_train)



# Use the transform function to normalize the data sets

X_train = scaler.transform(X_train)

X_test = scaler.transform(X_test)
# Create the MLP model with 2 layers, and as many neurons as there are features in the data set

mlp = MLPClassifier(hidden_layer_sizes=(8, 8))



# Fit the training data to the MLP model

mlp.fit(X_train, y_train)
# Make predictions on the data set using the MLP model

prediction = mlp.predict(X_test)



# Print results and analytics of the prediction

plt.figure(figsize=(10,7))

sns.heatmap(confusion_matrix(y_test, prediction), annot=True, fmt='.1f', cmap=sns.color_palette("Blues", 25))

plt.show()



print(confusion_matrix(y_test, prediction))

print(classification_report(y_test, prediction))