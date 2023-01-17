import numpy as np

# This is John's data
data = np.array([[0, 1, 5, 1, 0],  # record of John's 1st food delivery
                [1, 0, 7, 0, 1],   # record of John's 2nd food delivery
                [0, 1, 2, 1, 0],   # record of John's 3rd food delivery
                [1, 1, 4.2, 1, 0], # record of John's 4th food delivery
                [0, 0, 7.8, 0, 1], # ...
                [1, 0, 3.9, 1, 0],
                [0, 1, 4, 1, 0],
                [1, 1, 2, 0, 0],
                [0, 0, 3.5, 0, 1],
                [1, 0, 2.6, 1, 0],
                [0, 0, 4.1, 0, 1],
                [0, 1, 1.5, 0, 1],
                [1, 1, 1.75, 1, 0],
                [1, 0, 1.3, 0, 0],
                [1, 1, 2.1, 0, 0],
                [1, 1, 0.2, 1, 0],
                [1, 1, 5.2, 0, 1],
                [0, 1, 2, 1, 0],
                [1, 0, 5.5, 0, 1],
                [0, 0, 2, 1, 0],
                [1, 1, 1.7, 0, 0],
                [0, 1, 3, 1, 1],
                [1, 1, 1.9, 1, 0],
                [0, 1, 3.1, 0, 1],
                [0, 1, 2.3, 0, 0],
                [0, 0, 1.1, 1, 0],
                [1, 1, 2.5, 1, 1],
                [1, 1, 5, 0, 1],
                [1, 0, 7.5, 1, 1],
                [0, 0, 0.5, 1, 0],
                [0, 0, 1.5, 1, 0],
                [1, 0, 3.2, 1, 0],
                [0, 0, 2.15, 1, 0],
                [1, 1, 4.2, 0, 1],
                [1, 0, 6.5, 0, 1],
                [1, 0, 0.5, 0, 0],
                [0, 0, 3.5, 0, 1],
                [0, 0, 1.75, 0, 0],
                [1, 1, 5, 0, 1],
                [0, 0, 2, 1, 0],
                [0, 1, 1.3, 1, 1],
                [0, 1, 0.2, 0, 0],
                [1, 1, 2.2, 0, 0],
                [0, 1, 1.2, 1, 0],
                [1, 1, 4.2, 0, 1]])

print(data)
import pandas as pd

# Create the dataframe with this data, labeling the columns
delivery_data = pd.DataFrame(data, columns=["bad_weather", "is_rush_hour", "mile_distance", "urban_address", "late"])

# Print the first 5 rows
delivery_data.head(15)
input_data = delivery_data[["bad_weather", "is_rush_hour", "mile_distance", "urban_address"]]
target = delivery_data["late"]
from sklearn.neighbors import KNeighborsClassifier

# Use n_neighbors = 1
# This means the KNN will consider the "closest" record to make a decision.
classifier = KNeighborsClassifier(n_neighbors = 1)

# Fit the model to our data
classifier.fit(input_data, target)
import numpy as np

some_data = np.array([[0, 0, 2.1, 1]]) # bad_weather->0, is_rush_hour->0, mile_distance->2.1 and urban_address->1

# Use the fitted model to make predictions on new data
print(classifier.predict(some_data))
import numpy as np

some_data = np.array([[0, 0, 2.1, 1], # bad_weather->0, is_rush_hour->0, mile_distance->2.1 and urban_address->1
                 [0, 1, 5, 0],   # bad_weather->0, is_rush_hour->1, mile_distance->5.0 and urban_address->0
                 [1, 1, 3.1, 1]  # bad_weather->1, is_rush_hour->1, mile_distance->3.1 and urban_address->1
                ])

# Use the fitted model to make predictions on more new data
print(classifier.predict(some_data))
# Use the fitted model to make predictions on our training dataset
predictions = classifier.predict(input_data)

from sklearn.metrics import confusion_matrix

print(confusion_matrix(target, predictions))
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score

print(classification_report(target, predictions))

print("Accuracy:", accuracy_score(target, predictions))

delivery_data.shape
# Let's split our data into two sets: Training (85%) and Test (15%)
# This gives us 38 training records and 7 test records (total 45 records)

training_data = delivery_data.iloc[:38, :] # First 38
test_data = delivery_data.iloc[38:, :] # Remaining

# Print the first 5 rows
training_data.head()
from sklearn.neighbors import KNeighborsClassifier

X_train = training_data[["bad_weather", "is_rush_hour", "mile_distance", "urban_address"]].values
y_train = training_data["late"].values

# Use n_neighbors = 1
# This means the KNN will consider two other "closest" records to make a decision.
classifier = KNeighborsClassifier(n_neighbors = 1)

# Fit the model to our training data
classifier.fit(X_train, y_train)
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

# Use the fitted model to make predictions on the same dataset we trained the model on
train_predictions = classifier.predict(X_train)

print('Model evaluation on the training set: \n')
print(confusion_matrix(y_train, train_predictions))
print(classification_report(y_train, train_predictions))
print("Training accuracy:", accuracy_score(y_train, train_predictions))

from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

X_test = test_data[["bad_weather", "is_rush_hour", "mile_distance", "urban_address"]].values
y_test = test_data["late"].tolist()

# Use the fitted model to make predictions on the test dataset
test_predictions = classifier.predict(X_test)

print('Model evaluation on the training set: \n')
print(confusion_matrix(y_test, test_predictions))
print(classification_report(y_test, test_predictions))
print("Training accuracy:", accuracy_score(y_test, test_predictions))

# Let's further split our training data into two sets: Training (80%) and Validation (20%)
# This gives us 30 training records and 8 test records 

train_data = training_data.iloc[:30, :] # First 30
val_data = training_data.iloc[30:, :] # Remaining

X_train = train_data[["bad_weather", "is_rush_hour", "mile_distance", "urban_address"]].values
y_train = train_data["late"].tolist()

X_val = val_data[["bad_weather", "is_rush_hour", "mile_distance", "urban_address"]].values
y_val =val_data["late"].tolist()

K_values = [1, 2, 3, 4, 5, 6]

for K in K_values:
    classifier = KNeighborsClassifier(n_neighbors = K)
    classifier.fit(X_train, y_train)
    val_predictions = classifier.predict(X_val)
    print("K=%d, Validation accuracy score: %f" % (K, accuracy_score(y_val, val_predictions)))
classifier = KNeighborsClassifier(n_neighbors = 4)
classifier.fit(X_train, y_train)
test_predictions = classifier.predict(X_test)
print("Test accuracy score: %f" % (accuracy_score(y_test, test_predictions)))
