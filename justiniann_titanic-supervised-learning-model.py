import os

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier

TRAIN_FILE = os.path.join("..", "input", "train.csv")
TEST_FILE = os.path.join("..", "input", "test.csv")

# Training Data
training_dataset = pd.read_csv(TRAIN_FILE)
training_labels_preprocessed = training_dataset["Survived"]
training_dataset = training_dataset.drop(["Survived"], axis=1)

# Test Data
testing_dataset = pd.read_csv(TEST_FILE)
def preprocess(dataset):
    # remove name and passengerid, they are not needed for training
    # todo should name/ticket be removed
    dataset = dataset.drop(["Name", "PassengerId", "Ticket"], axis=1)

    # encode gender colunm
    encoder = LabelEncoder()
    dataset['Sex'] = encoder.fit_transform(dataset['Sex'])

    # one-hot encode embarked column
    dataset = pd.get_dummies(dataset, drop_first=True, columns=["Embarked"])

    # create a new feature, hasCabin
    dataset['Cabin'] = dataset['Cabin'].notnull().astype('int')

    # create a new feature, family size
    dataset['FamilySize'] = dataset['Parch'] + dataset['SibSp']

    # Fill in null values for 'Age' and 'Fare' column with mean
    # TODO explore better ways
    average_age = dataset['Age'].mean()
    dataset['Age'] = dataset['Age'].fillna(average_age)

    average_fare = dataset['Fare'].mean()
    dataset['Fare'] = dataset['Fare'].fillna(average_fare)

    return dataset

# get the labels for the data separated
training_data_preprocessed = preprocess(training_dataset)

# split training into training/validation set
training_data, validation_data, training_labels, validation_labels = train_test_split(training_data_preprocessed.values, training_labels_preprocessed)
def evaluate_model(model, data, labels):
    predictions = model.predict(data)
    return {"accuracy": accuracy_score(labels, predictions)}
print("Training Decision Tree Classifier on Full Dataset...")
model = DecisionTreeClassifier()
model.fit(training_data, training_labels)
print("Results are: {}\n".format(evaluate_model(model, validation_data, validation_labels)))
importance_map = {}
for i in range(0, len(training_data_preprocessed.columns)):
    importance_map[training_data_preprocessed.columns[i]] = model.feature_importances_[i]

margin = 0.05
labels_to_trim = [key for key, val in importance_map.items() if val < margin]
if labels_to_trim:
    print(
        "The following labels have a feature importance under {} and will be trimmed from the dataset: {}\n".format(
            margin, labels_to_trim))
    training_data_preprocessed = training_data_preprocessed.drop(labels_to_trim, axis=1)
    training_data, validation_data, training_labels, validation_labels = train_test_split(training_data_preprocessed.values, training_labels_preprocessed)
print("Training Decision Tree Classifier on Trimmed Dataset...")
model = DecisionTreeClassifier()
model.fit(training_data, training_labels)
print("Results are: {}\n".format(evaluate_model(model, validation_data, validation_labels)))
print("Training Random Forrest Model...")
ensemble_model = RandomForestClassifier(n_estimators=100)
ensemble_model.fit(training_data, training_labels)
print("Results are: {}\n".format(evaluate_model(ensemble_model, validation_data, validation_labels)))
testing_dataset = pd.read_csv(TEST_FILE)
testing_data_preprocessed = preprocess(testing_dataset)
if labels_to_trim:
    testing_data_preprocessed = testing_data_preprocessed.drop(labels_to_trim, axis=1)
model_predictions = ensemble_model.predict(testing_data_preprocessed.values)
i = 0
with open("result.csv", 'w+') as result_file:
    result_file.write("PassengerId,Survived\n")
    while i < len(model_predictions):
        result_file.write("{},{}\n".format(testing_dataset['PassengerId'][i], model_predictions[i]))
        i += 1