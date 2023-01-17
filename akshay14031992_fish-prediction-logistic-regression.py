import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
sns.set()
data = pd.read_csv(os.path.join(dirname, filename))
data.describe(include='all')
x_train, x_test, y_train, y_test = train_test_split(data.drop(columns=['Species']), data[["Species"]], test_size = 0.2, random_state = 14)
logistic_classifier = LogisticRegression(random_state = 14)
logistic_classifier.fit(x_test, y_test)
logistic_classifier.score(x_test, y_test)
logistic_classifier.score(x_train, y_train)
success_counter = 0
for i in range(len(data)):
    species = data.loc[i, "Species"]
    weight = data.loc[i, "Weight"]
    height = data.loc[i, "Height"]
    width = data.loc[i, "Width"]
    length_1 = data.loc[i, "Length1"]
    length_2 = data.loc[i, "Length2"]
    length_3 = data.loc[i, "Length3"]

    if logistic_classifier.predict([[weight, length_1, length_2, length_3, height, width]])[0] == species:
        success_counter += 1

success_percent = (success_counter/len(data))*100
print("Model Accuracy: " + str(round(success_percent, 2)) + " %")
sns.distplot(data['Weight'])
x_train_imp, x_test_imp, y_train_imp, y_test_imp = train_test_split(data.drop(columns=['Species', 'Weight']), data[["Species"]], test_size = 0.2, random_state = 14)
logistic_classifier = LogisticRegression(random_state = 14)
logistic_classifier.fit(x_test_imp, y_test_imp)
logistic_classifier.score(x_test_imp, y_test_imp)
logistic_classifier.score(x_train_imp, y_train_imp)
success_counter = 0
for i in range(len(data)):
    species = data.loc[i, "Species"]
    weight = data.loc[i, "Weight"]
    height = data.loc[i, "Height"]
    width = data.loc[i, "Width"]
    length_1 = data.loc[i, "Length1"]
    length_2 = data.loc[i, "Length2"]
    length_3 = data.loc[i, "Length3"]

    if logistic_classifier.predict([[length_1, length_2, length_3, height, width]])[0] == species:
        success_counter += 1

success_percent = (success_counter/len(data))*100
print("Model Accuracy: " + str(round(success_percent, 2)) + " %")
