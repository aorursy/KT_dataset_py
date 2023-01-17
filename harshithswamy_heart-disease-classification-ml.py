import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt



from sklearn.preprocessing import MinMaxScaler

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix



from sklearn.svm import SVC

from sklearn.naive_bayes import GaussianNB

from sklearn.tree import DecisionTreeClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier
data_frame = pd.read_csv("/kaggle/input/heart-disease-uci/heart.csv")
data_frame.head()
data_frame.describe()
data_frame.info()
data_frame.rename(columns={"age": "Age", "sex": "Gender", "cp": "ChestPainType", "trestbps": "RestingBP",

                          "chol": "SerumCholestoral", "fbs": "FastingBP", "restecg": "RestingECG", "thalach": "MaxHeartRate",

                          "exang": "Excercise", "oldpeak": "Depression", "slope": "Slope", "ca": "Vessels", "thal": "Thalassemia",

                          "target": "Target"}, inplace=True)
data_frame.hist(figsize=(15,10))

plt.show()
plt.figure(figsize=(5, 5))

sns.countplot(x="Target", data=data_frame)

plt.title("Target Count Plot")

plt.show()
plt.figure(figsize=(10,5))

plt.subplot(1,2,1)

sns.countplot(x="ChestPainType", data=data_frame)

plt.title("Chest Pain Types Count")

plt.subplot(1,2,2)

sns.countplot(x="ChestPainType", hue="Target", data=data_frame)

plt.title("Chest Pain Type vs Target")

plt.show()
plt.figure(figsize=(10, 5))

sns.countplot(x="Excercise", hue="Target", data=data_frame)
plt.figure(figsize=(10, 5))

sns.countplot(x="Gender", hue="Target", data=data_frame, palette="bwr")

plt.title("Gender Count vs Target")

plt.show()
plt.figure(figsize=(10, 5))

sns.scatterplot(x="Age", y="MaxHeartRate", hue="Target", data=data_frame)

plt.title("Age vs Max Heart Rate")

plt.show()
plt.figure(figsize=(10,5))

sns.distplot(data_frame["Age"], bins=50)

plt.title("Age Ditribution")

plt.show()
young_age = data_frame[(data_frame["Age"] >= 29) & (data_frame["Age"] < 40)]

middle_age = data_frame[(data_frame["Age"] >= 40) & (data_frame["Age"] < 55)]

elder_age = data_frame[(data_frame["Age"] > 55)]
plt.figure(figsize=(15,10))

plt.subplot(2,2,1)

sns.barplot(x=["Young Age", "Middle Age", "Elder Age"], y=[len(young_age), len(middle_age), len(elder_age)], palette="rocket")

plt.title("Different Age Counts")

plt.subplot(2,2,2)

sns.countplot(x=young_age["Age"], hue=young_age["Target"], data=data_frame, palette="husl")

plt.title("Young Age Count Plot")

plt.subplot(2,2,3)

sns.countplot(x=middle_age["Age"], hue=middle_age["Target"], data=data_frame, palette="deep")

plt.title("Middle Age Count Plot")

plt.subplot(2,2,4)

sns.countplot(x=elder_age["Age"], hue=elder_age["Target"], data=data_frame, palette="hls")

plt.title("Elder Age Count Plot")

plt.show()
plt.figure(figsize=(12,10))

sns.heatmap(data_frame.corr(), linewidths=0.05, fmt= ".2f", cmap="YlGnBu", annot=True)

plt.title("Correlation Plot")

plt.show()
data_frame = pd.get_dummies(data_frame, columns=["ChestPainType", "RestingECG", "Excercise", "Vessels", "Slope", "Thalassemia"])
X = data_frame.drop("Target", axis=1)

Y = data_frame["Target"]
min_max_scaler = MinMaxScaler()

data_frame["RestingBP"] = min_max_scaler.fit_transform(data_frame["RestingBP"].values.reshape(-1, 1))

data_frame["Depression"] = min_max_scaler.fit_transform(data_frame["Depression"].values.reshape(-1, 1))

data_frame["MaxHeartRate"] = min_max_scaler.fit_transform(data_frame["MaxHeartRate"].values.reshape(-1, 1))

data_frame["SerumCholestoral"] = min_max_scaler.fit_transform(data_frame["SerumCholestoral"].values.reshape(-1, 1))
X_Scale = data_frame.drop("Target", axis=1)

Y_Scale = data_frame["Target"]
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

x_train_scale, x_test_scale, y_train_scale, y_test_scale = train_test_split(X_Scale, Y_Scale, test_size=0.2, random_state=42)
def model_evaluation(y_test, y_preds, model_name):

    

    accuracy = accuracy_score(y_test, y_preds) * 100

    print("************ Accuracy Score ************")

    print(accuracy)

    print("\n\n************ Classification Report ************")

    print(classification_report(y_test, y_preds))

    print("\n\n************ Confusion Matrix ************")

    sns.heatmap(confusion_matrix(y_preds, y_test), annot = True, fmt = ".0f", cmap = "YlGnBu")

    plt.title("{} Validation Matrix\n\n".format(model_name))

    plt.show()

    

    return accuracy
random_forest_model = RandomForestClassifier(criterion="entropy", max_depth=5)

random_forest_model.fit(x_train, y_train)

random_forest_pred = random_forest_model.predict(x_test)
rf_accuracy = model_evaluation(y_test, random_forest_pred, "Random Forest")
logistic_regression_model = LogisticRegression()

logistic_regression_model.fit(x_train, y_train)

lr_model_preds = logistic_regression_model.predict(x_test)
lr_accuracy = model_evaluation(y_test, lr_model_preds, "Logistic Regression")
svm_model = SVC(kernel="linear")

svm_model.fit(x_train, y_train)

svm_preds = svm_model.predict(x_test)
svm_accuracy = model_evaluation(y_test, svm_preds, "SVM")
decision_tree_model = DecisionTreeClassifier(criterion="entropy")

decision_tree_model.fit(x_train, y_train)

dt_preds = decision_tree_model.predict(x_test)
dt_accuracy = model_evaluation(y_test, dt_preds, "Decision Tree")
score_list = []

for i in range(1,20):

    knn_model = KNeighborsClassifier(n_neighbors=i)

    knn_model.fit(x_train_scale, y_train_scale)

    score_list.append(knn_model.score(x_test_scale, y_test_scale))

    

plt.plot(range(1,20), score_list)

plt.xticks(np.arange(1,20,1))

plt.xlabel("K value")

plt.ylabel("Score")

plt.show()



accuracy = max(score_list)

best_k = score_list.index(accuracy) + 1

print("Best KNN Model Score: {}".format(accuracy * 100))

print("Best K Value: {}".format(best_k))

knn_model = KNeighborsClassifier(n_neighbors = 3)

knn_model.fit(x_train_scale, y_train_scale)

knn_preds = knn_model.predict(x_test_scale)
knn_accuracy = model_evaluation(y_test_scale, knn_preds, "KNN")
x = ["Random Forest", "Logistic Regression", "Decision Tree", "SVM", "KNN"]

y = [rf_accuracy, lr_accuracy, dt_accuracy, svm_accuracy, knn_accuracy]

plt.bar(x=x, height=y)

plt.title("Algorithm Accuracy Comparison")

plt.xticks(rotation=15)

plt.xlabel("Algorithms")

plt.ylabel("Accuracy")

plt.show()