import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, plot_precision_recall_curve, plot_confusion_matrix

from matplotlib import pyplot as plt

# seed the random
import random
random.seed(22)
!ls ../input/titanic
train_data = pd.read_csv("../input/titanic/train.csv", index_col="PassengerId")
train_data.head()
test_data = pd.read_csv("../input/titanic/test.csv", index_col="PassengerId")
test_data.head()
y_train_all = train_data.pop("Survived")
print("Shape of Train labels:", y_train_all.shape)
def extract_title(name):
    return name.split(",")[1].strip().split()[0].strip(".").strip()
train_data['Title'] = train_data.Name.map(extract_title)
test_data['Title'] = test_data.Name.map(extract_title)
X_train_no_names = train_data.drop(["Name"], axis=1)
X_test_no_names = test_data.drop(["Name"], axis=1)
print("Shape of Train Features:", X_train_no_names.shape)
print("Shape of Test Features:", X_test_no_names.shape)
X_train_no_names.head()
X_both_no_names = pd.concat([X_train_no_names, X_test_no_names])

X_both = pd.get_dummies(X_both_no_names)
X_train_all = X_both.iloc[:len(X_train_no_names),:]
X_test = X_both.iloc[len(X_train_no_names):,:]

X_train_all = X_train_all.fillna(0.0)
X_test = X_test.fillna(0.0)

X_train_all.head()
print("Train shape before and after one hot:",X_train_no_names.shape, " -> ", X_train_all.shape)
print("Test shape before and after one hot:",X_test_no_names.shape, " -> ", X_test.shape)
X_train, X_val, y_train, y_val = train_test_split(X_train_all, y_train_all, random_state=0)
X_train.shape, X_val.shape
model = RandomForestClassifier(verbose=1, criterion='entropy')
model.fit(X_train, y_train)
y_pred = model.predict(X_val)
print("Accuracy :", accuracy_score(y_val, y_pred))
print("Precision:", precision_score(y_val, y_pred))
print("Recall   :", recall_score(y_val, y_pred))
print("F1 Score :", f1_score(y_val, y_pred))
fig, ax = plt.subplots(1,2, sharey=True, figsize=(10,5))
plot_precision_recall_curve(model, X_train, y_train, ax=ax[0])
plot_precision_recall_curve(model, X_val, y_val, ax=ax[1])
ax[0].set_title("Training")
ax[1].set_title("Validation")
plt.plot()
fig, ax = plt.subplots(1,2, sharey=True, figsize=(10,5))
plot_confusion_matrix(model, X_train, y_train, ax=ax[0])
plot_confusion_matrix(model,X_val, y_val, ax=ax[1])
ax[0].set_title("Training")
ax[1].set_title("Validation")
plt.plot()
prediction = model.predict(X_test)
result = pd.DataFrame(prediction, index=X_test.index, columns=["Survived"])
result
result.to_csv("result.csv")
