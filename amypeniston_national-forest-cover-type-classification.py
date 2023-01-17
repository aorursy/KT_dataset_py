import os

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import ExtraTreesClassifier

from sklearn.model_selection import RandomizedSearchCV, KFold, cross_val_score

from sklearn.metrics import classification_report, accuracy_score



for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
train_df = pd.read_csv("../input/learn-together/train.csv")

test_df = pd.read_csv("../input/learn-together/test.csv")
train_df.shape, test_df.shape
train_df.head()
train_df.dtypes.unique()
target_name = "Cover_Type"

feature_names = [n for n in train_df.columns.values if n != target_name and n != "Id"]
print("Missing values in training data, target column?", train_df[target_name].isnull().any().sum())

print("Missing values in training data, feature columns?", train_df[feature_names].isnull().any().sum())

print("Missing values in test data?", test_df.isnull().any().sum())
len(train_df.select_dtypes(include=["object"]).columns)
w = train_df[["Wilderness_Area1", "Wilderness_Area2", "Wilderness_Area3", "Wilderness_Area4", "Cover_Type"]].groupby(["Cover_Type"]).sum()

w_percentages = w/w.sum()

w_percentages
labels = ["Wilderness_Area1", "Wilderness_Area2", "Wilderness_Area3", "Wilderness_Area4"]

colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']



x = np.arange(len(w_percentages.columns.values))



plt.figure(figsize=(16,8))



for i in range(len(w_percentages.index.values)):

    if i == 0:

        running_sum = 0

    else:

        running_sum += w_percentages.iloc[i-1,:]

    plt.bar(x, w_percentages.iloc[i,:], color=colors[i], bottom=running_sum)



plt.xticks(x, labels);



plt.legend(["1 - Spruce/Fir",

"2 - Lodgepole Pine",

"3 - Ponderosa Pine",

"4 - Cottonwood/Willow",

"5 - Aspen",

"6 - Douglas-fir",

"7 - Krummholz"], loc="center", bbox_to_anchor=(0.5,-0.2), ncol=2);
plt.figure(figsize=(16,8))

for i in range(1, 7):

    sns.distplot(train_df["Elevation"][train_df["Cover_Type"] == i]);



plt.legend(["1 - Spruce/Fir",

"2 - Lodgepole Pine",

"3 - Ponderosa Pine",

"4 - Cottonwood/Willow",

"5 - Aspen",

"6 - Douglas-fir",

"7 - Krummholz"], loc="center", bbox_to_anchor=(0.5,-0.2), ncol=2);

plt.title("Elevation Density by Cover Type");

plt.ylabel("Density");
train_df["Cover_Type"].value_counts()
train = train_df.copy()

test = test_df.copy()



y = train["Cover_Type"]

train.drop(columns=["Id", "Cover_Type"], inplace=True, axis=1)

test_ids = test["Id"]

test.drop(columns=["Id"], inplace=True, axis=1)
seed = 0
X_train, X_val, y_train, y_val = train_test_split(train,

                                                  y,

                                                  test_size = 0.2,

                                                  random_state = seed)

X_train.shape, X_val.shape, y_train.shape, y_val.shape
rf = RandomForestClassifier(n_estimators = 100,

                                            random_state = seed)



rf.fit(X_train, y_train)

importances = rf.feature_importances_

indices = np.argsort(importances)[::-1]



feature_importances = pd.DataFrame({"Feature": X_train.columns,

                                   "Importance": importances}).sort_values("Importance", ascending=False)



rf_predictions = rf.predict(X_val)
accuracy_score(y_val, rf_predictions)
plt.figure(figsize=(16,8))

sns.barplot(x="Feature", y="Importance", data=feature_importances)

plt.title("Feature Importances for Random Forest Model")

plt.xticks(rotation="vertical")

plt.show()
def generate_test_predictions(model):

    predictions = model.predict(X_test)

    output = pd.DataFrame({"ID": test_ids, "Cover_Type": predictions})

    

    return output

    

output = generate_test_predictions(rf)

output.to_csv("submission_rf100.csv", index=False)

output.head()
rf.get_params()
n_estimators = [int(x) for x in np.linspace(100, 2000, num=20)]



random_param_grid = {"n_estimators": n_estimators}



model = RandomForestClassifier()



rf_random_search = RandomizedSearchCV(estimator=model, param_distributions=random_param_grid,

                                     n_iter=20, cv=3, verbose=2, random_state=seed, n_jobs=-1)



rf_random_search.fit(train, y)
rf_random_search.best_params_