# Import statements

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
# Load the dataset and review the first rows

df = pd.read_csv("../input/mushroom-classification/mushrooms.csv")
df.head()
df.describe()
df["class"].value_counts()
df.isnull().any()
# Preparing class (the dependent variable or 'y')

y = pd.DataFrame(df["class"])
y.replace("e", 0, inplace=True)
y.replace("p", 1, inplace=True)
y = y.values.flatten()
y
# Preparing features (the independent variables or 'x' values)

x = df.copy(deep=True)
x.drop("class", axis = 1, inplace = True)
x = pd.get_dummies(x)
x.head()
# Splitting x and y into training and testing sets

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state = 10, stratify = y)
# Using RandomizedSearchCV to identify effective parameter values

# Range of parameters to sample from
n_estimators = list(range(10, 1000, 10))
max_features = ["auto", "sqrt", "log2"]
max_depth = list(range(10, 1000, 10))
max_depth.append(None)
min_samples_split = list(range(2, 100, 10))
min_samples_leaf = list(range(1, 100, 10))
bootstrap = [True, False]

param_grid = {"n_estimators": n_estimators,
               "max_features": max_features,
               "max_depth": max_depth,
               "min_samples_split": min_samples_split,
               "min_samples_leaf": min_samples_leaf,
               "bootstrap": bootstrap}

rfc = RandomForestClassifier()
rfc_search = RandomizedSearchCV(estimator=rfc, param_distributions=param_grid, n_iter=100, cv=5, random_state=10)
rfc_search = rfc_search.fit(x_train, y_train)

top_params = rfc_search.best_params_
print(top_params)
# Using GridSearchCV to identify effective parameter values (range informed by RandomizedSearchCV results)

# Range of parameters to sample from
n_estimators = [70, 80, 90, 100]
max_features = ["sqrt"]
max_depth = [270, 300, 330, 360]
max_depth.append(None)
min_samples_split = [15, 20, 25, 30]
min_samples_leaf = [1, 2, 3, 4]
bootstrap = [True]

param_grid = {"n_estimators": n_estimators,
               "max_features": max_features,
               "max_depth": max_depth,
               "min_samples_split": min_samples_split,
               "min_samples_leaf": min_samples_leaf,
               "bootstrap": bootstrap}

rfc = RandomForestClassifier()
rfc_gsearch = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=5)
rfc_gsearch = rfc_gsearch.fit(x_train, y_train)

top_gparams = rfc_gsearch.best_params_
top_gparams
# Train and fit the classifier on the top performing parameters

n_estimators = top_gparams["n_estimators"]
max_features = top_gparams["n_estimators"]
max_depth = top_gparams["max_depth"]
min_samples_split = top_gparams["min_samples_split"]
min_samples_leaf = top_gparams["min_samples_leaf"]
bootstrap = [True]

rfc = RandomForestClassifier(n_estimators=n_estimators, 
                             max_features=max_features,
                             max_depth=max_depth,
                             min_samples_split=min_samples_split,
                             min_samples_leaf=min_samples_leaf,
                             bootstrap=bootstrap)

rfc = rfc.fit(x_train, y_train)
# Use the classifer to predict the y test values and evaluate the outcome

y_pred = rfc.predict(x_test)
confusion_matrix(y_test, y_pred) 
accuracy_score(y_test, y_pred) 
# Create a dataframe of feature importance scores
feature_scores = pd.Series(rfc.feature_importances_, index=x.columns).sort_values(ascending=False)
feature_scores = feature_scores[0:15]
c = feature_scores.values

# Generate a plot to visualise the most important features
ax = feature_scores.plot(kind="barh", color="#21c8d9", figsize=(10,6), width=0.7)
ax.set_ylabel("Features", labelpad=15, size=14)
plt.yticks(size=12)
ax.set_xlabel("Feature Score", labelpad=15, size=14)
plt.xticks(size=12)
ax.set_title("Feature Importance", size=16)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_color('#DDDDDD')
ax.spines['left'].set_color('#DDDDDD')
ax.tick_params(bottom=False, left=False)
plt.tight_layout()
plt.show()
# Plot mushroom odour by class
ax = sns.countplot(x="odor", hue="class", data=df, palette={"e":"#5f60f2", "p":"#21c8d9"})
ax.set_ylabel("Frequency", labelpad=15, size=14)
plt.yticks(size=12)
ax.set_xlabel("Mushroom Odour", labelpad=15, size=14)
plt.xticks(size=12)
ax.set_title("Mushroom Odour by Class", size=16)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_color('#DDDDDD')
ax.spines['left'].set_color('#DDDDDD')
ax.tick_params(bottom=False, left=False)
plt.legend(title="Class", loc="upper left")
plt.tight_layout()
plt.show()
# Plot mushroom gill size by class
ax = sns.countplot(x="gill-size", hue="class", data=df, palette={"e":"#5f60f2", "p":"#21c8d9"})
ax.set_ylabel("Frequency", labelpad=15, size=14)
plt.yticks(size=12)
ax.set_xlabel("Gill size", labelpad=15, size=14)
plt.xticks(size=12)
ax.set_title("Mushroom Gill Size by Class", size=16)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_color('#DDDDDD')
ax.spines['left'].set_color('#DDDDDD')
ax.tick_params(bottom=False, left=False)
plt.legend(title="Class", loc="upper left")
plt.tight_layout()
plt.show()