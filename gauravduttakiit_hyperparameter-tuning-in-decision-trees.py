# Importing the required libraries

import pandas as pd, numpy as np

import matplotlib.pyplot as plt, seaborn as sns

%matplotlib inline
# Reading the csv file and putting it into 'df' object.

df = pd.read_csv(r"/kaggle/input/heart-disease-prediction/heart_v2.csv")
df.columns
df.head()
df.shape
df.info()
plt.figure(figsize = (10,5))

ax= sns.violinplot(df['age'])

plt.show()
plt.figure(figsize = (15,5))

ax= sns.countplot(df['sex'])

for p in ax.patches:

    ax.annotate(str(p.get_height()), (p.get_x() * 1.01 , p.get_height() * 1.01))

plt.xticks(rotation = 45)

plt.show()

plt.figure(figsize = (10,5))

ax= sns.violinplot(df['BP'])

plt.show()
percentiles = df['BP'].quantile([0.05,0.95]).values

df['BP'][df['BP'] <= percentiles[0]] = percentiles[0]

df['BP'][df['BP'] >= percentiles[1]] = percentiles[1]
plt.figure(figsize = (10,5))

ax= sns.violinplot(df['BP'])

plt.show()
plt.figure(figsize = (10,5))

ax= sns.violinplot(df['cholestrol'])

plt.show()
percentiles = df['cholestrol'].quantile([0.05,0.95]).values

df['cholestrol'][df['cholestrol'] <= percentiles[0]] = percentiles[0]

df['cholestrol'][df['cholestrol'] >= percentiles[1]] = percentiles[1]
plt.figure(figsize = (10,5))

ax= sns.violinplot(df['cholestrol'])

plt.show()
plt.figure(figsize = (15,5))

ax= sns.countplot(df['heart disease'])

for p in ax.patches:

    ax.annotate(str(p.get_height()), (p.get_x() * 1.01 , p.get_height() * 1.01))

plt.xticks(rotation = 45)

plt.show()
plt.figure(figsize = (10,5))

sns.violinplot(y = 'age', x = 'heart disease', data = df)

plt.show()
plt.figure(figsize = (10,5))

ax= sns.countplot(x = "sex", hue = "heart disease", data = df)

for p in ax.patches:

    ax.annotate(str(p.get_height()), (p.get_x() * 1.01 , p.get_height() * 1.01))

plt.xticks(rotation = 90)

plt.show()
plt.figure(figsize = (10,5))

sns.violinplot(y = 'BP', x = 'heart disease', data = df)

plt.show()
plt.figure(figsize = (10,5))

sns.violinplot(y = 'cholestrol', x = 'heart disease', data = df)

plt.show()
plt.figure(figsize = (10,5))

sns.heatmap(df.corr(), annot = True, cmap="rainbow")

plt.show()
df.describe()
# Putting feature variable to X

X = df.drop('heart disease',axis=1)



# Putting response variable to y

y = df['heart disease']
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.6, random_state=50)

X_train.shape, X_test.shape
from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier(max_depth=3)

dt.fit(X_train, y_train)
from sklearn import tree

fig = plt.figure(figsize=(25,20))

_ = tree.plot_tree(dt,

                   feature_names=X.columns,

                   class_names=['No Disease', "Disease"],

                   filled=True)
y_train_pred = dt.predict(X_train)

y_test_pred = dt.predict(X_test)
from sklearn.metrics import confusion_matrix, accuracy_score
print(accuracy_score(y_train, y_train_pred))

confusion_matrix(y_train, y_train_pred)
print(accuracy_score(y_test, y_test_pred))

confusion_matrix(y_test, y_test_pred)
def get_dt_graph(dt_classifier):

    fig = plt.figure(figsize=(25,20))

    _ = tree.plot_tree(dt_classifier,

                       feature_names=X.columns,

                       class_names=['No Disease', "Disease"],

                       filled=True)
def evaluate_model(dt_classifier):

    print("Train Accuracy :", accuracy_score(y_train, dt_classifier.predict(X_train)))

    print("Train Confusion Matrix:")

    print(confusion_matrix(y_train, dt_classifier.predict(X_train)))

    print("-"*50)

    print("Test Accuracy :", accuracy_score(y_test, dt_classifier.predict(X_test)))

    print("Test Confusion Matrix:")

    print(confusion_matrix(y_test, dt_classifier.predict(X_test)))
dt_default = DecisionTreeClassifier(random_state=42)

dt_default.fit(X_train, y_train)
gph = get_dt_graph(dt_default)

evaluate_model(dt_default)
dt_depth = DecisionTreeClassifier(max_depth=3)

dt_depth.fit(X_train, y_train)
gph = get_dt_graph(dt_depth) 
evaluate_model(dt_depth)
dt_min_split = DecisionTreeClassifier(min_samples_split=20)

dt_min_split.fit(X_train, y_train)
gph = get_dt_graph(dt_min_split) 
evaluate_model(dt_min_split)
dt_min_leaf = DecisionTreeClassifier(min_samples_leaf=20, random_state=42)

dt_min_leaf.fit(X_train, y_train)
gph = get_dt_graph(dt_min_leaf)
evaluate_model(dt_min_leaf)
dt_min_leaf_entropy = DecisionTreeClassifier(min_samples_leaf=20, random_state=42, criterion="entropy")

dt_min_leaf_entropy.fit(X_train, y_train)
gph = get_dt_graph(dt_min_leaf_entropy)
evaluate_model(dt_min_leaf_entropy)
dt = DecisionTreeClassifier(random_state=42)
from sklearn.model_selection import GridSearchCV
# Create the parameter grid based on the results of random search 

params = {

    'max_depth': [2, 3, 5, 10, 20],

    'min_samples_leaf': [5, 10, 20, 50, 100],

    'criterion': ["gini", "entropy"]

}
# grid_search = GridSearchCV(estimator=dt, 

#                            param_grid=params, 

#                            cv=4, n_jobs=-1, verbose=1, scoring = "f1")
# Instantiate the grid search model

grid_search = GridSearchCV(estimator=dt, 

                           param_grid=params, 

                           cv=4, n_jobs=-1, verbose=1, scoring = "accuracy")
%%time

grid_search.fit(X_train, y_train)
score_df = pd.DataFrame(grid_search.cv_results_)

score_df.head()
score_df.nlargest(5,"mean_test_score")
grid_search.best_estimator_
dt_best = grid_search.best_estimator_
evaluate_model(dt_best)
get_dt_graph(dt_best)
from sklearn.metrics import classification_report
print(classification_report(y_test, dt_best.predict(X_test)))