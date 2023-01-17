# Importing the required libraries

import pandas as pd, numpy as np

import matplotlib.pyplot as plt, seaborn as sns

%matplotlib inline
# Reading the csv file and putting it into 'df' object.

df = pd.read_csv(r"/kaggle/input/delhi-delights-data/DelhiDelightsData.csv")
df.columns
df.head(len(df))
df.shape
df.info()
df.describe()
plt.figure(figsize = (10,5))

ax= sns.violinplot(df['Average Delivery Rating (a1)'])

plt.show()
plt.figure(figsize = (15,5))

ax= sns.countplot(df['"Delighted Members" Purchase'])

for p in ax.patches:

    ax.annotate(str(p.get_height()), (p.get_x() * 1.01 , p.get_height() * 1.01))

plt.xticks(rotation = 45)

plt.show()
df['"Delighted Members" Purchase'].value_counts(ascending=False) * 100 / len(df)
plt.figure(figsize = (10,5))

ax= sns.violinplot(df['Average Orders per month (a2)'])

plt.show()
plt.figure(figsize = (10,5))

sns.violinplot(y = 'Average Delivery Rating (a1)', x = '"Delighted Members" Purchase', data = df)

plt.show()
plt.figure(figsize = (10,5))

sns.violinplot(y = 'Average Orders per month (a2)',x='Average Delivery Rating (a1)', 

               hue = '"Delighted Members" Purchase', split=True,data = df,inner="quartile")

plt.show()
plt.figure(figsize = (10,5))

sns.violinplot(y = 'Average Orders per month (a2)', x = '"Delighted Members" Purchase', data = df)

plt.show()
plt.figure(figsize = (10,5))

sns.heatmap(df.corr(), annot = True, cmap="rainbow")

plt.show()
# Putting feature variable to X

X = df.drop('"Delighted Members" Purchase',axis=1)



# Putting response variable to y

y = df['"Delighted Members" Purchase']
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=10)

X_train.shape, X_test.shape
from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier()

dt.fit(X, y)
from sklearn import tree

fig = plt.figure(figsize=(25,20))

_ = tree.plot_tree(dt,

                   feature_names=X.columns,

                   class_names=['Yes', "No"],

                   filled=True)
dt = DecisionTreeClassifier(random_state=7)

dt.fit(X_train, y_train)
from sklearn import tree

fig = plt.figure(figsize=(25,20))

_ = tree.plot_tree(dt,

                   feature_names=X.columns,

                   class_names=['Yes', "No"],

                   filled=True)
from sklearn.metrics import confusion_matrix, accuracy_score
def evaluate_model(dt_classifier):

    print("Train Accuracy :", round(accuracy_score(y_train, dt_classifier.predict(X_train)),2))

    print("Train Confusion Matrix:")

    print(confusion_matrix(y_train, dt_classifier.predict(X_train)))

    print("-"*50)

    print("Test Accuracy :", round(accuracy_score(y_test, dt_classifier.predict(X_test)),2))

    print("Test Confusion Matrix:")

    print(confusion_matrix(y_test, dt_classifier.predict(X_test)))
evaluate_model(dt)
dt = DecisionTreeClassifier(random_state=50)
from sklearn.model_selection import GridSearchCV
# Create the parameter grid based on the results of random search 

params = {

    'max_depth': [2, 3, 4,5, 6],

    'min_samples_leaf': [1,2,3,4,5],

    'criterion': ["gini", "entropy"]

}
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
fig = plt.figure(figsize=(25,10))

_ = tree.plot_tree(dt_best,

                   feature_names=X.columns,

                   class_names=['Yes', "No"],

                   filled=True)
evaluate_model(dt_best)
from sklearn.metrics import classification_report
print(classification_report(y_test, dt_best.predict(X_test)))
print(classification_report(y_train, dt_best.predict(X_train)))