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
df.describe()
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
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.6, random_state=42)

X_train.shape, X_test.shape
from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier(max_depth=3)

dt.fit(X_train, y_train)


from sklearn import tree

text_representation = tree.export_text(dt)

print(text_representation)


fig = plt.figure(figsize=(25,20))

_ = tree.plot_tree(dt,

                   feature_names=X.columns,

                   class_names=['No Disease', "Disease"],

                   filled=True)
import graphviz

# DOT data

dot_data = tree.export_graphviz(dt, out_file=None, 

                                feature_names=X.columns, 

                                class_names=['No Disease', "Disease"],

                                filled=True)



# Draw graph

graph = graphviz.Source(dot_data, format="png") 

graph
y_train_pred = dt.predict(X_train)

y_test_pred = dt.predict(X_test)
from sklearn.metrics import confusion_matrix, accuracy_score,classification_report
print(accuracy_score(y_train, y_train_pred))

confusion_matrix(y_train, y_train_pred)
print(accuracy_score(y_test, y_test_pred))

confusion_matrix(y_test, y_test_pred)


print (classification_report(y_train, y_train_pred))
print (classification_report(y_test, y_test_pred))