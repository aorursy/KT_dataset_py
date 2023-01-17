import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
df = pd.read_csv("../input/train.csv")
# I set the name of the column to be dropped from the data set and then drop it.
dropList = ["Ticket", "Name", "Cabin", "Embarked", "Age", "PassengerId"]
df.drop(dropList, axis=1, inplace=True)
df.info()
df['Sex'] = pd.DataFrame(np.where(df['Sex'] == 'male',1,0))
df["Fare"] = df["Fare"].fillna(0.0).apply(np.int64)
df.info()
X = df.drop(["Survived"],axis=1)
y = df.Survived
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score 
from sklearn.metrics import confusion_matrix as cm
import matplotlib.pyplot as plt

# For Visualization
from sklearn.tree import export_graphviz
from sklearn import tree
from IPython.display import SVG
from graphviz import Source
from IPython.display import display
from sklearn import ensemble
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3, random_state=42)
d_tree1 = DecisionTreeClassifier(max_depth = 2, random_state=42)
d_tree1.fit(X_train, y_train)
predictions = d_tree1.predict(X_test)
score = round(accuracy_score(y_test, predictions), 3)
cm1 = cm(y_test, predictions)
print("Confisuon Matrix:\n", cm1)
print("Our model score:", score)
graph = Source(tree.export_graphviz(d_tree1, out_file=None,feature_names=X.columns, 
                                    filled = True))
display(SVG(graph.pipe(format='svg')))
plt.figure(figsize=(16, 9))

d_tree2 = DecisionTreeClassifier(max_depth = 8, random_state=42)
d_tree2.fit(X_train, y_train)
ranking = d_tree2.feature_importances_
features = np.argsort(ranking)[::-1][:10]
columns = X.columns

plt.title("Feature importances based on Decision Tree Classifier", y = 1.03, size = 18)
plt.bar(range(len(features)), ranking[features], align="center")
plt.xticks(range(len(features)), columns[features], rotation=80)
plt.show()
df_test = pd.read_csv("../input/test.csv")
dropListForTest = ["Ticket","Name","Cabin","Embarked","Age"]
df_test.drop(dropListForTest,axis=1,inplace=True)
df_test['Sex'] = pd.DataFrame(np.where(df_test['Sex'] == 'male',1,0))
df_test["Fare"] = df_test["Fare"].fillna(0.0).apply(np.int64)
ids = df_test['PassengerId']
predictions = d_tree1.predict(df_test.drop('PassengerId', axis=1))

#set the output as a dataframe and convert to csv file named submission.csv
output = pd.DataFrame({ 'PassengerId' : ids, 'Survived': predictions })
output.to_csv('submission.csv', index=False)