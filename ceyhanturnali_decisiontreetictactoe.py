!pip install pydotplus
import pandas as pd
from IPython.display import Image
from io import StringIO
import pydotplus
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier 
from sklearn.model_selection import train_test_split 

df = pd.read_csv("../input/tictactoe-endgame-dataset-uci/tic-tac-toe-endgame.csv", delimiter=",")
df[0:5]
feature_names = df[['V1','V2','V3','V4','V5','V6','V7','V8','V9' ]].values
feature_names[0:5]
class_names =[v10[0], v10[1]]
class_names
df['V1'],v1 = pd.factorize(df['V1'], sort=True)
df['V2'],v2 = pd.factorize(df['V2'], sort=True)
df['V3'],v3 = pd.factorize(df['V3'], sort=True)
df['V4'],v4 = pd.factorize(df['V4'], sort=True)
df['V5'],v5 = pd.factorize(df['V5'], sort=True)
df['V6'],v6 = pd.factorize(df['V6'], sort=True)
df['V7'],v7 = pd.factorize(df['V7'], sort=True)
df['V8'],v8 = pd.factorize(df['V8'], sort=True)
df['V9'],v9 = pd.factorize(df['V9'], sort=True)
df['V10'],v10 = pd.factorize(df['V10'], sort=True)
[v1, v2, v3, v4, v5, v6, v7, v8, v9, v10] 


df
df.describe()

feature_names = ['V1','V2','V3','V4', 'V5', 'V6', 'V7', 'V8', 'V9']
x = df[feature_names] # Features
x
y = df['V10'] # Target
y
X_trainset, X_testset, y_trainset, y_testset = train_test_split(x, y, test_size=0.3, random_state=3)
clf = DecisionTreeClassifier(criterion='entropy', min_samples_split=80)
clf = clf.fit(X_trainset,y_trainset)
predictionTree = clf.predict(X_testset)
print (predictionTree [0:5])
print (y_testset [0:5])
from sklearn import metrics

print("DecisionTrees's Accuracy: ", metrics.accuracy_score(y_testset, predictionTree))
count_misclassified = (y_testset != predictionTree).sum()
print('Misclassified samples: {}'.format(count_misclassified))
def plot_decision_tree(clf, features, classes):
    dot_data = StringIO()
    tree.export_graphviz(clf, out_file=dot_data, feature_names=features, class_names=classes, filled=True, rounded=True, special_characters=True)
    graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
    return Image(graph.create_png())
plot_decision_tree(clf, feature_names, class_names)
