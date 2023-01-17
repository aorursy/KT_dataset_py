import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
df = pd.read_csv(filepath_or_buffer='https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', header=None, sep=',')
df.dropna(how="all", inplace=True) # drops the empty line at file-end but there are none here..
iris_features=['sepal_len', 'sepal_wid', 'petal_len', 'petal_wid']  # there are 4 features in this database. or X vector.
df.columns= iris_features + ['class']  # last column is the 'class', or Y.
df.describe()
fig, axes = plt.subplots(2, 2, figsize=(18,12))
for idx, pr in enumerate(iris_features):
    plt.sca(axes[int(idx/2)][idx%2])
    ax = sns.boxplot(df[pr], df["class"])
    ax = sns.swarmplot(df[pr], df["class"])
plt.show()
err = setosa = versicolor = virginica = 0
for i, row in df.iterrows():
    pl = row['petal_len']
    pw = row['petal_wid']
    if pl < 2.5:  # Setosa
        if row['class'] != 'Iris-setosa' : err += 1
        setosa += 1
    elif pl <= 5 and pw < 1.75:  # Versicolor  petal_len >= 2.5, <= 5 cm.
        if row['class'] != 'Iris-versicolor': err += 1
        versicolor += 1
    elif pl >  5.0 or (pl <=  5.0  and pw >= 1.75):  # Virginica
        if row['class'] != 'Iris-virginica': err += 1
        virginica += 1

detected = setosa + versicolor + virginica
acc = round((detected - err) / detected * 100, 1)
print("Detected:", detected, "  Error:", err, "  Accuracy:", acc, "%")


from sklearn import tree
from sklearn import metrics
from graphviz import Source
from IPython.display import Image

def iris_decision_tree(df, depth=10, img_format="png"):
    dt=tree.DecisionTreeClassifier(max_depth=depth)
    dt.fit(df[iris_features], df['class'])
    classes = ['setosa', 'versicolor', 'virginica']  # ideally should deduce these in-order..
    dot=tree.export_graphviz(dt, out_file=None, feature_names=iris_features, class_names= classes) # diag of tree
    grf = Source(dot)
    grf.format=img_format
    fname= "dtree"+ str(depth)
    grf.render(fname)
    acc = metrics.accuracy_score(df['class'], dt.predict(df[iris_features]))
    return (acc, fname)

acc,fn = iris_decision_tree(df, depth=2)
print("Decision Tree of depth: 2,  Accuracy:", round(acc*100,1), "%")
Image(fn + ".png")
acc,fn = iris_decision_tree(df, depth=3)
print("Decision Tree of depth: 3,  Accuracy:", round(acc*100,1), "%")
Image(fn + ".png")