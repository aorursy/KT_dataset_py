import pandas as pd



train = pd.read_csv('/kaggle/input/identify-sinkhole/train.csv')

test = pd.read_csv('/kaggle/input/identify-sinkhole/test.csv')
from sklearn import tree

clf = tree.DecisionTreeClassifier(max_depth=8)

clf.fit(X=train.drop(columns=['ID','IsSinkhole']), y=train.IsSinkhole)
filename = "decision_tree.csv"



predictions = clf.predict(test.drop(columns=['ID']),)

test['IsSinkhole'] = predictions

test.to_csv(filename,index_label='ID',columns=['IsSinkhole'])
from IPython.display import FileLink

FileLink(filename)