import pandas as pd

from sklearn import tree



train = pd.read_csv('/kaggle/input/sinkhole-or-not-2019/train.csv')

test = pd.read_csv('/kaggle/input/sinkhole-or-not-2019/test.csv')



clf = tree.DecisionTreeClassifier(max_depth=8)

clf.fit(X=train.drop(columns=['ID','IsSinkhole']), y=train.IsSinkhole)



predictions = clf.predict(test.drop(columns=['ID']),)

test['IsSinkhole'] = predictions

test.to_csv('d8_decision_tree.csv',index_label='ID',columns=['IsSinkhole'])