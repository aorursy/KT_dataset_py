import pandas as pd
from sklearn import tree

test = pd.read_csv('../input/test.csv')
train = pd.read_csv('../input/train.csv')

clf = tree.DecisionTreeClassifier()

trainLabel = train.label
train = train.drop('label',1)

clf.fit(train,trainLabel)

dt = clf.predict(test)
predictions = pd.DataFrame(data=dt,columns=["label"])
predictions.ImageId = list(range(1,len(test)+1))
predictions.to_csv("sklearn_decisiontree.csv",index_label="ImageId", sep='\t')
