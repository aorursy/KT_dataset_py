import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn import tree
# Input data files are available in the "../input/" directory.

#from subprocess import check_output
#print(check_output(["ls", "../input"]).decode("utf8"))

train = pd.read_csv('../input/train.csv')
train_target = train['label']
train_features = train.ix[:,'pixel0':]

test = pd.read_csv('../input/test.csv')
dtree = tree.DecisionTreeClassifier()
dtree = dtree.fit(train_features, train_target)

# Look at the importance and score of the included features
print(dtree.feature_importances_)
print(dtree.score(train_features, train_target))
prediction = dtree.predict(test)

solution = pd.DataFrame(prediction, columns = ['Label'])
solution.index += 1 
print(solution.head())
print(solution.shape)

# Write your solution to a csv file with the name my_solution.csv
solution.to_csv("solution.csv", index_label = ["ImageId"])
train_features.head()
test.head()
from subprocess import check_output
print(check_output(["ls", "."]).decode("utf8"))

