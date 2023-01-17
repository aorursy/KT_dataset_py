import pandas as pd
from sklearn.tree import DecisionTreeClassifier

# The competition datafiles are in the directory ../input
# Read competition data files:
train = pd.read_csv("../input/train.csv")
test  = pd.read_csv("../input/test.csv")

# Write to the log:
print("Training set has {0[0]} rows and {0[1]} columns".format(train.shape))
print("Test set has {0[0]} rows and {0[1]} columns".format(test.shape))
# Any files you write to the current directory get shown as outputs
df = train
tdf = test
X = df.drop('label', axis=1)
Y = df.label
cls = DecisionTreeClassifier()
cls.fit(X, Y)
predict = cls.predict(tdf)
submission = pd.DataFrame({'ImageId': tdf.index + 1,
                           'Label':predict})
submission.to_csv('decision_tree.csv', index=False)