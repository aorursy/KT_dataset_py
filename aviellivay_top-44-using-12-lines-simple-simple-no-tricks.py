import pandas as pd

from sklearn.ensemble import RandomForestClassifier

train = pd.read_csv("../input/learn-together/train.csv", index_col=0)

test = pd.read_csv("../input/learn-together/test.csv", index_col=0)

X = train.copy()

X = X.drop('Cover_Type', 1)

y = train['Cover_Type']

model = RandomForestClassifier(n_estimators=100)

model = model.fit(X,y)

predicts = model.predict(test)

output = pd.DataFrame({'ID': test.index, 'Cover_Type': predicts})

output.to_csv('my_model.csv', index=False)