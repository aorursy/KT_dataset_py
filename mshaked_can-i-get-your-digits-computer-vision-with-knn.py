import pandas as pd

from sklearn.ensemble import RandomForestClassifier

#from sklearn.metrics import accuracy_score





train = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")
train_x = train[train.columns[1:]]

train_y = train['label']



rfc = RandomForestClassifier(n_jobs=-1, n_estimators=10)



rfc.fit(train_x, train_y)



predicted = list(rfc.predict(test))



image_ids = list(range(1, len(test) + 1))



results = pd.DataFrame({"ImageId" : image_ids,"Label" : predicted})



results.to_csv('submission.csv')