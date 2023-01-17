import numpy as np

import pandas as pd



import matplotlib.pyplot as plt



from sklearn.ensemble import RandomForestClassifier

from sklearn.tree import DecisionTreeClassifier
data = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')
np.random.seed(0)



datalen_t = len(data)

splitmark = int(.83 * datalen_t)

print(datalen_t, datalen_t - splitmark)

train, valid = data.iloc[:splitmark, :], data.iloc[splitmark:, :] 
def extract_data(data):

    labels = data.iloc[:, 0]

    digits = data.iloc[:, 1:]

    return digits, labels
x, y = extract_data(train)

xv, yv = extract_data(valid)
best_depth = 0

best_score = 0

for depth in range(1, 100):

    dt = DecisionTreeClassifier(max_depth=depth)

    dt.fit(x, y)

    score = dt.score(xv, yv)

    if score > best_score:

        best_depth = depth

        best_score = score

print(best_depth, best_score)
best_estimators = 1

for estimators in range(1, 100):

    rf = RandomForestClassifier(max_depth=best_depth, n_jobs=-1, n_estimators=estimators)

    rf.fit(x, y)

    score = rf.score(xv, yv)

    if score > best_score:

        best_estimators = estimators

        best_score = score

print(best_estimators, best_score)
xt = np.array(test)

rf = RandomForestClassifier(max_depth=best_depth, n_jobs=-1, n_estimators=best_estimators)

rf.fit(x, y)

print(rf.score(xv, yv))

yt = rf.predict(xt)
submission = pd.DataFrame({'ImageId': [i + 1 for i in range(len(xt))], 'Label': yt})

submission.to_csv('submission.csv', index=False)