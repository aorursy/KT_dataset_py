import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.neural_network import MLPClassifier
data = pd.read_csv('../input/Kvant_data.csv', delimiter=";", decimal=",")



data
X = pd.DataFrame(data.iloc[0:9, 0:11])

y = pd.DataFrame(data.iloc[0:9, 11])

X
model = MLPClassifier(solver='sgd', activation='logistic')
model.fit(X, y)
model.predict(data.iloc[9:10,:11])
print("Trærning score: %f" % model.score(X, y))

print("Test score: %f" % model.score(data.iloc[9:10,:11], data.iloc[9:10,11:12]))