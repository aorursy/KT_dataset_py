import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score
data = pd.read_csv("../input/mushrooms.csv")

data.head()
features = data.iloc[:,1:]

features = pd.get_dummies(features).astype(float)  #one hot encoding

target = data.iloc[:,0]

features.head()
X = features.values

y = target.values
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.30)
from sklearn import tree

clf = tree.DecisionTreeClassifier()
clf.fit(x_train, y_train)
predictions = clf.predict(x_test)

print(accuracy_score(y_test,predictions))  # raw score
sample = data.iloc[-1,:]

sample
sample_x = features.iloc[-1]

sample_y = target.iloc[-1]
sample_pred = clf.predict(sample_x.values.reshape(1,-1))

print("Model Prediction: ", sample_pred)

print("Sample Value: ", sample_y)