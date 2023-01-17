import pandas as pd

import numpy as np

from sklearn import tree

from sklearn.metrics import accuracy_score
train = pd.read_csv("../input/train.csv")

test  = pd.read_csv("../input/test.csv")
def preprocess_data(data):

    new_dataFrame = pd.DataFrame()



    new_dataFrame['Age'] = data.Age.fillna(data.Age.mean())

    new_dataFrame['Sex'] = pd.Series([1 if s == 'male' else 0 for s in data.Sex], name = 'Sex')



    return new_dataFrame
train_data = preprocess_data(train)

train_labels = train.Survived
classifier = tree.DecisionTreeClassifier()
classifier.fit(train_data, train_labels)
test_data = preprocess_data(test)
predicao = classifier.predict(test_data)
submission = pd.DataFrame()

submission['PassengerId'] = test.PassengerId

submission['Survived'] = pd.Series(predicao)

submission.to_csv("kaggle.csv", index=False)
print('Score: {}'.format(classifier.score(train_data, train_labels)))