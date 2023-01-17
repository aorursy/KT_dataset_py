from sklearn import svm, metrics

from sklearn.neighbors import KNeighborsClassifier

import sklearn.model_selection as ms

import pandas as pd
df = pd.read_csv("../input/train.csv")

df_test = pd.read_csv("../input/test.csv")

df.head()
df = df.sample(frac=1)

df = df.head(5000)
x = df.drop("label", axis=1)

y = df["label"]

test = df_test.as_matrix()
x_train, x_val, y_train, y_val = ms.train_test_split(x, y, test_size=0.2, random_state=0)
classifier = KNeighborsClassifier()

classifier.fit(x_train, y_train)
predicted = classifier.predict(x_val)

print (metrics.classification_report(predicted, y_val))
classifier = svm.SVC(gamma=0.001, kernel='linear')

#Todo: put in lines to train model, classify validation set, and print out metrics
from sklearn.model_selection import cross_val_predict



preds = cross_val_predict(classifier, X=x, y=y, cv=3)

print (metrics.classification_report(y, preds, digits=4))

print (metrics.confusion_matrix(y, preds))
predict_test = classifier.predict(test)

out = zip(range(len(test)), predict_test)

with open("./solution.csv", 'w') as g:

    g.write("ImageId,Label\n")

    for id, cat in out:

        g.write(str(id + 1) + "," + str(cat) + "\n")