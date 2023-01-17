#importing libraries



import numpy as np

import pandas as pd



from sklearn.model_selection import train_test_split

from sklearn.svm import SVC

from sklearn import metrics



import seaborn as sns
# Reading Data

train = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")
train.head()
test.head()
train.info()
#train["label"].value_counts().sort_index()

sns.countplot(train["label"])
pixels = train.drop(["label"], axis=1)

target = train["label"]
pixels = pixels/255.0
x_train, x_val, y_train, y_val = train_test_split(pixels, target, test_size=0.25, random_state=2019)
x_train.head()
y_train.head()
mdl = SVC(C=400, kernel='rbf', random_state=2019, gamma="scale", verbose=True)
mdl.fit(x_train, y_train)
predicted = mdl.predict(x_val)

predicted
print("accuracy", metrics.accuracy_score(y_val, predicted))    # accuracy
sns.heatmap(pd.DataFrame(metrics.confusion_matrix(y_val, predicted)), annot=True, cmap="YlGn", fmt='g')
test = test/255.0
y_pred = mdl.predict(test)
submission = {}

submission['ImageId'] = range(1,28001)

submission['Label'] = y_pred

submission = pd.DataFrame(submission)



submission = submission[['ImageId', 'Label']]

submission = submission.sort_values(['ImageId'])

submission.to_csv("submisision.csv", index=False)

print(submission['Label'].value_counts().sort_index())