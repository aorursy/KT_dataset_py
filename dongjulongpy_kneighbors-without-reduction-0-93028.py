import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
import time

start_1 = time.time()
train_df = pd.read_csv('../input/train.csv')
train_df.head()
train_df = train_df[:5000] # to keep process short.
X = train_df.drop('label', axis=1)
y = train_df['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=True)
#svm = LinearSVC()
#svm.fit(X_train, y_train)
#svm_predictions = svm.predict(X_test)
#print(classification_report(y_test, svm_predictions))
#print(accuracy_score(y_test, svm_predictions))
kneigh = KNeighborsClassifier() # I have tried both svm and kneighbors.
kneigh.fit(X_train, y_train) #  Kneighbors is much better than svm with less data.
kneigh_predictions = kneigh.predict(X_test)
print(classification_report(y_test, kneigh_predictions))
print(accuracy_score(y_test, kneigh_predictions))
end = time.time()
print("Time: ", end - start_1)
X = train_df.drop('label', axis=1)
y = train_df['label']

kneigh.fit(X, y)
# I divided the test dataset becauce it took long time when I tried to predict with complete dataset.
test_df = pd.read_csv('../input/test.csv')
X_test_1 = test_df[:5000]
X_test_2 = test_df[5000:10000]
X_test_3 = test_df[10000:15000]
X_test_4 = test_df[15000:20000]
X_test_5 = test_df[20000:25000]
X_test_6 = test_df[25000:]
start = time.time()
submission_1 = kneigh.predict(X_test_1)
end = time.time()
print('...done', end-start)

# I have created each submission dataframe and added them.
submission = pd.DataFrame(submission_1, columns=['Label'])
start = time.time()
submission_2 = kneigh.predict(X_test_2)
end = time.time()
print('...done', end-start)

submission_2 = pd.DataFrame(submission_2, columns=['Label'])
submission = submission.append(submission_2, ignore_index=True, sort=False)
start = time.time()
submission_3 = kneigh.predict(X_test_3)
end = time.time()
print('...done', end-start)

submission_3 = pd.DataFrame(submission_3, columns=['Label'])
submission = submission.append(submission_3, ignore_index=True, sort=False)
start = time.time()
submission_4 = kneigh.predict(X_test_4)
end = time.time()
print('...done', end-start)

submission_4 = pd.DataFrame(submission_4, columns=['Label'])
submission = submission.append(submission_4, ignore_index=True, sort=False)
start = time.time()
submission_5 = kneigh.predict(X_test_5)
end = time.time()
print('...done', end-start)

submission_5 = pd.DataFrame(submission_5, columns=['Label'])
submission = submission.append(submission_5, ignore_index=True, sort=False)
start = time.time()
submission_6 = kneigh.predict(X_test_6)
end = time.time()
print('...done', end-start)

submission_6 = pd.DataFrame(submission_6, columns=['Label'])
submission = submission.append(submission_6, ignore_index=True, sort=False)
submission.head()
example_sub = pd.read_csv('../input/sample_submission.csv')
example_sub.head()
submission.set_index(example_sub['ImageId'], inplace=True)
submission.head()
submission.info()
submission.to_csv('submission.csv')
end = time.time()

print('Complete Submission ...done', end - start_1)