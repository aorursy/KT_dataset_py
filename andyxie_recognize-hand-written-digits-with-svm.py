import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from sklearn import datasets, svm, metrics
from sklearn.model_selection import train_test_split
data = datasets.load_digits()
print(data["DESCR"])
data.data.shape
data.target.shape
X = data.data
y = data.target
# from sklearn.preprocessing import scale
# mean = X.mean()
# std = X.std()
# X = (X - mean)/std
plt.hist(X[0,:], density=True, histtype="step")
def plot_digit(number_of_digit, data, target):
    row = data[number_of_digit,:]
    plt.imshow(row.reshape(8,8), cmap=plt.cm.gray_r, interpolation='nearest')
    plt.xticks(())
    plt.yticks(())
    plt.title("Target: " + str(target[number_of_digit]), size=20)
    
plot_digit(0,X,y)
fig = plt.figure(figsize=(10, 12))
fig.tight_layout()
for i in range(9):
    plt.subplot(3,3, i+1)
    plot_digit(i, X, y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

model = svm.SVC(gamma=0.001)
model.fit(X_train, y_train)
y_prediction = model.predict(X_test)
report = metrics.classification_report(y_prediction,y_test)
print("SVM Classification Report")
print(report)
print("Confusion Matrix")
confusion_matrix = metrics.confusion_matrix(y_prediction, y_test)
print(confusion_matrix)
fig = plt.figure(figsize=(10, 12))
fig.tight_layout()
for i in range(9):
    plt.subplot(3,3, i+1)
    plot_digit(i, X_test, y_prediction)