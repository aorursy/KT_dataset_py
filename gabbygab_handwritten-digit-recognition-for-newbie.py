from sklearn.datasets import load_digits
from sklearn import svm
import matplotlib.pyplot as plt
digits = load_digits()

plt.gray()
plt.matshow(digits.images[20])
plt.show
print(digits.images[20])
clf = svm.SVC()
clf.fit(digits.data[:-1], digits.target[:-1])
prediction = clf.predict(digits.data[20:21])
print("Predicted Digit", prediction)
plt.gray()
plt.matshow(digits.images[28])
plt.show
clf = svm.SVC()
clf.fit(digits.data[:-1], digits.target[:-1])
prediction = clf.predict(digits.data[28:29])
print("Predicted Digit", prediction)