from sklearn.datasets import fetch_openml
%matplotlib inline

import matplotlib

import matplotlib.pyplot as plt
import numpy as np
mnist = fetch_openml('mnist_784')
x, y = mnist['data'], mnist['target']
x.shape
# random_digit = x[50000]

# random_digit_image = random_digit.reshape(28, 28)



# plt.imshow(random_digit_image, cmap=matplotlib.cm.binary, interpolation='nearest')

# plt.axis('off')

# plt.show()
x_train, x_test, y_train, y_test = x[:60000], x[60000:], y[:60000], y[60000:]
x_test.shape
shuffle_index = np.random.permutation(60000)

x_train, y_train = x_train[shuffle_index], y_train[shuffle_index]

y_train = y_train.astype(np.int8)
names = np.array([1, 2, 5, 4, 5, 6, 7, 5, 7])

b_names = (names == 5)



print(b_names)
y_train_5 = (y_train == 5)

y_test_5 = (y_test == 5)
from sklearn.linear_model import SGDClassifier
y_train_5
sgd_test_clf = SGDClassifier(random_state=42)

sgd_test_clf.fit(x_train, y_train_5)
rd = x_test[5]

rdi = rd.reshape(28, 28)
sgd_test_clf.predict(rdi.reshape(1, -1))
sgd_clf = SGDClassifier(random_state=42)

sgd_clf.fit(x_train, y_train)
random_digit = x_test[114]

random_digit_image = random_digit.reshape(28, 28)

print(sgd_clf.predict(random_digit_image.reshape(1, -1)))



plt.imshow(random_digit_image)

plt.show()