#import libraries

import matplotlib.pyplot as plt

%matplotlib inline

from sklearn.datasets import load_digits
digits = load_digits()
dir(digits)
digits.data[0]
plt.gray()

for i in range(5):

    plt.matshow(digits.images[i])
digits.target[0:5]
#import library

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size = 0.2)
len(x_train)
len(x_test)
#import library

from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
model.fit(x_train, y_train)
model.score(x_test, y_test) #accuracy
plt.matshow(digits.images[67])
digits.target[67]
model.predict([digits.data[67]])
model.predict(digits.data[0:5])
#import library

#confusion matrix

y_predicted = model.predict(x_test)

from sklearn.metrics import confusion_matrix



cm = confusion_matrix(y_test, y_predicted)

cm
#visualization

#import library

import seaborn as sns

plt.figure(figsize = (10,7))

sns.heatmap(cm, annot = True)

plt.xlabel('Predicted')

plt.ylabel('Truth')