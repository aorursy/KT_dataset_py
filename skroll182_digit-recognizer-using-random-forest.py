import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
def grayscale_to_binary(X):
    X[X>0] = 1
train_dataset = pd.read_csv("../input/train.csv").iloc[:20000, :].values
test_dataset = pd.read_csv('../input/test.csv').values
X_train = train_dataset[:, 1:]
y_train = train_dataset[:, 0]
image = X_train[100].reshape((28,28))
plt.title(y_train[100])
plt.xticks([],[])
plt.yticks([],[])
plt.imshow(image, cmap='gray')
plt.show()
grayscale_to_binary(X_train)
image = X_train[100].reshape((28,28))
plt.title(y_train[100])
plt.xticks([],[])
plt.yticks([],[])
plt.imshow(image, cmap='gray')
plt.show()
rfc = RandomForestClassifier(n_estimators=110).fit(X_train, y_train)
grayscale_to_binary(test_dataset)
y_result = rfc.predict(test_dataset)
for i in range(1, 11):
    plt.subplot(2, 5, i)
    image = test_dataset[i-1].reshape((28,28))
    plt.title(y_result[i-1])
    plt.yticks([],[])
    plt.xticks([],[])
    plt.imshow(image, cmap='gray')
plt.show()
results = pd.DataFrame(y_result)
results.index.name='ImageID'
results.index+=1
results.columns=['Label']
results.to_csv("random_forest_results.csv", header=True)
