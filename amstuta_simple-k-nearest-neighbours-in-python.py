import numpy as np

import pandas as pd

import matplotlib.pyplot as plt



%matplotlib inline

plt.rcParams['figure.figsize'] = (20.0, 16.0)
class KNearestNeighbours:



    def __init__(self, k=1, samples_per_class=None):

        # Number of examples to save from each class (if None all is saved)

        self.samples_per_class = samples_per_class

        self.k = k





    def train(self, X, y):

        groups = list(set(y))

        zipped = list(zip(y, X))

        data = {a : np.array([b[1] for b in zipped if b[0] == a]) for a in groups}

        features, targets = [], []



        self.group_means = {}

        for group in data:



            # Calculation of a mean example for each class

            mean = np.mean(data[group], axis=0)

            features.append(mean)

            targets.append(group)

            self.group_means[group] = mean



            # Here we randomly choose self.samples_per_class examples for each class. 

            # A better approach would be to keep only a few 'typical' instances.

            nb_samples = len(data[group]) if not self.samples_per_class else self.samples_per_class

            idxs = np.random.choice(range(len(data[group])), nb_samples, replace=False)

            for i in idxs:

                features.append(data[group][i])

                targets.append(group)



        self.X_train = np.array(features)

        self.y_train = np.array(targets)





    def predict(self, X):

        # Calculation of Euclidean distances

        p_squared = np.square(X).sum(axis=1)

        q_squared = np.square(self.X_train).sum(axis=1)

        product   = -2 * X.dot(self.X_train.T)

        distances = np.sqrt(product + q_squared + np.matrix(p_squared).T)



        # Selection of the nearest neighbours for each example

        predictions = np.zeros(len(distances))

        for i in range(len(distances)):

            labels = self.y_train[np.argsort(distances[i])].flatten()

            closest = list(labels[:self.k])

            predictions[i] = max(closest, key=closest.count)

        return predictions
train_data = pd.read_csv("../input/train.csv")

X_train = train_data.iloc[:, 1:].as_matrix()

y_train = train_data["label"].as_matrix()

X_test = pd.read_csv("../input/test.csv").as_matrix()



print(X_train.shape, y_train.shape, X_test.shape)
classifier = KNearestNeighbours(k=3, samples_per_class=2000)

classifier.train(X_train, y_train)
for target, features in classifier.group_means.items():

    plt.subplot(1, 10, target + 1)

    plt.imshow(features.reshape((28,28)))

    plt.axis("off")

    plt.title(target)

plt.show()
batch = 2000

predictions = []

for i in range(0, len(X_test), batch):

    print("Computing tests {} to {}".format(i, i + batch))

    pred = classifier.predict(X_test[i: i + batch])

    predictions += list(pred)
predictions = pd.DataFrame(predictions, dtype=int)

predictions.index+=1

predictions.index.name='ImageId'

predictions.columns=['Label']

predictions.to_csv('preds.csv', header=True)