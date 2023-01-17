import pathlib

import time



import matplotlib.pyplot as plt

import numpy as np

import pandas as pd

import torch

from sklearn.decomposition import PCA

from sklearn.manifold import TSNE

from sklearn.metrics import confusion_matrix

from sklearn.model_selection import train_test_split
start = time.time()

train_fp = pathlib.Path('/kaggle/input/digit-recognizer/train.csv')

train_x, train_y = [], []

with open(train_fp, 'r') as fh:

    next(fh)

    for line in fh:

        label, *image = line.split(',')

        train_x.append([int(p) for p in image])

        train_y.append(int(label))

train_x = np.array(train_x).reshape(-1, 28, 28)

train_y = np.array(train_y)

print(f'Loaded training data in {time.time() - start:.1f}s')
_, counts = np.unique(train_y, return_counts=True)

counts = [*counts, sum(counts)]

index = [*list(range(10)), 'total']

columns = ['']

display(pd.DataFrame(counts, index=index, columns=columns).T)
plt.figure(figsize=(12, 12))

for i in range(6**2):

    plt.subplot(6, 6, i + 1)

    plt.imshow(train_x[i], cmap=plt.cm.gray)

    plt.xlabel(str(train_y[i]))

    plt.xticks([])

    plt.yticks([])

plt.show()
centroids = [train_x[train_y == i].mean(axis=0) for i in range(10)]

plt.figure(figsize=(12, 4))

for i, centroid in enumerate(centroids, 1):

    plt.subplot(2, 5, i)

    plt.imshow(centroid, cmap=plt.cm.gray)

    plt.xlabel(str(i))

    plt.xticks([])

    plt.yticks([])

plt.show()
plt.figure(figsize=(14, 6))

for i, alg in enumerate([PCA(2), TSNE(2)], 1):

    x_reduced = alg.fit_transform(train_x[:1000].reshape(-1, 784))

    y_reduced = train_y[:1000]

    data = [x_reduced[y_reduced == i] for i in range(10)]

    plt.subplot(1, 2, i)

    for j, points in enumerate(data):

        plt.scatter(points[:, 0], points[:, 1], s=16, alpha=0.7, label=str(j))

    plt.title(f'1000 images from training data using {type(alg).__name__}')

    plt.legend()

    plt.xticks([])

    plt.yticks([])

plt.show()
class ConvClassifier(torch.nn.Module):



    def __init__(self):

        super().__init__()

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        print(f'Using device: {self.device}')



        self.layers1 = torch.nn.ModuleList([

            torch.nn.Conv2d(1, 32, 5),

            torch.nn.ReLU(),

            torch.nn.MaxPool2d(2),

            torch.nn.Dropout2d(0.25),

            torch.nn.Conv2d(32, 64, 5),

            torch.nn.ReLU(),

            torch.nn.MaxPool2d(2),

            torch.nn.Dropout2d(0.25),

            torch.nn.Conv2d(64, 128, 1),

            torch.nn.ReLU(),

            torch.nn.MaxPool2d(2)])

        self.layers2 = torch.nn.ModuleList([

            torch.nn.Dropout(0.25),

            torch.nn.Linear(512, 150),

            torch.nn.ReLU(),

            torch.nn.Dropout(0.25),

            torch.nn.Linear(150, 10)])

        n_params = sum([np.prod(p.size()) for p in self.parameters() if p.requires_grad])

        print(f'Number of parameters: {n_params}')



        self.loss = torch.nn.functional.cross_entropy

        self.optimizer = torch.optim.Adam(self.parameters())



    def fit(self, train_x, train_y, val_x, val_y, epochs=50, batch_size=250):

        start = time.time()

        self.to(self.device)

        train_loader = self._build_loader_xy(

            train_x, train_y, batch_size=batch_size, shuffle=True)

        val_loader = self._build_loader_xy(val_x, val_y, batch_size=1000, shuffle=False)

        history = {'epoch': [], 'loss': [], 'acc': [], 'val_loss': [], 'val_acc': []}

        for epoch in range(1, epochs + 1):

            self.train()

            start_time = time.time()

            loss, acc = 0, 0

            for x, y in train_loader:

                # Batch training step

                self.optimizer.zero_grad()

                x, y = x.to(self.device), y.to(self.device)

                y_pred = self(x)

                loss_ = self.loss(y_pred, y)

                loss_.backward()

                self.optimizer.step()

                # Calculate metrics for batch

                loss += float(loss_) * len(x)

                acc += int((y_pred.argmax(dim=1) == y).sum())

            # Calculate metrics for epoch

            loss /= len(train_loader.dataset)

            acc /= len(train_loader.dataset)

            val_loss, val_acc = self._evaluate(val_loader)

            duration = time.time() - start_time

            # Save training history

            history['epoch'].append(epoch)

            history['loss'].append(loss)

            history['acc'].append(acc)

            history['val_loss'].append(val_loss)

            history['val_acc'].append(val_acc)

            print(

                f'[{epoch:{len(str(epochs))}}/{epochs}] {duration:.1f}s'

                f' - loss: {loss:.4f} - acc: {acc:.4f}'

                f' - val_loss: {val_loss:.4f} - val_acc: {val_acc:.4f}')

        return history



    def predict(self, x, batch_size=1000):

        self.to(self.device)

        data_loader = self._build_loader_x(x, batch_size=batch_size, shuffle=False)

        self.eval()

        preds = []

        with torch.no_grad():

            for x, in data_loader:

                x = x.to(self.device)

                preds.extend(self(x).argmax(dim=1).cpu())

        return np.array(preds)



    def forward(self, x):

        x = x.view(x.size()[0], 1, *x.size()[1:])

        for layer in self.layers1:

            x = layer(x)

        x = x.view(x.size()[0], -1)

        for layer in self.layers2:

            x = layer(x)

        return x



    def _evaluate(self, data_loader):

        self.to(self.device)

        self.eval()

        loss, acc = 0, 0

        with torch.no_grad():

            for x, y in data_loader:

                x, y = x.to(self.device), y.to(self.device)

                y_pred = self(x)

                loss += float(self.loss(y_pred, y)) * len(x)

                acc += int((y_pred.argmax(dim=1) == y).sum())

        loss /= len(data_loader.dataset)

        acc /= len(data_loader.dataset)

        return loss, acc



    @staticmethod

    def _build_loader_x(x, batch_size, shuffle):

        return torch.utils.data.DataLoader(

            torch.utils.data.TensorDataset(torch.FloatTensor(x) / 255),

            batch_size=batch_size, shuffle=shuffle)



    @staticmethod

    def _build_loader_xy(x, y, batch_size, shuffle):

        return torch.utils.data.DataLoader(

            torch.utils.data.TensorDataset(torch.FloatTensor(x) / 255, torch.LongTensor(y)),

            batch_size=batch_size, shuffle=shuffle)
model = ConvClassifier()
train_x, val_x, train_y, val_y = train_test_split(

    train_x, train_y, test_size=2000, random_state=1234)

history = model.fit(train_x, train_y, val_x, val_y)
plt.figure(figsize=(16, 5))

for i, (name, key) in enumerate([('accuracy', 'acc'), ('loss', 'loss')], 1):

    plt.subplot(1, 2, i)

    plt.plot(history['epoch'], history[key], label=f'training {name}')

    plt.plot(history['epoch'], history[f'val_{key}'], label=f'validation {name}')

    plt.title(f'{name} during training')

    plt.legend()

    plt.xlabel('epoch')

    plt.ylabel(name)

plt.show()
plt.figure(figsize=(12, 6))

cm = confusion_matrix(val_y, model.predict(val_x))

threshold = np.max(cm) / 2

plt.imshow(cm, cmap=plt.cm.binary)

for i in range(10):

    for j in range(10):

        plt.text(

            j, i, cm[i, j], color='white' if cm[i, j] > threshold else 'black',

            horizontalalignment='center', verticalalignment='center')

plt.title(f'confusion matrix')

plt.xlabel('predicted label')

plt.xticks(range(10), range(10))

plt.ylabel('true label')

plt.yticks(range(10), range(10))

plt.show()
start = time.time()

test_fp = pathlib.Path('/kaggle/input/digit-recognizer/test.csv')

test_x = []

with open(test_fp, 'r') as fh:

    next(fh)

    for line in fh:

        image = line.split(',')

        test_x.append([int(p) for p in image])

test_x = np.array(test_x).reshape(-1, 28, 28)

print(f'Loaded test data in {time.time() - start:.1f}s')
with open('/kaggle/working/digit-recognizer_submission.csv', 'w') as fh:

    fh.write('ImageId,Label\n')

    for i, label in enumerate(model.predict(test_x), 1):

        fh.write(f'{i},{label}\n')

print('Done')