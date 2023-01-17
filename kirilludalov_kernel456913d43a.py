import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import matplotlib as mlp

import seaborn as sns

from sklearn.neighbors import KNeighborsClassifier, KDTree

from sklearn.manifold import TSNE

from sklearn.datasets import load_wine

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import cross_val_score, KFold

from sklearn.metrics import accuracy_score

%matplotlib inline
mlp.rcParams['figure.figsize'] = [16.0, 10.0]
plt.style.use('ggplot')
wine_data = load_wine()
df = pd.DataFrame(wine_data['data'], columns=wine_data['feature_names'])
df['Target'] = wine_data['target']
df.rename(columns={'od280/od315_of_diluted_wines': 'od280/od315'}, inplace=True)
df.head()
tsne = TSNE(n_components=2)
scaler = StandardScaler()
x = scaler.fit_transform(df.drop('Target', axis=1))
x = tsne.fit_transform(x)
x_min, x_max = x[:, 0].min() - 1, x[:, 0].max() + 1

y_min, y_max = x[:, 1].min() - 1, x[:, 1].max() + 1

xx, yy = np.meshgrid(np.arange(x_min, x_max, .05), np.arange(y_min, y_max, .05))
def print_border(n_neighbors):

    clf = KNeighborsClassifier(n_neighbors=n_neighbors)

    clf.fit(x, df['Target'])

    predict = clf.predict(np.c_[xx.ravel(), yy.ravel()])

    predict = predict.reshape(xx.shape)

    plt.pcolormesh(xx, yy, predict, cmap='terrain');

    sns.scatterplot(x[:, 0], x[:, 1], hue=df['Target'], palette=['red', 'blue', 'green']);
print_border(1)
print_border(3)
print_border(9)
score = []

for i in range(1, 101):

    val = cross_val_score(KNeighborsClassifier(n_neighbors=i), df.drop('Target', axis=1), df['Target'], cv=5).mean()

    score.append(1 - val)
plt.plot(np.arange(1, 101), score);
df.describe()
scaler = StandardScaler()

scaler.fit(df.drop('Target', axis=1))
score = []

for i in range(1, 101):

    val = cross_val_score(KNeighborsClassifier(n_neighbors=i), 

                          scaler.transform(df.drop('Target', axis=1)), 

                          df['Target'], cv=5).mean()

    score.append(1 - val)
plt.plot(np.arange(1, 101), score);
min(score)
class ParsenWindowEpachnikov:

    

    def __init__(self, kernel_size):

        self.kernel_size = kernel_size

        self.tree = None

        self.y = None

        self.classes = None

        

    def fit(self, x, y):

        self.tree = KDTree(x)

        self.y = y

        self.classes = np.unique(y)

        

    def predict_proba(self, x):

        idx, dist = self.tree.query_radius(x, self.kernel_size, return_distance=True)

        if idx.shape[0] == 0:

            return np.zeros((x.shape[0], self.classes.shape[0]))

        pred = np.zeros((x.shape[0], self.classes.shape[0]))

        kernel = self.kernel(dist/self.kernel_size)

        for i in range(idx.shape[0]):

            row_target = self.y[idx[i]]

            for j, cl in enumerate(self.classes):

                cl_id = np.where(row_target == cl)[0]

                if cl_id.shape[0] > 0:

                    pred[i, j] = kernel[i][cl_id].sum()

        pred_sum = pred.sum(axis=1).reshape(-1, 1)

        pred_sum[pred_sum == 0] = 1

        pred /= pred_sum

        return pred

    

    def predict(self, x):

        proba = self.predict_proba(x)

        unnknown_idx = proba.sum(axis=1) == 0

        predict = np.argmax(proba, axis=1)

        predict[unnknown_idx] = -1

        return predict

            

    @staticmethod

    def kernel(r):

        return 3/4*(1 - r**2)
def print_border_pw(kernel_size):

    clf = ParsenWindowEpachnikov(kernel_size=kernel_size)

    clf.fit(x, df['Target'].values)

    predict = clf.predict(np.c_[xx.ravel(), yy.ravel()])

    predict = predict.reshape(xx.shape)

    plt.pcolormesh(xx, yy, predict, cmap='terrain');

    sns.scatterplot(x[:, 0], x[:, 1], hue=df['Target'], palette=['red', 'blue', 'green']);
print_border_pw(3)
print_border_pw(5)
print_border_pw(20)
score = []

for i in range(3, 101):

    est = ParsenWindowEpachnikov(kernel_size=i)

    cv = KFold(n_splits=5, shuffle=True)

    vals = []

    for train_idx, test_idx in cv.split(np.arange(df.shape[0])):

        scaler = StandardScaler()

        train_data = df.iloc[train_idx].drop('Target', axis=1)

        train_data = scaler.fit_transform(train_data)

        test_data = df.iloc[test_idx].drop('Target', axis=1)

        test_data = scaler.transform(test_data)

        est.fit(train_data, df.iloc[train_idx]['Target'].values)

        predict = est.predict(test_data)

        vals.append(1 - accuracy_score(df.iloc[test_idx]['Target'].values, predict))

    score.append(np.mean(vals))
plt.plot(np.arange(3, 101), score);
min(score)
np.argmin(score) + 3
class ParsenWindowEpachnikovNeighbors:

    

    def __init__(self, n_neighbors):

        self.n_neighbors = n_neighbors + 1

        self.tree = None

        self.y = None

        self.classes = None

        

    def fit(self, x, y):

        self.tree = KDTree(x)

        self.y = y

        self.classes = np.unique(y)

        

    def predict_proba(self, x):

        dist, idx = self.tree.query(x, self.n_neighbors, return_distance=True)

        if idx.shape[0] == 0:

            return np.zeros((x.shape[0], self.classes.shape[0]))

        pred = np.zeros((x.shape[0], self.classes.shape[0]))

        kernel = self.kernel(dist/dist.max(axis=1).reshape(-1, 1))

        for i in range(idx.shape[0]):

            row_target = self.y[idx[i]]

            for j, cl in enumerate(self.classes):

                cl_id = np.where(row_target == cl)[0]

                if cl_id.shape[0] > 0:

                    pred[i, j] = kernel[i][cl_id].sum()

        pred_sum = pred.sum(axis=1).reshape(-1, 1)

        pred_sum[pred_sum == 0] = 1

        pred /= pred_sum

        return pred

    

    def predict(self, x):

        proba = self.predict_proba(x)

        unnknown_idx = proba.sum(axis=1) == 0

        predict = np.argmax(proba, axis=1)

        predict[unnknown_idx] = -1

        return predict

            

    @staticmethod

    def kernel(r):

        k = 3/4*(1 - r**2)

        k[k < 0] = 0

        return k
def print_border_pwn(n_neighbors):

    clf = ParsenWindowEpachnikovNeighbors(n_neighbors=n_neighbors)

    clf.fit(x, df['Target'].values)

    predict = clf.predict(np.c_[xx.ravel(), yy.ravel()])

    predict = predict.reshape(xx.shape)

    plt.pcolormesh(xx, yy, predict, cmap='terrain');

    sns.scatterplot(x[:, 0], x[:, 1], hue=df['Target'], palette=['red', 'blue', 'green']);
plt.subplot(2, 1, 1);

print_border_pwn(3)

plt.subplot(2, 1, 2);

print_border(3)
plt.subplot(2, 1, 1);

print_border_pwn(5)

plt.subplot(2, 1, 2);

print_border(5)
plt.subplot(2, 1, 1);

print_border_pwn(25)

plt.subplot(2, 1, 2);

print_border(25)
score = []

for i in range(1, 101):

    est = ParsenWindowEpachnikovNeighbors(n_neighbors=i)

    cv = KFold(n_splits=5, shuffle=True)

    vals = []

    for train_idx, test_idx in cv.split(np.arange(df.shape[0])):

        scaler = StandardScaler()

        train_data = df.iloc[train_idx].drop('Target', axis=1)

        train_data = scaler.fit_transform(train_data)

        test_data = df.iloc[test_idx].drop('Target', axis=1)

        test_data = scaler.transform(test_data)

        est.fit(train_data, df.iloc[train_idx]['Target'].values)

        predict = est.predict(test_data)

        vals.append(1 - accuracy_score(df.iloc[test_idx]['Target'].values, predict))

    score.append(np.mean(vals))
plt.plot(np.arange(1, 101), score);


min(score)
np.argmin(score)
class ParsenWindowEpachnikovNeighbors:

    

    def __init__(self, n_neighbors):

        self.n_neighbors = n_neighbors + 1

        self.tree = None

        self.y = None

        self.classes = None

        

    def fit(self, x, y):

        self.tree = KDTree(x)

        self.y = y

        self.classes = np.unique(y)

        

    def predict_proba(self, x):

        dist, idx = self.tree.query(x, self.n_neighbors, return_distance=True)

        if idx.shape[0] == 0:

            return np.zeros((x.shape[0], self.classes.shape[0]))

        pred = np.zeros((x.shape[0], self.classes.shape[0]))

        kernel = self.kernel(dist/dist.max(axis=1).reshape(-1, 1))

        for i in range(idx.shape[0]):

            row_target = self.y[idx[i]]

            for j, cl in enumerate(self.classes):

                cl_id = np.where(row_target == cl)[0]

                if cl_id.shape[0] > 0:

                    pred[i, j] = kernel[i][cl_id].sum()

        pred_sum = pred.sum(axis=1).reshape(-1, 1)

        pred_sum[pred_sum == 0] = 1

        pred /= pred_sum

        return pred

    

    def predict(self, x):

        proba = self.predict_proba(x)

        unnknown_idx = proba.sum(axis=1) == 0

        predict = np.argmax(proba, axis=1)

        predict[unnknown_idx] = -1

        return predict

    

    def get_margin(self, x, y):

        dist, idx = self.tree.query(np.expand_dims(x, 0), self.n_neighbors, return_distance=True)

        idx = idx[:, 1:]

        dist = dist[:, 1:]

        kernel = self.kernel(dist/dist.max(axis=1).reshape(-1, 1))

        current_classes = self.y[idx]

        

        kernel_positive = kernel[current_classes == y]

        kernel_positive = 0 if len(kernel_positive) == 0 else kernel_positive.sum()

        

        kernel_negative_best = 0

        for y_uniq in self.classes:

            if y_uniq != y:

                kernel_negative = kernel[current_classes == y_uniq]

                kernel_negative = 0 if len(kernel_negative) == 0 else kernel_negative.sum()

                kernel_negative_best = max(kernel_negative, kernel_negative_best)

        

        return kernel_positive - kernel_negative

            

    @staticmethod

    def kernel(r):

        k = 3/4*(1 - r**2)

        k[k < 0] = 0

        return k
est = ParsenWindowEpachnikovNeighbors(85)

scaler = StandardScaler()

x = scaler.fit_transform(df.drop('Target', axis=1))
est.fit(x, df['Target'].values)
margins = np.array([est.get_margin(x[i], df['Target'].iloc[i]) for i in range(x.shape[0])])
sns.barplot(np.arange(x.shape[0]), np.sort(margins));
y = df['Target'].values
def get_best(margins, y, ftype='max'):

    best_idx = []

    for y_uniq in np.unique(y):

        idx = np.where(y == y_uniq)[0]

        if ftype == 'max':

            best_id = idx[np.argmax(margins[idx])]

        else:

            best_id = idx[np.argmin(margins[idx])]

        best_idx.append(best_id)

    return best_idx
delta = 0.1

wrong = 0.02

est = ParsenWindowEpachnikovNeighbors(85)

est.fit(x, y)

margins = np.array([est.get_margin(x[i], df['Target'].iloc[i]) for i in range(x.shape[0])])

x_data = x[margins > delta]

y_data = y[margins > delta]

margins = margins[margins > delta]

best_idx = get_best(margins, y_data)

x_best = x_data[best_idx]

y_best = y_data[best_idx]

other_idx = [i for i in range(x_data.shape[0]) if i not in best_idx]

x_data = x_data[other_idx]

y_data = y_data[other_idx]

margins = margins[other_idx]

while True:

    est = ParsenWindowEpachnikovNeighbors(x_best.shape[0]//2)

    est.fit(x_best, y_best)

    predictions = est.predict(x)

    errors = (y != predictions).sum()/y.shape[0]

    if errors < wrong or x_data.shape[0] == 0:

        break

    else:

        best_idx = get_best(margins, y_data)

        x_best = np.append(x_best, x_data[best_idx], axis=0)

        y_best = np.append(y_best, y_data[best_idx])

        other_idx = [i for i in range(x_data.shape[0]) if i not in best_idx]

        x_data = x_data[other_idx]

        y_data = y_data[other_idx]

        margins = np.array([est.get_margin(x_data[i], y_data[i]) for i in range(x_data.shape[0])])
x_best.shape
(y != predictions).sum()/y.shape[0]
tsne = TSNE(n_components=2)
visual = tsne.fit_transform(x_best)
colors = ['red', 'blue', 'green']

plt.scatter(x=visual[:, 0], y=visual[:, 1], color=[colors[val] for val in y_best]);
delta = 0.1

wrong = 0.02

est = ParsenWindowEpachnikovNeighbors(85)

est.fit(x, y)

margins = np.array([est.get_margin(x[i], df['Target'].iloc[i]) for i in range(x.shape[0])])

x_data = x[margins > delta]

y_data = y[margins > delta]

margins = margins[margins > delta]

best_idx = get_best(margins, y_data)

x_best = x_data[best_idx]

y_best = y_data[best_idx]

other_idx = [i for i in range(x_data.shape[0]) if i not in best_idx]

x_data = x_data[other_idx]

y_data = y_data[other_idx]

margins = margins[other_idx]

while True:

    est = ParsenWindowEpachnikovNeighbors(x_best.shape[0]//2)

    est.fit(x_best, y_best)

    predictions = est.predict(x)

    errors = (y != predictions).sum()/y.shape[0]

    if errors < wrong or x_data.shape[0] == 0:

        break

    else:

        best_idx = get_best(margins, y_data, ftype='min')

        x_best = np.append(x_best, x_data[best_idx], axis=0)

        y_best = np.append(y_best, y_data[best_idx])

        other_idx = [i for i in range(x_data.shape[0]) if i not in best_idx]

        x_data = x_data[other_idx]

        y_data = y_data[other_idx]

        margins = np.array([est.get_margin(x_data[i], y_data[i]) for i in range(x_data.shape[0])])
x_best.shape
(y != predictions).sum()/y.shape[0]
tsne = TSNE(n_components=2)
visual = tsne.fit_transform(x_best)
colors = ['red', 'blue', 'green']

plt.scatter(x=visual[:, 0], y=visual[:, 1], color=[colors[val] for val in y_best]);
clf = ParsenWindowEpachnikovNeighbors(n_neighbors=1)

clf.fit(visual, y_best)

x_min, x_max = visual[:, 0].min() - 10, visual[:, 0].max() + 10

y_min, y_max = visual[:, 1].min() - 10, visual[:, 1].max() + 10

xx, yy = np.meshgrid(np.arange(x_min, x_max, 1), np.arange(y_min, y_max, 1))

predict = clf.predict(np.c_[xx.ravel(), yy.ravel()])

predict = predict.reshape(xx.shape)

plt.pcolormesh(xx, yy, predict, cmap='terrain');

sns.scatterplot(visual[:, 0], visual[:, 1], hue=y_best, palette=['red', 'blue', 'green']);