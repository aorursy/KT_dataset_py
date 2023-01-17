# 데이터 전처리
import numpy as np
import pandas as pd

from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.preprocessing import StandardScaler

# 기계학습 모델 생성, 학습, 평가
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier

# 시각화
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
random_state = 2020
n_samples = 100
datasets = [
    make_moons(n_samples=n_samples, noise=0.2, random_state=random_state),
    make_circles(n_samples=n_samples, noise=0.2, factor=0.5, random_state=random_state),
    make_classification(n_samples=n_samples, n_features=2, n_redundant=0,
                        n_informative=2, n_clusters_per_class=1, random_state=random_state)
]
datasets_names = ['moon', 'circle', 'classification']
X, y = datasets[0]
print('X ----------\n', X[:10])
print('y ----------\n', y[:10])
cmap = ListedColormap(['royalblue', 'orangered'])
fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(18, 5))
axs = axs.ravel()

for i, (dataset, dataset_name) in enumerate(zip(datasets, datasets_names)):
    X, y = dataset
    
    axs[i].scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', cmap=cmap)
    axs[i].set_title(dataset_name, fontsize=15)
    
plt.show()
# 데이터 선정
X, y = datasets[0]
dataset_name = datasets_names[0]

# train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

# Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test= scaler.transform(X_test)
#sklearn =>tree => DecisionTreeClassifier
model = DecisionTreeClassifier(max_depth=None)
model.fit(X_train, y_train)
# train, test acc
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

acc_train = accuracy_score(y_train, y_train_pred)
acc_test = accuracy_score(y_test, y_test_pred)
#aa=1
#print("문자열".format(aa))
print('Training Accuracy: {:.3f}'.format(acc_train))
print('Testing Accuracy: {:.3f}'.format(acc_test))
# 시각화를 위한 격자 생성
X = scaler.transform(X)

x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02), np.arange(y_min, y_max, 0.02))

grid = np.hstack([xx.reshape(-1, 1), yy.reshape(-1, 1)])
# 격자 공간에 대한 예측 확률값
y_pred_prob = model.predict_proba(grid)[:, 1]

# Contour
Z = y_pred_prob.reshape(xx.shape)
# 시각화: contour를 먼저 그리고, test는 약간 투명하게 표기
plt.figure(figsize=(5, 5))

plt.contourf(xx, yy, Z, alpha=0.5, cmap=cmap)
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, edgecolors='k', cmap=cmap)
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, edgecolors='k', alpha=0.5, cmap=cmap)
plt.title('Train Acc = {:.3f} & Test Acc = {:.3f}'.format(acc_train, acc_test))

plt.show()
max_depths = [1, 2, 3, None]
fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(12, 12))
axs = axs.ravel()

for i, max_depth in enumerate(max_depths):
    
    # 모델 학습
    model = DecisionTreeClassifier(max_depth=max_depth)
    model.fit(X_train, y_train)
    
    # 예측
    # 1. train, test data
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    acc_train = accuracy_score(y_train, y_train_pred)
    acc_test = accuracy_score(y_test, y_test_pred)
    
    # 2. grid
    y_pred_prob = model.predict_proba(grid)[:, 1]
    Z = y_pred_prob.reshape(xx.shape)
    
    # plot
    axs[i].contourf(xx, yy, Z, alpha=0.5, cmap=cmap)
    axs[i].scatter(X_train[:, 0], X_train[:, 1], c=y_train, edgecolors='k', cmap=cmap)
    axs[i].scatter(X_test[:, 0], X_test[:, 1], c=y_test, edgecolors='k', alpha=0.5, cmap=cmap)
    axs[i].set_title('[{}] Train Acc = {:.3f} & Test Acc = {:.3f}'.format(max_depth, acc_train, acc_test))
    
plt.show()
