# 데이터 전처리
import numpy as np
import pandas as pd

from sklearn.datasets import make_regression
#데이터 전처리, 평균과 표준편차가 각각 0,1이 되도록 변환해주는 모듈
from sklearn.preprocessing import StandardScaler

# 기계학습 모델 생성, 학습, 평가
from sklearn.tree import DecisionTreeRegressor

# 시각화
import matplotlib.pyplot as plt
random_state = 2020
n_samples = 50
np.random.seed(random_state)

X = np.random.rand(n_samples, 1) * 10
print(X)
X = np.sort(X, axis=0)
print(X)
y = np.sin(X).reshape(-1, )
y[::5] = y[::5] + np.random.randn(int(np.ceil(n_samples/5)))
X_test = np.arange(0, 10, 0.01)
X_test.shape
X_test = np.arange(0, 10, 0.01).reshape(-1, 1)
X_test.shape
# Scaling
plt.figure(figsize=(8, 5))
plt.scatter(X, y, s=20, edgecolor='black', c='lightblue', label='data')
plt.legend(loc='lower right')
plt.show()
model = DecisionTreeRegressor(max_depth=None)
model.fit(X, y)
y_pred = model.predict(X_test)
y_pred
# 원본 데이터 및 결과 시각화
plt.figure(figsize=(8, 5))

plt.plot(X_test, y_pred, label='max_depth=None', linewidth=2)
plt.scatter(X, y, s=20, edgecolor='black', c='lightblue', label='data')

plt.xlabel('data')
plt.ylabel('target')
plt.legend(loc='lower right')
plt.show()
max_depths = [3, 5, None]
#max_depths = [3]
#max_depths = [None]
plt.figure(figsize=(12, 6))
plt.scatter(X, y, s=20, edgecolor='black', c='lightblue', label='data')
for i, max_depth in enumerate(max_depths):
    
    # 모델 학습
    model = DecisionTreeRegressor(max_depth=max_depth)
    model.fit(X, y)
    
    # 예측
    y_pred = model.predict(X_test)
    
    # plot
    plt.plot(X_test, y_pred, label='max_depth={}'.format(max_depth), linewidth=1.5, alpha=0.7)
    
plt.legend(loc='lower right')
plt.show()
dataset = make_regression(n_samples=n_samples, n_features=2,
                          n_informative=2, n_targets=1, random_state=random_state)
X, y = dataset
X
y
scaler = StandardScaler()
X = scaler.fit_transform(X)
# 시각화를 위한 격자 생성
x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02), np.arange(y_min, y_max, 0.02))

grid = np.hstack([xx.reshape(-1, 1), yy.reshape(-1, 1)])
model = DecisionTreeRegressor(max_depth=None)
model.fit(X, y)
# 격자 공간에 대한 예측 확률값
y_pred = model.predict(grid)

# Contour
Z = y_pred.reshape(xx.shape)
# 시각화
plt.figure(figsize=(5, 5))
#결과만
plt.contourf(xx, yy, Z, alpha=0.5, cmap='Blues')
#데이터만
plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', cmap='Blues')
plt.show()
