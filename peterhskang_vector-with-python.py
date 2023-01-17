# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # numpy
from numpy import linalg as LA # linear algebra
from fractions import Fraction
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# R^2
v1 = np.array([1,2])
v2 = np.array([2,1])
print(v1, v2)

# R^3
v3 = np.array([1,2,3])
v4 = np.array([3,2,1])
print(v3, v4)

# R^n
v5 = np.arange(1,10)
print(v5)

# norm (벡터 크기)
v = np.array([3, 4])
print(LA.norm(v))

# 정규화 단위벡터
v_hat = v / LA.norm(v)
print(v_hat, LA.norm(v_hat) == 1)

# 표준단위벡터
e1 = np.array([1,0])
e2 = np.array([0,1])
# numpy.eye()
e = np.eye(2)
print(e[0] == e1, e[1] == e2)
print(np.eye(3))

# 표준단위벡터로 표현
ve = v[0] * e1 + v[1] * e2
print(ve == v)
v = np.array([1,2])
w = np.array([2,1])

# 덧셈과 뺄셈
print(v + w, v - w)

# 실수배
kv = 2 * v;
print(kv)

v1 = np.array([1,2])
v2 = np.array([3,4])
v_linear = 2*v1 + 3*v2
print(v_linear)

v1 = np.array([1,1])
v2 = np.array([1,2])
inner_product = np.inner(v1,v2)
print(inner_product)
print("sin 0: {:.2f}".format(np.sin(0/180)))
print("sin 30: {:.2f}".format(np.sin(30*np.pi/180)))
print("sin 45: {:.2f}".format(np.sin(45*np.pi/180)))
print("sin 60: {:.2f}".format(np.sin(60*np.pi/180)))
print("sin 90: {:.2f}".format(np.sin(90*np.pi/180)))
print("cos 0: {:.2f}".format(np.cos(0/180)))
print("cos 30: {:.2f}".format(np.cos(30*np.pi/180)))
print("cos 45: {:.2f}".format(np.cos(45*np.pi/180)))
print("cos 60: {:.2f}".format(np.cos(60*np.pi/180)))
print("cos 90: {:.2f}".format(np.cos(90*np.pi/180)))
print("tan 0: {:.2f}".format(np.tan(0/180)))
print("tan 30: {:.2f}".format(np.tan(30*np.pi/180)))
print("tan 45: {:.2f}".format(np.tan(45*np.pi/180)))
print("tan 60: {:.2f}".format(np.tan(60*np.pi/180)))
print("tan 90: {:.2f}".format(np.tan(90*np.pi/180)))

u = np.array([1,5])
v = np.array([2,3])
w = np.array([3,8])
k, m = 2, 3

# 가감배
print(v + w == w + v)
print((u + v) + w == u + (v + w))
print(v + 0 == v)
print(v - v == 0)
print(k*(u + v) == k*u + k*v)
print((k + m)*v == k*v + m*v)
print(k*(m*v) == (k*m)*v)
print(1*v == v)
print(0*v == 0)

# 스칼라곱
print(np.inner(u,v) == np.inner(v,u))
print(np.inner(0,v) == 0)
print(np.inner(u,(v + w)) == np.inner(u,v) + np.inner(u,w))
print(np.inner((u + v),w) == np.inner(u,w) + np.inner(v,w))
print(k*np.inner(u,v) == np.inner(k*u,v) == np.inner(u,k*v))
# (u•v)•w != u•(v•w)
print(np.inner(np.inner(u,v),w)) # python은 연산 가능하지만 결합법칙은 불가능
print(np.inner(u, np.inner(v,w)))
# R_^3
v = np.array([1,2,3])
w = np.array([4,5,6])
cross_product = np.cross(v, w)
print(cross_product)
print(-cross_product == np.cross(w, v))
u = np.array([1,2,3])
v = np.array([4,5,6])
w = np.array([7,8,9])
z = np.zeros(3)
k = 7
print(np.cross(u,v) == np.cross(-v,u))
print(np.cross(u,(v+w)) == (np.cross(u,v) + np.cross(u,w)))
print(np.cross((u+v),w) == (np.cross(u,w) + np.cross(v,w)))
print((k*np.cross(u,v)) == np.cross(k*u,v))
print(np.cross(k*u,v) == np.cross(u,k*v))
print(np.cross(u,z) == 0)
print(np.cross(u,u) == 0)
# 두 벡터의 cosθ 값을 구하라
# ||v|| ||u|| cosθ = v ⋅ y
u = np.array([1,1])
v = np.array([0,1])
cos_theta = np.inner(u,v) / (LA.norm(u) * LA.norm(v))
print("{:.2f}".format(cos_theta))

# 두 벡터가 이루는 평행사변형의 넓이를 구하라
# u × v = w
# 평행사변형 넓이: ||w||
u = np.array([2,3,0])
v = np.array([-1,2,-2])
cross = np.cross(u,v)
norm = np.linalg.norm(cross)
print(norm)

# 두 점을 지나는 직선이 세 점을 지나는 평면과 만나는 교점과 좌표를 구하라
dots = np.array([[-1,-1,0],[2,0,1]])



## R_^2에서 1차 방정식 구하기
# 두 점 (x1, y1), (x2, y2)를 지나는 직선 방정식을
# ax + by = 1이라고 할 때,
# (x1 y1) (a) = (1)
# (x2 y2) (b) = (1)
# <=> (a) = (x1 y1)-1 (1)
#     (b) = (x2 y2)   (1)
#
# e.g. (0,1), (1,3)을 지나는 방정식 구하기
# (y = 2x + 1 <=> -2x + y = 1)

points = np.array([[0,1],[1,3]])
#points = np.array([[1,1],[2,2]])
det = round(LA.det(points), 8)
if (det != 0):
    C = LA.inv(points).dot(np.ones(2))
    print(C)
else:
    # TODO    
    print("det is zero.")

## R_^3에서 1차 방정식 구하기
# e.g. ()
points = np.array([[-1,-1,0],[2,0,1],[4,0,2]])
det = round(LA.det(points), 8)
if (det != 0):
    C = LA.pinv(points).dot(np.ones(3))
    print(C)
else:
    # TODO    
    print("det is zero.")

# 1차원 방정식 해 찾기
# x + 2y = 4
# x - y = 1
# AX = B
# A: 계수행렬, B: 상수행렬
# <=> X = A_-1 B
A = np.array([[1,2],[1,-1]])
B = np.array([4,1])
A_inv = LA.inv(A)
X = A_inv.dot(B)
print(X)
X = LA.solve(A,B)
print(X)