import matplotlib.pyplot as plt
import numpy as np


%matplotlib inline

x = [1, 2, 3, 4, 5, 6, 7]
y = [3.5, 3.8, 4.2, 4.5, 5, 5.5, 7]

plt.plot(x,y)
plt.show()
plt.scatter(x,y)
plt.show()
t = np.linspace(0, 10, num=51)

f = np.array(np.cos(t))
%config InlineBackend.figure_format = 'svg'

plt.plot(t,f, 'g')

title_font= {
    "fontsize": 16,
    "fontweight": "bold",
    "color": "black",
    "family": "serif"
    }
    
label_font = {
    "fontsize": 9,
    "family": "serif",
    }

plt.title("График f(t)", fontdict=title_font)
plt.xlabel("Значения t", fontdict=label_font)
plt.ylabel("Значения f", fontdict=label_font)

plt.axis([0.5, 9.5, -2.5, 2.5])

plt.show()

x = np.linspace(-3, 3, 51)
y1 = x ** 2
y2 = 2 * x + 0.5
y3 = -3 * x - 1.5
y4 = np.sin(x)
fig, ax = plt.subplots(nrows=2, ncols=2)

fig.set_size_inches(8, 6)
fig.subplots_adjust(wspace=0.3, hspace=0.3)

ax1, ax2, ax3, ax4 = ax.flatten()

ax1.plot(x, y1)
ax1.set_xlim([-5, 5])
ax1.set_title('График y1')

ax2.plot(x, y2)
ax2.set_title('График y2')

ax3.plot(x, y3)
ax3.set_title('График y3')

ax4.plot(x, y4)
ax4.set_title('График y4')
import pandas as pd

plt.style.use('fivethirtyeight')
df = pd.read_csv('./creditcard.csv')
df.iloc[:10, [0, 1, 2, 3, 4, 28, 29, 30]]
class_count = df['Class'].value_counts()
class_count
class_count.plot(kind='bar')
plt.show()
class_count.plot(kind='bar', logy=True)
plt.show()
fraud_transaction = df.loc[df['Class']==1, 'V1']
normal_transaction= df.loc[df['Class']==0, 'V1']

hist = plt.hist([fraud_transaction, normal_transaction], bins=20, density=True, color=['red','grey'], alpha=0.5)

label_font = {
    "fontsize": 12,
    "fontweight":"light",
    "family": "serif",
}
plt.xlabel('Class', fontdict=label_font, labelpad=20)
plt.legend(labels=['Class1', 'Class0'], frameon=False)
plt.show
a = np.arange(12,24)
a
a = a.reshape(2,6)
a
b = a.argsort(axis=0)
b
c = a + a
c
d = a.reshape(6,2) + np.ones((6, 2))
d
e = np.sqrt(a.reshape(6,2))
e
a1 = a.reshape(2,-1)
a1
b1 = a.reshape(12,-1)
b1
c1 = a.reshape(6,-1)
c1
a1 = a.reshape(-1,3)
a1
a1 = a.reshape(-1,1)
a1
A = np.array([1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12])
B = A.reshape (12,1)
A.ndim == B.ndim
A = np.random.randn(3, 4)
A.size
B = A.flatten()
B.size
A = np.arange(20, 0, -2)
print(A)
A.ndim, A.shape, A.size
B = np.arange(20, 1, -2)
B.reshape(1,10)
print(B)
B.ndim, B.shape, B.size
a = np.zeros((2, 2))

b = np.ones((3, 2))

v = np.concatenate((a, b), axis=0)
v.size
A = np.arange(12).reshape(4, 3)
A
At = A.T
At
B = A.dot(At)
print(B)
B.size
print (f'определитель матрицы равен: {np.linalg.det(B)}')
np.random.seed(42)

c = np.random.randint(0, 16, 16)
c
C = c.reshape(4, 4)
C.shape
D = B + C * 10
D
D_d = np.linalg.det(D)
np.linalg.matrix_rank(D)

D_inv = np.linalg.inv(D)
D_inv
D_inv = np.where (D_inv < 0, 0, 1)
D_inv

print(f'B : \n{B}\n\n C : \n{C}')
E = np.where(D_inv > 0, B, C)
print(E)