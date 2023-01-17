a = [1,2,3,4]
b = [2,3,4,5]

ab = []                        

for i in range(0, len(a)):
  ab.append(a[i]*b[i])

ab
import numpy as np 

a = np.array([1,2,3,4])

b = np.array([2,3,4,5])

a * b
x = 9
who
x
from IPython.display import Image
path = "/Users/mvahit/Documents/GitHub/pydsml/img/"
Image(filename = path + "python_vs_c.png", width=400, height=400)
L = [1,2,"a",1.2]
L
[type(i) for i in L]
from IPython.display import Image
path = "/Users/mvahit/Documents/GitHub/pydsml/img/"
Image(filename = path + "array_vs_list.png", width=400, height=400)
import numpy as np 

np.array([12,33,4,5])

a = np.array([12,33,4,5])

a

type(a)
np.array([3.14, 4,6,1.2])
np.array([3.14, 4,6,1.2], dtype = 'float32')
np.zeros(10, dtype = int)
np.ones((2,3))
np.full((2,3), 9)
np.arange(0,10, 2)
np.linspace(0,1,30)
np.random.normal(0,1, (3,4))
np.random.randint(0,10, (2,2))
np.eye(3)
import numpy as np
a = np.random.randint(10, size = 10)
a.ndim
a.shape
a.size
a.dtype
a = np.random.randint(10, size = (10, 5))
a
a.ndim
a.shape
a.size
a.dtype
a = np.random.randint(10, size = (3,5,2))
a
a.ndim
a.shape
a.size
a.dtype

np.arange(1,10)
np.arange(1,10).reshape((3,3))
a = np.array([1,2,3])
a
b = a.reshape((1,3))
b.ndim
a
a[np.newaxis, :]
a[:,np.newaxis]
#tek boyutlu
x = np.array([1, 2, 3])
y = np.array([4, 5, 6])
x
y
np.concatenate([x,y])
z = [1,2,3]
np.concatenate([x,y,z])
#iki boyutlu
a = np.array([[1, 2, 3], 
              [4, 5, 6]])

a
np.concatenate([a,a])
np.concatenate([a,a], axis = 1)
#farklı boyutlu 
a = np.array([1, 2, 3])

b = np.array([[9, 8, 7], 
              [6, 5, 4]])
a
b
np.vstack([a,b])
a = np.array([[99],
              [99]])
a
np.hstack([a,b])
#array ayirma splitting
x = [1, 2, 3, 99, 99, 3, 2, 1]
x
np.split(x, [3,5])
a,b,c = np.split(x, [3,5])
a
b
c

m = np.arange(16).reshape((4, 4))
np.vsplit(m, [2])
ust, alt = np.vsplit(m, [2])
ust
alt
np.vsplit(m, [2])
m
sag, sol = np.hsplit(m, [2])
sag
sol
#array siralama 
v = np.array([2, 1, 4, 3, 5])
v
np.sort(v)
v
v.sort()
v
v = np.array([2, 1, 4, 3, 5])
np.sort(v)
i = np.argsort(v)
i
v
v[i]
import numpy as np
a = np.random.randint(10, size = 10)
a
a[0]
a[-2]
a[0] = 1
a
a = np.random.randint(10, size = (3, 5))
a
a[0,0]
a[1,1]
a[0,0] = 1
a[0,0]
a[0,0] = 2.2
a[0,0]
a = np.arange(20,30)
a
a[0:3]
a[:3]
a[3:]
a[::2]
a[1::2]
a[2::2]
a[1::3]
a = np.random.randint(10, size = (5, 5))
a
a[:,0]
a[:,1]
a[:,4]
a
a[4,:]
a[0]
a[:2,:3]
a[0:2,0:3]
a[::,:2]
a[:,:2]
a[1:3,0:2]
a = np.random.randint(10, size = (5, 5))
a
alt_a = a[0:3,0:2]
alt_a
alt_a[0,0] = 9999
alt_a[1,1] = 9999
alt_a
a

alt_b = a[0:3,0:2].copy()
alt_b
alt_b[0,0] = 9999
alt_b[1,1] = 9999
a
v = np.arange(0, 30, 3)
v
v[1]
v[3]
[v[1], v[3]]
al_getir = [1,3,5]
v[al_getir]
m = np.arange(9).reshape((3, 3))
m
satir = np.array([0,1])
sutun = np.array([1,2])
m[satir, sutun]
m
m[0, [1,2]]
m[0:,[1,2]]

v = np.arange(10)
v
index = np.array([0,1,2])
index
v[index] = 99
v
v[[0,1]] = [4,6]
v
v = np.array([1,2,3,4,5])
v > 3
v < 3
v <= 3
v == 3
v != 3
(2 * v)
(v ** 2)
(2 * v) == (v ** 2)
#ufunc
v = np.array([1,2,3,4,5])
np.equal(3,v)
np.equal([0,1,3], np.arange(3))
np.arange(3)
v = np.random.randint(0, 10, (3, 3))
v
v > 5
np.sum(v > 5)
np.sum((v > 3) & (v < 7))
(v > 3) & (v < 7)
np.sum((v > 3) | (v < 7))

np.sum(v > 4, axis = 1)
np.sum(v > 4, axis = 0)
np.all(v > 4)
np.all(v > 0)
np.any(v > 4)
np.all(v > 4, axis = 0)
np.all(v > 4, axis = 1)
v = np.array([1,2,3,4,5])
v
v[0]
v > 3
v[(v > 1) & (v < 5)]
a = np.arange(5)
a
a - 1
a / 2
a * 5 
a ** 2
a % 2
5*(a/2*9)
a
np.add(a,2)
np.subtract(a,1)
np.multiply(a,3)
np.divide(a,3)
np.power(a,3)
a = np.arange(1,6)
a
np.add.reduce(a)
np.add.accumulate(a)
a = np.random.normal(0, 1, 30)
a
np.mean(a)
np.std(a)
np.var(a)
np.median(a)
sum(a)
np.min(a)
np.max(a)
a = np.random.normal(0, 1, (3,3))
a
a.sum()
a.sum(axis = 1)
import numpy as np

a = np.array([1, 2, 3])
b = np.array([1, 2, 3])

a + b
m = np.ones((3, 3))
m
a
a + m
a = np.arange(3)
a
b = np.arange(3)[:, np.newaxis]
b
a + b
M = np.ones((2, 3))
M
a = np.arange(3)
a
a + M

a = np.arange(3).reshape((3, 1))
a
b = np.arange(3)
b
a.shape
b.shape
a + b
m = np.ones((3,2))
m
a = np.arange(3)
a
m.shape
a.shape
a + m
isim = ['ali', 'veli', 'isik']
yas = [25, 22, 19]
boy = [168, 159, 172]
data = np.zeros(3, dtype={'names':('isim', 'yas', 'boy'),
                          'formats':('U10', 'i4', 'f8')})
data
data['isim'] = isim
data['yas'] = yas
data['boy'] = boy
data
data['isim']
data[0]
data[data['yas'] < 25]['isim']
import numpy as np
import pandas as pd
pd.Series([1,2,3,4])
seri = pd.Series([1,2,3,4])
type(seri)
seri.axes
seri.dtype
seri.empty
seri.ndim
seri.size
seri.values
seri.head(2)
seri[0:3]
seri.tail(2)
a = np.array([1,2,3,566,88])
a
type(a)
seri = pd.Series(a)
seri
seri.index
seri[0]
pd.Series([1,5,0.9,34], index = [1,3,5,7])
seri = pd.Series([1,5,0.9,34], index = ['a','b','c','d'])
seri["a"]
sozluk = { "reg" : 10, "loj" : 11, "cart" : 12}
sozluk
seri = pd.Series(sozluk)
seri["reg"]
seri["loj":"cart"]
seri
pd.concat([seri, seri])
seri.append(seri)
seri
a = np.array([1, 2, 3,33,55,66])
seri = pd.Series(a)
seri
seri[0]
seri[0:3]
seri = pd.Series([121, 200, 150, 99], 
                 index=["reg", "loj", "cart", "rf"])
seri
seri.index
seri.keys
list(seri.items())
seri.values
seri
"reg" in seri
seri["reg"]
seri[['rf', 'reg']]
seri["rf"] = 125
seri

seri['loj': 'rf']
seri[0:2]

seri = pd.Series([121, 200, 150, 99], index=["reg", "loj", "cart", "rf"])
seri
seri[(seri > 125) & (seri < 200)]

data = pd.Series(['a', 'b', 'c'], index=[1, 3, 5])
data
data[0]
data[1]
data[1:3]
data[0:3]
# loc label based indexing, tanımlandığı şekli ile index yakalamak
data
data.loc[0:5]
# iloc positional indexing, indexi sıfırlayarak yakalamak
data
data.iloc[2]
data.iloc[0:3]