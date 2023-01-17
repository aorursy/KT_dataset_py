#Pembuatan 2 variabel berisi matrik 3x3

import numpy as np
a = np.array([[2,4,6],[2,2,2],[5,5,5]])
b = np.array([[1,4,6],[3,3,3],[6,8,10]])
print("Array 3x3 a = ")
print(a)
print("\nArray 3x3 b = ")
print(b)

#Penjumlahan 2 buah matrik berordo 3x3
c = a+b

print(c)
#Pengurangan 2 buah matrik berordo 3x3
d = b-a

print(d)
#Perkalian 2 buah matrik berordo 3x3
e = a*b

print(e)
#Mencari mean(rata-rata) 2 buah matrik berordo 3x3
f = np.mean(a)
g = np.mean(b)

print(f)
print(g)
#Mencari median(nilai tengah) 2 buah matrik berordo 3x3
h = np.median(a)
i = np.median(b)

print(h)
print(i)
#Mencari nilai min  dari 2 buah marik berordo 3x3
j = np.min(a)
k = np.min(b)

print(j)
print(k)
#Mencari nilai max  dari 2 buah marik berordo 3x3
l = np.max(a)
m = np.max(b)

print(l)
print(m)
