import numpy as np
a = np.array([1,2,3,4,5])
print (a)
print("\n")
b = np.array([[1,2,3],[4,5,6]])

print(b)
print("\n")
bool_ar = np.array([1,0,'a',0.5,0],dtype=bool)
print(bool_ar)
print("\n")
arr = np.array([[1,2,3],[4,5,6]])
for i in arr:
    for j in i:
        if j%2 != 0:
            print(j, end=" ")
print("\n")
arr = np.array([1,4,16,3,22,9,11,15,54])
arr = np.where(arr>=15,0,arr)

print(arr)
print("\n")

a = np.array([1,2,3,4,5,6])
b = np.array([2,4,6])
c = np.intersect1d(a,b)

print(c)
print("\n")

a = np.array([1,2,3,4,5,6])
b = np.array([2,4,6])

for i in b:
    for j in a:
        if i == j:
            a = a[a!=j]
print(a)
print("\n")
a = np.array([1,2,3,4,5,6])
b = np.array([2,4,6])

for i in b:
    for j in a:
        if i == j:
            index = np.argwhere(a==j)
            print(index)
