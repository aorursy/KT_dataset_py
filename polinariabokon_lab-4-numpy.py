import numpy as np
np.__version__
#1

x = np.random.normal(170, 20,120)

print(x)
a = min(x) #минимальный

b = max(x) #максимальный1

delc = np.delete(x, np.argmax(x)) #удаляем b , масив без b

c = max(delc) #максимальный2

deld = np.delete(delc, np.argmax(delc))

d = max(deld)

print("Минимальный:",a,'Три максимальных:',b,c,d)
mean = np.mean(x) #среднее значение

median = np.median(x) #медиана

print(mean/median )
mean = np.mean(x) #среднее значение

no_mean = x[x > mean]

print(no_mean)

u = x.size

print(u)

v = no_mean.size

print(v)

print((v*100)/u)
print(np.all(x > 140)) #Верно ли, что все значения в выборке больше 140?
n = x[x < 160] #низкие

s = x[(x > 160) & (x < 175)] #средние

v = x[x > 175] #высокие

n1 = np.sort(n)

s1 = np.sort(s)

v1 = np.sort(v)

n1[:] = 1

s1[:] = 2

v1[:] = 3

print(n1, s1, v1)
z = np.append(x, np.random.uniform(150, 190, 10))

print(z, z.size)
#2

A = np.random.randint(0, 10, (20, 5))

print(A);
A1 = A[:, :2]

print(A1)
A10 = A[-10:, :]

print(A10);
 #в третьей строке

Amean =  np.mean(A, axis=1) # в каждой строке

A3 = A[Amean>5, :]

print(A3)

print(Amean)
print (max(A[:, -1]))
mean1 = np.mean(A[:, 0])

mean2 = np.mean(A[:, -1])

print(mean1, mean2)

print(np.any(mean1 > mean2))
mj = np.max(A, axis=0)

print(mj)

B = mj - A

print(B)

print(B[ : , B[-1, :]>=0])

Bmean = np.mean(B, axis=1)

for x in Bmean:

    if x<5:

        print(x)

print(Bmean)
S = B.transpose()

C = np.delete(S, [0, 1, 2,3,4] ,axis=1)

print(S)

print(C)
S = B.transpose()

print(S[:,5:])

K =  np.ones(20)

print(K)

D = np.hstack([K[:, np.newaxis], B])  # добавление к матрице столбец из 1 

print(D)
Z = np.zeros((5, 5))

E = np.vstack([Z, B])

print(E)