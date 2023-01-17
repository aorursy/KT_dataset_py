import numpy as np
print(np.__version__) 
a = np.zeros(10)

a[4] = 1

a[-3:] = 2

print(a)
b = np.random.uniform(150, 200, 100)

print(b)

c = np.mean(b)

print(c)

d = np.std(b)

print(d)

e = b[(c - 3*d < b) & (b < c + 3*d)]

print((e.size/b.size)*100)
f = np.linspace(0, np.pi/2, 20)

g = np.cos(f)

h = np.sin(np.pi/2 - f)

def close(x, y, eps):

    if np.all(np.abs(x - y)) < eps:

        return True

    else:

        return False

print(close(g, h, eps = 0.0001))
A = np.random.normal(5, 2.5, (10, 5)) #1

i = np.mean(A[4, :]) #2

j = np.median(A[:, 2]) #3

k = np.sum(A, axis = 1) #4

l = np.sum(A, axis = 0)

m = np.sum(A)

n = (A[:3, :3]) #5

print(np.linalg.det(n))



#6

print(len(list(filter(lambda x: np.sum(x) < np.mean([np.sum(t) for t in A]), A))))



#7

print(np.array(list(filter(lambda x: x[0] > x[-1], A))).shape)



#8

rank = np.linalg.matrix_rank(list(filter(lambda x: np.sum(x) > np.mean(A), A)))

print(rank)
dataset = np.loadtxt('../input/cardio_train.csv',delimiter=';', skiprows=1) #1

print(dataset.shape) #2

dataset[:, 1] = np.round(dataset[:, 1]/365.25) #3

print(dataset[:, 1])

k = dataset[:, 1] #4

s = 0

for i in range(dataset.shape[0]):

    if k[i] > 50:

        s = s + 1

print(s)

z = dataset[:, 2:4] #columns 2-3

m = z[z[:, 0] == 2, :] #z[:, 0] = dataset[:, 2]

f = z[z[:, 0] == 1, :] 

print(np.round(np.mean(m[:, 1]))) #male, avg of second column of z (dataset[:, 3])

print(np.round(np.mean(f[:, 1]))) #female

wei = dataset[:, 4]

print(np.min(wei), np.max(wei))

xxx = wei[wei == 200] #from wei, where wei == 200

print(xxx.size)

smoke = dataset[:, -4] #need 0

heart = dataset[:, -1] # need 1

combo = smoke + heart # 0 + 1

print(combo[combo == 1].size)