import numpy as np
import matplotlib.pyplot as plt
C = np.array([[0.022,0.01,-0.001,0.011,0.005],[0.01,0.033,0,0.014,0.01],[-0.001,0,0.019,-0.001,-0.001],[0.011,0.014,-0.001,0.064,0.011],[0.005,0.01,-0.001,0.011,0.045]])
print(C)
C_invers = np.linalg.inv(C)
print(C_invers)
u = np.array([[1],[1],[1],[1],[1]])
print(u)
u_T = u.transpose()
print(u_T)
mu = np.array([[0.092],[0.075],[0.041],[0.062],[0.043]])
print(mu)
mu_T = mu.transpose()
print(mu_T)
u_T.dot(C_invers).dot(u)
1/u_T.dot(C_invers).dot(u)
a = (1/(u_T.dot(C_invers).dot(u)))*(C_invers.dot(u))
print(a)
C_invers.dot(mu)
u_T.dot(C_invers).dot(mu)
(u_T.dot(C_invers).dot(mu))/(u_T.dot(C_invers).dot(u))
(u_T.dot(C_invers).dot(mu)/u_T.dot(C_invers).dot(u))*(C_invers.dot(u))
b = (C_invers.dot(mu))-(u_T.dot(C_invers).dot(mu)/u_T.dot(C_invers).dot(u))*(C_invers.dot(u))
print(b)
w = a + (0.2*b)
print(w)
t = list()
for i in range(25):
    t.append(i/100)
print(t)
s = list()
for i in range(-24, 25):
    s.append(i/100)
print(s)
W = list()
I = list()
for i in t:
    w = a+i*b
    W.append(w)
    I.append(i)

for j in range(len(W)):
    print(I[j],W[j])
    print('\n')
# ini untuk program sampai t=0.19 bang

W = list()
I = list()
for i in t:
    w = a+i*b
    status = True
    for baris in w:
        if baris < 0:
            status = False
    if status == False:
        break
    W.append(w)
    I.append(i)

for j in range(len(W)):
    print(I[j],W[j])
    print('\n')
W = list()
I = list()

for i in t:
    w = a+i*b
    status = True
    for baris in w:
        if baris < 0:
            status = False
    if status == False:
        continue
    W.append(w)
    I.append(i)
o = t.index(I[-1])
o = o+1
W.append(a+t[o]*b)
I.append(t[o])

for j in range(len(I)):
    print(I[j],W[j])
    print('\n')
W = list()
I = list()
for i in t:
    w = a+i*b
    W.append(w)
    I.append(i)
    
for j in range(len(W)):
    print(I[j],W[j])
    print('\n')
    if sum(sum(W[j]<0)) > 0:
        break
W[20]<0
sum(W[20]<0)
sum(sum(W[20]<0)) > 0