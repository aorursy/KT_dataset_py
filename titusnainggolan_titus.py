import numpy as np
import matplotlib.pyplot as plt
C = np.array([[0.022,0.01,-0.001,0.011,0.005],[0.01,0.033,0,0.014,0.01],[-0.001,0,0.019,-0.001,-0.001],[0.011,0.014,-0.001,0.064,0.011],[0.005,0.01,-0.001,0.011,0.045]])
print(C)
Cinv = np.linalg.inv(C)
print(Cinv)
u = np.array([[1],[1],[1],[1],[1]])
print(u)
uT = u.transpose()
print(uT)
mu = np.array([[0.092],[0.075],[0.041],[0.062],[0.043]])
print(mu)
muT = mu.transpose()
print(muT)
a = Cinv.dot(mu)
print(a)
b = uT.dot(Cinv).dot(mu)
print(b)
d = uT.dot(Cinv).dot(u)
print(d)
e = Cinv.dot(u)
print(e)
t = list()
for i in range(25):
    t.append(i/100)
print(t)
W = list()
I = list()
W2 = list()
I2 = list()

for j in t:
    w = (a*j) + (((0.9 - b*j)/d)*e)
    W.append(w)
    I.append(j)
for k in range(len(W)):
    I2.append(I[k])
    W2.append(W[k])
    if sum(sum(W2[k]<0))>0:
        break
for l in range(len(W2)):
    print(I2[l],W2[l])
    print('\n')
# hitung expected return atau buat sumbu vertikal
wf = 0.1
muf = 1.9178
ERP = list()
for w in W2:
    ERP.append((wf*muf) + (muT.dot(w)))
print(ERP)
# hitung varian atau buat sumbu horizontal
VP = list()
for w in W2:
    VP.append(w.transpose().dot(C).dot(w))
print(VP)
# cari portofolio optimal
for m in range(len(ERP)):
    print(ERP[m]/VP[m])
plt.scatter(VP, ERP)
plt.title('Judul')
plt.xlabel('Varian Portofolio')
plt.ylabel('Expected Return Portofolio')
plt.show()