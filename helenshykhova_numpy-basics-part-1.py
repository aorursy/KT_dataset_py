import numpy as np

df = np.loadtxt('../input/cardio_train.csv', delimiter=';', skiprows=1)

print(df)

print(df.shape)

a=df[:,1]

df[:, 1] = np.floor(df[:,1] / 365.25)

print(df[:6, :])

c=a[a>50]

print(c.size,'- больше 50')

d=df[:,2:4]

#print(d,'3,4')

print(d[d[:,0]==2,:])

b=d[d[:,0]==2,:]

mn=np.mean(b[:,1])

print(np.mean(b[:,1]),'- мужчины')

v=d[d[:,0]==1,:]

print(v)

wm=np.mean(v[:,1])

print(np.mean(v[:,1]),'- женщины')

if mn>wm:

     print('Верно,что средний рост мужчин больше среднего роста женщин')

else:

     print('Неверно,что средний рост мужчин больше среднего роста женщин')

f=df[:,4]

print(f)

print(np.min(f),'- минимум,',np.max(f),'- максимум')

print(f[f==200])

t=f[f==200]

print(t.size,'- кол-во людей с макс весом')

g=df[:,-4]

p=df[:,-1]

gp=g-p

print(g-p)

gpp=gp[gp==-1]

print(gpp.size,'- некурящие люди с сердечно-сосудистыми заболеваниями')








