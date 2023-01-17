import numpy as np
np.__version__
x=np.zeros(10)

x
x[5]=1

x
x[-3:]=2

print(x)
y=np.random.uniform(150,200,100)

y
print(np.mean(y))
print(np.std(y))
z=y[(y>(np.mean(y)-np.std(y)))&(y<(np.mean(y)+np.std(y)))].size

z
z=y[(y>(np.mean(y)-np.std(y)))&(y<(np.mean(y)+np.std(y)))].size

print((z/y.size*100),'%')
x=y[(y>(np.mean(y)-np.std(y)))&(y<(np.mean(y)+np.std(y)))]

x
m20=np.linspace(0,np.pi,20)

m20

cos_m20=np.cos(m20)

cos_m20
sin_m20=np.sin(np.pi/2-m20)

sin_m20
cos_m20-sin_m20
np.abs(cos_m20-sin_m20)<1e-6
def close (x,y,eps):

    if np.abs(x-y)<eps:

        return True

    else:

        return False
close_v=np.vectorize(close)

close_v(cos_m20,sin_m20,1e-6)

x = np.random.normal(5, 2.5, (10, 5))

x
print(np.mean(x[5, :]))
print(np.median(x[:, 3]))
print(np.sum(x, axis=1))

print(np.sum(x, axis=0))

print(np.sum(x))
print(np.linalg.det(x[:3, :3]))
z=np.sum(x, axis=1)[np.sum(x, axis=1)<np.mean(np.sum(x, axis=1))].size

z
z = x[(x[:, 0]>x[:, -1]), :]

print(z)

print(z.shape)
z = x[np.sum(x, axis=1)>np.mean(np.sum(x, axis=1)), :]

      

print(z)

print(np.linalg.matrix_rank(z))
df = np.loadtxt('../input/cardio_train.csv', delimiter=';', skiprows=1)

print(df[:5, :])
print(df.shape)
df[:, 1] = df[:, 1]//365.25

print(df[:5, :])
print(round(df[df[:, 1] > 50,:].shape[0]/df.shape[0]*100, 1),'%')
f = np.mean(df[df[:, 2] == 1, 3])

print(f)

m = np.mean(df[df[:, 2] == 2, 3])

print(m)

if m > f:

    print('Верно, что средний рост мужчин больше среднего роста женщин.')

else:

    print('Не верно, что средний рост мужчин больше среднего роста женщин')
print(np.min(df[:, 4]))

print(np.max(df[:, 4])) 

print(df[df[:, 4] == np.max(df[:, 4]), :].shape[0])
print(df[(df[:, -4] == 0) & (df[:, -1] == 1),:].shape[0])

print(df[(df[:, -4] == 0) & (df[:, -1] == 1),:].shape[0] + df[(df[:, -4] == 0) & (df[:, -1] == 0),:].shape[0] + 

      df[(df[:, -4] == 1) & (df[:, -1] == 1),:].shape[0] + df[(df[:, -4] == 1) & (df[:, -1] == 0),:].shape[0])