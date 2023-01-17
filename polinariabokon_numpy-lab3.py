import numpy as np

np.__version__



A = np.zeros(10)

A
A[4]=1

A
A[-3:]=2

A
x = np.random.uniform(150,200,100)

x
x.mean()
x.std()

s = x[(x.mean()-x.std()<x)&(x.mean()+x.std()>x)]

s1=s.size/x.size

s1
d = np.linspace(0,np.pi/2,20)

d

w =np.cos(d)

w
y=np.sin(np.pi/2 - d)

y
w-y
def close(x,y,eps):

    if np.abs(x-y)<eps:

        return True

    else:

        return False

    
close (w,y,1e-6)
np.abs(w - y)<0.1
close_v = np.vectorize(close)

close_v(w,y,1e-6)
x = np.random.normal(5, 2.5, (10, 5))

x
a = x[4,:].mean()

a
s = np.median(x[:,3])

s
print(np.sum(x, axis=1)) 

print(np.mean(x, axis=0)) 

print(np.sum(x)) 
print(x[:3, :3])
y = np.sum(x, axis = 1)#cyмма эл

print(y)

a = np.mean(y) #средняя сумм

print(a)

print(y[y < a])

s = np.array(y[y < a]) 

print(s.size)

w = (x[x[:, 0]>x[: ,-1 ]])

print(w)

print(w.ndim, w.shape)

    
y = np.sum(x, axis = 1) #сумма эл строки

a = np.mean(x , axis = 1)  #средняя сумма

s  = x[y > a]

print(s)

print(np.linalg.matrix_rank(s))
df = np.loadtxt('../input/cardio_train.csv', delimiter=';', skiprows=1)

print(df.shape)
x = df[:,1]

y = x / 365.25

s = np.floor(y)

print(s)

df[:,1] = s
k = (s[s>50])

print(k.size)
women_height =  df[df[:, 2]==1, 3]

men_height = df[df[:, 2]==2, 3]

w = np.mean(women_height)

m =  np.mean(men_height)

print(w<m, w , m)
f = df[:,4]

mmax = np.max(f)

mmin = np.min(f)

g = (f[f == mmin])

print(mmax, mmin,g.size)
k =  df[df[:, -4]==0 & (df[:, -1]==1)].shape[0]

print(k)
