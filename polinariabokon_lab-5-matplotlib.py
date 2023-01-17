import matplotlib as mpl

import matplotlib.pyplot as plt

import numpy as np
fig, ax = plt.subplots()

x = np.linspace(-6,6,30);

plt.plot(x, 1/(1+np.exp(-x) ) );

plt.xlabel('x') ;

plt.ylabel('y');

plt.title(r'График $y=\dfrac{1}{1+e^{-x}}$ ');

plt.plot(x,1/(1+np.exp(-x) ), color='red') ;

plt.plot(x, 1/(1+np.exp(-x) ), linestyle='dashdot');

plt.plot(x, 1/(1+np.exp(-x) ) , linewidth=4);

plt.plot(x, 1/(1+np.exp(-x) ), marker='o');
fig, ax = plt.subplots()

x = np.linspace(-2.5,2.5,1000)

plt.plot(x, 1/(1+np.exp(-x)), label='$y=\dfrac{1}{1+e^{-x}}$;')

plt.plot(x, x/(np.sqrt(1+x**2)),label='$y=\dfrac{x}{\sqrt{1+x^2}}$')

plt.axis('equal')

plt.legend(loc=4);
fig, ax = plt.subplots(1,2,sharey=True)

x = np.linspace(-2.5,2.5,1000)

a = 1/(1+np.exp(-x))

b = x/(np.sqrt(1+x**2))

ax[0].plot(x, a, color = 'red')

ax[1].plot(x, b);
fig, ax = plt.subplots()

x1 = np.linspace(-2*np.pi,0,1000)

x2 = np.linspace(0,5,1000)

a = np.sin(-2*x1)

b = (x2**2) - x2

plt.plot(x1, a)

plt.plot(x2, b);
fig, ax = plt.subplots()

x = [0]

for i in range(1,101):

    x.append(a*x[-1]+ np.random.normal(0,5))

ax.plot(x);

    
fig, ax = plt.subplots()

x = [0]

for i in range(1,101):

    x.append(x[-1]+ np.random.normal(0,5))

ax.plot(x);
fig, ax = plt.subplots(figsize=(20, 10))

for a in np.arange(1,0,-0.2): 

    x = [0]

    for i in range(1,101):

        x.append(a*x[-1]+ np.random.normal(0,10))

    ax.plot(x, label = str(round(a,1)));

    plt.legend();
import os

os.listdir('../input')
df = np.loadtxt('../input/cardiovascular-disease-dataset/cardio_train.csv', delimiter=';', skiprows=1)

print(df.shape)

df.dtype
#id =df[:, 0]

age =df[:,1]

plt.hist(age)

fig, ax = plt.subplots(figsize=(20, 7))

#plt.bar(id , height = age);
weight =df[:,4]

plt.hist(weight);
fig, ax = plt.subplots(figsize=(20, 7))

weight =df[:,4]

plt.boxplot(weight);
height = df[:100, 3]

weight = df[:100, 4]

fig,ax = plt.subplots(figsize=(7,7))

plt.scatter(weight, height)

plt.xlabel('weight, kg' )

plt.ylabel('height, kg' )

plt.title('Weight vs. Height')
fig,ax = plt.subplots()

hweight = df[df[:,-1]==0, 4]

nweight = df[df[:,-1]==1, 4]

plt.bar([1,2], height = [hweight.mean(), nweight.mean()],color = ['red','blue'])

ax.set_xticks([1,2])

ax.set_xticklabels(['health', 'not health']);
fig,ax = plt.subplots()

h1 = df[(df[:, -1]== 0) & (df[:, 7]==1)]

n1 = df[(df[:, -1]== 1) & (df[:, 7]==1)]

h2 = df[(df[:, -1]== 0) & (df[:, 7]==2)] 

n2 = df[(df[:, -1]== 1) & (df[:, 7]==2)]

h3 = df[(df[:, -1]== 0) & (df[:, 7]==3)] 

n3 = df[(df[:, -1]== 1) & (df[:, 7]==3)]

plt.bar([1,2,3,4,5,6],height = [len(h3), len(n3), len(h2), len(n2), len(h1), len(n1)],

        color = ['red','blue','red', 'blue', 'red', 'blue']);

ax.set_xticklabels(['','health', 'not health','health', 'not health','health', 'not health']);