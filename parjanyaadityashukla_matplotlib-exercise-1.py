import numpy as np

x = np.arange(0,100)

y = x*2

z = x**2
import matplotlib.pyplot as plt

%matplotlib inline
fig = plt.figure()

ax = fig.add_axes([0,0,1,1])

ax.plot(x,y)

ax.set_xlabel('x')

ax.set_ylabel('y')

ax.set_title('title')
fig = plt.figure()

ax1 = fig.add_axes([0,0,1,1])

ax2 = fig.add_axes([0.2,0.5,.2,.2])
ax1.plot(x,y)

ax2.plot(x,y)

fig
fig1 = plt.figure()

ax1 = fig1.add_axes([0,0,1,1])

ax2 = fig1.add_axes([0.2,0.5,.4,.4])
ax1.plot(x,z)

ax1.set_xlabel('x')

ax1.set_ylabel('z')

ax2.plot(x,y)

ax2.set_xlabel('x')

ax2.set_ylabel('y')

ax2.set_xlim([20,22])

ax2.set_ylim([30,50])

fig1
fig, axes = plt.subplots(nrows=1, ncols=2)
axes[0].plot(x,y,linestyle='--',color='blue')

axes[1].plot(x,z,linestyle='-',color='red')

fig
fig, axes = plt.subplots(nrows=1, ncols=2,figsize=(12,2))

axes[0].plot(x,y,linestyle='--',color='blue')

axes[1].plot(x,z,linestyle='-',color='red')

fig
                                             #Thank You 