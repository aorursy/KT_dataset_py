import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline
x = np.linspace(0, 5, 11)

y = x ** 2
x
y
# Functional

plt.plot(x, y)

plt.xlabel('X Label')

plt.ylabel('X Label')

plt.title('Title')
plt.subplot(1, 2, 1)

plt.plot(x, y, 'r')



plt.subplot(1, 2, 2)

plt.plot(y, x, 'b')
# Object Oriented Method

fig = plt.figure()



#fig.add_axes([left, bottom, width, height])

axes = fig.add_axes([0.1, 0.1, 0.8, 0.8])

axes.plot(x, y)

axes.set_xlabel('X Label')

axes.set_ylabel('Y Label')

axes.set_title('Set Title')
fig = plt.figure()

axes1 = fig.add_axes([0.1, 0.1, 0.8, 0.8])

axes2 = fig.add_axes([0.2, 0.5, 0.4, 0.3])



axes1.plot(x, y)

axes1.set_title('Larger Plot')



axes2.plot(y, x)

axes2.set_title('Smaller Plot')
fig, axes = plt.subplots(nrows = 1, ncols = 2)

# for current_ax in axes:

    # current_ax.plot(x, y)

    

axes[0].plot(x, y)

axes[0].set_title('First Plot')

axes[1].plot(y, x)

axes[1].set_title('Second Plot')

# axes.plot(x, y)

plt.tight_layout()
# Figure Size and DPI

fig = plt.figure(figsize = (8, 2))

ax = fig.add_axes([0, 0, 1, 1])

ax.plot(x, y)
fig, axes = plt.subplots(nrows = 2, ncols = 1, figsize = (8, 2))

axes[0].plot(x, y)

axes[1].plot(y, x)

plt.tight_layout()
fig.savefig('my_picture.png', dpi = 200)
fig = plt.figure()

ax = fig.add_axes([0, 0, 1, 1])

ax.plot(x, x**2, label = 'X Squared')

ax.plot(x, x**3, label = 'X Cubed')

ax.legend(loc=0)
fig = plt.figure()



ax = fig.add_axes([0, 0, 1, 1])

ax.plot(x, y, color = 'purple', lw=2, ls='--')



ax.set_xlim([0,1])

ax.set_ylim([0,2])