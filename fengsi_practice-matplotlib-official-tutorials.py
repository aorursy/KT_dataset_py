import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
fig = plt.figure()  # an empty figure with no axes
fig.suptitle('No axes on this figure')  # Add a title

fig, ax_lst = plt.subplots(2, 2)  # a figure with a 2x2 grid of Axes
a = pd.DataFrame(np.random.rand(4, 5), columns=list('abcde'))
a_asndarray = a.values
print(type(a), 'to', type(a_asndarray))

b = np.matrix([[1, 2], [3, 4]])
b_asarray = np.asarray(b)
print(type(b), 'to', type(b_asarray))
x = np.linspace(0, 2, 100)

plt.plot(x, x, label='linear')
plt.plot(x, x**2, label='quadratic')
plt.plot(x, x**3, label='cubic')

plt.xlabel('x label')
plt.ylabel('y label')
plt.title('Simple Plot')

plt.legend()

plt.show()
def my_plotter(ax, data1, data2, param_dict):
    """
    A helper function to make a graph
    
    Parameters
    ----------
    ax: Axes
        The axes to draw to
    
    data1: array
        The x data
        
    data2: array
        The y data
        
    param_dict: dict
        Dictionary of kwargs to pass to ax.plot
        
    Retures
    -------
    out: list
        list of artists added
    """
    out = ax.plot(data1, data2, **param_dict)
    return out

data1, data2, data3, data4 = np.random.randn(4, 100)
fig, (ax1, ax2) = plt.subplots(1, 2)
my_plotter(ax1, data1, data2, {'marker': 'x'})
my_plotter(ax2, data3, data4, {'marker': 'o'})
y = np.random.rand(100000)
y[50000:] *= 2
y[np.logspace(1, np.log10(50000), 400).astype(int)] = -1

fig, (ax1, ax2) = plt.subplots(2, 1)

mpl.rcParams['path.simplify'] = True

mpl.rcParams['path.simplify_threshold'] = 0.0
ax1.plot(y)

mpl.rcParams['path.simplify_threshold'] = 1.0
ax2.plot(y)
delta = 0.11
x = np.linspace(0, 10 - 2*delta, 200) + delta
y = np.sin(x) + 1.0 + delta

fig, (ax1, ax2) = plt.subplots(2, 1)

ax1.plot(x, y, 'o', ls='-', markevery=None)
ax2.plot(x, y, 'o', ls='-', markevery=8)
plt.plot([1, 2, 3, 4])
plt.ylabel('some numbers')
plt.plot([1, 2, 3, 4], [1, 4, 9, 16])
plt.plot([1, 2, 3, 4], [1, 4, 9, 16], 'ro')
plt.axis([0, 6, 0, 20])
t = np.arange(0.0, 5.0, 0.2)

plt.plot(t, t, 'r--', t, t**2, 'bs', t, t**3, 'g^')
data = {
    'a': np.arange(50),
    'c': np.random.randint(0, 50, 50),  # colors
    'd': np.random.randn(50)
}
data['b'] = data['a'] + 10 * np.random.randn(50)
data['d'] = np.abs(data['d']) * 100  # sizes

plt.scatter('a', 'b', c='c', s='d', data=data)
plt.xlabel('entry a')
plt.ylabel('entry b')
names = ['group_a', 'group_b', 'group_c']
values = [1, 10, 100]

plt.figure(1, figsize=(9, 3))

plt.subplot(131)
plt.bar(names, values)
plt.subplot(132)
plt.scatter(names, values)
plt.subplot(133)
plt.plot(names, values)
plt.suptitle('Categorical Plotting')
lines = plt.plot([1, 2, 3])
plt.setp(lines)
def f(t):
    return np.exp(-t) * np.cos(2*np.pi*t)

t1 = np.arange(0.0, 5.0, 0.1)
t2 = np.arange(0.0, 5.0, 0.02)

plt.figure(1)
plt.subplot(211)
print(plt.gca())
plt.plot(t1, f(t1), 'bo', t2, f(t2), 'k')

plt.subplot(212)
print(plt.gca())
plt.plot(t2, np.cos(2*np.pi*t2), 'r--')
mu, sigma = 100, 15
x = mu + sigma * np.random.randn(10000)

n, bins, patches = plt.hist(x, 50, density=1, facecolor='g', alpha=0.75)

plt.xlabel('Smarts')
plt.ylabel('Probability')
plt.title('Histogram of IQ')
param = plt.text(45, 0.025, r'$\mu=100,\ \sigma=15$')
plt.axis([40, 160, 0, 0.03])
plt.grid(True)

plt.setp(param, fontsize=16, color='g')
ax = plt.subplot(111)

t = np.arange(0.0, 5.0, 0.01)
s = np.cos(2*np.pi*t)
line, = plt.plot(t, s, lw=2)

plt.annotate('local max', 
             xy=(2, 1), 
             xytext=(3, 1.5), 
             arrowprops=dict(facecolor='black', shrink=0.05))
plt.ylim(-2, 2)
y = np.random.normal(loc=0.5, scale=0.4, size=1000)
y = y[(y > 0) & (y < 1)]
y.sort()
x = np.arange(len(y))

plt.figure(1)

# Linear
plt.subplot(221)
plt.plot(x, y)
plt.yscale('linear')
plt.title('linear')
plt.grid('True')

# Log
plt.subplot(222)
plt.plot(x, y)
plt.yscale('log')
plt.title('log')
plt.grid('True')

# Symmetric log
plt.subplot(223)
plt.plot(x, y - y.mean())
plt.yscale('symlog', linthreshy=0.01)
plt.title('symlog')
plt.grid('True')

# Logit
plt.subplot(224)
plt.plot(x, y)
plt.yscale('logit')
plt.title('logit')
plt.grid('True')
plt.gca().yaxis.set_minor_formatter(mpl.ticker.NullFormatter())

plt.subplots_adjust(top=1.2, bottom=0.08, left=0.10, right=0.95, hspace=0.35, wspace=0.35)
