import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline
#Q.1 (1)

import matplotlib.pyplot as plt

def logistic(r, x):

    return r * x * (1 - x)



x = np.linspace(0, 1)

fig, ax = plt.subplots(1, 1)

a=[0.5,1.0,1.5,2.0,2.5,3.0,3.5,3.8,4.0]



for i in range(len(a)):

    for j in a:

        ax.plot(x, logistic(j, x), 'k')
#Q1. (2)

def logistic_equation_orbit(seed, r, n_iter, n_skip=0):

    print('Orbit for seed {0}, growth rate of {1}' .format(seed, r))

    X_t=[]

    T=[]

    t=0

    x = seed;

    # Iterate the logistic equation, printing only if n_skip steps have been skipped

    for i in range(n_iter + n_skip):

        if i >= n_skip:

            X_t.append(x)

            T.append(t)

            t+=1

        x = logistic(r,x);

    # Configure and decorate the plot

    plt.plot(T, X_t)

    plt.ylim(0, 1)

    plt.xlim(0, T[-1])

    plt.xlabel('Time t')

    plt.ylabel('X_t')

    plt.show()
logistic_equation_orbit(0.1, 2.5, 100)
logistic_equation_orbit(0.2, 2.5, 100)
logistic_equation_orbit(0.1, 3.05, 100)
logistic_equation_orbit(0.2, 3.5, 100)
logistic_equation_orbit(0.2, 3.53, 100)
logistic_equation_orbit(0.2, 3.54, 100)
logistic_equation_orbit(0.1, 3.7, 100)
logistic_equation_orbit(0.2, 3.5,100) #x0=0.2 and a=3.5
logistic_equation_orbit(0.1, 3.9, 100)

logistic_equation_orbit(0.1, 3.9, 100, 1000) #iterating over 100 and plotting 1000 
#Q1 (3)

# Create the bifurcation diagram

def bifurcation_diagram(seed, n_skip, n_iter, step=0.0001, r_min=0):

    print("Starting with x0 seed {0}, skip plotting first {1} iterations, then plot next {2} iterations.".format(seed, n_skip, n_iter));

    # Array of r values, the x axis of the bifurcation plot

    R = []

    # Array of x_t values, the y axis of the bifurcation plot

    X = []

    

    # Create the r values to loop. For each r value we will plot n_iter points

    r_range = np.linspace(r_min, 4, int(1/step))



    for r in r_range:

        x = seed;

        # For each r, iterate the logistic function and collect datapoint if n_skip iterations have occurred

        for i in range(n_iter+n_skip+1):

            if i >= n_skip:

                R.append(r)

                X.append(x)

                

            x = logistic(r,x);

    # Plot the data    

    plt.plot(R, X, ls='', marker=',')

    plt.ylim(0, 1)

    plt.xlim(r_min, 4)

    plt.xlabel('r')

    plt.ylabel('X')

    plt.show()
bifurcation_diagram(0.2, 100, 5)
bifurcation_diagram(0.2, 100, 10)
bifurcation_diagram(0.2, 100, 10, r_min=2.8)
bifurcation_diagram(0.2,19500,500)
#Q1.(4)

n = 10000

r = np.linspace(2.5, 4.0, n)

iterations = 1000

last = 100

x=0.00001

lyapunov = np.zeros(n)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 9),

                               sharex=True)



for i in range(iterations):

    x = logistic(r, x)

    # We compute the partial sum of the

    # Lyapunov exponent.

    lyapunov += np.log(abs(r - 2 * r * x))

# We display the Lyapunov exponent.

# Horizontal line.

ax2.axhline(0, color='k', lw=.5, alpha=.5)

# Negative Lyapunov exponent.

ax2.plot(r[lyapunov < 0],

         lyapunov[lyapunov < 0] / iterations,

         '.k', alpha=.5, ms=.5)

# Positive Lyapunov exponent.

ax2.plot(r[lyapunov >= 0],

         lyapunov[lyapunov >= 0] / iterations,

         '.r', alpha=.5, ms=.5)

ax2.set_xlim(2.5, 4)

ax2.set_ylim(-2, 1)

ax2.set_title("Lyapunov exponent")

plt.tight_layout()
def plot_system(r, x0, n, ax=None):

    # Plot the function and the

    # y=x diagonal line.

    t = np.linspace(0, 1)

    ax.plot(t, logistic(r, t), 'k', lw=2)

    ax.plot([0, 1], [0, 1], 'k', lw=2)



    # Recursively apply y=f(x) and plot two lines:

    # (x, x) -> (x, y)

    # (x, y) -> (y, y)

    x = x0

    for i in range(n):

        y = logistic(r, x)

        # Plot the two lines.

        ax.plot([x, x], [x, y], 'k', lw=1)

        ax.plot([x, y], [y, y], 'k', lw=1)

        # Plot the positions with increasing

        # opacity.

        ax.plot([x], [y], 'ok', ms=10,

                alpha=(i + 1) / n)

        x = y



    ax.set_xlim(0, 1)

    ax.set_ylim(0, 1)

    ax.set_title(f"$r={r:.1f}, \, x_0={x0:.1f}$")







fig, (ax1, ax2,) = plt.subplots(1, 2, figsize=(12, 6),

                               sharey=True)

plot_system(2.5, .1, 10, ax=ax1)

plot_system(3.5, .1, 10, ax=ax2)