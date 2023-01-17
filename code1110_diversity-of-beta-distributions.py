# Libraries

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

sns.set_context("talk")



from scipy.stats import beta
# figure and axes

fig, ax = plt.subplots(5, 5, figsize=(20, 20))



# parameters

params = np.array([0.5, 1, 2, 3, 5])

colors = sns.cubehelix_palette(25)



# range

counts = 0

for r, a in enumerate(params):

    for c, b in enumerate(params):

        # probability densitiy function

        x = np.linspace(beta.ppf(0.01, a, b), beta.ppf(0.99, a, b), 100)

        ax[r, c].plot(x, beta.pdf(x, a, b), 'r-', color=colors[counts], lw=10, alpha=0.9)

        ax[r, c].set_title("a = " + str(a) + ", b = " + str(b))

        counts += 1

        if c == 0:

            ax[r, c].set_ylabel("PDF")

        if r == 0:

            ax[r, c].set_xlabel("x")

plt.tight_layout()
# figure and axes

fig, ax = plt.subplots(5, 5, figsize=(20, 20))



# parameters

params = np.array([0.5, 1, 2, 3, 5])



# range

counts = 0

for r, a in enumerate(params):

    for c, b in enumerate(params):

        # probability densitiy function

        x = np.linspace(beta.ppf(0.01, a, b), beta.ppf(0.99, a, b), 100)

        ax[r, c].plot(x, beta.cdf(x, a, b), 'r-', color=colors[counts], lw=10, alpha=0.9)

        ax[r, c].set_title("a = " + str(a) + ", b = " + str(b))

        counts += 1

        if c == 0:

            ax[r, c].set_ylabel("CDF")

        if r == 0:

            ax[r, c].set_xlabel("x")

plt.tight_layout()