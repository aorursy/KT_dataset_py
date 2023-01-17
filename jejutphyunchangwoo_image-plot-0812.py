from sklearn.datasets import load_digits



digits = load_digits()

digits.keys()
digits
digits.images[0]
import matplotlib as mpl

import matplotlib.pyplot as plt



%matplotlib inline
fig, ax= plt.subplots(10, 10, figsize=(5,5))

for i, ax_i in enumerate(ax.flat):

    ax_i.imshow(digits.images[i], cmap="binary")

    ax_i.set(xticks=[], yticks=[])