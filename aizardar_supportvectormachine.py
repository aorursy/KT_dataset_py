# Let's load essential modules



import numpy as np

from sklearn.datasets import make_moons

from sklearn.pipeline import Pipeline

from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt

import pandas as pd

from sklearn.svm import SVC







# IPython libraries



from ipywidgets import interactive

from IPython.display import display

import ipywidgets as widgets
# Let's load moons dataset into objects and visualize it 



X, y = make_moons(n_samples=100, noise=0.15, random_state=42)



def plot_dataset(X, y):

    plt.plot(X[:, 0][y==0], X[:, 1][y==0], "bs")

    plt.plot(X[:, 0][y==1], X[:, 1][y==1], "g^")

    plt.grid(True, which='both')

    plt.xlabel(r"$x_1$", fontsize=20)

    plt.ylabel(r"$x_2$", fontsize=20, rotation=0)



plot_dataset(X, y)

plt.show()
def svm_poly_kernel(degree, coef0):

    

    poly_kernel_svm = Pipeline([

            ("scaler", StandardScaler()),  # SVMs are sensitive to the feature scales, it is important to scale features !

            ("svm_clf", SVC(kernel="poly", degree=degree, coef0=coef0, C=5))

        ])

    poly_kernel_svm.fit(X, y)

    # Plotting the decision boundary



    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1

    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),

                         np.arange(y_min, y_max, 0.1))

    Z = poly_kernel_svm.predict(np.c_[xx.ravel(),yy.ravel()])

    Z = Z.reshape(xx.shape)

    plt.contourf(xx,yy,Z,alpha=0.4)

    plt.scatter(X[:,0], X[:,1], c = y, s = 20, edgecolor = 'k')

    plt.xlabel(r"$x_1$", fontsize=20)

    plt.ylabel(r"$x_2$", fontsize=20, rotation=0)

    plt.show()
style = {'description_width': 'initial'}

m = interactive(svm_poly_kernel,degree=widgets.IntSlider(min=3,max=10,step=1,description= 'Degree',

                                       stye=style,continuous_update=False),coef0=widgets.IntSlider(min=1,max=100,step=99,description= 'Coef0',

                                       stye=style,continuous_update=False))



# Set the height of the control.children[-1] so that the output does not jump and flicker

output = m.children[-1]

output.layout.height = '350px'



# Display the control

display(m)