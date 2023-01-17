# =============================================

# REPRODUCIBLE CODE 

# Note: t-SNE is meant only visualisation 2D/3D

# =============================================

from sklearn.manifold import TSNE

import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

import math





def get_tsne_df(X, labels, to_dims=2, perplexity_iter_combns=[(30,100)]):

    """

    - `perplexity_iter_combns` is not `None` to visualize only. Doesn't return df

        + [(p1, iters1), (p2, iters2), ...]

    """

    # for visualaisation

    legend = [str(i) for i in np.unique(labels)]

    colors = {0 :'red', 1 :'blue', 2 :'green', 3 :'black', 4 :'orange', 5 :'yellow',

              6 :'pink', 7 :'brown', 8 : 'purple', 9 : 'grey' }

    num_combns = None

    rows, cols, = None, None

    fig, axarr, = None, None

    if (len(perplexity_iter_combns) > 1):

        num_combns = len(perplexity_iter_combns)

        cols = 4

        rows = math.ceil(num_combns/cols)

        # create `axarr` w/ len=cols*rows

        fig, axarr = plt.subplots(rows,cols)

        axarr = axarr.flatten()

        fig.set_size_inches(5*cols, 5*rows)

    

    for idx, (perplexity_val, n_iter_val) in enumerate(perplexity_iter_combns):

        model = TSNE(

            n_components  = to_dims, 

            random_state  = 0,

            perplexity    = perplexity_val,

            n_iter        = n_iter_val

        )

        # configuring the parameteres

        # the number of components = to_dims

        # default perplexity = 30

        # default learning rate = 200

        # default n_iter = 1000



        tsne_data = model.fit_transform(X)



        # return a dataframe which help us in ploting the result data

        # (only if single perplexity_iter_combn is given)

        tsne_data = np.vstack((tsne_data.T, labels)).T

        tsne_df = pd.DataFrame(data=tsne_data, columns=("Dim_1", "Dim_2", "label"))

        if (len(perplexity_iter_combns) == 1):

            return tsne_df

        else:

            # plot sublots for all `perplexity_iter_combns`

            axarr[idx].scatter(tsne_df['Dim_1'], tsne_df['Dim_2'], c=tsne_df['label'].apply(lambda x: colors[int(x)]), alpha=0.5)

            axarr[idx].title.set_text(f"perplexity: {perplexity_val}\niterations: {n_iter_val}")

            axarr[idx].set_xlabel("Dim 1")

            axarr[idx].set_ylabel("Dim 2")

            axarr[idx].legend(legend)

            print(f"combination {idx+1}of{num_combns} >> perplexity: {perplexity_val} iterations: {n_iter_val} done ...")

    plt.show()

    

    

_ = """#Plotting 

sns.FacetGrid(df, hue="label", size=6).map(plt.scatter, 'Dim_1', 'Dim_2').add_legend()

plt.show()

"""
from sklearn.datasets import load_digits

digits = load_digits()

X_mnist = digits.data

y_mnist = digits.target
df = get_tsne_df(

    X_mnist, y_mnist, 

    to_dims = 2,

    perplexity_iter_combns = [(30,1000)]

)



df.head()
import seaborn as sns

import matplotlib.pyplot as plt



sns.FacetGrid(df, hue="label", height=6).map(plt.scatter, 'Dim_1', 'Dim_2').add_legend()



plt.title(f"MNIST Dataset\nm={X_mnist.shape[1]} n={X_mnist.shape[0]}")

plt.show()
# plot sublots (simply give needed combinations)

# or use zip()

perplexity_iter_combns = [

    # (perplexity_i, iter_i)

    (1,251),

    (3,251),

    (6,251),

    (10,251),

    (20,300),

    (30,500),

    (100,1000),

    (len(X_mnist)-1, 1000)

]



df = get_tsne_df(

    X_mnist, y_mnist, 

    to_dims = 2,

    perplexity_iter_combns = perplexity_iter_combns

)
from sklearn.datasets import load_breast_cancer

X_bc, y_bc = load_breast_cancer(return_X_y=True)
df_bc = get_tsne_df(

    X_bc, y_bc, 

    to_dims = 2,

    perplexity_iter_combns = [(30,1000)]

)
sns.FacetGrid(df_bc, hue="label", height=6).map(plt.scatter, 'Dim_1', 'Dim_2').add_legend()



plt.title(f"Boston Dataset\nm={X_bc.shape[1]} n={X_bc.shape[0]}")

plt.show()
# plot sublots (simply give needed combinations)

# or use zip()

perplexity_iter_combns = [

    # (perplexity_i, iter_i)

    (1,251),

    (3,251),

    (6,251),

    (10,251),

    (20,300),

    (30,500),

    (100,1000),

    (len(X_bc)-1, 1000)

]



df = get_tsne_df(

    X_bc, y_bc, 

    to_dims = 2,

    perplexity_iter_combns = perplexity_iter_combns

)
PERPLEXITY = 100

iters = np.arange(250, 1500,50)

perplexities = np.array([PERPLEXITY]*len(iters))



generator = zip(perplexities, iters)



perplexity_iter_combns = []

for ppxty,it in generator:

    perplexity_iter_combns.append((ppxty, it))

    

perplexity_iter_combns
df = get_tsne_df(

    X_bc, y_bc, 

    to_dims = 2,

    perplexity_iter_combns = perplexity_iter_combns

)