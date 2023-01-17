import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

from subprocess import check_output

from sklearn import decomposition

print(check_output(["ls", "../input"]).decode("utf8"))

%matplotlib inline

# Any results you write to the current directory are saved as output.
orders_train = pd.read_csv('../input/order_products__train.csv')

pvt = orders_train.pivot_table(index = 'order_id', columns = 'add_to_cart_order', values = 'product_id',fill_value = 0)



pvt.shape
matr = pvt.as_matrix()

colors = np.arange(0,0.9,0.01)

print(colors)

cmap = []

for i in range(len(matr)):

    cmap.append(colors[np.count_nonzero(matr[i])])
x_tsne = decomposition.PCA().fit_transform(matr)

x_tsne_x = x_tsne[:,0]

x_tsne_y = x_tsne[:,1]

plt.scatter(x_tsne_x,x_tsne_y, c = cmap, alpha = 0.5)

plt.show()                                            