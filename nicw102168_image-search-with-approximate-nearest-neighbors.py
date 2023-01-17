import numpy as np

import numpy.linalg

import matplotlib.pyplot as plt

%matplotlib inline   

plt.rcParams['image.cmap'] = 'gray'



import annoy
def image_generator(*filenames):

    for filename in filenames:

        with open(filename, 'rb') as fp:

            for _ in range(100000):

                yield np.array(np.fromstring(fp.read(401), dtype=np.uint8)[1:], dtype=float)



def img_gen(batches=10):

    return image_generator(*["../input/snake-eyes/snakeeyes_{:02d}.dat".format(nn) for nn in range(batches)])



def plotdice(x):

    plt.imshow(x.reshape(20,20))

    plt.axis('off')                

def plotdicetiling(xx):

    rows = len(xx) // 10

    plt.figure(figsize=(12, rows * 6 // 5))

    for n, q in enumerate(xx):

        plt.subplot(rows, 10, n+1)

        plotdice(q)

def plotdicez(x):

    plt.imshow(x.reshape(20,20), cmap=plt.cm.RdBu, vmin=-128, vmax=128)

    plt.axis('off')
vector_length = 400

Ntrees = 1

t = annoy.AnnoyIndex(vector_length)

for i, v in zip(range(100000), img_gen(1)):

    t.add_item(i, v)

t.build(Ntrees)
for ref in range(6):

    neighbors = t.get_nns_by_item(ref, 50)

    plotdicetiling([np.array(t.get_item_vector(i)) for i in neighbors])
Ntrees = 10

tbig = annoy.AnnoyIndex(vector_length)

for i, v in zip(range(1000000), img_gen(10)):

    tbig.add_item(i, v)

tbig.build(Ntrees)
for ref in range(6):

    neighbors = tbig.get_nns_by_item(ref, 50)

    plotdicetiling([np.array(tbig.get_item_vector(i)) for i in neighbors])
Ntrees = 10

tbad = annoy.AnnoyIndex(vector_length)

for i, v in zip(range(1000), img_gen(1)):

    tbad.add_item(i, v)

tbad.build(Ntrees)

for ref in range(6):

    neighbors = tbad.get_nns_by_item(ref, 50)

    plotdicetiling([np.array(tbad.get_item_vector(i)) for i in neighbors])