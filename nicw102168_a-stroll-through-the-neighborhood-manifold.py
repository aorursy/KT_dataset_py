import numpy as np

import numpy.linalg

import matplotlib.pyplot as plt

%matplotlib inline   

plt.rcParams['image.cmap'] = 'gray'
def image_generator(*filenames):

    for filename in filenames:

        with open(filename, 'rb') as fp:

            for _ in range(100000):

                yield np.array(np.fromstring(fp.read(401), dtype=np.uint8)[1:], dtype=float)



def img_gen():

    return image_generator(*["../input/snake-eyes/snakeeyes_{:02d}.dat".format(nn) for nn in range(10)])
def plotdice(x):

    plt.imshow(x.reshape(20,20))

    plt.axis('off')                

def plotdicez(x):

    plt.imshow(x.reshape(20,20), cmap=plt.cm.RdBu, vmin=-128, vmax=128)

    plt.axis('off')
refs = np.array([x for _, x in zip(range(6), img_gen())])



for n in range(6):

    plt.subplot(1,6,n+1)

    plotdice(refs[n])
query = [ww for ww in img_gen() if (np.max(np.abs(refs - ww), axis=1) < 128).any()]
qq = query[0]



sim = [ww for ww in query if np.max(np.abs((qq - ww))) < 128][1:]



plt.figure(figsize=(8,6))

for k, ww in enumerate(sim):

    if k >= 20:

        break

    plt.subplot(4, 5, k + 1)

    plotdice(ww)
plt.figure(figsize=(8,6))

for k, ww in enumerate(sim):

    if k >= 20:

        break

    plt.subplot(4, 5, k + 1)

    plotdicez(ww - qq)
u, s, v = np.linalg.svd(np.cov(np.array(sim).T))



plt.figure()

plt.plot(s[:40], '-+')

plt.grid()
plt.figure()

for n in range(3):

    plt.subplot(1, 3, n + 1)

    plotdicez(v[n] * 400)
qq = query[1]



sim = [ww for ww in query if np.max(np.abs((qq - ww))) < 128][1:]



plt.figure(figsize=(20,8))

for k, ww in enumerate(sim):

    if k >= 20:

        break

    plt.subplot(4, 10, k//5*10 + k%5 + 1)

    plotdice(ww)

    

for k, ww in enumerate(sim):

    if k >= 20:

        break

    plt.subplot(4, 10, k//5*10 + k%5 + 6)

    plotdicez(ww - qq)



u, s, v = np.linalg.svd(np.cov(np.array(sim).T))



plt.figure()

plt.plot(s[:40], '-+')

plt.grid()



plt.figure()

for n in range(3):

    plt.subplot(1, 3, n + 1)

    plotdicez(v[n] * 400)
qq = query[2]

sim = [ww for ww in query if np.max(np.abs((qq - ww))) < 128][1:]

len(sim)
qq = query[3]

sim = [ww for ww in query if np.max(np.abs((qq - ww))) < 128][1:]

len(sim)
qq = query[4]



sim = [ww for ww in query if np.max(np.abs((qq - ww))) < 128][1:]



plt.figure(figsize=(20,8))

for k, ww in enumerate(sim):

    if k >= 20:

        break

    plt.subplot(4, 10, k//5*10 + k%5 + 1)

    plotdice(ww)

    

for k, ww in enumerate(sim):

    if k >= 20:

        break

    plt.subplot(4, 10, k//5*10 + k%5 + 6)

    plotdicez(ww - qq)



u, s, v = np.linalg.svd(np.cov(np.array(sim).T))



plt.figure()

plt.plot(s[:40], '-+')

plt.grid()



plt.figure()

for n in range(3):

    plt.subplot(1, 3, n + 1)

    plotdicez(v[n] * 400)
qq = query[5]



sim = [ww for ww in query if np.max(np.abs((qq - ww))) < 128][1:]



plt.figure(figsize=(20,8))

for k, ww in enumerate(sim):

    if k >= 20:

        break

    plt.subplot(4, 10, k//5*10 + k%5 + 1)

    plotdice(ww)

    

for k, ww in enumerate(sim):

    if k >= 20:

        break

    plt.subplot(4, 10, k//5*10 + k%5 + 6)

    plotdicez(ww - qq)



u, s, v = np.linalg.svd(np.cov(np.array(sim).T))



plt.figure()

plt.plot(s[:40], '-+')

plt.grid()



plt.figure()

for n in range(3):

    plt.subplot(1, 3, n + 1)

    plotdicez(v[n] * 400)