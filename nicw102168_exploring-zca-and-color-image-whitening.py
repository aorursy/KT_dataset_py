import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline   

plt.rcParams['image.cmap'] = 'gray'



def unpickle(file):

    import pickle

    with open(file, 'rb') as fo:

        dict = pickle.load(fo, encoding='bytes')

    return dict
batch1 = unpickle("../input/data_batch_1")

batch2 = unpickle("../input/data_batch_2")

# dd = batch[b'data']

dd = np.vstack(tuple(unpickle("../input/data_batch_{}".format(n))[b'data'] for n in range(1,6)))



red_cat_bag = dd[3000].reshape(3,32,32).transpose([1,2,0])

plt.imshow(red_cat_bag)
from sklearn.decomposition import PCA

pca = PCA(n_components=400, random_state=0, svd_solver='randomized')

pca.fit(dd)
plt.plot(np.cumsum(pca.explained_variance_ratio_))

plt.ylim(0.8, 1.0)

plt.grid()
DD = np.dot(dd - pca.mean_, pca.components_.T)
def reconstruct(pca, vec):

    return pca.mean_ + np.dot(vec, pca.components_)



def vtoimg(v):

     return np.array(np.clip(v, 0, 255), dtype=np.uint8).reshape(3,32,32).transpose([1,2,0])

    

reconstructed_cat = vtoimg(reconstruct(pca, DD[3000]))

plt.subplot(1,2,1)

plt.imshow(red_cat_bag)

plt.subplot(1,2,2)

plt.imshow(reconstructed_cat)
def whiten(pca, vec):

    QQ = np.dot(vec - pca.mean_, pca.components_.T)

    return np.dot(QQ / pca.singular_values_, pca.components_) * np.sqrt(60000) * 64 + 128



whitened_cat = vtoimg(whiten(pca, dd[3000]))



plt.subplot(1,2,1)

plt.imshow(red_cat_bag)

plt.subplot(1,2,2)

plt.imshow(whitened_cat)
plt.figure(figsize=(12,10))

for xx in range(25):

    plt.subplot(5,10,xx*2+1)

    plt.imshow(vtoimg(dd[xx]))

    plt.axis('off')

    plt.subplot(5,10,xx*2+2)

    plt.imshow(vtoimg(whiten(pca, dd[xx])))

    plt.axis('off')
dirac_r = pca.mean_.copy()

dirac_g = pca.mean_.copy()

dirac_b = pca.mean_.copy()



dirac_r[32*16+16] += 1

dirac_g[32*32+32*16+16] += 1

dirac_b[2*32*32+32*16+16] += 1



plt.plot(whiten(pca, dirac_r))

plt.plot(whiten(pca, dirac_g))

plt.plot(whiten(pca, dirac_b))
def vftoimg(v):

     return v.reshape(3,32,32).transpose([1,2,0])



kk = [dirac_r,dirac_g,dirac_b]

for k in range(3):

    for c in range(3):

        plt.subplot(3,3,c+1+k*3)

        plt.imshow(vftoimg(whiten(pca, kk[k])-128)[:,:,c], cmap=plt.cm.coolwarm, vmin=-0.1, vmax=0.1)
kk = [dirac_r,dirac_g,dirac_b]

for k in range(3):

    for c in range(3):

        plt.subplot(3,3,c+1+k*3)

        plt.imshow(vftoimg(whiten(pca, kk[k])-128)[:,:,c], cmap=plt.cm.coolwarm, vmin=-0.2, vmax=0.2)

        plt.xlim(13,19)

        plt.ylim(13,19)
#kk = [dirac_r,dirac_g,dirac_b]

#for k in range(3):

#    for c in range(3):

#        plt.subplot(3,3,c+1+k*3)

#        print(np.array(100*vftoimg(whiten(pca, kk[k])-128)[13:20,13:20,c], dtype=np.int))