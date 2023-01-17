# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np

import cv2

import pandas as pd

import numpy as np

from sklearn.cluster import KMeans

from sklearn.metrics import silhouette_score

from sklearn.cluster import AgglomerativeClustering

from sklearn.preprocessing import StandardScaler

from scipy import misc

import sklearn as sk

from sklearn import metrics

from PIL import Image 

from sklearn.mixture import GaussianMixture as GMM

#from PIL import Convert 

import matplotlib.pyplot as plt 

from sklearn.decomposition import PCA

from matplotlib.pyplot import imread

from sklearn.decomposition import PCA

from sklearn.manifold import TSNE

import matplotlib.image as mpimg

from skimage import io, color

import seaborn as sns



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
plane = imread("../input/images/3096_colorPlane.jpg")

birdie = imread("../input/images/42049_colorBird.jpg")



gray = birdie[:,:,0]

grayP = plane[:,:,0]
imgbird = gray.astype(np.uint8)

#imgbird = cv2.cvtColor(birdie, cv2.COLOR_BGR2RGB)

imgbird = imgbird / 255.0

#pixel_values = imgbird.reshape((-1, 3))

# convert to float

#pixel_values = np.float32(pixel_values)

df = pd.DataFrame(imgbird)

print('Size of the dataframe: {}'.format(df.shape))

plt.imshow(imgbird)



imgplane = grayP.astype(np.uint8)

imgplane = imgplane / 255.0

#pixel_values = imgbird.reshape((-1, 3))

# convert to float

#pixel_values = np.float32(pixel_values)

dfP = pd.DataFrame(imgplane)

print('Size of the dataframe: {}'.format(dfP.shape))

plt.imshow(imgplane)

np.random.seed(4)

rndperm = np.random.permutation(df.shape[0])

rndpermP = np.random.permutation(dfP.shape[0])
pca = PCA(n_components=5)

pca_result = pca.fit_transform(df)

df['pca-one'] = pca_result[:,0]

df['pca-two'] = pca_result[:,1] 

df['pca-three'] = pca_result[:,2]

df['pca-four'] = pca_result[:,3]

df['pca-five'] = pca_result[:,4]

print('Bird Image explained variation per principal component: {}'.format(pca.explained_variance_ratio_))



pca_resultP = pca.fit_transform(dfP)

df['pca-one'] = pca_resultP[:,0]

df['pca-two'] = pca_resultP[:,1] 

df['pca-three'] = pca_resultP[:,2]

df['pca-four'] = pca_resultP[:,3]

df['pca-five'] = pca_resultP[:,4]

print('Plane Image explained variation per principal component: {}'.format(pca.explained_variance_ratio_))
pca = PCA().fit(df)

plt.plot(np.cumsum(pca.explained_variance_ratio_))

plt.xlabel('number of components')

plt.ylabel('cumulative explained variance')

plt.title("Scree Plot of Bird Image")

plt.show()





pcaP = PCA().fit(dfP)

plt.plot(np.cumsum(pcaP.explained_variance_ratio_))

plt.xlabel('number of components')

plt.ylabel('cumulative explained variance')

plt.title("Scree Plot of Plane Image")

plt.show()
N = 300

df_subset = df.loc[rndperm[:N],:].copy()

#data_subset = df_subset[df].values

pca = PCA(n_components=5)

pca_result = pca.fit_transform(df_subset)

df_subset['pca-one'] = pca_result[:,0]

df_subset['pca-two'] = pca_result[:,1] 

df_subset['pca-three'] = pca_result[:,2]

df_subset['pca-four'] = pca_result[:,3]

df_subset['pca-five'] = pca_result[:,4]

print('Explained variation per principal component: {}'.format(pca.explained_variance_ratio_))



#Plane Random

N = 300

dfP_subset = dfP.loc[rndpermP[:N],:].copy()

#data_subset = df_subset[df].values

pca = PCA(n_components=5)

pca_result = pca.fit_transform(dfP_subset)

dfP_subset['pca-one'] = pca_result[:,0]

dfP_subset['pca-two'] = pca_result[:,1] 

dfP_subset['pca-three'] = pca_result[:,2]

dfP_subset['pca-four'] = pca_result[:,3]

dfP_subset['pca-five'] = pca_result[:,4]

print('Explained variation per principal component: {}'.format(pca.explained_variance_ratio_))
#bird



#Perplexity Value 30

tsne = TSNE(n_components=2, verbose=1, perplexity=30, n_iter=300)

tsne_results = tsne.fit_transform(df)

print('t-SNE done!')



df['tsne-2d-one'] = tsne_results[:,0]

df['tsne-2d-two'] = tsne_results[:,1]

plt.figure(figsize=(16,10))

plt.title("TNSE with Perplexity as 30")

sns.scatterplot(

    x="tsne-2d-one", y="tsne-2d-two",

    #hue="tsne-2d-one",

    palette=sns.color_palette("hls", 10),

    data=df,

    legend="full",

    alpha=0.3

)

#Perplexity Value 40

tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)

tsne_results = tsne.fit_transform(df)

print('t-SNE done!')



df['tsne-2d-one'] = tsne_results[:,0]

df['tsne-2d-two'] = tsne_results[:,1]

plt.figure(figsize=(16,10))

plt.title("TNSE with Perplexity as 40")

sns.scatterplot(

    x="tsne-2d-one", y="tsne-2d-two",

    #hue="tsne-2d-one",

    palette=sns.color_palette("hls", 10),

    data=df,

    legend="full",

    alpha=0.3

)



#Perplexity Value 50

tsne = TSNE(n_components=2, verbose=1, perplexity=50, n_iter=300)

tsne_results = tsne.fit_transform(df)

print('t-SNE done!')



df['tsne-2d-one'] = tsne_results[:,0]

df['tsne-2d-two'] = tsne_results[:,1]

plt.figure(figsize=(16,10))

plt.title("TNSE with Perplexity as 50")

sns.scatterplot(

    x="tsne-2d-one", y="tsne-2d-two",

    #hue="tsne-2d-one",

    palette=sns.color_palette("hls", 10),

    data=df,

    legend="full",

    alpha=0.3

)
#plane



#Perplexity Value 30

tsne = TSNE(n_components=2, verbose=1, perplexity=30, n_iter=300)

tsne_results = tsne.fit_transform(dfP)

print('t-SNE done!')



dfP['tsne-2d-one'] = tsne_results[:,0]

dfP['tsne-2d-two'] = tsne_results[:,1]

plt.figure(figsize=(16,10))

plt.title("TNSE with Perplexity as 30")

sns.scatterplot(

    x="tsne-2d-one", y="tsne-2d-two",

    #hue="tsne-2d-one",

    palette=sns.color_palette("hls", 10),

    data=dfP,

    legend="full",

    alpha=0.3

)



#Perplexity Value 40

tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)

tsne_results = tsne.fit_transform(dfP)

print('t-SNE done!')



dfP['tsne-2d-one'] = tsne_results[:,0]

dfP['tsne-2d-two'] = tsne_results[:,1]

plt.figure(figsize=(16,10))

plt.title("TNSE with Perplexity as 40")

sns.scatterplot(

    x="tsne-2d-one", y="tsne-2d-two",

    #hue="tsne-2d-one",

    palette=sns.color_palette("hls", 10),

    data=dfP,

    legend="full",

    alpha=0.3

)



#Perplexity Value 50

tsne = TSNE(n_components=2, verbose=1, perplexity=50, n_iter=300)

tsne_results = tsne.fit_transform(dfP)

print('t-SNE done!')



dfP['tsne-2d-one'] = tsne_results[:,0]

dfP['tsne-2d-two'] = tsne_results[:,1]

plt.figure(figsize=(16,10))

plt.title("TNSE with Perplexity as 50")

sns.scatterplot(

    x="tsne-2d-one", y="tsne-2d-two",

    #hue="tsne-2d-one",

    palette=sns.color_palette("hls", 10),

    data=dfP,

    legend="full",

    alpha=0.3

)
plane = imread("../input/images/3096_colorPlane.jpg")

#birdie = io.imread("../input/images/42049_colorBird.jpg")

img = cv2.cvtColor(plane, cv2.COLOR_BGR2RGB)
pixel_values = img.reshape((-1, 3))

# convert to float

pixel_values = np.float32(pixel_values)



vectorized = img.reshape((-1,3))

vectorized = np.float32(vectorized)



df = pd.DataFrame(pixel_values)

print('Size of the dataframe: {}'.format(df.shape))

#df.head()
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
K = 2

attempts=10

res,label,center=cv2.kmeans(vectorized,K,None,criteria,attempts,cv2.KMEANS_PP_CENTERS)

center = np.uint8(center)

#print(center)

res = center[label.flatten()]

result_image = res.reshape((img.shape))

KMean2= KMeans(n_clusters=2)

KMean2.fit(df)

label2=KMean2.predict(df)
figure_size = 15

plt.figure(figsize=(figure_size,figure_size))

plt.subplot(1,2,1),plt.imshow(img)

plt.title('Original Image'), plt.xticks([]), plt.yticks([])

plt.subplot(1,2,2),plt.imshow(result_image)

plt.title('Segmented Image when K = %i' % K), plt.xticks([]), plt.yticks([])

plt.show()

print("Silhouette Score is:", metrics.silhouette_score(df, label2, metric = 'euclidean'))

print("CH Index is:", metrics.calinski_harabasz_score(df, label2))

sns.scatterplot(df[0],df[1],hue=label2)
K = 3

attempts=10

res,label,center=cv2.kmeans(vectorized,K,None,criteria,attempts,cv2.KMEANS_PP_CENTERS)

center = np.uint8(center)

#print(center)

res = center[label.flatten()]

result_image = res.reshape((img.shape))

KMean3= KMeans(n_clusters=3)

KMean3.fit(df)

label3=KMean3.predict(df)
figure_size = 15

plt.figure(figsize=(figure_size,figure_size))

plt.subplot(1,2,1),plt.imshow(img)

plt.title('Original Image'), plt.xticks([]), plt.yticks([])

plt.subplot(1,2,2),plt.imshow(result_image)

plt.title('Segmented Image when K = %i' % K), plt.xticks([]), plt.yticks([])

plt.show()

print("Silhouette Score is:", metrics.silhouette_score(df, label3, metric = 'euclidean'))

print("CH Index is:", metrics.calinski_harabasz_score(df, label3))

sns.scatterplot(df[0],df[1],hue=label3)
K = 4

attempts=10

res,label,center=cv2.kmeans(vectorized,K,None,criteria,attempts,cv2.KMEANS_PP_CENTERS)

center = np.uint8(center)

#print(center)

res = center[label.flatten()]

result_image = res.reshape((img.shape))

KMean4= KMeans(n_clusters=4)

KMean4.fit(df)

label4=KMean4.predict(df)
figure_size = 15

plt.figure(figsize=(figure_size,figure_size))

plt.subplot(1,2,1),plt.imshow(img)

plt.title('Original Image'), plt.xticks([]), plt.yticks([])

plt.subplot(1,2,2),plt.imshow(result_image)

plt.title('Segmented Image when K = %i' % K), plt.xticks([]), plt.yticks([])

plt.show()

print("Silhouette Score is:", metrics.silhouette_score(df, label4, metric = 'euclidean'))

print("CH Index is:", metrics.calinski_harabasz_score(df, label4))

sns.scatterplot(df[0],df[1],hue=label4)
K = 5

attempts=10

res,label,center=cv2.kmeans(vectorized,K,None,criteria,attempts,cv2.KMEANS_PP_CENTERS)

center = np.uint8(center)

#print(center)

res = center[label.flatten()]

result_image = res.reshape((img.shape))

KMean5= KMeans(n_clusters=5)

KMean5.fit(df)

label5=KMean5.predict(df)
figure_size = 15

plt.figure(figsize=(figure_size,figure_size))

plt.subplot(1,2,1),plt.imshow(img)

plt.title('Original Image'), plt.xticks([]), plt.yticks([])

plt.subplot(1,2,2),plt.imshow(result_image)

plt.title('Segmented Image when K = %i' % K), plt.xticks([]), plt.yticks([])

plt.show()

print("Silhouette Score is:", metrics.silhouette_score(df, label4, metric = 'euclidean'))

print("CH Index is:", metrics.calinski_harabasz_score(df, label5))

sns.scatterplot(df[0],df[1],hue=label5)
#plane = imread("../input/images/3096_colorPlane.jpg")

birdie = imread("../input/images/42049_colorBird.jpg")

img = cv2.cvtColor(birdie, cv2.COLOR_BGR2RGB)
pixel_values = img.reshape((-1, 3))

# convert to float

pixel_values = np.float32(pixel_values)



vectorized = img.reshape((-1,3))

vectorized = np.float32(vectorized)



df = pd.DataFrame(pixel_values)

print('Size of the dataframe: {}'.format(df.shape))

df.head()
K = 2

attempts=10

res,label,center=cv2.kmeans(vectorized,K,None,criteria,attempts,cv2.KMEANS_PP_CENTERS)

center = np.uint8(center)

#print(center)

res = center[label.flatten()]

result_image = res.reshape((img.shape))

KMean2= KMeans(n_clusters=2)

KMean2.fit(df)

label2=KMean2.predict(df)
figure_size = 15

plt.figure(figsize=(figure_size,figure_size))

plt.subplot(1,2,1),plt.imshow(img)

plt.title('Original Image'), plt.xticks([]), plt.yticks([])

plt.subplot(1,2,2),plt.imshow(result_image)

plt.title('Segmented Image when K = %i' % K), plt.xticks([]), plt.yticks([])

plt.show()

print("Silhouette Coefficient is:", metrics.silhouette_score(df, label2, metric = 'euclidean'))

print("CH Index is:", metrics.calinski_harabasz_score(df, label2))

sns.scatterplot(df[0],df[1],hue=label2)
K = 3

attempts=10

res,label,center=cv2.kmeans(vectorized,K,None,criteria,attempts,cv2.KMEANS_PP_CENTERS)

center = np.uint8(center)

#print(center)

res = center[label.flatten()]

result_image = res.reshape((img.shape))

KMean3= KMeans(n_clusters=3)

KMean3.fit(df)

label3=KMean3.predict(df)
figure_size = 15

plt.figure(figsize=(figure_size,figure_size))

plt.subplot(1,2,1),plt.imshow(img)

plt.title('Original Image'), plt.xticks([]), plt.yticks([])

plt.subplot(1,2,2),plt.imshow(result_image)

plt.title('Segmented Image when K = %i' % K), plt.xticks([]), plt.yticks([])

plt.show()

print("Silhouette Coefficient is:", metrics.silhouette_score(df, label3, metric = 'euclidean'))

print("CH Index is:", metrics.calinski_harabasz_score(df, label3))

sns.scatterplot(df[0],df[1],hue=label3)
K = 4

attempts=10

res,label,center=cv2.kmeans(vectorized,K,None,criteria,attempts,cv2.KMEANS_PP_CENTERS)

center = np.uint8(center)

#print(center)

res = center[label.flatten()]

result_image = res.reshape((img.shape))

KMean4= KMeans(n_clusters=4)

KMean4.fit(df)

label4=KMean4.predict(df)
figure_size = 15

plt.figure(figsize=(figure_size,figure_size))

plt.subplot(1,2,1),plt.imshow(img)

plt.title('Original Image'), plt.xticks([]), plt.yticks([])

plt.subplot(1,2,2),plt.imshow(result_image)

plt.title('Segmented Image when K = %i' % K), plt.xticks([]), plt.yticks([])

plt.show()

print("Silhouette Coefficient is:", metrics.silhouette_score(df, label4, metric = 'euclidean'))

print("CH Index is:", metrics.calinski_harabasz_score(df, label4))

sns.scatterplot(df[0],df[1],hue=label4)
K = 5

attempts=10

res,label,center=cv2.kmeans(vectorized,K,None,criteria,attempts,cv2.KMEANS_PP_CENTERS)

center = np.uint8(center)

#print(center)

res = center[label.flatten()]

result_image = res.reshape((img.shape))

KMean5= KMeans(n_clusters=3)

KMean5.fit(df)

label5=KMean5.predict(df)
figure_size = 15

plt.figure(figsize=(figure_size,figure_size))

plt.subplot(1,2,1),plt.imshow(img)

plt.title('Original Image'), plt.xticks([]), plt.yticks([])

plt.subplot(1,2,2),plt.imshow(result_image)

plt.title('Segmented Image when K = %i' % K), plt.xticks([]), plt.yticks([])

plt.show()

print("Silhouette Coefficient is:", metrics.silhouette_score(df, label5, metric = 'euclidean'))

print("CH Index is:", metrics.calinski_harabasz_score(df, label5))

sns.scatterplot(df[0],df[1],hue=label5)
plane = imread("../input/images/3096_colorPlane.jpg")

#birdie = io.imread("../input/images/42049_colorBird.jpg")

img = cv2.cvtColor(plane, cv2.COLOR_BGR2RGB)

pixel_values = img.reshape((-1, 3))

# convert to float

pixel_values = np.float32(pixel_values)



vectorized = img.reshape((-1,3))

vectorized = np.float32(vectorized)



df = pd.DataFrame(pixel_values)

print('Size of the dataframe: {}'.format(df.shape))

df.head()
gmm = GMM(n_components=2).fit(df)

labelGMM2 = gmm.predict(df)

print("Silhouette Coefficient is:", metrics.silhouette_score(df, labelGMM2, metric = 'euclidean'))

print("CH Index is:", metrics.calinski_harabasz_score(df, labelGMM2))

sns.scatterplot(df[0],df[1],hue=labelGMM2)
gmm = GMM(n_components=3).fit(df)

labelGMM3 = gmm.predict(df)

print("Silhouette Coefficient is:", metrics.silhouette_score(df, labelGMM3, metric = 'euclidean'))

print("CH Index is:", metrics.calinski_harabasz_score(df, labelGMM3))

sns.scatterplot(df[0],df[1],hue=labelGMM3)
gmm = GMM(n_components=4).fit(df)

labelGMM4 = gmm.predict(df)

print("Silhouette Coefficient is:", metrics.silhouette_score(df, labelGMM4, metric = 'euclidean'))

print("CH Index is:", metrics.calinski_harabasz_score(df, labelGMM4))

sns.scatterplot(df[0],df[1],hue=labelGMM4)
gmm = GMM(n_components=5).fit(df)

labelGMM5 = gmm.predict(df)

print("Silhouette Coefficient is:", metrics.silhouette_score(df, labelGMM5, metric = 'euclidean'))

print("CH Index is:", metrics.calinski_harabasz_score(df, labelGMM5))

sns.scatterplot(df[0],df[1],hue=labelGMM5)
#plane = imread("../input/images/3096_colorPlane.jpg")

birdie = imread("../input/images/42049_colorBird.jpg")

img = cv2.cvtColor(birdie, cv2.COLOR_BGR2RGB)

pixel_values = img.reshape((-1, 3))

# convert to float

pixel_values = np.float32(pixel_values)



vectorized = img.reshape((-1,3))

vectorized = np.float32(vectorized)



df = pd.DataFrame(pixel_values)

print('Size of the dataframe: {}'.format(df.shape))

df.head()
gmm = GMM(n_components=2).fit(df)

labelGMM2 = gmm.predict(df)

print("Silhouette Coefficient is:", metrics.silhouette_score(df, labelGMM2, metric = 'euclidean'))

print("CH Index is:", metrics.calinski_harabasz_score(df, labelGMM2))

sns.scatterplot(df[0],df[1],hue=labelGMM2)
gmm = GMM(n_components=3).fit(df)

labelGMM3 = gmm.predict(df)

print("Silhouette Coefficient is:", metrics.silhouette_score(df, labelGMM3, metric = 'euclidean'))

print("CH Index is:", metrics.calinski_harabasz_score(df, labelGMM3))

sns.scatterplot(df[0],df[1],hue=labelGMM3)
gmm = GMM(n_components=4).fit(df)

labelGMM4 = gmm.predict(df)

print("Silhouette Coefficient is:", metrics.silhouette_score(df, labelGMM4, metric = 'euclidean'))

print("CH Index is:", metrics.calinski_harabasz_score(df, labelGMM4))

sns.scatterplot(df[0],df[1],hue=labelGMM4)
gmm = GMM(n_components=5).fit(df)

labelGMM5 = gmm.predict(df)

print("Silhouette Coefficient is:", metrics.silhouette_score(df, labelGMM5, metric = 'euclidean'))

print("CH Index is:", metrics.calinski_harabasz_score(df, labelGMM5))

sns.scatterplot(df[0],df[1],hue=labelGMM5)