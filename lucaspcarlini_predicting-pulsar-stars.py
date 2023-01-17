import pandas as pd
data = pd.read_csv('../input/predicting-a-pulsar-star/pulsar_stars.csv')



data.head()
target = data[['target_class']]



data.drop('target_class', axis=1, inplace=True)



target['target_class'].value_counts()
data.describe()
data.corr()
from sklearn.manifold import TSNE

from sklearn.manifold import Isomap



import numpy as np



import matplotlib.pyplot as plt
tsne = TSNE(n_components=2, init='pca', perplexity = 40)

tsne_data = tsne.fit_transform(data)
not_pulsar = []

pulsar = []



for i in range(len(target)):

    if target['target_class'][i] == 0:

        not_pulsar.append(tsne_data[i])

    if target['target_class'][i] == 1:

        pulsar.append(tsne_data[i])

        

not_pulsar = np.array(not_pulsar)

pulsar = np.array(pulsar)
plt.figure(figsize=(7,7))

plt.scatter(not_pulsar[:,0], not_pulsar[:,1], c='blue', label='Not Pulsar Stars')

plt.scatter(pulsar[:,0], pulsar[:,1], c='red', label='Pulsar Stars')

plt.legend()

plt.title('Low dimensional visualization (t-SNE) - Pulsar Stars');
isomap = Isomap(n_components=2, n_neighbors=5, path_method='D', n_jobs=-1)



isomap_data = isomap.fit_transform(data)
not_pulsar = []

pulsar = []



for i in range(len(target)):

    if target['target_class'][i] == 0:

        not_pulsar.append(isomap_data[i])

    if target['target_class'][i] == 1:

        pulsar.append(isomap_data[i])

        

not_pulsar = np.array(not_pulsar)

pulsar = np.array(pulsar)
plt.figure(figsize=(7,7))

plt.scatter(not_pulsar[:,0], not_pulsar[:,1], c='blue', label='Not Pulsar Stars')

plt.scatter(pulsar[:,0], pulsar[:,1], c='red', label='Pulsar Stars')

plt.legend()

plt.title('Low dimensional visualization (ISOMAP) - Pulsar Stars');
from sklearn.decomposition import PCA

from sklearn.neighbors import KNeighborsClassifier



from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler



from scipy.stats import norm
data_train, data_test, target_train, target_test = train_test_split(data, np.array(target['target_class']), test_size=0.2, random_state=0)



print('train size = ', len(data_train))

print('test size = ', len(data_test))
pd.Series(target_train).value_counts()
pd.Series(target_test).value_counts()
scaler = StandardScaler()

scaler.fit(data_train)



data_train_scaled = scaler.transform(data_train)

data_test_scaled = scaler.transform(data_test)
pca = PCA().fit(data_train_scaled)

pca_data_train = pca.transform(data_train_scaled)

print("Variance explained by each component (%): ")

for i in range(len(pca.explained_variance_ratio_)):

      print("\n",i+1,"º:", pca.explained_variance_ratio_[i]*100)

        

print("\nTotal sum (%): ",sum(pca.explained_variance_ratio_)*100)



print("\nSum of the first two components (%): ",(pca.explained_variance_ratio_[0] + pca.explained_variance_ratio_[1])*100)
not_pulsar = []

pulsar = []



for i in range(len(target_train)):

    if target_train[i] == 0:

        not_pulsar.append(pca_data_train[i])

    if target_train[i] == 1:

        pulsar.append(pca_data_train[i])

        

not_pulsar = np.array(not_pulsar)

pulsar = np.array(pulsar)
plt.figure(figsize=(7,7))

plt.scatter(not_pulsar[:,0], not_pulsar[:,1], c='blue', label='Not Pulsar Stars')

plt.scatter(pulsar[:,0], pulsar[:,1], c='red', label='Pulsar Stars')

plt.legend()

plt.title('Low dimensional visualization (PCA) - Pulsar Stars');
pca = PCA(n_components = 2).fit(data_train_scaled)



pca_data_test = pca.transform(data_test_scaled)

pca_data_train = pca.transform(data_train_scaled)
accuracy = []

for k in range(1,20):

    knn = KNeighborsClassifier(n_neighbors=k, p=2)

    knn.fit(pca_data_train, target_train)

    accuracy.append(knn.score(pca_data_test, target_test))
plt.plot(range(1,20),accuracy, 'bx-');

plt.xlabel('k number of neighbors')

plt.ylabel('Accuracy')

plt.title('Optimal number of neighbors');



print( "The best accuracy was", np.round(np.array(accuracy).max()*100, 2), "% with k =",  np.array(accuracy).argmax()+1) 

lda = LDA(n_components=1).fit(data_train_scaled, target_train)



lda_data_train = lda.transform(data_train_scaled)

lda_data_test = lda.transform(data_test_scaled)
not_pulsar = []

pulsar = []



for i in range(len(target_train)):

    if target_train[i] == 0:

        not_pulsar.append(lda_data_train[i])

    if target_train[i] == 1:

        pulsar.append(lda_data_train[i])

        

not_pulsar = np.array(not_pulsar)

pulsar = np.array(pulsar)
pulsar_mean, pulsar_std = norm.fit(pulsar)

not_pulsar_mean, not_pulsar_std = norm.fit(not_pulsar)

all_mean, all_std = norm.fit(lda_data_train)



x = np.linspace(-7, 12, 10000)

pulsar_pdf = norm.pdf(x, pulsar_mean, pulsar_std)

not_pulsar_pdf = norm.pdf(x, not_pulsar_mean, not_pulsar_std)

all_pdf = norm.pdf(x, all_mean, all_std)
plt.figure(figsize=(10,6))

plt.scatter(pulsar_mean,0, marker='X', c='red',s=400)

plt.scatter(not_pulsar_mean,0, marker='X', c='blue',s=400)

plt.scatter(all_mean,0, marker='X', c='k',s=400)

plt.scatter(lda_data_train[:,0], np.zeros((len(lda_data_train),1)), c= ['red' if l==1  else 'blue' for l in target_train])

plt.ylim([-0.5,0.7])

plt.xlim([-7,12])

plt.plot(x, pulsar_pdf, 'r', linewidth=1.2, label='Pulsar Stars')

plt.plot(x, not_pulsar_pdf, 'b', linewidth=1.2, label='Not Pulsar Stars')

plt.plot(x, all_pdf, 'k', linewidth=1.2, label='All data')

plt.xlabel('Discriminant Hyperplane')

plt.ylabel('Probability Density Function')

plt.legend()

plt.title('LDA model');
accuracy = []

for k in range(1,20):

    knn = KNeighborsClassifier(n_neighbors=k, p=2)

    knn.fit(lda_data_train, target_train)

    accuracy.append(knn.score(lda_data_test, target_test))
plt.plot(range(1,20),accuracy, 'bx-');

plt.xlabel('k number of neighbors')

plt.ylabel('Accuracy')

plt.title('Optimal number of neighbors');



print( "The best accuracy was", np.round(np.array(accuracy).max()*100, 2), "% with k =",  np.array(accuracy).argmax()+1)
accuracy = []

for k in range(1,20):

    knn = KNeighborsClassifier(n_neighbors=k, p=2)

    knn.fit(data_train_scaled, target_train)

    accuracy.append(knn.score(data_test_scaled, target_test))
plt.plot(range(1,20),accuracy, 'bx-');

plt.xlabel('k number of neighbors')

plt.ylabel('Accuracy')

plt.title('Optimal number of neighbors');



print( "The best accuracy was", np.round(np.array(accuracy).max()*100, 2), "% with k =",  np.array(accuracy).argmax()+1)