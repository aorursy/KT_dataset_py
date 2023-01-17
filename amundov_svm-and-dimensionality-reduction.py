import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.manifold import TSNE #Only visualization, sklearn does noe support .transform - method

from sklearn import svm
from sklearn.linear_model import SGDClassifier as SGD
from sklearn.neighbors import KNeighborsClassifier as KNN

import time
import matplotlib.pyplot as plt
import matplotlib
%matplotlib inline

import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

data = pd.read_csv("../input/train.csv")

y_train = data[:2000]['label'].values
x_train_raw = data[:2000].drop(labels = 'label',axis = 1).values

y_test = data[40000:42000]['label'].values
x_test_raw = data[40000:42000].drop(labels = 'label',axis = 1).values

scaler = StandardScaler()

train_input = scaler.fit_transform(x_train_raw)
test_input = scaler.transform(x_test_raw)


for i in range(25):
    plt.subplot(5,5,i+1)
    plt.imshow((x_train_raw[i*42]).reshape(28,28)) 
    plt.xticks([])
    plt.yticks([])  
    


    
    
pca = PCA().fit(x_train_raw)
plt.plot(pca.explained_variance_)


pca = PCA(n_components = 100).fit(train_input)
eigenvalues = pca.components_[:5]
fig1, ax1 = plt.subplots() 
ax1.plot(eigenvalues[0])
ax1.plot(eigenvalues[1])
ax1.plot(eigenvalues[2])
ax1.set_title('Three first eigenvalues')
plt.show()

fig2,ax2 = plt.subplots()
ax2.plot(pca.explained_variance_)
ax2.set_title('Variance in components')
plt.show()

plt.figure(figsize = (10,5))
for i in range(len(eigenvalues)):
    plt.subplot(1,5,i+1)
    plt.imshow((eigenvalues[i]).reshape(28,28)) 
    plt.xticks([])
    plt.yticks([])  


pca = PCA(n_components = 2).fit(x_train_raw)
train_pca = pca.transform(x_train_raw)

lda = LDA(n_components =2).fit(x_train_raw,y_train)
train_lda = lda.transform(x_train_raw)

train_tsne = TSNE(n_components = 2).fit_transform(x_train_raw)



 
ax = sns.scatterplot(
                train_pca[:,0],train_pca[:,1],
                hue = y_train, 
                palette = sns.color_palette( "hls",n_colors = 10)
                )    
ax.set_title('PCA 2-comp')
plt.show()

ax = sns.scatterplot(
                train_lda[:,0],train_lda[:,1],
                hue = y_train, 
                palette = sns.color_palette( "hls",n_colors = 10)
                )    
ax.set_title('LDA 2-comp')
plt.show()

ax = sns.scatterplot(
                train_tsne[:,0],train_tsne[:,1],
                hue = y_train, 
                palette = sns.color_palette( "hls",n_colors = 10)
                )    
ax.set_title('TSNE')
plt.show()
pca = PCA(n_components = 9).fit(x_train_raw)
train_pca = pca.transform(x_train_raw)

lda = LDA(n_components =9).fit(x_train_raw,y_train)
train_lda = lda.transform(x_train_raw)
fig, axs = plt.subplots(2,2, figsize =(12,12))
axs[0,0].scatter(
    train_pca[:,0],train_pca[:,1],
    c = y_train,
    cmap = plt.get_cmap('gist_rainbow' ),
    alpha = 0.5) 
axs[0,0].set_title('PCA 1st and 2nd component')

axs[0,1].scatter(
    train_pca[:,1],train_pca[:,2],
    c = y_train,
    cmap = plt.get_cmap('gist_rainbow' ),
    alpha = 0.5) 
axs[0,1].set_title('PCA 2nd and 3rd component')

axs[1,0].scatter(
    train_pca[:,2],train_pca[:,3],
    c = y_train,
    cmap = plt.get_cmap('gist_rainbow' ),
    alpha = 0.5) 
axs[1,0].set_title('PCA 3rd and 4th component')

axs[1,1].scatter(
    train_pca[:,3],train_pca[:,4],
    c = y_train,
    cmap = plt.get_cmap('gist_rainbow' ),
    alpha = 0.5) 
axs[1,1].set_title('PCA 4th and 5th component')

fig, axs = plt.subplots(2,2, figsize =(12,12))
axs[0,0].scatter(
    train_lda[:,0],train_lda[:,1],
    c = y_train,
    cmap = plt.get_cmap('gist_rainbow' ),
    alpha = 0.5) 
axs[0,0].set_title('LDA 1st and 2nd component')

axs[0,1].scatter(
    train_lda[:,1],train_lda[:,2],
    c = y_train,
    cmap = plt.get_cmap('gist_rainbow' ),
    alpha = 0.5) 
axs[0,1].set_title('LDA 2nd and 3rd component')

axs[1,0].scatter(
    train_lda[:,2],train_lda[:,3],
    c = y_train,
    cmap = plt.get_cmap('gist_rainbow' ),
    alpha = 0.5) 
axs[1,0].set_title('LDA 3rd and 4th component')

axs[1,1].scatter(
    train_lda[:,3],train_lda[:,4],
    c = y_train,
    cmap = plt.get_cmap('gist_rainbow' ),
    alpha = 0.5) 
axs[1,1].set_title('LDA 4th and 5th component')
def testClassifiers(x_train_scaled, y_train, x_test_scaled, y_test,n_components,use_lda = False ):
    start = time.time()
    pca = PCA(n_components = n_components)      
    models = (
        svm.SVC(kernel = 'rbf', C = 10, gamma = 0.001),
        svm.SVC(kernel = 'poly' , degree = 2),
        KNN(n_neighbors = 5),
        #SGD(loss = 'log') #logistic regression    
       )    
    train_pca = pca.fit_transform(x_train_scaled)
    test_pca = pca.transform(x_test_scaled)    
    predictions = []
    
    for model in models:
        model.fit(train_pca,y_train)
        predictions.append(model.predict(test_pca))
    
    if use_lda == True:
        lda = LDA(n_components = n_components)
        train_lda = lda.fit_transform(x_train_scaled,y_train)
        test_lda = lda.transform(x_test_scaled)
        for model in models:
            model.fit(train_lda,y_train)
            predictions.append(model.predict(test_lda))
        labels = ['PCA_SVC_rbf','PCA_SVC_poly','PCA_KNN','LDA_SVC_rbf','LDA_SVC_poly','LDA_KNN']
    else:
        
        labels = ['SVC_rbf','SVC_poly','KNN']

    scores = pd.DataFrame()
    
    for i,label in enumerate(labels):
        scores[label] = [accuracy_score(y_test,predictions[i])]
    scores['time'] = [time.time()-start]
    scores['components'] = n_components
    return scores



results = pd.DataFrame()

for components in range(2,18,2):
    df = testClassifiers(train_input, y_train, test_input, y_test,components,use_lda=True)
    results = pd.concat([results,df])
     
print(results.head())

labels = ['PCA_SVC_rbf','PCA_SVC_poly','PCA_KNN','LDA_SVC_rbf','LDA_SVC_poly','LDA_KNN']
x = results['components'].values

f,axs = plt.subplots(1,2, figsize = (10,5))

for label in labels:
    axs[0].scatter(x , results[label].values)
axs[0].legend(labels)

for label in labels:
    axs[1].plot(x , results[label].values)
axs[1].legend(labels)

    


results = pd.DataFrame()

for components in range(2,100,4):
    df = testClassifiers(train_input, y_train, test_input, y_test,components)
    results = pd.concat([results,df])

for components in range(100,200,10):
    df = testClassifiers(train_input, y_train, test_input, y_test,components)
    results = pd.concat([results,df])
for components in range(200,784,40):
    df = testClassifiers(train_input, y_train, test_input, y_test,components)
    results = pd.concat([results,df])
     



labels = ['SVC_rbf','SVC_poly','KNN']

x = results['components'][5:].values

f,axs = plt.subplots(1,2, figsize = (10,5))

for label in labels:
    axs[0].scatter(x , results[label][5:].values)
axs[0].legend(labels)

for label in labels:
    axs[1].plot(x , results[label][5:].values)
axs[1].legend(labels)

y_train = data[:40000]['label'].values
x_train_raw = data[:40000].drop(labels = 'label',axis = 1).values

y_test = data[40000:42000]['label'].values
x_test_raw = data[40000:42000].drop(labels = 'label',axis = 1).values

scaler = StandardScaler()

train_input = scaler.fit_transform(x_train_raw)
test_input = scaler.transform(x_test_raw)
models = (
    svm.SVC(kernel = 'rbf', C = 10, gamma = 0.001),
    svm.SVC(kernel = 'poly' , degree = 3) 
    )
t_rbf = time.time()
pca = PCA(n_components = 90).fit(train_input)
train_pca = pca.transform(train_input)
test_pca = pca.transform(test_input)
    
models[0].fit(train_pca,y_train)
predictions = models[0].predict(test_pca)
rbf_score =  accuracy_score(y_test, predictions)
t_rbf = time.time()-t_rbf

t_poly = time.time()
pca = PCA(n_components = 200).fit(train_input)
train_pca = pca.transform(train_input)
test_pca = pca.transform(test_input)
    
models[1].fit(train_pca,y_train)
predictions = models[1].predict(test_pca)
poly_score =  accuracy_score(y_test, predictions)  
t_poly = time.time()-t_poly


scores_final = pd.DataFrame([[rbf_score,t_rbf],[poly_score,t_poly]],columns = ['score','time'],index = ['rbf','poly'])
print(scores_final)
