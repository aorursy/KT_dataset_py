import pandas as pd

import matplotlib.pyplot as plt
from sklearn import datasets

digits = datasets.load_digits() 


plt.gray() 

plt.matshow(digits.images[0]) # first digit

plt.show()



plt.matshow(digits.images[1]) #second digit

plt.show()
x=digits.data 

y=digits.target



print(x.shape)

print(x)

print()

print(y.shape)

print(y)
x=pd.DataFrame(x) #converting in to dataframe

y=pd.DataFrame(y,columns=['target']) 
#combining x and y

df=pd.DataFrame(x)

df['target']=y

df.head()
x=x.sub(x.mean(axis=0), axis=1)

x=x.values
from sklearn.decomposition import PCA

pca = PCA(n_components=2) #Selecting 2 component PCA

principalComponents = pca.fit_transform(x) #Fit the model with X and apply the dimensionality reduction on X.

principalDf = pd.DataFrame(data = principalComponents, columns = ['principal component 1', 'principal component 2'])

finalDf = pd.concat([principalDf, y], axis = 1) #Concatinating principal components with class(y)
#plot even digits

d1=finalDf



fig = plt.figure(figsize = (8,8))

ax = fig.add_subplot(1,1,1) 

ax.set_xlabel('Principal Component 1', fontsize = 15)

ax.set_ylabel('Principal Component 2', fontsize = 15)

ax.set_title('2 component PCA', fontsize = 20)



targets = [0,2,4,6,8]

colors = ['r', 'g', 'b','y','black']



for target, color in zip(targets,colors):

    indicesToKeep = (d1['target'] == target)

    ax.scatter(d1.loc[indicesToKeep, 'principal component 1']

               , d1.loc[indicesToKeep, 'principal component 2']

               , c = color

               , s = 50)

ax.legend(targets)

ax.grid()
#plot first five digits

d1=finalDf.head(5)



fig = plt.figure(figsize = (8,8))

ax = fig.add_subplot(1,1,1) 

ax.set_xlabel('Principal Component 1', fontsize = 15)

ax.set_ylabel('Principal Component 2', fontsize = 15)

ax.set_title('2 component PCA', fontsize = 20)



targets = [0,1,2,3,4]

colors = ['r', 'g', 'b','y','black']



for target, color in zip(targets,colors):

    indicesToKeep = (d1['target'] == target)

    ax.scatter(d1.loc[indicesToKeep, 'principal component 1']

               , d1.loc[indicesToKeep, 'principal component 2']

               , c = color

               , s = 50)

ax.legend(targets)

ax.grid()

plt.savefig("6a_without_lib")
#plot last five digits

d1=finalDf.tail(5)



fig = plt.figure(figsize = (8,8))

ax = fig.add_subplot(1,1,1) 

ax.set_xlabel('Principal Component 1', fontsize = 15)

ax.set_ylabel('Principal Component 2', fontsize = 15)

ax.set_title('2 component PCA', fontsize = 20)



targets = [9,0,8,9,8]

colors = ['r', 'g', 'b','r','b']



for target, color in zip(targets,colors):

    indicesToKeep = (d1['target'] == target)

    ax.scatter(d1.loc[indicesToKeep, 'principal component 1']

               , d1.loc[indicesToKeep, 'principal component 2']

               , c = color

               , s = 50)

ax.legend(targets)

ax.grid()
#plot odd digits

d1=finalDf



fig = plt.figure(figsize = (8,8))

ax = fig.add_subplot(1,1,1) 

ax.set_xlabel('Principal Component 1', fontsize = 15)

ax.set_ylabel('Principal Component 2', fontsize = 15)

ax.set_title('2 component PCA', fontsize = 20)



targets = [1,3,5,7,9]

colors = ['r', 'g', 'b','y','black']



for target, color in zip(targets,colors):

    indicesToKeep = (d1['target'] == target)

    ax.scatter(d1.loc[indicesToKeep, 'principal component 1']

               , d1.loc[indicesToKeep, 'principal component 2']

               , c = color

               , s = 50)

ax.legend(targets)

ax.grid()
#plot class 1,2,6,8

d1=finalDf



fig = plt.figure(figsize = (8,8))

ax = fig.add_subplot(1,1,1) 

ax.set_xlabel('Principal Component 1', fontsize = 15)

ax.set_ylabel('Principal Component 2', fontsize = 15)

ax.set_title('2 component PCA', fontsize = 20)



targets = [1,2,6,8]

colors = ['r', 'g', 'b','y']



for target, color in zip(targets,colors):

    indicesToKeep = (d1['target'] == target)

    ax.scatter(d1.loc[indicesToKeep, 'principal component 1']

               , d1.loc[indicesToKeep, 'principal component 2']

               , c = color

               , s = 50)

ax.legend(targets)

ax.grid()