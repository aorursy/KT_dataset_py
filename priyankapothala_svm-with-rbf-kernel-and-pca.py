import numpy as np

import pandas as pd

from sklearn.model_selection import cross_val_score

from sklearn.preprocessing import StandardScaler

from sklearn.decomposition import PCA

from sklearn.svm import SVC

from sklearn.metrics import accuracy_score,confusion_matrix

import matplotlib.pyplot as plt
train = pd.read_csv('../input/digit-recognizer/train.csv')

train.head()
test = pd.read_csv('../input/digit-recognizer/train.csv')

test.head()
train.shape,test.shape
X = train.drop('label', axis=1)

y = train['label']
#Standardizing the features

scaler = StandardScaler()

scaler.fit(X)

X = scaler.transform(X)
pca = PCA()

pca.fit(X)
plt.plot(np.cumsum(pca.explained_variance_ratio_))

plt.xlabel('No of Components')

plt.ylabel('Cumulative Explained Variance')

plt.show()
#Initializing PCA to retain principal components contributing to 95% variance 

pca = PCA(0.95)

pca.fit(X)
print('Principal Components', pca.n_components_)
X = pca.transform(X)
temp_df = pd.DataFrame(data = X[:,0:2],columns = ['Principal Component 1', 'Principal Component 2'])

label_df = pd.DataFrame(data = y,columns = ['label'])

pc_df = pd.concat([temp_df, label_df], axis = 1)
pc_df.head()
fig = plt.figure(figsize=(15, 10))

colors = ['red', 'blue', 'green', 'darkorange', 'maroon',

          'yellow', 'black', 'lightgreen', 'pink', 'skyblue']

markers = ['*', 's', '+', 'x', 'D', 'o', '1', '8', 'p', 'v']

for label, color, marker in zip(np.unique(y), colors, markers):

    index = (pc_df['label'] == label)

    plt.scatter(pc_df.loc[index, 'Principal Component 1'],

                pc_df.loc[index, 'Principal Component 2'],

                c=color, marker=marker, label=label)

plt.legend(loc='best')

plt.title('Scatterplot PC1 vs PC2')

plt.xlabel('Principal Component 1')

plt.ylabel('Principal Component 2')

plt.show()

fig.savefig('scatterplot_pc1_pc2.png', dpi=fig.dpi)
svclassifier = SVC(kernel='rbf',gamma="scale")
accuracy = cross_val_score(svclassifier, X, y, scoring='accuracy', cv = 3)
print("Accuracy:",round(accuracy.mean() * 100,2))
svclassifier.fit(X,y)
test = scaler.transform(test)

test = pca.transform(test)

results = svclassifier.predict(test)
submissions_df=pd.DataFrame({"ImageId": list(range(1,len(results)+1)),

                         "Label": results})

submissions_df.to_csv("../output/digit-recognizer/sample_submission.csv", index=False, header=True)