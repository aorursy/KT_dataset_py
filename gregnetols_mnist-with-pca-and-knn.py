# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
train = pd.read_csv('../input/train.csv')

submission = pd.read_csv('../input/test.csv')
y_train = train['label']

X_train = train.drop('label', axis=1)

X_submission = submission
y_train.head()
X_train.head()
X_submission.head()
from sklearn.decomposition import PCA



pca = PCA(n_components=2)

pca_result = pca.fit_transform(X_train)



print(pca.explained_variance_ratio_)
print(X_train.shape)

print(pca_result.shape)
plt.scatter(pca_result[:4000, 0], pca_result[:4000, 1], c=y_train[:4000], edgecolor='none', alpha=0.5,

           cmap=plt.get_cmap('jet', 10), s=5)

plt.colorbar()
pca = PCA(200)

pca_full = pca.fit(X_train)



plt.plot(np.cumsum(pca_full.explained_variance_ratio_))

plt.xlabel('# of components')

plt.ylabel('Cumulative explained variance')
pca = PCA(n_components=50)

X_train_transformed = pca.fit_transform(X_train)

X_submission_transformed = pca.transform(X_submission)
from sklearn.model_selection import train_test_split



X_train_pca, X_test_pca, y_train_pca, y_test_pca = train_test_split( 

    X_train_transformed, y_train, test_size=0.2, random_state=13)

components = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]

neighbors = [1, 2, 3, 4, 5, 6, 7]



scores = np.zeros( (components[len(components)-1]+1, neighbors[len(neighbors)-1]+1 ) )
from sklearn.neighbors import KNeighborsClassifier



for component in components:

    for n in neighbors:

        knn = KNeighborsClassifier(n_neighbors=n)

        knn.fit(X_train_pca[:,:component], y_train_pca)

        score = knn.score(X_test_pca[:,:component], y_test_pca)

        #predict = knn.predict(X_test_pca[:,:component])

        scores[component][n] = score

        

        print('Components = ', component, ', neighbors = ', n,', Score = ', score)   
scores = np.reshape(scores[scores != 0], (len(components), len(neighbors)))



x = [0, 1, 2, 3, 4, 5, 6]

y = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]



plt.rcParams["axes.grid"] = False



fig, ax = plt.subplots()

plt.imshow(scores, cmap='hot', interpolation='none', vmin=.90, vmax=1)

plt.xlabel('neighbors')

plt.ylabel('components')

plt.xticks(x, neighbors)

plt.yticks(y, components)

plt.title('KNN score heatmap')



plt.colorbar()

plt.show()
knn = KNeighborsClassifier(n_neighbors=5)

knn.fit(X_train_pca[:, :35], y_train_pca)



predict_labels = knn.predict(X_submission_transformed[:, :35])
predict_labels
Submission = pd.DataFrame({

        "ImageId": range(1, predict_labels.shape[0]+1),

        "Label": predict_labels

    })



Submission.to_csv("KnnMnistSubmission.csv", index=False)



Submission.head(5)
Submission.shape