import pandas as pd

import numpy as  np

import matplotlib.pyplot as plt

import seaborn as sns
data = pd.read_csv("../input/emails.csv")

data.head()
labels = data["spam"]

labels.head()
text_data = data["text"]

text_data.head()
from sklearn.feature_extraction.text import CountVectorizer
count_vect = CountVectorizer()
final_counts = count_vect.fit_transform(text_data)
final_counts.get_shape
final_counts = final_counts.todense()
from sklearn import decomposition
pca = decomposition.PCA()
pca.n_components = 2
pca_data = pca.fit_transform(final_counts)
pca_data = np.vstack((pca_data.T,labels)).T
pca_data = pd.DataFrame(data=pca_data,columns=("f1","f2","label"))

pca_data.head()
sns.FacetGrid(pca_data,hue='label',size=7).map(plt.scatter,'f1','f2').add_legend()

plt.show()
from sklearn.manifold import TSNE
tsne_model = TSNE(n_components=2)
tsne_data = tsne_model.fit_transform(final_counts)
tsne_data = np.vstack((tsne_data.T,labels)).T
tsne_df = pd.DataFrame(data=tsne_data,columns=("f1","f2","label"))
sns.FacetGrid(tsne_df,hue='label',size=7).map(plt.scatter,'f1','f2').add_legend()

plt.show()
X,x_test,Y,y_test = train_test_split(text_data,labels,random_state=42,test_size=0.3,shuffle=True)
X.shape
x_test.shape
y_test = np.c_[y_test]

y_test.shape
Y = np.c_[Y]

Y.shape
from sklearn.feature_extraction.text import CountVectorizer
count_vect = CountVectorizer()
X = count_vect.fit_transform(X)
X.get_shape
X = X.todense()
x_test = count_vect.fit_transform(x_test)
x_test.shape
x_test = x_test.todense()
from sklearn.manifold import TSNE
tsne_model = TSNE(n_components=10)
tsne_train = tsne_model.fit_transform(X)
tsne_test = tsne_model.fit_transform(x_test)
from sklearn.neighbors import KNeighborsClassifier
knn_model  = KNeighborsClassifier()
knn_model.fit(tsne_train,Y)
predictions = knn_model.predict(tsne_test)
from sklearn.metrics import accuracy_score
score = accuracy_score(y_test,predictions)

score