import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset = pd.read_csv('../input/winedata1/Wine.csv')
dataset.head()
#data preparation
x = dataset.iloc[:, 0:13].values
y = dataset.iloc[:, 13].values

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)

#feature scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
#Applying PCA
from sklearn.decomposition import PCA
pca = PCA(n_components = None)
#n_componentes: number of extracted features (independent variables) to get. 
                #None is used as we do not know what is the right amount of features
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)
explained_variance = pca.explained_variance_ratio_
#percentage of variance explained by each of the principal components that we extracted

explained_variance
# --------------------------------------------------------------------------------------------------
x = dataset.iloc[:, 0:13].values
y = dataset.iloc[:, 13].values

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)

#feature scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
#----------------------------------------------------------------------------------------------------
#Doing this part again just because X_train and X_test are already transformed from the previous step

pca = PCA(n_components = 2)
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)

X_train[:5, :]
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression()
classifier.fit(X_train, Y_train)
y_pred = classifier.predict(X_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test, y_pred)

cm
x = dataset.iloc[:, 0:13].values
y = dataset.iloc[:, 13].values

X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)

#feature scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
lda = LDA(n_components = 2)
X_train = lda.fit_transform(X_train, Y_train) #y_train is required as this is supervised learning
X_test = lda.transform(X_test)

classifier = LogisticRegression()
classifier.fit(X_train, Y_train)
y_pred = classifier.predict(X_test)

cm = confusion_matrix(Y_test, y_pred)
cm
dataset2 = pd.read_csv('../input/social-network-ads/Social_Network_Ads.csv')
dataset2.head()
x = dataset2.iloc[:, [2,3]].values
y = dataset2.iloc[:, 4].values

X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# ------------ Please pay attention to this following part ---------------
from sklearn.decomposition import KernelPCA
kpca = KernelPCA(n_components = 2, kernel = 'rbf') 
#rbf is "gaussian" method that maps values to a higher dimension

X_train = kpca.fit_transform(X_train)
X_test = kpca.transform(X_test)
# ------------------------------------------------------------------------

classifier = LogisticRegression()
classifier.fit(X_train, Y_train)
y_pred = classifier.predict(X_test)

cm = confusion_matrix(Y_test, y_pred)
cm
