# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
file = open("/kaggle/input/mnist-in-csv/mnist_train.csv")

data_train = pd.read_csv(file)



y_train = np.array(data_train.iloc[:, 0])

x_train = np.array(data_train.iloc[:, 1:])



file = open("/kaggle/input/mnist-in-csv/mnist_test.csv")

data_test = pd.read_csv(file)

y_test = np.array(data_test.iloc[:, 0])

x_test = np.array(data_test.iloc[:, 1:])





size_img = 28

threshold_color = 100 / 255

# show n_images numbers

def show_img(x):

    plt.figure(figsize=(8,7))

    if x.shape[0] > 100:

        print(x.shape)

        n_imgs = 16

        n_samples = x.shape[0]

        x = x.reshape(n_samples, size_img, size_img)

        for i in range(n_imgs):

            plt.subplot(4, 4, i+1) #devide figure into 4x4 and choose i+1 to draw

            plt.imshow(x[i])

        plt.show()

    else:

        plt.imshow(x)

        plt.show()
show_img(x_train)
#plot any image (keep in min 1st row n=0)

X=x_train.reshape(x_train.shape[0],28,28)

n=0

print(y_train[n])

plt.imshow(X[n])
data= y_train

counts, bins = np.histogram(data)

plt.hist(bins[:-1], bins, weights=counts)
data_train.head()
data= x_train

counts, bins = np.histogram(data)

print(bins)

plt.hist(bins[:-1], bins, weights=counts)
x_train = np.array(data_train.iloc[:, 1:])

x_train01=x_train

n=200

x_train01[x_train01<n] = 0

x_train01[x_train01>=n] = 1

#print(x_train01[0])

show_img(x_train01)

#plot any image (keep in min 1st row n=0)

X=x_train.reshape(x_train.shape[0],28,28)

n=0

print(y_train[n])

plt.imshow(X[n])
x_test01=x_test

n=200

x_test01[x_test01<n] = 0

x_test01[x_test01>=n] = 1



X=x_test01.reshape(x_test.shape[0],28,28)

n=0

print(y_test[n])

plt.imshow(X[n])
print(x_train01[1:5,:])

data_train.head()
from sklearn.decomposition import PCA

X=x_train01

pca = PCA(n_components=10)

X_pca = pca.fit_transform(X)

print(pca)

print('eigenvectors \n', pca.components_)

print('singular values ', pca.singular_values_)

print('normalized cumulative sum of eigenvalues \n', pca.explained_variance_ratio_)

#print(' mean vector ', pca.mean_)



#print('Projections of class 0 \n ', X_pca[y_train==0])

#print('Projections of class 1 \n ', X_pca[y_train==1])
eig_vals=  pca.singular_values_

eig_vecs= pca.components_

# Make a list of (eigenvalue, eigenvector) tuples

eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:,i]) for i in range(len(eig_vals))]



# Sort the (eigenvalue, eigenvector) tuples from high to low

eig_pairs.sort(key=lambda x: x[0], reverse=True)



# Visually confirm that the list is correctly sorted by decreasing eigenvalues

print('Eigenvalues in descending order:')

for i in eig_pairs:

    print(i[0])
tot = sum(eig_vals)

var_exp = [(i / tot)*100 for i in sorted(eig_vals, reverse=True)]

cum_var_exp = np.cumsum(var_exp)

print(cum_var_exp)
#matplotlib.pyplot.bar(x, height, width=0.8, bottom=None, *, align='center', data=None, **kwargs)

with plt.style.context('seaborn-whitegrid'):

    plt.figure(figsize=(6, 4))



    plt.bar(range(10), var_exp, alpha=0.5, align='center',

            label='individual explained variance')

    plt.step(range(10), cum_var_exp, where='mid',

             label='cumulative explained variance')

    plt.ylabel('Explained variance ratio')

    plt.xlabel('Principal components')

    plt.legend(loc='best')

    plt.tight_layout()
print(X_pca)

X_pca.shape
X=x_train01

pca = PCA(n_components=10)

X_pca = pca.fit_transform(X)

xtest_pca=pca.fit_transform(x_test01)
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis



X=x_train01

y=y_train

lda= LinearDiscriminantAnalysis(n_components=9)

x_lda=lda.fit(X,y).transform(X)



x_lda
xtest_lda=lda.fit(x_train01,y).transform(x_test01)

xtest_lda.shape
X_train= pd.DataFrame(data=x_train)

s=X_train.sum(axis=1)

s=s/784

s
import matplotlib.pyplot as plt

plt.plot(y_train,s, 'o')

plt.ylabel('ratio of ink used')

plt.show()
X=X_pca

from sklearn import svm

clf=svm.SVC()

clf.fit(X, y_train) 

confidence = clf.score(X, y_train) 

print("Precisão SVM = {}".format(confidence)) 

y1=clf.predict(X)

y1_test=clf.predict(xtest_pca)
from sklearn.metrics import confusion_matrix

X=X_pca

classifier=clf

y_tes=y_train

y_pred=clf.predict(X)

cnf_matrix = confusion_matrix(y_tes, y_pred)

cnf_matrix
import seaborn as sns 

plt.title("Confusion Matrix")

sns.heatmap(cnf_matrix,cbar=False,annot=True,cmap="Blues",fmt="d")

y_tes=y_test

X=X_pca

y_pred=y1_test



cnf_matrix = confusion_matrix(y_tes, y_pred)

cnf_matrix



from sklearn.metrics import precision_score

confidence=precision_score(y_tes, y_pred,average='micro')

print("Precisão pca_svm = {}".format(confidence)) 



import seaborn as sns 

plt.title("Confusion Matrix")

sns.heatmap(cnf_matrix,cbar=False,annot=True,cmap="Blues",fmt="d")
X=x_lda

from sklearn import svm

clf=svm.SVC()

clf.fit(X, y_train) 

confidence = clf.score(X, y_train) 

print("Precisão SVM_lda = {}".format(confidence)) 

y2=clf.predict(X)

y2_test=clf.predict(xtest_lda)



from sklearn.metrics import confusion_matrix

classifier=clf

y_tes=y_train

y_pred=y2

cnf_matrix = confusion_matrix(y_tes, y_pred)

cnf_matrix



import seaborn as sns 

plt.title("Confusion Matrix")

sns.heatmap(cnf_matrix,cbar=False,annot=True,cmap="Blues",fmt="d")



from sklearn.metrics import confusion_matrix

classifier=clf

y_tes=y_test

y_pred=y2_test

cnf_matrix = confusion_matrix(y_tes, y_pred)

cnf_matrix





from sklearn.metrics import precision_score

confidence=precision_score(y_tes, y_pred,average='micro')

print("Precisão SVM_lda = {}".format(confidence)) 





import seaborn as sns 

plt.title("Confusion Matrix")

sns.heatmap(cnf_matrix,cbar=False,annot=True,cmap="Blues",fmt="d")
X=x_lda

from sklearn import tree

clf= tree.DecisionTreeClassifier()

clf.fit(X, y_train) 

confidence = clf.score(X, y_train) 

print("Precisão TREE_lda = {}".format(confidence)) 

y3=clf.predict(X)

y3_test=clf.predict(xtest_lda)

X.shape
x_test.shape
from sklearn.metrics import confusion_matrix

y_tes=y_test

y_pred= y3_test



cnf_matrix = confusion_matrix(y_tes, y_pred)

cnf_matrix



from sklearn.metrics import precision_score

confidence=precision_score(y_tes, y_pred,average='micro')

print("Precisão SVM_lda = {}".format(confidence)) 





import seaborn as sns 

plt.title("Confusion Matrix")

sns.heatmap(cnf_matrix,cbar=False,annot=True,cmap="Blues",fmt="d")
X=X_pca

from sklearn import tree

clf= tree.DecisionTreeClassifier()

clf.fit(X, y_train) 

confidence = clf.score(X, y_train) 

print("Precisão TREE_pca = {}".format(confidence)) 

y4=clf.predict(X)
from sklearn.metrics import confusion_matrix

classifier=clf.fit(X_pca, y_train) 

y_tes= y_test

X=xtest_pca

y_pred=clf.predict(X)

y4_test=y_pred



cnf_matrix = confusion_matrix(y_tes, y_pred)

cnf_matrix



confidence = clf.score(xtest_pca, y_tes) 

print("Precisão TREE_pca = {}".format(confidence)) 



import seaborn as sns 

plt.title("Confusion Matrix")

sns.heatmap(cnf_matrix,cbar=False,annot=True,cmap="Blues",fmt="d")
X=X_pca

from sklearn.ensemble import RandomForestClassifier

clf= RandomForestClassifier(n_estimators=100)

clf.fit(X, y_train) 

confidence = clf.score(X, y_train) 

print("Precisão forest_pca = {}".format(confidence)) 

y5=clf.predict(X)
from sklearn.metrics import confusion_matrix

classifier=clf.fit(X_pca, y_train) 

y_tes= y_test

X=xtest_pca

y_pred=clf.predict(X)

y5_test=y_pred



cnf_matrix = confusion_matrix(y_tes, y_pred)

cnf_matrix



confidence = clf.score(xtest_pca, y_tes) 

print("Precisão forest_pca = {}".format(confidence)) 



import seaborn as sns 

plt.title("Confusion Matrix")

sns.heatmap(cnf_matrix,cbar=False,annot=True,cmap="Blues",fmt="d")
X=x_lda

from sklearn.ensemble import RandomForestClassifier

clf= RandomForestClassifier(n_estimators=100)

clf.fit(X, y_train) 

confidence = clf.score(X, y_train) 

print("Precisão forest_lda = {}".format(confidence)) 

y6=clf.predict(X)
from sklearn.metrics import confusion_matrix

classifier=clf.fit(x_lda, y_train) 

y_tes= y_test

X=xtest_lda

y_pred=clf.predict(X)

y6_test=y_pred



cnf_matrix = confusion_matrix(y_tes, y_pred)

cnf_matrix



confidence = clf.score(xtest_lda, y_tes) 

print("Precisão forest_lda = {}".format(confidence)) 



import seaborn as sns 

plt.title("Confusion Matrix")

sns.heatmap(cnf_matrix,cbar=False,annot=True,cmap="Blues",fmt="d")
train_results = pd.DataFrame(data = {'ImageId':y_train,'I1':y1,'I2':y2,'I3':y3,'I4':y4,'I5':y5,'I6':y6})
train_results
train= pd.DataFrame(data = {'I1':y1,'I2':y2,'I3':y3,'I4':y4,'I5':y5,'I6':y6})

mode= train.mode(axis='columns', numeric_only=True)

mode
notsure=mode.dropna()
#plot any image (keep in min 1st row n=0)

X=x_train.reshape(x_train.shape[0],28,28)

n=8468

print(y_train[n])

plt.imshow(X[n])
#plot any image (keep in min 1st row n=0)

X=x_train.reshape(x_train.shape[0],28,28)

n=19502

print(y_train[n])

print(train.loc[[19502]])

plt.imshow(X[n])
na_free = mode.dropna()

only_na = mode[~mode.index.isin(na_free.index)]

only_na.shape

y_tes = y_train[~mode.index.isin(na_free.index)]
from sklearn.metrics import confusion_matrix

y_pred=mode



cnf_matrix = confusion_matrix(y_train, y_pred)

cnf_matrix



from sklearn.metrics import precision_score

confidence=precision_score(y_train, y_pred,average='micro')

print("Precisão combo = {}".format(confidence)) 





import seaborn as sns 

plt.title("Confusion Matrix")

sns.heatmap(cnf_matrix,cbar=False,annot=True,cmap="Blues",fmt="d")
test= pd.DataFrame(data = {'I1':y1_test,'I2':y2_test,'I3':y3_test,'I4':y4_test,'I5':y5_test,'I6':y6_test})

test_mode= test.mode(axis='columns', numeric_only=True)

test_mode
na_free = test_mode.dropna()

only_na = test_mode[~test_mode.index.isin(na_free.index)]

y_tes = y_test[~test_mode.index.isin(na_free.index)]

na_free.shape
from sklearn.metrics import confusion_matrix

y_pred=only_na[0]



cnf_matrix = confusion_matrix(y_tes, y_pred)

cnf_matrix



from sklearn.metrics import precision_score

confidence=precision_score(y_tes, y_pred,average='micro')

print("Precisão combo = {}".format(confidence)) 





import seaborn as sns 

plt.title("Confusion Matrix")

sns.heatmap(cnf_matrix,cbar=False,annot=True,cmap="Blues",fmt="d")
test_mode.dropna()
#plot any image (keep in min 1st row n=0)

X=x_test.reshape(x_test.shape[0],28,28)

n=6569

print(y_test[n])

plt.imshow(X[n])
#import the libraries

from keras.models import Sequential

from keras.layers import Dense, Conv2D, Flatten

from keras.datasets import mnist

from keras.utils import to_categorical

import matplotlib.pyplot as plt

import numpy as np



import numpy as np

import tensorflow as tf

np.random.seed(1337)
#One-Hot Encoding

y_train_one_hot = to_categorical(y_train)

y_test_one_hot = to_categorical(y_test)



#Print the new label

print(y_train_one_hot[0])
# build

model = tf.keras.Sequential()

model.add(tf.keras.layers.Dense(128, activation = tf.nn.relu, input_shape = (28*28,)))

model.add(tf.keras.layers.Dense(10, activation = tf.nn.softmax))
# compile

model.compile(optimizer = 'adam',

              loss = 'sparse_categorical_crossentropy',

              metrics = ['accuracy'])
EPOCH = 10



model.fit(x_train01, y_train, epochs = EPOCH, verbose = 1, validation_split = 0.3)
pred_train = model.predict(x_train01)
pred_train[0]
y7=np.argmax(pred_train, axis = 1)
np.argmax(pred_train[0])
# true label

y_train[0]
from sklearn.metrics import confusion_matrix



cnf_matrix = confusion_matrix(y_train, y7)

cnf_matrix



from sklearn.metrics import precision_score

confidence=precision_score(y_train, y7,average='micro')

print("Precisão NN = {}".format(confidence)) 



import seaborn as sns 

plt.title("Confusion Matrix")

sns.heatmap(cnf_matrix,cbar=False,annot=True,cmap="Blues",fmt="d")
pred_test = model.predict(x_test01)
pred_test.shape
pred_test[0]
np.argmax(pred_test[0])
y_test[0]
test_id = np.arange(1, x_test.shape[0]+1,1)

test_id
predictions = np.argmax(pred_test, axis = 1)

y7_test=predictions
from sklearn.metrics import confusion_matrix



cnf_matrix = confusion_matrix(y_test, predictions)

cnf_matrix



from sklearn.metrics import precision_score

confidence=precision_score(y_test, predictions,average='micro')

print("Precisão NN = {}".format(confidence)) 



import seaborn as sns 

plt.title("Confusion Matrix")

sns.heatmap(cnf_matrix,cbar=False,annot=True,cmap="Blues",fmt="d")
train01= pd.DataFrame(data = {'I1':y1,'I2':y2,'I3':y3,'I4':y4,'I5':y5,'I6':y6,'I7':y7})

mode01= train.mode(axis='columns', numeric_only=True)

mode01
notsure=mode01.dropna()

notsure

#the same as before
na_free = mode01.dropna()

only_na = mode01[~mode01.index.isin(na_free.index)]

y_train01 = y_train[~mode01.index.isin(na_free.index)]

from sklearn.metrics import confusion_matrix

y_pred=mode01



cnf_matrix = confusion_matrix(y_train, y_pred)

cnf_matrix





import seaborn as sns 

plt.title("Confusion Matrix")

sns.heatmap(cnf_matrix,cbar=False,annot=True,cmap="Blues",fmt="d")

test01= pd.DataFrame(data = {'I1':y1_test,'I2':y2_test,'I3':y3_test,'I4':y4_test,'I5':y5_test,'I6':y6_test,'I7':y7_test })

test_mode01= test01.mode(axis='columns', numeric_only=True)

test_mode01
na_free = test_mode01.dropna()

only_na = test_mode01[~test_mode01.index.isin(na_free.index)]

y_tes01 = y_test[~test_mode01.index.isin(na_free.index)]

na_free.shape
from sklearn.metrics import confusion_matrix

y_pred=only_na[0]



cnf_matrix = confusion_matrix(y_tes01, y_pred)

cnf_matrix



from sklearn.metrics import precision_score

confidence=precision_score(y_tes01, y_pred,average='micro')

print("Precisão combo_NN = {}".format(confidence)) 





import seaborn as sns 

plt.title("Confusion Matrix")

sns.heatmap(cnf_matrix,cbar=False,annot=True,cmap="Blues",fmt="d")