import numpy as np 

import pandas as pd 

import tensorflow as tf



#print(check_output(["ls", "../input"]).decode("utf8"))



df_test = pd.read_csv("../input/fashionmnist/fashion-mnist_test.csv")

df_train = pd.read_csv("../input/fashionmnist/fashion-mnist_train.csv")
x = df_train[df_train.columns[1:]]
y = df_train['label']

y.shape
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.25, random_state = 0)
X_train.shape, X_test.shape, y_train.shape, y_test.shape
mean = X_train.mean()

std = X_train.std()



X_train=(X_train-mean)/std
from sklearn.linear_model import LogisticRegression

lr = LogisticRegression(max_iter=1)

lr.fit(X_train, y_train)
mean = X_test.mean()

std = X_test.std()



X_test=(X_test-mean)/std
lr.score(X_test,y_test)
x_dfTest=df_test[df_test.columns[1:]]

y_dfTest=df_test['label']



mean = x_dfTest.mean()

std = x_dfTest.std()



x_dfTest=(x_dfTest-mean)/std
d=lr.predict(x_dfTest.iloc[20:50,0:784])

print(d)
print(y_dfTest[20:50].values)
import matplotlib.pyplot as plt

%matplotlib inline

class_table = [

    "T-shirt",

    "Trouser",

    "Pullover",

    "Dress",

    "Coat",

    "Sandaled",

    "Shirt",

    "Sneaker",

    "Bag",

    "Boot"

]



def get_label_cls(label):

 

    return class_table[label]
b=0

for i in range(20,50): 

    sample = np.reshape(df_test[df_test.columns[1:]].iloc[i].values, (28,28))

    

    plt.figure()

    plt.title("labeled predict:  {} / labeled real: {}".format(get_label_cls(d[b]),get_label_cls(df_test["label"].iloc[i])))

    plt.imshow(sample, 'gray')

    b+=1
# PCA eklenmiş



from sklearn.decomposition import PCA

pca = PCA(n_components=400, whiten=True)

pca.fit(X_train)

pca.fit(X_test)
x_pca = pca.transform(X_train)

xtest_pca = pca.transform(X_test)
print("sum variance: ", sum(pca.explained_variance_ratio_))
pca_lr = LogisticRegression(max_iter=1)

pca_lr.fit(x_pca, y_train)
pca_lr.score(xtest_pca,y_test)
pca_x_dfTest = PCA(n_components=400, whiten=True)

pca_x_dfTest.fit(x_dfTest)
array=pca_x_dfTest.transform(x_dfTest)
#array[0:10000,0:256]

d_pca=pca_lr.predict(array[20:50,0:400])

print(d_pca)
print(y_dfTest[20:50].values)
b=0

for i in range(20,50): 

    sample = np.reshape(array[i,0:400], (20,20))

    print("predict: {}  /  real: {}".format(get_label_cls(d_pca[b]),get_label_cls(y_dfTest[i])))

    b+=1
#LDA eklenmiş Hali



from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

lda = LDA(n_components=625)

lda_train=lda.fit_transform(X_train,y_train)

lda_test=lda.fit_transform(X_test,y_test)
lda_lr = LogisticRegression(max_iter=2)
lda_lr.fit(lda_train, y_train)
lda_lr.score(lda_test,y_test)