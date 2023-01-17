import numpy as np 

import pandas as pd 

import tensorflow as tf



df_test = pd.read_csv("../input/mnist-in-csv/mnist_test.csv")

df_train = pd.read_csv("../input/mnist-in-csv/mnist_train.csv")
x = df_train[df_train.columns[1:]]
y = df_train['label']

y.shape
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.20, random_state = 0)
X_train.shape, X_test.shape, y_train.shape, y_test.shape
from sklearn import preprocessing

scaler = preprocessing.StandardScaler()

X_trainn = scaler.fit_transform(X_train)
from sklearn.linear_model import LogisticRegression

lr = LogisticRegression(max_iter=5)

lr.fit(X_trainn, y_train)
from sklearn import preprocessing

scaler2 = preprocessing.StandardScaler()

X_testn = scaler2.fit_transform(X_test)
print('Accurisie: {}'.format(lr.score(X_testn,y_test)))
x_dfTest=df_test[df_test.columns[1:]]

y_dfTest=df_test['label']

scaler1 = preprocessing.StandardScaler()

x_dfTest = scaler1.fit_transform(x_dfTest)
#print(x_dfTest[20:50,0:784])



d=lr.predict(x_dfTest[20:50,0:784])

print(d)
print(y_dfTest[20:50].values)
from sklearn.metrics import confusion_matrix

predict =  lr.predict(x_dfTest)

print(confusion_matrix(y_dfTest, predict))

from sklearn.metrics import classification_report

print(classification_report(y_dfTest, lr.predict(x_dfTest)))

import matplotlib.pyplot as plt

%matplotlib inline

class_table = ["0","1","2","3","4","5","6","7","8","9"]



def get_label_cls(label):

 

    return class_table[label]
b=0

for i in range(20,50): 

    sample = np.reshape(df_test[df_test.columns[1:]].iloc[i].values, (28,28))

    

    plt.figure()

    plt.title("labeled predict:  {} / labeled real: {}".format(get_label_cls(d[b]),get_label_cls(df_test["label"].iloc[i])))

    plt.imshow(sample, 'gray')

    b+=1
scaler = preprocessing.MinMaxScaler()

scaler.fit(X_train)

X_trainn = scaler.transform(X_train)

X_testn = scaler.transform(X_test)
# PCA eklenmiş



from sklearn.decomposition import PCA

pca = PCA(n_components=400)

pca=pca.fit(X_trainn)

x_pca = pca.transform(X_trainn)

xtest_pca= pca.transform(X_testn)
print("sum variance: ", sum(pca.explained_variance_ratio_))
pca_lr = LogisticRegression(max_iter=5)

pca_lr.fit(x_pca, y_train)
print('Accurisie: {}'.format(pca_lr.score(xtest_pca,y_test)))
array=pca.transform(x_dfTest)
#array[0:10000,0:256]

d_pca=pca_lr.predict(array[20:50,0:400])

print(d_pca)
print(y_dfTest[20:50].values)
from sklearn.metrics import confusion_matrix

predict_pca =  pca_lr.predict(array)

print(confusion_matrix(y_dfTest, predict_pca))
from sklearn.metrics import classification_report

print(classification_report(y_dfTest, predict_pca))
b=0

for i in range(20,50): 

    sample = np.reshape(array[i,0:400], (20,20))

    print("predict: {}  /  real: {}".format(get_label_cls(d_pca[b]),get_label_cls(y_dfTest[i])))

    b+=1
scaler = preprocessing.MinMaxScaler()

scaler.fit(X_train)

X_trainn = scaler.transform(X_train)

X_testn = scaler.transform(X_test)
#LDA eklenmiş Hali



from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

lda = LDA(n_components=400)

lda=lda.fit(X_trainn, y_train)

lda_train = lda.transform(X_trainn)

lda_test= lda.transform(X_testn)
lda_lr = LogisticRegression(max_iter=5)
lda_lr.fit(lda_train,y_train)


print('Accurisie: {}'.format(lda_lr.score(lda_test,y_test)))
array=lda.transform(x_dfTest)
d_lda=lda_lr.predict(array[20:50,0:400])

print(d_lda)
print(y_dfTest[20:50].values)
from sklearn.metrics import confusion_matrix

predict_lda =  lda_lr.predict(array)

print(confusion_matrix(y_dfTest, predict_lda))
from sklearn.metrics import classification_report

print(classification_report(y_dfTest, predict_lda))
#from skfeature.function.similarity_based import fisher_score
#score = fisher_score.fisher_score(X_train, y_train)