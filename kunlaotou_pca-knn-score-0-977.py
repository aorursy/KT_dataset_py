import pandas as pd

from sklearn.decomposition import PCA

from sklearn.preprocessing import Normalizer

from sklearn.cross_validation import train_test_split

import numpy as np

from sklearn.neighbors import KNeighborsClassifier



# The competition datafiles are in the directory ../input

# Read competition data files:

train = pd.read_csv("../input/train.csv")

test  = pd.read_csv("../input/test.csv")



# Write to the log:

print("Training set has {0[0]} rows and {0[1]} columns".format(train.shape))

print("Test set has {0[0]} rows and {0[1]} columns".format(test.shape))

# Any files you write to the current directory get shown as outputs



y = train[['label']]             #训练集只要label这一列

X = train.drop('label',1)        #训练集去掉label这一列

header = X.columns               #只要训练集的标题一行，如pixel0，pixel1.....pixel783





 # 避免过拟合，采用交叉验证，验证集占训练集30%，固定随机种子（random_state)

X_train, X_test, y_train, y_test = train_test_split(X, y,    #随机划分训练集和测试集，

    test_size=0.30, random_state=2)           



norm = Normalizer().fit(X_train)           #规范化是将不同变化范围的值映射到相同的固定范围，常见的是[0,1]，此时也称为归一化

X_train = norm.transform(X_train)          #换后每个样本的各维特征的平方和为1

X_test = norm.transform(X_test)

test = norm.transform(test)







X_train = pd.DataFrame(X_train, columns = header)

X_test = pd.DataFrame(X_test, columns = header)

test = pd.DataFrame(test, columns = header)

y_train = pd.DataFrame(y_train)

y_train = y_train.as_matrix()





def pca(X_tr, X_ts, teste, n):

    pca = PCA(n)

    pca.fit(X_tr)

    X_tr_pca = pca.transform(X_tr)

    X_ts_pca = pca.transform(X_test)

    teste = pca.transform(teste)    

    return X_tr_pca, X_ts_pca, teste





X_tr_pca, X_ts_pca, test = pca(X_train, X_test, test, 35)



X_tr_pca = pd.DataFrame(X_tr_pca) 

X_ts_pca = pd.DataFrame(X_ts_pca)

test = pd.DataFrame(test)



submission = pd.DataFrame()



model = KNeighborsClassifier(n_neighbors = 4, weights='distance') #近邻权weights,K个近邻样本的权重,越靠近样本权值越大

model.fit(X_tr_pca, y_train)

score = model.score(X_ts_pca, y_test)

print ('KNN ', score)

pred = model.predict(test)







submission = pd.DataFrame({

    "ImageId": np.arange(1, pred.shape[0] + 1),

    "Label": pred

})                                                          #value为Series的字典结构



submission.to_csv("submission_knn.csv", index=False)