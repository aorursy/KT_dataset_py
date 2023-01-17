import numpy as np

import pandas as pd



import matplotlib.pyplot as plt  # 绘图

import seaborn as sns



%matplotlib inline
df_train = pd.read_csv('../input/train.csv')

df_test = pd.read_csv('../input/test.csv') # 读取时间有点长
df_train.info()
df_test.info()
# 由于数据量太大，选取前4000行作为训练样本

df_train = df_train[:3000]

# 将样本空间划分成训练集和测试集

Y = df_train['label']

X = df_train.drop('label',axis=1)

from sklearn.model_selection import train_test_split

X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.25,random_state=0)
# 观察训练集和测试集信息

X_train.info()
X_test.info()
# 将部分数据可视化

sns.set()

fig, ax = plt.subplots(3,5,figsize=(10,7),subplot_kw={'xticks':(),'yticks':()})

ax = ax.ravel()

for i in range(15):

    pixels = X_train.iloc[i].values.reshape(-1,28)

    ax[i].imshow(pixels,cmap='viridis')

    ax[i].set_title('Digit-'+str(Y_train.iloc[i]))
# 查看数据分布是否均匀

sns.countplot(Y_train)
# from sklearn.preprocessing import StandardScaler

# ss = StandardScaler()

# ss.fit(X_train)

# X_train = ss.transform(X_train)

# X_test = ss.transform(X_test)

# 通过计算发现，标准化之后，模型预测的结果会变坏。因此不进行标准化。
from sklearn.decomposition import PCA

# 寻找最佳的主成分维度n_components

pca = PCA(random_state=0,whiten=True)

pca.fit(X_train)

# 求每个components对应的方差百分数

exp_var_cum = np.cumsum(pca.explained_variance_ratio_)

plt.step(range(exp_var_cum.size),exp_var_cum)
sns.set()

plt.step(range(exp_var_cum[15:101].size),exp_var_cum[15:101])
# 选择n_components=25。

pca =PCA(n_components=40,random_state=0,whiten=True)

pca.fit(X_train)

X_train_pca = pca.transform(X_train)

X_test_pca = pca.transform(X_test)
# 上面的结果还能不能优化？尝试改变n_neighbors

from sklearn.neighbors import KNeighborsClassifier

temp = []

for i in range(1,10):

    knn_pca = KNeighborsClassifier(n_neighbors=i,n_jobs=8)

    knn_pca.fit(X_train_pca,Y_train)

    train_score_pca = knn_pca.score(X_train_pca,Y_train)

    test_score_pca = knn_pca.score(X_test_pca,Y_test)

#     print(i,':','train_score_pca=',train_score_pca)

#     print(i,':','test_score_pca',test_score_pca)

    li = [i,train_score_pca,test_score_pca]

    temp.append(li)

temp
from sklearn.lda import LDA

temp = []

for i in range(1,40):

    lda = LDA(n_components=i)

    clf = lda.fit(X_train,Y_train)

    X_test_lda = clf.transform(X_test)

#     print(i,'=',clf.score(X_train,Y_train))

#     print(i,'=',clf.score(X_test,Y_test))

    li = [i,clf.score(X_train,Y_train),clf.score(X_test,Y_test)]

    temp.append(li)

temp
from sklearn.decomposition import NMF

nmf = NMF(n_components=30,random_state=0)

nmf.fit(X_train)

X_train_nmf = nmf.transform(X_train)

X_test_nmf = nmf.transform(X_test)



knn_nmf = KNeighborsClassifier(n_neighbors=4,n_jobs=8)

knn_nmf.fit(X_train_nmf,Y_train)

print('Train score:',knn_nmf.score(X_train_nmf,Y_train))

print('test score:',knn_nmf.score(X_test_nmf,Y_test))
from sklearn.manifold import TSNE

tsne = TSNE(n_components=2,random_state=0)

X_train_tsne = tsne.fit_transform(X_train,Y_train)



plt.scatter(X_train_tsne[:,0],X_train_tsne[:,1],c=Y_train.values,cmap='prism')
X_test_tsne = tsne.fit_transform(X_test,Y_test)

plt.scatter(X_test_tsne[:,0],X_test_tsne[:,1],c=Y_test.values,cmap='prism')