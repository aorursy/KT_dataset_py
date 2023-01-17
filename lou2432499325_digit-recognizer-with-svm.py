import pandas as pd

import numpy as np

from matplotlib import pyplot as plt

%matplotlib inline

train = pd.read_csv('../input/digit-recognizer/train.csv')

test = pd.read_csv('../input/digit-recognizer/test.csv')
train.head(2)
test.head(2)
#手写图像查看

feature = train.iloc[:,1:]

plt.figure(figsize=(16,9))

for i in range(25):

    plt.subplot(5,5,i+1)

    plt.imshow(feature.values[i].reshape((28,28)),cmap=plt.cm.binary)
feature = train.iloc[:,1:]

#数据量太大，先进行降维处理

from sklearn.decomposition import PCA

pca = PCA(n_components=100)

x_train_pac = pca.fit_transform(feature)

plt.plot(pca.explained_variance_ratio_)

pca.explained_variance_ratio_.sum()
y_train_pac = pca.transform(test)
# #特征值归一化处理，转化为二进制

x_train = np.rint(x_train_pac / 255)

x_label = train['label'].astype(int)

y_train = np.rint(y_train_pac/ 255)
from sklearn.svm import SVC

import warnings

warnings.filterwarnings('ignore')

from sklearn.model_selection import StratifiedKFold,cross_val_score

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=4)

svc= SVC()

np.mean(cross_val_score(svc,x_train,x_label,cv=skf))
svc.fit(x_train,x_label)

Label = svc.predict(y_train)

pd.DataFrame({'ImageId':range(1,len(y_train)+1),'Label':Label}).to_csv('output.csv', index=False)