# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import numpy as np

import pandas as pd

from time import time

import matplotlib.pyplot as plt

from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import train_test_split

from sklearn import svm

from sklearn.decomposition import PCA

from sklearn.preprocessing import MinMaxScaler



train = pd.read_csv('/kaggle/input/digit-recognizer/train.csv')

test = pd.read_csv('/kaggle/input/digit-recognizer/test.csv')
train.info()
test.info()
train.isnull().any().describe()
test.isnull().any().describe()
print(train.shape)

print(test.shape)
X = train.iloc[:,1:]

y = train.iloc[:,0]
plt.figure(figsize = (10,5))



for num in range(0,10):

    plt.subplot(2,5,num+1)

    grid_data = X.iloc[num].values.reshape(28,28)

    plt.imshow(grid_data, interpolation = "none", cmap = "Greys")
X = MinMaxScaler().fit_transform(X)

test = MinMaxScaler().fit_transform(test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1, random_state = 14)
all_scores = []

n_components = np.linspace(0.7,0.9,num=50, endpoint=False)



#当主成分为n时,计算模型预测的准确率

def get_accuracy_score(n, X_train, X_test, y_train, y_test):

    t0 = time()

    pca = PCA(n_components = n)

    pca.fit(X_train)

    X_train_pca = pca.transform(X_train)

    X_test_pca = pca.transform(X_test)

    

    # 使用支持向量机分类器

    clf = svm.SVC()

    clf.fit(X_train_pca, y_train)

    

    # 计算准确度

    accuracy = clf.score(X_test_pca, y_test)

    t1 = time()

    print('n_components:{:.2f} , accuracy:{:.4f} , time elaps:{:.2f}s'.format(n, accuracy, t1-t0))

    return accuracy 



for n in n_components:

    score = get_accuracy_score(n,X_train, X_test, y_train, y_test)

    all_scores.append(score)  
# 画出主成分和准确度的关系图，主成分n_components的临界值为0.80时，精确度最高

plt.plot(n_components, all_scores, '-o')

plt.xlabel('n_components')

plt.ylabel('accuracy')

plt.show()
# 找出识别有误的数据并打印

pca = PCA(n_components = 0.80)

pca.fit(X_train)

X_train_pca = pca.transform(X_train)

X_test_pca = pca.transform(X_test)



clf = svm.SVC()

clf.fit(X_train_pca, y_train)

y_pred = clf.predict(X_test_pca)



errors = (y_pred != y_test)

y_pred_errors = y_pred[errors]

y_test_errors = y_test[errors].values

X_test_errors = X_test[errors]



print(y_pred_errors[:5])

print(y_test_errors[:5])

print(X_test_errors[:5])
# 数据可视化,查看预测有误的数字

n = 0

nrows = 2

ncols = 5



fig, ax = plt.subplots(nrows,ncols,figsize=(10,6))



for row in range(nrows):

    for col in range(ncols):

        ax[row,col].imshow((X_test_errors[n]).reshape((28,28)), cmap = "Greys")

        ax[row,col].set_title("Predict:{}\nTrue: {}".format(y_pred_errors[n],y_test_errors[n]))

        n += 1
pca = PCA(n_components=1)

pca.fit(X)



# 打印主成分个数

print(pca.n_components_)



# 对训练集和测试集进行主成分转换

X = pca.transform(X)

test = pca.transform(test)
# 使用支持向量机预测,使用网格搜索进行调参

clf_svc = GridSearchCV(estimator=svm.SVC(C=100.0, kernel='rbf', gamma=0.03), param_grid={ 'C': [1, 2, 4, 5], 'kernel': [ 'linear', 'rbf', 'sigmoid' ] }, cv=5, verbose=2 ) 



# 训练算法

clf_svc.fit(X, y)



# 显示使模型准确率最高的参数

print(clf_svc.best_params_)



preds = clf_svc.predict(test)

image_id = pd.Series(range(1,len(preds)+1))

result_2 = pd.DataFrame({'ImageID': image_id,'Label':preds})



result_2.to_csv('result_svc.csv',index = False)

print('Done')