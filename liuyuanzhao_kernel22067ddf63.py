

#encoding: utf-8

import numpy as np

import pandas as pd

from time import time

import matplotlib.pyplot as plt

import warnings

from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import train_test_split

from sklearn import svm

from sklearn.decomposition import PCA

from sklearn.preprocessing import MinMaxScaler



warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn", lineno=193)

train = pd.read_csv('../input/Kannada-MNIST/train.csv')

test = pd.read_csv('../input/Kannada-MNIST/test.csv')



#train.info()

#test.info()

#train.isnull().any().describe()



X = train.iloc[:,1:]

y = train.iloc[:,0]



#抽取train数据进行绘图

'''

plt.figure(figsize = (10,5))



for num in range(0,10):

    plt.subplot(2,5,num+1)

    #将长度为784的向量数据转化为28*28的矩阵

    grid_data = X.iloc[num].values.reshape(28,28)

    #print(grid_data)

    #显示图片，颜色为黑白

    plt.imshow(grid_data, interpolation = "none", cmap = "Greys")

#plt.show()

'''



# 特征预处理,将特征的值域规范化

X = MinMaxScaler().fit_transform(X)

test = MinMaxScaler().fit_transform(test)



# 分开训练集和测试集

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1, random_state = 14)





# 使用主成分分析,降低维度

all_scores = []

# 生成n_components的取值列表

n_components = np.linspace(0.5,0.9,num=20, endpoint=False)

'''

def get_accuracy_score(n, X_train, X_test, y_train, y_test):

#当主成分为n时,计算模型预测的准确率 

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

'''

'''

# 画出主成分和准确度的关系图，主成分n_components的临界值为0.56时，精确度最高

plt.plot(n_components, all_scores, '-o')

plt.xlabel('n_components')

plt.ylabel('accuracy')

plt.show()

'''

#找出识别有误的数据

pca = PCA(n_components = 0.78)

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

'''

# 查看数据

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



plt.show()

'''



# n_components为0.56时, 模型的准确率最高

# 对训练集和测试集进行PCA降低维度处理, 主成分个数为39

pca = PCA(n_components=0.56)

pca.fit(X)

# 打印主成分个数

print(pca.n_components_)



# 修改

df_test = pd.DataFrame(test,index = None)

# 对训练集和测试集进行主成分转换

X = pca.transform(X)

test_no_id = df_test.iloc[:,1:]

test = pca.transform(test_no_id)



# 使用支持向量机预测,使用网格搜索进行调参



clf_svc = GridSearchCV(estimator=svm.SVC(), param_grid={ 'C': [1, 2, 4, 5], 'kernel': [ 'linear', 'rbf', 'sigmoid' ] }, cv=5, verbose=2 ) 

# 训练算法

clf_svc.fit(X, y)

# 显示使模型准确率最高的参数

print(clf_svc.best_params_)



# 预测

preds = clf_svc.predict(test)

image_id = pd.Series(range(0,len(preds)))

result_2 = pd.DataFrame({'id': image_id,'label':preds})

# 保存为CSV文件

result_2.to_csv('submission.csv',index = False)

print('Done')
