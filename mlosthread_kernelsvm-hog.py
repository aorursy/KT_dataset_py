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
import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn import svm

from sklearn.utils.multiclass import unique_labels

import numpy as np

from skimage import feature as ft





if __name__ == "__main__":



    '''

    数据预处理环节

    '''

    #读入数据

    data_train = pd.read_csv('../input/Kannada-MNIST/train.csv')

    data_test = pd.read_csv('../input/Kannada-MNIST/test.csv')



    data_train = pd.DataFrame(data_train)

    data_test = pd.DataFrame(data_test)



    #取出 除去标签列以外的dataframe格式的特征矩阵

    train_data_dataframe = data_train.iloc[:,1:]



    #取出标签列矩阵

    label_data_series = data_train['label']



    #特征值和标签列 转化为numpy 中的数组格式

    train_data = train_data_dataframe.values

    label_data = label_data_series.values



    #划分训练集合 为训练集和交叉验证集合，其中20%的数据为交叉验证

    trainData,valData,trainLabel,valLabel = train_test_split(train_data,label_data,test_size=0.2,random_state=33)



    testData = data_test.values[:, 1:]



    #图像数据的标准化和归一化 本来就是灰度图像 除以255

    trainData = trainData/255

    valData = valData/255

    testData = testData / 255



    '''

    特征工程 HOG边缘（形状/梯度）检测环节   

    '''



    list_picture_train = []

    train_features = []

    list_picture_val = []

    val_features = []

    list_picture_test = []

    test_features = []

    for i in range(0, len(trainData)):

        list_picture_train.append(trainData[i].reshape(28, 28))

    for i in range(0,len(valData)):

        list_picture_val.append(valData[i].reshape(28, 28))

    for i in range(0,len(testData)):

        list_picture_test.append(testData[i].reshape(28, 28))

    for i in range(0,len(trainData)):

        train_features.append(ft.hog(list_picture_train[i],  # 输入图像

                               orientations=8,  # 直方图bin的个数9

                               pixels_per_cell=(3, 3),  # 每一个细胞单元含8*8个pixel(8,8)

                               cells_per_block=(1, 1),  # 每一个块含有2*2个cell(2,2)

                               block_norm='L2-Hys',  # 块为单位的特征向量归一化L2

                               transform_sqrt=True,  # gamma校正 增强图像对比度

                               feature_vector=True,  # 铺平得到的最终的特征向量

                               visualize=False))  # 待定

    for i in range(0,len(valData)):

        val_features.append(ft.hog(list_picture_val[i],

                                orientations=8,

                                pixels_per_cell=(3, 3),

                                cells_per_block=(1, 1),

                                block_norm='L2-Hys',

                                transform_sqrt=True,

                                feature_vector=True,

                                visualize=False))

    for i in range(0,len(testData)):

        test_features.append(ft.hog(list_picture_test[i],

                                orientations=8,

                                pixels_per_cell=(3, 3),

                                cells_per_block=(1, 1),

                                block_norm='L2-Hys',

                                transform_sqrt=True,

                                feature_vector=True,

                                visualize=False))

    trainData = np.array(train_features)

    valData = np.array(val_features)

    testData = np.array(test_features)

    

    '''

    LinearSVC线性核函数的支持向量机SVM

    '''

    SVM = svm.LinearSVC(C=1,max_iter=5000)



    SVM.fit(trainData,trainLabel)



    out_train_SVM = SVM.predict(trainData)

    train_SVM = np.array(out_train_SVM).reshape(-1, 1)



    out_val_SVM = SVM.predict(valData)

    val_SVM = np.array(out_val_SVM).reshape(-1, 1)





    pre_label = SVM.predict(testData)

    

    submission = pd.DataFrame(pre_label, columns=['label'])



    submission['id'] = data_test['id']



    submission.to_csv('submission.csv', index=False, columns=['id', 'label'])








