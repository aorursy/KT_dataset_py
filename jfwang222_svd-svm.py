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
#         print(os.path.join(dirname, filename))
        pass
# Any results you write to the current directory are saved as output.
import scipy.io as scio
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from sklearn import svm
import pickle
from sklearn.metrics import confusion_matrix
from sklearn.metrics import recall_score
from sklearn.model_selection import train_test_split
import os
%matplotlib inline
# 将路径下的所有文件形成一个list
def getFileList(filename):
    fileList = []
    for dirname, _, filenames in os.walk(filename):   #遍历当前所有文件
        for filename in filenames:
            fileList.append(os.path.join(dirname, filename))
    return fileList
# 将路径下的n个文件夹形成n个list,返回文件名list 以及所有文件的二维list
def getFileLists(filePath):
    # 获取所有文件路径
    dirList = os.listdir(filePath)
    allFile = []
    for dirName in dirList:
        allFile.append(getFileList(filePath + "/" + dirName))
    return dirList,allFile;
def files_split(dirList,allFile,my_test_size):
    EmojiDic = {}
    srcList = []
    typeList = []
    for i in range(len(dirList)):
        EmojiDic[i] = dirList[i]
        srcList = srcList + allFile[i]
        typeList = typeList + [i]*len(allFile[i])
    src_train,src_valida,type_train,type_valida=train_test_split(srcList,typeList,test_size = my_test_size,random_state = 5)
    return EmojiDic,src_train,src_valida,type_train,type_valida
# 训练图像路径
train_path = "/kaggle/input/afewdata/AFEW/Train" 
# 测试图像路径
test_path = "/kaggle/input/afewdata/AFEW/Val"
# 图像大小
img_size = 48
dirList,allFile = getFileLists(train_path)
EmojiDic,train_src,valida_src,train_label,validata_label = files_split(dirList,allFile,0.5)
# 读取所有图像
train_imgs = []
for filename in train_src:
    img = Image.open(filename).convert('L').resize((img_size,img_size), Image.ANTIALIAS)
    train_imgs.append(np.array(img).flatten())
train_imgs = np.array(train_imgs)
train_imgs.shape
fetureV = np.dot(train_imgs.T,train_imgs) #右奇异矩阵
evals,evecs =np.linalg.eig(fetureV) #求特征值，特征向量
sorted_indices = np.argsort(-evals) #排序，排序结果为下标，从小到大
topk_evecs = evecs[:,sorted_indices[:40]]
train_data = train_imgs.dot(topk_evecs)
train_data.shape
svm_classifier = svm.SVC(C=1.0, kernel='rbf', decision_function_shape='ovr', gamma=0.01)
svm_classifier.fit(train_data, train_label)
dirList,allFile = getFileLists(test_path)
EmojiDic,src_train,src_valida,type_train,type_valida = files_split(dirList,allFile,0.4)
validaimgs = []
size_train = 48
for filename in src_valida:
    img = Image.open(os.path.join(dirname, filename)).convert('L').resize((size_train,size_train), Image.ANTIALIAS)
    validaimgs.append(np.array(img).flatten())
validaimgs = np.array(validaimgs)
validaimgs.shape
validafetureV = np.dot(validaimgs.T,validaimgs) #右奇异矩阵
validaevals,validaevecs =np.linalg.eig(validafetureV) #求特征值，特征向量
sorted_validaindices = np.argsort(-validaevals) #排序，排序结果为下标，从小到大
validatopk_evecs = evecs[:,sorted_validaindices[:40]]
validatopk_evecs.shape
validata_data = validaimgs.dot(validatopk_evecs)
print("测试集:", svm_classifier.score(validata_data, type_valida))
pred_type = svm_classifier.predict(validata_data)
confusion_matrix(type_valida, pred_type)