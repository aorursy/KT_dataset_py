# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import pandas as pd

import jieba



filepath = '/kaggle/input/chinese-text-multi-classification/nCoV_100k_train.labled.csv'

file_data = pd.read_csv(filepath)



# 取前10000条数据

data = file_data.head(10000)



# 选择中文内容和情感倾向

data = data[['微博中文内容', '情感倾向']]



# 处理缺失值

#data.isnull().sum()



data = data.dropna()



import re



def clean_zh_text(text):

    # keep English, digital and Chinese

    comp = re.compile('[^A-Z^a-z^0-9^\u4e00-\u9fa5]')

    return comp.sub('', text)

 



data['微博中文内容'] = data.微博中文内容.apply(clean_zh_text)



# 分词

def chinese_word_cut(mytext):

    return " ".join(jieba.cut(mytext))



data['cut_comment'] = data.微博中文内容.apply(chinese_word_cut)



# lentrain = int((10000-30)*0.7)

# lentest = int((10000-30)*0.3)

# x_train = data.head(lentrain)['cut_comment']

# y_train = data.head(lentrain)['情感倾向']

# x_test = data.tail(lentest)['cut_comment']

# y_test = data.tail(lentest)['情感倾向']

x = data['cut_comment']

y = data['情感倾向']



# 引入停用词

#从文件导入停用词表

stpwrdpath = "../input/stop-words/stop_words.txt"

stpwrd_dic = open(stpwrdpath, 'rb')

stpwrd_content = stpwrd_dic.read()

#将停用词表转换为list  

stpwrdlst = stpwrd_content.splitlines()

stpwrd_dic.close()



x_list = x.tolist()



from sklearn.feature_extraction.text import TfidfVectorizer



tfidf_vec = TfidfVectorizer(token_pattern=r"(?u)\b\w+\b", max_df=0.6, stop_words=stpwrdlst)

x_tiv = tfidf_vec.fit_transform(x_list).toarray()



# 准备数据

from sklearn.preprocessing import StandardScaler

scale = StandardScaler()

scale_fit = scale.fit(x_tiv)

#x = scale_fit.transform(x)

lentrain = int((10000-30)*0.7)

lentest = int((10000-30)*0.3)



x_train_tiv = scale_fit.transform(x_tiv[:lentrain])

y_train = y[:lentrain]

x_test_tiv = scale_fit.transform(x_tiv[(-1)*lentest-1:-1])

y_test = y[(-1)*lentest-1:-1]

print(x_train_tiv.shape)

print(y_train.shape)

print(x_test_tiv.shape)

print(y_test.shape)
from sklearn import svm



print(x_train_tiv.shape, 'and ', y_train.shape)

clf = svm.SVC(C=0.8, kernel='rbf', gamma=20, decision_function_shape='ovr')

tiv_model = clf.fit(x_train_tiv, y_train)
print(clf.score(x_train_tiv, y_train))  # 精度



# y_hat

y_hat_tiv = clf.predict(x_test_tiv)

#print(y_hat_cv)

#print(y_test)
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score



print('Precision_score: ', precision_score(y_hat_tiv, y_test, average='weighted'))

print('Recall_score: ', recall_score(y_hat_tiv, y_test, average='weighted'))

print('F1_score: ', f1_score(y_hat_tiv, y_test, average='weighted'))

print('Accuracy_score: ', accuracy_score(y_hat_tiv, y_test))
# roc_curve:真正率（True Positive Rate , TPR）或灵敏度（sensitivity）

# 横坐标：假正率（False Positive Rate , FPR）

from numpy import interp

from sklearn.preprocessing import label_binarize

import matplotlib.pyplot as plt

from itertools import cycle

from sklearn.metrics import roc_curve, auc



nb_classes = 3

# Binarize the output

Y_valid = label_binarize(y_test, classes=[i for i in range(nb_classes)])

Y_pred = label_binarize(y_hat_tiv, classes=[i for i in range(nb_classes)])



    

# Compute ROC curve and ROC area for each class

fpr = dict()

tpr = dict()

roc_auc = dict()

for i in range(nb_classes):

    fpr[i], tpr[i], _ = roc_curve(Y_valid[:, i], Y_pred[:, i])

    roc_auc[i] = auc(fpr[i], tpr[i])



# Compute micro-average ROC curve and ROC area

fpr["micro"], tpr["micro"], _ = roc_curve(Y_valid.ravel(), Y_pred.ravel())

roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])





# Compute macro-average ROC curve and ROC area



# First aggregate all false positive rates

all_fpr = np.unique(np.concatenate([fpr[i] for i in range(nb_classes)]))



# Then interpolate all ROC curves at this points

mean_tpr = np.zeros_like(all_fpr)

for i in range(nb_classes):

    mean_tpr += interp(all_fpr, fpr[i], tpr[i])



# Finally average it and compute AUC

mean_tpr /= nb_classes



fpr["macro"] = all_fpr

tpr["macro"] = mean_tpr

roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])







# Plot all ROC curves

lw = 2

plt.figure()

plt.plot(fpr["micro"], tpr["micro"],

         label='micro-average ROC curve (area = {0:0.2f})'

               ''.format(roc_auc["micro"]),

         color='deeppink', linestyle=':', linewidth=4)



plt.plot(fpr["macro"], tpr["macro"],

         label='macro-average ROC curve (area = {0:0.2f})'

               ''.format(roc_auc["macro"]),

         color='navy', linestyle=':', linewidth=4)



colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])

for i, color in zip(range(nb_classes), colors):

    plt.plot(fpr[i], tpr[i], color=color, lw=lw,

             label='ROC curve of class {0} (area = {1:0.2f})'

             ''.format(i, roc_auc[i]))



plt.plot([0, 1], [0, 1], 'k--', lw=lw)

plt.xlim([0.0, 1.0])

plt.ylim([0.0, 1.05])

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.title('Some extension of Receiver operating characteristic to multi-class')

plt.legend(loc="lower right")

plt.show()