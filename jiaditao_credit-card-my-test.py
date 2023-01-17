import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

from matplotlib.font_manager import FontProperties

import seaborn as sns 

%matplotlib inline

data = pd.read_csv("../input/creditcardfraud/creditcard.csv")

data.head()
data.info()

data.describe()

Class_1 = data['Class'][data['Class']==1].count()

Class_all = data['Class'].count()

print('信用卡欺诈率为',str(round (100*Class_1/Class_all ,2) )+"%")
import seaborn as sns

sns.distplot(data['Amount'])

plt.show()
from sklearn.preprocessing import StandardScaler

s1 = StandardScaler()

data['AmountSta'] = s1.fit_transform(data['Amount'].values.reshape(-1,1))

data.drop(['Amount'],axis=1,inplace=True)
data['Hours'] = data['Time'].apply(lambda x:int(x/3600))

del data['Time']

s2 = StandardScaler()

data['HoursSta'] = s1.fit_transform(data['Hours'].values.reshape(-1,1))

data.drop(['HoursSta'],axis=1,inplace=True)

data.head(10)

#查看一下相关性

data_corr=data.corr()

data_corr['Class'].sort_values(ascending=False)
#随机欠采样

number_records_fraud = len(data[data.Class == 1]) # class=1的样本函数

fraud_indices = np.array(data[data.Class == 1].index) # 样本等于1的索引值



normal_indices = data[data.Class == 0].index # 样本等于0的索引值



random_normal_indices = np.random.choice(normal_indices,number_records_fraud,replace = False)

random_normal_indices = np.array(random_normal_indices)



under_sample_indices = np.concatenate([fraud_indices,random_normal_indices]) # Appending the 2 indices



under_sample_data = data.iloc[under_sample_indices,:] # Under sample dataset

'''

#包导不进来...我恨

#SMOTE过采样

from imblearn.over_sampling import SMOTE

oversample = SMOTE(random_state=42)

x,y = oversample.fit_sample(data[list(data.columns).remove('Class')],data['Class'])

print('通过SMOTE方法平衡正负样本后')

n_sample = y.shape[0]

n_pos_sample = y[y == 0].shape[0]

n_neg_sample = y[y == 1].shape[0]

print('样本个数：{}; 正样本占{:.2%}; 负样本占{:.2%}'.format(n_sample,

                                                   n_pos_sample / n_sample,

                                                   n_neg_sample / n_sample))

'''                                                   

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import train_test_split

from sklearn.model_selection import KFold

from sklearn.metrics import confusion_matrix,precision_recall_curve,auc,roc_auc_score,roc_curve,recall_score,classification_report 

def choose_c(data_x,data_y):

    #k折检验，分为五折,每次分的时候不洗牌

    fold = KFold(n_splits=5,shuffle=False)

    print(fold)

    c_range = [0.01,0.1,1,10,100]

    list_recall = []

    for c in c_range:

        print("===================================C=",c,"=====================")

        list_recall_perc = []

        for train_index,test_index in fold.split(data_x):

            ''' 

            这里记一笔，把series当dataframe做了，憨憨落泪...

            #aaa = data_x.iloc[train_index,:]

            bbb =data_y.iloc[train_index]

            print(bbb)

            '''

            lr = LogisticRegression(C=c,penalty='l1')

            #模型训练

            lr.fit(data_x.iloc[train_index,:],data_y.iloc[train_index])

            #模型预测

            y_pred =lr.predict(data_x.iloc[test_index,:])

            #计算召回率

            recall_acc = recall_score(data_y.iloc[test_index].values,y_pred)

            print('recall=',recall_acc)

            list_recall_perc.append(recall_acc)

        list_recall.append(np.mean(list_recall_perc))

    print(list_recall)

        

list1 = ['V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10', 'V11',

       'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20', 'V21',

       'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28', 

       'AmountSta', 'Hours']

under_sample_data_x =under_sample_data[list1]

under_sample_data_y =under_sample_data['Class']

#将数据分为2/8两份

x_train_undersample,x_test_undersample,y_train_undersample,y_test_undersample=train_test_split(under_sample_data_x,

                                                                                               under_sample_data_y,

                                                                                               test_size=.2,

                                                                                               random_state=0)

choose_c(x_train_undersample,y_train_undersample)



from sklearn import metrics

import scipy.optimize as op

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import KFold,cross_val_score

from sklearn.metrics import (precision_recall_curve,

                             auc,roc_auc_score,

                             roc_curve,recall_score,

                             classification_report)

lrnodel = LogisticRegression(C=10,penalty='l2')

lrnodel.fit(x_train_undersample,y_train_undersample)

#查看模型

print(lrnodel)

#查看混淆矩阵（评价精度）

ypred_lr=lrnodel.predict(x_test_undersample)

print(metrics.confusion_matrix(y_test_undersample,ypred_lr))



#查看分类报告

print(metrics.classification_report(y_test_undersample,ypred_lr))



#查看预测精度与决策覆盖面

print('Accurancy:%f'%(metrics.accuracy_score(y_test_undersample,ypred_lr)))

print('Area under the curve:%f'%(metrics.roc_auc_score(y_test_undersample,ypred_lr)))

import itertools

def plot_confusion_matrix(cm, classes,

                          title='Confusion matrix',

                          cmap=plt.cm.Blues):

    plt.imshow(cm, interpolation='nearest', cmap=cmap)

    plt.title(title)

    plt.colorbar()

    tick_marks = np.arange(len(classes))

    plt.xticks(tick_marks, classes, rotation=0)

    plt.yticks(tick_marks, classes)



    thresh = cm.max() / 2.

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):

        plt.text(j, i, cm[i, j],

                 horizontalalignment="center",

                 color="white" if cm[i, j] > thresh else "black")



    plt.tight_layout()

    plt.ylabel('True label')

    plt.xlabel('Predicted label')

    

cnf_matrix=metrics.confusion_matrix(y_test_undersample,ypred_lr)

np.set_printoptions(precision=2)

print("Recall metric in the testing dataset: ", cnf_matrix[1,1]/(cnf_matrix[1,0]+cnf_matrix[1,1]))



# Plot non-normalized confusion matrix

class_names = [0,1]

plt.figure()

plot_confusion_matrix(cnf_matrix

                      , classes=class_names

                      , title='Confusion matrix')

plt.show()
fpr, tpr, thresholds = roc_curve(y_test_undersample,ypred_lr)

plt.title('Receiver Operating Characteristic')

plt.plot(fpr, tpr, 'b',label='AUC = %0.5f'%(metrics.roc_auc_score(y_test_undersample,ypred_lr)))

plt.legend(loc='lower right')

plt.plot([0,1],[0,1],'r--')

plt.xlim([-0.1,1.0])

plt.ylim([-0.1,1.01])

plt.ylabel('True Positive Rate')

plt.xlabel('False Positive Rate')

plt.show()