import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

#import 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, learning_curve, KFold
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, StandardScaler
import random
from sklearn.svm import SVC
import sklearn.metrics as sk
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
#读取数据集
df = pd.read_csv('/kaggle/input/bank-marketing/bank-additional-full.csv', sep = ';')
df.shape
#展示
df.head()
#信息
df.info()
#删除无关组
df1=df.drop(columns=['day_of_week','month','contact','poutcome','pdays'],axis=1)
df1
#将布尔变量替换成0和1
df1.y.replace(('yes', 'no'), (1, 0), inplace=True)
df1.default.replace(('yes', 'no'), (1, 0), inplace=True)
df1.housing.replace(('yes', 'no'), (1, 0), inplace=True)
df1.loan.replace(('yes', 'no'), (1, 0), inplace=True)
df1
#构建副本
df2 = pd.get_dummies(df1)
df2.head()
#对变量进行描述统计
df3=df2.drop(columns=['job_unknown','marital_divorced','education_unknown'],axis=1)
df3.describe().T
#绘图
plt.figure(figsize=(14,8))
df3.corr()['y'].sort_values(ascending = False).plot(kind='bar')
#创建二分类目标
df_target=df3[['y']].values
df_features=df3.drop(columns=['y'],axis=1).values
x1_train, x1_test, y1_train, y1_test = train_test_split(df_features, df_target, test_size = 0.3, random_state = 0)
sc = StandardScaler()
x1_train = sc.fit_transform(x1_train)
x1_test = sc.transform(x1_test)
#创建模型
print('MODEL SVM',end='\n')
lsvclassifier = SVC(kernel='linear')
lsvclassifier.fit(x1_train, y1_train)

#K折交叉验证（5）
accuracies = cross_val_score(estimator = lsvclassifier, X = x1_train, y = y1_train, cv = 5)
mean_svm_linear=accuracies.mean()
std_svm_linear=accuracies.std()

#评估
print('After 5 fold cross validation:')
print('Mean of Accuracies: ',mean_svm_linear*100,end='\n')
print('Standard deviation of Accuracies',std_svm_linear*100,end='\n')

#测试集上预测
y_predl = lsvclassifier.predict(x1_test)

#输出混淆矩阵（对预测结果输出包括 precision  recall  f1-score  support（被分入到某一类的支持项））
print('Test Output:')
print('Confusion Matrix:')
print(sk.confusion_matrix(y1_test,y_predl))
print('Classification Report:')
print(sk.classification_report(y1_test,y_predl))
print('Accuracy: ',sk.accuracy_score(y1_test, y_predl, normalize=True, sample_weight=None))
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import RFE

#创建模型
print('logistic Model',end='\n')
regressor = LogisticRegression()#LR
rfe = RFE(regressor, 20)#RFE反复构建LR模型，并递归选择最好的特征
rfe = rfe.fit(x1_train, y1_train)


#K折交叉验证（5）
accuracies = cross_val_score(estimator = rfe, X = x1_train, y = y1_train, cv = 5)
mean_lr=accuracies.mean()
std_lr=accuracies.std()

#评估
print('After 5 fold cross validation:')
print('Mean of Accuracies: ',mean_lr*100,end='\n')
print('Standard deviation of Accuracies',std_lr*100,end='\n')

#测试集上预测
y_predl = rfe.predict(x1_test)

#输出混淆矩阵（对预测结果输出包括 precision  recall  f1-score  support（被分入到某一类的支持项））
print('Test Output:')
print('Confusion Matrix:')
print(sk.confusion_matrix(y1_test,y_predl))
print('Classification Report:')
print(sk.classification_report(y1_test,y_predl))
print('Accuracy: ',sk.accuracy_score(y1_test, y_predl, normalize=True, sample_weight=None))
from sklearn.naive_bayes import GaussianNB
#创建模型
print('native_bayes Model',end='\n')
clf = GaussianNB()#BY
clf.fit(x1_train, y1_train)

#K折交叉验证（5）
accuracies = cross_val_score(estimator = clf, X = x1_train, y = y1_train, cv = 5)
mean_clf=accuracies.mean()
std_clf=accuracies.std()

#评估
print('After 5 fold cross validation:')
print('Mean of Accuracies: ',mean_clf*100,end='\n')
print('Standard deviation of Accuracies',std_clf*100,end='\n')

#测试集上预测
y_predl = clf.predict(x1_test)

#输出混淆矩阵（对预测结果输出包括 precision  recall  f1-score  support（被分入到某一类的支持项）））
print('Test Output:')
print('Confusion Matrix:')
print(sk.confusion_matrix(y1_test,y_predl))
print('Classification Report:')
print(sk.classification_report(y1_test,y_predl))
print('Accuracy: ',sk.accuracy_score(y1_test, y_predl, normalize=True, sample_weight=None))