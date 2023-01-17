import pandas as pd #数据分析包

import numpy as np #科学计算包

from pandas import Series,DataFrame

import os

original_data = pd.read_csv("../input/train.csv")

total_data = original_data.copy()

total_data.columns #查看数据集的字段
#看看数据集是什么样的

total_data.head(10)
total_data.info()
total_data_plot = total_data.copy()

total_data_plot = total_data_plot.drop(['Cabin'],axis = 1)

total_data_plot = total_data_plot.dropna()

total_data_plot['Sex'] = pd.factorize(total_data_plot.Sex)[0]

total_data_plot['Embarked'] = pd.factorize(total_data_plot.Embarked)[0]
%matplotlib inline

import matplotlib.pyplot as plt

import seaborn as sns #要注意的是一旦导入了seaborn，matplotlib的默认作图风格就会被覆盖成seaborn的格式

plt.figure(figsize=(10, 8))

sns.set() #使用默认配色  

sns.pairplot(total_data_plot.iloc[:,1:],hue="Survived")   #hue 选择分类列  

plt.show()  
plt.figure(figsize=(8, 6))

_ = sns.heatmap(total_data_plot.iloc[:,1:].corr(), annot=False)#很方便的利用.corr方法输出相似性矩阵
# Drop column 'Cabin'

total_data = total_data.drop(['Cabin'],axis = 1)

# Drop Embarked missing valuesb

total_data = total_data[total_data['Embarked'].notnull()]

#对于Age的处理，我们首先要对Name进行处理

import re

#利用正则表达式，提取称谓

total_data['Title'] = total_data['Name'].map(lambda x: re.compile(",(.*?)\.").findall(x)[0])

#这些称号有法语还有英语，需要依据当时的文化环境将其归类

total_data.loc[total_data[total_data.Title==' Jonkheer'].index,['Title']] = ' Master'

total_data.loc[total_data[total_data.Title.isin([' Ms',' Mlle'])].index,['Title']] = ' Miss'

total_data.loc[total_data[total_data.Title == ' Mme'].index,['Title']] = ' Mrs'

total_data.loc[total_data[total_data.Title.isin([' Capt', ' Don', ' Major', ' Col', ' Sir'])].index,['Title']] = ' Sir'

total_data.loc[total_data[total_data.Title.isin([' Dona', ' Lady', ' the Countess'])].index,['Title']] = ' Lady'
#现在，我们需要统计每一个Title所对应的年龄的均值

Title_list = list(total_data['Title'].drop_duplicates())

Title_age = {}

for i in Title_list:

    Title_age[i] = total_data[total_data.Title == i].Age.mean()

print (Title_age)
#更新缺失值

for i in Title_list:

    total_data.loc[total_data[(total_data['Age'].isnull()) & (total_data['Title'] == i)].index,['Age']] = Title_age[i]
dummies_Pclass = pd.get_dummies(total_data['Pclass'], prefix= 'Pclass')

dummies_Sex = pd.get_dummies(total_data['Sex'], prefix= 'Sex')

dummies_Embarked = pd.get_dummies(total_data['Embarked'], prefix= 'Embarked')

dummies_Title = pd.get_dummies(total_data['Title'], prefix= 'Title')

total_data = pd.concat([total_data, dummies_Pclass, dummies_Sex, dummies_Embarked, dummies_Title], axis=1)

#删除虚拟变量对应的原始变量

total_data.drop(['Pclass', 'Name', 'Sex', 'Ticket', 'Embarked'], axis=1, inplace=True)
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

total_data['Age_scaled'] = scaler.fit_transform(total_data['Age'].values.reshape(-1,1))

total_data['Fare_scaled'] = scaler.fit_transform(total_data['Fare'].values.reshape(-1,1))

total_data.head(10)
# 我们把需要的feature字段取出来，转成numpy格式，使用scikit-learn中的LogisticRegression建模

from sklearn.linear_model import LogisticRegression

from sklearn.cross_validation import train_test_split

from sklearn.ensemble import RandomForestClassifier

from sklearn.grid_search import GridSearchCV

from sklearn.learning_curve import learning_curve





X = total_data.filter(regex='Survived|Age_.*|SibSp|Parch|Fare_.*|Embarked_.*|Sex_.*|Pclass_.*').as_matrix()[:, 1:]

y = total_data.filter(regex='Survived|Age_.*|SibSp|Parch|Fare_.*|Embarked_.*|Sex_.*|Pclass_.*').as_matrix()[:, 0]





# 将数据集拆分成训练集与验证集

train_split,crossvalid_split = train_test_split(total_data, test_size=0.33, random_state=42)

train_df = train_split.filter(regex='Survived|Age_.*|SibSp|Parch|Fare_.*|Embarked_.*|Sex_.*|Pclass_.*').copy()

valid_df =crossvalid_split.filter(regex='Survived|Age_.*|SibSp|Parch|Fare_.*|Embarked_.*|Sex_.*|Pclass_.*').copy()



# y即Survival结果

y_train = train_df.as_matrix()[:, 0]

y_valid = valid_df.as_matrix()[:, 0]



# X即特征属性值

X_train = train_df.as_matrix()[:, 1:]

X_valid = valid_df.as_matrix()[:, 1:]





estimator = LogisticRegression()

parameter = {'penalty':('l1','l2'),'C':tuple(np.linspace(.05, 5., 10))}

clf = GridSearchCV(estimator,parameter,cv = 5)

clf.fit(X_train, y_train)
# 用sklearn的learning_curve得到training_score和cv_score，使用matplotlib画出learning curve

def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None, n_jobs=1, 

                        train_sizes=np.linspace(.05, 1., 20), verbose=0, plot=True):

    """

    画出data在某模型上的learning curve.

    参数解释

    ----------

    estimator : 你用的分类器。

    title : 表格的标题。

    X : 输入的feature，numpy类型

    y : 输入的target vector

    ylim : tuple格式的(ymin, ymax), 设定图像中纵坐标的最低点和最高点

    cv : 做cross-validation的时候，数据分成的份数，其中一份作为cv集，其余n-1份作为training(默认为3份)

    n_jobs : 并行的的任务数(默认1)

    """

    train_sizes, train_scores, test_scores = learning_curve(

        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes, verbose=verbose)

    

    train_scores_mean = np.mean(train_scores, axis=1)

    train_scores_std = np.std(train_scores, axis=1)

    test_scores_mean = np.mean(test_scores, axis=1)

    test_scores_std = np.std(test_scores, axis=1)

    

    if plot:

        plt.figure()

        plt.title(title)

        if ylim is not None:

            plt.ylim(*ylim)

        plt.xlabel("Traning Samples")

        plt.ylabel("Score")

        plt.gca().invert_yaxis()

        plt.grid()

        plt.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, 

                         alpha=0.1, color="b")

        plt.fill_between(train_sizes, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, 

                         alpha=0.1, color="r")

        plt.plot(train_sizes, train_scores_mean, 'o-', color="b", label="Score on test data")

        plt.plot(train_sizes, test_scores_mean, 'o-', color="r", label="Score on validation data")

    

        plt.legend(loc="best")

        plt.draw()

        plt.gca().invert_yaxis()

        plt.grid()

        plt.show()

    

    midpoint = ((train_scores_mean[-1] + train_scores_std[-1]) + (test_scores_mean[-1] - test_scores_std[-1])) / 2

    diff = (train_scores_mean[-1] + train_scores_std[-1]) - (test_scores_mean[-1] - test_scores_std[-1])

    return midpoint, diff
plot_learning_curve(clf, "Learning Curve", X, y)
#绘制混淆矩阵

from sklearn.metrics import confusion_matrix

import itertools



def plot_confusion_matrix(cm, classes,

                          normalize=False,

                          title='Confusion matrix',

                          cmap=plt.cm.Blues):

    """

    This function prints and plots the confusion matrix.

    Normalization can be applied by setting `normalize=True`.

    """

    plt.imshow(cm, interpolation='nearest', cmap=cmap)

    plt.title(title)

    plt.colorbar()

    tick_marks = np.arange(len(classes))

    plt.xticks(tick_marks, classes, rotation=45)

    plt.yticks(tick_marks, classes)



    if normalize:

        cm1 = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        print("Normalized confusion matrix")

        print(cm1)

    else:

        print('Confusion matrix, without normalization')

        print(cm)

    

    accuracy = (cnf_matrix[1][1]+cnf_matrix[0][0])/cnf_matrix.sum()

    alive_precision = cnf_matrix[1][1]/cnf_matrix.sum(axis=0)[1]

    alive_recall = cnf_matrix[1][1]/cnf_matrix.sum(axis = 1)[1]

    alive_f1 = alive_precision * alive_recall * 2 / (alive_precision + alive_recall)

    dead_precision = cnf_matrix[0][0]/cnf_matrix.sum(axis=0)[0]

    dead_recall = cnf_matrix[0][0]/cnf_matrix.sum(axis=1)[0]

    dead_f1 = dead_precision * dead_recall * 2 / (dead_precision + dead_recall)

    print ('\n')

    print ('accuracy  = ',accuracy)

    print ('alive_precision=',alive_precision,'    alive_recall=',alive_recall,'    alive_f1=',alive_f1)

    print ('dead_precision=',dead_precision,'    dead_recall=',dead_recall,'    dead_f1=',dead_f1)

    thresh = cm.max() / 2.

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):

        plt.text(j, i, cm[i, j],

                 horizontalalignment="center",

                 color="white" if cm[i, j] > thresh else "black")

    

    plt.tight_layout()

    plt.ylabel('True label')

    plt.xlabel('Predicted label')



#使用验证集测试结果

cnf_matrix = confusion_matrix(y_valid, clf.predict(X_valid))

plt.figure()

plot_confusion_matrix(cnf_matrix, classes=[0,1], normalize=True,title='confusion matrix')

plt.show()
#载入Tensorflow

import tensorflow as tf

import tflearn
network_train = total_data.filter(regex='Survived|Age_.*|SibSp|Parch|Fare_.*|Embarked_.*|Sex_.*|Pclass_.*').copy()

dummies_Survived = pd.get_dummies(network_train['Survived'], prefix= 'Survived')

network_train = pd.concat([network_train, dummies_Survived], axis=1)

#删除虚拟变量对应的原始变量

network_train.drop(['Survived'], axis=1, inplace=True)
# X即特征属性值

X_network = network_train.as_matrix()[:, :-2]

# y即Survival结果

y_network = network_train.as_matrix()[:, -2:]
# 创建网络

def create_sample_network():

    net = tflearn.input_data(shape=[None, 12])

    net = tflearn.fully_connected(net, 256,activation='relu')

    net = tflearn.dropout(net,keep_prob=0.8)

    net = tflearn.fully_connected(net, 128,activation='relu')

    net = tflearn.dropout(net,keep_prob=0.5)

    net = tflearn.fully_connected(net, 64,activation='relu')

    net = tflearn.dropout(net,keep_prob=0.5)

    net = tflearn.fully_connected(net, 32,activation='relu')

    net = tflearn.fully_connected(net, 16,activation='relu')

    net = tflearn.fully_connected(net, 10,activation='relu')

    net = tflearn.fully_connected(net, 2, activation='softmax')

    net = tflearn.regression(net, optimizer='adam', learning_rate=0.001,  

                         loss='categorical_crossentropy', name='target') 

    return net
# 模型类型

tf.reset_default_graph()

net = create_sample_network()

model = tflearn.DNN(net, checkpoint_path='.\model_titanic',max_checkpoints=1,

                    tensorboard_verbose=2,tensorboard_dir='tensorboard_output') 

# 这里增加了读取存档的模式。如果已经有保存了的模型，我们当然就读取它然后继续训练！

if os.path.isfile('model_save'):

    model.load('model_save')

model.fit(X_network, y_network, n_epoch=5000, validation_set=0.1, shuffle=True,

          batch_size = 800,snapshot_epoch=False, show_metric=True,run_id='network_titanic')

# 这里是保存已经运算好了的模型

model.save('model_save')
# 读取模型

tf.reset_default_graph()

model_new = tflearn.DNN(create_sample_network())

model_new.load(model_file = 'model_save')
predict_result = model_new.predict(X_valid)

trans_pred = []

for i in predict_result:

    if i[0]>i[1]:

        trans = 0.

    else:

        trans = 1.

    trans_pred.append(trans)

y_pred = np.array(trans_pred)
cnf_matrix = confusion_matrix(y_valid, y_pred)

plt.figure()

plot_confusion_matrix(cnf_matrix, classes=[0,1], normalize=True,title='confusion matrix')

plt.show()
original_test_data = pd.read_csv("../input/test.csv")

test_data = original_test_data.copy()
# Drop column 'Cabin'

test_data = test_data.drop(['Cabin'],axis = 1)

# Drop Embarked missing value

test_data = test_data[test_data['Embarked'].notnull()]

test_data['Title'] = test_data['Name'].map(lambda x: re.compile(",(.*?)\.").findall(x)[0])

# 这些称号有法语还有英语，需要依据当时的文化环境将其归类

test_data.loc[test_data[test_data.Title==' Jonkheer'].index,['Title']] = ' Master'

test_data.loc[test_data[test_data.Title.isin([' Ms',' Mlle'])].index,['Title']] = ' Miss'

test_data.loc[test_data[test_data.Title == ' Mme'].index,['Title']] = ' Mrs'

test_data.loc[test_data[test_data.Title.isin([' Capt', ' Don', ' Major', ' Col', ' Sir'])].index,['Title']] = ' Sir'

test_data.loc[test_data[test_data.Title.isin([' Dona', ' Lady', ' the Countess'])].index,['Title']] = ' Lady'

# 更新缺失值

for i in Title_list:

    test_data.loc[test_data[(test_data['Age'].isnull()) & (test_data['Title'] == i)].index,['Age']] = Title_age[i]

# 查看缺失值情况

test_data.info()
test_data.loc[test_data[test_data.Fare.isnull()].index,['Fare']] = total_data[total_data.Pclass_3 == 1].Fare.mean()
dummies_Pclass_test = pd.get_dummies(test_data['Pclass'], prefix= 'Pclass')

dummies_Sex_test = pd.get_dummies(test_data['Sex'], prefix= 'Sex')

dummies_Embarked_test = pd.get_dummies(test_data['Embarked'], prefix= 'Embarked')

dummies_Title_test = pd.get_dummies(test_data['Title'], prefix= 'Title')

test_data = pd.concat([test_data, dummies_Pclass_test, dummies_Sex_test, dummies_Embarked_test, dummies_Title_test], axis=1)

#删除虚拟变量对应的原始变量

test_data.drop(['Pclass', 'Name', 'Sex', 'Ticket', 'Embarked'], axis=1, inplace=True)
test_data['Age_scaled'] = scaler.transform(test_data['Age'].reshape(-1,1))

test_data['Fare_scaled'] = scaler.transform(test_data['Fare'].reshape(-1,1))
X_test = test_data.filter(regex='Survived|Age_.*|SibSp|Parch|Fare_.*|Embarked_.*|Sex_.*|Pclass_.*').as_matrix()[:,:]
LR_predict = clf.predict(X_test)

Network_test_result = model_new.predict(X_test)

trans_test_pred = []

for i in Network_test_result:

    if i[0]>i[1]:

        trans_test = 0.

    else:

        trans_test = 1.

    trans_test_pred.append(trans_test)

Network_predict = np.array(trans_test_pred)
LR_result = pd.DataFrame({'PassengerId':test_data['PassengerId'].as_matrix(), 'Survived':LR_predict.astype(np.int32)})

DNN_result = pd.DataFrame({'PassengerId':test_data['PassengerId'].as_matrix(), 'Survived':Network_predict.astype(np.int32)})

LR_result.to_csv("logistic_regression_predictions.csv", index=False)

DNN_result.to_csv("DNN_predictions.csv", index=False)