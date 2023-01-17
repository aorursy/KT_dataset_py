import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline
data = pd.read_csv('../input/train.csv')
data.head()
data.info()
data = data.set_index(['PassengerId'])#将旅客ID设置为索引

data, labels = data.drop(['Survived','Name', 'Ticket', 'Cabin'], axis=1), data['Survived']#分离数据集为数据和标签
data.describe()# 显示数值型数据的摘要信息
data.head()
data[['Age', 'SibSp', 'Parch', 'Fare']].hist(figsize=(20, 12), bins=20, align='mid')#显示数据的直方图
data[['Age', 'SibSp', 'Parch', 'Fare']].corr()#查看各个特征的相关关系
cor = data[['Age', 'SibSp', 'Parch', 'Fare']].corr()#绘制相关性矩阵

plt.matshow(cor)
# 分割数据集

from sklearn.model_selection import train_test_split

# x_train, x_test, y_train, y_test = train_test_split(data, labels)

x_train = data

y_train = labels
from sklearn.base import BaseEstimator, TransformerMixin



#列选择器

class DataFrameSelector(BaseEstimator, TransformerMixin):

    def __init__(self, attributes_name):

        self.attributes_name = attributes_name

    def fit(self, X, y=None):

        return self

    def transform(self, X, y=None):

        return X[self.attributes_name]

#字符串类型使用最多的类型填写    

class StringImputer(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):

        self.most_frequent_ = pd.Series([X[c].value_counts().index[0] for c in X],

                                       index=X.columns)

        return self

    def transform(self, X, y=None):

        return X.fillna(self.most_frequent_)
# 创建数据处理管道

from sklearn.compose import ColumnTransformer

from sklearn.pipeline import Pipeline

from sklearn.preprocessing import StandardScaler

from sklearn.impute import SimpleImputer

from sklearn.preprocessing import OneHotEncoder

from sklearn.pipeline import FeatureUnion

# 对类别型数据进行one-hot编码

cat_pipeline = Pipeline([

    ('selector', DataFrameSelector(['Sex', 'Embarked', 'Pclass'])),#选择类别数据

    ('impute', StringImputer()),#使用最多的类别填充

    ('binary', OneHotEncoder(sparse=False))#One-Hot编码

])



# 对数值型数据进行处理

num_pipeline = Pipeline([

    ('selector', DataFrameSelector(['Age', 'SibSp', 'Parch', 'Fare'])),#选择数值型数据

    ('impute', SimpleImputer(strategy='median')),#以中位数填充空值

    ('std',  StandardScaler())#对数值进行标准化

])



# # # 选取其它数据

# other_pipeline = Pipeline([

#     ('selector', DataFrameSelector(['Pclass', 'SibSp', 'Parch']))

# ])



# 使用FeatureUnion联合各个Pipeline获取数据

full_pipeline = FeatureUnion(transformer_list=[

    ('cat_pipeline', cat_pipeline),

    ('num_pipeline', num_pipeline),

#     ('other_pipeline', other_pipeline)

])



x_train = full_pipeline.fit_transform(x_train)
from sklearn.linear_model import SGDClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import cross_val_predict

from sklearn.metrics import precision_score, recall_score, precision_recall_curve

from sklearn.metrics import roc_curve, roc_auc_score

from sklearn.metrics import f1_score

from sklearn.metrics import accuracy_score

def plot_recall_predict_curve(precisions, recalls, threshold, xlim=[-5,5]):

    #画出预测和召回曲线

    plt.plot(threshold, precisions[:-1], 'b--', label='Precision')

    plt.plot(threshold, recalls[:-1], 'g-', label='Recall')

    plt.xlabel('Threshold')

    plt.legend(loc='center right')

    plt.ylim([0, 1])

    plt.xlim(xlim)

    

def plot_roc_curve(fpr, tpr, label=None):

    #画出roc曲线

    plt.plot(fpr, tpr, linewidth=2, label=label)

    plt.plot([0, 1], [0, 1], 'k--')

    plt.axis([0, 1, 0, 1])

    
from scipy.stats import reciprocal

from sklearn.model_selection import RandomizedSearchCV

sgd_clf = SGDClassifier(max_iter=500, tol=1e-3, loss='log',

                        random_state=42)

param_sgd = {

    'alpha': reciprocal(0.0001, 0.1),

    'epsilon': reciprocal(0.001, 0.1),

    'penalty': ['l1', 'l2']

}

sgd_grid_search_cv = RandomizedSearchCV(sgd_clf, param_distributions=param_sgd, verbose=True,

                                       n_jobs=-1, cv=3, random_state=42)

sgd_grid_search_cv.fit(x_train, y_train)
sgd_clf = sgd_grid_search_cv.best_estimator_

sgd_pred = cross_val_predict(sgd_clf, x_train, y_train, cv=7)

sgd_acc = accuracy_score(y_train, sgd_pred)#准确率
sgd_pre = precision_score(y_train, sgd_pred)#预测
sgd_recall =recall_score(y_train, sgd_pred)#召回
sgd_f1 = f1_score(y_train, sgd_pred)#f1分数
sgd_pred = cross_val_predict(sgd_clf, x_train, y_train, cv=7, method='decision_function')
precisions, recalls, threshold = precision_recall_curve(y_train, sgd_pred)

plot_recall_predict_curve(precisions, recalls, threshold)
fpr, tpr, threshold = roc_curve(y_train, sgd_pred)

plot_roc_curve(fpr, tpr)

sgd_auc = roc_auc_score(y_train, sgd_pred)
sgd_scores = [sgd_acc, sgd_pre, sgd_recall, sgd_f1, sgd_auc]
from sklearn.model_selection import GridSearchCV

rf_clf = RandomForestClassifier(random_state=42)

param_grid = [

    {'n_estimators': [i for i in range(100, 500)]},

    {'max_depth': [i for i in range(2, 10)]}

]

grid_cv = GridSearchCV(rf_clf, param_grid, cv=3, n_jobs=-1, verbose=True)#查找最好的森林

grid_cv.fit(x_train, y_train)
grid_cv.best_params_
rf_pred = cross_val_predict(grid_cv.best_estimator_, x_train, y_train, cv=3)

rf_acc = accuracy_score(y_train, rf_pred)#准确率
rf_pre = precision_score(y_train, rf_pred)#预测
rf_recall = recall_score(y_train, rf_pred)#召回
rf_f1 = f1_score(y_train, rf_pred)
rf_pred = cross_val_predict(grid_cv.best_estimator_, x_train, y_train, 

                            cv=3, method='predict_proba')

rf_pred = rf_pred[:, 1]
precisions, recalls, threshold = precision_recall_curve(y_train, rf_pred)

plot_recall_predict_curve(precisions, recalls, threshold, [0,1])
fpr, tpr, threshold = roc_curve(y_train, rf_pred)

plot_roc_curve(fpr, tpr)

rf_auc = roc_auc_score(y_train, rf_pred)
rf_scores = [rf_acc, rf_pre, rf_recall, rf_f1, rf_auc]
from sklearn.model_selection import learning_curve

train_size, train_scores, test_scores = learning_curve(grid_cv.best_estimator_, x_train, y_train, 

                                                       n_jobs=-1, verbose=True, cv=5, random_state=42)
train_scores_mean = np.mean(train_scores, axis=1)

train_scores_std = np.std(train_scores, axis=1)

test_scores_mean = np.mean(test_scores, axis=1)

test_scores_std = np.std(test_scores, axis=1)

plt.grid()



plt.fill_between(train_size, train_scores_mean - train_scores_std,

                train_scores_mean + train_scores_std, alpha=0.1,

                color='r')

plt.fill_between(train_size, test_scores_mean - test_scores_std,

                test_scores_mean + test_scores_std, alpha=0.1,

                color='b')

plt.plot(train_size, train_scores_mean, 'r-', label='train')

plt.plot(train_size, test_scores_mean, 'b--',label='val')

plt.legend(loc='best')
from sklearn.ensemble import GradientBoostingClassifier

param_grid = {

    'n_estimators': [100, 200, 300, 400, 500],

    'max_depth': np.arange(2, 10)

}

grid_gbrt_cv = GridSearchCV(GradientBoostingClassifier(random_state=42), param_grid=param_grid, 

                            cv=3, verbose=True, n_jobs=-1)

grid_gbrt_cv.fit(x_train, y_train)
grid_gbrt_cv.best_score_
gbrt_pred = cross_val_predict(grid_gbrt_cv.best_estimator_, x_train, y_train, 

                              cv=5)

gbrt_acc = accuracy_score(y_train, gbrt_pred)
gbrt_pre = precision_score(y_train, gbrt_pred)
gbrt_recall = recall_score(y_train, gbrt_pred)
gbrt_f1 = f1_score(y_train, gbrt_pred)
fpr, tpr, threshold = roc_curve(y_train, gbrt_pred)

plot_roc_curve(fpr, tpr)

gbrt_auc = roc_auc_score(y_train, gbrt_pred)
gbrt_scores = [gbrt_acc, gbrt_pre, gbrt_recall, gbrt_f1, gbrt_auc]
from sklearn.svm import SVC

from sklearn.ensemble import AdaBoostClassifier

from sklearn.model_selection import RandomizedSearchCV

from scipy.stats import reciprocal, uniform



svm_param_grd = {

    'gamma': reciprocal(0.001, 0.1),

    'C': uniform(1, 10),

    'kernel': ['rbf', 'poly']

}

svm_rnd_search_cv = RandomizedSearchCV(SVC(probability=True, ), param_distributions=svm_param_grd, 

                                       n_jobs=-1, n_iter=10, cv=3, random_state=42)

svm_rnd_search_cv.fit(x_train, y_train)
svm_pred = cross_val_predict(svm_rnd_search_cv.best_estimator_, x_train, y_train, 

                             cv=5)

svm_acc = accuracy_score(y_train, svm_pred)
svm_pre = precision_score( y_train, svm_pred)
svm_recall = recall_score(y_train, svm_pred)
svm_f1 = f1_score(y_train, svm_pred)
fpr, tpr, threshold = roc_curve(y_train, svm_pred)

plot_roc_curve(fpr, tpr)

svm_auc = roc_auc_score(y_train, svm_pred)
svm_scores = [svm_acc, svm_pre, svm_recall, svm_f1, svm_auc]
from sklearn.naive_bayes import GaussianNB

from sklearn.model_selection import GridSearchCV

bayes_param = {

    'var_smoothing': np.arange(0.0001, 0.9, 0.1)

}

bayes_grid_cv = GridSearchCV(GaussianNB(), bayes_param, cv=3,

                            verbose=True, n_jobs=-1)

bayes_grid_cv.fit(x_train, y_train)

bayes_pred = cross_val_predict(bayes_grid_cv.best_estimator_, x_train, y_train, cv=3, verbose=True)
bayes_acc = accuracy_score(y_train, bayes_pred)

bayes_pre = precision_score(y_train, bayes_pred)

bayes_recall = recall_score(y_train, bayes_pred)

bayes_f1 = f1_score(y_train, bayes_pred)

bayes_auc = roc_auc_score(y_train, bayes_pred)

bayes_scores = [bayes_acc, bayes_pre, bayes_recall, bayes_f1, bayes_auc]
fpr, tpr, threshold = roc_curve(y_train, bayes_pred)

plot_roc_curve(fpr, tpr)
from sklearn.neighbors import KNeighborsClassifier

param_knn = {

    'n_neighbors': np.arange(1, 10),

    'p': [1, 2],

    'weights': ['distance', 'uniform']

}

knn_grid_cv = GridSearchCV(KNeighborsClassifier(), param_grid=param_knn, n_jobs=-1, 

                           verbose=True, cv = 3)

knn_grid_cv.fit(x_train, y_train)
knn_clf = knn_grid_cv.best_estimator_

knn_pred = cross_val_predict(knn_clf, x_train, y_train, cv=3)
knn_acc = accuracy_score(y_train, knn_pred)

knn_pre = precision_score(y_train, knn_pred)

knn_recall = recall_score(y_train, knn_pred)

knn_f1 = f1_score(y_train, knn_pred)

knn_auc = roc_auc_score(y_train, knn_pred)



knn_scores = [knn_acc, knn_pre, knn_recall, knn_f1, knn_auc]
fpr, tpr, threshold = roc_curve(y_train, knn_pred)

plot_roc_curve(fpr, tpr)
from sklearn.ensemble import VotingClassifier

voting_clf = VotingClassifier(

    estimators=[('sgd_clf', sgd_clf), 

                ('rf_clf', grid_cv.best_estimator_), 

                ('gbrt_clf', grid_gbrt_cv.best_estimator_), 

                ('bayes_clf', bayes_grid_cv.best_estimator_),

                ('knn_clf', knn_clf),

                ('svm_clf', svm_rnd_search_cv.best_estimator_)],

    voting='soft',

)





voting_pred = cross_val_predict(voting_clf, x_train, y_train, cv=5)

voting_acc = accuracy_score(y_train, voting_pred)
voting_pre = precision_score(y_train, voting_pred)
voting_recall = recall_score(y_train, voting_pred,)
voting_f1 = f1_score(y_train, voting_pred)
fpr, tpr, threshold = roc_curve(y_train, voting_pred)

plot_roc_curve(fpr, tpr)

voting_auc = roc_auc_score(y_train, voting_pred)
voting_scores = [voting_acc, voting_pre, voting_recall, voting_f1, voting_auc]
fpr_sgd, tpr_sgd, threshold_sgd = roc_curve(y_train, sgd_pred)

fpr_rf, tpr_rf, threshold_rf = roc_curve(y_train, rf_pred)

fpr_gbrt, tpr_gbrt, threshold_gbrt = roc_curve(y_train, gbrt_pred)

fpr_svm, tpr_svm, threshold_svm = roc_curve(y_train, svm_pred)

fpr_voting, tpr_voting, threshold_voting = roc_curve(y_train, voting_pred)

fpr_bayes, tpr_bayes, threshold_bayes = roc_curve(y_train, bayes_pred)

fpr_knn, tpr_knn, threshold_knn = roc_curve(y_train, knn_pred)
plt.plot(fpr_sgd, tpr_sgd, 'r:', label=('SGD:'+str(sgd_auc)))

plt.plot(fpr_gbrt, tpr_gbrt, 'g--', label=('GBRT:'+str(gbrt_auc)))

plt.plot(fpr_svm, tpr_svm, 'p-', label=('SVM:'+str(svm_auc)))

plt.plot(fpr_voting, tpr_voting, 'y-', label=('VOTING:'+str(voting_auc)))

plt.plot(fpr_bayes, tpr_bayes, color='purple', marker='*', label=('bayes:'+str(bayes_auc)))

plt.plot(fpr_knn, tpr_knn, color='pink', marker='o', label=('knn:'+str(knn_auc)))

plot_roc_curve(fpr_rf, tpr_rf, label=('RandomForest:'+str(rf_auc)))

plt.legend(loc='lower right')
xticks = ['acc', 'pre', 'recall', 'f1', 'auc']

bar_width = 0.10

n = np.arange(5)

i=0

plt.figure(figsize=(24, 12))

for  score, color, label in zip([sgd_scores, rf_scores, gbrt_scores, svm_scores, voting_scores, bayes_scores, knn_scores],

                  ['r', 'b', 'y', 'black', 'g', 'purple', 'pink'],

                  ['sgd_scores', 'rf_scores', 'gbrt_scores', 'svm_scores', 'voting_scores', 'bayes_scores', 'knn_scores']):

    plt.bar(n+i*bar_width, score, bar_width , color=color, label=label, alpha=0.6)

    i += 1

plt.ylim([0,1])

plt.yticks(np.arange(0,1, 0.05))

plt.xticks(n+2*bar_width,xticks)

plt.legend(loc='best')
acc = [sgd_acc, rf_acc, gbrt_acc, svm_acc, voting_acc, bayes_acc, knn_acc]

pre = [sgd_pre, rf_pre, gbrt_pre, svm_pre, voting_pre, bayes_pre, knn_pre]

recall = [sgd_recall, rf_recall, gbrt_recall, svm_recall, voting_recall, bayes_recall, knn_recall]

f1 = [sgd_f1, rf_f1, gbrt_f1, svm_f1, voting_f1, bayes_f1, knn_f1]

auc = [sgd_auc, rf_auc, gbrt_auc, svm_auc, voting_auc, bayes_auc, knn_auc]

xticks = ['sgd', 'rf', 'gbrt', 'svm', 'voting', 'bayes', 'knn']

bar_width = 0.15

n = np.arange(7)

i=0

plt.figure(figsize=(24, 12))

for  score, color, label in zip([acc, pre, recall, f1, auc],

                  ['r', 'b', 'y', 'black', 'g'],

                  ['acc', 'pre', 'recall', 'f1', 'auc']):

    plt.bar(n+i*bar_width, score, bar_width , color=color, label=label, alpha=0.6)

    plt.axhline(y=np.mean(score), color=color, ls='--', alpha=0.6)

    i += 1

plt.yticks(np.arange(0,1, 0.05))

plt.xticks(n+2*bar_width,xticks)

plt.legend(loc='best')
test = pd.read_csv('../input/test.csv')

index = np.array(test[['PassengerId']])[:,0]

test = test.set_index('PassengerId')

test = full_pipeline.transform(test)

index.shape
svm_rnd_search_cv.best_estimator_.fit(x_train, y_train)

pred = svm_rnd_search_cv.best_estimator_.predict(test)

pred_df = pd.DataFrame({'PassengerId':index,

                       'Survived':pred})