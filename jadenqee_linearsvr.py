# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
DATA_PATH = '/kaggle/input/bi-attrition-predict/'
train = pd.read_csv(f'{DATA_PATH}/train.csv')
test = pd.read_csv(f'{DATA_PATH}/test.csv')
len(train), len(test)
train.head()
test.head()
train.info()
#user_id:员工Id(str) Age:年龄(int) Attrition:是否离职(bool) BusinessTravel:出差情况(categorical) 
#DailyRate：？(int) Department: 部门(categorical) DistanceFromHome:居住地与工作单位距离(int) Education:教育时间(int)
#EducationField:教育背景(categorical) EmployeeCount:?(int) EmployeeNumber:员工号码(int) EnvironmentSatisfaction:环境满意度(int/categorical)
#Gender:性别(bool) HourlyRate:?(int) JobInvolvement:工作投入度(int) JobLevel:职位等级(categorical) JobRole:职位
#JobSatisfaction:工作满意度(int/categorical) MaritalStatus:婚姻状态(bool) MonthlyIncome:月收入(int) MonthlyRate:?(int)
#NumCompaniesWorked:任职过的公司数(int) Over18:是否成年(bool) OverTime:是否加班(bool) PercentSalaryHike:工资提高比率(int)
#PerformanceRating:绩效评估(int) RelationshipSatisfaction:人际关系满意度(int) StandardHours:标准工作时间(int)
#StockOptionLevel:股票占有等级(int) TotalWorkingYears:总计工作年数(int) TrainingTimeLastYear:去年培训时长(int)
#WorkLifeBalance:工作生活平衡情况(int) #YearsAtCompany:在公司工作年数(int) #YearsInCurrentRole:在这一职位时长(int)
#YearsSinceLastPromotion:距离上一次升职时间(int) YearsWithCurrManager:与同一上级工作时长(int)
#0.80
id_col = 'user_id'
target_col = 'Attrition'

digital_cols = ['Age', 'DailyRate', 'MonthlyIncome', 'NumCompaniesWorked', 'PercentSalaryHike',
                'TotalWorkingYears', 'YearsAtCompany', 'YearsInCurrentRole', 'YearsSinceLastPromotion', 'YearsWithCurrManager']
category_cols = ['BusinessTravel', 'Department',  'Education', 'EducationField',
                'EmployeeNumber', 'EnvironmentSatisfaction', 'Gender', 'JobInvolvement', 'JobLevel','DistanceFromHome',
                'JobRole', 'JobSatisfaction', 'MaritalStatus', 'Over18', 'OverTime',
                'RelationshipSatisfaction', 'StandardHours', 'StockOptionLevel', 'PerformanceRating', 'TrainingTimesLastYear','WorkLifeBalance' ]
# Credits to https://www.kaggle.com/a763337092/lr-baseline-for-bi-class#Data-process
# For categorical data
for col in category_cols:
    nunique_tr = train[col].nunique()
    nunique_te = test[col].nunique()
    na_tr = len(train.loc[train[col].isna()]) / len(train)
    na_te = len(test.loc[test[col].isna()]) / len(test)
    print(f'Col name:{col:30}\tunique cate num in train:{nunique_tr:5}\tunique cate num in train:{nunique_te:5}\tnull sample in train:{na_tr:.2f}\tnull sample in test:{na_te:.2f}')
#For numerical data

for col in digital_cols:
    
    min_tr = train[col].min()
    max_tr = train[col].max()
    mean_tr = train[col].mean()
    median_tr = train[col].median()
    std_tr = train[col].std()
    x = ['min','mean','median','std','max']
    y = [min_tr,mean_tr,median_tr,std_tr,max_tr]

    
    
    min_te = test[col].min()
    max_te = test[col].max()
    mean_te = test[col].mean()
    median_te = test[col].median()
    std_te = test[col].std()
    x = ['min','mean','median','std','max']
    y = [min_tr,mean_tr,median_tr,std_tr,max_tr]
    
    na_tr = len(train.loc[train[col].isna()]) / len(train)
    na_te = len(test.loc[test[col].isna()]) / len(test)
    print(f'\tIn train data:\tnan sample rate:{na_tr:.2f}\t')
    print(f'\tIn test data\tnan sample rate:{na_te:.2f}\t')
plt.bar(x, y)
plt.title(col)
plt.show
#age and attrition
plt.figure(figsize=(4,3))
print(train['Attrition'])
sns.barplot(x='Attrition', y='Age', data = train , palette = 'Set2')
figure, ax = plt.subplots(figsize=(10, 10))
data = pd.concat([train.drop(['user_id','Attrition','EmployeeNumber','EmployeeCount', 'Over18','StandardHours'],axis = 1), test]).corr() ** 2
#data = np.tril(data, k=-1)
data[data==0] = np.nan
sns.heatmap(np.sqrt(data), annot=False, cmap='viridis', ax=ax)
print(type(data))
train = pd.read_csv(f'{DATA_PATH}/train.csv')
target_col_dict = {'Yes': 1, 'No': 0}
train1 = train
train1['Attrition'] = train1['Attrition'].map(target_col_dict).values
train2 = train1
#train2.drop(['Attrition'])
data = train2.corrwith(train1['Attrition']).agg('square')
data = data.drop('Attrition')
figure, ax = plt.subplots(figsize=(10, 10))
data.agg('sqrt').plot.bar(ax=ax)
# del data
from sklearn.preprocessing import MinMaxScaler

sacalar = MinMaxScaler()
train_digital = sacalar.fit_transform(train[digital_cols])
test_digital = sacalar.transform(test[digital_cols])

# from sklearn.preprocessing import LabelEncoder, OneHotEncoder

# train_category, test_category = None, None
# drop_cols = ['EmployeeNumber', 'Over18', 'StandardHours','BusinessTravel']
# for col in [var for var in category_cols if var not in drop_cols]:
#     lbe, ohe = LabelEncoder(), OneHotEncoder()
    
#     lbe.fit(pd.concat([train[col], test[col]]).values.reshape(-1, 1))
#     train[col] = lbe.transform(train[col])
#     test[col] = lbe.transform(test[col])
    
#     ohe.fit(pd.concat([train[col], test[col]]).values.reshape(-1, 1))
#     oht_train = ohe.transform(train[col].values.reshape(-1, 1)).todense()
#     oht_test = ohe.transform(test[col].values.reshape(-1, 1)).todense()
    
#     if train_category is None:
#         train_category = oht_train
#         test_category = oht_test
#     else:
#         train_category = np.hstack((train_category, oht_train))
#         test_category = np.hstack((test_category, oht_test))
# print(train_category[0,:])
# lbe.fit(pd.concat([train['BusinessTravel'], test['BusinessTravel']]).values.reshape(-1, 1))
# BT_train = lbe.transform(train['BusinessTravel'])
# BT_test = lbe.transform(test['BusinessTravel'])
# train_category = np.insert(train_category, 0, values=BT_train, axis=1)
# test_category = np.insert(test_category, 0, values=BT_test, axis=1)
# print(train_category)


from sklearn.preprocessing import LabelEncoder, OneHotEncoder

train_category, test_category = None, None
drop_cols = ['EmployeeNumber', 'Over18', 'StandardHours']
for col in [var for var in category_cols if var not in drop_cols]:
    lbe, ohe = LabelEncoder(), OneHotEncoder()
    
    lbe.fit(pd.concat([train[col], test[col]]).values.reshape(-1, 1))
    train[col] = lbe.transform(train[col])
    test[col] = lbe.transform(test[col])
    
    ohe.fit(pd.concat([train[col], test[col]]).values.reshape(-1, 1))
    oht_train = ohe.transform(train[col].values.reshape(-1, 1)).todense()
    oht_test = ohe.transform(test[col].values.reshape(-1, 1)).todense()
    
    if train_category is None:
        train_category = oht_train
        test_category = oht_test
    else:
        train_category = np.hstack((train_category, oht_train))
        test_category = np.hstack((test_category, oht_test))
train_digital.shape, test_digital.shape, train_category.shape, test_category.shape
feature_names = ['Age', 'DailyRate', 'MonthlyIncome', 'NumCompaniesWorked', 'PercentSalaryHike',
                'TotalWorkingYears', 'YearsAtCompany', 'YearsInCurrentRole', 'YearsSinceLastPromotion', 'YearsWithCurrManager',
                'BusinessTravel', 'Department', 'DistanceFromHome', 'Education', 'EducationField',
                'EmployeeNumber', 'EnvironmentSatisfaction', 'Gender', 'JobInvolvement', 'JobLevel',
                'JobRole', 'JobSatisfaction', 'MaritalStatus', 'Over18', 'OverTime',
                'PerformanceRating', 'RelationshipSatisfaction', 'StandardHours', 'StockOptionLevel', 'TrainingTimesLastYear',
                'WorkLifeBalance']
train_features = np.hstack((train_digital, train_category))
test_features = np.hstack((test_digital, test_category))
train_features.shape, test_features.shape
# target_col_dict = {'Yes': 1, 'No': 0}
# train_labels = train[target_col].map(target_col_dict).values
train_labels = train[target_col]
train_labels.shape
# from imblearn.over_sampling import KMeansSMOTE
# sm = KMeansSMOTE(random_state=42 ,cluster_balance_threshold = 0.3, k_neighbors=2)
# X_res, y_res = sm.fit_resample(X_train, y_train)

from imblearn.over_sampling import SMOTE
sm = SMOTE(random_state=42)
X_res, y_res = sm.fit_resample(X_train, y_train)

# from imblearn.over_sampling import ADASYN
# ad = ADASYN(random_state  = 42)
# X_res, y_res = ad.fit_resample(X_train, y_train)

# from imblearn.combine import SMOTEENN
# sme = SMOTEENN(random_state = 42)
# X_res, y_res = sme.fit_resample(X_train, y_train)
from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(train_features, train_labels, test_size=0.20, random_state=42)
X_train.shape,X_val.shape,y_train.shape,y_val.shape
# from sklearn.decomposition import PCA
# #pca = PCA(n_components = 'mle')
# pca = PCA(n_components = 110)
# X_val_new = pca.fit_transform(X_val)
# train_features_new = pca.fit_transform(train_features)
# test_features_new = pca.fit_transform(test_features)
# X_train_new = pca.fit_transform(X_train)
# print('保留的特征数：')
# print(pca.n_components_)
# print('特征所占比重分别为：')
# print(pca.explained_variance_ratio_)
# print('特征所占比重之和：')
# print(sum(pca.explained_variance_ratio_))


### fit model for train data
import xgboost as xgb
#dataset
dtrain = xgb.DMatrix(X_train, label = y_train)

#parameters
num_round = 500
param = {'bst:max_depth':2, 'bst:eta':1, 'silent':1, 'objective':'binary:logistic' }
param['nthread'] = 2
param['eval_metric'] = 'auc'
plst = param.items()
np.any(np.isnan(y_train))
bst = xgb.train(plst, dtrain, num_round)
#save the model
bst.save_model('0001.model')
dtest = xgb.DMatrix(X_val)
ypred = bst.predict(dtest)
test_auc = metrics.roc_auc_score(y_val,ypred)#验证集上的auc值
print(test_auc)
from xgboost import XGBClassifier

bst2 = XGBClassifier(#learning_rate=0.08,
                      n_estimators=300,         # 树的个数--300棵树建立xgboost
                      #max_depth=3,               # 树的深度
                      #min_child_weight = 1,      # 叶子节点最小权重
                      #gamma=0.3,                  # 惩罚项中叶子结点个数前的参数
                      #subsample=0.8,             # 随机选择80%样本建立决策树
                      #colsample_btree=0.8,       # 随机选择80%特征建立决策树
#                       objective='binary:logistic', # 指定损失函数
#                       scale_pos_weight=1,        # 解决样本个数不平衡的问题
#                       random_state=27            # 随机数
                      )
bst2.fit(X_train,
          y_train,
           eval_set = [(X_val,y_val)],
           eval_metric = "auc",
          #early_stopping_rounds = 10,
          verbose = False)
#save the model
bst2.save_model('0002.model')

ypred2 = bst2.predict(X_val)
test_auc = metrics.roc_auc_score(y_val,ypred2)#验证集上的auc值
print(test_auc)
from sklearn.preprocessing import minmax_scale
feature_importance = bst.get_score(importance_type = 'total_cover')
feature_values = list(feature_importance.values())
feature_keys = list(feature_importance.keys())
original_importance_sort = np.sort(feature_values)[::-1]
new_dict = {v : k for k, v in feature_importance.items()} #key,value 互换位置
feature_label = []
for i in range(len(new_dict)): feature_label.append(new_dict[original_importance_sort[i]])
print(feature_label)
print(feature_keys)
feature_importance = np.sort(minmax_scale(feature_values,feature_range = (0,1)))[::-1]
print(feature_importance)



from xgboost import plot_importance
fig,ax = plt.subplots(figsize=(15,15))
plot_importance(bst,
                height=0.5,
                ax=ax,
                max_num_features=64)
feature_importance_per = feature_importance / sum(feature_importance)
print(feature_importance_per)

threshold = 0.95
#根据重要性阈值选择前N个特征
def TopiElements(feature_importance_per,threshold):
    score = 0
    for i in range(len(feature_importance_per)):
        if score < threshold:
            score += feature_importance_per[i]
        else:
            return i+1
def TopiFeaturesName(feature_label,feature_importance_per,threshold):
    return feature_label[0:TopiElements(feature_importance_per,threshold)-1]
print('共有',TopiElements(feature_importance_per,threshold),'个特征')
TopiFeatures = TopiFeaturesName(feature_label,feature_importance_per,threshold)
print(TopiFeatures)
#输入features为矩阵,返回值为矩阵
def GenerateNewFeatures(features,TopiFeatures):
    feature_columns = []
    for i in range(110): feature_columns.append('f{}'.format(i))
    train_Features = pd.DataFrame(features,columns = [feature_columns])
    train_Features.head()
    new_train_Features = train_Features[TopiFeatures]
    return new_train_Features.values
new_train_features = GenerateNewFeatures(train_features,TopiFeatures)
new_test_features = GenerateNewFeatures(test_features,TopiFeatures)
# from sklearn.feature_selection import SelectFromModel

# thresholds = sorted(bst2.feature_importances_)
# for thresh in thresholds:
#     # select features using threshold
#     selection = SelectFromModel(bst2, threshold=thresh, prefit=True)
#     select_X_train = selection.transform(X_train)
#     # train model
#     selection_model = XGBClassifier()
#     selection_model.fit(select_X_train, y_train)
#     # eval model
#     select_X_val = selection.transform(X_val)
#     ypred2 = selection_model.predict(select_X_val)
#     predictions = [round(value) for value in ypred2]
#     auc = metrics.roc_auc_score(y_val,ypred2)
#     print("Thresh=%.3f, n=%d, auc: %.2f%%" % (thresh, select_X_train.shape[1], auc))

threshold_res = 0.5
def EncodeResult(threshold, results):
    encodeResult = []
    for i in range(len(results)):
        encodeResult.append(1) if results[i] > threshold else encodeResult.append(0)
    return encodeResult
from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(new_train_features, train_labels, test_size=0.20, random_state=42)
#dataset
dtrain = xgb.DMatrix(X_train, label = y_train)

#parameters
num_round = 300
param = {'bst:max_depth':3, 'bst:eta':1, 'silent':1, 'objective':'binary:logistic' }
param['nthread'] = 2
plst = param.items()

bst = xgb.train(plst, dtrain, num_round)
#save the model
bst.save_model('0001.model')

dtest = xgb.DMatrix(X_val)
ypred = bst.predict(dtest)

test_auc = metrics.roc_auc_score(y_val,ypred)#验证集上的auc值
print(test_auc)
print(ypred)
encode_result = EncodeResult(threshold_res,ypred)
print(encode_result)
test_acc = metrics.accuracy_score(y_val,encode_result)
print(test_acc)
from sklearn.linear_model import LinearRegression

clf_after = LinearRegression()
clf_after.fit(X_train, y_train)
ypred = clf_after.predict(X_val)
ypred.shape
test_auc = metrics.roc_auc_score(y_val,ypred)#验证集上的auc值
print(test_auc)
X_train, X_val, y_train, y_val = train_test_split(train_features, train_labels, test_size=0.20, random_state=42)
clf = LinearRegression()
clf.fit(X_train, y_train)
ypred = clf.predict(X_val)
ypred.shape
test_auc = metrics.roc_auc_score(y_val,ypred)#验证集上的auc值
print(test_auc)
#X_train, X_val, y_train, y_val = train_test_split(train_features, train_labels, test_size=0.20, random_state=42)
from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor(random_state=0,max_features = 0.40)
rf.fit(X_train,y_train)
ypred = rf.predict(X_val)
test_auc = metrics.roc_auc_score(y_val,ypred)#验证集上的auc值
print('Random forest auc:',test_auc)
#print('acc:',metrics.accuracy_score(y_val,ypred))
#print(ypred)
X_train, X_val, y_train, y_val = train_test_split(new_train_features, train_labels, test_size=0.20, random_state=42)
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(random_state=0,max_features = 0.30)
rf.fit(X_train,y_train)
ypred = rf.predict(X_val)
test_auc = metrics.roc_auc_score(y_val,ypred)#验证集上的auc值
print('auc:',test_auc)
print('acc:',metrics.accuracy_score(y_val,ypred))
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier,RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import RandomizedSearchCV
from sklearn.base import BaseEstimator, clone
from sklearn.pipeline import make_pipeline
import xgboost as xgb
X_train, X_val, y_train, y_val = train_test_split(train_features, train_labels, test_size=0.20, random_state=42)
n_folds = 5
def acc_cv(model):
    kf = KFold(n_folds, shuffle=True, random_state=42).get_n_splits(X_train)
    acc= cross_val_score(model, X_train, y_train, scoring="accuracy", cv = kf)
    return(acc)
Adaboost = make_pipeline(RobustScaler(),
                         AdaBoostClassifier(base_estimator=None,
                                            n_estimators = 56,
                                            learning_rate= 0.18,
                                            algorithm='SAMME.R',
                                            random_state = 1)
                        )
GBoosting = make_pipeline(RobustScaler(), 
                          GradientBoostingClassifier(loss='deviance',
                                                     learning_rate = 0.05,
                                                     n_estimators = 56,
                                                     min_samples_split = 9,
                                                     min_samples_leaf = 2,
                                                     max_depth = 4,
                                                     random_state = 1,
                                                     max_features = 9)
                         )
SVC =  make_pipeline(RobustScaler(), 
                     SVC(decision_function_shape = 'ovr',
                         random_state = 1,
                         max_iter = 14888,
                         kernel = 'poly',
                         degree = 2,
                         coef0 = 0.49, 
                         C =  9.6)
                     )
RF = make_pipeline(RobustScaler(), 
                   RandomForestClassifier(criterion='gini', 
                                          n_estimators=364,
                                          max_depth = 11,                    
                                          min_samples_split=6,
                                          min_samples_leaf=1,
                                          max_features='auto',
                                          oob_score=True,
                                          random_state=1,
                                          )
                  )
xgbc = make_pipeline(RobustScaler(), 
                     xgb.XGBClassifier(n_estimators=121,
                                       reg_lambda = 0.9,
                                       reg_alpha = 0.5,
                                       max_depth = 9,
                                       learning_rate = 0.55,
                                       gamma = 0.5,
                                       colsample_bytree = 0.4,
                                       coldsample_bynode = 0.15,
                                       colsample_bylevel = 0.5)
                    )
score = acc_cv(Adaboost)
print("Adaboost score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))

score = acc_cv(GBoosting)
print("Gradient Boosting score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))

score = acc_cv(SVC)
print("SVC  score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))

score = acc_cv(RF)
print("Random Forest score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))

score = acc_cv(xgbc)
print("Xgboost score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
class AveragingModels(BaseEstimator):
    def __init__(self, models):
        self.models = models
        
    # we define clones of the original models to fit the data in
    def fit(self, X, y):
        self.models_ = [clone(x) for x in self.models]
        
        # Train cloned base models
        for model in self.models_:
            model.fit(X, y)

        return self
    
    #Now we do the predictions for cloned models and average them
    def predict(self, X):
        predictions = np.column_stack([
            model.predict(X) for model in self.models_
        ])
        return np.mean(predictions, axis=1)
%%capture
averaged_models = AveragingModels(models = (Adaboost,SVC, GBoosting, RF,xgbc))
averaged_models.fit(X_train, y_train)
#predict
train_pred = averaged_models.predict(X_train)
test_pred = averaged_models.predict(X_val)
test_pred.shape
print('ensemble_auc:', metrics.roc_auc_score(y_val,test_pred))
# train_pred = np.round(train_pred)
# test_pred = np.round(test_pred)

acc_averaged = np.round((train_pred==y_train).sum()/train_pred.shape[0],5)
print(f"Averaged models accuracy: {acc_averaged}")
#new features
from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(new_train_features, train_labels, test_size=0.20, random_state=42)
#old features
from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(train_features, train_labels, test_size=0.20, random_state=42)
from pandas import DataFrame
X_train = DataFrame(X_train, index = None)
y_train = DataFrame(y_train, index = None)
X_val = DataFrame(X_val,index = None)
y_val = DataFrame(y_val,index = None)
import lightgbm as lgb
train_data = lgb.Dataset(data = X_train, label = y_train)
test_data = lgb.Dataset(data = X_val, label = y_val)
# parameters
param = {'num_leaves':20, 'num_trees':300, 'objective':'binary'}
param['metric'] = ['binary_logloss']
# use test_data as validation dataset
num_round = 100
bst = lgb.train(param, train_data, num_round, valid_sets = test_data, early_stopping_rounds =10)
y_pred = bst.predict(X_val)
print('lightGBM_auc:', metrics.roc_auc_score(y_val,y_pred))
from tpot import TPOTRegressor
tpot = TPOTRegressor(generations=3, population_size=50, verbosity=2)
tpot.fit(X_train, y_train)
print(tpot.score(X_val, y_val))
tpot.export('tpot_titanic_pipeline.py')

y_pred = tpot.predict(X_val)
print('topt AUC:',metrics.roc_auc_score(y_val,y_pred))
# bernoli distribution
from sklearn.naive_bayes import BernoulliNB
clf = BernoulliNB(alpha = 2, fit_prior=False)
clf.fit(X_train,y_train)
y_pred = clf.predict(X_val)
print('BernoulliNB AUC:',metrics.roc_auc_score(y_val,y_pred))
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline


exported_pipeline = make_pipeline(
    StandardScaler(),
    GradientBoostingClassifier(learning_rate=0.1, max_depth=2, max_features=0.45, min_samples_leaf=6, min_samples_split=5, n_estimators=100, subsample=0.25)
)
exported_pipeline.fit(X_train, y_train)
y_pred = exported_pipeline.predict(X_val)
print('GBDT AUC:',metrics.roc_auc_score(y_val,y_pred))
from sklearn.svm import SVR
from sklearn import metrics
X_train, X_val, y_train, y_val = train_test_split(new_train_features, train_labels, test_size=0.20, random_state=42)
svc = SVR(kernel = 'linear')
svc.fit(X_train,y_train)
y_pred = svc.predict(X_val)
print('SVM_auc:',metrics.roc_auc_score(y_val,y_pred))

import numpy as np
from catboost import Pool, CatBoostRegressor

id_col = 'user_id'
target_col = 'Attrition'

digital_cols = ['Age', 'DailyRate', 'MonthlyIncome', 'NumCompaniesWorked', 'PercentSalaryHike',
                'TotalWorkingYears', 'YearsAtCompany', 'YearsInCurrentRole', 'YearsSinceLastPromotion', 'YearsWithCurrManager']
category_cols = ['BusinessTravel', 'Department',  'Education', 'EducationField',
                 'EnvironmentSatisfaction', 'Gender', 'JobInvolvement', 'JobLevel','DistanceFromHome',
                'JobRole', 'JobSatisfaction', 'MaritalStatus', 'OverTime',
                'RelationshipSatisfaction',  'StockOptionLevel', 'PerformanceRating', 'TrainingTimesLastYear','WorkLifeBalance' ]
feature_cols = digital_cols + category_cols

X_train, X_val, y_train, y_val = train_test_split(train[feature_cols], train[target_col], test_size=0.20, random_state=42)

cat_feature_indice = [i for i in range(10,28)]
train_pool = Pool(X_train, 
                  y_train, 
                  cat_features=cat_feature_indice)
test_pool = Pool(X_val, 
                 cat_features=cat_feature_indice) 

# specify the training parameters 
# ctb = CatBoostRegressor(iterations=1200, 
#                           depth=2, 
#                           learning_rate=0.1, 
#                           loss_function='RMSE')
ctb = CatBoostRegressor(iterations=1200, 
                          depth=2, 
                          learning_rate=0.1, 
                          loss_function='RMSE')

#train the model
ctb.fit(train_pool)
# make the prediction using the resulting model
y_pred = ctb.predict(test_pool)
print('catboost_auc:',metrics.roc_auc_score(y_val,y_pred))
#catboost
import numpy as np
from catboost import Pool, CatBoostRegressor

id_col = 'user_id'
target_col = 'Attrition'

digital_cols = ['Age', 'DailyRate', 'MonthlyIncome', 'NumCompaniesWorked', 'PercentSalaryHike',
                'TotalWorkingYears', 'YearsAtCompany', 'YearsInCurrentRole', 'YearsSinceLastPromotion', 'YearsWithCurrManager']
category_cols = ['BusinessTravel', 'Department',  'Education', 'EducationField',
                'EmployeeNumber', 'EnvironmentSatisfaction', 'Gender', 'JobInvolvement', 'JobLevel','DistanceFromHome',
                'JobRole', 'JobSatisfaction', 'MaritalStatus', 'Over18', 'OverTime',
                'RelationshipSatisfaction', 'StandardHours', 'StockOptionLevel', 'PerformanceRating', 'TrainingTimesLastYear','WorkLifeBalance' ]
feature_cols = digital_cols + category_cols

#X_train, X_val, y_train, y_val = train_test_split(train[feature_cols], train[target_col], test_size=0.20, random_state=42)

cat_feature_indice = [i for i in range(10,31)]
train_pool = Pool(train[feature_cols], 
                  train[target_col], 
                  cat_features=cat_feature_indice)
test_pool = Pool(test[feature_cols], 
                 cat_features=cat_feature_indice) 

# specify the training parameters 
# ctb = CatBoostRegressor(iterations=1200, 
#                           depth=2, 
#                           learning_rate=0.1, 
#                           loss_function='RMSE')
ctb = CatBoostRegressor(iterations=1100, 
                          depth=2, 
                          learning_rate=0.1, 
                          loss_function='RMSE')

#train the model
ctb.fit(train_pool)
# make the prediction using the resulting model
ypred = ctb.predict(test_pool)
#linear
# from sklearn.linear_model import LinearRegression

# clf_after = LinearRegression()
# clf_after.fit(new_train_features, train_labels)
# ypred = clf_after.predict(new_test_features)

from sklearn.linear_model import LinearRegression

clf_after = LinearRegression()
clf_after.fit(train_features, train_labels)
ypred = clf_after.predict(test_features)

#ensemble__classification
averaged_models = AveragingModels(models = (Adaboost,SVC, GBoosting, RF,xgbc))
averaged_models.fit(train_features, train_labels)
#predict
ypred = averaged_models.predict(test_features)
#xgboost
### fit model for train data
import xgboost as xgb
#dataset
dtrain = xgb.DMatrix(train_features, label = train_labels)

#parameters
num_round = 500
param = {'bst:max_depth':2, 'bst:eta':1, 'silent':1, 'objective':'binary:logistic' }
param['nthread'] = 2
param['eval_metric'] = 'auc'
plst = param.items()
bst = xgb.train(plst, dtrain, num_round)
dtest = xgb.DMatrix(test_features)
ypred = bst.predict(dtest)
#SMOTE SVM
# from imblearn.over_sampling import SMOTE
# sm = SMOTE(random_state=42)
# X_res, y_res = sm.fit_resample(train_features, train_labels)


from sklearn.svm import SVR
from sklearn import metrics
#X_train, X_val, y_train, y_val = train_test_split(train_features, train_labels, test_size=0.20, random_state=42)
svc = SVR(kernel = 'linear')
svc.fit(new_train_features, train_labels)
ypred = svc.predict(new_test_features)

# #X_train, X_val, y_train, y_val = train_test_split(train_features, train_labels, test_size=0.20, random_state=42)
# svc = SVR(kernel = 'linear')
# svc.fit(X_res, y_res)
# ypred_up = svc.predict(test_features)

sub = test[['user_id']].copy()
sub['Attrition'] = ypred
sub['Attrition'] = sub['Attrition'].apply(lambda x: x if x >=0 else 0.0005)
sub.to_csv('submission.csv', index=False)

# sub['Attrition'] = ypred_up
# sub['Attrition'] = sub['Attrition'].apply(lambda x: x if x >=0 else 0.0005)
# sub.to_csv('submission_up.csv', index=False)