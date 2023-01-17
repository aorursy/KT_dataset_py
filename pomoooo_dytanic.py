import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

data_raw = pd.read_csv("../input/train.csv")
data_val = pd.read_csv("../input/test.csv")

combine = [data_raw, data_val]

# 预览数据
data_raw.head()
data_val.head()
print(data_raw.info())
print('*'*40)
print(data_val.info())
#数值变量描述
data_raw.describe() 
data_val.describe(include='all')
sex_mapping = {'male':1, 'female':0}
for df in combine:
    df['Sex'] = df['Sex'].map(sex_mapping)
data_raw.head()
for df in combine:
    df['Age'] = df['Age'].fillna(df['Age'].median())
data_raw.info()
data_raw['Embarked'] = data_raw['Embarked'].fillna('S')
dummies1 = pd.get_dummies(data_raw['Embarked'], prefix='Embarked')
data_raw = pd.concat([data_raw, dummies1], axis=1)
data_raw.drop(['Embarked'], axis=1, inplace=True)
# for df in combine:
#     dummies = pd.get_dummies(df['Embarked'])
#     df = pd.concat([df, dummies], axis=1)
#     df.drop(['Embarked'], axis=1, inplace=True)
data_raw.head()

dummies2 = pd.get_dummies(data_val['Embarked'], prefix='Embarked')
data_val = pd.concat([data_val, dummies2], axis=1)
data_val.drop(['Embarked'], axis=1, inplace=True)
data_val.head()
# 缺失值处理
data_val['Fare'].fillna(data_val['Fare'].mean(), inplace=True)
from sklearn import model_selection

columns = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked_C', 'Embarked_Q', 'Embarked_S']
x_train = data_raw[columns]
y_train = data_raw['Survived']
#  将数据分为训练集和测试集
# train_x, test_x, train_y, test_y = model_selection.train_test_split(x_raw, y_raw, random_state = 0)
x_test = data_val[columns]
x_train.shape

print(len(x_train),len(y_train))
colormap = plt.cm.RdBu
plt.figure(figsize=(14,12))
plt.title('Pearson Correlation of Features', y=1.05, size=15)
sns.heatmap(data.astype(float).corr(),linewidths=0.1,vmax=1.0, 
            square=True, cmap=colormap, linecolor='white', annot=True)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import cross_validate

lr_model = LogisticRegression()
lr_model.fit(train_x, train_y)
scores = cross_validate(lr_model,x_raw,y_raw,cv=5)
# 准确率
acc = lr_model.score(train_x, train_y)
print("训练集准确率:", acc)
print("测试集准确率:", lr_model.score(test_x, test_y))

y_pred_lr = lr_model.predict(x_val)
scores
from sklearn.svm import SVC
from sklearn.metrics import mean_absolute_error
svc = SVC()
svc.fit(train_x, train_y)

# 准确率
acc_svc = svc.score(train_x, train_y)
print("SVC准确率:", acc_svc)
print("测试集准确率:", svc.score(test_x, test_y))

y_pred_svc = svc.predict(x_val)
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators=200)
rf.fit(train_x, train_y)
# rf_feature = rf.fit(train_x, train_y).feature_importances_
# print(rf_feature)

# 准确率
acc_rf = rf.score(train_x, train_y)
print("训练集准确率:", acc_rf)
print("测试集准确率:", rf.score(test_x, test_y))

y_pred_rf = rf.predict(x_val)
from xgboost import XGBClassifier

xgb = XGBClassifier(n_estimators=200)
xgb.fit(train_x, train_y)
# rf_feature = rf.fit(train_x, train_y).feature_importances_
# print(rf_feature)

# 准确率
acc_xgb = xgb.score(train_x, train_y)
print("训练集准确率:", acc_xgb)
print("测试集准确率:", xgb.score(test_x, test_y))

y_pred_xgb = xgb.predict(x_val)

from sklearn import svm, tree, linear_model, neighbors, naive_bayes, ensemble, discriminant_analysis, gaussian_process
from xgboost import XGBClassifier
import sklearn.model_selection
MLA = [
    #Ensemble Methods
    ensemble.AdaBoostClassifier(),
    ensemble.BaggingClassifier(),
    ensemble.ExtraTreesClassifier(),
    ensemble.GradientBoostingClassifier(),
    ensemble.RandomForestClassifier(),

    #Gaussian Processes
    gaussian_process.GaussianProcessClassifier(),
    
    #GLM
    linear_model.LogisticRegressionCV(),
    linear_model.PassiveAggressiveClassifier(),
    linear_model.RidgeClassifierCV(),
    linear_model.SGDClassifier(),
    linear_model.Perceptron(),
    
    #Navies Bayes
    naive_bayes.BernoulliNB(),
    naive_bayes.GaussianNB(),
    
    #Nearest Neighbor
    neighbors.KNeighborsClassifier(),
    
    #SVM
    svm.SVC(probability=True),
    svm.NuSVC(probability=True),
    svm.LinearSVC(),
    
    #Trees    
    tree.DecisionTreeClassifier(),
    tree.ExtraTreeClassifier(),
    
    #Discriminant Analysis
    discriminant_analysis.LinearDiscriminantAnalysis(),
    discriminant_analysis.QuadraticDiscriminantAnalysis(),

    
    #xgboost: http://xgboost.readthedocs.io/en/latest/model.html
    XGBClassifier()    
    ]

cv_split = model_selection.ShuffleSplit(n_splits = 10, test_size = 0.25, random_state = 0)


MLA_columns = ['MLA Name', 'MLA Parameters','MLA Train Accuracy Mean', 'MLA Test Accuracy Mean', 'MLA Test Accuracy 3*STD' ,'MLA Time']
MLA_compare = pd.DataFrame(columns = MLA_columns)

#create table to compare MLA predictions
# MLA_predict = y_raw

#index through MLA and save performance to table
row_index = 0
for alg in MLA:

    #set name and parameters
    MLA_name = alg.__class__.__name__
    MLA_compare.loc[row_index, 'MLA Name'] = MLA_name
    MLA_compare.loc[row_index, 'MLA Parameters'] = str(alg.get_params())
    
    #score model with cross validation: http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_validate.html#sklearn.model_selection.cross_validate
    cv_results = model_selection.cross_validate(alg, x_raw, y_raw, cv=cv_split)

    MLA_compare.loc[row_index, 'MLA Time'] = cv_results['fit_time'].mean()
    MLA_compare.loc[row_index, 'MLA Train Accuracy Mean'] = cv_results['train_score'].mean()
    MLA_compare.loc[row_index, 'MLA Test Accuracy Mean'] = cv_results['test_score'].mean()   
    #if this is a non-bias random sample, then +/-3 standard deviations (std) from the mean, should statistically capture 99.7% of the subsets
    MLA_compare.loc[row_index, 'MLA Test Accuracy 3*STD'] = cv_results['test_score'].std()*3   #let's know the worst that can happen!
    

    #save MLA predictions - see section 6 for usage
    alg.fit(x_raw, y_raw)
#     MLA_predict[MLA_name] = alg.predict(x_val)
    
    row_index+=1

    
#print and sort table: https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.sort_values.html
MLA_compare.sort_values(by = ['MLA Test Accuracy Mean'], ascending = False, inplace = True)
MLA_compare
import seaborn as sns
import matplotlib.pyplot as plt
sns.barplot(x='MLA Test Accuracy Mean', y = 'MLA Name', data = MLA_compare, color = 'm')

#prettify using pyplot: https://matplotlib.org/api/pyplot_api.html
plt.title('Machine Learning Algorithm Accuracy Score \n')
plt.xlabel('Accuracy Score (%)')
plt.ylabel('Algorithm')
from sklearn.cross_validation import KFold
from sklearn.ensemble import (RandomForestClassifier, AdaBoostClassifier, 
                              GradientBoostingClassifier, ExtraTreesClassifier)
from sklearn.svm import SVC

# Class to extend the Sklearn classifier
class SklearnHelper(object):
    def __init__(self, clf, seed=0, params=None):
        params['random_state'] = seed
        self.clf = clf(**params)

    def train(self, x_train, y_train):
        self.clf.fit(x_train, y_train)

    def predict(self, x):
        return self.clf.predict(x)
    
    def fit(self,x,y):
        return self.clf.fit(x,y)
    
    def feature_importances(self,x,y):
        print(self.clf.fit(x,y).feature_importances_)
    
ntrain = x_train.shape[0]
ntest = x_test.shape[0]
SEED = 0 # for reproducibility
NFOLDS = 5 # set folds for out-of-fold prediction
kf = KFold(ntrain, n_folds= NFOLDS, random_state=SEED) 

def get_oof(clf, x_train, y_train, x_test):
    oof_train = np.zeros((ntrain,))
    oof_test = np.zeros((ntest,))
    oof_test_skf = np.empty((NFOLDS, ntest))
    
    # 将训练集分成5份，4份用作训练，1份用作预测，共进行5次
    for i, (train_index, test_index) in enumerate(kf):
        x_tr = x_train[train_index]
        y_tr = y_train[train_index]
        x_te = x_train[test_index]

        clf.train(x_tr, y_tr) # 4/5训练集用作训练

        oof_train[test_index] = clf.predict(x_te)  # 1/5训练集用作预测
        oof_test_skf[i, :] = clf.predict(x_test)   # 对测试集进行预测

    oof_test[:] = oof_test_skf.mean(axis=0)
    return oof_train.reshape(-1, 1), oof_test.reshape(-1, 1)



rf_params = {
    'n_jobs': -1,
    'n_estimators': 500,
     'warm_start': True, 
     #'max_features': 0.2,
    'max_depth': 6,
    'min_samples_leaf': 2,
    'max_features' : 'sqrt',
    'verbose': 0
}

# Extra Trees Parameters
et_params = {
    'n_jobs': -1,
    'n_estimators':500,
    #'max_features': 0.5,
    'max_depth': 8,
    'min_samples_leaf': 2,
    'verbose': 0
}

# AdaBoost parameters
ada_params = {
    'n_estimators': 500,
    'learning_rate' : 0.75
}

# Gradient Boosting parameters
gb_params = {
    'n_estimators': 500,
     #'max_features': 0.2,
    'max_depth': 5,
    'min_samples_leaf': 2,
    'verbose': 0
}

# Support Vector Classifier parameters 
svc_params = {
    'kernel' : 'linear',
    'C' : 0.025
    }


# Create 5 objects that represent our 4 models
rf = SklearnHelper(clf=RandomForestClassifier, seed=SEED, params=rf_params)
et = SklearnHelper(clf=ExtraTreesClassifier, seed=SEED, params=et_params)
ada = SklearnHelper(clf=AdaBoostClassifier, seed=SEED, params=ada_params)
gb = SklearnHelper(clf=GradientBoostingClassifier, seed=SEED, params=gb_params)
svc = SklearnHelper(clf=SVC, seed=SEED, params=svc_params)


# Create Numpy arrays of train, test and target ( Survived) dataframes to feed into our models
y_train = y_train.ravel()
x_train = x_train.values # Creates an array of the train data
x_test = x_test.values # Creats an array of the test data

# Create our OOF train and test predictions. These base results will be used as new features
et_oof_train, et_oof_test = get_oof(et, x_train, y_train, x_test) # Extra Trees
rf_oof_train, rf_oof_test = get_oof(rf,x_train, y_train, x_test) # Random Forest
ada_oof_train, ada_oof_test = get_oof(ada, x_train, y_train, x_test) # AdaBoost 
gb_oof_train, gb_oof_test = get_oof(gb,x_train, y_train, x_test) # Gradient Boost
svc_oof_train, svc_oof_test = get_oof(svc,x_train, y_train, x_test) # Support Vector Classifier

print("Training is complete")

rf_feature = rf.feature_importances(x_train,y_train)
et_feature = et.feature_importances(x_train, y_train)
ada_feature = ada.feature_importances(x_train, y_train)
gb_feature = gb.feature_importances(x_train,y_train)
import xgboost as xgb

x_train2 = np.concatenate(( et_oof_train, rf_oof_train, ada_oof_train, gb_oof_train, svc_oof_train), axis=1)
x_test2 = np.concatenate(( et_oof_test, rf_oof_test, ada_oof_test, gb_oof_test, svc_oof_test), axis=1)

gbm = xgb.XGBClassifier(
    #learning_rate = 0.02,
 n_estimators= 2000,
 max_depth= 4,
 min_child_weight= 2,
 #gamma=1,
 gamma=0.9,                        
 subsample=0.8,
 colsample_bytree=0.8,
 objective= 'binary:logistic',
 nthread= -1,
 scale_pos_weight=1).fit(x_train2, y_train)

predictions = gbm.predict(x_test2)
# 准确率
acc_xgb = gbm.score(x_train2, y_train)
print("训练集准确率:", acc_xgb)
my_submission = pd.DataFrame({'PassengerId':data_val.PassengerId, 'Survived':predictions})
my_submission.to_csv('titanic_submission.csv', index=False)

