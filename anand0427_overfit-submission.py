import warnings

warnings.filterwarnings('ignore')

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
df = pd.read_csv("../input/train.csv")

test_df = pd.read_csv("../input/test.csv")
df.head()
df["target"].value_counts(), df.info(), df.describe()
y = df["target"]

X = df.drop(["target", "id"], axis = 1)
def logisticRegression(X,y):

    train_x, test_x, train_y, test_y = train_test_split(X, y, test_size = 0.2, random_state = 42)

    from sklearn.linear_model import LogisticRegression

    from sklearn.metrics import accuracy_score, log_loss

    lrModel = LogisticRegression()

    lrModel.fit(train_x, train_y)

    print(accuracy_score(lrModel.predict(test_x), test_y))

    print(log_loss(lrModel.predict(test_x), test_y))

    return lrModel
logisticRegression(X,y)
y.value_counts()
# from imblearn.over_sampling import SMOTE

# sm = SMOTE(random_state=42, k_neighbors=3, n_jobs = 5)

# X,y = sm.fit_resample(X,y)
logisticRegression(X,y)
X = pd.DataFrame(X)

X.std().plot()
from sklearn.preprocessing import StandardScaler

ss = StandardScaler()

X = ss.fit_transform(X)

lrModel = logisticRegression(X,y)

test_df_pred = lrModel.predict_proba(test_df.iloc[:,1:])
X = pd.DataFrame(X)

X.head()
test_df_pred
submit_df = pd.read_csv("../input/sample_submission.csv")

submit_df.head()

submit_df["target"] = test_df_pred

# submit_df.to_csv("lrModel.csv", index = False) 
from sklearn.model_selection import StratifiedShuffleSplit

from sklearn.metrics import accuracy_score, log_loss

from sklearn.neighbors import KNeighborsClassifier

from sklearn.svm import SVC

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis

from sklearn.linear_model import LogisticRegression

from xgboost import XGBClassifier

import lightgbm as lgb



classifiers = [

    KNeighborsClassifier(3),

    SVC(probability=True),

    DecisionTreeClassifier(),

    RandomForestClassifier(),

    AdaBoostClassifier(),

    GradientBoostingClassifier(),

    GaussianNB(),

    LinearDiscriminantAnalysis(),

    QuadraticDiscriminantAnalysis(),

    LogisticRegression(),

    XGBClassifier()]
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2, random_state = 42)
for clf in classifiers:

    name = clf.__class__.__name__

    clf.fit(X_train, y_train)

    acc = clf.score(X_test,y_test)

    print("{0}: {1}".format(name,acc))
def applyModel(model, X,y):

    train_x, test_x, train_y, test_y = train_test_split(X, y, test_size = 0.2, random_state = 42)

    from sklearn.metrics import accuracy_score, log_loss

    model.fit(train_x, train_y)

#     print(accuracy_score(model.predict(test_x), test_y))

#     print(log_loss(model.predict(test_x), test_y))

    return (accuracy_score(model.predict(test_x), test_y), log_loss(model.predict(test_x), test_y))
qdaModel = QuadraticDiscriminantAnalysis()

applyModel(qdaModel, X, y)
submit_df = pd.read_csv("../input/sample_submission.csv")

submit_df.head()

test_df_pred = qdaModel.predict_proba(test_df.iloc[:,1:])

submit_df["target"] = test_df_pred

# submit_df.to_csv("qdaModel Base 2.csv", index = False)
from sklearn.model_selection import StratifiedShuffleSplit

StratifiedShuffleSplit()
def featureExtractionAndModel(stat_model, feature_ext_model, X, y):

    if(clf.__class__.__name__ == "LinearDiscriminantAnalysis") or (clf.__class__.__name__ == "RFE") or (clf.__class__.__name__ == "SelectKBest"):

        train_x, test_x, train_y, test_y = train_test_split(feature_ext_model.fit_transform(X, y), y, test_size = 0.2, random_state = 42)

    else:

        train_x, test_x, train_y, test_y = train_test_split(feature_ext_model.fit_transform(X), y, test_size = 0.2, random_state = 42)

    from sklearn.metrics import accuracy_score, log_loss

    stat_model.fit(train_x, train_y)

#     print(accuracy_score(model.predict(test_x), test_y))

#     print(log_loss(model.predict(test_x), test_y))

    

    return (accuracy_score(stat_model.predict(test_x), test_y), log_loss(stat_model.predict(test_x), test_y))
lrModel_FE = LogisticRegression()
from sklearn.manifold import LocallyLinearEmbedding

from sklearn.manifold import SpectralEmbedding

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.decomposition import FastICA

from sklearn.decomposition import PCA

from sklearn.manifold import TSNE

from sklearn.feature_selection import RFE

from sklearn.feature_selection import SelectKBest



extractors = [

    LocallyLinearEmbedding(n_components = 150),

    SpectralEmbedding(n_components = 150),

    PCA(n_components=150,svd_solver='full'),

    LinearDiscriminantAnalysis(n_components = 150),

    FastICA(n_components = 150),

    TSNE(n_components = 3),

    RFE(lrModel_FE, n_features_to_select = 100),

    SelectKBest(k = 150)

    ]
for clf in extractors:

    name = clf.__class__.__name__

#     print(name)

    acc = featureExtractionAndModel(lrModel_FE, clf, X, y)

    print("{0}: {1:.4f} {2:.4f}".format(name,acc[0], acc[1]))
acc_lst = []

log_loss_lst = []

for i in range(25, 150, 5):

    acc = applyModel(lrModel_FE, RFE(lrModel_FE, n_features_to_select = i).fit_transform(X, y), y)

#     print("{0}: {1:.4f} {2:.4f}".format(i, acc[0], acc[1]))

    acc_lst.append(acc[0])

    log_loss_lst.append(acc[1])

plt.plot(range(25, 150, 5), acc_lst)
acc_lst = []

log_loss_lst = []

for i in range(155, 250, 1):

    acc = applyModel(lrModel_FE, SelectKBest(k = i).fit_transform(X, y), y)

    print("{0}: {1:.4f} {2:.4f}".format(i, acc[0], acc[1]))

    acc_lst.append(acc[0])

    log_loss_lst.append(acc[1])

plt.plot(range(155, 250, 1), acc_lst)
rfe = RFE(lrModel_FE, n_features_to_select = 50)

applyModel(lrModel_FE, rfe.fit_transform(X, y), y)

submit_df = pd.read_csv("../input/sample_submission.csv")

submit_df.head()

test_df_pred = lrModel_FE.predict(rfe.transform(test_df.iloc[:,1:]))

submit_df["target"] = test_df_pred

# submit_df.to_csv("lrModel RFE.csv", index = False) #0.519
kbest = SelectKBest(k = 173)

applyModel(lrModel_FE, kbest.fit_transform(X, y), y)

submit_df = pd.read_csv("../input/sample_submission.csv")

submit_df.head()

test_df_pred = lrModel_FE.predict(kbest.transform(test_df.iloc[:,1:]))

submit_df["target"] = test_df_pred

# submit_df.to_csv("lrModel SelectKBest.csv", index = False) 0.500
acc_lst = []

log_loss_lst = []

for i in range(70, 299, 1):

    acc = applyModel(tuned_lr_model, PCA(n_components= i).fit_transform(X), y)

    print("{0}: {1:.4f} {2:.4f}".format(i, acc[0], acc[1]))

    acc_lst.append(acc[0])

    log_loss_lst.append(acc[1])

plt.plot(range(70, 299, 1), acc_lst)
tuned_lr_model = LogisticRegression(C=2.4, class_weight=None, 

                                                      dual=False, fit_intercept=True,

          intercept_scaling=1, max_iter=8, multi_class='warn',

          n_jobs=None, penalty='l2', random_state=42, solver='warn',

          tol=0.0001, warm_start=False)
pca = PCA(n_components = 73)

applyModel(tuned_lr_model, pca.fit_transform(X), y)

submit_df = pd.read_csv("../input/sample_submission.csv")

submit_df["target"] = tuned_lr_model.predict(pca.transform(test_df.iloc[:,1:]))

# submit_df.to_csv("lrModel tuned and PCA.csv", index = False) #0.499
import eli5

weights_df = eli5.formatters.as_dataframe.explain_weights_df(tuned_lr_model, top=200)

weights_df["weight"] = weights_df["weight"].abs()

weights_df = weights_df.sort_values(by="weight", ascending=False)

weights_df.iloc[:20,:]
best_features = [int(feature[1:]) for feature in weights_df["feature"] if feature!="<BIAS>"][:10]

# X.loc[:, best_features]
acc_lst = []

log_loss_lst = []

for i in range(5, 150, 1):

    best_features = [int(feature[1:]) for feature in weights_df["feature"] if feature!="<BIAS>"][:i]

    acc = applyModel(lrModel_FE, X.loc[:, best_features], y)

    print("{0}: {1:.4f} {2:.4f}".format(i, acc[0], acc[1]))

    acc_lst.append(acc[0])

    log_loss_lst.append(acc[1])

plt.plot(range(5, 150, 1), acc_lst)
best_features = [int(feature[1:]) for feature in weights_df["feature"] if feature!="<BIAS>"][:10]

applyModel(tuned_lr_model, X.loc[:, best_features], y)

submit_df = pd.read_csv("../input/sample_submission.csv")

submit_df["target"] = tuned_lr_model.predict(test_df.iloc[:,best_features])

submit_df.to_csv("lrModel tuned and ELI5.csv", index = False) #0.499
best_features = [int(feature[1:]) for feature in weights_df["feature"] if feature!="<BIAS>"][:10]

applyModel(lrModel_FE, X.loc[:, best_features], y)

submit_df = pd.read_csv("../input/sample_submission.csv")

submit_df["target"] = lrModel_FE.predict(test_df.iloc[:,best_features])

submit_df.to_csv("lrModel_FE and ELI5.csv", index = False) #0.499
### Lets create a model on GaussianNB and use it on test data
# gnb = GaussianNB()

# gnb.fit(X,y)

# test_df_pred = pd.DataFrame({"target": gnb.predict_proba(test_df.iloc[:,1:])})
# pd.read_csv("../input/sample_submission.csv").head()
# test_df_pred["id"] = test_df["id"]
# test_df_pred.head()
# test_df_pred.to_csv("Submission.csv", index=False)
# import matplotlib.pyplot as plt
# from sklearn.decomposition import PCA

# pca = PCA().fit(X)

# plt.plot(np.cumsum(pca.explained_variance_ratio_))

# plt.xlim(0,250,1)
# sklearn_pca = PCA(n_components=250)

# print(sklearn_pca)
# X_train_pca = sklearn_pca.fit_transform(X_train)

# print(X_train_pca.shape)



# X_test_pca = sklearn_pca.transform(X_test)

# print(X_test_pca.shape)
# classifiers = [

#     KNeighborsClassifier(3),

#     SVC(probability=True),

#     DecisionTreeClassifier(),

#     RandomForestClassifier(),

#     AdaBoostClassifier(),

#     GradientBoostingClassifier(),

#     GaussianNB(),

#     LinearDiscriminantAnalysis(),

#     QuadraticDiscriminantAnalysis(),

#     LogisticRegression(),

#     XGBClassifier()]

# for clf in classifiers:

#     name = clf.__class__.__name__

#     clf.fit(X_train_pca, y_train)

#     acc = clf.score(X_test_pca,y_test)

#     print("{0}: {1}".format(name,acc))
# sklearn_pca = PCA(n_components=200)

# print(sklearn_pca)
# X_pca = sklearn_pca.fit_transform(X)

# print(X_pca.shape)



# test_pca = sklearn_pca.transform(test_df.iloc[:,1:])

# print(test_pca.shape)
# qda = QuadraticDiscriminantAnalysis()

# qda.fit(X_pca,y)

# test_df_pred = pd.DataFrame({"target": qda.predict_proba(test_pca)})

# test_df_pred["id"] = test_df["id"]

# test_df_pred.to_csv("Submission with pca and qda.csv", index=False)
# qda = QuadraticDiscriminantAnalysis()

# qda.fit(X,y)

# test_df_pred = pd.DataFrame({"target": qda.predict_proba(test_df)})

# test_df_pred["id"] = test_df["id"]

# test_df_pred.to_csv("Submission with qda.csv", index=False)
# from sklearn.model_selection import GridSearchCV
# xgb_tuning = XGBClassifier(learning_rate =0.1,

#  n_estimators=1000,

#  max_depth=5,

#  min_child_weight=1,

#  gamma=0,

#  subsample=0.8,

#  colsample_bytree=0.8,

#  objective= 'binary:logistic',

#  nthread=4,

#  scale_pos_weight=1,

#  seed=27,

# n_gpus=1)
# xgb_tuning.fit(X_train,y_train)
# xgb_tuning.score(X_test,y_test)
# param_test1 = {

#  'max_depth':[7,8,9,10],

#  'min_child_weight':[0,1,2]

# }
# gsearch1 = GridSearchCV(estimator = XGBClassifier( learning_rate =0.1, n_estimators=140, max_depth=5,

#                                                  min_child_weight=1, gamma=0, 

#                                                   subsample=0.8, colsample_bytree=0.8,

#                                                   objective= 'binary:logistic',

#                                                   nthread=32, scale_pos_weight=1, seed=27,n_gpus=1), 

#                         param_grid = param_test1, scoring='roc_auc',n_jobs=32,iid=False, cv=5,verbose=4)

# gsearch1.fit(X_train,y_train)

# gsearch1.best_params_, gsearch1.best_score_
# xgboost_params = { "n_estimators": 400, 'tree_method':'gpu_hist', 'predictor':'gpu_predictor' }
# param_test3 = {

#  'gamma':[i/10.0 for i in range(0,5)]

# }

# gsearch3 = GridSearchCV(estimator = XGBClassifier(learning_rate =0.1, max_depth=9,

#  min_child_weight=0, gamma=0, subsample=0.8, colsample_bytree=0.8,

#  objective= 'binary:logistic', nthread=32, scale_pos_weight=1,seed=27,n_gpus=1,**xgboost_params), 

#  param_grid = param_test3, scoring='roc_auc',n_jobs=32,iid=False,verbose=4 ,cv=5)

# gsearch3.fit(X,y)

# gsearch3.best_params_, gsearch3.best_score_
# param_test4 = {

#  'colsample_bytree':[i/10.0 for i in range(7,10)],

#     'subsample': [i/10.0 for i in range(7,10)]

# }

# gsearch4 = GridSearchCV(estimator = XGBClassifier(learning_rate =0.1, max_depth=9,

#  min_child_weight=0, gamma=0.0, subsample=0.8, colsample_bytree=0.8,

#  objective= 'binary:logistic', nthread=32, scale_pos_weight=1,seed=27,n_gpus=1,**xgboost_params), 

#  param_grid = param_test4, scoring='roc_auc',n_jobs=32,iid=False,verbose=5 ,cv=5)

# gsearch4.fit(X,y)

# gsearch4.best_params_, gsearch4.best_score_
# param_test5 = {

#     'subsample': [i/10.0 for i in range(5,8)]

# }

# gsearch5 = GridSearchCV(estimator = XGBClassifier(learning_rate =0.1, max_depth=9,

#  min_child_weight=0, gamma=0.0, subsample=0.8, colsample_bytree=0.8,

#  objective= 'binary:logistic', nthread=32, scale_pos_weight=1,seed=27,n_gpus=1,**xgboost_params), 

#  param_grid = param_test5, scoring='roc_auc',n_jobs=32,iid=False,verbose=5 ,cv=5)

# gsearch5.fit(X,y)

# gsearch5.best_params_, gsearch5.best_score_
# param_test7 = {

#  'reg_alpha':[i/10.0 for i in range(0,3)],

#     'reg_lambda':[i/10.0 for i in range(9,11)]

# }

# gsearch7 = GridSearchCV(estimator = XGBClassifier(learning_rate =0.11, max_depth=9,

#  min_child_weight=0, gamma=0.0, subsample=0.7, colsample_bytree=0.8,

#  objective= 'binary:logistic', nthread=5, scale_pos_weight=1,seed=27,n_gpus=1,**xgboost_params), 

#  param_grid = param_test7, scoring='roc_auc',n_jobs=32,iid=False,verbose=5 ,cv=5)

# gsearch7.fit(X,y)

# gsearch7.best_params_, gsearch7.best_score_
# param_test8 = {

#  'learning_rate' :[0.08,0.09,0.1,0.11,0.12]

# }

# gsearch8 = GridSearchCV(estimator = XGBClassifier(

#                                 learning_rate =0.1,max_depth=9,min_child_weight=0, gamma=0.0, 

#                                 subsample=0.7, colsample_bytree=0.8,objective= 'binary:logistic', 

#                                 nthread=5,reg_alpha= 0.1, reg_lambda= 0.9, 

#                                 scale_pos_weight=1,seed=27,n_gpus=1,**xgboost_params), 

#                         param_grid = param_test8, scoring='roc_auc',

#                         n_jobs=32,iid=False,verbose=5 ,cv=5

#                        )

# gsearch8.fit(X,y)

# gsearch8.best_params_, gsearch8.best_score_
# xgb_tuning = XGBClassifier(learning_rate =0.11,

#  n_estimators=400,

#  max_depth=9,

#  min_child_weight=0,

#  gamma=0.0,

#  subsample=0.7,

#  colsample_bytree=0.8,

#  objective= 'binary:logistic',

#  nthread=4,

#  scale_pos_weight=1,

#  seed=27,

# reg_alpha= 0.1, reg_lambda= 0.9,

# n_gpus=1,

# tree_method='gpu_hist',

# predictor='gpu_predictor')
# xgb_tuning.fit(X_train,y_train)
# xgb_tuning.score(X_test,y_test)
# X.shape
# test_df.head()
# xgb_tuning.fit(X,y)

# test_df_pred = pd.DataFrame({"target": xgb_tuning.predict_proba(np.array(test_df.drop("id",axis=1)))})

# test_df_pred["id"] = test_df["id"]

# test_df_pred.to_csv("Submission xgb tuning.csv", index=False)
# from sklearn.linear_model import LogisticRegression
# lrcv = LogisticRegression(C=50000,penalty="l2")

# lrcv.fit(X_train,y_train)
# lrcv.score(X_test,y_test)
# param1 = {

#  'penalty':["l1","l2"]

# }

# gsearch = GridSearchCV(estimator = LogisticRegression(C=5, class_weight=None, dual=False,

#                                                       fit_intercept=True,

#           intercept_scaling=1, max_iter=100, multi_class='warn',

#           n_jobs=None, penalty='l1', random_state=42, solver='warn',

#           tol=0.0001, verbose=0, warm_start=False), 

#  param_grid = param1, scoring='roc_auc',n_jobs=4,iid=False, cv=5)

# gsearch.fit(X,y)

# gsearch.best_params_,gsearch.best_score_
# param2 = {

#  'C':[i/1000 if i!=0 else 1 for i in range(0,100000,10)]

# }

# gsearch2 = GridSearchCV(estimator = LogisticRegression(C=5, class_weight=None, dual=False,

#                                                        fit_intercept=True,

#           intercept_scaling=1, max_iter=100, multi_class='warn',

#           n_jobs=None, penalty='l2', random_state=42, solver='warn',

#           tol=0.0001, verbose=0, warm_start=False), 

#  param_grid = param2, scoring='roc_auc',n_jobs=4,iid=False, cv=5)

# gsearch2.fit(X,y)

# gsearch2.best_params_, gsearch2.best_score_
# param3 = {

#  'tol':[i/10000 if i!=0 else 1 for i in range(0,1000,1)]

# }

# gsearch3 = GridSearchCV(estimator = LogisticRegression(C=2.4, class_weight=None, 

#                                                       dual=False, fit_intercept=True,

#           intercept_scaling=1, max_iter=100, multi_class='warn',

#           n_jobs=None, penalty='l2', random_state=42, solver='warn',

#           tol=0.0001, verbose=0, warm_start=False), 

#  param_grid = param3, scoring='roc_auc',verbose=4,n_jobs=32,iid=False, cv=5)

# gsearch3.fit(X,y)

# gsearch3.best_params_, gsearch3.best_score_
# param4 = {

#  'max_iter':[i for i in range(8,100,1)]

# }

# gsearch4 = GridSearchCV(estimator = LogisticRegression(C=2.4, class_weight=None, 

#                                                       dual=False, fit_intercept=True,

#           intercept_scaling=1, max_iter=100, multi_class='warn',

#           n_jobs=None, penalty='l2', random_state=42, solver='warn',

#           tol=0.0001, verbose=0, warm_start=False), 

#  param_grid = param4, scoring='roc_auc',verbose=4,n_jobs=32,iid=False, cv=5)

# gsearch4.fit(X,y)

# gsearch4.best_params_, gsearch4.best_score_
# param5 = {

#  'class_weight':["balanced",None],

# }

# gsearch5 = GridSearchCV(estimator = LogisticRegression(C=2.4, class_weight=None, 

#                                                       dual=False, fit_intercept=True,

#           intercept_scaling=1, max_iter=8, multi_class='warn',

#           n_jobs=None, penalty='l2', random_state=42, solver='warn',

#           tol=0.0001, verbose=0, warm_start=False), 

#  param_grid = param5, scoring='roc_auc',verbose=4,n_jobs=-1,iid=False, cv=5)

# gsearch5.fit(X,y)

# gsearch5.best_params_, gsearch5.best_score_
# from sklearn.linear_model import LogisticRegression
# lr = LogisticRegression(C=2.4, class_weight="balanced", 

#                                                       dual=False, fit_intercept=True,

#           intercept_scaling=1, max_iter=8, multi_class='warn',

#           n_jobs=None, penalty='l2', random_state=42, solver='warn',

#           tol=0.0001, warm_start=False)

# lr.fit(X_train,y_train)
# lr.score(X_test,y_test)
# lr.fit(X,y)

# y_pred = pd.DataFrame({"target":lr.predict_proba(test_df.iloc[:,1:]).flatten()}, index=test_df.index)

# y_pred["id"] = test_df["id"] 

# y_pred.to_csv("submission with tuned logistic regression.csv", index = False)
# from sklearn.ensemble import RandomForestClassifier 
# rf = RandomForestClassifier(n_estimators=200, n_jobs=4, class_weight='balanced', max_depth=6)

# rf.fit(X_train,y_train)

# rf.score(X_test, y_test)
# from sklearn.feature_selection import RFE
# lr_rfe = RFE(lr, 75, step=1)

# lr_rfe.fit(X_train,y_train)

# # scores_table(selector, 'selector_clf')

# lr_rfe.score(X_test, y_test) 

# y_pred = lr_rfe.predict_proba(test_df.iloc[:,1:])

# s = pd.read_csv('../input/sample_submission.csv')

# s["target"] = y_pred

# s.to_csv("submission with RFE.csv", index = False)