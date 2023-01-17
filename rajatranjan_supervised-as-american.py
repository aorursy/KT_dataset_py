# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
%matplotlib inline
import matplotlib.pyplot as plt
import seaborn as sns

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
head=pd.read_csv('../input/header.csv')
train=pd.read_csv('../input/train.csv',names=head.columns)
test=pd.read_csv('../input/test.csv',names=head.columns)

head
test.head()
test_pr=test.drop(['key','label'],axis=1)
train.label.value_counts()
des=pd.concat([train.describe().T.drop('count',axis=1),test.describe().T.drop('count',axis=1)], axis=1)
des
# from sklearn.utils import shuffle
# # train = shuffle(train)

# sample0=train[train.label==0].sample(198357,random_state=101,axis=0).copy()
# sample1=train[train.label==1].copy()
# print(sample0.shape)
# print(sample1.shape)
# train_1=sample0.append(sample1)
# print(train_1.shape)
# train_1 = shuffle(train_1)
# train_1.head()
# test.describe()
train_1=train.copy()
train_1.hist(column='V1',by='label',figsize=(12,5))
# train.set_index('key',inplace=True)
# train.head()
# train.V1.plot.hist(by='label',bins=30)
# for j in ['V1','V2','V3','V4','V5','V6','V7','V8','V9','V10']:
# #     plt.figure(figsize=(12,9))
# #     sns.countplot(train[j],bins=30,hue=train['label'])
#     train_1.hist(column=j,by='label',figsize=(12,5),bins=50)
    
# for j in ['V1','V2','V3','V4','V5','V6','V7','V8','V9','V10']:
# #     plt.figure(figsize=(12,9))
# #     sns.countplot(train[j],bins=30,hue=train['label'])
#     test_pr.hist(column=j,figsize=(12,5),bins=50)
from sklearn.metrics import recall_score
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split


def evaluate_model_auc(model, X_test_parameter, y_test_parameter):
    ## The predictions
#     y_pred = model.predict_proba(X_test_parameter)[:,1]
    ## False positive rate, true positive rate and treshold
    y_pred = model.predict(X_test_parameter)
    fp_rate, tp_rate, treshold = roc_curve(y_test_parameter, y_pred)
    print("false positive",fp_rate)
    print('true postive',tp_rate)
    ## Calculate the auc score
    auc_score = auc(fp_rate, tp_rate)
    print('auc',auc_score)
    lAUC = np.trapz(tp_rate, fp_rate,dx=0.05)
    print('lAUC',lAUC)
#     lauc=fp_rate*0.05
#     ## Returns the score to the model
    print ("Metric needed",auc_score*0.35+0.65*lAUC)
    return (auc_score)
def plot_auc(model, X_test, y_test):
    ## Predictions
    y_pred = model.predict(X_test)
    
    ## Calculates auc score
    fp_rate, tp_rate, treshold = roc_curve(y_test, y_pred)
    auc_score = auc(fp_rate, tp_rate)
    
    ## Creates a new figure and adds its parameters
    plt.figure(figsize=(12,10))
    plt.title('ROC Curve')
    ## Plot the data - false positive rate and true positive rate
    plt.plot(fp_rate, tp_rate, 'b', label = 'AUC = %0.2f' % auc_score)
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
# def evaluate_model_score(model, X_test, y_test):
#     ## Return the score value to the model
#     return model.score(X_test, y_test)
# def evaluate_classification_report(model, y_test):
#     return classification_report(y_test, model.predict(X_test), target_names=['Regular transaction',
#                                                                       'Fraudulent transaction'])
def evaluate_model(model_param, X_test_param, y_test_param,feat=[],f=True):
    print("Model evaluation")
#     print("Accuracy: {:.5f}".format(evaluate_model_score(model_param, X_test_param, y_test_param)))
    print("Metric AUC LAUC: {:.5f}".format(evaluate_model_auc(model_param, X_test_param, y_test_param)))
#     print("\n#### Classification Report ####\n")
#     print(evaluate_classification_report(model_param, y_test_param))
    d={}
    plot_auc(model_param, X_test_param, y_test_param)
    if f==True:
        col=pd.DataFrame({'importance': model_param.feature_importances_, 'feature': x.columns})
        for f in feat:
            print('selecting '+str(f)+' features')
            main_col=col.sort_values(by=['importance'], ascending=[False])[:f]['feature'].values
            d[f]=main_col
        return d
# ## This is a shared function used to print out the results of a gridsearch process
# def gridsearch_results(gridsearch_model):
#     print('Best score: {} '.format(gridsearch_model.best_score_))
#     print('\n#### Best params ####\n')
#     print(gridsearch_model.best_params_)
# # the results of the default classifier
# # min_estimator - min number of estimators to run
# # max_estimator - max number of estimators to run
# # X_train, y_train, X_test, y_test - splitted dataset
# # scoring function: accuracy or auc
# def model_selection(min_estimator, max_estimator, X_train_param, y_train_param,
#                    X_test_param, y_test_param, scoring='accuracy'):
#     scores = [] 
#     ## Returns the classifier with highest accuracy score
#     if (scoring == 'accuracy'):
#         for n in range(min_estimator, max_estimator,100):
#             rfc_selection = RandomForestRegressor(n_estimators=n, random_state=42).fit(X_train_param, y_train_param)
#             score = evaluate_model_score(rfc_selection, X_test_param, y_test_param)
#             print('Number of estimators: {} - Score: {:.5f}'.format(n, score))
#             scores.append((rfc_selection, score))
            
#     ## Returns the classifier with highest auc score
#     elif (scoring == 'auc'):
#          for n in range(min_estimator, max_estimator,100):
#             rfc_selection = RandomForestRegressor(n_estimators=n, random_state=42).fit(X_train_param, y_train_param)
#             score = evaluate_model_auc(rfc_selection, X_test_param, y_test_param)
#             print('Number of estimators: {} - AUC: {:.5f}'.format(n, score))
#             scores.append((rfc_selection, score))
#     return sorted(scores, key=lambda x: x[1], reverse=True)[0][0]
# from sklearn.ensemble import GradientBoostingRegressor
# model = GradientBoostingRegressor(random_state=21, n_estimators=400)
# model.fit(X_train,y_train)

# transform the validation dataset
# rescaled_X_test = scaler.transform(X_test)

# maincolumns=evaluate_model(model, X_test, y_test,False)
# import lightgbm as lgb
# params = {
#     'boosting_type': 'gbdt',
#     'objective': 'regression',
#     'metric': 'rmsle',
#     'max_depth': 6, 
#     'learning_rate': 0.01,
#     'verbose': 1}
# n_estimators = 500

# n_iters = 5
# preds_buf = []
# err_buf = []
# for i in range(n_iters): 
#     x_train, x_valid, y_train, y_valid = train_test_split(x, y, test_size=0.10, random_state=i)
#     d_train = lgb.Dataset(x_train, label=y_train)
#     d_valid = lgb.Dataset(x_valid, label=y_valid)
#     watchlist = [d_valid]

#     model = lgb.train(params, d_train, n_estimators, watchlist, verbose_eval=1)
#     evaluate_model(model, x_valid, y_valid,False)
# #     preds = model.predict(x_valid)
# #     preds = np.exp(preds)
#     preds[preds < 0] = median_trip_duration
#     err = rmsle(np.exp(y_valid), preds)
#     err_buf.append(err)
#     print('RMSLE = ' + str(err))
    
#     preds = model.predict(X_test)
#     preds = np.exp(preds)
#     preds[preds < 0] = median_trip_duration
#     preds_buf.append(preds)

# print('Mean RMSLE = ' + str(np.mean(err_buf)) + ' +/- ' + str(np.std(err_buf)))
# # Average predictions
# preds = np.mean(preds_buf, axis=0)

x,y=train_1.drop(['key','label'],axis=1),train_1['label']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=101)
# from sklearn.model_selection import RandomizedSearchCV
# # Number of trees in random forest
# n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
# # Number of features to consider at every split
# max_features = ['auto', 'sqrt','log']
# # Maximum number of levels in tree
# max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
# max_depth.append(None)
# # Minimum number of samples required to split a node
# min_samples_split = [2, 5, 10]
# # Minimum number of samples required at each leaf node
# min_samples_leaf = [1, 2, 4]
# # Method of selecting samples for training each tree
# bootstrap = [True, False]
# # Create the random grid
# random_grid = {'n_estimators': n_estimators,
#                'max_features': max_features,
#                'max_depth': max_depth,
#                'min_samples_split': min_samples_split,
#                'min_samples_leaf': min_samples_leaf,
#                'bootstrap': bootstrap}
# print(random_grid)
# # {'bootstrap': [True, False],
# #  'max_depth': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, None],
# #  'max_features': ['auto', 'sqrt'],
# #  'min_samples_leaf': [1, 2, 4],
# #  'min_samples_split': [2, 5, 10],
# #  'n_estimators': [200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000]}
# param_grid = {
#     'max_depth': [10,50, 90],
#     'max_features': [30,40,50],
#     'min_samples_leaf': [3, 5],
#     'min_samples_split': [2,8, 10, 12],
#     'n_estimators': [100, 250, 400]
# }
from sklearn.ensemble import RandomForestRegressor
# rf=RandomForestRegressor(n_estimators=450,random_state=42,verbose=1,max_features=36)
# rf.fit(x_train,y_train)

# Use the random grid to search for best hyperparameters
# First create the base model to tune
# rf = RandomForestRegressor()
# Random search of parameters, using 3 fold cross validation, 
# search across 100 different combinations, and use all available cores
# rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = -1)
# grid_search = GridSearchCV(estimator = rfc, param_grid = param_grid, verbose = 2,cv=2)
# # Fit the random search model
# grid_search.fit(x_train,y_trai
# grid_search.best_params_
# main_cols=evaluate_model(rf, x_test, y_test,[50])
# evaluate_model(rfc, x_test, y_test)
# maincolumns=evaluate_model(rf, x_test, y_test,[30,40,45,50,52,54])
# main_cols
# from sklearn.decomposition import PCA
# pca=PCA(n_components=5)
# pca.fit(x)
# print(pca.explained_variance_ratio_.sum())
# x_pca=pca.transform(x)
# testpca=pca.transform(test_pr)

# from sklearn.ensemble import RandomForestRegressor
# rfpca=RandomForestRegressor(bootstrap=True,
#             max_depth=None, max_features='auto', max_leaf_nodes=None,
#             min_impurity_decrease=0.0, min_impurity_split=None,
#             min_samples_leaf=1, min_samples_split=2,
#             min_weight_fraction_leaf=0.0, n_estimators=200, n_jobs=-1,
#             oob_score=False, random_state=42, verbose=1)
# rfpca.fit(x_pca,y)
# predpca=rfpca.predict(testpca)
# subpca=pd.DataFrame({'key':test.key,'score':predpca})
# subpca.to_csv('hckamerAIpca1.csv',index=False)



# X_trainrf, X_testrf, y_trainrf, y_testrf = train_test_split(x, y, random_state=42)
# from sklearn.ensemble import RandomForestRegressor
# rf=RandomForestRegressor()
# rf.fit(X_trainrf,y_trainrf)
# maincolumns=evaluate_model(rf, X_testrf, y_testrf)
# rfc_model = model_selection(200,501, X_train, y_train, X_test, y_test, scoring='auc')
# maincolumns

# x1= x[list(maincolumns)]
# print(x1.head())
# from sklearn.preprocessing import StandardScaler
# m1=StandardScaler()
# xscaled1=m1.fit_transform(x1)
# x_train1,x_test1,y_train1, y_test1 = train_test_split(x1, y, test_size=0.25, random_state=101)
# rf1=RandomForestRegressor(n_estimators=200,n_jobs=-1,bootstrap=True,verbose=1)
# rf1.fit(x_train1,y_train1)

# rfc_model
# evaluate_model(rf1, x_test1, y_test1,False)
# # for j in [30,40,45,50,52,54]:
    
# #     test_pr+str(j)=test[d[j]]
# test_pr30=test[list(maincolumns[30])]
# test_pr40=test[list(maincolumns[40])]
# test_pr45=test[list(maincolumns[45])]
# test_pr50=test[list(maincolumns[50])]
# test_pr52=test[list(maincolumns[52])]
# test_pr54=test[list(maincolumns[54])]
# x30=x[list(maincolumns[30])]
# x40=x[list(maincolumns[40])]
# x45=x[list(maincolumns[45])]
# x50=x[list(maincolumns[50])]
# x52=x[list(maincolumns[52])]
# x54=x[list(maincolumns[54])]

# from sklearn.ensemble import RandomForestRegressor
# rfmain=RandomForestRegressor(n_estimators=200,random_state=101, verbose=1,bootstrap=True)
# #  [30,40,45,50,52,54]
# rfmain.fit(x30,y)
# ypred30=rfmain.predict(test_pr30)
# sub30=pd.DataFrame({'key':test.key,'score':ypred30})
# sub30.to_csv('hckamerAIsub30.csv',index=False)

# rfmain.fit(x40,y)
# ypred40=rfmain.predict(test_pr40)
# sub40=pd.DataFrame({'key':test.key,'score':ypred40})
# sub40.to_csv('hckamerAIsub40.csv',index=False)

# rfmain.fit(x45,y)
# ypred45=rfmain.predict(test_pr45)
# sub45=pd.DataFrame({'key':test.key,'score':ypred45})
# sub45.to_csv('hckamerAIsub45.csv',index=False)

# rfmain.fit(x50,y)
# ypred50=rfmain.predict(test_pr50)
# sub50=pd.DataFrame({'key':test.key,'score':ypred50})
# sub50.to_csv('hckamerAIsub50.csv',index=False)

# rfmain.fit(x52,y)
# ypred52=rfmain.predict(test_pr52)
# sub52=pd.DataFrame({'key':test.key,'score':ypred52})
# sub52.to_csv('hckamerAIsub52.csv',index=False)

# rfmain.fit(x54,y)
# ypred54=rfmain.predict(test_pr54)
# sub54=pd.DataFrame({'key':test.key,'score':ypred54})
# sub54.to_csv('hckamerAIsub54.csv',index=False)
# sub1=pd.DataFrame({'key':test.key,'score':pred1})
# sub1
# from sklearn.ensemble import RandomForestRegressor
# rfm=RandomForestRegressor(bootstrap=True,
#             max_depth=None, max_features='auto', max_leaf_nodes=None,
#             min_impurity_decrease=0.0, min_impurity_split=None,
#             min_samples_leaf=1, min_samples_split=2,
#             min_weight_fraction_leaf=0.0, n_estimators=400, n_jobs=-1,
#             oob_score=False, random_state=42, verbose=1)
# rfm.fit(x,y)
# pred=rfm.predict(test_pr)
# sub=pd.DataFrame({'key':test.key,'score':pred})
# sub.to_csv('hckamerAIBalanced.csv',index=False)
# sub1.to_csv('s1cate.csv',index=False)
# test_main=test[list(maincolumns)]
# from sklearn.ensemble import RandomForestClassifier
# rfc=RandomForestClassifier(bootstrap=True,n_estimators=2000,random_state=42,verbose=1)
# rfc.fit(x,y)
# print('predicting')
# pred1=rfc.predict_proba(test_pr)[:,1]
# pred=rfc.predict_proba(test_pr)[:,0]
# sub=pd.DataFrame({'key':test.key,'score':pred})
# sub.to_csv('hckamerAIclass.csv',index=False)
# sub1=pd.DataFrame({'key':test.key,'score':pred1})
# sub1.to_csv('hckamerAIclass1.csv',index=False)
x1=x[list(['V1', 'V6', 'V10', 'V4', 'V5', 'V8', 'V2', 'V7', 'V9', 'V14', 'V3',
        'V26', 'V36', 'V11', 'V37', 'V18', 'V13', 'V46', 'V16', 'V43',
        'V47', 'V38', 'V45', 'V12', 'V52', 'V27', 'V24', 'V44', 'V34',
        'V53', 'V25', 'V31', 'V41', 'V33', 'V30', 'V17', 'V35', 'V54',
        'V48', 'V40', 'V42', 'V23', 'V20', 'V39', 'V50', 'V32', 'V22',
        'V49', 'V15', 'V28'])]

x1.shape
test_pr1=test_pr[list(['V1', 'V6', 'V10', 'V4', 'V5', 'V8', 'V2', 'V7', 'V9', 'V14', 'V3',
        'V26', 'V36', 'V11', 'V37', 'V18', 'V13', 'V46', 'V16', 'V43',
        'V47', 'V38', 'V45', 'V12', 'V52', 'V27', 'V24', 'V44', 'V34',
        'V53', 'V25', 'V31', 'V41', 'V33', 'V30', 'V17', 'V35', 'V54',
        'V48', 'V40', 'V42', 'V23', 'V20', 'V39', 'V50', 'V32', 'V22',
        'V49', 'V15', 'V28'])]
test_pr1.shape

# test_pr
from sklearn.ensemble import RandomForestRegressor

# rf=RandomForestRegressor(n_estimators=350,random_state=42,verbose=1,max_features=30,min_samples_split=4)
# rf.fit(x,y)
# print('predicting')
# pred=rf.predict(test_pr)
# sub=pd.DataFrame({'key':test.key,'score':pred})
# sub.to_csv('hckamerAIme3.csv',index=False)
# sub1=pd.DataFrame({'key':test.key,'score':pred1})
# sub1.to_csv('hckamerAIclass1.csv',index=False)


# 0.96615
# rf=RandomForestRegressor(n_estimators=350,random_state=42,verbose=1,max_features=30,min_samples_split=4)
# rf.fit(x1,y)
# print('predicting')
# pred=rf.predict(test_pr1)
# sub=pd.DataFrame({'key':test.key,'score':pred})
# sub.to_csv('hckamerAIme4.csv',index=False)

# 0.96615
# rf=RandomForestRegressor(n_estimators=400,random_state=42,verbose=1,max_features=30,min_samples_split=6)
# rf.fit(x,y)
# print('predicting')
# pred=rf.predict(test_pr)
# sub=pd.DataFrame({'key':test.key,'score':pred})
# sub.to_csv('hckamerAIme5.csv',index=False)

# 0.96606
# rf=RandomForestRegressor(n_estimators=400,random_state=42,verbose=1,max_features=30,min_samples_split=8)
# rf.fit(x,y)
# print('predicting')
# pred=rf.predict(test_pr)
# sub=pd.DataFrame({'key':test.key,'score':pred})
# sub.to_csv('hckamerAIme6.csv',index=False)
# 0.96596
# rf=RandomForestRegressor(n_estimators=350,random_state=42,verbose=1,max_features=28)
# rf.fit(x,y)
# print('predicting')
# pred=rf.predict(test_pr)
# sub=pd.DataFrame({'key':test.key,'score':pred})
# sub.to_csv('hckamerAIme7.csv',index=False)
#0.96617
# rf=RandomForestRegressor(n_estimators=350,random_state=42,verbose=1,max_features=26)
# rf.fit(x,y)
# print('predicting')
# pred=rf.predict(test_pr)
# sub=pd.DataFrame({'key':test.key,'score':pred})
# sub.to_csv('hckamerAIme8.csv',index=False)
# 0.96614
# rf=RandomForestRegressor(n_estimators=350,random_state=42,verbose=1,max_features=28,max_depth=4000)
# rf.fit(x,y)
# print('predicting')
# pred=rf.predict(test_pr)
# sub=pd.DataFrame({'key':test.key,'score':pred})
# sub.to_csv('hckamerAIme9.csv',index=False)
# 0.96617
# rf=RandomForestRegressor(n_estimators=350,random_state=42,verbose=1,max_features=28,max_depth=2000)
# rf.fit(x,y)
# print('predicting')
# pred=rf.predict(test_pr)
# sub=pd.DataFrame({'key':test.key,'score':pred})
# sub.to_csv('hckamerAIme10.csv',index=False)
# 0.96617
# rf=RandomForestRegressor(n_estimators=350,random_state=42,verbose=1,max_features=28,min_samples_leaf=2)
# rf.fit(x,y)
# print('predicting')
# pred=rf.predict(test_pr)
# sub=pd.DataFrame({'key':test.key,'score':pred})
# sub.to_csv('hckamerAIme13.csv',index=False)
# 0.96602

# rf=RandomForestRegressor(n_estimators=350,random_state=42,verbose=1,max_features=28,min_samples_leaf=10)
# rf.fit(x,y)
# print('predicting')
# pred=rf.predict(test_pr)
# sub=pd.DataFrame({'key':test.key,'score':pred})
# sub.to_csv('hckamerAIme14.csv',index=False)
# 0.96415
rf=RandomForestRegressor(n_estimators=500,random_state=42,verbose=1,max_features=28,min_samples_leaf=5)
rf.fit(x,y)
print('predicting')
pred=rf.predict(test_pr)
sub=pd.DataFrame({'key':test.key,'score':pred})
sub.to_csv('hckamerAIme15.csv',index=False)


rf=RandomForestRegressor(n_estimators=500,random_state=42,verbose=1,max_features=28,min_samples_leaf=5)
rf.fit(x1,y)
print('predicting')
pred=rf.predict(test_pr1)
sub=pd.DataFrame({'key':test.key,'score':pred})
sub.to_csv('hckamerAIme16.csv',index=False)
rf=RandomForestRegressor(n_estimators=350,random_state=42,verbose=1,max_features=28,min_samples_leaf=5)
rf.fit(x,y)
print('predicting')
pred=rf.predict(test_pr)
sub=pd.DataFrame({'key':test.key,'score':pred})
sub.to_csv('hckamerAIme17.csv',index=False)

rf=RandomForestRegressor(n_estimators=400,random_state=42,verbose=1,max_features=28,min_samples_leaf=3)
rf.fit(x,y)
print('predicting')
pred=rf.predict(test_pr)
sub=pd.DataFrame({'key':test.key,'score':pred})
sub.to_csv('hckamerAIme18.csv',index=False)