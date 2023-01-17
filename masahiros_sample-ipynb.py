import pandas as pd
import numpy as np
import os
import matplotlib as mpl
import matplotlib.pyplot as plt
%matplotlib inline
import datetime
import warnings
warnings.filterwarnings('ignore')
mpl.rcParams['font.family'] = 'AppleGothic'
# os.chdir("/Users/admin/Desktop/workspace/call_predict/")
train = pd.read_csv('../input/train_call_history.csv')
def cleaning_data(data_set):    
    data_set['week'] = data_set['week'].map( {'月曜日': "Mon", '火曜日': "Tue", '水曜日': "Wed","木曜日":"Thu","金曜日":"Fri","土曜日":"Sat","日曜日":"Sun"} )
    data_set["establishment"] = data_set["establishment"].astype(str).str[:3]+ "0"
    data_set["call_time"] = data_set["call_time"].astype(str).str[:2]
    data_set["industry_code1"] = data_set["industry_code1"].astype(str).str[:2]
    data_set["birthday"] = data_set["birthday"].astype(str).str[:3] + "0"
    data_set["sogyotoshitsuki"] = data_set["sogyotoshitsuki"].astype(str).str[:3]+ "0"
    data_set["tokikessan_uriagedaka_category"]  = pd.cut(data_set["tokikessan_uriagedaka"] , [0, 1000000, 3000000, 10000000,50000000,999999999999])
    data_set["tokikessan_uriagedaka"] = data_set["tokikessan_uriagedaka"].fillna(data_set["tokikessan_uriagedaka"].mean())
    data_set["shihonkin"] = data_set["shihonkin"].fillna(data_set["shihonkin"].mean())
    data_set["employee_num"] = data_set["employee_num"].fillna(data_set["employee_num"].mean())
    data_set["employee_num"] = data_set["employee_num"].fillna(data_set["employee_num"].mean())
    data_set["tokikessan_riekikin"] = data_set["tokikessan_riekikin"].fillna(0)
    data_set["kojokazu"] = data_set["kojokazu"].fillna(0)
    data_set["jigyoshokazu"] = data_set["jigyoshokazu"].fillna(0)
    data_set["danjokubun"] = data_set["danjokubun"].fillna(1)
    return data_set
train = cleaning_data(train)
#対象となるカテゴリーデータを指定
category_col = ['week', 'call_time', 'list_type','establishment', 'sogyotoshitsuki','tokikessan_uriagedaka_category']
fig, axes = plt.subplots(nrows = len(category_col), ncols = 2, figsize = (80, 120))

for x in category_col:
    freqs = pd.crosstab(train[x],train.result) #出現頻度データを作成
    freqs.plot(ax = axes[category_col.index(x), 0], kind = 'bar', stacked = True) #左カラムにプロット
    axes[category_col.index(x)][0].set_xticklabels(freqs.index, rotation=45, size=12)
    props = freqs.div(freqs.sum(1).astype(float), axis = 0) #出現頻度データから割合（100%）データを作成
    props.plot(ax = axes[category_col.index(x), 1], kind = 'bar', stacked = True) #右カラムにプロット
    axes[category_col.index(x)][1].set_xticklabels(props.index, rotation = 45, size = 12)
    fig.tight_layout()
plt.show()
select_col =["result",'week', 'call_time','list_type','employee_num','tokikessan_uriagedaka','tokikessan_riekikin']
#       ['id', 'result', 'charger_id', 'call_date', 'week', 'call_time','service', 'list_type', 're_call_date', 're_call_time', 'address',
#        'kabushiki_code', 'sogyotoshitsuki', 'establishment', 'shihonkin','employee_num', 'kojokazu', 'jigyoshokazu', 'industry_code1',
#        'industry_code2', 'industry_code3', 'industry1', 'industry2','industry3', 'atsukaihin_code_1', 'atsukaihin_code_2',
#        'atsukaihin_code_3', 'atsukaihin_code_4', 'atsukaihin_code_5','atsukaihin_code_6', 'atsukaihin1', 'atsukaihin2', 'atsukaihin3',
#        'atsukaihin4', 'atsukaihin5', 'atsukaihin6', 'eigyoshumokumeisho','yakuimmeisho', 'okabunushimeisho', 'shiiresakimeisho',
#        'hambaisakimeisho', 'kojo_shiten_eigyoshomeisho', 'gaikyo','maemaekikessan_kessantoshitsuki', 'maemaekikessan_tsukisu',
#        'maemaekikessan_uriagedaka', 'maemaekikessan_zeikomihikikubun','maemaekikessan_riekikin', 'maemaekikessan_uriagedaka_original',
#        'maemaekikessan_riekikin_original', 'zenkikessan_kessantoshitsuki','zenkikessan_tsukisu', 'zenkikessan_uriagedaka',
#        'zenkikessan_zeikomihikikubun', 'zenkikessan_riekikin','zenkikessan_uriagedaka_original', 'zenkikessan_riekikin_original',
#        'tokikessan_kessantoshitsuki', 'tokikessan_tsukisu','tokikessan_uriagedaka', 'tokikessan_zeikomihikikubun',
#        'tokikessan_riekikin', 'tokikessan_uriagedaka_original','tokikessan_riekikin_original', 'tokiuriage_shinchoritsu',
#        'tokiuriage_shinchohitai', 'zenkiuriage_shinchoritsu','zenkiuriage_shinchohitai', 'tokirieki_shinchoritsu',
#        'tokirieki_shinchohitai', 'zenkirieki_shinchoritsu','zenkirieki_shinchohitai', 'hitoriataririgekkan_uriagekinhitai',
#        'hitoriataririgekkan_riekikingaku', 'representative', 'birthday','danjokubun', 'eto_meisho', 'position', 'shusshinchi', 'jukyo',
#        'saishugakureki_gakko', 'shumi1', 'shumi2', 'shumi3', 'tosankeireki','race_area']
data = cleaning_data(train)
data = data[select_col]
data = pd.get_dummies(data)
#Prediction
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, log_loss
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn import cross_validation, metrics
from sklearn.grid_search import GridSearchCV

label = data.columns
data = data.values
X = data[0::, 1::]
y = data[0::, 0]
(X_train, X_test, y_train, y_test) = train_test_split(X, y, test_size=0.3, random_state=8,)
# 決定木の作成
# ROIを最適化する最適なmax_depth と　閾値を探ってみる。
# 一件APが取れた時のLTVが20000円、一件電話をするコストが1000円と過程した時にROIが最大になるmax_depthとThresholdを求めている。
# そもそもグリッドサーチを行えばこのコードは不要。
pre_ROI = 0
pre_roc_auc_score = 0
for i_1 in range(1,10):
    DecisionTree = RandomForestClassifier(random_state=1, max_depth=i_1,class_weight="balanced",max_features=4,n_estimators = 120)
    DecisionTree.fit(X_train, y_train)
    predict_Xtest = DecisionTree.predict_proba(X_test)[:,1]
    now_roc_auc_score = roc_auc_score(y_test, DecisionTree.predict_proba(X_test)[:,1], average='macro', sample_weight=None)
    if pre_roc_auc_score < now_roc_auc_score:
        pre_roc_auc_score = now_roc_auc_score
    for i_2 in range(1,100):
        test = np.where(predict_Xtest >(i_2/100),1,0)
        now_ROI = confusion_matrix(y_test, test)[1,1]*20000 - confusion_matrix(y_test, test)[0::, 1].sum()*1000
        if pre_ROI < now_ROI:
            pre_ROI = now_ROI
            best_depth,best_Threshold = i_1,i_2
        else:
            pass

# 最適な深さと閾値でモデルを作る。
DecisionTree = RandomForestClassifier(random_state=1, max_depth=best_depth,class_weight='balanced',max_features=4,n_estimators = 120)
DecisionTree.fit(X_train, y_train)
predict_Xtest = DecisionTree.predict_proba(X_test)[:,1]
predict_Xtest = np.where(predict_Xtest >(best_Threshold/100),1,0)


# 評価
print('Train score: {:.3f}'.format(DecisionTree.score(X_train, y_train)))
print('Test score: {:.3f}'.format(DecisionTree.score(X_test, y_test)))
print('precision_score: {:.3f}'.format(precision_score(y_test, predict_Xtest)))
print('f1 score: {:.3f}'.format(f1_score(y_test, predict_Xtest)))
print('best_roc_auc_score: {:.3f}'.format(pre_roc_auc_score))
print('roc_auc_score: {:.3f}'.format(roc_auc_score(y_test, DecisionTree.predict_proba(X_test)[:,1], average='macro', sample_weight=None)))
max_ROI = confusion_matrix(y_test, test)[1:,].sum()*19000
print("=======================")
print('ROI: {:,}円 /最大ROI:{:,}円'.format(pre_ROI,max_ROI))
print('比率： {:.1f} %'.format(pre_ROI/max_ROI*100))
print("=======================")
print('Confusion matrix:\n{}'.format(confusion_matrix(y_test, predict_Xtest)))
print("=======================")
print('best_depth: {}'.format(best_depth))
print('best_Threshold: {}'.format(best_Threshold/100))
importance = DecisionTree.feature_importances_
data = pd.DataFrame({"name":label[1:],"importance":importance})
data = data.sort_values('importance', ascending=False)
fig, axes = plt.subplots(figsize = (20, 10))
sns.barplot(x=data['importance'], y=data['name'],color="b")
# #Grid search

# from sklearn.grid_search import GridSearchCV

#  # use a full grid over all parameters
# param_grid = {"max_depth": [1,5,10],
#                "n_estimators":[90,120,150],
#                "max_features": [1,2,3,4,5],
# #                "min_samples_split": [2, 3],
# #                "min_samples_leaf": [1, 3],
#               "bootstrap": [True, False],
#               "class_weight":["balanced"],
#               "criterion": ["gini", "entropy"]}

# forest_grid = GridSearchCV(estimator=RandomForestClassifier(random_state=0),
#                  param_grid = param_grid,   
#                  scoring="roc_auc",  #metrics
#                  cv = 3,              #cross-validation
#                  n_jobs = 1)          #number of core

# forest_grid.fit(X_train,y_train) #fit

# forest_grid_best = forest_grid.best_estimator_ #best estimator
# print("Best Model Parameter: ",forest_grid.best_params_)
# test = pd.read_csv("test_call_history.csv")
# test_ID = test["id"]
# test_data = cleaning_data(test)
# test_select_col = select_col.copy()
# test_select_col.remove("result")
# # ここは本気でイケてないので、真似しないこと。
# # 何をしているかというと、ダミー変数をpd.get_dummies　で作った時にtestデータにしかない。trainデータにしかないカテゴリーができてしまったので強引に調整している。
# test_data = test_data[test_select_col]
# test_data = pd.get_dummies(test_data)
# test_data = test_data.drop(["call_time_01","call_time_06"],axis = 1)
# test_data["call_time_24"] = 0
# test_data = test_data[['employee_num', 'tokikessan_uriagedaka',
#        'tokikessan_riekikin', 'call_time_07', 'call_time_08', 'call_time_09',
#        'call_time_10', 'call_time_11', 'call_time_12', 'call_time_13',
#        'call_time_14', 'call_time_15', 'call_time_16', 'call_time_17',
#        'call_time_18', 'call_time_19', 'call_time_20', 'call_time_21',
#        'call_time_22', 'call_time_23', 'call_time_24', 'list_type_源泉',
#        'list_type_管S']]
# test_data_label = test_data.columns
# test_data = test_data.values


# predict_test_data = DecisionTree.predict_proba(test_data)
# x = pd.DataFrame({'id':test_ID,"result":predict_test_data[:,1]})
# x = x.set_index('id')
# x.to_csv("predict.csv")





