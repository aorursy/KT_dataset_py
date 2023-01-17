# Data file
import numpy as np
import pandas as pd
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
# Import
%matplotlib inline
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns 

from sklearn.svm import SVR
from sklearn.svm import LinearSVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor #  KneighborsRegressorではない
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_validate
from sklearn.model_selection import KFold
from sklearn.linear_model import RANSACRegressor
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import LassoCV
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import SGDRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.ensemble import VotingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from xgboost.sklearn import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
train = pd.read_csv('/kaggle/input/restaurant-revenue-prediction/train.csv.zip', header=0)
train.head(10)
test = pd.read_csv('/kaggle/input/restaurant-revenue-prediction/test.csv.zip')
test.head(10)
submission = pd.read_csv('/kaggle/input/restaurant-revenue-prediction/sampleSubmission.csv')
submission.head(10)
train_mid = train.copy()
train_mid['train_or_test'] = 'train'

test_mid = test.copy()
test_mid['train_or_test'] = 'test'

test_mid['revenue'] = 9

alldata = pd.concat([train_mid, test_mid], sort=False, axis=0).reset_index(drop=True)

print('The size of the train data:' + str(train.shape))
print('The size of the test data:' + str(test.shape))
print('The size of the submission data:' + str(submission.shape))
print('The size of the alldata data:' + str(alldata.shape))
train.describe()
test.describe()
alldata.describe()

print('=====Train=====')
train.info()
print('\n=====Test=====')
test.info()
# Check for duplicates
idsUnique = len(set(alldata['Id']))
idsTotal = alldata.shape[0]
idsDupli = idsTotal - idsUnique
print("There are " + str(idsDupli) + " duplicate IDs for " + str(idsTotal) + " total entries")

# Missing data in Alldata
def Missing_table(df):
    # null_val = df.isnull().sum()
    null_val = df.isnull().sum()[df.isnull().sum()>0].sort_values(ascending=False)
    percent = 100 * null_val/len(df)
    na_col_list = df.isnull().sum()[df.isnull().sum()>0].index.tolist() # 欠損を含むカラムをリスト化
    list_type = df[na_col_list].dtypes.sort_values(ascending=False) #データ型
    Missing_table = pd.concat([null_val, percent, list_type], axis = 1)
    missing_table_len = Missing_table.rename(
    columns = {0:'Missing data', 1:'%', 2:'type'})
    return missing_table_len.sort_values(by=['Missing data'], ascending=False)

Missing_table(alldata)
# EDA
# Histogram
alldata.hist(figsize = (12,12))
# Heatmap. Understand feature related to survived
fig, ax = plt.subplots(figsize=(15,15))
sns.heatmap(train.corr(),annot=True, center=0, square=True, linewidths=0.1, vmax=1.0, linecolor='white', cmap="RdBu")
plt.title('Restaurant Revenue Prediction', fontsize = 20)
plt.xlabel('x-axis', fontsize = 15)
plt.ylabel('y-axis', fontsize = 15)
alldata['City'].value_counts()
alldata['City Group'].value_counts()
alldata['Type'].value_counts()
alldata['City'].replace(['İstanbul', 'Ankara', 'İzmir'], 0,inplace=True)
alldata['City'].replace(['Bursa', 'Samsun', 'Antalya', 'Sakarya', 'Kayseri', 'Diyarbakır', 'Tekirdağ', 'Eskişehir', 'Adana', 'Aydın', 'Muğla', 'Konya', 'Trabzon', 'Amasya', 'Uşak', 'Kastamonu', 'Karabük', 'Kütahya', 'Bolu', 'Şanlıurfa', 'Edirne', 'Kırklareli', 'Afyonkarahisar', 'Osmaniye', 'Denizli', 'Tokat', 'Balıkesir', 'Gaziantep', 'Kocaeli', 'Elazığ', 'Isparta', 'Mersin', 'Manisa', 'Çanakkale', 'Hatay', 'Zonguldak', 'Aksaray', 'Yalova', 'Kırıkkale', 'Malatya', 'Mardin', 'Batman', 'Rize', 'Artvin', 'Bilecik', 'Nevşehir', 'Sivas', 'Kırşehir', 'Erzincan', 'Erzurum', 'Ordu', 'Kahramanmaraş', 'Siirt', 'Niğde', 'Giresun', 'Çankırı', 'Çorum', 'Düzce', 'Tanımsız', 'Kars'], 1,inplace=True)

alldata['City Group'] = alldata['City Group'].replace("Big Cities",0).replace("Other",1)

alldata['Type'] = alldata['Type'].replace("FC",0).replace("IL",1).replace("DT",2).replace("MB",3)

alldata["Open Date"] = pd.to_datetime(alldata["Open Date"])
alldata["Year"] = alldata["Open Date"].apply(lambda x:x.year)
alldata["Month"] = alldata["Open Date"].apply(lambda x:x.month)
alldata["Day"] = alldata["Open Date"].apply(lambda x:x.day)
alldata["kijun"] = "2015-04-27"
alldata["kijun"] = pd.to_datetime(alldata["kijun"])
alldata["BusinessPeriod"] = (alldata["kijun"] - alldata["Open Date"]).apply(lambda x: x.days)

alldata = alldata.drop('Open Date', axis=1)
alldata = alldata.drop('kijun', axis=1)

alldata
# Check all of datatype
alldata.dtypes
train = alldata.query('train_or_test == "train"')
test = alldata.query('train_or_test == "test"')

target_column = 'revenue'
train_target = train[target_column]

train_target
drop_column = ['Id', 'train_or_test', 'revenue']
train_feature = train.drop(columns=drop_column)

train_feature
# Before deleting the Id column of test data, extract only the Id column used for output. The first time I merged train and test, the first index started at 137 and the index is off by that amount. It's not a problem in the final output, but I don't like the way it looks, so I'll just reindex it
test
# Index reset
test = test.reset_index()
# Delete unnecessary index column
del test["index"]
test
# Idカラムを、submission_idとしてだけ抜き出しておく
submission_id = test['Id']

# 最後のテスト出力用の説明変数データを作成。学習データとカラムを合わせて、Id, train_or_test, 9のデータが入っているrevenueを削除し、学習に必要な特徴量のみを保持
test_feature = test.drop(columns=drop_column)
test_feature

# 有効な特微量を探す（SelectKBestの場合）
from sklearn.feature_selection import SelectKBest, f_regression
# 特に重要な4つの特徴量のみを探すように設定してみる
selector = SelectKBest(score_func=f_regression, k=4) 
selector.fit(train_feature, train_target)
mask_SelectKBest = selector.get_support()    # 各特徴量を選択したか否かのmaskを取得

# 有効な特微量を探す（SelectPercentileの場合）
from sklearn.feature_selection import SelectPercentile, f_regression
# 特徴量のうち40%を選択
selector = SelectPercentile(score_func=f_regression, percentile=40) 
selector.fit(train_feature, train_target)
mask_SelectPercentile = selector.get_support()

# 有効な特微量を探す（モデルベース選択の場合：SelectFromModel）
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestRegressor
# estimator として RandomForestRegressor を使用。重要度が median 以上のものを選択
selector = SelectFromModel(RandomForestRegressor(n_estimators=100, random_state=42), threshold="median")    
selector.fit(train_feature, train_target)
mask_SelectFromModel = selector.get_support()

# 有効な特微量を探す（RFE：再帰的特徴量削減 : n_features_to_select）
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestRegressor
# estimator として RandomForestRegressor を使用。特徴量を2個選択させる
selector = RFE(RandomForestRegressor(n_estimators=100, random_state=42), n_features_to_select=2)
selector.fit(train_feature, train_target)
mask_RFE = selector.get_support()

print(train.columns)
print(mask_SelectKBest)
print(mask_SelectPercentile)
print(mask_SelectFromModel)
print(mask_RFE)

important_feature = pd.DataFrame({"Index":train.columns[1], "SelectKBest":mask_SelectKBest, "SelectPercentile":mask_SelectPercentile, "SelectFromModelKBest":mask_SelectFromModel, "RFE":mask_RFE})
important_feature.to_csv("important_feature.csv", index=False)

# 読み込む
result = pd.read_csv('important_feature.csv')
result.head(50)
# 新しいカラムを作成して合計のTrue数を記載する。その後ソートで表示する
result["Total_True_Number"] = result.sum(axis=1)
result.sort_values('Total_True_Number', ascending = False)

# ホールドアウト法で検証するため、あらかじめデータを学習用と検証用に分割
X_train, X_test, y_train, y_test = train_test_split(train_feature, train_target, test_size=0.2, random_state=0, shuffle=True)

# 警告が多いので、いったん警告を表示されないようにする
# 本来は表示を消すのはお勧めしない。廃止予定の関数や例外が表示されるほうが良い
import warnings
warnings.filterwarnings('ignore')


# RandomForest==============

rf = RandomForestRegressor(n_estimators=200, max_depth=5, max_features=0.5,  verbose=True, random_state=0, n_jobs=-1) # RandomForest のオブジェクトを用意する
rf.fit(X_train, y_train)
print('='*20)
print('RandomForestRegressor')
print(f'accuracy of train set: {rf.score(X_train, y_train)}')
print(f'accuracy of test set: {rf.score(X_test, y_test)}')

# 学習させたRandomForestをtestデータに適用して、売上を予測しましょう
rf_prediction = rf.predict(test_feature)
rf_prediction

'''
# Create submission data
rf_submission = pd.DataFrame({"Id":submission_id, "Prediction":rf_prediction})
rf_submission.to_csv("RandomForest_submission.csv", index=False)
'''

# SVR（Support Vector Regression）==============
# ※[LibSVM]や[LibLinear]は台湾国立大学の方で開発されたらしくどうしてもその表示が入るようになっている

svr = SVR(verbose=True)
svr.fit(X_train, y_train)
print('='*20)
print('SVR')
print(f'accuracy of train set: {svr.score(X_train, y_train)}')
print(f'accuracy of test set: {svr.score(X_test, y_test)}')

svr_prediction = svr.predict(test_feature)
svr_prediction

# LinearSVR==============

lsvr = LinearSVR(verbose=True, random_state=0)
lsvr.fit(X_train, y_train)
print('='*20)
print('LinearSVR')
print(f'accuracy of train set: {lsvr.score(X_train, y_train)}')
print(f'accuracy of test set: {lsvr.score(X_test, y_test)}')

lsvr_prediction = lsvr.predict(test_feature)
lsvr_prediction

# SGDRegressor==============

sgd = SGDRegressor(verbose=0, random_state=0)
sgd.fit(X_train, y_train)
print('='*20)
print('SGDRegressor')
print(f'accuracy of train set: {sgd.score(X_train, y_train)}')
print(f'accuracy of test set: {sgd.score(X_test, y_test)}')

sgd_prediction = sgd.predict(test_feature)
sgd_prediction

# k-近傍法（k-NN）==============

knn = KNeighborsRegressor()
knn.fit(X_train, y_train)
print('='*20)
print('KNeighborsRegressor')
print(f'accuracy of train set: {knn.score(X_train, y_train)}')
print(f'accuracy of test set: {knn.score(X_test, y_test)}')

knn_prediction = knn.predict(test_feature)
knn_prediction

# 決定木==============

decisiontree = DecisionTreeRegressor(max_depth=3, random_state=0)
decisiontree.fit(X_train, y_train)
print('='*20)
print('DecisionTreeRegressor')
print(f'accuracy of train set: {decisiontree.score(X_train, y_train)}')
print(f'accuracy of test set: {decisiontree.score(X_test, y_test)}')

decisiontree_prediction = decisiontree.predict(test_feature)
decisiontree_prediction

# LinearRegression (線形回帰)==============

lr = LinearRegression()
lr.fit(X_train, y_train)
print('='*20)
print('LinearRegression')
print(f'accuracy of train set: {lr.score(X_train, y_train)}')
print(f'accuracy of test set: {lr.score(X_test, y_test)}')
# 回帰係数とは、回帰分析において座標平面上で回帰式で表される直線の傾き。 原因となる変数x（説明変数）と結果となる変数y（目的変数）の平均的な関係を、一次式y＝ax＋bで表したときの、係数aを指す。
print("回帰係数:",lr.coef_)
print("切片:",lr.intercept_)

lr_prediction = lr.predict(test_feature)
lr_prediction


# RANSACRegressor==============

# ロバスト回帰を行う（自然界のデータにはたくさんノイズがある。ノイズなどの外れ値があると、法則性をうまく見つけられないことがある。そんなノイズをうまく無視してモデルを学習させるのがRANSAC）
#線形モデルをRANSACでラッピング　（外れ値の影響を抑える）
from sklearn.linear_model import RANSACRegressor
 
ransac=RANSACRegressor(lr,#基本モデルは、LinearRegressionを流用
                       max_trials=100,#イテレーションの最大数100
                       min_samples=50,#ランダムに選択されるサンプル数を最低50に設定
                       loss="absolute_loss",#学習直線に対するサンプル店の縦の距離の絶対数を計算
                       residual_threshold=5.0,#学習直線に対する縦の距離が5以内のサンプルだけを正常値
                       random_state=0)
 
ransac.fit(X_train, y_train)
print('='*20)
print('RANSACRegressor')
print(f'accuracy of train set: {lr.score(X_train, y_train)}')
print(f'accuracy of test set: {lr.score(X_test, y_test)}')
print("RANSAC回帰係数:",ransac.estimator_.coef_[0])
print("RANSAC切片:",ransac.estimator_.intercept_)

ransac_prediction = ransac.predict(test_feature)
ransac_prediction

# RIDGE回帰==============

ridge = Ridge(random_state=0)
ridge.fit(X_train, y_train)
print('='*20)
print('Ridge')
print(f'accuracy of train set: {ridge.score(X_train, y_train)}')
print(f'accuracy of test set: {ridge.score(X_test, y_test)}')

ridge_prediction = ridge.predict(test_feature)
ridge_prediction

ridge_submission = pd.DataFrame({"Id":submission_id, "Prediction":ridge_prediction})
ridge_submission.to_csv("Ridge_submission.csv", index=False)



# LASSO回帰==============

lasso = LassoCV(alphas = [1, 0.1, 0.001, 0.0005], verbose=True, random_state=0)
lasso.fit(X_train, y_train)
print('='*20)
print('LassoCV')
print(f'accuracy of train set: {lasso.score(X_train, y_train)}')
print(f'accuracy of test set: {lasso.score(X_test, y_test)}')

lasso_prediction = lasso.predict(test_feature)
lasso_prediction


# ElasticNet==============

en = ElasticNet(random_state=0)
en.fit(X_train, y_train)
print('='*20)
print('ElasticNet')
print(f'accuracy of train set: {en.score(X_train, y_train)}')
print(f'accuracy of test set: {en.score(X_test, y_test)}')

en_prediction = en.predict(test_feature)
en_prediction

# Kernel Ridge Regression(l2制約付き最小二乗学習)==============

kernelridge = KernelRidge(alpha=0.6, kernel='polynomial', degree=2, coef0=2.5)
kernelridge.fit(X_train, y_train)
print('='*20)
print('KernelRidge')
print(f'accuracy of train set: {kernelridge.score(X_train, y_train)}')
print(f'accuracy of test set: {kernelridge.score(X_test, y_test)}')

kernelridge_prediction = kernelridge.predict(test_feature)
kernelridge_prediction


# Gradient Boosting Regression==============
# Boostingとは弱学習器をたくさん集めて強学習器を作ろうという話が出発点で、PAC Learningと呼ばれています

gradientboost = GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05, max_depth=4, max_features='sqrt', min_samples_leaf=15, min_samples_split=10, loss='huber', verbose=0, random_state=0)
gradientboost.fit(X_train, y_train)
print('='*20)
print('GradientBoostingRegressor')
print(f'accuracy of train set: {gradientboost.score(X_train, y_train)}')
print(f'accuracy of test set: {gradientboost.score(X_test, y_test)}')

gradientboost_prediction = gradientboost.predict(test_feature)
gradientboost_prediction


# XGB==============

xgb = XGBRegressor(objective ='reg:squarederror', verbose=True, random_state=0)  
xgb.fit(X_train, y_train) 
print('='*20)
print('XGBClassifier')
print(f'accuracy of train set: {xgb.score(X_train, y_train)}')
print(f'accuracy of test set: {xgb.score(X_test, y_test)}')

xgb_prediction = xgb.predict(test_feature)
xgb_prediction


# lightgbm==============

lgbm = LGBMRegressor(random_state=0)
lgbm.fit(X_train, y_train)
print('='*20)
print('LGBMRegressor')
print(f'accuracy of train set: {lgbm.score(X_train, y_train)}')
print(f'accuracy of test set: {lgbm.score(X_test, y_test)}')

lgbm_prediction = lgbm.predict(test_feature)
lgbm_prediction


# catboost==============

catboost = CatBoostRegressor(verbose=0, random_state=0)
catboost.fit(X_train, y_train)
print('='*20)
print('CatBoostRegressor')
print(f'accuracy of train set: {catboost.score(X_train, y_train)}')
print(f'accuracy of test set: {catboost.score(X_test, y_test)}')

catboost_prediction = catboost.predict(test_feature)
catboost_prediction


# VotingRegressor==============

# voting に使う分類器を用意する
estimators = [
  ("rf", rf),
  ("svr", svr),
  ("lsvr", lsvr),
  ("sgd", sgd),
  ("knn", knn),
  ("decisiontree", decisiontree),
  ("lr", lr),
  ("ransac", ransac),
  ("ridge", ridge),
  ("lasso", lasso),
  ("en", en),
  ("kernelridge", kernelridge),
  ("gradientboost", gradientboost),
  ("xgb", xgb),
  ("lgbm", lgbm),
  ("catboost", catboost),
]

vote = VotingRegressor(estimators=estimators)
vote.fit(X_train, y_train)
print('='*20)
print('VotingRegressor')
print(f'accuracy of train set: {vote.score(X_train, y_train)}')
print(f'accuracy of test set: {vote.score(X_test, y_test)}')

vote_prediction = vote.predict(test_feature)
vote_prediction


# ※重要な特微量を探す（RandomForestを利用する）
plt.figure(figsize=(20,10))
plt.barh(
    X_train.columns[np.argsort(rf.feature_importances_)],
    rf.feature_importances_[np.argsort(rf.feature_importances_)],
    label='RandomForestRegressor'
)
plt.title('RandomForestRegressor feature importance')

# ※重要な特微量を探す（決定木やXGBを利用する）


from sklearn import tree
text_representation = tree.export_text(decisiontree)
print(text_representation)

with open("decistion_tree.log", "w") as fout:
    fout.write(text_representation)


fig = plt.figure(figsize=(25,20))
_ = tree.plot_tree(decisiontree, 
                   feature_names=X_train.columns,  
                   class_names=target_column,
                   filled=True)
fig.savefig("decistion_tree.png")

import graphviz
# DOT data
dot_data = tree.export_graphviz(decisiontree, out_file=None, 
                                feature_names=X_train.columns,  
                                class_names=target_column,
                                filled=True)

# Draw graph
graph = graphviz.Source(dot_data, format="png") 
graph
graph.render("decision_tree_graphivz")
'decision_tree_graphivz.png'

# 重要度を出力
for n, v in zip(X_train.columns, decisiontree.feature_importances_):
    print(f'importance of {n} is :{v}')


import warnings
warnings.filterwarnings('ignore')

# RandomForest==============

kf = KFold(n_splits = 5, shuffle = True, random_state=0)

rf = RandomForestRegressor(n_estimators=200, max_depth=5, max_features=0.5,  verbose=True, random_state=0, n_jobs=-1) # RandomForest のオブジェクトを用意する
rf_cross_score = cross_validate(rf, train_feature, train_target, cv=kf)
rf_cross_score
print('='*20)
print('RandomForestRegressor 交差検証(Cross-validation)')
print(f'平均値 mean:{rf_cross_score["test_score"].mean()}, 標準偏差 std:{rf_cross_score["test_score"].std()}')
print("交差検証トレーニングのscore:",format(rf_cross_score))
#print("交差検証テストのscore:",format(np.mean(rf_cross_score)))

# SVR（Support Vector Regression）==============
# ※[LibSVM]や[LibLinear]は台湾国立大学の方で開発されたらしくどうしてもその表示が入るようになっている

svr = SVR(verbose=True)
svr_cross_score = cross_validate(svr, train_feature, train_target, cv=kf)
print('='*20)
print('SVR 交差検証(Cross-validation)')
print(f'平均値 mean:{svr_cross_score["test_score"].mean()}, 標準偏差 std:{svr_cross_score["test_score"].std()}')

# LinearSVR==============

lsvr = LinearSVR(verbose=True, random_state=0)
lsvr_cross_score = cross_validate(lsvr, train_feature, train_target, cv=kf)
print('='*20)
print('LinearSVR 交差検証(Cross-validation)')
print(f'平均値 mean:{lsvr_cross_score["test_score"].mean()}, 標準偏差 std:{lsvr_cross_score["test_score"].std()}')


# SGDRegressor==============

sgd = SGDRegressor(verbose=0, random_state=0)
sgd_cross_score = cross_validate(sgd, train_feature, train_target, cv=kf)
print('='*20)
print('SGDRegressor 交差検証(Cross-validation)')
print(f'平均値 mean:{sgd_cross_score["test_score"].mean()}, 標準偏差 std:{sgd_cross_score["test_score"].std()}')


# k-近傍法（k-NN）==============

knn = KNeighborsRegressor()
knn_cross_score = cross_validate(knn, train_feature, train_target, cv=kf)
print('='*20)
print('KNeighborsRegressor 交差検証(Cross-validation)')
print(f'平均値 mean:{knn_cross_score["test_score"].mean()}, 標準偏差 std:{knn_cross_score["test_score"].std()}')



# 決定木==============

decisiontree = DecisionTreeRegressor(max_depth=3, random_state=0)
decisiontree_cross_score = cross_validate(decisiontree, train_feature, train_target, cv=kf)
print('='*20)
print('DecisionTreeRegressor 交差検証(Cross-validation)')
print(f'平均値 mean:{decisiontree_cross_score["test_score"].mean()}, 標準偏差 std:{decisiontree_cross_score["test_score"].std()}')




# LinearRegression (線形回帰)==============

lr = LinearRegression()
lr_cross_score = cross_validate(lr, train_feature, train_target, cv=kf)
print('='*20)
print('LinearRegression 交差検証(Cross-validation)')
print(f'平均値 mean:{lr_cross_score["test_score"].mean()}, 標準偏差 std:{lr_cross_score["test_score"].std()}')


# RANSACRegressor==============

# ロバスト回帰を行う（自然界のデータにはたくさんノイズがある。ノイズなどの外れ値があると、法則性をうまく見つけられないことがある。そんなノイズをうまく無視してモデルを学習させるのがRANSAC）
#線形モデルをRANSACでラッピング　（外れ値の影響を抑える）
from sklearn.linear_model import RANSACRegressor
 
ransac=RANSACRegressor(lr,#基本モデルは、LinearRegressionを流用
                       max_trials=100,#イテレーションの最大数100
                       min_samples=50,#ランダムに選択されるサンプル数を最低50に設定
                       loss="absolute_loss",#学習直線に対するサンプル店の縦の距離の絶対数を計算
                       residual_threshold=5.0,#学習直線に対する縦の距離が5以内のサンプルだけを正常値
                       random_state=0)
 
ransac_cross_score = cross_validate(ransac, train_feature, train_target, cv=kf)
print('='*20)
print('RANSACRegressor 交差検証(Cross-validation)')
print(f'平均値 mean:{ransac_cross_score["test_score"].mean()}, 標準偏差 std:{ransac_cross_score["test_score"].std()}')


# RIDGE回帰==============

ridge = Ridge(random_state=0)
ridge_cross_score = cross_validate(ridge, train_feature, train_target, cv=kf)
print('='*20)
print('Ridge 交差検証(Cross-validation)')
print(f'平均値 mean:{ridge_cross_score["test_score"].mean()}, 標準偏差 std:{ridge_cross_score["test_score"].std()}')




# LASSO回帰==============

lasso = LassoCV(alphas = [1, 0.1, 0.001, 0.0005], verbose=True, random_state=0)
lasso_cross_score = cross_validate(lasso, train_feature, train_target, cv=kf)
print('='*20)
print('LassoCV 交差検証(Cross-validation)')
print(f'平均値 mean:{lasso_cross_score["test_score"].mean()}, 標準偏差 std:{lasso_cross_score["test_score"].std()}')


# ElasticNet==============

en = ElasticNet(random_state=0)
en_cross_score = cross_validate(en, train_feature, train_target, cv=kf)
print('='*20)
print('ElasticNet 交差検証(Cross-validation)')
print(f'平均値 mean:{en_cross_score["test_score"].mean()}, 標準偏差 std:{en_cross_score["test_score"].std()}')


# Kernel Ridge Regression(l2制約付き最小二乗学習)==============

kernelridge = KernelRidge(alpha=0.6, kernel='polynomial', degree=2, coef0=2.5)
kernelridge_cross_score = cross_validate(kernelridge, train_feature, train_target, cv=kf)
print('='*20)
print('KernelRidge 交差検証(Cross-validation)')
print(f'平均値 mean:{kernelridge_cross_score["test_score"].mean()}, 標準偏差 std:{kernelridge_cross_score["test_score"].std()}')


# Gradient Boosting Regression==============
# Boostingとは弱学習器をたくさん集めて強学習器を作ろうという話が出発点で、PAC Learningと呼ばれています

gradientboost = GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05, max_depth=4, max_features='sqrt', min_samples_leaf=15, min_samples_split=10, loss='huber', verbose=0, random_state=0)
gradientboost_cross_score = cross_validate(gradientboost, train_feature, train_target, cv=kf)
print('='*20)
print('GradientBoostingRegressor 交差検証(Cross-validation)')
print(f'平均値 mean:{gradientboost_cross_score["test_score"].mean()}, 標準偏差 std:{gradientboost_cross_score["test_score"].std()}')


# XGB==============

xgb = XGBRegressor(objective ='reg:squarederror', verbose=True, random_state=0)  
xgb_cross_score = cross_validate(xgb, train_feature, train_target, cv=kf)
print('='*20)
print('XGBClassifier 交差検証(Cross-validation)')
print(f'平均値 mean:{xgb_cross_score["test_score"].mean()}, 標準偏差 std:{xgb_cross_score["test_score"].std()}')


# lightgbm==============

lgbm = LGBMRegressor(random_state=0)
lgbm_cross_score = cross_validate(lgbm, train_feature, train_target, cv=kf)
print('='*20)
print('LGBMRegressor 交差検証(Cross-validation)')
print(f'平均値 mean:{lgbm_cross_score["test_score"].mean()}, 標準偏差 std:{lgbm_cross_score["test_score"].std()}')


# catboost==============

catboost = CatBoostRegressor(verbose=0, random_state=0)
catboost_cross_score = cross_validate(catboost, train_feature, train_target, cv=kf)
print('='*20)
print('CatBoostRegressor 交差検証(Cross-validation)')
print(f'平均値 mean:{catboost_cross_score["test_score"].mean()}, 標準偏差 std:{catboost_cross_score["test_score"].std()}')

