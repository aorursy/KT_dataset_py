# Data files
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
# Import
%matplotlib inline
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns 

from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost.sklearn import XGBClassifier 
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
credit = pd.read_csv('/kaggle/input/creditcardfraud/creditcard.csv')
credit
del credit['Time']
del credit['Amount']

train_feature = credit.drop(columns='Class')
train_target = credit['Class']

# Normalization of train data
from sklearn.preprocessing import MinMaxScaler
mmsc = MinMaxScaler()
train_feature = mmsc.fit_transform(train_feature)
print(train_feature.max())
print(train_feature.min())
X_train, X_test, y_train, y_test = train_test_split(train_feature, train_target, test_size=0.2, random_state=0, shuffle=True)
# Searching for Effective feature
'''
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

print(train_feature.columns)
print(mask_SelectKBest)
print(mask_SelectPercentile)
print(mask_SelectFromModel)
print(mask_RFE)
'''
# RandomForest==============

rf = RandomForestClassifier()
rf.fit(X_train, y_train)
print('='*20)
print('RandomForestClassifier')
print(f'accuracy of train set: {rf.score(X_train, y_train)}')
print(f'accuracy of test set: {rf.score(X_test, y_test)}')

# SVC==============

svc = SVC(verbose=True, random_state=0)
svc.fit(X_train, y_train)
print('='*20)
print('SVC')
print(f'accuracy of train set: {svc.score(X_train, y_train)}')
print(f'accuracy of test set: {svc.score(X_test, y_test)}')


# LinearSVC==============

lsvc = LinearSVC(verbose=True)
lsvc.fit(X_train, y_train)
print('='*20)
print('LinearSVC')
print(f'accuracy of train set: {lsvc.score(X_train, y_train)}')
print(f'accuracy of test set: {lsvc.score(X_test, y_test)}')

# k-近傍法（k-NN）==============

knn = KNeighborsClassifier(n_neighbors=3) #引数は分類数
knn.fit(X_train, y_train)
print('='*20)
print('KNeighborsClassifier')
print(f'accuracy of train set: {knn.score(X_train, y_train)}')
print(f'accuracy of test set: {knn.score(X_test, y_test)}')


# 決定木==============

decisiontree = DecisionTreeClassifier(max_depth=3, random_state=0)
decisiontree.fit(X_train, y_train)
print('='*20)
print('DecisionTreeClassifier')
print(f'accuracy of train set: {decisiontree.score(X_train, y_train)}')
print(f'accuracy of test set: {decisiontree.score(X_test, y_test)}')


# SGD Classifier==============

sgd = SGDClassifier()
sgd.fit(X_train, y_train)
print('='*20)
print('SGD Classifier')
print(f'accuracy of train set: {sgd.score(X_train, y_train)}')
print(f'accuracy of test set: {sgd.score(X_test, y_test)}')


# XGBClassifier==============

xgb = XGBClassifier()
xgb.fit(X_train, y_train)
print('='*20)
print('XGB Classifier')
print(f'accuracy of train set: {xgb.score(X_train, y_train)}')
print(f'accuracy of test set: {xgb.score(X_test, y_test)}')

# LGBMClassifier==============

lgbm = LGBMClassifier()
lgbm.fit(X_train, y_train)
print('='*20)
print('LGBM Classifier')
print(f'accuracy of train set: {lgbm.score(X_train, y_train)}')
print(f'accuracy of test set: {lgbm.score(X_test, y_test)}')

# CatBoostClassifier==============

catboost = CatBoostClassifier()
catboost.fit(X_train, y_train)
print('='*20)
print('CatBoost Classifier')
print(f'accuracy of train set: {catboost.score(X_train, y_train)}')
print(f'accuracy of test set: {catboost.score(X_test, y_test)}')
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Output expected data(y_pred) from X_test
y_pred = rf.predict(X_test)

print('Correct answer data (y_test) : ' + str(y_test))
print('Expected data(y_pred) : ' + str(y_pred))
print('\n======Creating Matrix======')

# The first argument is the actual class (the correct class) and the second argument is a list or array of predicted classes
print(confusion_matrix(y_test, y_pred))
matrix = confusion_matrix(y_test, y_pred)
print(type(matrix))

# Change table easier to read using Pandas
class_names = ["0","1"]
df = pd.DataFrame(matrix, index=class_names, columns=class_names)
df

# accuracy_score for multi-class classification
print('Accuracy_score:{:.3f}'.format(accuracy_score(y_test, y_pred)))

# Visualization of Matrix
import seaborn as sns
plt.figure(figsize=(15, 10)) 
sns.heatmap(matrix, annot = True, cmap = 'Blues')
plt.savefig('sklearn_confusion_matrix.png')

from sklearn.metrics import classification_report
import pprint
print(classification_report(y_test, y_pred, target_names=["0","1"]))
repo = classification_report(y_test, y_pred, output_dict=True)
pprint.pprint(repo)
print(repo['0'])
print(repo['0']['precision'])
print(type(repo['0']['precision']))
df = pd.DataFrame(repo)
df
from sklearn.metrics import plot_confusion_matrix
np.set_printoptions(precision=2)

# Title of the visualization and designate how to normalize
titles_options = [
        ("Confusion matrix, without normalization", None),
        ("Normalized confusion matrix: true", 'true'),
        ("Normalized confusion matrix: pred", 'pred'),
        ("Normalized confusion matrix: all", 'all'),
    ]

fig = plt.figure(figsize=(20, 20), facecolor="w")
fig.subplots_adjust(hspace=0.2, wspace=0.4)
i = 0
for title, normalize in titles_options:
    i += 1
    ax = fig.add_subplot(2, 2, i)
    disp = plot_confusion_matrix(
                        rf,
                        X_test,
                        y_test,
                        display_labels=class_names,
                        cmap=plt.cm.Blues,
                        normalize=normalize,
                        ax=ax,
                    )

    # Showing title on Pic
    disp.ax_.set_title(title)

    print(title)
    print(disp.confusion_matrix)
plt.show()
