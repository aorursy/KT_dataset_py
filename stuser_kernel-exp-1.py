import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.family']='SimHei' #顯示中文

%matplotlib inline
import warnings
warnings.filterwarnings('ignore')
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input/input-iris"))

# Any results you write to the current directory are saved as output.
# Load in the train datasets
train = pd.read_csv('../input/input-iris/train.csv', encoding = "utf-8", dtype = {'type': np.int32})
test = pd.read_csv('../input/input-iris/test.csv', encoding = "utf-8")
submission = pd.read_csv('../input/input-iris/submission.csv', encoding = "utf-8", dtype = {'type': np.int32})
train.head(3)
test.head(3)
submission.head(3)
df1 = pd.get_dummies(train['屬種'])
df1.sample(5)
df2 = train['屬種'].replace({'Iris-setosa':1,'Iris-versicolor':2,'Iris-virginica':3})
df2.sample(5)
#missing data
miss_sum = train.isnull().sum().sort_values(ascending=False)
miss_sum
#查詢那幾筆是空值
print(train[train['花萼寬度'].isnull()])
print("--------------------------------")
print(train[train['花萼長度'].isnull()])
#直接把 NaN drop (如果筆數很少,不影響建模的時候)
train_d_na = train.dropna().reset_index(drop=True)
train_d_na.isnull().sum().sort_values(ascending=False)
#將空值補平均數
#train.loc[train['花萼寬度'].isnull(),['花萼寬度']] = train['花萼寬度'].mean() #花萼寬度:第2欄
train[['花萼寬度']] = train[['花萼寬度']].fillna(np.mean(train[['花萼寬度']]))

train.plot(kind='line',y='花萼寬度',figsize=(10,6),fontsize=14,title='花萼寬度')
#將空值補眾數
#train.loc[train['花萼長度'].isnull(),['花萼長度']] = train['花萼長度'].mode()[0] #花萼長度:第1欄
train[['花萼長度']] = train[['花萼長度']].fillna(train['花萼長度'].mode()[0])

train.plot(kind='line',y='花萼長度',figsize=(10,6),fontsize=14,title='花萼長度')
from pandas.plotting import scatter_matrix
scatter_matrix( train[['花瓣寬度','花瓣長度','花萼寬度','花萼長度']],figsize=(10, 10),color='b')
corr = train[['花瓣寬度','花瓣長度','花萼寬度','花萼長度']].corr()
print(corr)
import seaborn as sns
plt.rcParams['font.family']='DFKai-SB' #顯示中文
plt.figure(figsize=(10,10))
sns.heatmap(corr, square=True, annot=True, cmap="RdBu_r") #center=0, cmap="YlGnBu"
#sns.plt.show()

# http://seaborn.pydata.org/tutorial/color_palettes.html
#train[['花瓣寬度','花瓣長度','花萼寬度','花萼長度']]
fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(10, 10), sharey=True)

axes[0, 0].boxplot(train['花萼寬度'],showmeans=True)
axes[0, 0].set_title('訓:花萼寬度')

axes[0, 1].boxplot(train['花瓣寬度'],showmeans=True)
axes[0, 1].set_title('訓:花瓣寬度')

axes[0, 2].boxplot(train['花瓣長度'],showmeans=True)
axes[0, 2].set_title('訓:花瓣長度')

axes[0, 3].boxplot(train['花萼長度'],showmeans=True)
axes[0, 3].set_title('訓:花萼長度')

axes[1, 0].boxplot(test['花萼寬度'],showmeans=True)
axes[1, 0].set_title('測:花萼寬度')

axes[1, 1].boxplot(test['花瓣寬度'],showmeans=True)
axes[1, 1].set_title('測:花瓣寬度')

axes[1, 2].boxplot(test['花瓣長度'],showmeans=True)
axes[1, 2].set_title('測:花瓣長度')

axes[1, 3].boxplot(test['花萼長度'],showmeans=True)
axes[1, 3].set_title('測:花萼長度')
train.plot(kind='bar',y='花萼寬度',figsize=(30,6),fontsize=14,title='花萼寬度')
#IQR = Q3-Q1
IQR = np.percentile(train['花萼寬度'],75) - np.percentile(train['花萼寬度'],25)
#outlier = Q3 + 1.5*IQR , or. Q1 - 1.5*IQR
train[train['花萼寬度'] > np.percentile(train['花萼寬度'],75)+1.5*IQR]
#outlier = Q3 + 1.5*IQR , or. Q1 - 1.5*IQR
train[train['花萼寬度'] < np.percentile(train['花萼寬度'],25)-1.5*IQR]
#fix_X = X.drop(X.index[[5,23,40]])
#fix_y = y.drop(y.index[[5,23,40]])
#把示範用的 type 4, 資料去除, 以免干擾建模
train = train[train['type']!=4]
from sklearn.model_selection import train_test_split

X = train[['花瓣寬度','花瓣長度','花萼寬度','花萼長度']]
y = train['type']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state=100)
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)
X_train_std[0:5]
y_test[0:5]
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics

knn = KNeighborsClassifier(n_neighbors=3, weights='uniform')
knn.fit(X_train_std, y_train)

print(metrics.classification_report(y_test, knn.predict(X_test_std)))
print(metrics.confusion_matrix(y_test, knn.predict(X_test_std)))
from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier(n_estimators=500, criterion='gini', max_features='auto', oob_score=True)
rfc.fit(X_train, y_train) #不標準化

print("oob_score(accuary):",rfc.oob_score_)
print(metrics.classification_report(y_test, rfc.predict(X_test)))
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
gnb.fit(X_train_std, y_train)

print(metrics.classification_report(y_test, gnb.predict(X_test_std)))
print(metrics.confusion_matrix(y_test, gnb.predict(X_test_std)))
from sklearn.svm import SVC

svc = SVC(C=1.0, kernel="rbf", probability=True)
svc.fit(X_train_std, y_train)

print(metrics.classification_report(y_test, svc.predict(X_test_std)))
print(metrics.confusion_matrix(y_test, svc.predict(X_test_std)))
#from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from mlxtend.classifier import StackingClassifier
import xgboost as xgb

clf1 = KNeighborsClassifier(n_neighbors=3, weights='uniform')
clf2 = RandomForestClassifier(n_estimators=500, criterion='gini', max_features='auto', oob_score=True)
clf3 = GaussianNB()
clf4 = SVC(C=1.0, kernel="rbf", probability=True)
meta_clf = xgb.XGBClassifier(n_estimators= 2000, max_depth= 4)
stacking_clf = StackingClassifier(classifiers=[clf1, clf2, clf3, clf4], meta_classifier=meta_clf)

clf1.fit(X_train_std, y_train)
clf2.fit(X_train, y_train)
clf3.fit(X_train_std, y_train)
clf4.fit(X_train_std, y_train)
stacking_clf.fit(X_train_std, y_train)

print('KNN Score:',clf1.score(X_test_std, y_test))
print('RF Score:',clf2.score(X_test, y_test))
print('GNB Score:',clf3.score(X_test_std, y_test))
print('SVC Score:',clf4.score(X_test_std, y_test))
print('Stacking Score:',stacking_clf.score(X_test_std, y_test))
import xgboost as xgb

gbm = xgb.XGBClassifier(n_estimators= 2000, max_depth= 4).fit(X_train, y_train)

print(metrics.classification_report(y_test, gbm.predict(X_test)))
print("Score:", gbm.score(X_test, y_test))
print(gbm.feature_importances_)
from xgboost import plot_importance
plot_importance(gbm, )
plt.show()
pred = gbm.predict(test[['花瓣寬度','花瓣長度','花萼寬度','花萼長度']])
pred
# Generate Submission File 
StackingSubmission = pd.DataFrame({ 'id': submission.id, 'type': pred })
StackingSubmission.to_csv("submission.csv", index=False)
submission = pd.read_csv('submission.csv', encoding = "utf-8", dtype = {'type': np.int32})
submission
test[20:30]
#使用先前 training set的scale fit做縮放
test_std = sc.transform(test[['花瓣寬度','花瓣長度','花萼寬度','花萼長度']])
submission_stk = stacking_clf.predict(test_std)
submission_stk
submission_rfc = rfc.predict(test[['花瓣寬度','花瓣長度','花萼寬度','花萼長度']])
submission_rfc
submission_knn =knn.predict(test_std)
submission_knn
submission_gnb = gnb.predict(test_std)
submission_gnb
submission_svc = svc.predict(test_std)
submission_svc
from sklearn.ensemble import VotingClassifier
clf1 = knn
clf2 = rfc
clf3 = gnb
clf4 = svc

eclf = VotingClassifier(estimators=[('knn', clf1), ('rfc', clf2),('gnb', clf3),('svc',clf4)], voting='hard', weights=[1, 1, 1, 4])
eclf.fit(X_train_std, y_train)
print(metrics.classification_report(y_test, eclf.predict(X_test_std)))
submission_eclf = eclf.predict(test_std)
submission_eclf
