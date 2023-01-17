import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline

pd.set_option('max_columns', 30, 'max_rows', 50)
tr_train = pd.read_csv('../input/X_train.csv', encoding='cp949')
tr_test = pd.read_csv('../input/X_test.csv', encoding='cp949')
tr = pd.concat([tr_train, tr_test])
tr.head()
y_train = pd.read_csv('../input/y_train.csv')
tr_train_all = pd.merge(tr_train, y_train, how="left")
f = tr_train_all.groupby('cust_id')['store_nm'].agg([('주구매지점', lambda x: x.value_counts().index[0])]).reset_index()
f.head()
tr_train_all = pd.merge(tr_train_all, f, how= 'left')
tr_train_all.head()
f= tr_train_all.groupby(['주구매지점'])['gender'].mean().sort_values().to_frame().reset_index()
f
male_preferred_store=f[f['gender']>0.4]['주구매지점'].tolist()
female_preferred_store=f[f['gender']<0.26]['주구매지점'].tolist()
print(male_preferred_store)
print(female_preferred_store)
f = tr_train_all.groupby('cust_id')['gds_grp_mclas_nm'].agg([('주구매코너', lambda x: x.value_counts().index[0])]).reset_index()
f.head()
tr_train_all = pd.merge(tr_train_all, f, how= 'left')
tr_train_all.head()
f= tr_train_all.groupby(['주구매코너'])['gender'].mean().sort_values().to_frame().reset_index()
f
male_preferred_corner=f[f['gender']>0.6]['주구매코너'].tolist()
female_preferred_corner=f[f['gender']<0.22]['주구매코너'].tolist()
print(male_preferred_corner)
print(female_preferred_corner)
features = []
f = tr.groupby('cust_id')['amount'].agg([('총구매액', 'sum')]).reset_index()
features.append(f)
f.head()
f = tr.groupby('cust_id')['amount'].agg([('구매건수', 'size')]).reset_index()
features.append(f)
f.head()
f = tr.groupby('cust_id')['amount'].agg([('평균구매가격', 'mean')]).reset_index()
features.append(f)
f.head()
n = tr.gds_grp_nm.nunique()
f = tr.groupby('cust_id')['gds_grp_nm'].agg([('구매상품다양성', lambda x: len(x.unique()) / n)]).reset_index()
features.append(f)
f.head()
tr['sales_date'] = tr.tran_date.str[:10]
f = tr.groupby(by = 'cust_id')['sales_date'].agg([('내점일수','nunique')]).reset_index()
features.append(f)
f.head()
def weekday(x):
    w = x.dayofweek 
    if w < 4:
        return 1 # 주중
    else:
        return 0 # 주말
f = tr.groupby(by = 'cust_id')['sales_date'].agg([('요일구매패턴', lambda x : pd.to_datetime(x).apply(weekday).value_counts().index[0])]).reset_index()
features.append(f)
f.head()
def f1(x):
    k = x.month
    if 3 <= k <= 5 :
        return('봄_구매건수')
    elif 6 <= k <= 8 :
        return('여름_구매건수')
    elif 9 <= k <= 11 :    
        return('가을_구매건수')
    else :
        return('겨울_구매건수')    
    
tr['season'] = pd.to_datetime(tr.sales_date).apply(f1)
f = pd.pivot_table(tr, index='cust_id', columns='season', values='amount', 
                   aggfunc=np.size, fill_value=0).reset_index()
features.append(f)
f.head()
f = tr.groupby('cust_id')['gds_grp_mclas_nm'].agg([('주구매코너', lambda x: x.value_counts().index[0])]).reset_index()
f = pd.get_dummies(f, columns=['주구매코너'])  # This method performs One-hot-encoding
features.append(f)
f.head()
def f2(x):
    k = x.dayofweek 
    if k == 0 :
        return('월_구매건수')
    elif k== 1 :
        return('화_구매건수')
    elif k== 2 :
        return('수_구매건수')
    elif k== 3 :
        return('목_구매건수')
    elif k== 4 :
        return('금_구매건수')
    elif k== 5 :
        return('토_구매건수')
    else :
        return('일_구매건수')    
    
tr['weekday'] = pd.to_datetime(tr.sales_date).apply(f2)
f = pd.pivot_table(tr, index='cust_id', columns='weekday', values='amount', 
                   aggfunc=np.size, fill_value=0).reset_index()
features.append(f)
f.head()
def f3(x):
    k = x.dayofweek 
    if k == 0 :
        return('월_구매액')
    elif k== 1 :
        return('화_구매액')
    elif k== 2 :
        return('수_구매액')
    elif k== 3 :
        return('목_구매액')
    elif k== 4 :
        return('금_구매액')
    elif k== 5 :
        return('토_구매액')
    else :
        return('일_구매액')    
    
tr['weekday_1'] = pd.to_datetime(tr.sales_date).apply(f3)
f = pd.pivot_table(tr, index='cust_id', columns='weekday_1', values='amount', 
                   aggfunc=np.sum, fill_value=0).reset_index()
features.append(f)
f.head()
#최대 구매 요일
def f6(x):
    k = x.dayofweek 
    if k == 0 :
        return('월')
    elif k== 1 :
        return('화')
    elif k== 2 :
        return('수')
    elif k== 3 :
        return('목')
    elif k== 4 :
        return('금')
    elif k== 5 :
        return('토')
    else :
        return('일') 


f = tr.groupby(by = 'cust_id')['sales_date'].agg([('주구매요일', lambda x : pd.to_datetime(x).apply(f6).value_counts().index[0])]).reset_index()
f = pd.get_dummies(f, columns=['주구매요일']) 
features.append(f)
f.head()
def f4(x):
    k = x.month
    if k==1 :
        return('1월_구매건수')
    elif k==2 :
        return('2월_구매건수')
    elif k==3 :
        return('3월_구매건수')
    elif k==4 :
        return('4월_구매건수')
    elif k==5 :
        return('5월_구매건수')
    elif k==6 :
        return('6월_구매건수')
    elif k==7 :
        return('7월_구매건수')
    elif k==8 :
        return('8월_구매건수')
    elif k==9 :
        return('9월_구매건수')
    elif k==10 :
        return('10월_구매건수')
    elif k==11 :
        return('11월_구매건수')
    else :
        return('12월_구매건수')    
    
tr['month'] = pd.to_datetime(tr.sales_date).apply(f4)
f = pd.pivot_table(tr, index='cust_id', columns='month', values='amount', 
                   aggfunc=np.size, fill_value=0).reset_index()
features.append(f)
f.head()
def f5(x):
    k = x.month
    if k==1 :
        return('1월_구매액')
    elif k==2 :
        return('2월_구매액')
    elif k==3 :
        return('3월_구매액')
    elif k==4 :
        return('4월_구매액')
    elif k==5 :
        return('5월_구매액')
    elif k==6 :
        return('6월_구매액')
    elif k==7 :
        return('7월_구매액')
    elif k==8 :
        return('8월_구매액')
    elif k==9 :
        return('9월_구매액')
    elif k==10 :
        return('10월_구매액')
    elif k==11 :
        return('11월_구매액')
    else :
        return('12월_구매액')    
    
tr['month_1'] = pd.to_datetime(tr.sales_date).apply(f5)
f = pd.pivot_table(tr, index='cust_id', columns='month_1', values='amount', 
                   aggfunc=np.sum, fill_value=0).reset_index()
features.append(f)
f.head()
#주구매월
def f7(x):
    k = x.month
    if k==1 :
        return('1월')
    elif k==2 :
        return('2월')
    elif k==3 :
        return('3월')
    elif k==4 :
        return('4월')
    elif k==5 :
        return('5월')
    elif k==6 :
        return('6월')
    elif k==7 :
        return('7월')
    elif k==8 :
        return('8월')
    elif k==9 :
        return('9월')
    elif k==10 :
        return('10월')
    elif k==11 :
        return('11월')
    else :
        return('12월')    
    
f = tr.groupby(by = 'cust_id')['sales_date'].agg([('주구매월', lambda x : pd.to_datetime(x).apply(f7).value_counts().index[0])]).reset_index()
f = pd.get_dummies(f, columns=['주구매월']) 
features.append(f)
f.head()
def f8(x):
    k = x.day
    if 1<= k <= 10 :
        return('월초_구매액')
    elif 10 < k <= 20 :
        return('월중순_구매액')
    else :
        return('월말_구매액')    
    
tr['day'] = pd.to_datetime(tr.sales_date).apply(f8)
f = pd.pivot_table(tr, index='cust_id', columns='day', values='amount', 
                   aggfunc=np.sum, fill_value=0).reset_index()
features.append(f)
f.head()
def f9(x):
    k = x.day
    if 1<= k <= 10 :
        return('월초_구매건수')
    elif 10 < k <= 20 :
        return('월중순_구매건수')
    else :
        return('월말_구매건수')    
    
tr['day_1'] = pd.to_datetime(tr.sales_date).apply(f9)
f = pd.pivot_table(tr, index='cust_id', columns='day_1', values='amount', 
                   aggfunc=np.size, fill_value=0).reset_index()
features.append(f)
f.head()
#주구매일
def f10(x):
    k = x.day
    if 1<= k <= 10 :
        return('월초')
    elif 10 < k <= 20 :
        return('월중순')
    else :
        return('월말')    
    
f = tr.groupby(by = 'cust_id')['sales_date'].agg([('주구매일', lambda x : pd.to_datetime(x).apply(f10).value_counts().index[0])]).reset_index()
f = pd.get_dummies(f, columns=['주구매일']) 
features.append(f)
f.head()
f = tr.groupby('cust_id')['gds_grp_mclas_nm'].agg([('주구매코너', lambda x: x.value_counts().index[0])]).reset_index()
f['주구매코너_성별']=np.where(f['주구매코너'].isin(male_preferred_corner),1,0)
f['주구매코너_성별']=np.where(f['주구매코너'].isin(female_preferred_corner),-1,f['주구매코너_성별'])
f.drop(labels = ["주구매코너"],axis = 1,inplace=True)
features.append(f)
f.head()
f = tr.groupby('cust_id')['store_nm'].agg([('주구매지점', lambda x: x.value_counts().index[0])]).reset_index()
f['주구매지점_성별']=np.where(f['주구매지점'].isin(male_preferred_store),1,0)
f['주구매지점_성별']=np.where(f['주구매지점'].isin(female_preferred_store),-1,f['주구매지점_성별'])
f.drop(labels = ["주구매지점"],axis = 1,inplace=True)
features.append(f)
f.head()
X_train = pd.DataFrame({'cust_id': tr_train.cust_id.unique()})
for f in features :
    X_train = pd.merge(X_train, f, how='left')
display(X_train)

X_test = pd.DataFrame({'cust_id': tr_test.cust_id.unique()})
for f in features :
    X_test = pd.merge(X_test, f, how='left')
display(X_test)
IDtest = X_test.cust_id;
X_train.drop(['cust_id'], axis=1, inplace=True)
X_test.drop(['cust_id'], axis=1, inplace=True)
y_train = pd.read_csv('../input/y_train.csv').gender
#!pip install xgboost
from sklearn import metrics #accuracy measure
from sklearn.metrics import confusion_matrix #for confusion matrix

from collections import Counter

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier, VotingClassifier, BaggingClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, cross_val_score, cross_val_predict, KFold, StratifiedKFold, learning_curve, train_test_split
from xgboost import XGBClassifier
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('fivethirtyeight')
import warnings
warnings.filterwarnings('ignore')
%matplotlib inline
kfold = StratifiedKFold(n_splits=10)

random_state = 0
classifiers = []
classifiers.append(SVC(random_state=random_state))
classifiers.append(DecisionTreeClassifier(random_state=random_state))
classifiers.append(AdaBoostClassifier(DecisionTreeClassifier(random_state=random_state),random_state=random_state,learning_rate=0.1))
classifiers.append(RandomForestClassifier(random_state=random_state))
classifiers.append(ExtraTreesClassifier(random_state=random_state))
classifiers.append(GradientBoostingClassifier(random_state=random_state))
classifiers.append(XGBClassifier(random_state=random_state))
classifiers.append(BaggingClassifier(base_estimator=DecisionTreeClassifier(),random_state=random_state,n_estimators=100))
classifiers.append(LogisticRegression(random_state = random_state))
classifiers.append(LinearDiscriminantAnalysis())

cv_results = []
for classifier in classifiers :
    cv_results.append(cross_val_score(classifier, X_train, y = y_train, scoring = "accuracy", cv = kfold, n_jobs=4))

cv_means = []
cv_std = []
for cv_result in cv_results:
    cv_means.append(cv_result.mean())
    cv_std.append(cv_result.std())

cv_res = pd.DataFrame({"CrossValMeans":cv_means,"CrossValerrors": cv_std,"Algorithm":["SVC","DecisionTree","AdaBoost",
"RandomForest","ExtraTrees","GradientBoosting","XGBoost","BaggingDecisionTree","LogisticRegression","LinearDiscriminantAnalysis"]})

g = sns.barplot("CrossValMeans","Algorithm",data = cv_res, palette="Set3",orient = "h",**{'xerr':cv_std})
g.set_xlabel("Mean Accuracy")
g = g.set_title("Cross validation scores")
cv_res
# Gradient boosting tunning

GBC = GradientBoostingClassifier()
gb_param_grid = {'loss' : ["deviance"],
              'n_estimators' : [100],
              'learning_rate': [0.1, 0.05, 0.01],
              'max_depth': [4],
              'min_samples_leaf': [100,150],
              'max_features': [0.3, 0.1] 
              }

gsGBC = GridSearchCV(GBC,param_grid = gb_param_grid, cv=kfold, scoring="accuracy", n_jobs= 4, verbose = 1)

gsGBC.fit(X_train,y_train)

GBC_best = gsGBC.best_estimator_

# Best score
print(gsGBC.best_score_)
print(GBC_best)
from sklearn.metrics import accuracy_score, precision_score, precision_recall_curve
train_Gender = pd.Series(GBC_best.predict(X_train), name="gender")
sns.heatmap(confusion_matrix(y_train,train_Gender),cmap='winter',annot=True,fmt='2.0f')
plt.show()

print("accuracy_score: {}".format( accuracy_score(y_train, train_Gender)))
# XG boosting tunning

XGBC = XGBClassifier()
xgb_param_grid = {'min_child_weight' : [1],
              'n_estimators' : [250],
              'learning_rate': [0.07],
              'max_depth': [3, 4],
              'reg_alpha': [0,0.01]
              }

gsXGBC = GridSearchCV(XGBC,param_grid = xgb_param_grid, cv=kfold, scoring="accuracy", n_jobs= 4, verbose = 1)

gsXGBC.fit(X_train,y_train)

XGBC_best = gsXGBC.best_estimator_

# Best score
print(gsXGBC.best_score_)
print(XGBC_best)
train_Gender = pd.Series(XGBC_best.predict(X_train), name="gender")
sns.heatmap(confusion_matrix(y_train,train_Gender),cmap='winter',annot=True,fmt='2.0f')
plt.show()

print("accuracy_score: {}".format( accuracy_score(y_train, train_Gender)))
# RFC Parameters tunning 
RFC = RandomForestClassifier()


## Search grid for optimal parameters
rf_param_grid = {"max_depth": [None],
              "max_features": [1, 3],
              "min_samples_split": [3, 10],
              "min_samples_leaf": [3, 10],
              "bootstrap": [False],
              "n_estimators" :[300],
              "criterion": ["gini"]}


gsRFC = GridSearchCV(RFC,param_grid = rf_param_grid, cv=kfold, scoring="accuracy", n_jobs= 4, verbose = 1)

gsRFC.fit(X_train,y_train)

RFC_best = gsRFC.best_estimator_

# Best score
print(gsRFC.best_score_)
print(RFC_best)
from sklearn.metrics import accuracy_score, precision_score, precision_recall_curve
train_Gender = pd.Series(RFC_best.predict(X_train), name="gender")
sns.heatmap(confusion_matrix(y_train,train_Gender),cmap='winter',annot=True,fmt='2.0f')
plt.show()

print("accuracy_score: {}".format( accuracy_score(y_train, train_Gender)))
# Bagging DT tunning

DTC = DecisionTreeClassifier()

BDT = BaggingClassifier(DTC, random_state=7)

bdt_param_grid = {"base_estimator__criterion" : ["gini"],
              "base_estimator__splitter" :   ["best", "random"],
              'n_estimators' : [300],
              'base_estimator__min_samples_leaf': [1],
              'base_estimator__min_samples_split': [2],
              'max_features': [1,3],
              'bootstrap':[False]
              }

gsBDT = GridSearchCV(BDT,param_grid = bdt_param_grid, cv=kfold, scoring="accuracy", n_jobs= 4, verbose = 1)

gsBDT.fit(X_train,y_train)

BDT_best = gsBDT.best_estimator_

# Best score
print(gsBDT.best_score_)
print(BDT_best)
sns.set(style='white', context='notebook', palette='deep')

def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=-1, train_sizes=np.linspace(.1, 1.0, 5)):
    """Generate a simple plot of the test and training learning curve"""
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt

g = plot_learning_curve(gsRFC.best_estimator_,"RF learning curves",X_train,y_train,cv=kfold)
g = plot_learning_curve(gsXGBC.best_estimator_,"XG boost learning curves",X_train,y_train,cv=kfold)
g = plot_learning_curve(gsBDT.best_estimator_,"Bagging DecisionTree learning curves",X_train,y_train,cv=kfold)
g = plot_learning_curve(gsGBC.best_estimator_,"Gradient Boosting learning curves",X_train,y_train,cv=kfold)
from matplotlib import font_manager, rc
font_name = font_manager.FontProperties(fname="c:/Windows/Fonts/malgun.ttf").get_name()
rc('font', family=font_name)

nrows = ncols = 2
fig, axes = plt.subplots(nrows = nrows, ncols = ncols, sharex="all", figsize=(15,15))

RFC.fit(X_train,y_train)
XGBC.fit(X_train,y_train)

names_classifiers = [("RandomForest", RFC_best),("XG Boost",XGBC_best),("BasicRF",RFC),("BasicXGB",XGBC)]

nclassifier = 0
for row in range(nrows):
    for col in range(ncols):
        name = names_classifiers[nclassifier][0]
        classifier = names_classifiers[nclassifier][1]
        indices = np.argsort(classifier.feature_importances_)[::-1][:40]
        g = sns.barplot(y=X_train.columns[indices][:40],x = classifier.feature_importances_[indices][:40] , orient='h',ax=axes[row][col])
        g.set_xlabel("Relative importance",fontsize=12)
        g.set_ylabel("Features",fontsize=12)
        g.tick_params(labelsize=9)
        g.set_title(name + " feature importance")
        nclassifier += 1
test_Survived_GBC = pd.Series(GBC_best.predict(X_test), name="GBC")
test_Survived_XGBC = pd.Series(XGBC_best.predict(X_test), name="XGBC")
test_Survived_BDT = pd.Series(BDT_best.predict(X_test), name="BDT")

# Concatenate all classifier results
ensemble_results = pd.concat([test_Survived_GBC,test_Survived_XGBC,test_Survived_BDT],axis=1)

g= sns.heatmap(ensemble_results.corr(),annot=True)
votingC = VotingClassifier(estimators=[('rfc',RFC_best),('gbc',GBC_best),('xgbc',XGBC_best)], voting='soft', n_jobs=4)

votingC = votingC.fit(X_train, y_train)

cross = cross_val_score(votingC, X_train, y_train, cv = 10, scoring = "accuracy")
print('The cross validated score is',cross.mean())
from sklearn.metrics import accuracy_score, precision_score, precision_recall_curve

train_Gender = pd.Series(votingC.predict(X_train), name="gender")
sns.heatmap(confusion_matrix(y_train,train_Gender),cmap='winter',annot=True,fmt='2.0f')
plt.show()

print("accuracy_score: {}".format( accuracy_score(y_train, train_Gender)))
votingC.fit(X_train, y_train)
pred = votingC.predict_proba(X_test)[:,1]
fname = 'submissions.csv'
submissions = pd.concat([IDtest, pd.Series(pred, name="gender")] ,axis=1)
submissions.to_csv(fname, index=False)
print("'{}' is ready to submit." .format(fname))