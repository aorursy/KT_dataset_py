import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

import time



from sklearn.preprocessing import MinMaxScaler,StandardScaler

from sklearn import cross_validation

from sklearn.cross_validation import train_test_split

from sklearn.model_selection import StratifiedKFold,learning_curve,GridSearchCV

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier,BaggingClassifier,GradientBoostingClassifier

from sklearn.metrics import roc_curve,roc_auc_score,auc

from sklearn.decomposition import PCA



import sklearn as sl

import xgboost as xgb



import warnings

warnings.filterwarnings('ignore')
train_data = pd.read_csv('../input/train.csv')

train_data
train_df = train_data.drop(['TARGET','ID'],axis=1).copy()

target = train_data.TARGET

# train_df[:] = StandardScaler().fit_transform(train_df)

train_df

# target.info()

sns.countplot(x='TARGET',data=train_data)
X = train_df.as_matrix()

y = target.as_matrix()


train_df[:] = StandardScaler().fit_transform(train_df)

X_scaled = train_df.as_matrix()
def plot_learning_curve(estimator, title, X, y, ylim=(0.4,1.1), cv=None,

                        train_sizes=np.linspace(.1, 1.0, 5)):

    """

    画出data在某模型上的learning curve.

    参数解释

    ----------

    estimator : 你用的分类器。

    title : 表格的标题。

    X : 输入的feature，numpy类型

    y : 输入的target vector

    ylim : tuple格式的(ymin, ymax), 设定图像中纵坐标的最低点和最高点

    cv : 做cross-validation的时候，数据分成的份数，其中一份作为cv集，其余n-1份作为training(默认为3份)

    """

    start_time = time.time()

    plt.figure()

    train_sizes, train_scores, test_scores = learning_curve(

        estimator, X, y, cv=3, n_jobs=1, train_sizes=train_sizes)

    train_scores_mean = np.mean(train_scores, axis=1)

    train_scores_std = np.std(train_scores, axis=1)

    test_scores_mean = np.mean(test_scores, axis=1)

    test_scores_std = np.std(test_scores, axis=1)



    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,

                     train_scores_mean + train_scores_std, alpha=0.1,

                     color="r")

    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,

                     test_scores_mean + test_scores_std, alpha=0.1, color="g")

    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",

             label="Training score")

    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",

             label="Cross-validation score")

    

    plt.annotate(test_scores_mean[-1],xy=(train_sizes[-1],test_scores_mean[-1]))



    plt.xlabel("Training examples")

    plt.ylabel("Score")

    plt.legend(loc="best")

    plt.grid("on") 

    if ylim:

        plt.ylim(ylim)

    plt.title(title+'(time=%fs)'%(time.time()-start_time))

    plt.show()
def print_grid_search(search_clf):

    search_clf.grid_scores_,search_clf.best_params_,search_clf.best_score_

    return
def grid_search(clf,params,X,y):

    start_time = time.time()

    search_clf = GridSearchCV(estimator=clf,param_grid=params,scoring='roc_auc',n_jobs=4,cv=5)

    search_clf.fit(X,y)

#     print_grid_search(search_clf)

    print('time: ',time.time()-start_time)

    return search_clf
xgb_clf = xgb.XGBClassifier(learning_rate=0.1, n_estimators=147,max_depth=4,min_child_weight=9,gamma=0.2,subsample=0.9,colsample_bytree=0.55)
xgb_clf.fit(X_scaled,y)
clf = xgb_clf
test_data = pd.read_csv('../input/test.csv')

test_data
ids = test_data.ID

test_df = test_data.drop('ID',axis=1).copy()

test_df[:] = StandardScaler().fit_transform(test_df)
predict = clf.predict_proba(test_df.as_matrix())

predict
predict_xgb = xgb_clf.predict_proba(test_df.as_matrix())

predict_xgb
result_df = pd.DataFrame(predict_xgb[:,1],columns=['TARGET'])

result_df.insert(0,'ID',ids)

result_df
result_df.to_csv('result.csv',index=False)