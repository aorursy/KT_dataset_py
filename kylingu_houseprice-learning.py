# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

# disable copy warning in pandas
pd.options.mode.chained_assignment = None  # default='warn'
# disable sklearn deprecation warning
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input/"))
# Any results you write to the current directory are saved as output.

# 限定numpy输出进度到4
np.set_printoptions(precision=4)
pd.set_option('precision', 4)
df_train = pd.read_csv('../input/train.csv').set_index('Id')
df_test = pd.read_csv('../input/test.csv').set_index('Id')
dataset = pd.concat([df_train, df_test], axis=0, sort=True)
id_test = df_test.index
id_train = df_train.index
dataset.head()
dataset.shape, df_train.shape, df_test.shape
dataset.info()
# 定量，数值变量
quantitative = [col for col in df_test.columns if df_test.dtypes[col] != 'object']
need_convert = ['MSSubClass', 'YearBuilt', 'YearRemodAdd', 'YrSold', 'GarageYrBlt']
quantitative = set(quantitative) - set(need_convert)
quantitative = list(quantitative)
# 其中一些是rate变量，比如OverallCond，OverallQual都是评分，但是评分有高低，所以仍然算是定量变量。

# 定性，类型变量
qualitative = [col for col in df_test.columns if df_test.dtypes[col] == 'object']
qualitative.extend(need_convert)
len(quantitative), len(qualitative)
dataset.loc[:, qualitative] = dataset[qualitative].astype('object')
dataset[quantitative].describe().T
dataset[qualitative].describe(include=['O']).T
missing = dataset.drop(columns='SalePrice').isna().sum()
fig = plt.figure(figsize=(9, 6))
missing[missing > 0].sort_values().plot(kind='barh', ax=fig.gca(), title='Missing value feature amount')
sns.despine(bottom=True)
fig = plt.figure(figsize=(14, 14))
sns.heatmap(dataset.loc[id_train, quantitative+ ['SalePrice']].corr(), cmap='cool', ax=fig.gca())
corrs = dataset.loc[id_train, quantitative + ['SalePrice']].corr().SalePrice.drop('SalePrice')
corrs.sort_values().plot(kind='bar', figsize=(9, 6), title='Correlations to SalePrice', rot=60)
sns.despine(left=True, bottom=True)
corrs[np.abs(corrs) >= 0.4].sort_values().plot.bar(figsize=(9, 6), title='Correlations above middle strength to SalePrice')
sns.despine()
fig = plt.figure(figsize=(9, 6))
sns.heatmap(dataset.loc[id_train, list(corrs[np.abs(corrs) > 0.4].index)].corr(), cmap='cool', ax=fig.gca(), annot=True)
dataset.loc[id_train, ['GrLivArea', 'TotRmsAbvGrd', 'TotalBsmtSF', '1stFlrSF', 'GarageCars', 'GarageArea']].count()
selected_quantitive = ['OverallQual', 'MasVnrArea', 'FullBath', 'GarageCars', 'GrLivArea', 'TotalBsmtSF', 'Fireplaces']
train_set = dataset.loc[id_train, selected_quantitive + ['SalePrice']]
train_set.info()
from scipy import stats
# p-value 是否> 0.05
train_set.dropna().apply(lambda x: stats.shapiro(x)[1] < 0.05, axis=0)
# 画所有的变量两两散点图分布。
# sns.pairplot(train_set.dropna())

# 画单个变量分布
# sns.distplot(train_set.OverallQual);
# sns.distplot(train_set.SalePrice, fit=stats.norm)
# 画各个变量分布。
g = sns.FacetGrid(pd.melt(train_set), col='variable', col_wrap=4, sharex=False, sharey=False)
g.map(sns.distplot, 'value', kde=False, fit=stats.norm)
temp = train_set.SalePrice.apply(lambda x: np.log(x))
sns.distplot(temp)

for col in ['MasVnrArea', 'GrLivArea', 'SalePrice']:
    print(col, ', skew: ', train_set[col].skew(), 'after log: ',train_set[col].apply(lambda x: np.log(x) if x > 0 else 0).skew())
fig, axes = plt.subplots(2, 4, figsize=(16, 7))
plt.subplots_adjust(wspace=0.3, hspace=0.3)
fig.suptitle("Scatter plot to SalePrice", fontsize=16)
y = train_set.dropna()['SalePrice'].apply(lambda x: np.log(x))

for row, ax_row in enumerate(axes):
    for col, ax in enumerate(ax_row):
        element = col + row * 4
        if element == 7:
            break
        
        mcol = selected_quantitive[col + row * 4]
        x = train_set.dropna()[mcol]
        if mcol in ['MasVnrArea', 'GrLivArea']:
            x = x.apply(lambda x: np.log(x) if x!=0 else 0)
        ax.scatter(x=x, y=y);
        ax.set_title(mcol)
qualitative.__len__()
qualitive_set = dataset.loc[id_train, qualitative+['SalePrice']].fillna('Missing')
qualitive_set[qualitative] = qualitive_set[qualitative].astype('category')
# 将数据转换成长型，将（列名和值）与其他的（列名和值）竖向连接下去，生成类似于这样的
# SalePrice	variable	value
# 0	208500.0	MSZoning	RL
# 1	181500.0	MSZoning	RL
# 2	223500.0	MSZoning	RL
# 3	140000.0	MSZoning	RL
# 4	250000.0	MSZoning	RL

long_data = pd.melt(qualitive_set, id_vars=['SalePrice'], value_vars=qualitative)

def mboxplot(x, y, **kwargs):
    ax = sns.boxplot(x, y, palette= kwargs.get('palette', None))
    ax.xaxis.set_tick_params(rotation=70)

def plotbox(start, end):
    g = sns.FacetGrid(long_data[long_data.variable.isin(qualitative[start: end])], col='variable', col_wrap=3, sharex=False, sharey=False, size=4.5)
    g.map(mboxplot, 'value', 'SalePrice', palette='cool')
    plt.subplots_adjust(top=0.9)
    g.fig.suptitle('Boxplot of qualitive col to SalePrice')
# 绘制前6列的boxplot
plotbox(0, 6)
plotbox(6, 12)
plotbox(12, 18)
plotbox(18, 24)
plotbox(24, 30)
plotbox(30, 36)
plotbox(36, 42)
plotbox(42, 48)
def anova(frame):
    anv = pd.DataFrame()
    anv['feature'] = qualitative
    pvals = []
    for c in qualitative:
        samples = []
        for cls in frame[c].unique():
            s = frame[frame[c] == cls]['SalePrice'].values
            samples.append(s)
        pval = stats.f_oneway(*samples)[0]
        pvals.append(pval)
    anv['fval'] = pvals
    return anv.sort_values('fval')

a = anova(qualitive_set)
# 单纯的P值太小，即我们要的是pval小于0.05，因此进行一次转换
# a['disparity'] = np.log(1./a['pval'].values)

fig = plt.figure(figsize=(14, 6))
sns.barplot(data=a, x='feature', y='fval', orient='v', ax=fig.gca())
x=plt.xticks(rotation=90)
selected_qualitive = a[a.fval > 10].feature
qualitive_set = dataset.loc[id_train, selected_qualitive]
qualitive_set.shape
dataset[quantitative].isna().sum()
dataset.MasVnrArea.fillna(dataset.MasVnrArea.median(), inplace=True)
dataset.GarageCars.fillna(dataset.GarageCars.median(), inplace=True)
dataset.TotalBsmtSF.fillna(dataset.TotalBsmtSF.median(), inplace=True)
dataset.GarageArea.fillna(dataset.GarageArea.median(), inplace=True)
dataset.BsmtFinSF1.fillna(dataset.BsmtFinSF1.median(), inplace=True)
dataset.BsmtFinSF2.fillna(dataset.BsmtFinSF2.median(), inplace=True)
dataset.BsmtUnfSF.fillna(dataset.BsmtUnfSF.median(), inplace=True)
dataset.BsmtFullBath.fillna(dataset.BsmtFullBath.median(), inplace=True)
dataset.BsmtHalfBath.fillna(dataset.BsmtHalfBath.median(), inplace=True)
# LotFrontage is missing too much, delete it.
dataset = dataset.drop(columns='LotFrontage')
dataset[qualitative].isna().sum().sort_values(ascending=False)
# 我们上面有说过，这些NA表示没有，所以我们填入Missing。
dataset.loc[:, qualitative] = dataset.loc[:, qualitative].fillna('Missing')
dataset.loc[:, qualitative] = dataset.loc[:, qualitative].astype('category')
dataset.isna().sum().sort_values(ascending=False).head(15)
dataset.MasVnrArea = dataset.MasVnrArea.apply(lambda x: np.log(x) if x > 0 else 0)
dataset.GrLivArea = dataset.GrLivArea.apply(lambda x: np.log(x) if x > 0 else 0)
dataset.SalePrice = dataset.SalePrice.apply(lambda x: np.log(x) if x > 0 else 0)
dataset.iloc[id_train].groupby('PoolQC').SalePrice.mean().sort_values().plot.bar()
dataset.PoolQC = dataset.PoolQC.apply(lambda x: 1 if x == 'Ex' else 0)
a = anova(dataset.loc[id_train, qualitative + ['SalePrice']])
# 单纯的P值太小，即我们要的是pval小于0.05，因此进行一次转换
# a['disparity'] = np.log(1./a['pval'].values)

fig = plt.figure(figsize=(14, 6))
sns.barplot(data=a, x='feature', y='fval', orient='v', ax=fig.gca())
x=plt.xticks(rotation=90)
selected_qualitive = list(a[a.fval > 10].feature)
train = dataset.loc[id_train, selected_quantitive + selected_qualitive + ['SalePrice']]
train = pd.get_dummies(train, columns=selected_qualitive)
X = train.drop(columns='SalePrice')
Y = train.SalePrice
Y.shape, X.shape
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, Y, random_state = 5)

dtr = DecisionTreeRegressor(random_state = 0).fit(X_train, y_train)
fig = plt.figure(figsize=(14, 6))
ax.xaxis.set_tick_params(rotation=70)
importances = pd.Series(dtr.feature_importances_, index=X.columns)
# importances.hist(bins=20, ax=fig.gca())
importances[importances > 0.001].sort_values().plot.barh()
dtr.score(X_test, y_test)
from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import Ridge
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict

X_train, X_test, y_train, y_test = train_test_split(X, Y, random_state = 5)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, random_state = 5)
# 控制 是否要运行寻参函数，以便节省时间。
RELEASE = True

def get_scaled_features(X, columns, X2=None):
    scaler = MinMaxScaler()
    X[columns] = scaler.fit_transform(X[columns])
    if X2 is not None:
        X2[columns] = scaler.transform(X2[columns])
    return X, X2

def get_polyed_features(X, columns, X2=None, degree=2):
    poly = PolynomialFeatures(degree=degree, include_bias=False)
    polyed_X = poly.fit_transform(X[columns])
    polyed_X = pd.DataFrame(polyed_X, columns=poly.get_feature_names(columns), index=X.index)
    polyed_X = pd.concat([X.loc[:, set(X.columns) - set(columns)], polyed_X], axis=1)
    
    polyed_X2 = None
    if X2 is not None:
        polyed_X2 = poly.transform(X2[columns])
        polyed_X2 = pd.DataFrame(polyed_X2, columns=poly.get_feature_names(columns), index=X2.index)
        polyed_X2 = pd.concat([X2.loc[:, set(X2.columns) - set(selected_quantitive)], polyed_X2], axis=1)
    
    return polyed_X, polyed_X2

def mcross_validation(esimator, X, Y, cv=3, random_state=None, need_scaled=True, need_poly=False):
    kf = KFold(n_splits=cv, shuffle=True, random_state=random_state)
    scores = []
    for train_index, test_index in kf.split(X, Y):
        x_train, x_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = Y.iloc[train_index], Y.iloc[test_index]
        
        if need_scaled:
            x_train, x_test = get_scaled_features(x_train, selected_quantitive, x_test)
    
        if need_poly:
            x_train, x_test = get_polyed_features(x_train, selected_quantitive, x_test)

        est =esimator.fit(x_train, y_train)
        y_pred = est.predict(x_test)
        scores.append(est.score(x_test, y_test))
    return scores

def plot_residual(y, y_pred, se=None, ylabeltext='Residual value', title='Residual plot for regression analysis'):
    residual = (y - y_pred)
#     I don't know how to calculate the studentized residual using sklearn or scipy.
#     if se != None:
#         residual = residual/se
    plt.scatter(x=y_pred, y= residual, s=20, marker='*', c='b')
    plt.plot([np.min(y_pred), np.max(y_pred)], [0, 0], 'g--')
    plt.title(title)
    plt.xlabel('Fitted Value')
    plt.ylabel(ylabeltext)
    sns.despine()

def plot_train_val_score(trainscores, valscores, alphas):
    plt.plot(alphas, trainscores, 'r-', label='train')
    plt.plot(alphas, valscores, label='val')
    plt.legend()
    plt.title('Best parameter is {:.5f} fot best score {:.4f}'.format(alphas[np.argmax(valscores)], np.max(valscores)))
    sns.despine()
X_train1, X_val1= get_scaled_features(X_train, selected_quantitive, X2=X_val)
lr = LinearRegression().fit(X_train1, y_train)
lr.score(X_val1, y_val)
y_pred = lr.predict(X_val1)
plot_residual(y_val, y_pred)
X_train1, X_val1= get_scaled_features(X_train, selected_quantitive, X2=X_val)
X_train_polyed, X_val_polyed = get_polyed_features(X_train1, selected_quantitive, X_val1)
lr = LinearRegression().fit(X_train_polyed, y_train)
lr.score(X_val_polyed, y_val)
plot_residual(y_val, lr.predict(X_val_polyed))
X_train1, X_val1= get_scaled_features(X_train, selected_quantitive, X2=X_val)
X_train_polyed, X_val_polyed = get_polyed_features(X_train1, selected_quantitive, X_val1)
ridge = Ridge(alpha=1, max_iter=10000).fit(X_train_polyed, y_train)
ridge.score(X_val_polyed, y_val)
fig = plt.figure(figsize=(18, 6))
ax = fig.gca()
coef = pd.DataFrame({'coef': ridge.coef_}, index=X_train_polyed.columns).sort_values('coef')
coef.plot.bar(ax=ax)
ax.set_xticklabels(ax.get_xticklabels(), rotation=80);
sns.despine()
coef.head(5)
coef.tail(5)
y_pred = ridge.predict(X_val_polyed)
plot_residual(y_val, y_pred)
X_train1, X_val1= get_scaled_features(X_train, selected_quantitive, X2=X_val)
X_train_polyed, X_val_polyed = get_polyed_features(X_train1, selected_quantitive, X_val1, degree=2)
alphas = range(1, 50)
scores_train = []
scores_val = []
for i, alpha in enumerate(alphas):
    ridge = Ridge(alpha=alpha, max_iter=10000).fit(X_train_polyed, y_train)
    scores_train.append(ridge.score(X_train_polyed, y_train))
    scores_val.append(ridge.score(X_val_polyed, y_val))

plot_train_val_score(scores_train, scores_val, alphas)
ridge = Ridge(alpha=17, max_iter=10000)
scores = mcross_validation(ridge, X, Y, need_poly=True)
'ridge', np.mean(scores), scores
plot_residual(y_val, ridge.predict(X_val_polyed))
X_train1, X_val1= get_scaled_features(X_train, selected_quantitive, X2=X_val)
X_train_polyed, X_val_polyed = get_polyed_features(X_train1, selected_quantitive, X_val1)
lasso = Lasso(alpha=1, max_iter=10000).fit(X_train_polyed, y_train)
lasso.score(X_val_polyed, y_val)
y_pred = lasso.predict(X_val_polyed)
plot_residual(y_val, y_pred)
Y.mean()
alphas = np.linspace(0, 0.02, 100)
scores_train = []
scores_val = []
for i, alpha in enumerate(alphas):
    lasso = Lasso(alpha=alpha, max_iter=10000).fit(X_train_polyed, y_train)
    scores_train.append(lasso.score(X_train_polyed, y_train))
    scores_val.append(lasso.score(X_val_polyed, y_val))

plot_train_val_score(scores_train, scores_val, alphas)
lasso = Lasso(alpha=0.00101, max_iter=10000)
scores = mcross_validation(lasso, X, Y, need_poly=True)
'Lasso', format(np.mean(scores), '.4f'), scores
plot_residual(y_val, lasso.predict(X_val_polyed))
from sklearn.neighbors import KNeighborsRegressor
X_train1, X_val1= get_scaled_features(X_train, selected_quantitive, X2=X_val)
X_train_polyed, X_val_polyed = get_polyed_features(X_train1, selected_quantitive, X_val1)
knn = KNeighborsRegressor().fit(X_train_polyed, y_train)
knn.score(X_val_polyed, y_val)
neibors = range(3, 10)
result = pd.DataFrame()
for neibor in neibors:
    scores = mcross_validation(KNeighborsRegressor(n_neighbors=neibor), X, Y, need_poly=True)
    result[str(neibor)] = pd.Series(scores)
result.T.mean(axis=1)
from sklearn.svm import SVR
X_train1, X_val1= get_scaled_features(X_train, selected_quantitive, X2=X_val)
X_train_polyed, X_val_polyed = get_polyed_features(X_train1, selected_quantitive, X_val1)
svr = SVR().fit(X_train_polyed, y_train)
'train score: ', svr.score(X_train_polyed, y_train), ', val score: ', svr.score(X_val_polyed, y_val)
# X数量是1460个，cross vali分成3份。所以每份的数量是486,我们想要获得更高训练分
# 我们先看看分隔的结果，然后再看看如何使用gamma，调节support vector数量。
epsilon = [0, 0.01, 0.1, 0.3, 0.5, 0.7]
C = [1, 5, 50, 100]

def tune_svm(epsilons, Cs, kernels=['rbf'], gammas=['auto'], release=False):
    result = pd.DataFrame()
    if release:
        print('Runtime is in release version, it won\'t run')
        return result

    i = 1
    length = len(epsilons) * len(Cs) * len(kernels) * len(gammas)
    for eps in epsilons:
        for c in Cs:
            if eps < c:
                for k in kernels:
                    for gamma in gammas:
                        scores = mcross_validation(SVR(kernel=k, C=c, epsilon=eps, gamma=gamma, verbose=False), X, Y, need_poly=True)
                        result[str((k, eps, c))] = pd.Series(scores)
                        print('progress {:.2f}'.format(100 * (i / length)))
                        i += 1
    return result

result = tune_svm(epsilon, C, release=RELEASE)
result.T.mean(axis=1).sort_values(ascending=False)
epsilon = [0.01]
C = [1, 4.86, 5, 6, 7, 8, 9]
kernel = ['rbf', 'linear']
result = tune_svm(epsilon, C,  kernels=kernel, release=RELEASE)
result.T.mean(axis=1).sort_values(ascending=False)
epsilon = [0.01, 0.03, 0.05, 0.07, 0.09]
C = [1, 4.86, 5, 6, 7, 8, 9, 11]
kernel = ['rbf']
result = tune_svm(epsilon, C, release=RELEASE)
result.T.mean(axis=1).sort_values(ascending=False).head(5)
gammas = ['auto', 0.0001, 0.01, 0.1]
result = tune_svm([0.07], [8], gammas=gammas, release=RELEASE)
result.T.mean(axis=1).sort_values(ascending=False)
# second tune for gamma
gammas = [0.007, 0.01, 0.03, 0.05, 0.07, 0.1]
result = tune_svm([0.07], [8], gammas=gammas, release=RELEASE)
result.T.mean(axis=1).sort_values(ascending=False)
X_polyed, _ = get_polyed_features(X, selected_quantitive)

dt = DecisionTreeRegressor(random_state=0)
scores = cross_val_score(dt, X_polyed, Y, scoring='r2', cv=3)
scores, '{:.4f}'.format(scores.mean())
from sklearn.ensemble import RandomForestRegressor
rfr = RandomForestRegressor(random_state=0)
scores = cross_val_score(rfr, X_polyed, Y, scoring='r2', cv=3)
scores, '{:.4f}'.format(scores.mean())
from sklearn.model_selection import GridSearchCV
def tune_estimator(name, estimator, params, release=False, scoring='r2'):
    if release:
        return name, ', Runtime is in release version, it won\'t run'
    gscv_training = GridSearchCV(estimator=estimator, param_grid=params, scoring=scoring, n_jobs=-1, cv=3, verbose=1)
    gscv_training.fit(X_polyed, Y)
    return name, gscv_training.best_score_, gscv_training.best_params_
params = {'n_estimators':range(10, 120, 10), 'max_features':['auto', 'sqrt', 'log2']}
tune_estimator('rf', rfr, params, RELEASE)
# second round, since sqrt is best on the upper, so adjust it into ranges.
params = {'n_estimators':range(100, 120), 'max_features':range(20, 29)}
tune_estimator('rf', rfr, params, RELEASE)
from sklearn.ensemble import GradientBoostingRegressor
gbr = GradientBoostingRegressor(random_state=0)
scores = cross_val_score(gbr, X_polyed, Y, scoring='r2', cv=3)
scores, '{:.4f}'.format(scores.mean())
params = {'learning_rate':[0.01, 0.1, 1], 'n_estimators':range(90, 200, 10), 'max_depth':[3, 4, 5], 'max_features':['auto', 'sqrt', 'log2']}
tune_estimator('gbr', gbr, params, RELEASE)
# second tune
params = {'n_estimators':range(140, 210), 'max_depth':[4, 5], 'max_features':range(20, 29)}
tune_estimator('gbr', gbr, params, RELEASE)
X_scaled, _ = get_scaled_features(X, selected_quantitive)
X_polyed, _ = get_polyed_features(X_scaled, selected_quantitive)

models = [Ridge(alpha=17),
          Lasso(alpha=0.00101),
          KNeighborsRegressor(n_neighbors=8),
          SVR(kernel = 'rbf', C=8, epsilon=0.07, gamma=0.01),
          DecisionTreeRegressor(),
          RandomForestRegressor(max_features=28, n_estimators=101), 
          GradientBoostingRegressor(random_state=0, max_depth=4, max_features=24, n_estimators=209)]
names = ['ridge', 'lasso', 'knn', 'svr', 'dtr', 'rfr', 'gbr']

scores = []
i = 1
preds = pd.DataFrame()

for name, model in zip(names, models):
    score = cross_val_score(model, X_polyed, Y, cv=3, scoring='r2', n_jobs=1, verbose=False)
    print('score on {}, mean:{:.4f}, from {}'.format(name, score.mean(), score))
    scores.append(score.mean())
    y_pred = cross_val_predict(model, X_polyed, Y, cv=3, n_jobs=1, verbose=False)
    preds[name] = pd.Series(y_pred)
    print('progress {:.2f}'.format(100 * (i / len(names))))
    i += 1

model_scores = pd.Series(scores, index=names, name='modelscore')
model_scores.sort_values().plot.barh()
from sklearn.model_selection import KFold
test = dataset.loc[id_test, selected_quantitive + selected_qualitive]
test = pd.get_dummies(test, columns=selected_qualitive)
test.shape
test_scaled, _= get_scaled_features(test, selected_quantitive)
test_polyed, _= get_polyed_features(test, selected_quantitive)

n_train=X_polyed.shape[0]
n_test=test_polyed.shape[0]
kf=KFold(n_splits=3,random_state=1,shuffle=True)
def get_oof(clf, X, y, test_X):
    oof_train = np.zeros((n_train, ))
    oof_test_mean = np.zeros((n_test, ))
    # 5 is kf.split
    oof_test_single = np.empty((kf.get_n_splits(), n_test))
    for i, (train_index, val_index) in enumerate(kf.split(X,y)):
        kf_X_train = X.iloc[train_index]
        kf_y_train = y.iloc[train_index]
        kf_X_val = X.iloc[val_index]
        
        clf.fit(kf_X_train, kf_y_train)
        
        oof_train[val_index] = clf.predict(kf_X_val)
        oof_test_single[i,:] = clf.predict(test_X)
    # oof_test_single, 将生成一个5行*n_test列的predict value。那么mean(axis=0), 将对5行，每列的值进行求mean。然后reshape返回   
    oof_test_mean = oof_test_single.mean(axis=0)
    return oof_train.reshape(-1,1), oof_test_mean.reshape(-1,1)

ridge_train, ridge_test = get_oof(models[0], X_polyed, Y, test_polyed)
lasso_train, lasso_test = get_oof(models[1], X_polyed, Y, test_polyed)
knn_train, knn_test = get_oof(models[2], X_polyed, Y, test_polyed)
svr_train, svr_test = get_oof(models[4], X_polyed, Y, test_polyed)
rfr_train, rfr_test = get_oof(models[5], X_polyed, Y, test_polyed)
gbr_train, gbr_test=get_oof(models[6], X_polyed, Y, test_polyed)


y_train_pred_stack = np.concatenate([ridge_train, lasso_train, knn_train, svr_train, rfr_train, gbr_train], axis=1)
y_train_stack = Y.reset_index(drop=True)
y_test_pred_stack = np.concatenate([ridge_test, lasso_test, knn_test, svr_test, rfr_test, gbr_test], axis=1)

y_train_pred_stack.shape, y_train_stack.shape, y_test_pred_stack.shape
params = {'alpha':np.linspace(2, 4, 100)}
reg = Ridge()

gscv_test= GridSearchCV(estimator=reg, param_grid=params, scoring='r2', n_jobs=-1, cv=5, verbose=False)
gscv_test.fit( y_train_pred_stack, y_train_stack)
gscv_test.best_score_, gscv_test.best_params_

# reg = Ridge()
# scores = cross_val_score(reg, preds, Y, scoring='r2', cv=5)
# 'stacking score: ', scores.mean(), ' from ', scores
stacker = gscv_test.best_estimator_
from scipy import stats
se = pd.Series(stats.sem(preds), index=names, name='se')
se.sort_values(ascending=False).plot.bar()
a = pd.concat([model_scores, se], axis=1)
plt.scatter(x=a['modelscore'], y=a['se'], marker='*', cmap='cool')
plt.xlabel('model score')
plt.ylabel('se')
sns.despine()
plt.title('se versus model score')
print('confidence interval: [', np.exp(preds.gbr.mean() - 1.96*se.gbr), np.exp(preds.gbr.mean() + 1.96*se.gbr), ']')
print('real price mean', np.exp(Y.mean()))
plot_residual(Y.values, preds.gbr.values)
sns.residplot(Y.values, Y.values - preds.gbr.values, lowess=True, color="g")
from statsmodels.stats.outliers_influence import variance_inflation_factor

# vif = [variance_inflation_factor(X_polyed.iloc[:, -35:].values, i) for i in range(X_polyed.shape[1])]
# vif2 = pd.DataFrame({'vif': vif}, index=X_polyed.columns)
# print(vif)

# vif2 = pd.read_csv('vif.csv', index_col=0)
# vif2 = vif2[(vif2.vif != np.inf) & (vif2.vif.notna())]
# vif2.plot.bar()
# # 我们试着删除掉VIF > 200，虽说大于5的都算是共线性，当我们的结果是几乎都大于，因此，只能选择部分删除了。
# deleted_columns = vif2[vif2.vif>200].index.values
# fig = plt.figure(figsize=(12, 14))
# sns.heatmap(X_polyed.iloc[:, -35:].corr(), ax=fig.gca(), cmap='viridis')
# a = X_polyed.iloc[:, -35:].corr()
# X_new =  X_polyed[list(set(X_polyed.columns.values) - set(deleted_columns))]
# scores = cross_val_score(gbr, X_new, Y, scoring='r2', cv=5)
# scores, '{:.4f}'.format(scores.mean())
# y_pred = cross_val_predict(gbr, X_new, Y, cv=5, n_jobs=1, verbose=1)

# # stats.sem(y_pred), se.gbr
# plot_residual(Y, y_pred)
X_polyed, _ = get_polyed_features(X, selected_quantitive)
gbr = GradientBoostingRegressor(random_state=0, max_depth=4, max_features=24, n_estimators=209)
# gbr = GradientBoostingRegressor(random_state=0, max_depth=4, max_features=27, n_estimators=187)
y_pred = cross_val_predict(gbr, X_polyed, Y, cv=3, verbose=1)
plot_residual(Y, y_pred)
X_new = X[np.abs(Y - y_pred)<0.7]
Y_new = Y[np.abs(Y - y_pred)<0.7]
X_new_polyed, _ = get_polyed_features(X_new, selected_quantitive)

scores = cross_val_score(gbr, X_new_polyed, Y_new, scoring='r2', cv=3)
print(scores, '{:.4f}'.format(scores.mean()))

y_pred = cross_val_predict(gbr, X_new_polyed, Y_new, cv=3, verbose=1)
plot_residual(Y_new, y_pred)
X_new = X_new[np.abs(Y_new - y_pred)<0.7]
Y_new = Y_new[np.abs(Y_new - y_pred)<0.7]
X_new_polyed, _ = get_polyed_features(X_new, selected_quantitive)

scores = cross_val_score(gbr, X_new_polyed, Y_new, scoring='r2', cv=5)
scores, '{:.4f}'.format(scores.mean())
y_pred2 = cross_val_predict(gbr, X_new_polyed, Y_new, cv=5, verbose=1)
plot_residual(Y_new, y_pred2)
gbr_final = GradientBoostingRegressor(random_state=0, max_depth=4, max_features=24, n_estimators=209)
gbr_final.fit(X_new_polyed, Y_new)
dataset.loc[id_test, selected_quantitive + selected_qualitive].info()
test = dataset.loc[id_test, selected_quantitive + selected_qualitive]
test = pd.get_dummies(test, columns=selected_qualitive)
test.shape
test_scaled, _= get_scaled_features(test, selected_quantitive)
test_polyed, _= get_polyed_features(test, selected_quantitive)
# test_pred = gbr_final.predict(test_polyed)
test_pred = stacker.predict(y_test_pred_stack)
test_pred = np.exp(test_pred)
result_df = pd.DataFrame({'Id': test.index, 'SalePrice':test_pred}).set_index('Id')
result_df.to_csv('predicted_price.csv')
pd.read_csv('predicted_price.csv').head()
pd.read_csv('../input/sample_submission.csv').head()

