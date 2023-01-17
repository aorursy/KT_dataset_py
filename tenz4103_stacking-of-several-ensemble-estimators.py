import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt



import matplotlib.pyplot as plt

from scipy.stats import skew

from scipy.stats.stats import pearsonr

from sklearn.model_selection import train_test_split

from sklearn.ensemble import GradientBoostingRegressor

from sklearn.model_selection import cross_val_score

from sklearn.metrics import mean_squared_error

from sklearn.ensemble import AdaBoostRegressor

from sklearn.tree import DecisionTreeRegressor

from sklearn.model_selection import GridSearchCV

from sklearn.ensemble import ExtraTreesRegressor

from sklearn.ensemble import RandomForestRegressor

from sklearn.ensemble import BaggingRegressor

from sklearn import tree

from sklearn.linear_model import LassoCV



%config InlineBackend.figure_format = 'retina' #set 'png' here when working on notebook

%matplotlib inline
train = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")
all_data = pd.concat((train.loc[:,'MSSubClass':'SaleCondition'],

                      test.loc[:,'MSSubClass':'SaleCondition']))



#log transform the target:

train["SalePrice"] = np.log1p(train["SalePrice"])



#log transform skewed numeric features:

numeric_feats = all_data.dtypes[all_data.dtypes != "object"].index



skewed_feats = train[numeric_feats].apply(lambda x: skew(x.dropna())) #compute skewness

skewed_feats = skewed_feats[skewed_feats > 0.75]

skewed_feats = skewed_feats.index



all_data[skewed_feats] = np.log1p(all_data[skewed_feats])



all_data = pd.get_dummies(all_data)

#filling NA's with the mean of the column:

all_data = all_data.fillna(all_data.mean())

#creating matrices for sklearn:

X_train = all_data[:train.shape[0]]

X_test = all_data[train.shape[0]:]

y = train.SalePrice
clf_gbr = GradientBoostingRegressor(learning_rate=0.075, n_estimators=200)

clf_adb=AdaBoostRegressor(DecisionTreeRegressor(max_depth=4), n_estimators=300, learning_rate = 1.6)

clf_extr=ExtraTreesRegressor(n_estimators=70)

clf_rfr=RandomForestRegressor(n_estimators=200)

clf_bg=BaggingRegressor(n_estimators=100)

clf_tree = tree.DecisionTreeRegressor(min_samples_leaf=8, max_depth=7)

clf_lasso = LassoCV(alphas = [1, 0.1, 0.001, 0.0005])

clf_adls =AdaBoostRegressor(LassoCV(alphas = [1, 0.1, 0.5, 0.05, 0.01, 0.005]), n_estimators=200 )

clf_bgls=BaggingRegressor(LassoCV(alphas = [1, 0.1, 0.5, 0.05, 0.01, 0.005]), n_estimators=100)
clf_gbr.fit(X_train, y)

preds_gbr= clf_gbr.predict(X_train)

clf_adb.fit(X_train, y)

preds_adb= clf_adb.predict(X_train)

clf_extr.fit(X_train, y)

preds_extr= clf_extr.predict(X_train)

clf_rfr.fit(X_train, y)

preds_rfr= clf_rfr.predict(X_train)

clf_bg.fit(X_train, y)

preds_bg= clf_bg.predict(X_train)

clf_tree.fit(X_train, y)

preds_tree= clf_tree.predict(X_train)

clf_lasso.fit(X_train, y)

preds_lasso= clf_lasso.predict(X_train)

clf_adls.fit(X_train, y)

preds_adls= clf_adls.predict(X_train)

clf_bgls.fit(X_train, y)

preds_bgls= clf_bgls.predict(X_train)
preds= pd.DataFrame({"preds_gbr":preds_gbr, "preds_adb":preds_adb, "preds_extr":preds_extr, "preds_rfr":preds_rfr,

                      "preds_bg":preds_bg, "preds_tree":preds_tree, "preds_lasso":preds_lasso, "preds_adls":preds_adls,

                      "preds_bgls":preds_bgls, "true":y})
preds.head()
sns.jointplot( x="preds_adb", y="true", data=preds, kind="reg", size=10, color='black')

sns.jointplot( x="preds_adls", y="true", data=preds, kind="reg", size=10, color='black')

sns.jointplot( x="preds_bg", y="true", data=preds, kind="reg", size=10, color='black')

sns.jointplot( x="preds_bgls", y="true", data=preds, kind="reg", size=10, color='black')

sns.jointplot( x="preds_extr", y="true", data=preds, kind="reg", size=10, color='black')

sns.jointplot( x="preds_gbr", y="true", data=preds, kind="reg", size=10, color='black')

sns.jointplot( x="preds_lasso", y="true", data=preds, kind="reg", size=10, color='black')

sns.jointplot( x="preds_rfr", y="true", data=preds, kind="reg", size=10, color='black')

sns.jointplot( x="preds_tree", y="true", data=preds, kind="reg", size=10, color='black')
estimators=[clf_gbr, clf_adb,  clf_rfr, clf_bg, clf_tree, clf_lasso, clf_adls, clf_bgls]

scores_train=[]



for estimator in estimators:

    scores_train.append(cross_val_score(estimator, X_train, y, cv=5).mean()) 
estimators_labels=['clf_gbr', 'clf_adb',  'clf_rfr', 'clf_bg', 'clf_tree', 'clf_lasso', 'clf_adls', 'clf_bgls']
scores= pd.DataFrame({"estimator":estimators_labels, "score": scores_train}) 
scores
sns.set(style="whitegrid", color_codes=True)

sns.pointplot(x="estimator", y="score", data=scores, color='black')
plt.figure(figsize=(12,10))

foo = sns.heatmap(preds.corr().sort_values('true', ascending=False), vmax=1.0, square=True, annot=True, cmap='BuGn')
X_train.columns
summary = pd.DataFrame(list(zip(X_train.columns, \

    np.transpose(clf_gbr.feature_importances_), \

    np.transpose(clf_adb.feature_importances_), \

    np.transpose(clf_rfr.feature_importances_), \

    np.transpose(clf_tree.feature_importances_), \

    np.transpose(clf_lasso.coef_), 

         )), columns=['Feature', 'gbr', 'adb', 'rfr', 'tree', 'Lasso'])

  

summary['Median'] = summary.median(1)

summary.sort_values('Median', ascending=False)
gbr_features=pd.DataFrame({"Feature":summary.Feature, "gbr":summary.gbr}).sort_values('gbr', ascending=False).head(10)

adb_features=pd.DataFrame({"Feature":summary.Feature, "adb":summary.adb}).sort_values('adb', ascending=False).head(10)

rfr_features=pd.DataFrame({"Feature":summary.Feature, "rfr":summary.rfr}).sort_values('rfr', ascending=False).head(10)

tree_features=pd.DataFrame({"Feature":summary.Feature, "tree":summary.tree}).sort_values('tree', ascending=False).head(10)

lasso_features=pd.DataFrame({"Feature":summary.Feature, "lasso":summary.Lasso}).sort_values('lasso', ascending=False).head(10)
plt.figure(figsize=(12,10))

sns.pointplot(x="Feature", y="adb", data=adb_features, color="r")

sns.pointplot(x="Feature", y="gbr", data=gbr_features)

sns.pointplot(x="Feature", y="rfr", data=rfr_features, color='green')

sns.pointplot(x="Feature", y="tree", data=tree_features, color='yellow')

sns.pointplot(x="Feature", y="lasso", data=lasso_features, color='cyan')

plt.xticks(rotation=90) 
test_gbr= clf_gbr.predict(X_test)

test_adb= clf_adb.predict(X_test)

test_rfr= clf_rfr.predict(X_test)

test_bg= clf_bg.predict(X_test)

test_tree= clf_tree.predict(X_test)

test_lasso= clf_lasso.predict(X_test)

test_adls= clf_adls.predict(X_test)

test_bgls= clf_bgls.predict(X_test)

test_preds= pd.DataFrame({"test_gbr":test_gbr, "test_adb":test_adb, 

                      "test_bg":test_bg, "test_tree":test_tree, "test_lasso":test_lasso, 

                      "test_bgls":test_bgls})
columns=['preds_adb', 'preds_bg', 'preds_bgls', 'preds_gbr', 'preds_lasso', 'preds_tree']

X_preds = preds.loc[:, columns]

y_preds = preds.true
gbr = GradientBoostingRegressor()

lasso = LassoCV(alphas = [1, 0.1, 0.001, 0.0005])

adb=AdaBoostRegressor(DecisionTreeRegressor(max_depth=4))

bagging=BaggingRegressor(n_estimators=100)
gbr.fit(X_preds, y_preds)

lasso.fit(X_preds, y_preds)

adb.fit(X_preds, y_preds)

bagging.fit(X_preds, y_preds)
cross_val_score(lasso, X_preds, y_preds, cv=5).mean()
cross_val_score(gbr, X_preds, y_preds, cv=5).mean()
cross_val_score(adb, X_preds, y_preds, cv=5).mean()
cross_val_score(bagging, X_preds, y_preds, cv=5).mean()
gbr_fin=gbr.predict(X_preds)

lasso_fin=lasso.predict(X_preds)

adb_fin=adb.predict(X_preds)

bagging_fin=bagging.predict(X_preds)

final= pd.DataFrame({"gbr":gbr_fin, "lasso":lasso_fin, "adb":adb_fin, "bagging":bagging_fin, "target":y})
final.head()
fig=sns.jointplot( x="gbr", y="target", data=final, kind="reg", size=10, color='black')

fig=sns.jointplot( x="adb", y="target", data=final, kind="reg", size=10, color='black')

fig=sns.jointplot( x="lasso", y="target", data=final, kind="reg", size=10, color='black')

fig=sns.jointplot( x="bagging", y="target", data=final, kind="reg", size=10, color='black')
lasso_pred = np.expm1(lasso.predict(X_preds))

adb_pred = np.expm1(adb.predict(X_preds))

gbr_pred = np.expm1(gbr.predict(X_preds))

bagging_pred = np.expm1(bagging.predict(X_preds))

target=np.expm1(y)
pred_exp= pd.DataFrame({"gbr":gbr_pred, "lasso":lasso_pred, "adb":adb_pred, "bagging": bagging_pred, "target":target})
fig=sns.jointplot( x="gbr", y="target", data=pred_exp, kind="resid", size=10, color='black')

fig=sns.jointplot( x="adb", y="target", data=pred_exp, kind="resid", size=10, color='black')

fig=sns.jointplot( x="lasso", y="target", data=pred_exp, kind="resid", size=10, color='black')

fig=sns.jointplot( x="bagging", y="target", data=pred_exp, kind="resid", size=10, color='black')
prediction=np.expm1(bagging.predict(test_preds))
solution = pd.DataFrame({"id":test.Id, "SalePrice":prediction})

solution.to_csv("so.csv", index = False)