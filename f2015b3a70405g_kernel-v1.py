import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
trdf = pd.read_csv('train.csv')
trdf.head()
trdf.nunique()
trdf.corr()
trdf.shape

trdf.info()
trdf.isnull().values.any()
trdf = trdf.drop_duplicates()

trdf.info()
trdf.corr()['AveragePrice'].abs().sort_values()
corr = trdf.corr(method="kendall")



# Generate a mask for the upper triangle

mask = np.zeros_like(corr, dtype=np.bool)

mask[np.triu_indices_from(mask)] = True



# Set up the matplotlib figure

f, ax = plt.subplots(figsize=(11, 9))



# Generate a custom diverging colormap

cmap = sns.diverging_palette(220, 10, as_cmap=True)



# Draw the heatmap with the mask and correct aspect ratio

sns.heatmap(corr, mask=mask, cmap=cmap, vmax=1, center=0.5,

            square=True, linewidths=.5, cbar_kws={"shrink": .5},annot=True)



plt.show()
trdfc = trdf.copy()

##trdfc = pd.get_dummies(trdfc, columns = ['year'])

trdfc = trdfc.drop(columns = ['Total Bags'])

trdfc.head()
X = trdfc.drop(['AveragePrice'],axis=1)

y = trdfc['AveragePrice'].tolist()



from sklearn.model_selection import train_test_split

Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.33)
from sklearn.preprocessing import MinMaxScaler

m = MinMaxScaler()



m.fit_transform(X)

from sklearn.ensemble import RandomForestRegressor

regr = RandomForestRegressor(random_state=42)



X_tran = m.transform(Xtr)

X_val= m.transform(Xte)

from sklearn.model_selection import GridSearchCV

from sklearn.metrics import make_scorer

from sklearn.metrics import r2_score



parameters = {'n_estimators' : [30,40,60,100,140, 170, 250], 'max_features': [2,3,5,6,9]}

scorer = make_scorer(r2_score)

grid_obj = GridSearchCV(regr,parameters,scoring=scorer, n_jobs=6)

grid_fit = grid_obj.fit(X_tran,ytr)

best_clf = grid_fit.best_estimator_

unoptimized_predictions = (regr.fit(X_tran, ytr)).predict(X_val)      

optimized_predictions = best_clf.predict(X_val)       



acc_unop = r2_score(yte, unoptimized_predictions)*100       

acc_op = r2_score(yte, optimized_predictions)*100         



print(acc_op)
from sklearn.metrics import mean_squared_error

print((mean_squared_error(yte,optimized_predictions)))
"""from sklearn.linear_model import LinearRegression

lr = LinearRegression()

lr.fit(X_tran,ytr)

pred = lr.predict(X_val)

from sklearn.model_selection import cross_val_score

scorer = make_scorer(r2_score)

scores = cross_val_score(lr,X_tran, ytr, cv=5, scoring = scorer)

print(scores.mean())"""
"""import xgboost

xgb = xgboost.XGBRegressor()

xgb.fit(X_tran, ytr)



parameters = {'n_estimators' : [30,40,60,100,140, 170], 'learning_rate': [0.01, 0.03,0.05,0.08,0.1,0.3]}

scorer = make_scorer(r2_score)

grid_obj = GridSearchCV(xgb,parameters,scoring=scorer)

grid_fit = grid_obj.fit(X_tran,ytr)

best_xgb = grid_fit.best_estimator_

unoptimized_predictions = xgb.predict(X_val)      

optimized_predictions = best_xgb.predict(X_val)       



acc_unop = r2_score(yte, unoptimized_predictions)*100       

acc_op = r2_score(yte, optimized_predictions)*100  """
"""print(acc_unop)

print(acc_op)

print(best_xgb)

print((mean_squared_error(yte,optimized_predictions)))"""
"""from sklearn.ensemble import GradientBoostingRegressor

clf = GradientBoostingRegressor(n_estimators = 30,max_depth=4)        #Initialize the classifier object



parameters = {'loss':['ls', 'lad', 'huber', 'quantile'],'learning_rate':[0.01,0.05,0.1,0.5,1]}    #Dictionary of parameters



scorer = make_scorer(r2_score)         #Initialize the scorer using make_scorer



grid_obj = GridSearchCV(clf,parameters,scoring=scorer)

grid_fit = grid_obj.fit(X_tran,ytr)

best_clf = grid_fit.best_estimator_

##unoptimized_predictions = clf.predict(X_val)      

optimized_predictions = best_clf.predict(X_val)       



##acc_unop = r2_score(yte, unoptimized_predictions)*100       

acc_op = r2_score(yte, optimized_predictions)*100 

print(acc_op)"""
test = pd.read_csv('test.csv')

testc = test.copy()

testc = test.drop(columns = ['Total Bags'])

##testc = pd.get_dummies(testc, columns = ['year'])

##testc.info()

##X.info()

best_clf.fit(m.transform(X), y)

testc_tr = m.transform(testc)

pred = best_clf.predict(testc_tr)



pre = pd.Series(data=pred)



newd = pd.concat([test['id'],pre], axis=1)

newd.rename(index=str, columns = {0:'AveragePrice'}, inplace=True)
newd.to_csv("submit.csv", index=False)