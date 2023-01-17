import pandas as pd
import numpy as np
from sklearn.metrics import cohen_kappa_score,confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from functools import partial 
import scipy as sp
df = pd.read_csv('../input/train/train.csv')
dfDropped = df.drop(['Name','RescuerID','Description','PetID','AdoptionSpeed'],axis=1)
X = dfDropped.values
y = df["AdoptionSpeed"]
class OptimizedRounder(object):
    def __init__(self):
        self.coef_ = 0

    def _kappa_loss(self, coef, X, y):
        X_p = np.copy(X)
        for i, pred in enumerate(X_p):
            if pred < coef[0]:
                X_p[i] = 0
            elif pred >= coef[0] and pred < coef[1]:
                X_p[i] = 1
            elif pred >= coef[1] and pred < coef[2]:
                X_p[i] = 2
            elif pred >= coef[2] and pred < coef[3]:
                X_p[i] = 3
            else:
                X_p[i] = 4

        ll = cohen_kappa_score(y, X_p,weights='quadratic')
        return -ll

    def fit(self, X, y):
        loss_partial = partial(self._kappa_loss, X=X, y=y)
        initial_coef = [0.5, 1.5, 2.5, 3.5]
        self.coef_ = sp.optimize.minimize(loss_partial, initial_coef, method='nelder-mead')

    def predict(self, X, coef):
        X_p = np.copy(X)
        for i, pred in enumerate(X_p):
            if pred < coef[0]:
                X_p[i] = 0
            elif pred >= coef[0] and pred < coef[1]:
                X_p[i] = 1
            elif pred >= coef[1] and pred < coef[2]:
                X_p[i] = 2
            elif pred >= coef[2] and pred < coef[3]:
                X_p[i] = 3
            else:
                X_p[i] = 4
        return X_p

    def coefficients(self):
        return self.coef_['x']
#from sklearn.model_selection import StratifiedKFold
#def skfold_gen(X,y,n_splits=5, random_state=42, shuffle=True):
#    for train, test in skf.split(X, y):
#        yield train, test
#clf = GridSearchCV(RandomForestRegressor()
#                   , dict(max_depth=[5,10,15,20], n_estimators=[ 200, 250,300,350,400,450]), cv=skfold_gen(X,y),
#                 scoring='neg_mean_squared_error',verbose=2,n_jobs=3)

#clf.fit(X,y)
# (-1.1598095155985777, {'max_depth': 10, 'n_estimators': 300})
# clf.best_score_,clf.best_params_
# optr.fit(clf.predict(X),y)
# optr.coefficients()
# array([0.44385399, 2.07305103, 2.47256317, 2.93407633])
rfr = RandomForestRegressor(max_depth=10, n_estimators=300)
rfr.fit(X,y)
optr = OptimizedRounder()
optr.fit(rfr.predict(X),y)
optr.coefficients()
df_t = pd.read_csv('../input/test/test.csv')
X_t = df_t.drop(['Name','RescuerID','Description','PetID'],axis=1).values
X_t.shape
import pandas as pd

submission = pd.DataFrame(dict(PetID=df_t['PetID'], AdoptionSpeed=optr.predict(rfr.predict(X_t), optr.coefficients()).astype(int)))
submission.to_csv('submission.csv', index=False)
!head submission.csv
