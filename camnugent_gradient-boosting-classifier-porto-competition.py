import pandas as pd

import numpy as np

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.model_selection import GridSearchCV
test_dat = pd.read_csv('../input/test.csv')

train_dat = pd.read_csv('../input/train.csv')

submission = pd.read_csv('../input/sample_submission.csv')

train_dat.info()
train_dat.describe()



train_y = train_dat['target']

train_x = train_dat.drop(['target', 'id'], axis = 1)

"""

gb_params = {

    'n_estimators' : [100,200,300],

    'learning_rate' : [.1,.2,.3],

    'max_depth' : [3,5,7]

}



gb_class = GradientBoostingClassifier()



gb_grid = GridSearchCV(gb_class, gb_params, cv = 5, n_jobs=-1)

gb_grid.fit(train_x, train_y)



gb_grid.best_estimator_

"""
gb_opt = GradientBoostingClassifier(criterion='friedman_mse', init=None,

                            learning_rate=0.1, loss='deviance', max_depth=3,

                            max_features=None, max_leaf_nodes=None, min_impurity_split=None,

                            min_samples_leaf=1, min_samples_split=2,

                            min_weight_fraction_leaf=0.0, n_estimators=100,

                            presort='auto', random_state=None, subsample=1.0, verbose=0,

                            warm_start=False)

    

gb_opt.fit(train_x, train_y)

test_y_gb = gb_opt.predict_proba(test_x)



gb_out = submission

gb_out['target'] = test_y_gb



gb_out['target'] = 1-gb_out['target']
gb_out.to_csv('gb_predictions1.csv', index=False, float_format='%.4f')
