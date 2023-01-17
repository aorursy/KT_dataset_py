!pip install ultimate==2.1.2
from __future__ import print_function
from ultimate.mlp import MLP

import numpy as np
import sys, random

ratio = 2.2

X = np.random.rand(50000, 5).astype("float32") * 2 - 1 
Y = 50*(X[:,0]**3) + 20*(X[:,1]**2) + ratio*X[:,2] + 1*X[:,3] + 0*X[:,4]

print("X", X.shape, X.min(), X.max())
print("Y", Y.shape, Y.min(), Y.max())

param = {
    'loss_type': 'mse',
    'layer_size': [X.shape[1],16,16,16,1],
    'activation': 'am2',
    'output_range': [0, 1],
    'output_shrink': 0.001,
    'importance_mul': 0.0001,
    'importance_out': True,
    'rate_init': 0.02, 
    'rate_decay': 0.9, 
    'epoch_train': 50, 
    'epoch_decay': 1,
    'verbose': 0,
}

mlp = MLP(param).fit(X, Y)

print("feature importances:", mlp.feature_importances_)

print("nn ratio", mlp.feature_importances_[2]/mlp.feature_importances_[3], ratio)

import lightgbm as lgb

gbm = lgb.LGBMRegressor(objective='regression',
                        num_leaves=2**7, # 32
                        learning_rate=0.05,
                        n_estimators=100)

gbm.fit(X, Y, eval_metric='rmse')  # rmse l1

print('feature importances:', list(gbm.feature_importances_))
print("lightgbm ratio", gbm.feature_importances_[2]/gbm.feature_importances_[3], ratio)