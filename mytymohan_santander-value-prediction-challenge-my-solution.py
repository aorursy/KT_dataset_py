import xgboost as xgb
import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
import warnings as ws
ws.simplefilter("ignore")
#Loading required packages for analysis
import numpy as np
import pandas as pd

#Reading the training data
vp_train = pd.read_csv("../input/train.csv", header='infer')
vp_test = pd.read_csv("../input/test.csv", header='infer')
from scipy.stats import skew
vp_train_log_x = np.log1p(vp_train.iloc[:,2:])
vp_test_log_x = np.log1p(vp_test.iloc[:,1:])
vp_train_y = vp_train.iloc[:,1]
model = xgb.XGBRegressor(colsample_bytree=0.4,
                 gamma=0,                 
                 learning_rate=0.07,
                 max_depth=3,
                 min_child_weight=1.5,
                 n_estimators=1000,                                                                    
                 reg_alpha=0.75,
                 reg_lambda=0.45,
                 subsample=0.6,
                 seed=42)
from collections import OrderedDict
model.fit(vp_train_log_x, vp_train_y)
xgb_fea_imp=pd.DataFrame(list(model.get_booster().get_fscore().items()),columns=['feature','importance']).sort_values('importance', ascending=False)
#Selecting features with importance greater than 10
feature_g10 = xgb_fea_imp[xgb_fea_imp.importance >=10]
#Relevent Feature list
feature_list = list(feature_g10.feature)
gbm_param_grid = {
     'colsample_bytree': np.linspace(0.5, 0.9, 5),
     'n_estimators':[100, 200],
     'max_depth': [10, 15, 20, 25]
}

gbm = xgb.XGBRegressor()

grid_mse = GridSearchCV(estimator = gbm, param_grid = gbm_param_grid, scoring = 'neg_mean_squared_error', cv = 5, verbose = 1)
grid_mse.fit(vp_train_log_x[feature_list], vp_train_y)
pred = grid_mse.predict(vp_test_log_x[feature_list])
#Saving the final results
y_pred_final = pd.DataFrame({'target1':pred})
x_id = vp_test.loc[:,['ID']]
result = pd.concat([x_id, y_pred_final], axis = 1, ignore_index=True)

result.columns = ['ID', 'target1']
result['target'] = result['target1'].abs()
result = result.drop(['target1'], axis = 1)
result.to_csv('Submission_7_Aug_2018_2.csv', sep=',', index=False)