import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import scipy.sparse as sps
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
start = time.time()
vetor_train_target = sps.load_npz('../input/arquivos-full/sparse_matrix_train_target.npz').tocsr()[:150000]
matriz_train       = sps.load_npz('../input/arquivos-full/sparse_matrix_train.npz').tocsr()[:150000]
matriz_test        = sps.load_npz('../input/arquivos-full/sparse_matrix_test.npz' ).tocsr()
end = time.time()

%env JOBLIB_TEMP_FOLDER=/tmp

print('Tempo de carregamento das Matrizes: '+str("%.2f" % (end - start))+'s\n')
from sklearn.model_selection import ParameterGrid
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.dummy import DummyRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.utils import check_random_state
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import BaggingRegressor
from mlxtend.regressor import StackingRegressor
from lightgbm import LGBMRegressor


X_train= matriz_train 
y_train= vetor_train_target 
X_test  = matriz_test 
#X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.33, random_state=2)


lr = LinearRegression()
svr_lin = SVR(kernel='linear')
ridge = Ridge(random_state=1)
svr_rbf = SVR(kernel='rbf')

estimators_L1 = [
    #( ExtraTreesRegressor(random_state=0, n_jobs=-1, 
    #                           n_estimators=40, max_depth=3)),
        
    (BaggingRegressor(Ridge(alpha = 29.0,random_state=5), bootstrap = False, bootstrap_features= True, max_features = 1.0, max_samples = 1.0, n_jobs=-1)),
        
    (LGBMRegressor(boosting_type='gbdt', class_weight=None, colsample_bytree=1.0,
                     learning_rate=0.05, max_depth=-1, min_child_samples=20,
                     min_child_weight=0.001, min_split_gain=0.0, n_estimators=25,
                     n_jobs=-1, num_leaves=270, objective='regression', random_state=None,
                     reg_alpha=0.0, reg_lambda=0.0, silent=True, subsample=1.0,
                     subsample_for_bin=200000, subsample_freq=1, task='train')),
    (XGBRegressor(n_estimators=30))
]
start = time.time()
final_estimator = BaggingRegressor(ExtraTreesRegressor(random_state=0, n_jobs=-1,
                               n_estimators=35, max_depth=4), bootstrap = False,
                                   bootstrap_features= True, max_features = 1.0, max_samples = 0.5, n_jobs=-1,random_state=0)

stregr = StackingRegressor(regressors=estimators_L1, 
                           meta_regressor=final_estimator)



stregr.fit(X_train, y_train.toarray().ravel())
print("Feito fit")
y_pred = stregr.predict(X_test)
end = time.time()

print('Tempo de processamento: ' + str("%.2f" % (end - start))+'s\n')       

#rmse = np.sqrt(mean_squared_error(y_test.todense(), y_pred))
#print(rmse)
df_test_item_id  = pd.read_csv('../input/arquivos-full/test.csv' , encoding='utf8')['item_id'].head(y_pred.shape[0])
df_y_pred = pd.DataFrame(y_pred, columns = ['deal_probability'])
df_resultado = pd.concat([df_test_item_id,df_y_pred],axis =1)
df_resultado.to_csv('submission.csv', encoding='utf-8', index=False)

