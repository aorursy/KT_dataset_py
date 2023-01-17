!pip install lime
!pip install lightgbm

import pandas as pd
from sklearn import preprocessing, model_selection
import random, os, torch
import numpy as np
import lightgbm as lgb
from pandas.api.types import is_numeric_dtype
from sklearn.metrics import f1_score
import sklearn
import lightgbm as lgb
import xgboost as xgb

from lightgbm import LGBMClassifier

seed = 1234

def lgb_f1_score(y_hat, data):
    y_true = data.get_label()
    print(y_hat.shape)
    y_hat = np.round(y_hat) # scikits f1 doesn't like probabilities
    return 'f1', f1_score(y_true, y_hat), True

def seed_everything(seed=seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
seed_everything()
sklearn.__version__
codebook = pd.read_csv('../input/costa-rican-household-poverty-prediction/codebook.csv')
codebook
train = pd.read_csv('../input/costa-rican-household-poverty-prediction/train.csv')
train_size = train.shape[0]
train
train.info()
test = pd.read_csv('../input/costa-rican-household-poverty-prediction/test.csv')
test
alldata = pd.concat([train, test], 0)
alldata
y = np.array(alldata.Target[:train_size], dtype=np.int) - 1
alldata = alldata.drop('Target', axis=1)
alldata
too_much_na = []
categorical_feat = []
id_encoder = preprocessing.LabelEncoder()
alldata.Id = id_encoder.fit_transform(alldata.Id)
for feat in alldata.columns:
    if not is_numeric_dtype(alldata[feat]):
        categorical_feat.append(feat)
    if alldata[feat].isna().sum() / alldata.shape[0] > 0:
        too_much_na.append(feat)
    

alldata = alldata.drop(too_much_na, axis=1)
alldata
encoders = dict()
for feat in categorical_feat:
    encoders[feat] = preprocessing.LabelEncoder()
    alldata[feat] = encoders[feat].fit_transform(alldata[feat])

categorical_feat
train = alldata.iloc[:train_size]
test = alldata.iloc[train_size:]
train, test
train_all = pd.DataFrame(train, copy=True)
train_all['Target'] = y
train_all
train_all.groupby(['area1', 'Target']).count()
train_all.groupby(['male', 'Target']).count()
tX, vX, ty, vy = model_selection.train_test_split(train, y, test_size=0.1, random_state=seed, stratify=y)
train_data = lgb.Dataset(tX, label=ty)
eval_data = lgb.Dataset(vX, label=vy)
tX.shape, vX.shape, ty.shape, vy.shape
class LGB:
    def __init__(self, param, lgb_config, n_round):
        self.param = param
        self.lgb_config = lgb_config
        self.n_round = n_round
        
    def fit(self, X, y):
        data = lgb.Dataset(X, y)
        self.gbm = lgb.train(self.param, data, **lgb_config)
        
    def predict(self, X):
        return self.gbm.predict(X)

def evaluate_macroF1_lgb(predictions, truth):  
    # this follows the discussion in https://github.com/Microsoft/LightGBM/issues/1483
    pred_labels = predictions.argmax(axis=1)
    truth = truth.get_label()
    f1 = f1_score(truth, pred_labels, average='macro')
    return ('macroF1', 1-f1) 

def learning_rate_power_0997(current_iter):
    base_learning_rate = 0.1
    min_learning_rate = 0.02
    lr = base_learning_rate  * np.power(.995, current_iter)
    return max(lr, min_learning_rate)

opt_parameters = {'max_depth':35, 
                  'eta':0.15, 
                  'silent':1, 
                  'objective':'multi:softmax', 
                  'min_child_weight': 2, 
                  'num_class': 4, 
                  'gamma': 2.5, 
                  'colsample_bylevel': 1, 
                  'subsample': 0.95, 
                  'colsample_bytree': 0.85, 
                  'reg_lambda': 0.35 }
fit_params={"early_stopping_rounds":500,
            "eval_metric" : evaluate_macroF1_lgb, 
            "eval_set" : [(tX, ty), (vX, vy)],
#             'verbose': 50,
}
clf =  xgb.XGBClassifier(random_state=seed, n_estimators=300, learning_rate=0.15, n_jobs=4, **opt_parameters)
clf.fit(tX, ty, **fit_params)
f1 = f1_score(clf.predict(vX), vy, average='macro')
f1
final_y = clf.predict(test)
final_y
test
sample_sub = pd.read_csv('../input/costa-rican-household-poverty-prediction/sample_submission.csv')
sample_sub
test
sub = pd.DataFrame(test, copy=True).loc[:, ['Id', 'hacdor']]
sub['Target'] = final_y + 1
sub = sub.drop('hacdor', 1)
sub
sub.Id = id_encoder.inverse_transform(sub.Id)
sub
sub.to_csv('submission.csv', index=False)