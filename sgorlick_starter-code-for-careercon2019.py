from sklearn.utils import check_array

from sklearn.compose import ColumnTransformer

from sklearn.preprocessing import StandardScaler,MinMaxScaler,QuantileTransformer

from sklearn.base import BaseEstimator,TransformerMixin,RegressorMixin

from sklearn.pipeline import make_pipeline

from sklearn.model_selection import StratifiedKFold

from sklearn.metrics import accuracy_score

import pandas as pd

import numpy as np

import lightgbm as lgb

import matplotlib.pyplot as plt

from scipy.stats import skew,kurtosis
train = pd.read_csv('../input/X_train.csv',index_col=1)

train.drop('row_id',axis=1,inplace=True)

test = pd.read_csv('../input/X_test.csv',index_col=1)

test.drop('row_id',axis=1,inplace=True)

target =  pd.read_csv('../input/y_train.csv')

sub = pd.read_csv('../input/sample_submission.csv')
def fe_step0 (actual):

    

    # https://www.mathworks.com/help/aeroblks/quaternionnorm.html

    # https://www.mathworks.com/help/aeroblks/quaternionmodulus.html

    # https://www.mathworks.com/help/aeroblks/quaternionnormalize.html

        

    actual['norm_quat'] = (actual['orientation_X']**2 + actual['orientation_Y']**2 + actual['orientation_Z']**2 + actual['orientation_W']**2)

    actual['mod_quat'] = (actual['norm_quat'])**0.5

    actual['norm_X'] = actual['orientation_X'] / actual['mod_quat']

    actual['norm_Y'] = actual['orientation_Y'] / actual['mod_quat']

    actual['norm_Z'] = actual['orientation_Z'] / actual['mod_quat']

    actual['norm_W'] = actual['orientation_W'] / actual['mod_quat']

    

    return actual



# https://stackoverflow.com/questions/53033620/how-to-convert-euler-angles-to-quaternions-and-get-the-same-euler-angles-back-fr?rq=1

def quaternion_to_euler(x, y, z, w):

    import math

    t0 = +2.0 * (w * x + y * z)

    t1 = +1.0 - 2.0 * (x * x + y * y)

    X = math.atan2(t0, t1)



    t2 = +2.0 * (w * y - z * x)

    t2 = +1.0 if t2 > +1.0 else t2

    t2 = -1.0 if t2 < -1.0 else t2

    Y = math.asin(t2)



    t3 = +2.0 * (w * z + x * y)

    t4 = +1.0 - 2.0 * (y * y + z * z)

    Z = math.atan2(t3, t4)



    return X, Y, Z



def fe_step1 (actual):

    """Quaternions to Euler Angles"""

    

    x, y, z, w = actual['norm_X'].tolist(), actual['norm_Y'].tolist(), actual['norm_Z'].tolist(), actual['norm_W'].tolist()

    nx, ny, nz = [], [], []

    for i in range(len(x)):

        xx, yy, zz = quaternion_to_euler(x[i], y[i], z[i], w[i])

        nx.append(xx)

        ny.append(yy)

        nz.append(zz)

    

    actual['euler_x'] = nx

    actual['euler_y'] = ny

    actual['euler_z'] = nz

    return actual





train = fe_step0(train)

test = fe_step0(test)



train = fe_step1(train)

test = fe_step1(test)





def extra_feats(data):

    data['totl_anglr_vel'] = (data['angular_velocity_X']**2 + data['angular_velocity_Y']**2 + data['angular_velocity_Z']**2)** 0.5

    data['totl_linr_acc'] = (data['linear_acceleration_X']**2 + data['linear_acceleration_Y']**2 + data['linear_acceleration_Z']**2)**0.5

    data['totl_xyz'] = (data['orientation_X']**2 + data['orientation_Y']**2 + data['orientation_Z']**2)**0.5

    data['acc_vs_vel'] = data['totl_linr_acc'] / data['totl_anglr_vel']

    return data



train = extra_feats(train)

test = extra_feats(test)



for i in range(1,train.shape[1]):

    train=pd.concat([train,train.iloc[:,i].diff().fillna(0)],axis=1)

    test=pd.concat([test,test.iloc[:,i].diff().fillna(0)],axis=1)

    

    

train.columns=list(range(train.shape[1]))

test.columns=list(range(test.shape[1]))
train_pivot=pd.pivot_table(data=train,index=train.index,columns=train.columns[0])

test_pivot=pd.pivot_table(data=test,index=test.index,columns=train.columns[0])



train_stats=[]

for i in np.linspace(0,128*(train.shape[1]-2),train.shape[1]-1).astype(int):

    train_stats.append( np.mean(train_pivot.iloc[:,i:i+128],axis=1).values )

    train_stats.append( np.median(train_pivot.iloc[:,i:i+128],axis=1) )

    train_stats.append( np.sum(train_pivot.iloc[:,i:i+128],axis=1).values )    

    train_stats.append( np.max(train_pivot.iloc[:,i:i+128],axis=1).values )

    train_stats.append( np.min(train_pivot.iloc[:,i:i+128],axis=1).values )

    train_stats.append( np.max(train_pivot.iloc[:,i:i+128],axis=1).values - np.min(train_pivot.iloc[:,i:i+128],axis=1).values )

    train_stats.append( (np.max(train_pivot.iloc[:,i:i+128],axis=1).values / np.min(train_pivot.iloc[:,i:i+128],axis=1)).fillna(0).values )

    train_stats.append( np.std(train_pivot.iloc[:,i:i+128],axis=1).values )

    train_stats.append( np.var(train_pivot.iloc[:,i:i+128],axis=1).values )

    train_stats.append( skew(train_pivot.iloc[:,i:i+128],axis=1) )

    train_stats.append( kurtosis(train_pivot.iloc[:,i:i+128],axis=1) )

train_pivot=pd.concat([train_pivot,pd.DataFrame(train_stats).T],axis=1)



test_stats=[]

for i in np.linspace(0,128*(test.shape[1]-2),test.shape[1]-1).astype(int):

    test_stats.append( np.mean(test_pivot.iloc[:,i:i+128],axis=1).values )

    test_stats.append( np.median(test_pivot.iloc[:,i:i+128],axis=1) )

    test_stats.append( np.sum(test_pivot.iloc[:,i:i+128],axis=1).values )    

    test_stats.append( np.max(test_pivot.iloc[:,i:i+128],axis=1).values )

    test_stats.append( np.min(test_pivot.iloc[:,i:i+128],axis=1).values )

    test_stats.append( np.max(test_pivot.iloc[:,i:i+128],axis=1).values - np.min(test_pivot.iloc[:,i:i+128],axis=1).values )

    test_stats.append( (np.max(test_pivot.iloc[:,i:i+128],axis=1).values / np.min(test_pivot.iloc[:,i:i+128],axis=1)).fillna(0).values )

    test_stats.append( np.std(test_pivot.iloc[:,i:i+128],axis=1).values )

    test_stats.append( np.var(test_pivot.iloc[:,i:i+128],axis=1).values )

    test_stats.append( skew(test_pivot.iloc[:,i:i+128],axis=1) )

    test_stats.append( kurtosis(test_pivot.iloc[:,i:i+128],axis=1) )

test_pivot=pd.concat([test_pivot,pd.DataFrame(test_stats).T],axis=1)





train_pivot.columns=list(range(train_pivot.shape[1]))

test_pivot.columns=list(range(test_pivot.shape[1]))
target_working = target['surface'].map({

    'concrete':0,

    'soft_pvc':1,

    'wood':2,

    'tiled':3,

    'fine_concrete':4,

    'hard_tiles_large_space':5,

    'soft_tiles':6,

    'carpet':7,

    'hard_tiles':8,

})
train_pivot=check_array(np.where(train_pivot== np.inf,0,train_pivot))

test_pivot=check_array(np.where(test_pivot== np.inf,0,test_pivot))
class LGBClassifierCV(BaseEstimator,RegressorMixin) :

    def __init__(self,fit_params=None,n_splits=3) :#,feature_name=feature_name) :

        self.fit_params = fit_params

        self.n_splits = n_splits

        #self.feature_name = feature_name

    def fit(self,X,y) : 

        print('begin fit . . .')

        self.oof_preds = np.zeros((X.shape[0],target['group_id'].nunique())) 

        self.M = []

        #X = np.where(X == np.inf,0,X)

        X = check_array(X,force_all_finite ='allow-nan')

        y = y.values

        folds = StratifiedKFold(n_splits= self.n_splits, shuffle=True,random_state=1600)

        M_fit=0

        M_cv=0

        for n_fold, (train_idx, valid_idx) in enumerate(folds.split(X, y)):

            dtrain = lgb.Dataset(data=X[train_idx,:], 

                                 label=y[train_idx], 

                                 #feature_name=self.feature_name,

                                 #categorical_feature=['col'],

                                 )

            dvalid = lgb.Dataset(data=X[valid_idx,:], 

                                 label=y[valid_idx], 

                                 #feature_name=self.feature_name,

                                 #categorical_feature=['col'],

                                 )

            m = lgb.train(

                train_set=dtrain,

                valid_sets=[dtrain, dvalid],

                params=self.fit_params,

                num_boost_round=100000,

                early_stopping_rounds=100,

                verbose_eval=False

            )

            self.M.append(m)

            self.oof_preds[valid_idx,:] = m.predict(X[valid_idx,:])

            print(n_fold, accuracy_score(y[valid_idx],np.argmax(self.oof_preds[valid_idx,:],axis=1)))

        print('final', accuracy_score(y,np.argmax(self.oof_preds,axis=1)))

        return self

    @property

    def cv_scores_(self):

        return self.oof_preds

    def predict(self,X) :

        sub_preds=np.zeros((X.shape[0],target['group_id'].nunique()))

        #X = np.where(X == np.inf,0,X)

        X = check_array(X,force_all_finite ='allow-nan')

        for m in self.M :

            sub_preds=np.add(sub_preds,m.predict(X)/5)

        return sub_preds
fit_params = {



    'num_leaves': 18,

    'min_data_in_leaf': 40,

    'objective': 'multiclass',

    'metric': 'multi_error',

    'max_depth': 8,

    'learning_rate': 0.01,

    "boosting": "gbdt",

    "bagging_freq": 5,

    "bagging_fraction": 0.812667,

    "bagging_seed": 11,

    "verbosity": -1,

    'reg_alpha': 0.2,

    'reg_lambda': 0,

    "num_class": target['group_id'].nunique(),

    'nthread': -1

}



M=make_pipeline(QuantileTransformer(output_distribution='normal'),StandardScaler(),MinMaxScaler(),LGBClassifierCV(fit_params=fit_params) )

#M=make_pipeline(StandardScaler(),LGBClassifierCV(fit_params=fit_params) )

target

M.fit(train_pivot,target['group_id'])#target_working)
for i in range(target['group_id'].nunique()) :

    print(i,np.where(target['group_id'] == i,1,0).sum(),np.round(np.where(np.argmax(M.named_steps['lgbclassifiercv'].cv_scores_,axis=1)[target['group_id'] == i]==i,1,0).mean(),2) )

oof_preds_grp_to_label=pd.Series(np.argmax(M.named_steps['lgbclassifiercv'].cv_scores_,axis=1)).map({a:b for a,b in target.loc[:,['group_id','surface']].drop_duplicates().values})
oof_preds_grp_to_label_map = oof_preds_grp_to_label.map({

    'concrete':0,

    'soft_pvc':1,

    'wood':2,

    'tiled':3,

    'fine_concrete':4,

    'hard_tiles_large_space':5,

    'soft_tiles':6,

    'carpet':7,

    'hard_tiles':8,

})



for i in range(target_working.nunique()) :

    print(i,np.where(target_working == i,1,0).sum(),np.round(np.where(oof_preds_grp_to_label_map[target_working == i]==i,1,0).mean(),2) )
sub_preds=pd.Series(np.argmax(M.predict(test_pivot),axis=1)).map({a:b for a,b in target.loc[:,['group_id','surface']].drop_duplicates().values})
sub['surface']=sub_preds.values
sub.to_csv('submittal.csv',index=False)