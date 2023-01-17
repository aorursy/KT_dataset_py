# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
def Prepare_data(label_name,sample_id):
    label=pd.read_csv('../input/train.csv').loc[:,label_name]
    predictors=pd.read_csv('../input/train.csv').drop(columns=[label_name,sample_id])
    
    test_id=pd.read_csv('../input/test.csv').loc[:,sample_id]
    test_predictors=pd.read_csv('../input/test.csv').drop(columns=sample_id)
    return(predictors,label,test_predictors,test_id)


def one_hot_encoding(predictors,test_predictors):
    #丢掉所有因为特征过于特殊而没有意义的列（此处为大于15的）
    drop_col=[dname for dname in predictors.columns if predictors[dname].nunique()>15 and predictors[dname].dtype=='object']
    
    predictors.drop(columns=drop_col,inplace=True)
    test_predictors.drop(columns=drop_col,inplace=True)
    
    ohe_predictors=pd.get_dummies(predictors)
    ohe_test_predictors=pd.get_dummies(test_predictors)
    #要把测试集和训练集的特征对齐
    ali_ohe_train,ali_ohe_test=ohe_predictors.align(ohe_test_predictors,join='left',axis=1)
    return(ali_ohe_train,ali_ohe_test)

def impute_value(pred,test_pred):
    from sklearn.impute import SimpleImputer
    my_imp=SimpleImputer()
    im_train_pred=my_imp.fit_transform(pred)
    im_test_pred=my_imp.transform(test_pred)
    return(im_train_pred,im_test_pred)
    
def random_forest_model(train_pred,train_label,test_pred):
    from sklearn.ensemble import RandomForestRegressor
    import pandas as pd
    rfm=RandomForestRegressor()
    rfm.fit(train_pred,train_label)
    prediction=rfm.predict(test_pred)
    return(prediction)

def xgboost_model(features,label,test_X):
    from xgboost import XGBRegressor
    xgbmodel=XGBRegressor(n_estimators=1000)
    xgbmodel.fit(features,label,verbose=False)
    return(xgbmodel.predict(test_X))

def create_submission(test_id,prediction):
    import pandas as pd
    sub_df=pd.DataFrame({'Id':test_id,'SalePrice':prediction})
    sub_df.to_csv('submission.csv',index=False)
    return
#这个方框内只是使用了编码和简单插补
features,label,test_features,test_id=Prepare_data('SalePrice','Id')
ohe_features,ohe_test_features=one_hot_encoding(features,test_features)
im_ohe_features,im_ohe_test_features=impute_value(ohe_features,ohe_test_features)
'''
test_prediction=random_forest_model(im_ohe_pred,label,im_ohe_test_pred)
create_submission(test_id,test_prediction)'''
prediction=xgboost_model(im_ohe_features,label,im_ohe_test_features)
create_submission(test_id,prediction)