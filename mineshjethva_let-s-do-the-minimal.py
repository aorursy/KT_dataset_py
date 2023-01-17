# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
# Adding needed libraries and reading data
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import ensemble, tree, linear_model
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.utils import shuffle

%matplotlib inline
import warnings
warnings.filterwarnings('ignore')

# Prints R2 and RMSE scores
def get_score(prediction, lables):    
#     print('R2: {}'.format(r2_score(prediction, lables)))
    print('RMSE: {}'.format(np.sqrt(mean_squared_error(prediction, lables))),end=" ")
    return np.sqrt(mean_squared_error(prediction, lables))

# Shows scores for train and validation sets    
def train_test(estimator, x_trn, x_tst, y_trn, y_tst):
    prediction_train = estimator.predict(x_trn)
    # Printing estimator
#     print(estimator)
    # Printing train scores
    print("\nTrain",end=" :\t")
    tr_rmse = get_score(prediction_train, y_trn)
    prediction_test = estimator.predict(x_tst)
    # Printing test scores
    print("\tTest",end=" :\t")
    ts_rmse = get_score(prediction_test, y_tst)
    
    return [tr_rmse,ts_rmse]
# Reading Data in to pandas data frame
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
# Visualizing top 5 rows of training data
train.head()
# Printing description about training data
# train.info()
# counting "NA" (NaN) values per column (featu)
# train.isna().sum()
train = train.select_dtypes(include=['float64','int64'])
train.fillna(train.median(),inplace=True)
print(train.columns)
# train.info()
# train.isna().sum()
train_bkp = train
selected_cols = train.columns
# selected_cols = ["YearBuilt","LotArea","OverallQual","YearBuilt","SalePrice"]
feat_cols = selected_cols[:-1]
train = train_bkp[selected_cols]
train.shape
x_train_np = train[feat_cols].values
y_train_np = train["SalePrice"].values
# from sklearn.model_selection import ShuffleSplit
# splits = 10
# score = []
# rs = ShuffleSplit(n_splits= splits,test_size = 0.3, random_state = 110)

# for train_index, test_index in rs.split(range(len(train))):
    
# #     print(type(train_index))
#     x_train_st , x_test_st = x_train_np[train_index], x_train_np[test_index]
#     y_train_st , y_test_st = y_train_np[train_index], y_train_np[test_index]
# #     print(x_train_st.shape,x_test_st.shape)
# #     print(y_train_st.shape,y_test_st.shape)
    
#     LR_model = linear_model.LinearRegression()
#     LR_model.fit(x_train_st, y_train_st)
#     tr_rmse,ts_rmse = train_test(LR_model, x_train_st, x_test_st, y_train_st, y_test_st)
# #     train_test(LR_model, x_train_st, x_train_np, y_train_st, y_train_np)
#     score.append(ts_rmse)
# ts_avg = np.array(score).mean()
# print("\n\nts_avg",ts_avg)
LR_model = linear_model.LinearRegression()
LR_model.fit(x_train_np, y_train_np)
from sklearn.model_selection import cross_val_score

def rmse_cv(model):
    rmse= np.sqrt(-cross_val_score(estimator = model
                                   ,X= x_train_np
                                   ,y= y_train_np
                                   , scoring="neg_mean_squared_error"
                                   , cv = 12, n_jobs=2, verbose=1))
    return rmse
print(x_train_np.shape)
rmse_cv(LR_model).mean()
pred = LR_model.predict(x_test_st)
act = y_test_st
x = act
y = pred
#plt.plot(act,pred,"o")

fit = np.polyfit(x,y,1)
fit_fn = np.poly1d(fit) 
# fit_fn is now a function which takes in x and returns an estimate for y

plt.plot(x,y, 'o', x, fit_fn(x), '--k')
test.fillna(test.median(),inplace=True)
kag_pred = LR_model.predict(test[feat_cols])
kag_pred
sub = pd.DataFrame()
sub['Id'] = test["Id"]
sub['SalePrice'] = kag_pred
sub.to_csv('submission.csv',index=False)
