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
    print('R2: {}'.format(r2_score(prediction, lables)))
    print('RMSE: {}'.format(np.sqrt(mean_squared_error(prediction, lables))))
    return np.sqrt(mean_squared_error(prediction, lables))
# Shows scores for train and validation sets    
def train_test(estimator, x_trn, x_tst, y_trn, y_tst):
    prediction_train = estimator.predict(x_trn)
    # Printing estimator
    print(estimator)
    # Printing train scores
    out = []
    out.append(get_score(prediction_train, y_trn))
    prediction_test = estimator.predict(x_tst)
    # Printing test scores
    print("Test")
    out.append(get_score(prediction_test, y_tst))
    return out
train = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')
test = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')
test.head()
train.head()
ntrain = train.shape[0]
y_train = train["SalePrice"]
train = train[train.columns[:-1]]
train = train.append(test)

obj_cols = train.columns[train.dtypes == "object"]
train[obj_cols] = train[obj_cols].fillna("None") # No pool ?
train = pd.get_dummies(train,columns=obj_cols)

#for col_i in obj_cols:
#    print("Col_name : ", col_i)
#    #print("values : \n", train[col_i].value_counts())
#    print("NAs : ", train[col_i].isna().sum()) 
train = train.select_dtypes(include=['float64','int64','uint8'])
train.fillna(train.median(),inplace=True)
print(train.columns)
train.info()
# get a list of columns
#cols = list(train)
# move the column to head of list using index, pop and insert
#cols.append(cols.pop(cols.index('SalePrice')))
#cols
# use ix to reorder
#train = train.ix[:, cols]
#train.head()
feat_cols = train.columns
train = train[feat_cols]
train.shape
skewed_columns = train.columns[abs(train.skew()) > 1.7]
skewed_columns

train_bkp = train.copy()
train[skewed_columns] = np.log2(train[skewed_columns])

inf_issue = train.columns[np.isinf(train).sum() > 0]
#inf_issue
def diff(first, second):
        second = set(second)
        return [item for item in first if item not in second]
cols_2_log = diff(skewed_columns,inf_issue)
cols_2_log
train = train_bkp.copy()
train[cols_2_log] = np.log2(train[cols_2_log])

train.shape
y_train = np.log2(y_train)

train_new = train[:ntrain]
train_new = train_new[np.abs(train_new -train_new.mean()) <= (3*train_new.std())] #keep only the ones that are within +3 to -3 standard deviations in the column 'Data'.
print( "removed : {}".format(train[:ntrain].shape[0] - train_new.shape[0]))
x_train_st, x_test_st, y_train_st, y_test_st = train_test_split(train[:ntrain],y_train,train_size=0.7)
print(x_train_st.shape)
print(x_test_st.shape)
SLR_Test = linear_model.LinearRegression(normalize=True)
SLR_Test.fit(x_train_st.values, y_train_st)
train_test(SLR_Test, x_train_st, x_test_st, y_train_st, y_test_st)
pred = SLR_Test.predict(x_train_st)
act = y_train_st
x = act
y = pred

fit = np.polyfit(x,y,1)
fit_fn = np.poly1d(fit) 
# fit_fn is now a function which takes in x and returns an estimate for y

plt.plot(x,y, 'o', x, fit_fn(x), '--k')
pred = SLR_Test.predict(x_test_st)
act = y_test_st
x = act
y = pred
#plt.plot(act,pred,"o")

fit = np.polyfit(x,y,1)
fit_fn = np.poly1d(fit) 
# fit_fn is now a function which takes in x and returns an estimate for y

plt.plot(x,y, 'o', x, fit_fn(x), '--k')
test = train[ntrain:]
print(test.shape)
s_kag_pred = SLR_Test.predict(test)
s_kag_pred = np.power(2,s_kag_pred)
s_kag_pred
np.isinf(s_kag_pred).sum()
#np.median(s_kag_pred)
s_kag_pred[np.isinf(s_kag_pred)] = np.median(s_kag_pred)
np.isinf(s_kag_pred).sum()
LR_Test = linear_model.ElasticNetCV(normalize=True,cv=5,l1_ratio=0.1)
LR_Test.fit(x_train_st.values, y_train_st)
train_test(LR_Test, x_train_st, x_test_st, y_train_st, y_test_st)
#plt.plot(np.sort(LR_Test.coef_))
#LR_Test.l1_ratio_
#plt.plot(LR_Test.alphas_)
plt.plot(LR_Test.alphas_,LR_Test.mse_path_)
(LR_Test.alpha_, LR_Test.mse_path_.min())
def opt_l2(l2_ratios):
    out = []
    for l2 in l2_ratios:
        LR_Test = linear_model.ElasticNetCV(normalize=True,cv=5,l1_ratio=l2)
        LR_Test.fit(x_train_st.values, y_train_st)
        out.append(train_test(LR_Test, x_train_st, x_test_st, y_train_st, y_test_st))
    return out
l2_ratios = np.linspace(0.1,1,num=10)
output_opt_l2 = opt_l2(l2_ratios)
output_opt_l2
plt.plot(l2_ratios,output_opt_l2,"-o")
alphas = np.logspace(-4,-2,num=100)
LR_Test = linear_model.ElasticNetCV(alphas=alphas,normalize=True,cv=15,l1_ratio=1)
LR_Test.fit(x_train_st.values, y_train_st)
train_test(LR_Test, x_train_st, x_test_st, y_train_st, y_test_st)
plt.plot(np.log10(LR_Test.alphas_),LR_Test.mse_path_,"-o")
(LR_Test.alpha_, np.log10(LR_Test.alpha_) ,LR_Test.mse_path_.min())
import plotly.offline as py
from plotly.offline import init_notebook_mode
import plotly.graph_objs as go
init_notebook_mode(connected=True)
data = [
    go.Contour(
 #       x= np.log10(LR_Test.alphas_),
        y=np.log10(LR_Test.alphas_),
        z=LR_Test.mse_path_
        #,y=[0, 1, 4, 5, 7]
    )
]
#data
dataS = [
    go.Surface(
 #       x= np.log10(LR_Test.alphas_),
        x=LR_Test.l1_ratio_,
        y=np.log10(LR_Test.alphas_),
        z=LR_Test.mse_path_
        #,y=[0, 1, 4, 5, 7]
    )
]
#dataS


layout = go.Layout(
    title='MSE',
    scene=dict(xaxis=dict(title="Folds")
               ,yaxis=dict(title="Alphas")
               ,zaxis=dict(title='RMSE')
              ),
    autosize=False,
    width=500,
    height=500,
    margin=dict(
        l=65,
        r=50,
        b=65,
        t=90
    )
)

py.iplot(go.Figure(data=data, layout=layout)
         , filename='elevations-3d-surface')

py.iplot(go.Figure(data=dataS, layout=layout)
         , filename='elevations-3d-surface')
pred = LR_Test.predict(x_train_st)
act = y_train_st
x = act
y = pred

fit = np.polyfit(x,y,1)
fit_fn = np.poly1d(fit) 
# fit_fn is now a function which takes in x and returns an estimate for y

plt.plot(x,y, 'o', x, fit_fn(x), '--k')
pred = LR_Test.predict(x_test_st)
act = y_test_st
x = act
y = pred
#plt.plot(act,pred,"o")

fit = np.polyfit(x,y,1)
fit_fn = np.poly1d(fit) 
# fit_fn is now a function which takes in x and returns an estimate for y

plt.plot(x,y, 'o', x, fit_fn(x), '--k')
test = train[ntrain:]
print(test.shape)
kag_pred = LR_Test.predict(test)
kag_pred = np.power(2,kag_pred)
kag_pred
SLR_pred = pd.read_csv("../input/the-minimal-normalize-x-skew-y-exploratory/submission.csv")
SLR_pred.head()
ELR_pred = pd.read_csv("../input/beginelasticv5/submission(1).csv")
ELR_pred.head()
sub = pd.DataFrame()
sub['Id'] = test["Id"]
sub['SalePrice'] = (kag_pred*3 + SLR_pred.SalePrice*1 + ELR_pred.SalePrice*2) /6
sub.to_csv('submission.csv',index=False)
sub.head()
