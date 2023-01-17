# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))
print(os.listdir("../input/dmbi-18-metro/"))
print(os.listdir("../input/weather-model-input/"))

# Any results you write to the current directory are saved as output.
model_input = pd.read_csv("../input/weather-model-input/model_input.csv",index_col=0)
print(model_input.shape)
model_input.head()

def labels(row):
    winter = [12, 1, 2] # coldest
    # summer = [6, 7, 8] # hottest
    fall = [9, 10, 11] # hotter
    spring = [3, 4, 5] # colder
    if row in winter:
        return 11

    if row in fall:
        return 33

    if row in spring:
        return 22

    return 44

# adding month number
model_input["Basis Month N"] = model_input["Basis date"].map(lambda x: x[3:5])
model_input["Basis Month N"] = model_input["Basis Month N"].map(int)

# adding season number
model_input["Basis Season N"] = model_input["Basis Month N"].map(lambda row: labels(row))
model_input_observedMaxTemp = model_input.groupby(model_input.index)['observedMaxTemp'].mean()
print(model_input_observedMaxTemp.shape)
model_input_observedMaxTemp=model_input_observedMaxTemp.to_frame()
model_input_observedMaxTemp.head()

print(model_input.shape)
del model_input["observedMaxTemp"]
model_input = model_input.merge(model_input_observedMaxTemp,how="left",right_index=True,left_index=True)
print(model_input.shape)
# del model_input

model_input.head()
model_input_basis = pd.to_datetime(model_input["Basis date"], format="%d/%m/%Y")
# datetime.date(2010, 6, 16).isocalendar()[1]
model_input["week n"] = model_input_basis.map(lambda x: 
#                       x.isocalendar()[0]
                     x.strftime("%V")) 
model_input["week n"] = abs(model_input["week n"].astype("int")-26)

model_input["Basis Month N"] = abs(model_input["Basis Month N"].astype("int")-6)
# week_n
# selected_months = ["01","02","03","12"]
# train=train.loc[train["Basis date"].map(lambda x: x[3:5]).isin(selected_months)]
# test["Basis date"].map(lambda x: x[3:5]).value_counts()
# 01    930
# 03    915
# 02    840
# 12     15
# model_input.corr()[["Time","observedMaxTemp"]].sort_values("observedMaxTemp",ascending=False) #inplace=True
model_input.corr().sort_values("observedMaxTemp",ascending=False) #inplace=True
# model_input[["Basis Season N","C3"]].corr().plot(marker="o")
import seaborn as sns
plt.figure(figsize=(15, 15))
cmap = sns.cubehelix_palette(n_colors=100, start=2, rot=1, light=0.9, dark=0.2)
sns.heatmap(abs(model_input.corr()).sort_values("observedMaxTemp",ascending=False),cmap=cmap)
# model_input.skew().sort_values()
# model_input.groupby('week n', as_index=False)['observedMaxTemp'].mean().plot() #["week n"].max()
# model_input.groupby('Basis Season N', as_index=False)['observedMaxTemp'].mean().plot()
model_input.groupby('Basis Month N', as_index=False)['observedMaxTemp'].mean().plot()

train = model_input[model_input.trainOrTest=="train"]
test = model_input[model_input.trainOrTest=="test"]
print(train.shape,test.shape)

# test["Basis Month N"].value_counts()
# selected_months = ["01","02","03","12"]
selected_months = [3,4,5,6]
# train=train.loc[train["Basis date"].map(lambda x: x[3:5]).isin(selected_months)]
train=train.loc[train["Basis Month N"].isin(selected_months)]

print(train.shape,test.shape)
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
from sklearn.linear_model import Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor
import xgboost as xgb
ntrain = train.shape[0]
y_train = train["observedMaxTemp"]
y_test = test["observedMaxTemp"]
train = train[train.columns[:-1]]
train = train.append(test)
del train["observedMaxTemp"]
train.shape
# train.info()
train = train.select_dtypes(include=np.number)
train.fillna(train.median(),inplace=True)
train.shape
selected_cols = train.columns

# coef_Lasso = pd.read_csv("../input/housepricinglassocoef/coef_Lasso.csv")
# selected_cols = coef_Lasso.feature.tolist()
# selected_cols = ["C3","Basis Season N","AverageHeight","Basis Month N"]
feat_cols = selected_cols #[:-1]
train = train[selected_cols]
print(train.shape)
x_train_np = train[feat_cols][:ntrain].values
y_train_np = y_train.values #train["SalePrice"].values

# y_train_np = np.log(y_train_np)
y_train_np = y_train_np*25
y_train_np
# plt.plot(y_train_np)
plt.hist(y_train_np) 
# train for test moths only
from sklearn.model_selection import cross_val_score,GridSearchCV

def rmse_cv(model):
    rmse= np.sqrt(-cross_val_score(estimator = model
                                   ,X= x_train_np
                                   ,y= y_train_np
                                   , scoring="neg_mean_squared_error"
                                   , cv = 5, n_jobs=4, verbose=1))
    return rmse
from sklearn.metrics import mean_squared_error, make_scorer

def rmse(y,y_pred,**args):
    return -np.sqrt(mean_squared_error(y,y_pred,**args))

rmse_scorer = make_scorer(rmse)
# ss = pd.read_csv('../input/dmbi-18-metro/')
actual=pd.DataFrame([test.index,y_test])
actual=actual.transpose()
actual.columns = ["validityDate_city","predictedMaxTemp"]
actual.predictedMaxTemp = actual.predictedMaxTemp.astype("float")
actual = actual.groupby('validityDate_city', as_index=False)['predictedMaxTemp'].mean()

print(actual.shape)
actual.head()

x_train_st, x_test_st, y_train_st , y_test_st = train_test_split(x_train_np,y_train_np,train_size=0.7,random_state=1)

param_dict = {
    "alpha": np.logspace(base=10,start=-5,stop=-1,num=10) #[0.00059948425] [2.7825e-5]# [0.00021544] #
    , "fit_intercept" : [True]
    , "normalize" : [False]
    , "precompute" : [False]
#     , copy_X=True
    , "max_iter":[10000]
#     , "tol": [0.01]
#     , warm_start=False
#     , positive=False
    , "random_state" : [1]
    , "selection":['cyclic']
}

def cv_score(model):
    
    
    cv = GridSearchCV(estimator=model
                      ,param_grid=param_dict
                      ,n_jobs=4
                      ,scoring=rmse_scorer #"neg_mean_squared_error"
                      ,cv = 5 #[x_train_np,y_train_np]
                      ,return_train_score=True
                      ,verbose=1)
    bst_model =cv.fit(x_train_st,y_train_st).best_estimator_
    
    tr_cv = np.mean(np.sqrt(-cross_val_score(bst_model,x_train_st,y_train_st,scoring="neg_mean_squared_error", cv = 5, n_jobs=4, verbose=1)))
    ts_cv = np.mean(np.sqrt(-cross_val_score(bst_model,x_test_st,y_test_st,scoring="neg_mean_squared_error", cv = 5, n_jobs=4, verbose=1)))
    kag_tr_cv = np.mean(np.sqrt(-cross_val_score(bst_model,x_train_np,y_train_np,scoring="neg_mean_squared_error", cv = 5, n_jobs=4, verbose=1)))
    
#     kag_ts_cv = np.mean(np.sqrt(-cross_val_score(bst_model,x_train_np,y_train_np,scoring="neg_mean_squared_error", cv = 5, n_jobs=4, verbose=1)))

    print(bst_model)
    print(tr_cv,ts_cv,kag_tr_cv)
    
    return [cv,tr_cv,ts_cv,kag_tr_cv]

from sklearn.linear_model import Lasso
regr_2 = Lasso()
[cv,tr_cv,ts_cv,kag_tr_cv] = cv_score(regr_2)
bst_model=cv.best_estimator_
print(tr_cv,ts_cv,kag_tr_cv)
# 1.1420425153793534 1.1502175181626113 1.1695601377209102
# 1.1471954077668247 1.1348489887238495 1.1447285972450576
# 1.2357725032212596 1.2698250677532683 1.257759532339718
# 1.2357665170236989 1.2697838610664882 1.2577630558254609

# 1.2332648956638774 1.2668910627000134 1.260790429804996

# 1.7803217340964195 1.8387023082525544 1.8089948802206457 #C3
# 1.7513519776573183 1.813701170531879 1.7845298093601738  #"C3" "Basis Season N"
# 1.657934502831966 1.7359621487508492 1.6965000530648233 #"C3" "Basis Season N" "AverageHeight"
# 1.6387905916654208 1.6207499528051543 1.6441882467248707 # month problem solved
# 1.6233098019523815 1.6095852297658084 1.6315874167841788 #"C3" "Basis Season N" "AverageHeight" "Basis Month N"
# 1.1607475292392255 1.1438864246606277 1.1662043395266997 #all
# 73.86819099023465 73.73704913572304 75.22436790277827 # pow 2 bad idea
# times*25

29.15/25
coef_Lasso = pd.DataFrame([feat_cols,bst_model.coef_.tolist()])
coef_Lasso = coef_Lasso.transpose()
coef_Lasso.columns=["feature","coef"]

coef_Lasso.sort_values("coef",ascending=False,inplace=True)

print(coef_Lasso.head())

(coef_Lasso.coef==0).value_counts()
coef_Lasso[coef_Lasso.coef!=0]#.hist()

# coef_Lasso[coef_Lasso.coef!=0].to_csv("coef_Lasso_weather.csv")
results= cv.cv_results_

scoring = {'score': rmse_scorer}

plt.figure(figsize=(5, 5))
plt.title("GridSearchCV evaluating using multiple scorers simultaneously",
          fontsize=16)

plt.xlabel("alpha")
plt.ylabel("Score")
plt.grid()

ax = plt.axes()
# ax.set_xlim(0, 100)
# ax.set_ylim(0,0.4)
ax.set_xscale("log")
# Get the regular numpy array from the MaskedArray
X_axis = np.array(results['param_alpha'].data, dtype=float)

for scorer, color in zip(sorted(scoring), ['g', 'k']):
    for sample, style in (('train', '--'), ('test', '-')):
        sample_score_mean = results['mean_%s_%s' % (sample, scorer)]
        sample_score_std = results['std_%s_%s' % (sample, scorer)]
        ax.fill_between(X_axis, sample_score_mean - sample_score_std,
                        sample_score_mean + sample_score_std,
                        alpha=0.1 if sample == 'test' else 0, color=color)
        ax.plot(X_axis, sample_score_mean, style, color=color,
                alpha=1 if sample == 'test' else 0.7,
                label="%s (%s)" % (scorer, sample))

    best_index = np.nonzero(results['rank_test_%s' % scorer] == 1)[0][0]
    best_score = results['mean_test_%s' % scorer][best_index]

    # Plot a dotted vertical line at the best score for that scorer marked by x
    ax.plot([X_axis[best_index], ] * 2, [0, best_score],
            linestyle='-.', color=color, marker='x', markeredgewidth=3, ms=8)

    # Annotate the best score for that scorer
#     ax.annotate("%0.2f" % best_score,
#                 (X_axis[best_index], best_score + 0.005))

plt.legend(loc="best")
# plt.grid('off')
plt.show()
# Predict
# y_1 = regr_1.predict(x_train_np)
y_2 = bst_model.predict(x_train_np)

# Plot the results
plt.figure()
# plt.plot(y_train_np, y_1, c="g",marker="o",ls="", label="n_estimators=1", linewidth=2)
plt.plot(y_train_np, y_2, c="r",marker="o",ls="", label="n_estimators=300", linewidth=2)
plt.xlabel("actual")
plt.ylabel("prediction")
plt.title("Boosted Decision Tree Regression")
plt.legend()
plt.show()
import seaborn as sns
# Plot the residuals after fitting a linear model
# sns.residplot(y_train_np, y_1, lowess=True, color="r")
sns.residplot(y_train_np, y_2, lowess=True)
test = train[ntrain:]
test.fillna(test.median(),inplace=True)
kag_pred = bst_model.predict(test[feat_cols])
kag_pred = kag_pred/25
kag_pred
# test = pd.read_csv('../input/test.csv')
sub = pd.DataFrame()
# validityDate_city,predictedMaxTemp
sub['validityDate_city'] = test.index
sub['predictedMaxTemp'] = kag_pred
sub = sub.groupby('validityDate_city', as_index=False)['predictedMaxTemp'].mean()
sub.to_csv('submission.csv',index=False)

local_kag=rmse(y_pred=sub.predictedMaxTemp,y=actual.predictedMaxTemp)
print(local_kag)
# -9.665382059430602
# -1.2771051394002935
# -1.2939625518782338
sub.head()

