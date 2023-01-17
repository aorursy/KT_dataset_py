# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kag?gle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import eli5
from eli5.sklearn import PermutationImportance
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
#from plotly.offline import init_notebook_mode, iplot
#init_notebook_mode(connected = True)
#import plotly.graph_objs as go
import warnings
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
warnings.filterwarnings('ignore')
plt.style.use('fivethirtyeight')

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
train = pd.read_csv('../input/train_V2.csv')
test = pd.read_csv('../input/test_V2.csv')
train.describe()
train.info()
def reduce_mem_usage(df):
    """ iterate through all the columns of a dataframe and modify the data type
        to reduce memory usage.        
    """
    #start_mem = df.memory_usage().sum() / 1024**2
    #print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))

    for col in df.columns:
        col_type = df[col].dtype

        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int': #Encode with the most relevant datatype.g
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)

    #end_mem = df.memory_usage().sum() / 1024**2
    #print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    #print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))

    return df
train = reduce_mem_usage(train)
test = reduce_mem_usage(test)
train.info()
#Univariate Distribution
for col in train.columns:
    if 'Id' in col:
        pass
    elif train[col].dtype == 'float16' or len(train[col].unique()) > 100:
        print(col)
        plt.figure(figsize = (15, 8))
        sns.kdeplot(train[col])
        plt.show()
    else:
        print(col)
        plt.figure(figsize = (15, 8))
        train[col].value_counts().plot(kind = 'bar')
        plt.show()

for col in test.columns:
    if 'Id' in col:
        pass
    elif test[col].dtype == 'float16' or len(test[col].unique()) > 100:
        print(col)
        plt.figure(figsize = (15, 8))
        sns.kdeplot(test[col])
        plt.show()
    else:
        print(col)
        plt.figure(figsize = (15, 8))
        test[col].value_counts().plot(kind = 'bar')
        plt.show()
for col in train.columns:
    if 'Id' in col or col == 'winPlacePerc':
        pass
    elif train[col].dtype == 'float16':
        print(col)
        sns.jointplot(x = col, y = 'winPlacePerc', data = train, height = 10, ratio = 3)
        plt.show()
    else:
        print(col)
        sns.catplot(x = col, y = 'winPlacePerc', data=train, kind = 'boxen', aspect=3)
        plt.show()
#Correlation map of variables in the dataset
plt.figure(figsize = (15, 8))
sns.heatmap(train.corr())
#Drop Null Vals
train.dropna(inplace = True)
from sklearn.linear_model import LinearRegression, Lasso
def fit_linear_model(train, y):
    lm = Lasso(alpha = 0.0000001)
    lm.fit(train, y)
    return lm
import xgboost

def fit_tree_model(train, y):
    # train XGBoost model
    model = xgboost.train({"learning_rate": 0.05}, xgboost.DMatrix(train, label=y), 100)
    return model
from sklearn.model_selection import train_test_split
cols = ['assists', 'boosts', 'damageDealt', 'DBNOs',
           'headshotKills', 'heals', 'killPlace', 'killPoints', 'kills',
           'killStreaks', 'longestKill', 'maxPlace', 'numGroups', 'revives',
           'rideDistance', 'roadKills', 'swimDistance', 'teamKills',
           'vehicleDestroys', 'walkDistance', 'weaponsAcquired', 'winPoints']
trainX, valX, trainY, valY = train_test_split(train[cols], train['winPlacePerc'], test_size = 0.2)
lm = fit_linear_model(trainX, trainY)
print('Score:{:.4f} & RMSE: {:.4f}'.format(lm.score(valX[cols], valY), np.sqrt(mean_squared_error(lm.predict(valX[cols]), valY))))
from eli5.sklearn import PermutationImportance
perm = PermutationImportance(lm, random_state=1).fit(train[cols], train['winPlacePerc'])
eli5.show_weights(perm, feature_names = cols)
from matplotlib import pyplot as plt
from pdpbox import pdp, get_dataset, info_plots

for feat_name in cols:
    pdp_dist = pdp.pdp_isolate(model=lm, dataset=valX, model_features=cols, feature=feat_name)
    pdp.pdp_plot(pdp_dist, feat_name)
    plt.show()
import shap
# load JS visualization code to notebook
shap.initjs()

# explain the model's predictions using SHAP values
# (same syntax works for LightGBM, CatBoost, and scikit-learn models)
explainer = shap.LinearExplainer(lm, data=trainX)
shap_values = explainer.shap_values(valX.iloc[:1000,:])

# visualize the first prediction's explanation
#shap.force_plot(explainer.expected_value, shap_values[0,:], valX.iloc[0,:])
shap.force_plot(explainer.expected_value, shap_values, valX.iloc[:1000,:], link="logit")
explainer = shap.LinearExplainer(lm, data=trainX)
shap_values = explainer.shap_values(valX)
shap.summary_plot(shap_values, valX)
tree = fit_tree_model(trainX, trainY)
print('RMSE: {:.4f}'.format(np.sqrt(mean_squared_error(tree.predict(data=xgboost.DMatrix(valX[cols])), valY))))
# load JS visualization code to notebook
shap.initjs()

# explain the model's predictions using SHAP values
# (same syntax works for LightGBM, CatBoost, and scikit-learn models)
explainer = shap.TreeExplainer(tree)
shap_values = explainer.shap_values(valX)

# visualize the first prediction's explanation
shap.force_plot(explainer.expected_value, shap_values[0,:], valX.iloc[0,:])
shap.force_plot(explainer.expected_value, shap_values[1,:], valX.iloc[1,:])
explainer = shap.TreeExplainer(tree)
shap_values = explainer.shap_values(valX)
shap.summary_plot(shap_values, valX)
shap_values = explainer.shap_values(valX.iloc[:1000,:])
shap.force_plot(explainer.expected_value, shap_values, valX.iloc[:1000,:], link="logit")