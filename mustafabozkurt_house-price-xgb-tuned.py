# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import numpy as np

import pandas as pd



# data visualization libraries:

import matplotlib.pyplot as plt

import seaborn as sns

import missingno as msno



# to ignore warnings:

import warnings

warnings.filterwarnings('ignore')



# to display all columns:

pd.set_option('display.max_columns', None)

pd.set_option('display.max_rows', None)

import statsmodels.api as sm

import statsmodels.formula.api as smf

import seaborn as sns

from sklearn.preprocessing import scale 

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score

from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

from sklearn.metrics import roc_auc_score,roc_curve

import statsmodels.formula.api as smf

import matplotlib.pyplot as plt

from sklearn.neighbors import KNeighborsClassifier

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC

from sklearn.naive_bayes import GaussianNB

from sklearn import tree

from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.model_selection import train_test_split, GridSearchCV

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC

from sklearn.naive_bayes import GaussianNB

from sklearn import tree

from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import GradientBoostingClassifier

from xgboost import XGBClassifier 

from lightgbm import LGBMClassifier

from catboost import CatBoostClassifier

import numpy as np, pandas as pd, os, gc

from sklearn.model_selection import GroupKFold

from sklearn.metrics import roc_auc_score

import matplotlib.pyplot as plt

import seaborn as sns

import seaborn as sns

import lightgbm as lgb

import gc

from time import time

import datetime

from tqdm import tqdm_notebook

from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import KFold, TimeSeriesSplit

from sklearn.metrics import roc_auc_score

warnings.simplefilter('ignore')

sns.set()

%matplotlib inline

import seaborn as sns

import matplotlib.pyplot as plt

import numpy as np,gc # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

pd.set_option('display.max_columns', 500)

import numpy as np

import pandas as pd 

import statsmodels.api as sm

import statsmodels.formula.api as smf

import seaborn as sns

from sklearn.preprocessing import scale 

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score

from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

from sklearn.metrics import roc_auc_score,roc_curve

import statsmodels.formula.api as smf

import matplotlib.pyplot as plt

from sklearn.neighbors import KNeighborsClassifier

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC

from sklearn.naive_bayes import GaussianNB

from sklearn import tree

from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import GradientBoostingClassifier

from xgboost import XGBClassifier

from lightgbm import LGBMClassifier

from catboost import CatBoostClassifier



from warnings import filterwarnings

filterwarnings('ignore')

from sklearn.model_selection import StratifiedKFold

!pip install ycimpute



from sklearn.preprocessing import StandardScaler

from sklearn.decomposition import PCA

from sklearn.preprocessing import LabelEncoder

from sklearn import datasets, metrics, model_selection, svm

import missingno as msno

from ycimpute.imputer import iterforest,EM

from fancyimpute import KNN

from sklearn.preprocessing import OrdinalEncoder



import numpy as np

import pandas as pd 

import statsmodels.api as sm

import statsmodels.formula.api as smf

import seaborn as sns

from sklearn.preprocessing import scale 

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score

from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

from sklearn.metrics import roc_auc_score,roc_curve

import statsmodels.formula.api as smf

import matplotlib.pyplot as plt

from sklearn.neighbors import KNeighborsClassifier

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC

from sklearn.naive_bayes import GaussianNB

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import GradientBoostingClassifier

from xgboost import XGBClassifier

from lightgbm import LGBMClassifier

from catboost import CatBoostClassifier

import numpy as np

import pandas as pd 

from sklearn.model_selection import train_test_split, GridSearchCV

from sklearn.metrics import mean_squared_error, r2_score

import matplotlib.pyplot as plt

from sklearn.preprocessing import scale 

from sklearn.preprocessing import StandardScaler

from sklearn import model_selection

from sklearn.linear_model import LinearRegression

from sklearn.tree import DecisionTreeRegressor

from sklearn.neighbors import KNeighborsRegressor

from sklearn.neural_network import MLPRegressor

from sklearn.ensemble import RandomForestRegressor

from sklearn.ensemble import GradientBoostingRegressor

from sklearn import neighbors

from sklearn.svm import SVR

from warnings import filterwarnings

filterwarnings('ignore')

from lightgbm import LGBMRegressor

import xgboost

from xgboost import XGBRegressor

pd.set_option('display.max_columns', None)

import gc
train=pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/train.csv")

test=pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/test.csv")
train.head()
test.head()
# FREQUENCY ENCODE TOGETHER

def encode_FE(df1, df2, cols):

    for col in cols:

        df = pd.concat([df1[col],df2[col]])

        vc = df.value_counts(dropna=True, normalize=True).to_dict()

        vc[-1] = -1

        nm = col+'_FE'

        df1[nm] = df1[col].map(vc)

        df1[nm] = df1[nm].astype('float32')

        df2[nm] = df2[col].map(vc)

        df2[nm] = df2[nm].astype('float32')

        print(nm,', ',end='')

        

# LABEL ENCODE

def encode_LE(col,train,test,verbose=False):

    df_comb = pd.concat([train[col],test[col]],axis=0)

    df_comb,_ = df_comb.factorize(sort=True)

    nm = col

    if df_comb.max()>32000: 

        train[nm] = df_comb[:len(train)].astype('int32')

        test[nm] = df_comb[len(train):].astype('int32')

    else:

        train[nm] = df_comb[:len(train)].astype('int16')

        test[nm] = df_comb[len(train):].astype('int16')

    del df_comb; x=gc.collect()

    if verbose: print(nm,', ',end='')

        

# GROUP AGGREGATION MEAN AND STD

# https://www.kaggle.com/kyakovlev/ieee-fe-with-some-eda

def encode_AG(main_columns, uids, aggregations, train_df, test_df, 

              fillna=True, usena=False):

    # AGGREGATION OF MAIN WITH UID FOR GIVEN STATISTICS

    for main_column in main_columns:  

        for col in uids:

            for agg_type in aggregations:

                new_col_name = main_column+'_'+col+'_'+agg_type

                temp_df = pd.concat([train_df[[col, main_column]], test_df[[col,main_column]]])

                if usena: temp_df.loc[temp_df[main_column]==-1,main_column] = np.nan

                temp_df = temp_df.groupby([col])[main_column].agg([agg_type]).reset_index().rename(

                                                        columns={agg_type: new_col_name})



                temp_df.index = list(temp_df[col])

                temp_df = temp_df[new_col_name].to_dict()   



                train_df[new_col_name] = train_df[col].map(temp_df).astype('float32')

                test_df[new_col_name]  = test_df[col].map(temp_df).astype('float32')

                

                if fillna:

                    train_df[new_col_name].fillna(-1,inplace=True)

                    test_df[new_col_name].fillna(-1,inplace=True)

                

                print("'"+new_col_name+"'",', ',end='')

                

# COMBINE FEATURES

def encode_CB(col1,col2,train,test):

    nm = str(col1)+'_'+str(col2)

    train[nm] = train[col1].astype(str)+'_'+train[col2].astype(str)

    test[nm] = test[col1].astype(str)+'_'+test[col2].astype(str)

    print("'"+nm+"'",', ',end='')

    encode_LE(nm,train,test)

# GROUP AGGREGATION NUNIQUE

def encode_AG2(main_columns, uids, train_df, test_df):

    for main_column in main_columns:  

        for col in uids:

            comb = pd.concat([train_df[[col]+[main_column]],test_df[[col]+[main_column]]],axis=0)

            mp = comb.groupby(col)[main_column].agg(['nunique'])['nunique'].to_dict()

            train_df[col+'_'+main_column+'_ct'] = train_df[col].map(mp).astype('float32')

            test_df[col+'_'+main_column+'_ct'] = test_df[col].map(mp).astype('float32')

            print(col+'_'+main_column+'_ct, ',end='')
main_columns=test._get_numeric_data().columns.drop(['Id'])

categorical_columns=test.columns.drop(main_columns)

categorical_columns=categorical_columns.drop(['Id'])
# FREQUENCY ENCODE TOGETHER

encode_FE(train, test, [ 'GarageCars', 'ExterQual',"OverallQual","BsmtQual","Fireplaces","KitchenQual","CentralAir","FullBath","GrLivArea"])

# COMBINE FEATURES

encode_CB('GarageCars', 'ExterQual',train,test)

encode_CB("BsmtQual","OverallQual",train,test)

encode_CB("Fireplaces","KitchenQual",train,test)

encode_CB("CentralAir","FullBath",train,test)

encode_CB('GarageCars_ExterQual','BsmtQual_OverallQual',train,test)

encode_CB('Fireplaces_KitchenQual','CentralAir_FullBath' , train,test)

encode_CB('GarageCars_ExterQual_BsmtQual_OverallQual' , 'Fireplaces_KitchenQual_CentralAir_FullBath' , train,test)

# GROUP AGGREGATION MEAN AND STD

#encode_AG(main_columns, ['GarageCars_ExterQual_BsmtQual_OverallQual' , 'Fireplaces_KitchenQual_CentralAir_FullBath' , 'GarageCars_ExterQual_BsmtQual_OverallQual_Fireplaces_KitchenQual_CentralAir_FullBath'], ["mean","std"], train, test)

#GROUP AGGREGATION NUNIQUE

encode_AG2(categorical_columns, ["MSSubClass","OverallQual"], train, test)
train.head()
def make_corr(Vs):

    cols = Vs.columns



    plt.figure(figsize=(15,15))

    sns.heatmap(train[cols].corr(), cmap='RdBu_r', annot=True, center=0.0)

    #plt.title(Vs[0]+' - '+Vs[-1],fontsize=14)

    plt.show()

make_corr(train.loc[:,"MSSubClass":"HouseStyle"])
#test = pd.get_dummies(test, columns = ["Neighborhood"])

#train=pd.get_dummies(train, columns = ["Neighborhood"])
encoder=OrdinalEncoder()

imputer=KNN()



def encode(data):

    '''function to encode non-null data and replace it in the original data'''

    #retains only non-null values

    nonulls = np.array(data.dropna())

    #reshapes the data for encoding

    impute_reshape = nonulls.reshape(-1,1)

    #encode date

    impute_ordinal = encoder.fit_transform(impute_reshape)

    #Assign back encoded values to non-null values

    data.loc[data.notnull()] = np.squeeze(impute_ordinal)

    return data
main_columns=test._get_numeric_data().columns.drop(['Id'])

categorical_columns=test.columns.drop(main_columns)

categorical_columns=categorical_columns.drop(['Id'])
from sklearn import preprocessing

for i in categorical_columns:

    lbe=preprocessing.LabelEncoder()

    train[i]=lbe.fit_transform(train[i].astype(str))

    test[i]=lbe.fit_transform(test[i].astype(str))
#for i in categorical_columns:

  #      encode(train[i])

  #      encode(test[i])
train=train.fillna(-1)

test=test.fillna(-1)
for i in categorical_columns:

    if (test[i].max()== train[i].max())&(train[i].max()<10):

                test = pd.get_dummies(test, columns = [i])

                train=pd.get_dummies(train, columns = [i])
train.shape, test.shape




y = train['SalePrice']

X= train.drop(['Id',"SalePrice"], axis=1)



models = [LGBMRegressor,

          XGBRegressor,

          GradientBoostingRegressor,

          RandomForestRegressor,

          DecisionTreeRegressor,

          MLPRegressor,

          KNeighborsRegressor,

          SVR]
def compML(df, y, alg):

    #train-test ayrimi

    #for name, clf in zip(names, classifiers):

    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.30,random_state=42)

    #modelleme

 

    model = alg().fit(X_train, y_train)

    y_pred = model.predict(X_test)

    RMSE = np.sqrt(mean_squared_error(y_test, y_pred))

    model_name = alg.__name__

    print(model_name, "Model Test error:",RMSE)
for i in models:

    compML(X, y, i)
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.30,random_state=42)

xgb = XGBRegressor().fit(X_train, y_train)

y_pred = xgb.predict(X_test)

np.sqrt(mean_squared_error(y_test, y_pred))
xgb_params = {"learning_rate": [0.1,0.01,0.5],

             "max_depth": [2,3,4,5,8],

             "n_estimators": [100,200,500,1000,3000,6000],

             "colsample_bytree": [0.4,0.7,1],

             "min_child_weight":[0,1,2],

            "gamma":[0.6,0.8,0.2],

                       "subsample":[0.7,0.5,0.9],

                       "colsample_bytree":[0.7,0.01,0.2],

                       

                       "nthread":[-1,-2],

                       "scale_pos_weight":[1,2,5,7],

                       "seed":[27,5,40,60],

                       "reg_alpha":[0.0000,0.005,0.5,0.00006]}
#xgb_cv_model  = GridSearchCV(xgb,xgb_params, cv = 5, n_jobs = -1, verbose = 2).fit(X_train, y_train)
#xgb_cv_model.best_params_
xgb_tuned = XGBRegressor(learning_rate=0.01,

                       n_estimators=6000,

                       max_depth=4,

                       min_child_weight=0,

                       gamma=0.6,

                       subsample=0.7,

                       colsample_bytree=0.7,

                       objective='reg:linear',

                       nthread=-1,

                       scale_pos_weight=1,

                       seed=27,

                       reg_alpha=0.00006,

                       random_state=42).fit(X_train, y_train)
y_pred = xgb_tuned.predict(X_test)
np.sqrt(mean_squared_error(y_test, y_pred))

from sklearn.metrics import mean_absolute_error
print("Mean Absolute Error : " + str(mean_absolute_error(y_test, y_pred)))
ids=test["Id"]

test= test.drop(["Id"], axis=1).astype("float64")
prediction=xgb_tuned.predict(test)

output=pd.DataFrame({"Id":ids, "SalePrice":prediction})

output.to_csv("submission_xgb.csv", index=False)
ft_weights = pd.DataFrame(xgb_tuned.feature_importances_, columns=['weights'], index=X.columns)

ft_weights=ft_weights.reset_index()

feature_imp=ft_weights.head(60)

feature_imp.to_excel('feature_importances.xlsx')

plt.figure(figsize=(20, 10))

sns.barplot(x='weights', y="index", data=feature_imp.sort_values(['weights'],ascending=False))

plt.title('XGB Features (avg over folds)')

plt.tight_layout()

plt.show()

plt.savefig('xgb_importances-01.png')