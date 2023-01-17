# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

            

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
train_df=pd.read_csv("/kaggle/input/allstate-claims-severity/train.csv")

test_df=pd.read_csv("/kaggle/input/allstate-claims-severity/test.csv")

submission=pd.read_csv("/kaggle/input/allstate-claims-severity/sample_submission.csv")

from sklearn.decomposition import FactorAnalysis

fa= FactorAnalysis(n_components=15, random_state=0)
contFeatureslist=[]

for colName,x in train_df.iloc[1,:].iteritems():

    if(not str(x).isalpha()):

        contFeatureslist.append(colName)

contFeatureslist.remove("id")

contFeatureslist.remove("loss")

import matplotlib.pyplot as plt





newdf=train_df[contFeatureslist]

newdf.columns

    
import seaborn as sns

%matplotlib inline

print(contFeatureslist)

plt.figure(figsize=(23,9))

sns.boxplot(x=contFeatureslist,data=newdf)

correlationMatrix = train_df[contFeatureslist].corr().abs()

plt.subplots(figsize=(13, 9))

sns.heatmap(correlationMatrix,annot=True)



# Mask unimportant features

sns.heatmap(correlationMatrix, mask=correlationMatrix < 1, cbar=False)

plt.show()
sns.boxplot(np.log1p(train_df["loss"]))

train_df["loss"]=np.log1p(train_df["loss"])
catFeatureslist = []

for colName,x in train_df.iloc[1,:].iteritems():

    if(str(x).isalpha()):

        catFeatureslist.append(colName)

from sklearn.preprocessing import LabelEncoder

for cf1 in catFeatureslist:

    le = LabelEncoder()

    le.fit(train_df[cf1].unique())

    train_df[cf1] = le.transform(train_df[cf1])
print(len(train_df.columns))

sum(train_df[catFeatureslist].apply(pd.Series.nunique) > 2)

print(len(train_df.columns))
filterG5_10 = list((train_df[catFeatureslist].apply(pd.Series.nunique) > 5) & 

                (train_df[catFeatureslist].apply(pd.Series.nunique) < 10))
catFeaturesG5_10List = [i for (i, v) in zip(catFeatureslist, filterG5_10) if v]
correlationMatrix = train_df[catFeatureslist].corr().abs()

crrm=correlationMatrix[correlationMatrix>0.9]

catFeatureslist = []

for colName,x in test_df.iloc[1,:].iteritems():

    if(str(x).isalpha()):

        catFeatureslist.append(colName)

from sklearn.preprocessing import LabelEncoder

for cf1 in catFeatureslist:

    le = LabelEncoder()

    le.fit(test_df[cf1].unique())

    test_df[cf1] = le.transform(test_df[cf1])
cat2=train_df.drop("cat2",axis=1)

Ytrain=train_df["loss"]

train_df=train_df.drop("loss",axis=1)

Xtrain=train_df

Xtest=test_df

cat_features=list(np.where(all_data.dtypes==np.object)[0])

Xtrain.columns


all_data=pd.concat((train_df, test_df))

all_data.shape



from xgboost import XGBRegressor

from lightgbm import LGBMRegressor

from catboost import CatBoostRegressor

from sklearn.model_selection import cross_val_score

model_xgb=XGBRegressor(tree_method='gpu_hist', seed=18, objective='reg:linear', n_jobs=-1, verbosity=0,

                       colsample_bylevel=0.764115402027029, colsample_bynode=0.29243734009596956, 

                       colsample_bytree= 0.7095719673041723, gamma= 4.127534050725986, learning_rate= 0.02387231810322894, 

                       max_depth=14, min_child_weight=135, n_estimators=828,reg_alpha=0.3170105723222332, 

                       reg_lambda= 0.3660379465131937, subsample=0.611471430211575)

model_xgb



model_LGB=LGBMRegressor(objective='regression_l1', random_state=18, subsample_freq=1,

                        colsample_bytree=0.3261853512759363, min_child_samples=221, n_estimators=2151, num_leaves= 45, 

                        reg_alpha=0.9113713668943361, reg_lambda=0.8220990333713991, subsample=0.49969995651550947, 

                        max_bin=202, learning_rate=0.02959820893211799) #,device='gpu')

model_LGB





# model_Cat=CatBoostRegressor(loss_function='MAE', random_seed=18, task_type='GPU', cat_features=cat_features, verbose=False,

#                             iterations=2681, learning_rate=0.2127106032536721, depth=7, l2_leaf_reg=5.266150673910493, 

#                             random_strength=7.3001140226199315, bagging_temperature=0.26098669708900213)

# model_Cat





# model_Cat.fit(Xtrain, Ytrain)

model_LGB.fit(Xtrain, Ytrain)

model_xgb.fit(Xtrain, Ytrain)



lgb_predictions=model_LGB.predict(Xtest)

# cat_predictions=model_Cat.predict(Xtest)

xgb_predictions=model_xgb.predict(Xtest)



predictions=(lgb_predictions  + xgb_predictions)/2



predictions=np.exp(predictions)-1

submission['loss']=predictions

submission.to_csv('Result.csv')

submission.head()