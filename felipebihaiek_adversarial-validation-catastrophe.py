import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import optuna

#pd.set_option('display.max_columns', 500)

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split









from collections import OrderedDict

import numpy as np

from matplotlib.pylab import plt

%matplotlib inline



from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler, MinMaxScaler

from sklearn.metrics import mean_squared_error







import lightgbm as lgb

import gc



import sys

sys.path.append('../input/iterative-stratification/iterative-stratification-master')

#sys.path.append('..')

from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
def reduce_mem_usage(props):

    start_mem_usg = props.memory_usage().sum() / 1024**2 

    print("Memory usage of properties dataframe is :",start_mem_usg," MB")

    NAlist = [] # Keeps track of columns that have missing values filled in. 

    for col in props.columns:

        if props[col].dtype != object:  # Exclude strings

            

            # Print current column type

            #print("******************************")

           # print("Column: ",col)

            #print("dtype before: ",props[col].dtype)

            

            # make variables for Int, max and min

            IsInt = False

            mx = props[col].max()

            mn = props[col].min()

            

            # Integer does not support NA, therefore, NA needs to be filled

            if not np.isfinite(props[col]).all(): 

                NAlist.append(col)

                props[col].fillna(mn-1,inplace=True)  

                   

            # test if column can be converted to an integer

            asint = props[col].fillna(0).astype(np.int64)

            result = (props[col] - asint)

            result = result.sum()

            if result > -0.01 and result < 0.01:

                IsInt = True



            

            # Make Integer/unsigned Integer datatypes

            if IsInt:

                if mn >= 0:

                    if mx < 255:

                        props[col] = props[col].astype(np.uint8)

                    elif mx < 65535:

                        props[col] = props[col].astype(np.uint16)

                    elif mx < 4294967295:

                        props[col] = props[col].astype(np.uint32)

                    else:

                        props[col] = props[col].astype(np.uint64)

                else:

                    if mn > np.iinfo(np.int8).min and mx < np.iinfo(np.int8).max:

                        props[col] = props[col].astype(np.int8)

                    elif mn > np.iinfo(np.int16).min and mx < np.iinfo(np.int16).max:

                        props[col] = props[col].astype(np.int16)

                    elif mn > np.iinfo(np.int32).min and mx < np.iinfo(np.int32).max:

                        props[col] = props[col].astype(np.int32)

                    elif mn > np.iinfo(np.int64).min and mx < np.iinfo(np.int64).max:

                        props[col] = props[col].astype(np.int64)    

            

            # Make float datatypes 32 bit

            else:

                props[col] = props[col].astype(np.float32)

            

            # Print new column type

           # print("dtype after: ",props[col].dtype)

           # print("******************************")

    

    # Print final result

    print("___MEMORY USAGE AFTER COMPLETION:___")

    mem_usg = props.memory_usage().sum() / 1024**2 

    print("Memory usage is: ",mem_usg," MB")

    print("This is ",100*mem_usg/start_mem_usg,"% of the initial size")

    return props, NAlist


def train_short_form_loader(feature_file,target_file,extra_target_file=None):

    '''takes the original target and features and creates a train dataset 

    in col long format'''





    train_features = pd.read_csv(feature_file)



    train_targets = pd.read_csv(target_file)

    train_features,_= reduce_mem_usage(train_features)

    train_targets,_ = reduce_mem_usage(train_targets)





    if extra_target_file is not None:

        extra_targets = pd.read_csv(extra_target_file)

        extra_targets,_ = reduce_mem_usage(extra_targets)

        train_targets = pd.concat([train_targets,extra_targets])

        del extra_targets



    targets = train_targets.columns[1:]



    train_melt=train_targets.merge(train_features,how="left",on="sig_id")





    del train_features,train_targets





    train_melt.set_index("sig_id",inplace=True)



    #train_melt["variable"]= train_melt["variable"].astype('category')

    train_melt["cp_type"]= train_melt["cp_type"].astype('category')

    train_melt["cp_dose"]= train_melt["cp_dose"].astype('category')



    return train_melt , targets







def test_short_form_loader(feature_file):

    '''takes the original target and features and creates a train dataset 

    in col long format'''





    train_features = pd.read_csv(feature_file)



    #train_targets = pd.read_csv(target_file)

    train_features,_= reduce_mem_usage(train_features)

    #train_targets,_ = reduce_mem_usage(train_targets)



    train_melt =  train_features.copy()

    del train_features





    train_melt.set_index("sig_id",inplace=True)



    #train_melt["variable"]= train_melt["variable"].astype('category')

    train_melt["cp_type"]= train_melt["cp_type"].astype('category')

    train_melt["cp_dose"]= train_melt["cp_dose"].astype('category')



    return train_melt 

def dataset_splitter(data,**kwargs):

    '''splits by index and does not mix sig ids

    @might want to allow some mixing for regularization later on '''

    train_ids,test_ids=train_test_split(data.index.unique(),**kwargs)

    return data.loc[train_ids], data.loc[test_ids]
train_df ,targets= train_short_form_loader("../input/lish-moa/train_features.csv","../input/lish-moa/train_targets_scored.csv")
train_df.drop(targets,axis=1,inplace=True)
train_df.head()
online_df = test_short_form_loader("../input/lish-moa/test_features.csv")
train_df["is_test"]=0

online_df["is_test"]=1



train_df=pd.concat([train_df,online_df])
del online_df

gc.collect()
train_df.head()
train_df, test_df =dataset_splitter(train_df)
test_df.head()
model=lgb.LGBMClassifier(n_jobs=4)
test_df.index.isin(train_df.index).any()
recorder={}

model.fit(train_df.drop("is_test",axis=1),train_df["is_test"],eval_metric=["logloss",'auc'],eval_set=[(train_df.drop("is_test",axis=1),train_df["is_test"]),(test_df.drop("is_test",axis=1),test_df["is_test"])],callbacks=[lgb.callback.record_evaluation(recorder)],verbose=False)



ig, ax = plt.subplots()

ax.plot(recorder['valid_0']['binary_logloss'], label='Train')

ax.plot(recorder['valid_1']['binary_logloss'], label='Val')

ax.legend()

plt.ylabel('logloss')

plt.title('Lightgbm first take logloss')

plt.show()
ig, ax = plt.subplots()

ax.plot(recorder['valid_0']['auc'], label='Train')

ax.plot(recorder['valid_1']['auc'], label='Val')

ax.legend()

plt.ylabel('AUC')

plt.title('AUC')

plt.show()
lgb.plot_importance(model,max_num_features=30)
import shap
%time shap_values = shap.TreeExplainer(model).shap_values(test_df.drop("is_test",axis=1))
len(shap_values)
plt.figure(figsize=(16,12))

shap.summary_plot(shap_values[0], test_df.drop("is_test",axis=1) ,max_display=50)
train_df.groupby("is_test")['c-62'].agg(['mean','std','median'])
test_df.groupby("is_test")['c-62'].agg(['mean','std','median'])
import seaborn as sn

sn.boxplot(x='is_test',y='c-62',data=train_df)



#sn.distplot(test_df.loc[train_df["is_test"]==1,'c-62'],hist=True, rug=True)

sn.violinplot(x='is_test',y='c-62',data=train_df)
sn.violinplot(x='is_test',y='c-62',data=test_df)
sn.violinplot(x='is_test',y='g-375',data=test_df)