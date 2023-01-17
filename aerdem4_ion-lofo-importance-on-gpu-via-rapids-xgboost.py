import sys

!cp ../input/rapids/rapids.0.12.0 /opt/conda/envs/rapids.tar.gz

!cd /opt/conda/envs/ && tar -xzvf rapids.tar.gz

sys.path = ["/opt/conda/envs/rapids/lib/python3.6/site-packages"] + sys.path

sys.path = ["/opt/conda/envs/rapids/lib/python3.6"] + sys.path

sys.path = ["/opt/conda/envs/rapids/lib"] + sys.path 

!cp /opt/conda/envs/rapids/lib/libxgboost.so /opt/conda/lib/
!git clone https://github.com/aerdem4/rapids-kaggle-utils.git

%cd rapids-kaggle-utils/
!pip install lofo-importance
!pip install -U xgboost
import numpy as np 

import pandas as pd

from sklearn import *

import lightgbm as lgb

import cudf



train = cudf.read_csv('/kaggle/input/liverpool-ion-switching/train.csv')
# rapids-kaggle-utils

from cu_utils.transform import cu_min_transform, cu_max_transform, cu_mean_transform





CU_FUNC = {"min": cu_min_transform, "max": cu_max_transform, "mean": cu_mean_transform}
%%time





def features(df):

    df = df.sort_values(by=['time']).reset_index(drop=True)

    df["index"] = (df["time"] * 10_000) - 1

    df['batch'] = df["index"] // 50_000

    df['batch_index'] = df["index"]  - (df["batch"] * 50_000)

    df['batch_slices'] = df['batch_index']  // 5_000

    df['batch_slices2'] = df['batch'].astype(str) + "_" + df['batch_slices'].astype(str)

    

    for c in ['batch','batch_slices2']:



        df["abs_signal"] = df["signal"].abs()

        for abs_val in [True, False]:

            for func in ["min", "max", "mean"]:

                output_col = func + c

                input_col = "signal"

                if abs_val:

                    output_col = "abs_" + output_col

                    input_col = "abs_" + input_col

                df = df.groupby([c], method='cudf').apply_grouped(CU_FUNC[func],

                                                                  incols={input_col: 'x'},

                                                                  outcols=dict(y_out=np.float32),

                                                                  tpb=32).rename({'y_out': output_col})

        

        df['range'+c] = df['max'+c] - df['min'+c]

        df['maxtomin'+c] = df['max'+c] / df['min'+c]

        df['abs_avg'+c] = (df['abs_min'+c] + df['abs_max'+c]) / 2



    for c in [c1 for c1 in df.columns if c1 not in ['time', 'signal', 'open_channels', 'batch', 'batch_index', 'batch_slices', 'batch_slices2']]:

        df[c+'_msignal'] = df[c] - df['signal']

        

    return df



train = features(train)

train.shape
from lofo import LOFOImportance, Dataset, plot_importance

from sklearn.model_selection import KFold

import xgboost



# Convert to pandas for now. Xgboost supports cudf but LOFO doesn't support yet

sample_df = train.to_pandas().sample(frac=0.1, random_state=0)

sample_df.sort_values("time", inplace=True)



# define the validation scheme

cv = KFold(n_splits=5, shuffle=False, random_state=0)



# define the binary target and the features

features = [c for c in train.columns if c not in ['time', 'open_channels', 'batch', 'batch_index', 'batch_slices', 'batch_slices2']]

dataset = Dataset(df=sample_df, target="open_channels", features=features)



# define the validation scheme and scorer

params ={'learning_rate': 0.8, 'max_depth': 4, "n_estimators ": 100, "tree_method": 'gpu_hist', "gpu_id": 0}

xgb_reg = xgboost.XGBRegressor(**params)

lofo_imp = LOFOImportance(dataset, cv=cv, scoring="neg_mean_squared_error", model=xgb_reg)



# get the mean and standard deviation of the importances in pandas format

importance_df = lofo_imp.get_importance()
plot_importance(importance_df, figsize=(12, 20))