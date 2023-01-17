import numpy  as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

import gc

import warnings

warnings.filterwarnings("ignore")

from sklearn import preprocessing

import os



from lightgbm import LGBMRegressor

from sklearn.base import BaseEstimator, RegressorMixin, clone

from sklearn.metrics import mean_squared_log_error



#将图表嵌入到notebook中

%matplotlib inline 
def input_file(file):

    path = f"/kaggle/input/ashrae-energy-prediction/{file}"

    if not os.path.exists(path): return path + ".gz"

    return path

## Function to reduce the DF size

def compress_dataframe(df):

    result = df.copy()

    for col in result.columns:

        col_data = result[col]

        dn = col_data.dtype.name

        if dn == "object":

            result[col] = pd.to_numeric(col_data.astype("category").cat.codes, downcast="integer")

        elif dn == "bool":

            result[col] = col_data.astype("int8")

        elif dn.startswith("int") or (col_data.round() == col_data).all():

            result[col] = pd.to_numeric(col_data, downcast="integer")

        else:

            result[col] = pd.to_numeric(col_data, downcast='float')

    return result
def make_is_bad_zero(Xy_subset, min_interval=48, summer_start=3000, summer_end=7500):

    """Helper routine for 'find_bad_zeros'.

    

    This operates upon a single dataframe produced by 'groupby'. We expect an 

    additional column 'meter_id' which is a duplicate of 'meter' because groupby 

    eliminates the original one."""

    

    meter = Xy_subset.meter_id.iloc[0]

    is_zero = Xy_subset.meter_reading == 0

    #print(is_zero)

    

    if meter == 0:

        # Electrical meters should never be zero. Keep all zero-readings in this table so that

        # they will all be dropped in the train set.

        return is_zero

    

    transitions = (is_zero != is_zero.shift(1))

    

    #print(transitions)

    all_sequence_ids = transitions.cumsum()

    

    

    #print(all_sequence_ids)

    ids = all_sequence_ids[is_zero].rename("ids")

    #print(ids)

    

    

    if meter in [2, 3]:

        # It's normal for steam and hotwater to be turned off during the summer

        keep = set(ids[(Xy_subset.timestamp < summer_start) |

                       (Xy_subset.timestamp > summer_end)].unique())

        is_bad = ids.isin(keep) & (ids.map(ids.value_counts()) >= min_interval)

    elif meter == 1:

        time_ids = ids.to_frame().join(Xy_subset.timestamp).set_index("timestamp").ids

        is_bad = ids.map(ids.value_counts()) >= min_interval



        # Cold water may be turned off during the winter

        jan_id = time_ids.get(0, False)

        dec_id = time_ids.get(8283, False)

        if (jan_id and dec_id and jan_id == time_ids.get(500, False) and

                dec_id == time_ids.get(8783, False)):

            is_bad = is_bad & (~(ids.isin(set([jan_id, dec_id]))))

    else:

        raise Exception(f"Unexpected meter type: {meter}")



    result = is_zero.copy()

    result.update(is_bad)

    return result



def find_bad_zeros(X,y):

    """Return an Index object containing only the rows which should be deleted."""

    Xy=X.assign(meter_reading=y,meter_id=X.meter)

    is_bad_zero=Xy.groupby(["building_id","meter"]).apply(make_is_bad_zero)

    return is_bad_zero[is_bad_zero].index.droplevel([0,1])
def read_train():

    df=pd.read_csv('/kaggle/input/ashrae-energy-prediction/train.csv',parse_dates=["timestamp"])

    df.timestamp=(df.timestamp-pd.to_datetime("2016-01-01")).dt.total_seconds()//3600

    return compress_dataframe(df)
def read_test():

    df=pd.read_csv('/kaggle/input/ashrae-energy-prediction/test.csv',parse_dates=["timestamp"])

    df.timestamp=(df.timestamp-pd.to_datetime("2016-01-01")).dt.total_seconds()//3600

    

    

    return compress_dataframe(df).set_index("row_id")
def read_building_metadata():

    return compress_dataframe(pd.read_csv('/kaggle/input/ashrae-energy-prediction/building_metadata.csv').fillna(-1).set_index("building_id"))
#从EDA 时间偏移分析出来的时间偏移量

site_GMT_offsets = [-5, 0, -9, -6, -8, 0, -6, -6, -5, -7, -8, -6, 0, -7, -6, -6]
#这里取消了 add_na_indicators

def read_weather_train(fix_timestamps=True,interpolate_na=True):

    df=pd.read_csv('/kaggle/input/ashrae-energy-prediction/weather_train.csv',parse_dates=["timestamp"])

    df["timestamp"]=(df.timestamp-pd.to_datetime("2016-01-01")).dt.total_seconds()//3600

    

    if fix_timestamps:

        GMT_offset_map={site:offset for site,offset in enumerate(site_GMT_offsets)}

        df.timestamp=df.timestamp+df.site_id.map(GMT_offset_map)

    if interpolate_na:

        site_dfs=[]

        

        for site_id in df.site_id.unique():

            #训练集2016年有366天

            site_df=df[df.site_id==site_id].set_index("timestamp").reindex(range(8784))  

            site_df.site_id=site_id

            

            for col in [c for c in site_df.columns if c!="site_id"]:

                

                #经过EDA测试，天气的相关缺失值使用线性填补

                site_df[col]=site_df[col].interpolate(limit_direction='both',method='linear')

            site_dfs.append(site_df)

        df=pd.concat(site_dfs).reset_index()

    return compress_dataframe(df).set_index(["site_id","timestamp"])
#这里取消了 add_na_indicators

def read_weather_test(fix_timestamps=True,interpolate_na=True):

    df=pd.read_csv('/kaggle/input/ashrae-energy-prediction/weather_test.csv',parse_dates=["timestamp"])

    df["timestamp"]=(df.timestamp-pd.to_datetime("2016-01-01")).dt.total_seconds()//3600

    

    if fix_timestamps:

        GMT_offset_map={site:offset for site,offset in enumerate(site_GMT_offsets)}

        df.timestamp=df.timestamp+df.site_id.map(GMT_offset_map)

    

    if interpolate_na:

        site_dfs=[]

        

        for site_id in df.site_id.unique():

            #测试集2017年有365天

            site_df=df[df.site_id==site_id].set_index("timestamp").reindex(range(8760))  

            site_df.site_id=site_id

            

            for col in [c for c in site_df.columns if c!="site_id"]:

                

                #经过EDA测试，天气的相关缺失值使用线性填补

                site_df[col]=site_df[col].interpolate(limit_direction='both',method='linear')

            site_dfs.append(site_df)

        df=pd.concat(site_dfs).reset_index()

    return compress_dataframe(df).set_index(["site_id","timestamp"])
def combined_train_data(fix_timestamps=True,interpolate_na=True,add_na_indicators=True):

    Xy=compress_dataframe(read_train()

                        .join(read_building_metadata(),on="building_id")

                        .join( read_weather_train(fix_timestamps,interpolate_na),on=["site_id","timestamp"])

                        .fillna(-1))

    

    return Xy.drop(columns=["meter_reading"]), Xy.meter_reading
def _add_time_features(X):

    return X.assign(tm_day_of_week=(X.timestamp//24)%7,tm_hour_of_day=(X.timestamp%24))
def combined_test_data(fix_timestamps=True, interpolate_na=True, add_na_indicators=True):

    X = compress_dataframe(read_test().join(read_building_metadata(), on="building_id").join(

        read_weather_test(fix_timestamps, interpolate_na),

        on=["site_id", "timestamp"]).fillna(-1))

    return X
#根据EDA显示去掉电力中的  building_id  0-104 是site_id==0

def find_bad_sitezero(X):

    return X[(X.timestamp<3378)&(X.site_id==0)&(X.meter==0)].index
#经过EDA 发现1099号建筑的蒸汽数据有问题，需要去掉

def find_bad_building1099(X,y):

    return X[(X.building_id==1099)&(X.meter==2)].index
# 移除经过EDA发现的异常数据

#包括四种能源数据的季节性清洗以及大块0数据以及异常高的数据

def find_bad_rows(X,y):

    return find_bad_zeros(X,y).union(find_bad_sitezero(X)).union(find_bad_building1099(X,y))
class CatSplitRegressor(BaseEstimator,RegressorMixin):

    def __init__(self,model,col):

        self.model=model

        self.col=col

    

    def fit(self,X,y):

        self.fitted={}

        importances=[]

        

        for val in X[self.col].unique():

            X1=X[X[self.col]==val].drop(columns=[self.col])

            self.fitted[val]=clone(self.model).fit(X1,y.reindex_like(X1))

            importances.append(self.fitted[val].feature_importances_)

            del X1

        

        fi=np.average(importances,axis=0)

        col_index=list(X.columns).index(self.col)

        self.feature_importances_=[*fi[:col_index],0,*fi[col_index:]]

        return self

    

    def predict(self,X):

        result=np.zeros(len(X))

        for val in X[self.col].unique():

            ix=np.nonzero((X[self.col]==val).to_numpy())

            predictions=self.fitted[val].predict(X.iloc[ix].drop(columns=[self.col]))

            result[ix]=predictions

        return result
categorical_columns=["building_id","meter","site_id","primary_use","tm_day_of_week","tm_hour_of_day"]
class LGBMWrapper(BaseEstimator,RegressorMixin):

    def __init__(self,categorical_feature=None,**params):

        self.model=LGBMRegressor(**params)

        self.categorical_feature=categorical_feature

    def fit(self,X,y):

        with warnings.catch_warnings():

            cats = None if self.categorical_feature is None else list(

                X.columns.intersection(self.categorical_feature)

            )

            warnings.filterwarnings("ignore","categorical_feature in Dataset is overriden".lower())

            self.model.fit(X,y,**({} if cats is None else {"categorical_feature":cats}))

            self.feature_importances_=self.model.feature_importances_

            

            return self

    

    def predict(self,X):

        return self.model.predict(X)

    

    def get_params(self,deep=True):

        return {**self.model.get_params(deep),"categorical_feature":self.categorical_feature}

    

    def set_params(self,**params):

        ctf=params.pop("categorical_feature",None)

        if ctf is not None: self.categorical_feature=ctf

        self.model.set_params(params)
X,y=combined_train_data()

bad_rows=find_bad_rows(X,y)

pd.Series(bad_rows.sort_values()).to_csv("rows_to_drop.csv",header=False,index=False)



X=X.drop(index=bad_rows)

y=y.reindex_like(X)



#Additional preprocessing



#增加了 dayofweek 以及hourofday

X=compress_dataframe(_add_time_features(X))



#square_feet 平滑处理

X['square_feet']=np.log1p(X['square_feet'])



#对 primary_use 进行转换

#labelEncoder = preprocessing.LabelEncoder()

#X['primary_use'] = labelEncoder.fit_transform(X['primary_use'].astype(str))



#原始的时间戳特征在预测的时候并没有什么作用

X=X.drop(columns="timestamp")



y=np.log1p(y)



model=CatSplitRegressor(

    LGBMWrapper(random_state=0,n_jobs=-1,categorical_feature=categorical_columns),"meter"

)



model.fit(X,y)

del X,y
X=combined_test_data()



#增加了 dayofweek 以及hourofday

X=compress_dataframe(_add_time_features(X))



#原始的时间戳特征在预测的时候并没有什么作用

X=X.drop(columns="timestamp")



#对 primary_use 进行转换

#labelEncoder = preprocessing.LabelEncoder()

#X['primary_use'] = labelEncoder.fit_transform(X['primary_use'].astype(str))



#square_feet 平滑处理

X['square_feet']=np.log1p(X['square_feet'])



predictions=pd.DataFrame({

    "row_id":X.index,

    "meter_reading":np.clip(np.expm1(model.predict(X)),0,None)

    

})



del X
predictions.to_csv("submission.csv",index=False,float_format="%.4f")