import numpy as np

import pandas as pd

from catboost import CatBoostClassifier,CatBoostRegressor, Pool

from sklearn.metrics import accuracy_score, mean_squared_error

from sklearn.model_selection import train_test_split

from math import sqrt

from datetime import datetime

pd.options.mode.chained_assignment = None
class DataProvider():

    def __init__(self, y_columns, gender='wta', start_year=2011, test_year=None, test_date=None, oddsMode='No', diffsMode='No', 

                 noStatsGenerated=False, noStats=False, diffs=[], isDropSources=False, verbose=False, isCatboost=False):

        self.gender = gender

        self.start_year = start_year

        self.test_year = test_year

        self.test_date = test_date

        self.y_columns = y_columns

        self.verbose = verbose

        self.isDropSources = isDropSources

        self.isCatboost=isCatboost

        # No - No lines info

        # Full - Full lines info (drop nans)

        # Equal - Equal odds only (.47 - .53)

        self.oddsMode = oddsMode 

        

        # Time - Last to Current

        # Player - One player via another

        # Both - Diffs on diffs

        self.diffsMode = diffsMode 

        

        self.noStatsGenerated = noStatsGenerated

        self.noStats=noStats

        self.diffs = diffs

        

        self.cols_info_orig=['TourRank','RID_CUR','TName','GameD','Year', 'Name_1','Name_2','Result_CUR_1','GRes_CUR_1','PS_CUR_1','TTL_CUR_1',

                             'TPoints_CUR_1','K_CUR_1','K_CUR_2','F_CUR_1', 'F_CUR_2','SETS_CUR_1','K20_CUR_1','K20_CUR_2','K21_CUR_1','K21_CUR_2'] if isCatboost else ['TourRank','RID_CUR','TName','GameD',

                                'Year', 'Name_1','Name_2','Result_CUR_1','GRes_CUR_1','PS_CUR_1','TTL_CUR_1', 'SETS_0-2_CUR_1','SETS_1-2_CUR_1','SETS_2-0_CUR_1','SETS_2-1_CUR_1',

                                'TPoints_CUR_1','K_CUR_1','K_CUR_2','F_CUR_1', 'F_CUR_2'] 

        self.cols_info=['TourRank','RID','Tour','GameD','Year','Name_1','Name_2','Result','GRes','PS','TTL', 'TPoints','K1','K2', 'F1','F2','SETS',

                        'K20','K02','K21','K12' ] if isCatboost else ['TourRank','RID','Tour','GameD','Year','Name_1','Name_2','Result','GRes','PS','TTL', 'SETS_0-2','SETS_1-2','SETS_2-0',

                                                       'SETS_2-1','TPoints','K1','K2','F1','F2'] 

        self.cols_ex=['TourCountry','TName','GameD','Name_1','Name_2','Result_CUR_1','GRes_CUR_1','PS_CUR_1','TTL_CUR_1','StatsGenerated_CUR_1','StatsGenerated_CUR_2',

                      'SETS_CUR_1', 'Unnamed: 0', 'Result_CUR_2', 'PS_CUR_2', 'Year'] if isCatboost else ['TourCountry','TName','GameD','Name_1','Name_2','Result_CUR_1','GRes_CUR_1','PS_CUR_1','TTL_CUR_1','StatsGenerated_CUR_1','StatsGenerated_CUR_2',

                      'SETS_0-2_CUR_1','SETS_1-2_CUR_1','SETS_2-0_CUR_1','SETS_2-1_CUR_1', 'Unnamed: 0', 'Result_CUR_2', 'PS_CUR_2', 'Year']

    

    def load_data(self):

        df=pd.read_csv(f'../input/tennis-20112019/{self.gender}.csv')



        if self.isCatboost:

            di = {'SETS_2-0_CUR_1' : 0, 'SETS_0-2_CUR_1' : 1, 'SETS_2-1_CUR_1' : 2, 'SETS_1-2_CUR_1' : 3}

            df['SETS_CUR_1']=df[['SETS_0-2_CUR_1','SETS_1-2_CUR_1','SETS_2-0_CUR_1','SETS_2-1_CUR_1']].idxmax(axis=1)

            df['SETS_CUR_1']=df['SETS_CUR_1'].map(di)



            di = {'SETS_2-0_1' : 0, 'SETS_0-2_1' : 1, 'SETS_2-1_1' : 2, 'SETS_1-2_1' : 3}

            df['SETS_1']=df[['SETS_2-0_1', 'SETS_0-2_1', 'SETS_2-1_1', 'SETS_1-2_1']].idxmax(axis=1)

            df['SETS_1']=df['SETS_1'].map(di)

            

            di = {'SETS_2-0_2' : 0, 'SETS_0-2_2' : 1, 'SETS_2-1_2' : 2, 'SETS_1-2_2' : 3}

            df['SETS_2']=df[['SETS_2-0_2', 'SETS_0-2_2', 'SETS_2-1_2', 'SETS_1-2_2']].idxmax(axis=1)

            df['SETS_2']=df['SETS_2'].map(di)

            df.drop(['SETS_0-2_CUR_1','SETS_1-2_CUR_1','SETS_2-0_CUR_1','SETS_2-1_CUR_1','SETS_0-2_1','SETS_1-2_1','SETS_2-0_1','SETS_2-1_1',

                     'SETS_0-2_2','SETS_1-2_2','SETS_2-0_2','SETS_2-1_2'], axis=1, inplace=True)

        

        if self.verbose:

            print(f'Set start to {self.start_year} year ({len(df)} records).')

        

        if self.noStatsGenerated:

            df=df[(df['StatsGenerated_CUR_1']==0) & (df['StatsGenerated_CUR_2']==0)]

            if self.verbose:

                print(f'Remove generated stats ({len(df)} records).')

        

        if self.oddsMode=='No':

            dfi = df[self.cols_info_orig]

            df.drop(['K_CUR_1', 'TPoints_CUR_1', 'F_CUR_1','K_CUR_2', 'TPoints_CUR_2', 'F_CUR_2','K20_CUR_1','K21_CUR_1','K20_CUR_2','K21_CUR_2'], axis=1, inplace=True)

        elif self.oddsMode=='Full':

            if not self.isCatboost:

                df.dropna(subset=['K_CUR_1', 'TPoints_CUR_1', 'F_CUR_1','K_CUR_2', 'TPoints_CUR_2', 'F_CUR_2'], inplace=True)

            if self.verbose:

                print(f'Full odds mode has applied ({len(df)} records).')

        elif self.oddsMode=='Equal':

            df=df[(df['K_CUR_1']>=0.47) & (df['K_CUR_1']<=0.53)]

            if self.verbose:

                print(f'Equal odds mode 2 has applied ({len(df)} records).')

            

        if self.oddsMode!='No':

            dfi = df[self.cols_info_orig]

        dfi.columns=self.cols_info

        # ********* DIFFS CALCULATION *********

        for diff in self.diffs:

            if diff=='RID':

                if self.diffsMode=='Time' or self.diffsMode=='Both':

                    df[f'{diff}_DT_1']=df[f'{diff}_1']-df[f'{diff}_CUR']

                    df[f'{diff}_DT_2']=df[f'{diff}_2']-df[f'{diff}_CUR']

                if self.diffsMode=='Player' or self.diffsMode=='Both':

                    df[f'{diff}_DP']=df[f'{diff}_1']-df[f'{diff}_2']

                if self.diffsMode=='Both':

                    df[f'{diff}_DTP']=df[f'{diff}_DT_1']-df[f'{diff}_DT_2']

                    if self.isDropSources:

                        df.drop([f'{diff}_DT_1',f'{diff}_DT_2'], axis=1, inplace=True)

            elif diff=='TourRank':

                if self.diffsMode=='Time' or self.diffsMode=='Both':

                    df[f'{diff}_DT_1']=df[f'{diff}_1']-df[f'{diff}']

                    df[f'{diff}_DT_2']=df[f'{diff}_2']-df[f'{diff}']

                if self.diffsMode=='Player' or self.diffsMode=='Both':

                    df[f'{diff}_DP']=df[f'{diff}_1']-df[f'{diff}_2']

                if self.diffsMode=='Both':

                    df[f'{diff}_DTP']=df[f'{diff}_DT_1']-df[f'{diff}_DT_2']

                    if self.isDropSources:

                        df.drop([f'{diff}_DT_1',f'{diff}_DT_2'], axis=1, inplace=True)

            elif diff=='K' or diff=='F' or diff=='TPoints':

                if self.diffsMode=='Time' or self.diffsMode=='Both':

                    df[f'{diff}_DT1_1']=df[f'{diff}_CUR_1']-df[f'{diff}_1']

                    df[f'{diff}_DT1_2']=df[f'{diff}_CUR_2']-df[f'{diff}_1']

                    df[f'{diff}_DT5_1']=df[f'{diff}_CUR_1']-df[f'{diff}_L5_1']

                    df[f'{diff}_DT5_2']=df[f'{diff}_CUR_2']-df[f'{diff}_L5_1']

                    df[f'{diff}_DTA_1']=df[f'{diff}_CUR_1']-df[f'{diff}_A_1']

                    df[f'{diff}_DTA_2']=df[f'{diff}_CUR_2']-df[f'{diff}_A_1']

                if self.diffsMode=='Player' or self.diffsMode=='Both':

                    df[f'{diff}_DP']=df[f'{diff}_CUR_1']-df[f'{diff}_CUR_2']

                if self.diffsMode=='Both':

                    df[f'{diff}_DTP1']=df[f'{diff}_DT1_1']-df[f'{diff}_DT1_2']

                    df[f'{diff}_DTP5']=df[f'{diff}_DT5_1']-df[f'{diff}_DT5_2']

                    df[f'{diff}_DTPA']=df[f'{diff}_DTA_1']-df[f'{diff}_DTA_2']

                    if self.isDropSources:

                        df.drop([f'{diff}_DT1_1',f'{diff}_DT1_2',f'{diff}_DT5_1',f'{diff}_DT5_2',f'{diff}_DTA_1',f'{diff}_DTA_2'], axis=1, inplace=True)

            elif diff=='Age':

                if self.diffsMode=='Player' or self.diffsMode=='Both':

                    df[f'{diff}_DP']=df[f'{diff}_CUR_1']-df[f'{diff}_CUR_2']

                    if self.isDropSources:

                        df.drop([f'{diff}_CUR_1',f'{diff}_CUR_1'], axis=1, inplace=True)

            else:

                if self.diffsMode=='Time' or self.diffsMode=='Both':

                    df[f'{diff}_DT1_1']=df[f'{diff}_1']-df[f'{diff}_A_1']

                    df[f'{diff}_DT1_2']=df[f'{diff}_2']-df[f'{diff}_A_2']

                    df[f'{diff}_DT5_1']=df[f'{diff}_L5_1']-df[f'{diff}_A_1']

                    df[f'{diff}_DT5_2']=df[f'{diff}_L5_2']-df[f'{diff}_A_2']

                if self.diffsMode=='Player' or self.diffsMode=='Both':

                    df[f'{diff}_DP']=df[f'{diff}_1']-df[f'{diff}_2']

                    df[f'{diff}_DP5']=df[f'{diff}_L5_1']-df[f'{diff}_L5_2']

                    df[f'{diff}_DPA']=df[f'{diff}_A_1']-df[f'{diff}_A_2']

                if self.diffsMode=='Both':

                    df[f'{diff}_DT1P']=df[f'{diff}_DT1_1']-df[f'{diff}_DT1_2']

                    df[f'{diff}_DT5P']=df[f'{diff}_DT5_1']-df[f'{diff}_DT5_2']

                    if self.isDropSources:

                        df.drop([f'{diff}_DT1_1',f'{diff}_DT1_2',f'{diff}_DT5_1',f'{diff}_DT5_2'], axis=1, inplace=True)

                if self.isDropSources:

                    df.drop([f'{diff}_1',f'{diff}_2',f'{diff}_L5_1',f'{diff}_L5_2',f'{diff}_A_1',f'{diff}_A_2'], axis=1, inplace=True)

        return self.split_xy(df, dfi)

        

    def split_xy(self, df, dfi):

        if self.test_year:

            dfyr = df[df['Year'] == self.test_year]

            df = df[df['Year'] < self.test_year]

            dfyr_info = dfi[dfi['Year'] >= self.test_year]

            df_info = dfi[dfi['Year'] < self.test_year]

            if self.verbose:

                print(f'Data was divided to main ({len(df)} rows) and {self.test_year} ({len(dfyr)} rows) parts.')

        elif self.test_date:

            dfyr = df[df['GameD'] == self.test_date]

            df = df[df['GameD'] < self.test_date]

            dfyr_info = dfi[dfi['GameD'] == self.test_date]

            df_info = dfi[dfi['GameD'] < self.test_date]

            

        if self.noStats:

            cols_Aces=[col for col in dfyr if col.startswith('Aces')]

            cols_BreakPoints=[col for col in dfyr if col.startswith('BreakPoints')]

            cols_DoubleFaults=[col for col in dfyr if col.startswith('DoubleFaults')]

            cols_ReceivingPoints=[col for col in dfyr if col.startswith('ReceivingPoints')]

            cols_TotalPoints=[col for col in dfyr if col.startswith('TotalPoints')]

            cols_Serve=[col for col in dfyr if col.startswith('Serve')] # Both

            cols_stats=cols_Aces+cols_BreakPoints+cols_DoubleFaults+cols_ReceivingPoints+cols_TotalPoints+cols_Serve

        

        df_y=df[self.y_columns]

        df.drop(self.cols_ex, axis=1, inplace=True, errors='ignore')

        dfyr_y=dfyr[self.y_columns]

        dfyr.drop(self.cols_ex, axis=1, inplace=True, errors='ignore')

        if self.noStats:

            df.drop(cols_stats, axis=1, inplace=True, errors='ignore')

            dfyr.drop(cols_stats, axis=1, inplace=True, errors='ignore')

                

        return (df_info, df, df_y, dfyr_info, dfyr, dfyr_y)
from sklearn.base import BaseEstimator



class BasicTransformer(BaseEstimator):

    

    def __init__(self, num_strategy='median',  return_df=False, isCatboost=True):

        self.cols_cat=['RID_CUR','Surface','TourRank','Month','WeekDay','RID_1','TourRank_1','RID_2','TourRank_2',

                       'SETS_1','SETS_2'] if isCatboost else ['RID_CUR','Surface','TourRank','Month','WeekDay','RID_1','TourRank_1','RID_2','TourRank_2']

        self.cols_bin=['IsHome_CUR_1','IsBirthDay_CUR_1','IsLastRet_CUR_1','IsHome_1','HomeChanged_CUR_1','TourChanged_CUR_1','SurfaceChanged_CUR_1', 'GRes_1', 'IsHome_CUR_2',

                       'IsBirthDay_CUR_2','IsLastRet_CUR_2','IsHome_2','HomeChanged_CUR_2','TourChanged_CUR_2','SurfaceChanged_CUR_2', 'GRes_2'] if isCatboost else ['IsHome_CUR_1',

                       'IsBirthDay_CUR_1','IsLastRet_CUR_1','IsHome_1','HomeChanged_CUR_1','TourChanged_CUR_1','SurfaceChanged_CUR_1',

                        'GRes_1','SETS_0-2_1','SETS_1-2_1','SETS_2-0_1','SETS_2-1_1',

                        'IsHome_CUR_2','IsBirthDay_CUR_2','IsLastRet_CUR_2','IsHome_2','HomeChanged_CUR_2','TourChanged_CUR_2','SurfaceChanged_CUR_2',

                        'GRes_2','SETS_0-2_2','SETS_1-2_2','SETS_2-0_2','SETS_2-1_2',]

        

        if num_strategy not in ['mean', 'median']:

            raise ValueError('num_strategy must be either "mean" or "median"')

        self.num_strategy = num_strategy

        self.return_df = return_df

        self.isCatboost = isCatboost



    def transform(self, X):

        # check that we have a DataFrame with same column names as the one we fit

        if set(self._columns) != set(X.columns):

            raise ValueError('Passed DataFrame has different columns than fit DataFrame')

        elif len(self._columns) != len(X.columns):

            raise ValueError('Passed DataFrame has different number of columns than fit DataFrame')

            

        # fill missing values

        num_cols = self._column_dtypes['num']

        X_num = X[num_cols] if self.isCatboost else X[num_cols].fillna(self._num_fill) 

        # Copy binary columns

        X_bin=X[self._column_dtypes['bin']].values.astype(int)



        if not self.isCatboost:

            # Standardize numerics

            std = X_num.std()

            X_num = (X_num - X_num.mean()) / std

            zero_std = np.where(std == 0)[0]



            # If there is 0 standard deviation, then all values are the 

            # same. Set them to 0.

            if len(zero_std) > 0:

                X_num.iloc[:, zero_std] = 0

        X_num = X_num.values

        if self.isCatboost:

            X_cat = X[self._column_dtypes['cat']].values.astype(int)

        else:

            # create separate array for new encoded categoricals

            X_cat = np.empty((len(X), self._total_cat_cols), dtype='uint8')

            i = 0

            for col in self._column_dtypes['cat']:

                vals = self._cat_cols[col]

                for val in vals:

                    X_cat[:, i] = X[col] == val

                    i += 1

            X_cat=X_cat.astype(int)

        # concatenate transformed numeric and categorical arrays

        data = np.column_stack((X_bin, X_num, X_cat))

        # return either a DataFrame or an array

        if self.return_df:

            dfbin = pd.DataFrame(data=X_bin).astype('int32')

            dfnum = pd.DataFrame(data=X_num)

            dfcat = pd.DataFrame(data=X_cat).astype('int32')

            df = pd.concat([dfbin, dfnum, dfcat], axis=1)

            df.columns=self._feature_names

            return df

        else:

            return np.column_stack((X_bin, X_num, X_cat))

    

    def fit_transform(self, X, y=None):

        return self.fit(X).transform(X)

    

    def get_feature_names(self):

        return self._feature_names



    def fit(self, X, y=None):

        # Assumes X is a DataFrame



        # Set cat cols type to uint8

        X=self.set_binary(X)

        self._columns = X.columns.values



        # Split data into categorical and numeric

        self._dtypes = X.dtypes.values

        self._kinds = np.array([dt.kind for dt in X.dtypes])

        self._column_dtypes = {}

        is_num = self._kinds == 'f'

        self._column_dtypes['bin'] = np.intersect1d(self.cols_bin,X.columns)

        self._column_dtypes['cat'] = np.intersect1d(self.cols_cat,X.columns)

        self._column_dtypes['num'] = self._columns[is_num]

        self._feature_names = np.append(self._column_dtypes['bin'],self._column_dtypes['num'])

        if self.isCatboost:

            self._feature_names = np.append(self._feature_names, self._column_dtypes['cat'] )

        else:

            # Create a dictionary mapping categorical column to unique 

            # values above threshold

            self._cat_cols = {}

            for col in self._column_dtypes['cat']:

                vc = X[col].value_counts()

                vals = vc.index.values

                self._cat_cols[col] = vals

                self._feature_names = np.append(self._feature_names, [f'{col}_{val}' for val in vals])



            # get total number of new categorical columns    

            self._total_cat_cols = sum([len(v) for col, v in self._cat_cols.items()])

        

        # get mean or median

        num_cols = self._column_dtypes['num']

        self._num_fill = X[num_cols].agg(self.num_strategy)

        return self

        



    def set_binary(self,X):

        cols=self.cols_bin+self.cols_cat

        cols=np.intersect1d(cols,X.columns)

        for c in cols:

            X[c]=X[c].fillna(0).astype(np.uint8)

        return X
dp=DataProvider(['GRes_CUR_1'], gender='atp', start_year=2011, test_year=2019, oddsMode='No', diffsMode='Both', noStatsGenerated=False,isCatboost=True,

            diffs=['Age', 'RID', 'TourRank', 'GRes', 'TTL', 'PS', 'Aces', 'DoubleFaults', 'TotalPointsWon', 'ReceivingPointsWon', 

             'Serve1stPCT', 'Serve1stWonPCT', 'Serve2ndWonPCT', 'BreakPointsConvertedPCT', 'ReceivingPointsWonPCT'])

            #diffs=['Age', 'RID', 'TourRank', 'GRes', 'TTL', 'PS'])

df_info, df_x, df_y, info_year, df_year, y_year=dp.load_data() 

df_train, df_test, y_train, y_test = train_test_split(df_x, df_y, test_size=0.2, random_state=42) 

bt = BasicTransformer(return_df=True, num_strategy='mean',) 

X_train = bt.fit_transform(df_train) 

X_test=bt.transform(df_test) 

X_year=bt.transform(df_year) 

cate_features_index = np.where(X_train.dtypes != float)[0] 

train_pool = Pool(X_train,y_train,cat_features=cate_features_index) 

test_pool = Pool(X_test,y_test,cat_features=cate_features_index)



model = CatBoostClassifier(verbose=False, iterations=1000, loss_function= 'Logloss', eval_metric='AUC',use_best_model=True, random_seed=42, 

                           learning_rate=0.02, has_time=False, depth=8, l2_leaf_reg=5, random_strength=1, bagging_temperature=0)



model.fit(train_pool,eval_set=test_pool) 

preds_class = model.predict(X_year) 

atp_ml_acc=accuracy_score(preds_class,y_year)
dp=DataProvider(['GRes_CUR_1'], gender='wta', start_year=2011, test_year=2019, oddsMode='No', diffsMode='Both', noStatsGenerated=False,isCatboost=True,

            diffs=['Age', 'RID', 'TourRank', 'GRes', 'TTL', 'PS', 'Aces', 'DoubleFaults', 'TotalPointsWon', 'ReceivingPointsWon', 

             'Serve1stPCT', 'Serve1stWonPCT', 'Serve2ndWonPCT', 'BreakPointsConvertedPCT', 'ReceivingPointsWonPCT'])

            #diffs=['Age', 'RID', 'TourRank', 'GRes', 'TTL', 'PS'])

df_info, df_x, df_y, info_year, df_year, y_year=dp.load_data() 

df_train, df_test, y_train, y_test = train_test_split(df_x, df_y, test_size=0.2, random_state=42) 

bt = BasicTransformer(return_df=True, num_strategy='mean',) 

X_train = bt.fit_transform(df_train) 

X_test=bt.transform(df_test) 

X_year=bt.transform(df_year) 

cate_features_index = np.where(X_train.dtypes != float)[0] 

train_pool = Pool(X_train,y_train,cat_features=cate_features_index) 

test_pool = Pool(X_test,y_test,cat_features=cate_features_index)



model = CatBoostClassifier(verbose=False, iterations=1000, loss_function= 'Logloss', eval_metric='AUC',use_best_model=True, random_seed=42, 

                           learning_rate=0.02, has_time=False, depth=8, l2_leaf_reg=5, random_strength=1, bagging_temperature=0)



model.fit(train_pool,eval_set=test_pool) 

preds_class = model.predict(X_year) 

wta_ml_acc=accuracy_score(preds_class,y_year)
print('Accuracy of models above is {:.2%} for ATP and {:.2%} for WTA matches for 2019 year.'.format(atp_ml_acc, wta_ml_acc))
dp=DataProvider(['TTL_CUR_1'], gender='atp', start_year=2011, test_year=2019, oddsMode='No', diffsMode='Both', noStatsGenerated=False,

            diffs=['Age', 'RID', 'TourRank', 'GRes', 'TTL', 'PS', 'Serve1stWonPCT','BreakPointsConvertedPCT', 'ReceivingPointsWonPCT'])

            #diffs=['Age', 'RID', 'TourRank', 'GRes', 'TTL', 'PS'])

df_info, df_x, df_y, info_year, df_year, y_year=dp.load_data() 

df_train, df_test, y_train, y_test = train_test_split(df_x, df_y, test_size=0.2, random_state=42) 

bt = BasicTransformer(return_df=True, num_strategy='mean') 

X_train = bt.fit_transform(df_train) 

X_test=bt.transform(df_test) 

X_year=bt.transform(df_year) 

cate_features_index = np.where(X_train.dtypes != float)[0] 

train_pool = Pool(X_train,y_train,cat_features=cate_features_index) 

test_pool = Pool(X_test,y_test,cat_features=cate_features_index) 

model = CatBoostRegressor(verbose=False, iterations=500, loss_function= 'RMSE', eval_metric='RMSE',use_best_model=True, random_seed=42, 

                          learning_rate=0.03, has_time=False, depth=6, l2_leaf_reg=5, random_strength=0.5, bagging_temperature=0) 

model.fit(train_pool,eval_set=test_pool) 

preds_proba = model.predict(X_year)

atp_total=sqrt(mean_squared_error(preds_proba,y_year))
dp=DataProvider(['TTL_CUR_1'], gender='wta', start_year=2011, test_year=2019, oddsMode='No', diffsMode='Both', noStatsGenerated=False,

            diffs=['Age', 'RID', 'TourRank', 'GRes', 'TTL', 'PS', 'Serve1stWonPCT','BreakPointsConvertedPCT', 'ReceivingPointsWonPCT'])

            #diffs=['Age', 'RID', 'TourRank', 'GRes', 'TTL', 'PS'])

df_info, df_x, df_y, info_year, df_year, y_year=dp.load_data() 

df_train, df_test, y_train, y_test = train_test_split(df_x, df_y, test_size=0.2, random_state=42) 

bt = BasicTransformer(return_df=True, num_strategy='mean') 

X_train = bt.fit_transform(df_train) 

X_test=bt.transform(df_test) 

X_year=bt.transform(df_year) 

cate_features_index = np.where(X_train.dtypes != float)[0] 

train_pool = Pool(X_train,y_train,cat_features=cate_features_index) 

test_pool = Pool(X_test,y_test,cat_features=cate_features_index) 

model = CatBoostRegressor(verbose=False, iterations=500, loss_function= 'RMSE', eval_metric='RMSE',use_best_model=True, random_seed=42, 

                          learning_rate=0.03, has_time=False, depth=6, l2_leaf_reg=5, random_strength=0.5, bagging_temperature=0) 

model.fit(train_pool,eval_set=test_pool) 

preds_proba = model.predict(X_year)

wta_total=sqrt(mean_squared_error(preds_proba,y_year))
print('Total mean error is {:.2} for ATP and {:.2} for WTA matches.'.format(atp_total,wta_total))
dp=DataProvider(['SETS_CUR_1'], gender='atp', start_year=2011, test_year=2019, oddsMode='No', diffsMode='Both', noStatsGenerated=False,isCatboost=True,

            diffs=['Age', 'RID', 'TourRank', 'GRes', 'TTL', 'PS', 'Aces', 'DoubleFaults', 'TotalPointsWon', 'ReceivingPointsWon', 

            'Serve1stPCT', 'Serve1stWonPCT', 'Serve2ndWonPCT', 'BreakPointsConvertedPCT', 'ReceivingPointsWonPCT'])

            #diffs=['Age', 'RID', 'TourRank', 'GRes', 'TTL', 'PS'])



df_info, df_x, df_y, info_year, df_year, y_year=dp.load_data() 

df_train, df_test, y_train, y_test = train_test_split(df_x, df_y, test_size=0.2, random_state=42) 

bt = BasicTransformer(return_df=True, num_strategy='mean',) 

X_train = bt.fit_transform(df_train) 

X_test=bt.transform(df_test) 

X_year=bt.transform(df_year) 

cate_features_index = np.where(X_train.dtypes != float)[0] 

train_pool = Pool(X_train,y_train,cat_features=cate_features_index) 

test_pool = Pool(X_test,y_test,cat_features=cate_features_index)



model = CatBoostClassifier(verbose=False, iterations=250, loss_function= 'MultiClass', eval_metric='MultiClass',use_best_model=True, random_seed=42, 

                           learning_rate=0.2, has_time=False, depth=6, l2_leaf_reg=5, random_strength=1, bagging_temperature=0)



model.fit(train_pool,eval_set=test_pool) 

preds_class = model.predict(X_year) 

atp_sc=accuracy_score(preds_class,y_year)
dp=DataProvider(['SETS_CUR_1'], gender='wta', start_year=2011, test_year=2019, oddsMode='No', diffsMode='Both', noStatsGenerated=False,isCatboost=True,

            diffs=['Age', 'RID', 'TourRank', 'GRes', 'TTL', 'PS', 'Aces', 'DoubleFaults', 'TotalPointsWon', 'ReceivingPointsWon', 

            'Serve1stPCT', 'Serve1stWonPCT', 'Serve2ndWonPCT', 'BreakPointsConvertedPCT', 'ReceivingPointsWonPCT'])

            #diffs=['Age', 'RID', 'TourRank', 'GRes', 'TTL', 'PS'])



df_info, df_x, df_y, info_year, df_year, y_year=dp.load_data() 

df_train, df_test, y_train, y_test = train_test_split(df_x, df_y, test_size=0.2, random_state=42) 

bt = BasicTransformer(return_df=True, num_strategy='mean',) 

X_train = bt.fit_transform(df_train) 

X_test=bt.transform(df_test) 

X_year=bt.transform(df_year) 

cate_features_index = np.where(X_train.dtypes != float)[0] 

train_pool = Pool(X_train,y_train,cat_features=cate_features_index) 

test_pool = Pool(X_test,y_test,cat_features=cate_features_index)



model = CatBoostClassifier(verbose=False, iterations=250, loss_function= 'MultiClass', eval_metric='MultiClass',use_best_model=True, random_seed=42, 

                           learning_rate=0.2, has_time=False, depth=6, l2_leaf_reg=5, random_strength=1, bagging_temperature=0)



model.fit(train_pool,eval_set=test_pool) 

preds_class = model.predict(X_year) 

wta_sc=accuracy_score(preds_class,y_year)
print('Exact scores accuracy is {:.2%} for ATP and {:.2%} for WTA matches.'.format(atp_sc,wta_sc))