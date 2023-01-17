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

#import housing-pricing data set
DATA_SET="/kaggle/input/house-prices-advanced-regression-techniques/train.csv"
TARGET_COL='SalePrice'

hp_data = pd.read_csv(DATA_SET, index_col=0)
hp_data.head(5)
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler


class HousingDataPreprocessor:
    '''Preproess house-prices dataset
    '''

    def __init__(self,target_col,peason_coef_thresh=0.5):
        '''
        '''
        self.__target_col=target_col
        self.__peason_coef_thresh=peason_coef_thresh

        #will initialize in `fit` method
        self.__imputer=None
        self.__mapping_table_for_col=None
        self.__peason_high_cols=None
        self.__scaler=None


    def fit(self,Xy_data):
        '''
        '''
        features_cols=Xy_data.columns.drop(self.__target_col)
        X_data=Xy_data[features_cols]
        y_data=Xy_data[self.__target_col]
        
        self.__mapping_table_for_col=self.__create_count_mapping_table(X_data)
        X_labeled_data=self.__categorical_data_to_number(X_data)

        self.__imputer=self.__create_fit_imputer_na_data_by_mean(X_labeled_data)
        X_imputed_data=pd.DataFrame(self.__imputer.transform(X_labeled_data.values), columns=X_labeled_data.columns)

        y_data=y_data.reset_index(drop=True)
        Xy_imputed_data = pd.concat([X_imputed_data, y_data], axis=1)

        self.__peason_high_cols=self.__drop_less_pearson_features_cols(Xy_imputed_data).drop(self.__target_col)
        X_high_peason_data=Xy_imputed_data[self.__peason_high_cols]

        self.__scaler=self.__create_fit_standard_scaler(X_high_peason_data)
        

    def transform(self,X_data):
        '''Preprocessing dataset
        '''
        X_labeled_data=self.__categorical_data_to_number(X_data)

        X_imputed_data=pd.DataFrame(self.__imputer.transform(X_labeled_data.values), columns=X_labeled_data.columns)

        X_high_peason_data = X_imputed_data[self.__peason_high_cols]

        X_scaled_data=pd.DataFrame(self.__scaler.transform(X_high_peason_data.values), columns=X_high_peason_data.columns)
        return X_scaled_data
        

    def __create_count_mapping_table(self,Xy_data):
        '''Create mapping table by number of counts
        '''
        s = (Xy_data.dtypes == 'object')
        object_cols = s[s].index

        mapping_table_for_col = {}
        for col in object_cols:
            categorical_values = Xy_data[col].value_counts().index
            mapping_table = { label : index for index, label in enumerate(categorical_values) }
            mapping_table_for_col[col] = mapping_table

        return mapping_table_for_col


    def __create_fit_imputer_na_data_by_mean(self,Xy_data):
        '''Fill N/A data by mean value
        '''
        imputer = Imputer(missing_values='NaN', strategy='mean', axis=0)
        imputer.fit(Xy_data)

        return imputer


    def __create_fit_standard_scaler(self,data):
        '''create and fit standard scaler
        '''
        scaler=StandardScaler()
        scaler.fit(data)
        return scaler


    def __create_fit_min_max_scaler(self,data):
        '''create and fit standard scaler
        '''
        scaler=MinMaxScaler()
        scaler.fit(data)
        return scaler


    def __categorical_data_to_number(self,data):
        '''
        '''
        labeled_data = data.copy()

        s = (labeled_data.dtypes == 'object')
        object_cols = s[s].index

        mapping_table_for_col = {}
        for col in object_cols:
            if col in self.__mapping_table_for_col.keys():
                labeled_data[col] = labeled_data[col].map(self.__mapping_table_for_col[col])

        return labeled_data



    def __drop_less_pearson_features_cols(self,Xy_data):
        '''Remote low correlation features

        Parameters:
            Xy_data: Pandas DataFrame
        '''
        target_col=self.__target_col
        thresh=self.__peason_coef_thresh
        hd_corr = Xy_data.corr(method='pearson')

        hd_high_corr = hd_corr.loc[:,(hd_corr[target_col] > thresh) | (hd_corr[target_col] < (-1*thresh))]
        return hd_high_corr.columns
        
hd_preprocessor=HousingDataPreprocessor(target_col=TARGET_COL)
hd_preprocessor.fit(hp_data)

feature_cols = hp_data.columns.drop(TARGET_COL)
X=hp_data[feature_cols]
y=hp_data[TARGET_COL]

X_preprocessed_hp_data=hd_preprocessor.transform(X)
X_preprocessed_hp_data.head()
from sklearn.model_selection import train_test_split

X_train, X_valid, y_train, y_valid = train_test_split(X_preprocessed_hp_data, y, test_size=0.2, random_state=42)
X_train.head()
#from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

#model=LogisticRegression(C=1, penalty="l1", random_state=7)
#model=DecisionTreeRegressor(random_state=7)
model=RandomForestRegressor(random_state=7)
model.fit(X_train, y_train)
from sklearn.metrics import mean_absolute_error


def predict_and_calc_mae(model, X, y):
    '''
    '''
    pred_y = model.predict(X)
    mae = mean_absolute_error(pred_y, y)
    return mae


mae = predict_and_calc_mae(model, X_valid, y_valid)
print(f'Validation AUC Score : {mae:.4f}')

hd_test = pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/test.csv", index_col=0)
hd_test.head()
X_hd_test_preprocessed=hd_preprocessor.transform(hd_test)
test_preds = model.predict(X_hd_test_preprocessed)

output = pd.DataFrame({'Id': hd_test.index, 'SalePrice': test_preds})
output.to_csv('submission.csv', index=False)# サブミットのための予測評価