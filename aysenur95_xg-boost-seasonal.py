import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import xgboost

import csv as csv

from xgboost import plot_importance

from matplotlib import pyplot

from sklearn.model_selection import cross_val_score,KFold

from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_absolute_error

import matplotlib.pyplot as plt

from sklearn.model_selection import GridSearchCV#Perforing grid search

from scipy.stats import skew

from collections import OrderedDict
#read data

training_features_data = pd.read_csv("../input/flu-shot-learning-h1n1-seasonal-flu-vaccines/training_set_features.csv",

                    sep=',')





training_set_labels = pd.read_csv("../input/flu-shot-learning-h1n1-seasonal-flu-vaccines/training_set_labels.csv",

                    sep=',')





test_features_data = pd.read_csv("../input/flu-shot-learning-h1n1-seasonal-flu-vaccines/test_set_features.csv",

                    sep=',')

#eliminate null values



#for float types

training_features_data=training_features_data.fillna(training_features_data.mean())



#for string types

training_features_data=training_features_data.fillna('out-of-category')
#check no missing values are left 

training_features_data.isna().sum()
#encoding categorical features (str-->float)



from sklearn.preprocessing import OrdinalEncoder

enc = OrdinalEncoder()



enc.fit(training_features_data)

training_features_data_arr=enc.transform(training_features_data)



col_names_list=training_features_data.columns

encoded_categorical_df=pd.DataFrame(training_features_data_arr, columns=col_names_list)
#normalization(make all values bet. 0-1)



from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

scaler.fit(encoded_categorical_df)

normalized_arr=scaler.transform(encoded_categorical_df)



normalized_df=pd.DataFrame(normalized_arr, columns=col_names_list)
#check if data types are correct or not 

normalized_df.info()
#check types of test dataset

test_features_data.info()
#eliminate null values



#for float types

test_features_data=test_features_data.fillna(test_features_data.mean())



#for string types

test_features_data=test_features_data.fillna('out-of-category')
#check no missing values are left 

test_features_data.isna().sum()
#encoding categorical features  (str-->float)



from sklearn.preprocessing import OrdinalEncoder

enc = OrdinalEncoder()

enc.fit(test_features_data)

test_features_data_arr=enc.transform(test_features_data)



col_names_list=test_features_data.columns

test_encoded_categorical_df=pd.DataFrame(test_features_data_arr, columns=col_names_list)
#check data types

test_encoded_categorical_df.info()
#normalization(bet. 0-1)



#using minmax scaler(look up)

test_normalized_arr=scaler.transform(test_encoded_categorical_df)

test_normalized_df=pd.DataFrame(test_normalized_arr, columns=col_names_list)
# split df to X and Y

y = training_set_labels.loc[:, 'seasonal_vaccine'].values  #train_y

X = normalized_df  #train_x

best_xgb_model = xgboost.XGBRegressor(

                 colsample_bytree=0.4,

                 gamma=0,                 

                 learning_rate=0.1,

                 max_depth=5,

                 min_child_weight=1.5,

                 n_estimators=10000,                                                                    

                 reg_alpha=0.75,

                 reg_lambda=0.45,

                 subsample=0.6,

                 seed=42)

best_xgb_model.fit(X,y)
y_pred = best_xgb_model.predict(test_normalized_df)
y_pred[:10]
import numpy as np



np.sum(np.logical_or(np.array(y_pred) > 1, np.array(y_pred) < 0), axis=0)
y_pred = 1/(1+np.exp(-y_pred))

#pred sonuçlarını dosyaya yazdırma



df_pred_seasonal_vaccine_xg=pd.DataFrame(y_pred, columns=['seasonal_vaccine'])

df_pred_seasonal_vaccine_xg["respondent_id"] = df_pred_seasonal_vaccine_xg.index



df_pred_seasonal_vaccine_xg=df_pred_seasonal_vaccine_xg[['respondent_id', 'seasonal_vaccine']]



df_pred_seasonal_vaccine_xg.to_csv('/kaggle/working/df_seasonal_xg_son.csv', columns=['respondent_id', 'seasonal_vaccine'], 

                            index=False, sep=',')
df_pred_seasonal_vaccine_xg.head()
df_pred_h1n1_xg = pd.read_csv("../input/h1n1-xgboost/df_h1n1_xg_son.csv",

                    sep=',')



df_final = df_pred_h1n1_xg.merge(df_pred_seasonal_vaccine_xg, on="respondent_id", how = 'inner')
df_final['respondent_id'] = df_final['respondent_id'].astype(int) + 26707
df_final.head()
#pred sonuçlarını dosyaya yazdırma



#df_final=df[['respondent_id', 'h1n1_vaccine', 'seasonal_vaccine' ]]



df_final.to_csv('/kaggle/working/df_xgboost.csv', columns=['respondent_id', 'h1n1_vaccine', 'seasonal_vaccine' ], 

                            index=False, sep=',')