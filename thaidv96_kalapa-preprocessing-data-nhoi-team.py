#import dữ liệu 

import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

from collections import Counter



%matplotlib inline
train_df = pd.read_csv('../input/kalapacreditscoring/train.csv')

test_df = pd.read_csv('../input/kalapacreditscoring/test.csv')



total_df = pd.concat([train_df, test_df])
#xử lý biến tuổi

#Vi du lieu tuoi source2 nhieu hon source1 nen fill tu source 1 sang source2

#Cac gia tri duoi 18 cho ve bang nan

total_df['age']=total_df['age_source2'].mask(pd.isnull, total_df['age_source1'])

#total_df.age[total_df.age<18] = np.nan

total_df.drop(["age_source1", "age_source2"], axis = 1, inplace = True)



# FIELD 7:



total_df['FIELD_7_ISNA'] = total_df.FIELD_7.isna()

total_df.FIELD_7.fillna('[]', inplace=True)

total_df.FIELD_7 = total_df.FIELD_7.map(eval)



# # Tạo cột đếm số lượng phần tử

total_df['FIELD_7_len'] = total_df.FIELD_7.str.len()



total_df.loc[total_df.FIELD_7_ISNA,'FIELD_7_len'] = np.nan



total_df.drop("FIELD_7_ISNA",axis=1,inplace=True)

#total_df.drop("FIELD_7", axis=1, inplace=True)
#xử lý biến "province"

total_df.replace('Tỉnh Hòa Bình', 'Tỉnh Hoà Bình', inplace=True)

total_df.replace('Tỉnh Vĩnh phúc', 'Tỉnh Vĩnh Phúc', inplace=True)



# xử lý biến maCv bị lẫn dữ liệu số

total_df.maCv.replace(['0','7','2253','15097','135','147','2853','2983','4999','5339','8059','10958','1006481','1006694','1020883','1020096','17169','275744'],np.nan,inplace = True)



#xử lý FIELD_12 bị lẫn dữ liệu chữ

total_df.FIELD_12.replace(['DN','GD','TN','HT','DN','XK','DT','DK'],np.nan,inplace = True)



# Xử lý FIELD_9 bị lẫn dữ liệu số 

total_df.FIELD_9.replace(['75','74','80','86','79'],np.nan,inplace = True)



# Xử lý FIELD_13 bị lẫn dữ liệu số 

total_df.FIELD_13.replace(['0','4','8'],np.nan,inplace = True)



# Xử lý FIELD_39 bị lẫn dữ liệu số 

total_df.FIELD_39.replace(['1'],np.nan,inplace = True)



# Xử lý FIELD_40 bị lẫn dữ liệu loại khác

total_df.FIELD_40.replace(['02 05 08 11','05 08 11 02','08 02'],np.nan,inplace = True)



# Xử lý FIELD_43 bị lẫn dữ liệu số 

total_df.FIELD_43.replace(['0','5'],np.nan,inplace = True)



# đồng nhất dữ liệu true, false

map_true = ['True', 'TRUE']

total_df.replace(map_true, True, inplace = True)

map_false = ['False', 'FALSE']

total_df.replace(map_false, False, inplace = True)



# đồng nhất dữ liệu đưa NaN, na, nan về NaN

total_df.replace('NaN', np.nan,inplace = True)

total_df.replace('nan', np.nan, inplace=True)

total_df.replace('na', np.nan, inplace=True)



total_df.replace('None',-1, inplace=True)



total_df.FIELD_3.replace(-1,np.nan, inplace = True)

total_df['FIELD_3_365'] = (total_df.FIELD_3/365).round(0)

total_df['FIELD_3_RESIDUAL'] = total_df.FIELD_3_365*365 - total_df.FIELD_3

total_df[['FIELD_3', 'FIELD_3_365', 'FIELD_3_RESIDUAL']]



# Trong tài chính, người ta quan tâm tới số ngày quá hạn hơn là số năm vay.

total_df.drop('FIELD_3_365', axis=1, inplace = True)
train_df= total_df[total_df['id'] < 30000]

test_df = total_df[total_df['id'] >= 30000].drop(columns=['label'])
train_df.to_csv('train_to_randomForest.csv', index=False)

test_df.to_csv('test_to_randomForest.csv', index=False)