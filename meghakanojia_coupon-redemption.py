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
import seaborn as sns
# Viewing training dataset

train_df=pd.read_csv('/kaggle/input/predicting-coupon-redemption/train/train.csv')

train_df
#checking for null values in training dataset

train_df.info()
# Viewing test dataset

test_df=pd.read_csv('/kaggle/input/predicting-coupon-redemption/test.csv')

test_df
# checking for null values in test dataset

test_df.info()
#Customer transaction dataset

cus_tsc_df=pd.read_csv('/kaggle/input/predicting-coupon-redemption/train/customer_transaction_data.csv')

cus_tsc_df
# checking for null values in customer demographics dataset

cus_tsc_df.info()
cus_tsc_df=cus_tsc_df.sample(n=5000)

cus_tsc_df
# selecting features with high importance only

cus_tsc_df= cus_tsc_df.drop(['date','item_id'],axis=1)

cus_tsc_df
# merging train_df and cus_tsc_df

train_tsc_df=pd.merge(train_df,cus_tsc_df,left_on='customer_id',right_on='customer_id')



# merging test_df and cus_tsc_df

test_tsc_df=pd.merge(test_df, cus_tsc_df, left_on='customer_id',right_on='customer_id')

test_tsc_df
train_tsc_df
#Customer demographics dataset

cus_dem_df=pd.read_csv('/kaggle/input/predicting-coupon-redemption/train/customer_demographics.csv')

cus_dem_df
# checking for null values in customer demographics dataset

cus_dem_df.info()
#preprocessing dataset: replacing null values

cus_dem_df['no_of_children'].replace({np.NaN:0},inplace=True)

cus_dem_df['marital_status'].replace({np.NaN:'Unknown'},inplace=True)

# Again, checking for null values after preprocessing customer demographics dataset

cus_dem_df.info()
# Selecting features with high importance only

cus_dem_df= cus_dem_df[['customer_id','rented','income_bracket']]

cus_dem_df
# merging train_tsc_df and cus_dem_df

train_tsc_dem_df=pd.merge(train_tsc_df,cus_dem_df,left_on='customer_id',right_on='customer_id')



# merging test_tsc_df and cus_dem_df

test_tsc_dem_df=pd.merge(test_tsc_df, cus_dem_df, left_on='customer_id',right_on='customer_id')

train_tsc_dem_df
test_tsc_dem_df
#Campaign dataset

camp_df=pd.read_csv('/kaggle/input/predicting-coupon-redemption/train/campaign_data.csv')

camp_df
camp_df.info()
# Selecting features with high importance only

camp_df=camp_df[['campaign_id','campaign_type']]

camp_df
# Merging train_tsc_dem_df with camp_df

train_tsc_dem_camp_df=pd.merge(train_tsc_dem_df,camp_df,left_on='campaign_id',right_on='campaign_id')

test_tsc_dem_camp_df=pd.merge(test_tsc_dem_df,camp_df,left_on='campaign_id',right_on='campaign_id')
train_tsc_dem_camp_df
test_tsc_dem_camp_df
#Coupon item mapping Dataset

cpn_map_df=pd.read_csv('/kaggle/input/predicting-coupon-redemption/train/coupon_item_mapping.csv')

cpn_map_df
cpn_map_df.info()
# Item Dataset

item_df=pd.read_csv('/kaggle/input/predicting-coupon-redemption/train/item_data.csv')

item_df
item_df.info()
# Merging coupon item mapping and item dataset

cpn_item_df=pd.merge(cpn_map_df,item_df,left_on='item_id',right_on='item_id')

cpn_item_df
# Selecting features with high importance only

cpn_item_df= cpn_item_df[['coupon_id','brand']]

cpn_item_df
# Merging train_tsc_dem_camp_df with cpn_item_df

train_tsc_dem_camp_cpn_df = pd.merge(train_tsc_dem_camp_df,cpn_item_df,left_on='coupon_id',right_on='coupon_id')

# Merging test_tsc_dem_camp_df with cpn_item_df

test_tsc_dem_camp_cpn_df = pd.merge(test_tsc_dem_camp_df, cpn_item_df, left_on='coupon_id', right_on='coupon_id')

train_tsc_dem_camp_cpn_df
test_tsc_dem_camp_cpn_df
# Taking a small sample



TRAIN=train_tsc_dem_camp_cpn_df.sample(n=2500000)

TEST=test_tsc_dem_camp_cpn_df.sample(n=2500000)
#One hot encoding

training_data=pd.get_dummies(TRAIN)

testing_data=pd.get_dummies(TEST)

final_train,final_test=training_data.align(testing_data,join='inner',axis=1)
final_train.columns
final_test.columns
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import cross_val_score



X=final_train[['quantity',

       'selling_price', 'other_discount', 'coupon_discount', 'rented',

       'income_bracket', 'brand', 'campaign_type_X', 'campaign_type_Y']]

y=training_data['redemption_status']



model=RandomForestClassifier(n_estimators=10).fit(X,y)



cv_scores=cross_val_score(model,X,y,cv=10)



print('cross validation scores:',cv_scores,'\n','mean of cross validation scores:',np.mean(cv_scores))
df1=pd.DataFrame([[ 'quantity',

       'selling_price', 'other_discount', 'coupon_discount', 'rented',

       'income_bracket', 'brand', 'campaign_type_X', 'campaign_type_Y'],model.feature_importances_])
df1.T
test_data= pd.get_dummies(test_tsc_dem_camp_cpn_df)

test_data.columns
prediction=model.predict(test_data[['quantity',

       'selling_price', 'other_discount', 'coupon_discount', 'rented',

       'income_bracket', 'brand', 'campaign_type_X', 'campaign_type_Y']])
# Coupon Redemption Predictions

prediction
print('Redeem:',sum([1 for i in prediction if i==1]),'\nNot Redeem:',sum([1 for i in prediction if i==0]))