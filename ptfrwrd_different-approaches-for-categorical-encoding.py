import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

sns.set_style('whitegrid')



import warnings

warnings.simplefilter(action='ignore')



# use this code to show all columns

pd.set_option('display.max_columns', None)

pd.set_option('display.max_rows', None)
train = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')

test = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv')



train_test = pd.concat((train, test), sort=False).reset_index(drop=True)

train_test.drop(['Id'], axis=1, inplace=True)



train_test.head()
print('Number of categorial types: ', len(train.select_dtypes(include='object').columns))

print('Categorial types: ', train.select_dtypes(include='object').columns)
# see different categorical encoders



from category_encoders.ordinal import OrdinalEncoder

from category_encoders.woe import WOEEncoder

from category_encoders.target_encoder import TargetEncoder

from category_encoders.sum_coding import SumEncoder

from category_encoders.m_estimate import MEstimateEncoder

from category_encoders.leave_one_out import LeaveOneOutEncoder

from category_encoders.helmert import HelmertEncoder

from category_encoders.cat_boost import CatBoostEncoder

from category_encoders.james_stein import JamesSteinEncoder

from category_encoders.one_hot import OneHotEncoder
%%time

encoder = OrdinalEncoder()

ordinal_encoder_example  = encoder.fit_transform(train_test['LotConfig'])
# see  data

ordinal_encoder_example['Original_data'] = train_test['LotConfig']

ordinal_encoder_example = ordinal_encoder_example.rename(columns={'LotConfig': 'Ordinal_data'})

ordinal_encoder_example.head()
%%time

TE_encoder = TargetEncoder()

train_te = TE_encoder.fit_transform(train['LotShape'], train['SalePrice'])

test_te = TE_encoder.transform(test['LotShape'])
# see for train data

target_encoder_example = train_te.rename(columns={'LotShape': 'Target_encoder_data'})

target_encoder_example['Original_data'] = train['LotShape']

target_encoder_example.head()
%%time

SE_encoder = SumEncoder('GarageType')

train_se = SE_encoder.fit_transform(train['GarageType'], train['SalePrice'])

test_se = SE_encoder.transform(test['GarageType'])
# see for train data

sum_encoder_example = train_se.rename(columns={'GarageType': 'Sum_encoder_data'})

sum_encoder_example['Original_data'] = train['GarageType']

sum_encoder_example.head()
%%time

MEE_encoder = MEstimateEncoder()

train_mee = MEE_encoder.fit_transform(train['KitchenQual'], train['SalePrice'])

test_mee = MEE_encoder.transform(train_test['KitchenQual'])
# see for train data

me_encoder_example = train_mee.rename(columns={'KitchenQual': 'ME_encoder_data'})

me_encoder_example['Original_data'] = train['KitchenQual']

me_encoder_example.head()
%%time

LOOE_encoder = LeaveOneOutEncoder()

train_looe = LOOE_encoder.fit_transform(train['GarageFinish'], train['SalePrice'])

test_looe = LOOE_encoder.transform(test['GarageFinish'])
# see for train data

loo_encoder_example = train_looe.rename(columns={'GarageFinish': 'LOO_encoder_data'})

loo_encoder_example['Original_data'] = train['GarageFinish']

loo_encoder_example.head()
%%time

HE_encoder = HelmertEncoder('Foundation')

train_he = HE_encoder.fit_transform(train['Foundation'], train['SalePrice'])

test_he = HE_encoder.transform(test['Foundation'])
# see for train data

he_encoder_example = train_he.rename(columns={'Foundation': 'HE_encoder_data'})

he_encoder_example['Original_data'] = train['Foundation']

he_encoder_example.head()
%%time

CB_encoder = CatBoostEncoder()

train_cb = CB_encoder.fit_transform(train['Neighborhood'], train['SalePrice'])

test_cb = CB_encoder.transform(test['Neighborhood'])
# see for train data

cb_encoder_example = train_cb.rename(columns={'Neighborhood': 'CB_encoder_data'})

cb_encoder_example['Original_data'] = train['Neighborhood']

cb_encoder_example.head()
%%time

JS_encoder = JamesSteinEncoder()

train_js = JS_encoder.fit_transform(train['SaleCondition'], train['SalePrice'])

test_js = JS_encoder.transform(test['SaleCondition'])
# see for train data

js_encoder_example = train_js.rename(columns={'SaleCondition': 'JS_encoder_data'})

js_encoder_example['Original_data'] = train['SaleCondition']

js_encoder_example.head()
%%time

# with  category_encoders, but the most common approach with pandas dummy.

OHE_encoder = OneHotEncoder('RoofStyle')

train_ohe = OHE_encoder.fit_transform(train['RoofStyle'])

test_ohe = OHE_encoder.transform(test['RoofStyle'])
# see for train data

oh_encoder_example = train_ohe.rename(columns={'RoofStyle': 'OH_encoder_data'})

oh_encoder_example['Original_data'] = train['RoofStyle']

oh_encoder_example.head()