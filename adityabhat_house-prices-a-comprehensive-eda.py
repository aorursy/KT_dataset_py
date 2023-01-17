# Import Libraries

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import altair as alt

import pandas as pd



# Set Options

#pd.set_option('display.max_rows', 500)

pd.set_option('display.max_columns', 500)

#pd.set_option('display.width', 1000)
# List Files in the directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Read Data

train = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')

test = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv')        
alt.Chart(train).mark_bar(opacity=0.7).encode(

    alt.X('SalePrice', bin=True),

    alt.Y('count()')

)
train.Utilities.isnull().any()
alt.Chart(train[['Utilities','SalePrice']]).mark_bar().encode(

    y=alt.X('Utilities:N', sort='-x'),

    x=alt.Y('count(SalePrice):Q'),

    tooltip='count(SalePrice):Q'

)
C1 = alt.Chart(train[['Heating','SalePrice']]).mark_boxplot().encode(y='Heating:N', x='SalePrice').properties(width=400,height=200)

C2 = alt.Chart(train[['Heating','SalePrice']]).mark_bar().encode(x='count(SalePrice):Q',y=alt.Y('Heating:N', sort='-x'),tooltip='count(SalePrice):Q')

C1 | C2
C1 = alt.Chart(train[['HeatingQC','SalePrice']]).mark_boxplot().encode(y='HeatingQC:N', x='SalePrice').properties(width=400,height=200)

C2 = alt.Chart(train[['HeatingQC','SalePrice']]).mark_bar().encode(x='count(SalePrice):Q',y=alt.Y('HeatingQC:N', sort='-x'),tooltip='count(SalePrice):Q')

C1 | C2
C1 = alt.Chart(train[['Electrical','SalePrice']]).mark_boxplot().encode(y='Electrical:O', x='SalePrice').properties(width=400,height=200)

C2 = alt.Chart(train[['Electrical','SalePrice']]).mark_bar().encode(x='count(SalePrice):Q',y=alt.Y('Electrical:O', sort='-x'),tooltip='count(SalePrice):Q')

C1 | C2
C1 = alt.Chart(train[['Fireplaces','SalePrice']]).mark_boxplot().encode(y='Fireplaces:O', x='SalePrice').properties(width=400,height=200)

C2 = alt.Chart(train[['Fireplaces','SalePrice']]).mark_bar().encode(x='count(SalePrice):Q',y=alt.Y('Fireplaces:O', sort='-x'),tooltip='count(SalePrice):Q')

C1 | C2
C1 = alt.Chart(train[['FireplaceQu','SalePrice']]).mark_boxplot().encode(y='FireplaceQu', x='SalePrice').properties(width=400,height=200)

C2 = alt.Chart(train[['FireplaceQu','SalePrice']]).mark_bar().encode(x='count(SalePrice):Q',y=alt.Y('FireplaceQu:O', sort='-x'),tooltip='count(SalePrice):Q')

C1 | C2
C1 = alt.Chart(train[['PoolQC','SalePrice']]).mark_boxplot().encode(y='PoolQC', x='SalePrice').properties(width=400,height=200)

C2 = alt.Chart(train[['PoolQC','SalePrice']]).mark_bar().encode(x='count(SalePrice):Q',y=alt.Y('PoolQC:O', sort='-x'),tooltip='count(SalePrice):Q')

C1 | C2
C1=alt.Chart(train[['Fence','SalePrice']]).mark_boxplot().encode(y='Fence', x='SalePrice').properties(width=400,height=200)

C2=alt.Chart(train[['Fence','SalePrice']]).mark_bar().encode(x='count(SalePrice):Q',y=alt.Y('Fence:O', sort='-x'),tooltip='count(SalePrice):Q')

C1 | C2
C1=alt.Chart(train[['CentralAir','SalePrice']]).mark_boxplot().encode(y='CentralAir', x='SalePrice').properties(width=400,height=200)

C2=alt.Chart(train[['CentralAir','SalePrice']]).mark_bar().encode(x='count(SalePrice):Q',y=alt.Y('CentralAir:O', sort='-x'),tooltip='count(SalePrice):Q')

C1 | C2
C1=alt.Chart(train[['Fireplaces','SalePrice']]).mark_boxplot().encode(y='Fireplaces:O', x='SalePrice:Q').properties(width=400,height=200)

C2=alt.Chart(train[['Fireplaces','SalePrice']]).mark_bar().encode(x='count(SalePrice):Q',y=alt.Y('Fireplaces:O', sort='-x'),tooltip='count(SalePrice):Q')

C1 | C2
C1=alt.Chart(train[['PoolQC','SalePrice']]).mark_boxplot().encode(y='PoolQC:O', x='SalePrice:Q').properties(width=400,height=200)

C2=alt.Chart(train[['PoolQC','SalePrice']]).mark_bar().encode(x='count(SalePrice):Q',y=alt.Y('PoolQC:O', sort='-x'),tooltip='count(SalePrice):Q')

C1 | C2


C1=alt.Chart(train[['Fence','SalePrice']]).mark_boxplot().encode(y='Fence:O', x='SalePrice:Q').properties(width=400,height=200)

C2=alt.Chart(train[['Fence','SalePrice']]).mark_bar().encode(x='count(SalePrice):Q',y=alt.Y('Fence:O', sort='-x'),tooltip='count(SalePrice):Q')

C1 | C2
train.LotArea.isnull().any()
chart = alt.Chart(train[['LotArea','SalePrice']]).mark_point(opacity=0.5).encode(

            x=alt.X('LotArea'),

            y=alt.Y('SalePrice')    

)



chart + chart.transform_regression('LotArea', 'SalePrice',method="linear").mark_line(color="red")
chart = alt.Chart(train[['MasVnrArea','SalePrice']]).mark_point(opacity=0.5).encode(

            x=alt.X('MasVnrArea'),

            y=alt.Y('SalePrice')    

)



chart + chart.transform_regression('MasVnrArea', 'SalePrice',method="linear").mark_line(color="red")
train.BsmtFinSF1.isnull().any()
chart = alt.Chart(train[['BsmtFinSF1','SalePrice']]).mark_point(opacity=0.5).encode(

            x=alt.X('BsmtFinSF1'),

            y=alt.Y('SalePrice')    

)



chart + chart.transform_regression('BsmtFinSF1', 'SalePrice',method="linear").mark_line(color="red")
train.BsmtFinSF2.isnull().any()
chart = alt.Chart(train[['BsmtFinSF2','SalePrice']]).mark_point(opacity=0.5).encode(

            x=alt.X('BsmtFinSF2'),

            y=alt.Y('SalePrice')    

)



chart + chart.transform_regression('BsmtFinSF2', 'SalePrice',method="linear").mark_line(color="red")
chart = alt.Chart(train[['BsmtUnfSF','SalePrice']]).mark_point(opacity=0.5).encode(

            x=alt.X('BsmtUnfSF'),

            y=alt.Y('SalePrice')    

)



chart + chart.transform_regression('BsmtUnfSF', 'SalePrice',method="linear").mark_line(color="red")
### 