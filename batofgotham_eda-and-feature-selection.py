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
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import scipy.stats #For Chi-square Test
input_path = '/kaggle/input/porto-seguro-safe-driver-prediction/'

df = pd.read_csv(input_path+"train.csv")

df.shape
df.info()
df['target'].unique()
count = {'count': df['target'].value_counts()}

target_info_df = pd.DataFrame(count)
target_info_df['Percent'] = (target_info_df/df.shape[0])*100
target_info_df
target = df['target']
df.drop(columns=['target'],inplace=True)
df_metedata = pd.DataFrame({'DTypes':df.dtypes})
for col in df.columns:

    if '_cat' in col or '_bin' in col:

        df_metedata.loc[col,'DTypes'] = 'Categorical'

    elif df[col].dtype == 'int64':

        df_metedata.loc[col,'DTypes'] = 'int64'

    elif df[col].dtype == 'float64':

        df_metedata.loc[col,'DTypes'] = 'float64'
df_metedata['Dropped']=False

df_metedata['Missing'] = np.nan
df_metedata.loc['id','Missing'] = np.nan

df_metedata.loc['id','Dropped'] = True
df.replace(to_replace=-1,value=np.nan,inplace=True)
df.isnull().sum().sort_values(ascending=False)
(411231/df.shape[0])*100
df_metedata.loc['ps_car_03_cat','Dropped'] = True
df.drop(columns=['ps_car_03_cat'],inplace=True)
for col in df.columns:

    if '_cat' in col or '_bin' in col:

        df[col].fillna(int(df[col].mode()[0]),inplace=True)

        df_metedata.loc[col,'Missing'] = int(df[col].mode()[0])

    elif df[col].dtype == 'int64':

        df[col].fillna(int(df[col].mode()[0]),inplace=True)

        df_metedata.loc[col,'Missing'] = int(df[col].mode()[0])

    else:

        df[col].fillna(df[col].mean(),inplace=True)

        df_metedata.loc[col,'Missing'] = df[col].mean()
df.columns
df_minority = df.loc[target == 1].copy()

df_minority['target'] = target.loc[target == 1].copy()
df_majority = df.loc[target == 0].copy()

df_majority['target'] = target.loc[target == 0].copy()
df_majority.shape, df_minority.shape
splitted_frame = np.array_split(df_majority, 20)
def chi2_test(col):

    count = 0

    splitted_frames = splitted_frame

    for frame in splitted_frames:

        chunks = [frame,df_minority]

        df_test = pd.concat(chunks)

        crosstab_col = pd.crosstab(df_test[col],df_test['target'])

        pValue = scipy.stats.chi2_contingency(crosstab_col)[1]

        if pValue < 0.05:

            count = count + 1

    if count >= 10:

        print('Consider this feature')

    else:

        print("Don't consider this feature")
def anova_test(col):

    count = 0

    splitted_frames = splitted_frame

    for frame in splitted_frames:

        chunks = [frame,df_minority]

        df_test = pd.concat(chunks)

        pValue = scipy.stats.f_oneway(df_test[col],df_test['target'])[1]

        if pValue < 0.05:

            count = count + 1

    if count >= 10:

        print('Consider this feature')

    else:

        print("Don't consider this feature")
df['ps_ind_01'].nunique()
df_metedata.loc['ps_ind_01','DTypes'] = 'Ordinal'
crosstab_ps_ind_01 = pd.crosstab(df['ps_ind_01'],target)

crosstab_ps_ind_01
chi2_test('ps_ind_01')
df['ps_ind_02_cat'].nunique()
df_metedata.loc['ps_ind_02_cat','DTypes']
crosstab_ps_ind_02_cat = pd.crosstab(df['ps_ind_02_cat'],target)

crosstab_ps_ind_02_cat
chi2_test('ps_ind_02_cat')
df['ps_ind_03'].nunique()
df_metedata.loc['ps_ind_03','DTypes']
crosstab_ps_ind_03 = pd.crosstab(df['ps_ind_03'],target)

crosstab_ps_ind_03
chi2_test('ps_ind_03')
df['ps_ind_04_cat'].nunique()
crosstab_ps_ind_04_cat = pd.crosstab(df['ps_ind_04_cat'],target)

crosstab_ps_ind_04_cat
chi2_test('ps_ind_04_cat')
df['ps_ind_05_cat'].nunique()
crosstab_ps_ind_05_cat = pd.crosstab(df['ps_ind_05_cat'],target)

crosstab_ps_ind_05_cat
chi2_test('ps_ind_05_cat')
df['ps_ind_06_bin'].nunique()
crosstab_ps_ind_06_bin = pd.crosstab(df['ps_ind_06_bin'],target)

crosstab_ps_ind_06_bin
chi2_test('ps_ind_06_bin')
df['ps_ind_07_bin'].nunique()
crosstab_ps_ind_07_bin = pd.crosstab(df['ps_ind_07_bin'],target)

crosstab_ps_ind_07_bin
chi2_test('ps_ind_07_bin')
df['ps_ind_08_bin'].nunique()
crosstab_ps_ind_08_bin = pd.crosstab(df['ps_ind_08_bin'],target)

crosstab_ps_ind_08_bin
chi2_test('ps_ind_08_bin')
df['ps_ind_09_bin'].nunique()
crosstab_ps_ind_09_bin = pd.crosstab(df['ps_ind_09_bin'],target)

crosstab_ps_ind_09_bin
chi2_test('ps_ind_09_bin')
df['ps_ind_10_bin'].nunique()
crosstab_ps_ind_10_bin = pd.crosstab(df['ps_ind_10_bin'],target)

crosstab_ps_ind_10_bin
chi2_test('ps_ind_10_bin')
df_metedata.loc['ps_ind_10_bin','Dropped'] = True
df['ps_ind_11_bin'].nunique()
crosstab_ps_ind_11_bin = pd.crosstab(df['ps_ind_11_bin'],target)

crosstab_ps_ind_11_bin
chi2_test('ps_ind_11_bin')
df_metedata.loc['ps_ind_11_bin','Dropped'] = True
df['ps_ind_12_bin'].nunique()
crosstab_ps_ind_12_bin = pd.crosstab(df['ps_ind_12_bin'],target)

crosstab_ps_ind_12_bin
chi2_test('ps_ind_12_bin')
df['ps_ind_13_bin'].nunique()
crosstab_ps_ind_13_bin = pd.crosstab(df['ps_ind_13_bin'],target)

crosstab_ps_ind_13_bin
chi2_test('ps_ind_13_bin')
df_metedata.loc['ps_ind_13_bin','Dropped'] = True
df['ps_ind_16_bin'].nunique()
crosstab_ps_ind_16_bin = pd.crosstab(df['ps_ind_16_bin'],target)

crosstab_ps_ind_16_bin
chi2_test('ps_ind_16_bin')
df['ps_ind_17_bin'].nunique()
crosstab_ps_ind_17_bin = pd.crosstab(df['ps_ind_17_bin'],target)

crosstab_ps_ind_17_bin
chi2_test('ps_ind_17_bin')
df['ps_ind_18_bin'].nunique()
crosstab_ps_ind_18_bin = pd.crosstab(df['ps_ind_18_bin'],target)

crosstab_ps_ind_18_bin
chi2_test('ps_ind_18_bin')
df['ps_car_01_cat'].nunique()
crosstab_ps_car_01_cat = pd.crosstab(df['ps_car_01_cat'],target)

crosstab_ps_car_01_cat
chi2_test('ps_car_01_cat')
df['ps_car_02_cat'].nunique()
crosstab_ps_car_02_cat = pd.crosstab(df['ps_car_02_cat'],target)

crosstab_ps_car_02_cat
chi2_test('ps_car_02_cat')
df['ps_car_04_cat'].nunique()
crosstab_ps_car_04_cat = pd.crosstab(df['ps_car_04_cat'],target)

crosstab_ps_car_04_cat
chi2_test('ps_car_04_cat')
df['ps_car_05_cat'].nunique()
crosstab_ps_car_05_cat = pd.crosstab(df['ps_car_05_cat'],target)

crosstab_ps_car_05_cat
chi2_test('ps_car_05_cat')
df['ps_car_06_cat'].nunique()
crosstab_ps_car_06_cat = pd.crosstab(df['ps_car_06_cat'],target)

crosstab_ps_car_06_cat
chi2_test('ps_car_06_cat')
df['ps_car_07_cat'].nunique()
crosstab_ps_car_07_cat = pd.crosstab(df['ps_car_07_cat'],target)

crosstab_ps_car_07_cat
chi2_test('ps_car_07_cat')
df['ps_car_08_cat'].nunique()
crosstab_ps_car_08_cat = pd.crosstab(df['ps_car_08_cat'],target)

crosstab_ps_car_08_cat
chi2_test('ps_car_08_cat')
df['ps_car_09_cat'].nunique()
crosstab_ps_car_09_cat = pd.crosstab(df['ps_car_09_cat'],target)

crosstab_ps_car_09_cat
chi2_test('ps_car_09_cat')
df['ps_car_10_cat'].nunique()
crosstab_ps_car_10_cat = pd.crosstab(df['ps_car_10_cat'],target)

crosstab_ps_car_10_cat
chi2_test('ps_car_10_cat')
df_metedata.loc['ps_car_10_cat','Dropped'] = True
df['ps_car_11_cat'].nunique()
crosstab_ps_car_11_cat = pd.crosstab(df['ps_car_11_cat'],target)

crosstab_ps_car_11_cat
chi2_test('ps_car_11_cat')
df['ps_calc_15_bin'].nunique()
crosstab_ps_calc_15_bin = pd.crosstab(df['ps_calc_15_bin'],target)

crosstab_ps_calc_15_bin
chi2_test('ps_calc_15_bin')
df_metedata.loc['ps_calc_15_bin','Dropped']= True
df['ps_calc_16_bin'].nunique()
crosstab_ps_calc_16_bin = pd.crosstab(df['ps_calc_16_bin'],target)

crosstab_ps_calc_16_bin
chi2_test('ps_calc_16_bin')
df_metedata.loc['ps_calc_16_bin','Dropped'] = True
df['ps_calc_17_bin'].nunique()
crosstab_ps_calc_17_bin = pd.crosstab(df['ps_calc_17_bin'],target)

crosstab_ps_calc_17_bin
chi2_test('ps_calc_17_bin')
df_metedata.loc['ps_calc_17_bin','Dropped'] = True
df['ps_calc_18_bin'].nunique()
crosstab_ps_calc_18_bin = pd.crosstab(df['ps_calc_18_bin'],target)

crosstab_ps_calc_18_bin
chi2_test('ps_calc_18_bin')
df_metedata.loc['ps_calc_18_bin','Dropped'] = True
df['ps_calc_19_bin'].nunique()
crosstab_ps_calc_19_bin = pd.crosstab(df['ps_calc_19_bin'],target)

crosstab_ps_calc_19_bin
chi2_test('ps_calc_19_bin')
df_metedata.loc['ps_calc_19_bin','Dropped'] = True
df['ps_calc_20_bin'].nunique()
crosstab_ps_calc_20_bin = pd.crosstab(df['ps_calc_20_bin'],target)

crosstab_ps_calc_20_bin
chi2_test('ps_calc_20_bin')
df_metedata.loc['ps_calc_20_bin','Dropped'] = True
df['ps_ind_14'].nunique()
df_metedata.loc['ps_ind_14','DTypes']
df_metedata.loc['ps_ind_14','DTypes'] = 'Ordinal'
crosstab_ps_ind_14 = pd.crosstab(df['ps_ind_14'],target)

crosstab_ps_ind_14
chi2_test('ps_ind_14')
df['ps_ind_15'].nunique()
df_metedata.loc['ps_ind_15','DTypes'] = 'Ordinal'
crosstab_ps_ind_15 = pd.crosstab(df['ps_ind_15'],target)

crosstab_ps_ind_15
chi2_test('ps_ind_15')
df['ps_reg_01'].nunique()
df_metedata.loc['ps_reg_01','DTypes']
df['ps_reg_01']
df_metedata.loc['ps_reg_01','DTypes'] = 'Ordinal'
crosstab_ps_reg_01 = pd.crosstab(df['ps_reg_01'],target)

crosstab_ps_reg_01
chi2_test('ps_reg_01')
df['ps_reg_02'].nunique()
df_metedata.loc['ps_reg_02','DTypes']
df['ps_reg_02'].value_counts()
df_metedata.loc['ps_reg_02','DTypes'] = 'Ordinal'
crosstab_ps_reg_02 = pd.crosstab(df['ps_reg_02'],target)

crosstab_ps_reg_02
chi2_test('ps_reg_02')
df['ps_reg_03'].nunique()
df_metedata.loc['ps_reg_03','DTypes']
df['ps_reg_03'].value_counts()
df['ps_reg_03'].max()
df['ps_reg_03'].min()
fig,ax = plt.subplots(2,1,figsize=(14,8))

ax1,ax2 = ax.flatten()

sns.set_style("whitegrid")

sns.distplot(df['ps_reg_03'],ax=ax1)

sns.boxplot(x=target,y=df['ps_reg_03'],showmeans=True,ax=ax2)
anova_test('ps_reg_03')
df['ps_car_11'].nunique()
df['ps_car_11'].dtype
df_metedata.loc['ps_car_11','DTypes']
df_metedata.loc['ps_car_11','DTypes'] = 'Ordinal'
crosstab_ps_car_11 = pd.crosstab(df['ps_car_11'],target)

crosstab_ps_car_11
chi2_test('ps_car_11')
df['ps_car_12'].nunique()
df_metedata.loc['ps_car_12','DTypes']
df['ps_car_12'].value_counts()
fig,ax = plt.subplots(2,1,figsize=(14,8))

ax1,ax2 = ax.flatten()

sns.set_style("whitegrid")

sns.distplot(df['ps_car_12'],ax=ax1)

sns.boxplot(x=target,y=df['ps_car_12'],showmeans=True,ax=ax2)
df['ps_car_12'].max()
df['ps_car_12'].min()
anova_test('ps_car_12')
df['ps_car_13'].nunique()
df_metedata.loc['ps_car_13','DTypes']
df['ps_car_13'].value_counts()
fig,ax = plt.subplots(2,1,figsize=(20,10))

ax1,ax2 = ax.flatten()

sns.distplot(df['ps_car_13'],ax=ax1)

sns.boxplot(x=target,y=df['ps_car_13'],showmeans=True,ax=ax2)
df['ps_car_13'].max()
anova_test('ps_car_13')
df['ps_car_14'].nunique()
df_metedata.loc['ps_car_14','DTypes']
df['ps_car_14'].value_counts()
fig,ax = plt.subplots(2,1,figsize=(20,10))

ax1,ax2 = ax.flatten()

sns.distplot(df['ps_car_14'],ax=ax1)

sns.boxplot(x=target,y=df['ps_car_14'],showmeans=True,ax=ax2)
df['ps_car_14'].max()
df['ps_car_14'].min()
anova_test('ps_car_14')
df['ps_car_15'].nunique()
df_metedata.loc['ps_car_15','DTypes']
df['ps_car_15'].value_counts()
df_metedata.loc['ps_car_15','DTypes'] = 'Ordinal'
crosstab_ps_car_15 = pd.crosstab(df['ps_car_15'],target)

crosstab_ps_car_15
chi2_test('ps_car_15')
df['ps_calc_01'].nunique()
df_metedata.loc['ps_calc_01','DTypes']
df['ps_calc_01'].value_counts()
df_metedata.loc['ps_calc_01','DTypes'] = 'Ordinal'
crosstab_ps_calc_01 = pd.crosstab(df['ps_calc_01'],target)

crosstab_ps_calc_01
chi2_test('ps_calc_01')
df_metedata.loc['ps_calc_01','Dropped'] =True
df['ps_calc_02'].nunique()
df_metedata.loc['ps_calc_02','DTypes']
df['ps_calc_02'].value_counts()
df_metedata.loc['ps_calc_02','DTypes'] = 'Ordinal'
crosstab_ps_calc_02 = pd.crosstab(df['ps_calc_02'],target)

crosstab_ps_calc_02
chi2_test('ps_calc_02')
df_metedata.loc['ps_calc_02','Dropped'] =True
df['ps_calc_03'].nunique()
df_metedata.loc['ps_calc_03','DTypes']
df['ps_calc_03'].value_counts()
df_metedata.loc['ps_calc_03','DTypes'] = 'Ordinal'
crosstab_ps_calc_03 = pd.crosstab(df['ps_calc_03'],target)

crosstab_ps_calc_03
chi2_test('ps_calc_03')
df_metedata.loc['ps_calc_03','Dropped'] =True
df['ps_calc_04'].nunique()
df_metedata.loc['ps_calc_04','DTypes']
df['ps_calc_04'].value_counts()
df_metedata.loc['ps_calc_04','DTypes'] = 'Ordinal'
crosstab_ps_calc_04 = pd.crosstab(df['ps_calc_04'],target)

crosstab_ps_calc_04
chi2_test('ps_calc_04')
df_metedata.loc['ps_calc_04','Dropped'] =True
df['ps_calc_05'].nunique()
df_metedata.loc['ps_calc_05','DTypes']
df['ps_calc_05'].value_counts()
df_metedata.loc['ps_calc_05','DTypes'] = 'Ordinal'
crosstab_ps_calc_05 = pd.crosstab(df['ps_calc_05'],target)

crosstab_ps_calc_05
chi2_test('ps_calc_05')
df_metedata.loc['ps_calc_05','Dropped'] =True
df['ps_calc_06'].nunique()
df_metedata.loc['ps_calc_06','DTypes']
df['ps_calc_06'].value_counts()
df_metedata.loc['ps_calc_06','DTypes'] = 'Ordinal'
crosstab_ps_calc_06 = pd.crosstab(df['ps_calc_06'],target)

crosstab_ps_calc_06
chi2_test('ps_calc_06')
df_metedata.loc['ps_calc_06','Dropped'] =True
df['ps_calc_07'].nunique()
df_metedata.loc['ps_calc_07','DTypes']
df['ps_calc_07'].value_counts()
df_metedata.loc['ps_calc_07','DTypes'] = 'Ordinal'
crosstab_ps_calc_07 = pd.crosstab(df['ps_calc_07'],target)

crosstab_ps_calc_07
chi2_test('ps_calc_07')
df_metedata.loc['ps_calc_07','Dropped'] =True
df['ps_calc_08'].nunique()
df_metedata.loc['ps_calc_08','DTypes']
df['ps_calc_08'].value_counts()
df_metedata.loc['ps_calc_08','DTypes'] = 'Ordinal'
crosstab_ps_calc_08 = pd.crosstab(df['ps_calc_08'],target)

crosstab_ps_calc_08
chi2_test('ps_calc_08')
df_metedata.loc['ps_calc_08','Dropped'] =True
df['ps_calc_09'].nunique()
df_metedata.loc['ps_calc_09','DTypes']
df['ps_calc_09'].value_counts()
df_metedata.loc['ps_calc_09','DTypes'] = 'Ordinal'
crosstab_ps_calc_09 = pd.crosstab(df['ps_calc_09'],target)

crosstab_ps_calc_09
chi2_test('ps_calc_09')
df_metedata.loc['ps_calc_09','Dropped'] =True
df['ps_calc_10'].nunique()
df_metedata.loc['ps_calc_10','DTypes']
df['ps_calc_10'].value_counts()
df_metedata.loc['ps_calc_10','DTypes'] = 'Ordinal'
crosstab_ps_calc_10 = pd.crosstab(df['ps_calc_10'],target)

crosstab_ps_calc_10
chi2_test('ps_calc_10')
df_metedata.loc['ps_calc_10','Dropped'] =True
df['ps_calc_11'].nunique()
df_metedata.loc['ps_calc_11','DTypes']
df['ps_calc_11'].value_counts()
df_metedata.loc['ps_calc_11','DTypes'] = 'Ordinal'
crosstab_ps_calc_11 = pd.crosstab(df['ps_calc_11'],target)

crosstab_ps_calc_11
chi2_test('ps_calc_11')
df_metedata.loc['ps_calc_11','Dropped'] =True
df['ps_calc_12'].nunique()
df_metedata.loc['ps_calc_12','DTypes']
df['ps_calc_12'].value_counts()
df_metedata.loc['ps_calc_12','DTypes'] = 'Ordinal'
crosstab_ps_calc_12 = pd.crosstab(df['ps_calc_12'],target)

crosstab_ps_calc_12
chi2_test('ps_calc_12')
df_metedata.loc['ps_calc_12','Dropped'] =True
df['ps_calc_13'].nunique()
df_metedata.loc['ps_calc_13','DTypes']
df['ps_calc_13'].value_counts()
df_metedata.loc['ps_calc_13','DTypes'] = 'Ordinal'
crosstab_ps_calc_13 = pd.crosstab(df['ps_calc_13'],target)

crosstab_ps_calc_13
chi2_test('ps_calc_13')
df_metedata.loc['ps_calc_13','Dropped'] =True
df['ps_calc_14'].nunique()
df_metedata.loc['ps_calc_14','DTypes']
df['ps_calc_14'].value_counts()
df_metedata.loc['ps_calc_14','DTypes'] = 'Ordinal'
crosstab_ps_calc_14 = pd.crosstab(df['ps_calc_14'],target)

crosstab_ps_calc_14
chi2_test('ps_calc_14')
df_metedata.loc['ps_calc_14','Dropped'] =True
df_metedata
df_new = pd.read_csv(input_path+'train.csv')

target_new = df_new['target']

df_new.drop(columns=['target'],inplace = True)
df_new.replace(to_replace=-1,value=np.nan,inplace=True)
for columns in df_new.columns.values:

    if df_metedata.loc[columns,'Dropped']:

        df_new.drop(columns=[columns],inplace=True)
for col in df_new.columns:

    if ((df_metedata.loc[col,'DTypes'] == 'Categorical') or (df_metedata.loc[col,'DTypes'] == 'Ordinal')):

        df_new[col].fillna(df_new[col].mode()[0],inplace=True)

        df_metedata.loc[col,'Missing'] = df_new[col].mode()[0]

    else:

        df_new[col].fillna(df_new[col].mean(),inplace=True)

        df_metedata.loc[col,'Missing'] = df_new[col].mean()

        
df_new.shape
import pickle

working_path = '/kaggle/working'

pickle.dump(df_metedata,open(working_path+'porto-seguro-safe-driver-prediction_df_metedata_pickle','wb'))