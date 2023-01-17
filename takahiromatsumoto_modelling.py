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
import gc

import warnings

warnings.filterwarnings('ignore')



import numpy as np

import scipy as sp

import pandas as pd

from pandas import DataFrame, Series



import matplotlib.pyplot as plt

plt.style.use('ggplot')

%matplotlib inline



from sklearn.metrics import roc_auc_score

from sklearn.model_selection import StratifiedKFold

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn import preprocessing as pp

from sklearn.preprocessing import MinMaxScaler

from category_encoders import OrdinalEncoder, OneHotEncoder, TargetEncoder

from tqdm import tqdm_notebook as tqdm



from sklearn.ensemble import GradientBoostingClassifier



import lightgbm as lgb

from lightgbm import LGBMClassifier



import eli5

from eli5.sklearn import PermutationImportance
df_train = pd.read_csv('../input/homework-for-students2/train.csv', parse_dates=['issue_d','earliest_cr_line'], index_col=0)

df_test = pd.read_csv('../input/homework-for-students2/test.csv', parse_dates=['issue_d','earliest_cr_line'], index_col=0)
zip = pd.read_csv('../input/homework-for-students2/free-zipcode-database.csv', index_col=0, low_memory=False)
#年収の外れ値を除外

df_train = df_train[df_train['annual_inc'] < df_train['annual_inc'].quantile(0.999)]
#古いデータを除外

df_train = df_train[df_train.issue_d.dt.year >= 2014]
df_train_true = df_train[df_train.loan_condition >= 1]



df_train_false = df_train[df_train.loan_condition <= 0]



f = 'tot_cur_bal'

df_train_true[f]= df_train_true[f].apply(np.log1p)

df_train_false[f]= df_train_false[f].apply(np.log1p)





plt.figure(figsize=[20,7])

df_train_true[f].hist(density=True, alpha=0.5, bins=51)

df_train_false[f].hist(density=True, alpha=0.5, bins=51)

plt.xlabel(f)

plt.ylabel('density')

plt.show()
y_train = df_train.loan_condition

X_train = df_train.drop(['loan_condition'], axis=1)



X_test = df_test
#年収のローン対比を特徴量に追加

X_train['loan/annual_inc']= X_train['loan_amnt']/((X_train['annual_inc'])+1)

X_test['loan/annual_inc']= X_test['loan_amnt']/((X_test['annual_inc'])+1)



X_train['loan/tot_cur_bal'] = X_train['loan_amnt']/((X_train['tot_cur_bal'])+1)

X_test['loan/tot_cur_bal'] = X_test['loan_amnt']/((X_test['tot_cur_bal'])+1)
#issue_dから、年と月を取得

# X_train['year'] = X_train['issue_d'].str[4:8].astype(int)

# X_test['year'] = X_test['issue_d'].str[4:8].astype(int)

# X_train['month'] = X_train['issue_d'].str[0:3].astype(str)

# X_test['month'] = X_test['issue_d'].str[0:3].astype(str)

# X_train.head(20)



X_train['year'] = X_train.issue_d.dt.year

X_test['year'] = X_test.issue_d.dt.year

X_train['month'] = X_train.issue_d.dt.month

X_test['month'] = X_test.issue_d.dt.month



X_train['e_year'] = X_train.earliest_cr_line.dt.year

X_test['e_year'] = X_test.earliest_cr_line.dt.year

X_train['e_month'] = X_train.earliest_cr_line.dt.month

X_test['e_month'] = X_test.earliest_cr_line.dt.month



del X_train['issue_d']

del X_test['issue_d']

del X_train['earliest_cr_line']

del X_test['earliest_cr_line']



X_train.head()

X_test.head()

#Zipcodeごとの緯度・経度を算出

col2 = 'Zipcode'

zip_group = zip.groupby([col2]).mean()

del zip_group['Xaxis']

del zip_group['Yaxis']

del zip_group['Zaxis']

del zip_group['Decommisioned']

del zip_group['TaxReturnsFiled']

del zip_group['EstimatedPopulation']

del zip_group['TotalWages']



zip_group.head()
#Zipcodeごとの緯度・経度をテーブルにjoin

X_train['zip_code']=X_train['zip_code'].str[:3].astype(int)

X_train = pd.merge(X_train, zip_group, left_on='zip_code', right_on='Zipcode',how='left')

del X_train['zip_code']



X_test['zip_code']=X_test['zip_code'].str[:3].astype(int)

X_test = pd.merge(X_test, zip_group, left_on='zip_code', right_on='Zipcode',how='left')

del X_test['zip_code']
#州ごとの一人当たり賃金を算出

col3 = 'State'

state_group = zip.groupby([col3]).sum()

state_group['wpc']=state_group['TotalWages']/state_group['EstimatedPopulation']



del state_group['Xaxis']

del state_group['Yaxis']

del state_group['Zaxis']

del state_group['Decommisioned']

del state_group['TaxReturnsFiled']

del state_group['EstimatedPopulation']

del state_group['TotalWages']

del state_group['Lat']

del state_group['Long']

del state_group['Zipcode']



state_group.head()



#州ごとの一人当たり賃金をテーブルにマージ

X_train = pd.merge(X_train, state_group, left_on='addr_state', right_on='State',how='left')

del X_train['addr_state']

X_train.head()



X_test = pd.merge(X_test, state_group, left_on='addr_state', right_on='State',how='left')

del X_test['addr_state']

X_test.head()
#欠損数

X_train['nan_sum'] = 0

X_test['nan_sum'] = 0

nan_train = X_train.isnull().sum(axis=1) 

nan_test = X_test.isnull().sum(axis=1)

X_train['nan_sum'] = nan_train

X_test['nan_sum'] = nan_test
#欠損値を埋める

#欠損値を埋める

X_train.fillna({'loan_amnt': X_train['loan_amnt'].median(),

                'installment':X_train['installment'].median(),

                'grade':'#',

                'sub_grade':'#',

                'emp_title': '#',

                'emp_length': 'n/a',

                'home_ownership': '#',

                'annual_inc' : 0,

                #'issue_d':X_train['issue_d'].mode(),

                'purpose':'other',

                'title':'Other',

                'zip_code':'000xx',

                'addr_state':'#',

                'dti':X_train['dti'].mean(),

                'delinq_2yrs':X_train['delinq_2yrs'].mean(),

                #'earliest_cr_line':X_train['earliest_cr_line'].mode(),

                'inq_last_6mths':X_train['inq_last_6mths'].mean(),

                'mths_since_last_delinq': 0,

                'mths_since_last_record':0,

                'open_acc':X_train['open_acc'].mean(),

                'pub_rec':0,

                'revol_bal':X_train['revol_bal'].mean(),

                'revol_util':X_train['revol_util'].mean(),

                'total_acc':X_train['total_acc'].mean(),

                'initial_list_status':'#',

                'collections_12_mths_ex_med':X_train['collections_12_mths_ex_med'].mean(),

                'mths_since_last_major_derog':0,

                'application_type':'#',

                'acc_now_delinq':X_train['acc_now_delinq'].mean(),

                'tot_coll_amt':0,

                'tot_cur_bal':0, 

                'Lat': 0,

                'Long': 0,

                'wpc': X_train['wpc'].median(),

                'year':2015,

                'month':7,

                'e_year':X_train['e_year'].median(),

                'e_month':X_train['e_month'].median(),

                'loan/annual_inc':1},               

               inplace = True)



X_test.fillna({'loan_amnt': X_train['loan_amnt'].median(),

                'installment':X_train['installment'].median(),

                'grade':'#',

                'sub_grade':'#',

                'emp_title': '#',

                'emp_length': 'n/a',

                'home_ownership': '#',

                'annual_inc' : 0,

                #'issue_d':X_train['issue_d'].mode(),

                'purpose':'other',

                'title':'Other',

                'zip_code':'000xx',

                'addr_state':'#',

                'dti':X_train['dti'].mean(),

                'delinq_2yrs':X_train['delinq_2yrs'].mean(),

                #'earliest_cr_line':X_train['earliest_cr_line'].mode(),

                'inq_last_6mths':X_train['inq_last_6mths'].mean(),

                'mths_since_last_delinq': 0,

                'mths_since_last_record':0,

                'open_acc':X_train['open_acc'].mean(),

                'pub_rec':0,

                'revol_bal':X_train['revol_bal'].mean(),

                'revol_util':X_train['revol_util'].mean(),

                'total_acc':X_train['total_acc'].mean(),

                'initial_list_status':'#',

                'collections_12_mths_ex_med':X_train['collections_12_mths_ex_med'].mean(),

                'mths_since_last_major_derog':0,

                'application_type':'#',

                'acc_now_delinq':X_train['acc_now_delinq'].mean(),

                'tot_coll_amt':0,

                'tot_cur_bal':0,

                'Lat': 0,

                'Long': 0,

                'wpc': X_train['wpc'].median(),

                'year':2015,

                'month':7,

                'e_year':X_train['e_year'].median(),

                'e_month':X_train['e_month'].median(),

                'loan/annual_inc':1},  

               inplace = True)
#annual_inc以外、yeo-johnson変換

from sklearn.preprocessing import PowerTransformer



# pt1 = PowerTransformer(method='box-cox')

# pt2 = PowerTransformer(method='box-cox')

# pt3 = PowerTransformer(method='box-cox')

# pt4 = PowerTransformer(method='box-cox')

# pt5 = PowerTransformer(method='box-cox')

# pt6 = PowerTransformer(method='box-cox')



pt1 = PowerTransformer(method='yeo-johnson')

pt2 = PowerTransformer(method='yeo-johnson')

pt3 = PowerTransformer(method='yeo-johnson')

pt4 = PowerTransformer(method='yeo-johnson')

pt5 = PowerTransformer(method='yeo-johnson')

# pt6 = PowerTransformer(method='yeo-johnson')



# num_cols = 'loan_amnt'

yj_col = 'loan_amnt'

train_data1 = X_train[yj_col].values.reshape(-1,1)

test_data1 = X_test[yj_col].values.reshape(-1,1)

pt1.fit(train_data1)

X_train[yj_col] = pt1.transform(train_data1)

X_test[yj_col] = pt1.transform(test_data1)



yj_col = 'installment'

train_data2 = X_train[yj_col].values.reshape(-1,1)

test_data2 = X_test[yj_col].values.reshape(-1,1)

pt2.fit(train_data2)

X_train[yj_col] = pt2.transform(train_data2)

X_test[yj_col] = pt2.transform(test_data2)



yj_col = 'dti'

train_data3 = X_train[yj_col].values.reshape(-1,1)

test_data3 = X_test[yj_col].values.reshape(-1,1)

pt3.fit(train_data3)

X_train[yj_col] = pt3.transform(train_data3)

X_test[yj_col] = pt3.transform(test_data3)



yj_col = 'revol_bal'

train_data4 = X_train[yj_col].values.reshape(-1,1)

test_data4 = X_test[yj_col].values.reshape(-1,1)

pt4.fit(train_data4)

X_train[yj_col] = pt4.transform(train_data4)

X_test[yj_col] = pt4.transform(test_data4)



yj_col = 'tot_cur_bal'

train_data5 = X_train[yj_col].values.reshape(-1,1)

test_data5 = X_test[yj_col].values.reshape(-1,1)

pt5.fit(train_data5)

X_train[yj_col] = pt5.transform(train_data5)

X_test[yj_col] = pt5.transform(test_data5)
#年収を対数変換

# X_train['loan_amnt'] = X_train['loan_amnt'].apply(np.log1p)

# X_test['loan_amnt'] = X_test['loan_amnt'].apply(np.log1p)



X_train['annual_inc'] = X_train['annual_inc'].apply(np.log1p)

X_test['annual_inc'] = X_test['annual_inc'].apply(np.log1p)



# X_train['installment'] = X_train['installment'].apply(np.log1p)

# X_test['installment'] = X_test['installment'].apply(np.log1p)



# X_train['dti'] = X_train['dti'].apply(np.log1p)

# X_test['dti'] = X_test['dti'].apply(np.log1p)



# X_train['revol_bal'] = X_train['revol_bal'].apply(np.log1p)

# X_test['revol_bal'] = X_test['revol_bal'].apply(np.log1p)



# X_train['tot_cur_bal'] = X_train['tot_cur_bal'].apply(np.log1p)

# X_test['tot_cur_bal'] = X_test['tot_cur_bal'].apply(np.log1p)
#OrdinalEncoderに必要な書き換えで上書きされるデータをtempに避難

col_grade = X_train.columns.get_loc('grade')

col_sub_grade = X_train.columns.get_loc('sub_grade')

col_years = X_train.columns.get_loc('emp_length')

# col_month = X_train.columns.get_loc('month')



grade_temp = []

sub_grade_temp = []

years_temp = []

# month_temp = []



for i in range(7):     

    if X_train.iloc[i,col_grade] == "A":

        grade_temp.append(1)

    elif X_train.iloc[i,col_grade] == "B":

        grade_temp.append(2)

    elif X_train.iloc[i,col_grade] == "C":

        grade_temp.append(3)

    elif X_train.iloc[i,col_grade] == "D":

        grade_temp.append(4)

    elif X_train.iloc[i,col_grade] == "E":

        grade_temp.append(5)

    elif X_train.iloc[i,col_grade] == "F":

        grade_temp.append(6)

    elif X_train.iloc[i,col_grade] == "G":

        grade_temp.append(7)







for i in range(35):

    if X_train.iloc[i,col_sub_grade] == "A1":

        sub_grade_temp.append(1)

    elif X_train.iloc[i,col_sub_grade] == "A2":

        sub_grade_temp.append(2)

    elif X_train.iloc[i,col_sub_grade] == "A3":

        sub_grade_temp.append(3)

    elif X_train.iloc[i,col_sub_grade] == "A4":

        sub_grade_temp.append(4)

    elif X_train.iloc[i,col_sub_grade] == "A5":

        sub_grade_temp.append(5)

    elif X_train.iloc[i,col_sub_grade] == "B1":

        sub_grade_temp.append(6)

    elif X_train.iloc[i,col_sub_grade] == "B2":

        sub_grade_temp.append(7)

    elif X_train.iloc[i,col_sub_grade] == "B3":

        sub_grade_temp.append(8)

    elif X_train.iloc[i,col_sub_grade] == "B4":

        sub_grade_temp.append(9)

    elif X_train.iloc[i,col_sub_grade] == "B5":

        sub_grade_temp.append(10)

    elif X_train.iloc[i,col_sub_grade] == "C1":

        sub_grade_temp.append(11)

    elif X_train.iloc[i,col_sub_grade] == "C2":

        sub_grade_temp.append(12)

    elif X_train.iloc[i,col_sub_grade] == "C3":

        sub_grade_temp.append(13)

    elif X_train.iloc[i,col_sub_grade] == "C4":

        sub_grade_temp.append(14)

    elif X_train.iloc[i,col_sub_grade] == "C5":

        sub_grade_temp.append(15)

    elif X_train.iloc[i,col_sub_grade] == "D1":

        sub_grade_temp.append(16)

    elif X_train.iloc[i,col_sub_grade] == "D2":

        sub_grade_temp.append(17)       

    elif X_train.iloc[i,col_sub_grade] == "D3":

        sub_grade_temp.append(18)

    elif X_train.iloc[i,col_sub_grade] == "D4":

        sub_grade_temp.append(19)

    elif X_train.iloc[i,col_sub_grade] == "D5":

        sub_grade_temp.append(20)

    elif X_train.iloc[i,col_sub_grade] == "E1":

        sub_grade_temp.append(21)

    elif X_train.iloc[i,col_sub_grade] == "E2":

        sub_grade_temp.append(22)

    elif X_train.iloc[i,col_sub_grade] == "E3":

        sub_grade_temp.append(23)

    elif X_train.iloc[i,col_sub_grade] == "E4":

        sub_grade_temp.append(24)

    elif X_train.iloc[i,col_sub_grade] == "E5":

        sub_grade_temp.append(25)

    elif X_train.iloc[i,col_sub_grade] == "F1":

        sub_grade_temp.append(26)

    elif X_train.iloc[i,col_sub_grade] == "F2":

        sub_grade_temp.append(27)

    elif X_train.iloc[i,col_sub_grade] == "F3":

        sub_grade_temp.append(28)

    elif X_train.iloc[i,col_sub_grade] == "F4":

        sub_grade_temp.append(29)

    elif X_train.iloc[i,col_sub_grade] == "F5":

        sub_grade_temp.append(30)

    elif X_train.iloc[i,col_sub_grade] == "G1":

        sub_grade_temp.append(31)

    elif X_train.iloc[i,col_sub_grade] == "G2":

        sub_grade_temp.append(32)

    elif X_train.iloc[i,col_sub_grade] == "G3":

        sub_grade_temp.append(33)    

    elif X_train.iloc[i,col_sub_grade] == "G4":

        sub_grade_temp.append(34)

    elif X_train.iloc[i,col_sub_grade] == "G5":

        sub_grade_temp.append(2)      

        



for i in range(12):

    if X_train.iloc[i,col_years] == "< 1 year":

        years_temp.append(1) 

    elif X_train.iloc[i,col_years] == "1 year":

        years_temp.append(2) 

    elif X_train.iloc[i,col_years] == "2 years":

        years_temp.append(3) 

    elif X_train.iloc[i,col_years] == "3 years":

        years_temp.append(4) 

    elif X_train.iloc[i,col_years] == "4 years":

        years_temp.append(5) 

    elif X_train.iloc[i,col_years] == "5 years":

        years_temp.append(6) 

    elif X_train.iloc[i,col_years] == "6 years":

        years_temp.append(7) 

    elif X_train.iloc[i,col_years] == "7 years":

        years_temp.append(8) 

    elif X_train.iloc[i,col_years] == "8 years":

        years_temp.append(9) 

    elif X_train.iloc[i,col_years] == "9 years":

        years_temp.append(10) 

    elif X_train.iloc[i,col_years] == "10+ years":

        years_temp.append(11) 

    elif X_train.iloc[i,col_years] == "n/a":

        years_temp.append(12) 



        

# for i in range(12):

#     if X_train.iloc[i,col_month] == "Jan":

#         month_temp.append(1) 

#     elif X_train.iloc[i,col_month] == "Feb":

#         month_temp.append(2) 

#     elif X_train.iloc[i,col_month] == "Mar":

#         month_temp.append(3) 

#     elif X_train.iloc[i,col_month] == "Apr":

#         month_temp.append(4) 

#     elif X_train.iloc[i,col_month] == "May":

#         month_temp.append(5) 

#     elif X_train.iloc[i,col_month] == "Jun":

#         month_temp.append(6) 

#     elif X_train.iloc[i,col_month] == "Jul":

#         month_temp.append(7) 

#     elif X_train.iloc[i,col_month] == "Aug":

#         month_temp.append(8) 

#     elif X_train.iloc[i,col_month] == "Sep":

#         month_temp.append(9) 

#     elif X_train.iloc[i,col_month] == "Oct":

#         month_temp.append(10) 

#     elif X_train.iloc[i,col_month] == "Nov":

#         month_temp.append(11) 

#     elif X_train.iloc[i,col_month] == "Dec":

#         month_temp.append(12) 



print(grade_temp)

print(sub_grade_temp)

print(years_temp)

# print(month_temp)


X_train.iloc[0,col_grade]= "A"

X_train.iloc[1,col_grade]= "B"

X_train.iloc[2,col_grade]= "C"

X_train.iloc[3,col_grade]= "D"

X_train.iloc[4,col_grade]= "E"

X_train.iloc[5,col_grade]= "F"

X_train.iloc[6,col_grade]= "G"







X_train.iloc[0,col_sub_grade]= "A1"

X_train.iloc[1,col_sub_grade]= "A2"

X_train.iloc[2,col_sub_grade]= "A3"

X_train.iloc[3,col_sub_grade]= "A4"

X_train.iloc[4,col_sub_grade]= "A5"

X_train.iloc[5,col_sub_grade]= "B1"

X_train.iloc[6,col_sub_grade]= "B2"

X_train.iloc[7,col_sub_grade]= "B3"

X_train.iloc[8,col_sub_grade]= "B4"

X_train.iloc[9,col_sub_grade]= "B5"

X_train.iloc[10,col_sub_grade]= "C1"

X_train.iloc[11,col_sub_grade]= "C2"

X_train.iloc[12,col_sub_grade]= "C3"

X_train.iloc[13,col_sub_grade]= "C4"

X_train.iloc[14,col_sub_grade]= "C5"

X_train.iloc[15,col_sub_grade]= "D1"

X_train.iloc[16,col_sub_grade]= "D2"

X_train.iloc[17,col_sub_grade]= "D3"

X_train.iloc[18,col_sub_grade]= "D4"

X_train.iloc[19,col_sub_grade]= "D5"

X_train.iloc[20,col_sub_grade]= "E1"

X_train.iloc[21,col_sub_grade]= "E2"

X_train.iloc[22,col_sub_grade]= "E3"

X_train.iloc[23,col_sub_grade]= "E4"

X_train.iloc[24,col_sub_grade]= "E5"

X_train.iloc[25,col_sub_grade]= "F1"

X_train.iloc[26,col_sub_grade]= "F2"

X_train.iloc[27,col_sub_grade]= "F3"

X_train.iloc[28,col_sub_grade]= "F4"

X_train.iloc[29,col_sub_grade]= "F5"

X_train.iloc[30,col_sub_grade]= "G1"

X_train.iloc[31,col_sub_grade]= "G2"

X_train.iloc[32,col_sub_grade]= "G3"

X_train.iloc[33,col_sub_grade]= "G4"

X_train.iloc[34,col_sub_grade]= "G5"







X_train.iloc[0,col_years]= "n/a"

X_train.iloc[1,col_years]= "< 1 year"

X_train.iloc[2,col_years]= "1 year"

X_train.iloc[3,col_years]= "2 years"

X_train.iloc[4,col_years]= "3 years"

X_train.iloc[5,col_years]= "4 years"

X_train.iloc[6,col_years]= "5 years"

X_train.iloc[7,col_years]= "6 years"

X_train.iloc[8,col_years]= "7 years"

X_train.iloc[9,col_years]= "8 years"

X_train.iloc[10,col_years]= "9 years"

X_train.iloc[11,col_years]= "10+ years"





# X_train.iloc[0,col_month]= "Jan"

# X_train.iloc[1,col_month]= "Feb"

# X_train.iloc[2,col_month]= "Mar"

# X_train.iloc[3,col_month]= "Apr"

# X_train.iloc[4,col_month]= "May"

# X_train.iloc[5,col_month]= "Jun"

# X_train.iloc[6,col_month]= "Jul"

# X_train.iloc[7,col_month]= "Aug"

# X_train.iloc[8,col_month]= "Sep"

# X_train.iloc[9,col_month]= "Oct"

# X_train.iloc[10,col_month]= "Nov"

# X_train.iloc[11,col_month]= "Dec"

#purposeのcount encoding

X_all = pd.concat([X_train, X_test], axis=0)



summary = X_all['purpose'].value_counts()

summary



X_train['purpose_count'] = X_train['purpose'].map(summary)

X_test['purpose_count'] = X_test['purpose'].map(summary)



# del X_train['purpose']

# del X_test['purpose']
#OneHotEncoder

# cols = ['home_ownership']

cols = ['purpose', 'home_ownership']

ohe = OneHotEncoder(cols=cols)

X_train = ohe.fit_transform(X_train)

X_test = ohe.transform(X_test)



# del X_train['home_ownership_4']

# del X_test['home_ownership_4']

# del X_train['home_ownership_5']

# del X_test['home_ownership_5']

# del X_train['home_ownership_6']

# del X_test['home_ownership_6']
#空欄を無職と推定

X_train['no_job_flag'] = 0

X_test['no_job_flag'] = 0

X_train.loc[X_train['emp_title'] == '#', 'no_job_flag'] = 1

X_test.loc[X_test['emp_title'] == '#', 'no_job_flag'] = 1
#テキスト用にコピーしておく

TXT_train_emp_title = X_train.emp_title.copy()

TXT_test_emp_title = X_test.emp_title.copy()



TXT_train_title = X_train.title.copy()

TXT_test_title = X_test.title.copy()



#Count Encoding

emp_title_count = X_train['emp_title'].value_counts()

X_train['emp_title'] = 1

X_train['emp_title'] = X_train['emp_title'].map(emp_title_count)

X_test['emp_title'] = 1

X_test['emp_title'] = X_test['emp_title'].map(emp_title_count)

del X_train['emp_title']

del X_test['emp_title']#'emp_title'のCount Encodingは、不要であることが判明



title_count = X_train['title'].value_counts()

X_train['title'] = 1

X_train['title'] = X_train['title'].map(title_count)

X_test['title'] = 1

X_test['title'] = X_test['title'].map(title_count)

# del X_train['emp_title']

# del X_test['emp_title']



# del X_train['title']

# del X_test['title']
#object typeを抽出

cats = []

for col in X_train.columns:

    if X_train[col].dtype == 'object':

        cats.append(col)

        

        print(col, X_train[col].nunique())
#OrdinalEncoder

oe = OrdinalEncoder(cols=cats, return_df=False)



X_train[cats] = oe.fit_transform(X_train[cats])

X_test[cats] = oe.transform(X_test[cats])
#避難していたデータを復元

for i in range(7):

     X_train.iloc[i,col_grade] = grade_temp[i]





for i in range(35):

     X_train.iloc[i,col_sub_grade] = sub_grade_temp[i]





for i in range(12):

     X_train.iloc[i,col_years] = years_temp[i]



        

# for i in range(12):

#      X_train.iloc[i,col_month] = month_temp[i]
#時系列に変換

X_train['time_series'] = X_train['e_year'] * 12 + X_train['e_month']

X_test['time_series'] = X_test['e_year'] * 12 + X_test['month']



X_train['e_time_series'] = X_train['e_year'] * 12 + X_train['e_month']

X_test['e_time_series'] = X_test['e_year'] * 12 + X_test['e_month']

#年収とグレードの相互作用を勘案

X_train['inc*grade'] = X_train['annual_inc'] * X_train['grade']

X_test['inc*grade'] = X_test['annual_inc'] * X_test['grade']
del X_train['year']

del X_test['year']

del X_train['month']

del X_test['month']

del X_train['e_year']

del X_test['e_year']

del X_train['e_month']

del X_test['e_month']



#テキスト

tdidf = TfidfVectorizer(max_features = 50)



TXT_train_emp_title = tdidf.fit_transform(TXT_train_emp_title)

TXT_test_emp_title = tdidf.transform(TXT_test_emp_title)



X_train = sp.sparse.hstack([X_train, TXT_train_emp_title])

X_test = sp.sparse.hstack([X_test, TXT_test_emp_title])



X_train = pd.DataFrame(X_train.todense())

X_test = pd.DataFrame(X_test.todense())







tdidf = TfidfVectorizer(max_features = 25)



TXT_train_title = tdidf.fit_transform(TXT_train_title)

TXT_test_title = tdidf.transform(TXT_test_title)



X_train = sp.sparse.hstack([X_train, TXT_train_title])

X_test = sp.sparse.hstack([X_test, TXT_test_title])



X_train = pd.DataFrame(X_train.todense())

X_test = pd.DataFrame(X_test.todense())



X_train.head()

num_split = 7 #best number of split based on simulation

num_iter = 20

stop_round = 100

scores = []

y_pred_cva = np.zeros(len(X_test)) #cvaデータ収納用

# skf = StratifiedKFold(n_splits=num_split, random_state=71, shuffle=True)



for h in range (num_iter):

    skf = StratifiedKFold(n_splits=num_split, random_state=h, shuffle=True)



    for i, (train_ix, test_ix) in tqdm(enumerate(skf.split(X_train, y_train))):

        X_train_, y_train_ = X_train.values[train_ix], y_train.values[train_ix]

        X_val, y_val = X_train.values[test_ix], y_train.values[test_ix]





        #clf = GradientBoostingClassifier()

    #     clf = LGBMClassifier() 

    #     clf.fit(X_train_, y_train_)



        clf = LGBMClassifier(boosting_type='gbdt', class_weight=None, colsample_bytree=0.9,

                                    importance_type='split', learning_rate=0.05, max_depth=-1,

                                    min_child_samples=20, min_child_weight=0.001, min_split_gain=0.0,

                                    n_estimators=9999, n_jobs=-1, num_leaves=15, objective=None,

                                    random_state=71, reg_alpha=0.0, reg_lambda=0.0, silent=True,

                                    subsample=1.0, subsample_for_bin=200000, subsample_freq=0) 

        clf.fit(X_train_, y_train_, early_stopping_rounds=stop_round, eval_metric='auc', eval_set=[(X_val, y_val)])    



        y_pred = clf.predict_proba(X_val)[:,1]

        score = roc_auc_score(y_val, y_pred)

        scores.append(score)

#         print(y_pred)



        y_pred_cva += clf.predict_proba(X_test)[:,1]

#         print(clf.predict_proba(X_test)[:,1])

#         print(y_pred_cva)



#         print('CV Score of Fold_%d is %f' % (i, score))



print(np.mean(scores))

print(scores)



y_pred_cva /= (num_split * num_iter)
y_pred_cva
submission = pd.read_csv('../input/homework-for-students2/sample_submission.csv', index_col=0)



submission.loan_condition = y_pred_cva

submission.to_csv('submission.csv')