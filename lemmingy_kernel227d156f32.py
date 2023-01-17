import pandas as pd

import numpy as np

import category_encoders as ce

import matplotlib as plt



from sklearn.impute import SimpleImputer

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.preprocessing import StandardScaler

from sklearn.pipeline import Pipeline

from sklearn.decomposition import PCA

from sklearn.metrics import roc_auc_score

from tqdm import tqdm_notebook as tqdm

from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import cross_val_score

import seaborn as sns

from scipy import stats

from sklearn.model_selection import train_test_split

from sklearn.feature_extraction.text import TfidfVectorizer

from scipy.sparse import hstack

import lightgbm as lab

from lightgbm import LGBMRegressor

from sklearn.linear_model import LogisticRegression



import matplotlib.pyplot as plt

plt.style.use('ggplot')

%matplotlib inline



pd.set_option('display.max_columns', 500)

# X_train= pd.read_csv('../input/train.csv',parse_dates=['issue_d','earliest_cr_line'])

# X_test = pd.read_csv('../input/test.csv',parse_dates=['issue_d','earliest_cr_line'])

df= pd.read_csv('../input/train.csv')

X_test = pd.read_csv('../input/test.csv')

y_train= df["ConvertedSalary"]

X_train=df.drop("ConvertedSalary",axis=1)
print(X_train.shape)

print(X_test.shape)

##addr_stateに緯度経度をjoinする場合<-外部データが必要だったら。

add = pd.read_csv('../input/country_info.csv')

X_all=pd.concat([X_train,X_test],axis=0)

#国を変換

county={'Bahamas':'Bahamas, The', 'Bosnia and Herzegovina':'Bosnia & Herzegovina', 'Congo, Republic of the...':'Congo, Repub. of the',

 "Côte d'Ivoire":"Cote d'Ivoire",

 'Democratic Republic of the Congo':'Congo, Dem. Rep.',

 'Gambia':'Gambia, The',

 'Hong Kong (S.A.R.)':'Hong Kong',

 'Iran, Islamic Republic of...':'Iran',

 'Libyan Arab Jamahiriya':'Libya',

 'Montenegro':'Other',

 'Myanmar':'Other',

 'Other Country (Not Listed Above)':'Other',

 'Republic of Korea':'Korea, South',

 'Republic of Moldova':'Moldova',

 'Russian Federation':'Russia',

 'South Korea':'Korea, South',

 'Syrian Arab Republic':'Syria',

 'The former Yugoslav Republic of Macedonia':'Macedonia',

 'Trinidad and Tobago':'Trinidad & Tobago',

 'United Republic of Tanzania':'Tanzania',

 'Venezuela, Bolivarian Republic of...':'Venezuela',

 'Viet Nam':'Vietnam',

}

X_all.Country = X_all.Country.replace(county)

X_all=pd.merge(X_all, add, on='Country', how='left')

X_train= X_all.iloc[:X_train.shape[0],:]

X_test=X_all.iloc[X_train.shape[0]:,:]

print(X_train.shape)

print(X_test.shape)
#MilitaryUS_Hobby_OpenSource

X_train["MilitaryUS"]=X_train["MilitaryUS"].fillna("No")

X_test["MilitaryUS"]=X_test["MilitaryUS"].fillna("No")

YN = {'Yes':1, 'No':0}

MHO=["MilitaryUS","Hobby","OpenSource","AdBlocker","AdBlockerDisable"]



for col in MHO:

    X_train[col] = X_train[col].replace(YN)

    X_test[col]=X_test[col].replace(YN)

print(X_train["Student"].value_counts())

print(X_test["Student"].value_counts())
YN = {'No':0, 'Yes, full-time' :2,'Yes, part-time':1}

MHO=["Student"]



for col in MHO:

    X_train[col] = X_train[col].replace(YN)

    X_test[col]=X_test[col].replace(YN)
YN = {'0-2 years':1,

 '12-14 years':13,

 '15-17 years':16,

 '18-20 years':19,

 '21-23 years':22,

 '24-26 years':25,

 '27-29 years':28,

 '3-5 years':4,

 '30 or more years':30,

 '6-8 years':7,

 '9-11 years':10}



lis=["YearsCoding"]



for col in lis:

    X_train[col] = X_train[col].replace(YN)

    X_test[col]=X_test[col].replace(YN)

YN={'20 to 99 employees':50,

'100 to 499 employees':250,

'10,000 or more employees':10000,

'1,000 to 4,999 employees':2500,

'10 to 19 employees':15,

'Fewer than 10 employees':5,

'500 to 999 employees':750,

'5,000 to 9,999 employees':7500}

lis=["CompanySize"]



for col in lis:

    X_train[col] = X_train[col].replace(YN)

    X_test[col]=X_test[col].replace(YN)





X_train["CompanySize"].value_counts()
YN={"3-5 years":4,

"0-2 years":1,

"6-8 years":7,

"12-14 years":13, 

"9-11 years":10,

"15-17 years":16,

"18-20 years":19,

"21-23 years":22,

"30 or more years":31,

"24-26 years":25,

"27-29 years":28}



lis=["YearsCodingProf"]



for col in lis:

    X_train[col] = X_train[col].replace(YN)

    X_test[col]=X_test[col].replace(YN)

X_train["YearsCodingProf"].value_counts()
YN={"Less than a year ago":1,

"Between 1 and 2 years ago":2,

"Between 2 and 4 years ago":3,

"More than 4 years ago":4, 

"I've never had a job":0,}



lis=["LastNewJob"]



for col in lis:

    X_train[col] = X_train[col].replace(YN)

    X_test[col]=X_test[col].replace(YN)

X_train["LastNewJob"].value_counts()
YN={'Agree':3,

 'Disagree':1,

 'Neither Agree nor Disagree':2,

 'Strongly agree':4,

 'Strongly disagree':0}



lis=["AgreeDisagree1","AgreeDisagree2","AgreeDisagree3"]



for col in lis:

    X_train[col] = X_train[col].replace(YN)

    X_test[col]=X_test[col].replace(YN)

X_train["AgreeDisagree1"].value_counts()
YN={'Neither agree nor disagree':3,

 'Somewhat agree':4,

 'Somewhat disagree':2,

 'Strongly agree':5,

 'Strongly disagree':1}



lis=["AdsAgreeDisagree1","AdsAgreeDisagree2","AdsAgreeDisagree3"]



for col in lis:

    X_train[col] = X_train[col].replace(YN)

    X_test[col]=X_test[col].replace(YN)

# set(X_train["AdsAgreeDisagree1"].values)
YN={'1 - 4 hours':3,

 '5 - 8 hours':6,

 '9 - 12 hours':10,

 'Less than 1 hour':0,

 'Over 12 hours':12}



lis=["HoursComputer"]



for col in lis:

    X_train[col] = X_train[col].replace(YN)

    X_test[col]=X_test[col].replace(YN)

YN={'1 - 2 hours':2,

 '3 - 4 hours':4,

 '30 - 59 minutes':1,

 'Less than 30 minutes':0,

 'Over 4 hours':5}



lis=["HoursOutside"]



for col in lis:

    X_train[col] = X_train[col].replace(YN)

    X_test[col]=X_test[col].replace(YN)
set(X_train["Age"].values)

YN={'18 - 24 years old':22,

 '25 - 34 years old':27,

 '35 - 44 years old':40,

 '45 - 54 years old':50,

 '55 - 64 years old':60,

 '65 years or older':65,

 'Under 18 years old':18}

lis=["Age"]



for col in lis:

    X_train[col] = X_train[col].replace(YN)

    X_test[col]=X_test[col].replace(YN)
set(X_train["Employment"].values)

YN={'Employed full-time':4,

 'Employed part-time':3,

 'Independent contractor, freelancer, or self-employed':5,

 'Not employed, and not looking for work':0,

 'Not employed, but looking for work':2,

 'Retired':1}

lis=["Employment"]



for col in lis:

    X_train[col] = X_train[col].replace(YN)

    X_test[col]=X_test[col].replace(YN)
YN={'A little bit interested':2,

 'Extremely interested':5,

 'Not at all interested':1,

 'Somewhat interested':3,

 'Very interested':4}

lis=["HypotheticalTools1","HypotheticalTools2","HypotheticalTools3","HypotheticalTools4","HypotheticalTools5"]



for col in lis:

    X_train[col] = X_train[col].replace(YN)

    X_test[col]=X_test[col].replace(YN)

YN={'A few times per week':3,

 'Less than once per month':1,

 'Multiple times per day':5,

 'Never':0,

 'Once a day':4,

 'Weekly or a few times per month':2}

lis=["CheckInCode"]



for col in lis:

    X_train[col] = X_train[col].replace(YN)

    X_test[col]=X_test[col].replace(YN)

# set(X_train["CheckInCode"].values)
X_train.head()
###ヒストグラム-1

df_X_train=X_train

df_X_test=X_test

# # 数値カラムを抽出 (float or int)

list_cols_num = []

for i in df_X_train.columns:

    if df_X_train[i].dtype == 'float64' or df_X_train[i].dtype == 'int64':

        list_cols_num.append(i)

        

print(list_cols_num)
# オブジェクトカラムを抽出 (object)

df_X_train=X_train

df_X_test=X_test



list_cols_cat = []

for i in df_X_train.columns:

    if df_X_train[i].dtype == 'object':

        list_cols_cat.append(i)

        

print(list_cols_cat)





# 各統計量を作成

# X_train

statics_X_train_cat = pd.DataFrame([df_X_train[list_cols_cat].nunique().values.tolist(),  # ユニーク数

                                df_X_train[list_cols_cat].isnull().sum().values.tolist()], # 欠損数

                              index=['unique', 'trainnull'],

                              columns=list_cols_cat).T

# X_test

statics_X_test_cat =  pd.DataFrame([df_X_test[list_cols_cat].nunique().values.tolist(),  # ユニーク数

                                df_X_test[list_cols_cat].isnull().sum().values.tolist()],  # 欠損数

                                index=['unique', 'null'],

                                columns=list_cols_cat).T



statics_X_train_cat
statics_X_test_cat
#ユニーク数の差をチェック

# check=X_train.columns



list_cols_cat = []

for i in df_X_train.columns:

    if df_X_train[i].dtype == 'object':

        list_cols_cat.append(i)

        

# print(list_cols_cat)

# for col in list_cols_cat:

#     a=set(X_train[col].unique())-set(X_test[col].unique())

#     print("train-test",col,len(a))

#     a=set(X_train[col].unique())-set(add[col].unique())

#     print("train-add",col,len(a))

#     a=set(X_test[col].unique())-set(X_train[col].unique())

#     print("test-train",col,len(a))

#     a=set(X_test[col].unique())-set(add[col].unique())

#     print("test-add",col,len(a))



    

# for col in categorical:

#     a=set(X_train[col].unique())-set(X_test[col].unique())

#     print("train",col,len(a))

#     a=set(X_test[col].unique())-set(X_train[col].unique())

#     print("test",col,len(a))
# set(X_all["Country"].values)-set(add["Country"])
# set(add["Country"].values)-set(X_all["Country"])
set(X_test["Country"].values)-set(X_train["Country"])
numeric=list_cols_num

# ['loan_amnt', 'installment',

#        'annual_inc','dti', 'delinq_2yrs',

#         'inq_last_6mths', 'mths_since_last_delinq',

#        'mths_since_last_record', 'open_acc', 'pub_rec', 'revol_bal',

#        'revol_util', 'total_acc',

#        'collections_12_mths_ex_med', 'mths_since_last_major_derog',

#        'acc_now_delinq', 'tot_coll_amt', 'tot_cur_bal','loan_condition','grade','sub_grade','emp_length']





text=["DevType","CommunicationTools","FrameworkWorkedWith","AdsActions","RaceEthnicity"]

categorical=list(set(list_cols_cat)-set(text))
X_train[numeric].corr()
# #ヒートマップ

# sns.heatmap(X_train[numeric].corr(), vmax=1, vmin=-1, center=0)
# #ヒートマップ

# sns.heatmap(X_test[numeric].corr(), vmax=1, vmin=-1, center=0)
# ###ヒストグラム-1

# df_X_train=X_train

# df_X_test=X_test

# # 数値カラムを抽出 (float or int)

# list_cols_num = []

# for i in df_X_train.columns:

#     if df_X_train[i].dtype == 'float64' or df_X_train[i].dtype == 'int64':

#         list_cols_num.append(i)

        

# print(list_cols_num)



# # 各統計量を作成

# # X_train

# statics_X_train_num = pd.DataFrame([df_X_train[list_cols_num].nunique().values.tolist(),  # ユニーク数

#                                 df_X_train[list_cols_num].isnull().sum().values.tolist(),  # 欠損数

#                               df_X_train[list_cols_num].mean().values.tolist(),  # 平均値

#                               df_X_train[list_cols_num].std().values.tolist(),  # 標準偏差

#                               df_X_train[list_cols_num].median().values.tolist(),  # 中央値

#                               df_X_train[list_cols_num].min().values.tolist(),  # 最小値

#                               df_X_train[list_cols_num].max().values.tolist()],  # 最大値

#                               index=['unique', 'null', 'mean', 'std', 'median', 'min', 'max'],

#                               columns=list_cols_num).T

# # X_test

# statics_X_test_num =  pd.DataFrame([df_X_test[list_cols_num].nunique().values.tolist(),  # ユニーク数

#                                 df_X_test[list_cols_num].isnull().sum().values.tolist(),  # 欠損数

#                                 df_X_test[list_cols_num].mean().values.tolist(),  # 平均値

#                                 df_X_test[list_cols_num].std().values.tolist(),  # 標準偏差

#                                 df_X_test[list_cols_num].median().values.tolist(),  # 中央値

#                                 df_X_test[list_cols_num].min().values.tolist(),  # 最小値

#                                 df_X_test[list_cols_num].max().values.tolist()],  # 最大値

#                                 index=['unique', 'null', 'mean', 'std', 'median', 'min', 'max'],

#                                 columns=list_cols_num).T



# statics_X_train_num
# # ヒストグラム-2

# statics_X_test_num
# # ヒストグラム-3



# plotpos = 1

# fig = plt.figure(figsize = (30, 30))



# for i in list_cols_num:

#     plotdata1 = df_X_train[i]

#     plotdata2 = df_X_test[i]



#     ax = fig.add_subplot(10, 3, plotpos)

#     ax.hist(plotdata1, bins=50, alpha=0.4)

#     ax.hist(plotdata2, bins=50, alpha=0.4)

#     ax.set_xlabel(i)

    

#     plotpos = plotpos + 1



# plt.show()
# ##クリッピング



# for col in numeric:

#     upperbound, lowerbound= np.percentile(X_train[col],[1,99])

#     X_train[col]=np.clip(X_train[col],upperbound,lowerbound)

#     X_test[col]=np.clip(X_test[col],upperbound,lowerbound)

# X_train['issue_d_y']=X_train['issue_d'].dt.year

# X_train['issue_d_m']=X_train['issue_d'].dt.month

# X_train['issue_d_d']=X_train['issue_d'].dt.day

# X_train['earliest_cr_line_y']=X_train['earliest_cr_line'].dt.year

# X_train['earliest_cr_line_m']=X_train['earliest_cr_line'].dt.month

# X_train['earliest_cr_line_d']=X_train['earliest_cr_line'].dt.day

# X_train['issue-cr_line']=X_train['issue_d']-X_train['earliest_cr_line']



# X_year=X_train.groupby('issue_d_y').mean()

# X_year
# X_test['issue_d_y']=X_test['issue_d'].dt.year

# X_test['issue_d_m']=X_test['issue_d'].dt.month

# X_test['issue_d_d']=X_test['issue_d'].dt.day

# X_test['earliest_cr_line_y']=X_test['earliest_cr_line'].dt.year

# X_test['earliest_cr_line_m']=X_test['earliest_cr_line'].dt.month

# X_test['earliest_cr_line_d']=X_test['earliest_cr_line'].dt.day

# X_test['issue-cr_line']=X_test['issue_d']-X_test['earliest_cr_line']

# X_test.groupby('issue_d_y').mean()

# X_test
# X_train.groupby(['issue_d_y','initial_list_status']).mean()
# X_test.groupby(['issue_d_y','initial_list_status']).mean()
# X_train.groupby(['issue_d_y','initial_list_status']).count()
# X_test.groupby(['issue_d_y','initial_list_status']).count()
# # 貸付日が2014年以降のデータのみ使用

# X_train = X_train[X_train['issue_d_y'] >= 2014]
# #ユニーク数の差をチェック

# # check=X_train.columns

# for col in text:

#     a=set(X_train[col].unique())-set(X_test[col].unique())

#     print("train",col,len(a))

#     a=set(X_test[col].unique())-set(X_train[col].unique())

#     print("test",col,len(a))



    

# for col in categorical:

#     a=set(X_train[col].unique())-set(X_test[col].unique())

#     print("train",col,len(a))

#     a=set(X_test[col].unique())-set(X_train[col].unique())

#     print("test",col,len(a))
###数字データの可視化用



#for col in numeric:

#     ax = sns.distplot(X_train[col].fillna(-9999))

#     plt.show()
#欠損値カウント

X_train['null_count']=X_train.isnull().sum(axis=1)

X_test['null_count'] = X_test.isnull().sum(axis=1)
text
X_train_1=X_train.copy()

X_test_1=X_test.copy()
X_train=X_train_1.copy()

X_test=X_test_1.copy()
# # テキスト特徴量を抽出・欠損値を'#'で補完

# from sklearn.model_selection import KFold







# dev_train = X_train['DevType']

# dev_train.fillna('#', inplace=True)

# dev_test = X_test['DevType']

# dev_test.fillna('#', inplace=True)

    

# ct_train = X_train['CommunicationTools']

# ct_train.fillna('#', inplace=True)

# ct_test = X_test['CommunicationTools']

# ct_test.fillna('#', inplace=True)



# fw_train = X_train['FrameworkWorkedWith']

# fw_train.fillna('#', inplace=True)

# fw_test = X_test['FrameworkWorkedWith']

# fw_test.fillna('#', inplace=True)



# aa_train = X_train['AdsActions']

# aa_train.fillna('#', inplace=True)

# aa_test = X_test['AdsActions']

# aa_test.fillna('#', inplace=True)



# re_train = X_train['RaceEthnicity']

# re_train.fillna('#', inplace=True)

# re_test = X_test['RaceEthnicity']

# re_test.fillna('#', inplace=True)









# # TF-IDF 

# tfidf = TfidfVectorizer(max_features=10000)



# dev_train = tfidf.fit_transform(dev_train)

# dev_test = tfidf.transform(dev_test)



# ct_train = tfidf.fit_transform(ct_train)

# ct_test = tfidf.transform(ct_test)



# fw_train = tfidf.fit_transform(fw_train)

# fw_test= tfidf.transform(fw_test)



# aa_train = tfidf.fit_transform(aa_train)

# aa_test = tfidf.transform(aa_test)



# re_train = tfidf.fit_transform(re_train)

# re_test = tfidf.transform(re_test)





# # 全てのTF-IDF matrixを結合

# txt_train = hstack([dev_train, ct_train,fw_train,aa_train,re_train]).tocsr()

# txt_test = hstack([dev_train, ct_train,fw_train,aa_train,re_train]).tocsr()

# print(txt_train.shape, txt_test.shape)



# # Stacking

# oof_train = np.zeros(len(X_train))

# oof_test = np.zeros(len(X_test))





# SEED = 71

# skf = KFold(n_splits=5, shuffle=False, random_state=None)





# for i, (train_ix, test_ix) in enumerate(skf.split(X_train, y_train)):

#     # トレーニングデータ・検証データに分割

#     X_tr, y_tr = txt_train[train_ix], y_train.iloc[train_ix]

#     X_te, y_te = txt_train[test_ix], y_train.iloc[test_ix]

    

#     # トレーニングデータからモデルを作成

#     clf = LogisticRegression(

#         solver='sag'

#     )

#     clf.fit(X_tr, y_tr)

    

#     # 検証データに対して予測

#     y_pred = clf.predict_proba(X_te)[:,1]

#     oof_train[test_ix] = y_pred

#     score = roc_auc_score(y_te, y_pred)

#     print('CV Score of Fold_%d is %f' % (i, score))

    

#     # テストデータに対して予測

#     oof_test += clf.predict_proba(txt_test)[:,1]



# oof_test /= NFOLDS
# X_train['tfidf'] = oof_train

# X_test['tfidf'] = oof_test
print(X_train.shape)

print(X_test.shape)

X_train
print(X_train.isnull().any())
# numeric.remove('loan_condition')

# numeric
print(X_train.shape)

print(X_test.shape)
numeric
# num=["loan_amnt","annual_inc","revol_bal","tot_cur_bal","dti"]

# num
##全部掛けて割って足して引く (初回だけにして、重要じゃないものを消していく)

# for i in numeric:

#     for k in numeric:

#         print(i,k)

#         X_train[i+"+"+k]=X_train[i]+X_train[k]

#         X_train[i+"-"+k]=X_train[i]-X_train[k]

#         X_train[i+"*"+k]=X_train[i]*X_train[k]

#         X_train[i+"/"+k]=X_train[i]/(X_train[k]+1)

#         X_test[i+"+"+k]=X_test[i]+X_test[k]

#         X_test[i+"-"+k]=X_test[i]-X_test[k]

#         X_test[i+"*"+k]=X_test[i]*X_test[k]

#         X_test[i+"/"+k]=X_test[i]/(X_test[k]+1)

# X_train.head()
# def encode(df, col):

#     # この方法だと場合によって最大値が変化するデータでは正確な値は出ない

#     # 例：月の日数が30日や31日の場合がある

#     df[col + '_cos'] = np.cos(2 * np.pi * df[col] / df[col].max())

#     df[col + '_sin'] = np.sin(2 * np.pi * df[col] / df[col].max())

#     return df



# # df = encode(df, 'dow')

# # df = encode(df, 'hour')

# df = encode(X_train, 'issue_d_d')

# df = encode(X_test, 'issue_d_d')

# df = encode(X_train, 'issue_d_m')

# df = encode(X_test, 'issue_d_m')

# df = encode(X_train, 'earliest_cr_line_m')

# df = encode(X_test, 'earliest_cr_line_m')

# df = encode(X_train, 'earliest_cr_line_d')

# df = encode(X_test, 'earliest_cr_line_d')
# X_train["issue-cr_line"]=X_train["issue-cr_line"].dt.days

# X_test["issue-cr_line"]=X_test["issue-cr_line"].dt.days

# X_train.head()
print(X_train.shape)

print(X_test.shape)

print(set(X_train.columns)-set(X_test.columns))
# #統計量を作成するものを指定。

# lis=["loan_amnt","emp_length","annual_inc","dti","revol_bal","revol_util","tot_coll_amt"]

# df=pd.DataFrame()

# gb=["addr_state"]



# for col in lis:

#     print(col)

#     tmp=X_train.groupby(gb).mean()

#     df[col+"_mean_add"]=tmp[col]

#     tmp=X_train.groupby(gb).max()

#     df[col+"_max_add"]=tmp[col]

#     tmp=X_train.groupby(gb).min()

#     df[col+"_min_add"]=tmp[col]

#     tmp=X_train.groupby(gb).std()

#     df[col+"_std_add"]=tmp[col]

#     tmp=X_train.groupby(gb).median()

#     df[col+"_median_add"]=tmp[col]    

# X_train=X_train.merge(df,how="left",on=gb)

# X_test=X_test.merge(df,how="left",on=gb)

print(X_train.shape)

print(X_test.shape)
# ##ランク付けをする

# X_all=pd.concat([X_train,X_test],axis=0)

# df=pd.DataFrame()



# for col in lis:

#     print(col)

#     tmp=X_all.groupby(gb)[col].rank()

#     df[col+"_rk_add"]=tmp    

# X_all=pd.concat([X_all,df],axis=1)

# X_train= X_all.iloc[:X_train.shape[0],:]

# X_test=X_all.iloc[X_train.shape[0]:,:]









#Save

X_train_n=X_train.copy()

X_test_n=X_test.copy()

y_train_n=y_train.copy()

print(X_train.shape)

print(X_test.shape)
#Load

X_train=X_train_n.copy()

X_test=X_test_n.copy()

y_train=y_train_n.copy()

print(X_train.shape)

print(X_test.shape)
#Ordinary-encoding

tmp=[]

for col in categorical:

    print(col)

    oe = ce.OrdinalEncoder(cols=[col])

    print(oe)

    tmp=oe.fit_transform(X_train)

    X_train[col+"_oe"]=tmp[col]

    tmp = oe.transform(X_test)

    X_test[col+"_oe"]=tmp[col]



X_train.head()



##LightGBMへ投入###過学習に超注意！

for col in categorical:

    X_train[col+"_oe"] = X_train[col+"_oe"].astype('category')

    X_test[col+"_oe"] = X_test[col+"_oe"].astype('category')

for col in categorical:

    X_train[col] = X_train[col].str.lower()

    X_test[col] = X_test[col].str.lower()

#count-encoding

for col in categorical:

    X_count=pd.concat([X_train, X_test])

    count_mean=X_count.groupby(col)["Respondent"].count()

    print(col)

    X_train[col+"_count"]= X_train[col].map(count_mean)

    X_test[col+"_count"]=X_test[col].map(count_mean)

X_train.head()



for col in text:

    print(col)

    X_count=pd.concat([X_train, X_test])

    count_mean=X_count.groupby(col)["Respondent"].count()

    X_train[col+"_count"]= X_train[col].map(count_mean)

    X_test[col+"_count"]=X_test[col].map(count_mean)

X_train.head()
# #グレードを標的にしたターゲットエンコーディング



# for cols in categorical:

#     print(cols)

#     X_target=pd.concat([X_train, X_test])

#     grouped_cat = X_target.groupby(cols)[cols].count().reset_index(name='cat_counts')

#     grouped_grade = X_target.groupby(cols)["SalaryType_oe"].sum().reset_index(name='grade_counts')

#     X_train = X_train.merge(grouped_cat, how = "left", on = cols)

#     X_train = X_train.merge(grouped_grade, how = "left", on = cols)

#     X_test = X_test.merge(grouped_cat, how = "left", on = cols)

#     X_test  = X_test.merge(grouped_grade, how = "left", on = cols)

#     X_train[cols+"_tag"] = X_train["grade_counts"]/X_train["cat_counts"]

#     X_test[cols+"_tag"] = X_test["grade_counts"]/X_test["cat_counts"]

#     X_train=X_train.drop(columns=["grade_counts","cat_counts"])

#     X_test=X_test.drop(columns=["grade_counts","cat_counts"])

# X_train.head()



# for cols in text:

#     print(cols)

#     X_target=pd.concat([X_train, X_test])

#     grouped_cat = X_target.groupby(cols)[cols].count().reset_index(name='cat_counts')

#     grouped_grade = X_target.groupby(cols)["grade"].sum().reset_index(name='grade_counts')

#     X_train = X_train.merge(grouped_cat, how = "left", on = cols)

#     X_train = X_train.merge(grouped_grade, how = "left", on = cols)

#     X_test = X_test.merge(grouped_cat, how = "left", on = cols)

#     X_test  = X_test.merge(grouped_grade, how = "left", on = cols)

#     X_train[cols+"_tag"] = X_train["grade_counts"]/X_train["cat_counts"]

#     X_test[cols+"_tag"] = X_test["grade_counts"]/X_test["cat_counts"]

#     X_train=X_train.drop(columns=["grade_counts","cat_counts"])

#     X_test=X_test.drop(columns=["grade_counts","cat_counts"])

# X_train.head()
# #One-hot_encoding

# ohe=["home_ownership","addr_state"]



# X_all=pd.concat([X_train,X_test],axis=0)



# X_all= pd.get_dummies(X_all,

#                        dummy_na = True,

#                        columns = ohe)



# ohes=set(X_all)-set(X_train.columns)



# X_train= X_all.iloc[:X_train.shape[0],:]

# X_test=X_all.iloc[X_train.shape[0]:,:]



# ohes
# home=['home_ownership_ANY',

#  'home_ownership_MORTGAGE',

#  'home_ownership_OWN',

#  'home_ownership_RENT',

#  'home_ownership_nan']

# addr=list(set(ohes)-set(home))



# #Ohe-hot交互特徴量

# for i in home:

#     for k in addr:

#         print(i,k)

#         X_train[i+"_"+k]=X_train[i]*X_train[k]

#         X_test[i+"_"+k]=X_test[i]*X_test[k]

# X_train.head()

# cat_ohe=list(set(categorical)-set(["addr_state","home_ownership"]))
X_train
X_train=X_train.drop(columns=categorical)

X_test=X_test.drop(columns=categorical)



X_train=X_train.drop(columns=text)

X_test=X_test.drop(columns=text)
X_train.head()
print(X_train.shape)

print(X_test.shape)
X_test_id=X_test[["Respondent"]]
#Save

X_train_c=X_train.copy()

X_test_c=X_test.copy()
#Load

X_train=X_train_c.copy()

X_test=X_test_c.copy()
###特徴量選定

drop_columns=[]



X_train=X_train.drop(columns=drop_columns)

X_test=X_test.drop(columns=drop_columns)

print(set(X_train.columns)-set(X_test.columns))

print(set(X_test.columns)-set(X_train.columns))
print(X_train.shape)

X_train.head()
X_train_v=X_train.copy()

X_test_v=X_test.copy()

y_train_v=y_train.copy()
X_train=X_train_v.copy()

X_test=X_test_v.copy()

y_train=y_train_v.copy()



print(X_train.shape)

print(X_test.shape)
# ##initial_list_statusごとに分けた場合のSplit

# X_train_1, X_val_1, y_train_1, y_val_1 = train_test_split(X_train,

#                                                    y_train,

#                                                   test_size=0.20,

#                                                   random_state=1)





# X_train_2, X_val_2, y_train_2, y_val_2 = train_test_split(X_train,

#                                                    y_train,

#                                                   test_size=0.20,

#                                                   random_state=42)





# X_train_3, X_val_3, y_train_3, y_val_3 = train_test_split(X_train,

#                                                    y_train,

#                                                   test_size=0.20,

#                                                   random_state=71)
# clf=LGBMRegressor(boosting_type='gbdt', class_weight=None, colsample_bytree=1.0,

#               importance_type='split', learning_rate=0.1, max_depth=-1,

#               min_child_samples=20, min_child_weight=0.001, min_split_gain=0.0,

#               n_estimators=9999, n_jobs=-1, num_leaves=31, objective=None,

#               random_state=1, reg_alpha=0.0, reg_lambda=0.0, silent=True,

#               subsample=1.0, subsample_for_bin=200000, subsample_freq=0)

# clf2=LGBMRegressor(boosting_type='gbdt', class_weight=None, colsample_bytree=1.0,

#               importance_type='split', learning_rate=0.1, max_depth=-1,

#               min_child_samples=20, min_child_weight=0.001, min_split_gain=0.0,

#               n_estimators=9999, n_jobs=-1, num_leaves=31, objective=None,

#               random_state=71, reg_alpha=0.0, reg_lambda=0.0, silent=True,

#               subsample=1.0, subsample_for_bin=200000, subsample_freq=0)

# clf3=LGBMRegressor(boosting_type='gbdt', class_weight=None, colsample_bytree=1.0,

#               importance_type='split', learning_rate=0.1, max_depth=-1,

#               min_child_samples=20, min_child_weight=0.001, min_split_gain=0.0,

#               n_estimators=9999, n_jobs=-1, num_leaves=31, objective=None,

#               random_state=42, reg_alpha=0.0, reg_lambda=0.0, silent=True,

#               subsample=1.0, subsample_for_bin=200000, subsample_freq=0)



# clf.fit(X_train_1, y_train_1, early_stopping_rounds=20, eval_set=[(X_val_1,y_val_1)],verbose=100)

# clf2.fit(X_train_2, y_train_2, early_stopping_rounds=20, eval_set=[(X_val_2,y_val_2)],verbose=100)

# clf3.fit(X_train_3, y_train_3, early_stopping_rounds=20, eval_set=[(X_val_3,y_val_3)],verbose=100)







# from sklearn.model_selection import StratifiedKFold



# scores = []





# cv_results = cross_val_score(clf,

#                              X_train,

#                              y_train,

#                              cv=5,

#                              scoring='r2')

# print(cv_results)

# print(cv_results.mean(),'+-', cv_results.std())

# cv_results = cross_val_score(clf2,

#                              X_train,

#                              y_train,

#                              cv=5,

#                              scoring='r2')

# print(cv_results)

# print(cv_results.mean(),'+-', cv_results.std())

# cv_results = cross_val_score(clf3,

#                              X_train,

#                              y_train,

#                              cv=5,

#                              scoring='r2')

# print(cv_results)

# print(cv_results.mean(),'+-', cv_results.std())



# for i, (train_ix, test_ix) in enumerate(tqdm(skf.split(X_train, y_train))):

#     X_train_, y_train_ = X_train.values[train_ix], y_train.values[train_ix]

#     X_val, y_val = X_train.values[test_ix], y_train.values[test_ix]

    

#     clf.fit(X_train_, y_train_, early_stopping_rounds=20, eval_metric='auc', eval_set=[(X_val,y_val)],verbose=100)

#     y_pred = clf.predict(X_val)

#     score = roc_auc_score(y_val, y_pred)

#     scores.append(score)

    

#     print('CV Score of Fold_%d is %f' % (i, score))

    



    





#予測用

clf=LGBMRegressor(boosting_type='gbdt', class_weight=None, colsample_bytree=1.0,

              importance_type='split', learning_rate=0.1, max_depth=-1,

              min_child_samples=20, min_child_weight=0.001, min_split_gain=0.0,

              n_estimators=100, n_jobs=-1, num_leaves=31, objective=None,

              random_state=1, reg_alpha=0.0, reg_lambda=0.0, silent=True,

              subsample=1.0, subsample_for_bin=200000, subsample_freq=0)

clf2=LGBMRegressor(boosting_type='gbdt', class_weight=None, colsample_bytree=1.0,

              importance_type='gain', learning_rate=0.1, max_depth=-1,

              min_child_samples=20, min_child_weight=0.001, min_split_gain=0.0,

              n_estimators=100, n_jobs=-1, num_leaves=26, objective=None,

              random_state=71, reg_alpha=0.0, reg_lambda=0.0, silent=True,

              subsample=1.0, subsample_for_bin=300000, subsample_freq=0)

clf3=LGBMRegressor(boosting_type='gbdt', class_weight=None, colsample_bytree=1.0,

              importance_type='split', learning_rate=0.1, max_depth=-1,

              min_child_samples=20, min_child_weight=0.001, min_split_gain=0.0,

              n_estimators=100, n_jobs=-1, num_leaves=36, objective=None,

              random_state=42, reg_alpha=0.0, reg_lambda=0.0, silent=True,

              subsample=1.0, subsample_for_bin=400000, subsample_freq=0)



clf.fit(X_train, y_train)

print(1)

clf2.fit(X_train, y_train)

print(2)

clf3.fit(X_train, y_train)

print(3)

###　予測出力コーナー(Data split用) only for light GBM

# y_pred=clf.predict(X_test)

y_pred=0.4*clf.predict(X_test)+0.3*clf2.predict(X_test)+0.3*clf3.predict(X_test)

submission=pd.DataFrame()

submission["Respondent"]=X_test["Respondent"]

submission["ConvertedSalary"]=y_pred

submission.to_csv('submission.csv',index=False)

# ###　特徴量の重要度確認コーナー

import lightgbm as lgb

# y_pred = clf.predict_proba(X_train)[:,1]

# score=roc_auc_score(y_train,y_pred)

# print(score)





fig, ax = plt.subplots(figsize=(10,15))

lgb.plot_importance(clf,max_num_features=50, ax=ax, importance_type='gain')

# importanceを表示する

importance = pd.DataFrame(clf.feature_importances_, index=X_train.columns, columns=['importance'])

importance=importance.sort_values('importance')

importance.index
importance
# all_col=list(set(X_train.columns)-set(ohes))
##ランク化

# from sklearn.preprocessing import quantile_transform



# X_all=pd.concat([X_train,X_test],axis=0)

# X_all[all_col]=quantile_transform(X_all[all_col],n_quantiles=100,random_state=0,output_distribution='normal')

# X_train= X_all.iloc[:X_train.shape[0],:]

# X_test=X_all.iloc[X_train.shape[0]:,:]


