# !pip install missingno

# !pip install autoimpute

# !pip install sklearn

# !pip install seaborn

# !pip install catboost

# !pip install xgboost

# !pip install lightgbm

# !pip install missingno

# !pip install numpy

# !pip3 install datawig
from sklearn.ensemble import RandomForestClassifier

import shap

import eli5

from eli5.sklearn import PermutationImportance

import lightgbm as lgb

# from pyecharts import *

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import missingno as mn

import math

import datetime

import warnings

from sklearn.preprocessing import LabelEncoder

from sklearn.metrics import f1_score

import itertools

warnings.filterwarnings(action='ignore')



from sklearn import preprocessing

from sklearn.metrics import f1_score

from sklearn.metrics import *

from lightgbm import LGBMClassifier







from IPython.core.display import display, HTML

display(HTML('<style>.container {width:98% !important;}</style>'))





pd.options.display.max_columns = 999

pd.options.display.max_rows = 50



from sklearn.neural_network import MLPClassifier

from sklearn.ensemble import AdaBoostClassifier

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.ensemble import ExtraTreesClassifier

from sklearn.ensemble import BaggingClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import VotingClassifier

from sklearn.preprocessing import *

from sklearn.metrics import *

from xgboost import XGBClassifier

from sklearn.metrics import *

import seaborn as sns
train = pd.read_csv("../input/PJT002_train.csv") # Train Data Input

val = pd.read_csv("../input/PJT002_validation.csv") # Validation Data Input

test = pd.read_csv("../input/PJT002_test.csv") # Test Data Input
train["set"] = 1 

val["set"] = 2 

test["set"] = 3 



sub = pd.concat([train,val,test], axis = 0)

sub = sub.reset_index()

sub.pop('index')

sub2 = sub.copy() 



main = pd.DataFrame() 

copy_cols=['id','set','fr_yn']

main[copy_cols] = sub[copy_cols]
# Funcitions



def ratio(col, frame, type=0): # 수치형 이상치 & 명목형 칼럼의 칼럼별 화재 확률

    

    if type == 0: # 수치형 == 0 / 명목형 == 1

        df = sub[sub.set == 2]

        ratio_frame = pd.DataFrame()

        ratio_frame['type'] = ['0 Values','Missing Values']

        

        ratio_frame['total_counts'] = [frame[frame[col] == 0].fr_yn.count(),

                              frame[frame[col].isnull()].fr_yn.count(),

                              ]

        

        ratio_frame['Y_counts'] = [frame[(frame[col] == 0)&(frame.fr_yn == 'Y')].id.count(),

                              frame[(frame[col].isnull())&(frame.fr_yn == 'Y')].id.count(),

                              ]

        

        ratio_frame['fire ratio'] = [str(round((frame[(frame[col] == 0)&(frame.fr_yn == 'Y')].fr_yn.count() / frame[frame[col] == 0].fr_yn.count())*100, 2)) + '%',

                          str(round((frame[(frame[col].isnull())&(frame.fr_yn == 'Y')].fr_yn.count() / frame[frame[col].isnull()].fr_yn.count())*100 , 2)) + "%",

                          ]

        

        ratio_frame['test_counts'] = [(frame[(frame[col] == 0)&(frame.set == 3)].id.count()),

                          (frame[(frame[col].isnull())&(frame.set == 3)].id.count())

                          ]

        return ratio_frame

    

    else:

        data = frame.copy()

        data[col] = frame[col].fillna('NONE')

        ratio_frame1 = pd.DataFrame()

        lst = list(data[col].unique())

        lst2 = []

        lst3 = []

        lst4 = []

        lst5 = []

        lst6 = []

        lst7 = []

        for i in lst:

            lst2.append(round(data[(data[col] == i)&(data.fr_yn == 'Y')].fr_yn.count() / data[data[col] == i].fr_yn.count(), 2))

            lst3.append(data[data[col] == i].fr_yn.count())

            lst4.append(data[(data[col] == i)&(data.fr_yn == 'Y')].fr_yn.count())

            lst5.append(data[(data[col] == i)&(data.set == 1)].id.count())

            lst6.append(data[(data[col] == i)&(data.set == 3)].id.count())

            



        ratio_frame1['label_name'] = lst

        ratio_frame1['fire_ratio'] = lst2

        ratio_frame1['total_counts'] = lst3

        ratio_frame1['Y_counts'] = lst4

        ratio_frame1['val_counts'] = lst6

        

        return ratio_frame1.sort_values(by = ['fire_ratio','total_counts'], ascending=False)





# pre_check function





def check(col, frame): # null / 0 수치 카운트

    

    print('Column Name : ' + str(col))

    print('')

    print('[Train Set]'+'  null count: ' + str(frame[(frame.set==1)&(frame[col].isnull())].count()[0]) + " // " + 'ZERO count: ' + str(frame[(frame.set==1)&(frame[col] == 0)].count()[0]))

    print('[Val Set]'+'   null count: ' + str(frame[(frame.set==2)&(frame[col].isnull())].count()[0]) + " // " +  'ZERO count: ' + str(frame[(frame.set==2)&(frame[col]==0)].count()[0]))

    print('[Test Set]'+'    null count: ' + str(frame[(frame.set==3)&(frame[col].isnull())].count()[0]) + " // " +  'ZERO count: ' + str(frame[(frame.set==3)&(frame[col]==0)].count()[0]))



        

# drop_check function        

        

def check_df(df):

    if (df[df.set==3].id.count() == 2957):

        print('pass')

    else:

        print('check_again')

        

# Matthew Correlation Coeffitients



from sklearn.metrics import matthews_corrcoef



def matt_corr(df):           # matt 상관계수 (for DataFrame)

    

    obj_df = df.select_dtypes(include='object')

    obj_df = obj_df.fillna('nodata')

    obj_df.drop(['dt_of_fr','dt_of_athrztn','fr_yn'],1,inplace=True)

    

    obj_cols = []

    obj_cols = list(obj_df.columns)



    mcc = [] 



    for i in obj_cols:

        for j in obj_cols:

            obj_df_temp = obj_df.copy()

            n = len(obj_df_temp[i].dropna())

            m = len(obj_df_temp[j].dropna())

            if n-m > 0:

                obj_df_temp[i].dropna()

            else:

                obj_df_temp[j].dropna()

                

            x = obj_df_temp[i].values

            y = obj_df_temp[j].values

            z = matthews_corrcoef(x,y)

            mcc.append(z)



    mcc_data = np.array(mcc).reshape(len(obj_cols),len(obj_cols))



    mcc_df = pd.DataFrame(data=mcc_data, columns = obj_cols, index = obj_cols)



    plt.figure(figsize=(30,15))

    sns.heatmap(mcc_df, annot=True, cmap='coolwarm')





def col_matt(df, col_name):    # 컬럼별 matt상관계수 (명목형)

    

    obj_df = df.select_dtypes(include='object')

    obj_df = obj_df.fillna('nodata')

    obj_df.drop(['dt_of_fr','dt_of_athrztn','fr_yn'],1,inplace=True)



    obj_cols = list(obj_df.columns)

    obj_cols.remove(col_name)



    mcc = [] 



    for i in obj_cols:

        x = obj_df[col_name].values

        y = obj_df[i].values

        z = matthews_corrcoef(x,y)

        mcc.append(z)



    mcc_data = np.array(mcc).reshape(len(obj_cols),1)



    mcc_df = pd.DataFrame(data=mcc_data, columns = [col_name], index = obj_cols)

    mcc_df = mcc_df.sort_values(by=col_name,ascending=False)

    

    return mcc_df





def check_label_rate(df, col, df_set=None, decimal_point=2, dependent_variable='fr_yn'):

    

    if df_set == None:

        item_cols = df[col].unique().tolist()

        item_cnt = [] # 각 항목별 개수

        ttl_cnt = len(df) # 데이터 프레임 행 개수

        fr_yn_Y_cnt = [] # 각 항목별 화재 발생 개수

        

        # 각 항목별 개수 확인

        for i in item_cols:

            if str(i) == 'nan':

                item_cnt.append(len(df[df[col].isnull()]))

            else:

                item_cnt.append(len(df[df[col]==i]))

        # 각 항목별 Yes 개수        

        for i in item_cols:

            if str(i) == 'nan':

                fr_yn_Y_cnt.append(len(df[df[col].isnull()&((df[dependent_variable]=="Y")|(df[dependent_variable]=="1"))]))

            else:

                fr_yn_Y_cnt.append(len(df[(df[col]==i)&((df[dependent_variable]=="Y")|(df[dependent_variable]=="1"))]))          

                        

        return pd.DataFrame(data=[np.round(np.array(item_cnt)/ttl_cnt*100,decimal_point),item_cnt,np.round(np.array(fr_yn_Y_cnt)/np.array(item_cnt)*100,decimal_point)], index=['item_per_ttl_rate(%)','item_cnt','fr_probability(%)'],columns=df[col].unique().tolist())

                

    elif len(df_set) == 2:

        item_cols = df[col].unique().tolist()

        item_cnt_1 = [] # 1번에서의 각 항목별 개수

        item_cnt_2 = [] # 2번에서의 각 항목별 개수

        ttl_cnt_1 = len(df[df.set==df_set[0]]) # 1번에서 데이터 프레임 행 개수

        ttl_cnt_2 = len(df[df.set==df_set[1]]) # 1번에서 데이터 프레임 행 개수

        fr_yn_Y_cnt_1 = [] # 1번에서의 각 항목별 화재 발생 수

        fr_yn_Y_cnt_2 = [] # 2번에서의 각 항목별 화재 발생 수

        

        #각 항목별 개수 확인

        for i in item_cols:

            if str(i) == 'nan':

                item_cnt_1.append(len(df[(df[col].isnull())&(df.set==df_set[0])]))

                item_cnt_2.append(len(df[(df[col].isnull())&(df.set==df_set[1])]))

            else:

                item_cnt_1.append(len(df[(df[col]==i)&(df.set==df_set[0])]))

                item_cnt_2.append(len(df[(df[col]==i)&(df.set==df_set[1])]))

        

        # 각 항목별 Yes 개수        

        for i in item_cols:

            if str(i) == 'nan':

                fr_yn_Y_cnt_1.append(len(df[df[col].isnull()&(df.set==df_set[0])&((df[dependent_variable]=="Y")|(df[dependent_variable]=="1"))]))

                fr_yn_Y_cnt_2.append(len(df[df[col].isnull()&(df.set==df_set[1])&((df[dependent_variable]=="Y")|(df[dependent_variable]=="1"))]))

            else:

                fr_yn_Y_cnt_1.append(len(df[(df[col]==i)&(df.set==df_set[0])&((df[dependent_variable]=="Y")|(df[dependent_variable]=="1"))]))

                fr_yn_Y_cnt_2.append(len(df[(df[col]==i)&(df.set==df_set[1])&((df[dependent_variable]=="Y")|(df[dependent_variable]=="1"))]))

                

        return pd.DataFrame(data=[np.round(np.array(item_cnt_1)/ttl_cnt_1*100,decimal_point),item_cnt_1,np.round(np.array(fr_yn_Y_cnt_1)/np.array(item_cnt_1)*100,decimal_point),np.round(np.array(item_cnt_2)/ttl_cnt_2*100,decimal_point),item_cnt_2,np.round(np.array(fr_yn_Y_cnt_2)/np.array(item_cnt_2)*100,decimal_point)], index=[['set1','set1','set1','set2','set2','set2'],['item_per_ttl_rate(%)','item_cnt','fr_probability(%)','item_per_ttl_rate(%)','item_cnt','fr_probability(%)']],columns=df[col].unique().tolist())

    

    else:

        item_cols = df[col].unique().tolist()

        item_cnt_1 = [] # 1번에서의 각 항목별 개수

        item_cnt_2 = [] # 2번에서의 각 항목별 개수

        item_cnt_3 = [] # 3번에서의 각 항목별 개수

        ttl_cnt_1 = len(df[df.set==df_set[0]]) # 1번에서 데이터 프레임 행 개수

        ttl_cnt_2 = len(df[df.set==df_set[1]]) # 1번에서 데이터 프레임 행 개수

        ttl_cnt_3 = len(df[df.set==df_set[2]]) # 1번에서 데이터 프레임 행 개수

        fr_yn_Y_cnt_1 = [] # 1번에서의 각 항목별 화재 발생 수

        fr_yn_Y_cnt_2 = [] # 2번에서의 각 항목별 화재 발생 수

        

    #각 항목별 개수 확인

        for i in item_cols:

            if str(i) == 'nan':

                item_cnt_1.append(len(df[(df[col].isnull())&(df.set==df_set[0])]))

                item_cnt_2.append(len(df[(df[col].isnull())&(df.set==df_set[1])]))

                item_cnt_3.append(len(df[(df[col].isnull())&(df.set==df_set[2])]))

            else:

                item_cnt_1.append(len(df[(df[col]==i)&(df.set==df_set[0])]))

                item_cnt_2.append(len(df[(df[col]==i)&(df.set==df_set[1])]))

                item_cnt_3.append(len(df[(df[col]==i)&(df.set==df_set[2])]))

    

    # 각 항목별 Yes 개수        

        for i in item_cols:

            if str(i) == 'nan':

                fr_yn_Y_cnt_1.append(len(df[df[col].isnull()&(df.set==df_set[0])&((df[dependent_variable]=="Y")|(df[dependent_variable]=="1"))]))

                fr_yn_Y_cnt_2.append(len(df[df[col].isnull()&(df.set==df_set[1])&((df[dependent_variable]=="Y")|(df[dependent_variable]=="1"))]))

            else:

                fr_yn_Y_cnt_1.append(len(df[(df[col]==i)&(df.set==df_set[0])&((df[dependent_variable]=="Y")|(df[dependent_variable]=="1"))]))

                fr_yn_Y_cnt_2.append(len(df[(df[col]==i)&(df.set==df_set[1])&((df[dependent_variable]=="Y")|(df[dependent_variable]=="1"))]))



        return pd.DataFrame(data=[np.round(np.array(item_cnt_1)/ttl_cnt_1*100,decimal_point),item_cnt_1,np.round(np.array(fr_yn_Y_cnt_1)/np.array(item_cnt_1)*100,decimal_point),np.round(np.array(item_cnt_2)/ttl_cnt_2*100,decimal_point),item_cnt_2,np.round(np.array(fr_yn_Y_cnt_2)/np.array(item_cnt_2)*100,decimal_point),np.round(np.array(item_cnt_3)/ttl_cnt_3*100,decimal_point),item_cnt_3], index=[['set1','set1','set1','set2','set2','set2','set3','set3'],['item_per_ttl_rate(%)','item_cnt','fr_probability(%)','item_per_ttl_rate(%)','item_cnt','fr_probability(%)','item_per_ttl_rate(%)','item_cnt']],columns=df[col].unique().tolist())







# 수치형 데이터 EDA 함수    

    

def percentile(df, column, n):

    # 원하는 변수의 percentile 구간별 Y인 비율(그 구간의 Y 개수/그 구간의 전체 fr_yn)

    # i=> 분위수

    percentiles = []

    s = 0

    values = 100/n

    values = values * 0.01



    for d in range(n):

        s += values

        s = round(s,2)

        percentiles.append(s)

    percentile = df[column].describe(percentiles=percentiles)

    

    index = []

    values = []

    fr_rate = []

    grp_cnt = []

    Y_cnt = []

    

    for i in range(4, len(percentile)-1):

        if i>4 and i<len(percentile)-1:

            index.append(percentile.index[i])

            values.append(percentile.values[i])

            fr_rate.append(df[(df[column] <= percentile.values[i])&(df[column] > percentile.values[i-1])&(df.fr_yn=="Y")]["fr_yn"].count()/df[(df[column] <= percentile.values[i])&(df[column] > percentile.values[i-1])]["fr_yn"].count())

            grp_cnt.append(df[(df[column] <= percentile.values[i])&(df[column] > percentile.values[i-1])]["fr_yn"].count())

            Y_cnt.append(df[(df[column] <= percentile.values[i])&(df[column] > percentile.values[i-1])&(df.fr_yn=="Y")]["fr_yn"].count())

        else:

            index.append(percentile.index[i])

            values.append(percentile.values[i])  

            fr_rate.append(df[(df[column] <= percentile.values[i])&(df.fr_yn=="Y")]["fr_yn"].count()/df[(df[column] <= percentile.values[i])]["fr_yn"].count())

            grp_cnt.append(df[(df[column] <= percentile.values[i])]["fr_yn"].count())

            Y_cnt.append(df[(df[column] <= percentile.values[i])&(df.fr_yn=="Y")]["fr_yn"].count())

    

    result_df = pd.DataFrame(index=index, columns=[column,'fr_rate','grp_cnt','Y_cnt'])       

    result_df[column] = values

    result_df['fr_rate'] = fr_rate

    result_df['grp_cnt'] = grp_cnt

    result_df['Y_cnt'] = Y_cnt

    

    return result_df
sub.head(5) # Raw-Data
mn.matrix(sub,color =(0.25,0.4,0.6)) # Missing Values Visualize(Missingno)
# 전기, 가스, lw 관련 컬럼들은 따로 묶어 처리



ele_info =sub[sub.columns[sub.columns.map(lambda x : x.startswith('ele'))]] 

gas_info = sub[sub.columns[sub.columns.map(lambda x : x.startswith('gas'))]]

lw_info = sub[sub.columns[sub.columns.map(lambda x : x.startswith('lw'))]]



sub = sub[sub.columns[sub.columns.map(lambda x : True if not x.startswith("ele") else False)]]

sub = sub[sub.columns[sub.columns.map(lambda x : True if not x.startswith("gas") else False)]]

sub = sub[sub.columns[sub.columns.map(lambda x : True if not x.startswith("lw") else False)]]



ele_info['isele'] = ele_info.median(axis=1)

gas_info['isgas'] = gas_info.median(axis=1)

lw_info['islw'] = lw_info.median(axis=1)



sub['isele'] = ele_info['isele']

sub['isgas'] = gas_info['isgas']

sub['islw'] = lw_info['islw']
#1-1) 지역별(대단위) 화재 비율

print('전체 화재 비율: {}'.format(round(sub[(sub.fr_yn=='Y')].fr_yn.count() / sub.fr_yn.count(),2)))

print('경남 화재 비율: {}'.format(round(sub[(sub.set==1)&(sub.fr_yn=='Y')].fr_yn.count() / sub[sub.set==1].fr_yn.count(), 2)))

print('김해 화재 비율: {}'.format(round(sub[(sub.set==2)&(sub.fr_yn=='Y')].fr_yn.count() / sub[sub.set==2].fr_yn.count(), 2)))
#1-2) 컬럼 Missing Value 비율 & 해당 컬럼 Missing Value 의 화재확률

# Missing Value DataFrame



mis = np.array(sub.drop(['id','dt_of_fr','dt_of_athrztn','fr_yn','set'],1).columns)

mis_val = np.array((69054 - sub.drop(['id','dt_of_fr','dt_of_athrztn','fr_yn','set'],1).count().values) / 69054)

miss_df = pd.DataFrame()

miss_df['cols'] = mis

miss_df['mis_val_ratio'] = mis_val



lis = []

for i in list(mis):

    fr_ratio = sub[(sub[i].isnull())&(sub.fr_yn == 'Y')].fr_yn.count() / sub[sub[i].isnull()].fr_yn.count()

    lis.append(fr_ratio)

    

lis = pd.Series(lis).map(lambda x : 0 if str(x)=='nan' else x).values

miss_df['mis_val_fr_ratio']=lis

miss_df = miss_df.sort_values(by='mis_val_fr_ratio',ascending=False)

miss_df = miss_df[miss_df.mis_val_ratio != 0].sort_values(by='mis_val_ratio',ascending=False)

mis_cols = np.array(miss_df['cols'])

mis_val_fr_ratio = np.array(miss_df['mis_val_fr_ratio'].values)

mis_mis_val_ratio = np.array(miss_df['mis_val_ratio'].values)

tt = miss_df.cols.tolist()

miss_df.index = tt

miss_df.drop('cols',1,inplace=True)

miss_df
plt.figure(figsize=(20,10)) # 수치형 변수 Corr

sns.heatmap(sub.corr(),vmax=1,vmin=-1,annot=True,cmap='vlag')
matt_corr(sub) # 명목형 변수 Matt Corr
sub.emd_nm[9165] = '경상남도 함안군 칠원읍'

sub.emd_nm.fillna('경상남도 진주시 하대동',inplace = True)

sub['emd_nm2'] = sub['emd_nm'].apply(lambda x : x.split(" ")[1]) # 파생변수 (가장 큰 지역단위로 재정의)

main['emd_nm2'] = sub['emd_nm2']



# 지역별 화재확률 비교



reg_gr = pd.DataFrame(sub.groupby(['emd_nm2']).fr_yn.count()).reset_index()

reg_gr2 = pd.DataFrame(sub[sub.fr_yn=='Y'].groupby(['emd_nm2']).fr_yn.count()).reset_index()

reg_gr3 = pd.DataFrame(sub[sub.fr_yn=='N'].groupby(['emd_nm2']).fr_yn.count()).reset_index()

reg_gr = reg_gr.merge(reg_gr2, how='left',on='emd_nm2')

reg_gr = reg_gr.merge(reg_gr3, how='left',on='emd_nm2')

reg_gr['fire_ratio'] = reg_gr['fr_yn_y'] / (reg_gr['fr_yn_y']+reg_gr['fr_yn'])

reg_gr.drop(['fr_yn_y','fr_yn'],1,inplace=True)

reg_gr.rename(columns={'fr_yn_x':'total_counts'},inplace=True)

reg_gr['fire_ratio'] = reg_gr['fire_ratio'].apply(lambda x : 0 if str(x) == 'nan' else x)

reg_gr = reg_gr.sort_values(by='fire_ratio',ascending=False)

reg_gr
main["fr_years"] = sub.dt_of_fr.apply(lambda x : x.split(' ')[0])

main['fr_years'] = main['fr_years'].apply(lambda x : pd.to_datetime(str(x),format='%Y-%m-%d'))

main['fr_years'] = main['fr_years'].apply(lambda x : x.isocalendar()[0]).astype(str)



# fr_weekend// 사고 발생일 주말 / 평일 분류



sub["weekend"] = sub.dt_of_fr.apply(lambda x : x.split(' ')[0])

sub['weekend'] = sub['weekend'].apply(lambda x : pd.to_datetime(str(x),format='%Y-%m-%d'))

sub['weekend'] = pd.to_datetime(sub['weekend']).dt.dayofweek



sub['weekend'] = sub['weekend'].apply(lambda x : 'weekend' if x > 4 else 'weekday')

main['weekend'] = sub['weekend'].astype(str)



ratio('weekend',sub,1)
# fr_months// 1달 단위 전처리



main["fr_months"] = sub.dt_of_fr.apply(lambda x : x.split(' ')[0])

main['fr_months'] = main['fr_months'].apply(lambda x : pd.to_datetime(str(x),format='%Y-%m-%d'))

main['fr_months'] = main['fr_months'].apply(lambda x : str(x).split('-')[1]).astype(str)
sns.barplot(x = check_label_rate(main, 'fr_months', df_set=[1,2]).loc['set1','fr_probability(%)'].index, y = check_label_rate(main, 'fr_months', df_set=[1,2]).loc['set1','fr_probability(%)'].values)

plt.ylim(8,17)

plt.show()

ratio('fr_months',main,1)
# 계절처리



main['season'] = main['fr_months']



def season(x):

    if 3 <= int(x.season) <= 5:

        x.season = 'spring'

        return x

    elif 6 <= int(x.season) <= 8:

        x.season = 'sum'

        return x

    elif 9 <= int(x.season) <= 11:

        x.season = 'fall'

        return x

    else:

        x.season = 'winter'

        return x



main = main.apply(season, axis=1)



sns.barplot(x = ratio('season',main,1)['label_name'], y = ratio('season',main,1)['fire_ratio'])

plt.ylim(0,0.2)

plt.show()

ratio('season',main,1)
# 발생 1시간 단위 전처리 (fr_hours)

main["fr_hours"] = sub.dt_of_fr.apply(lambda x : x.split(' ')[1])

main['fr_hours'] = main.fr_hours.apply(lambda x: str(x.split(':')[0]))



# 시간 3시간 단위 처리



main['hours_3'] = main['fr_hours'].apply(lambda x : math.floor(int(x)/3) + 1)
# Make Climate DataFrame



sub["date_"] = sub2.dt_of_fr.map(lambda x : datetime.datetime.strptime(x, '%Y-%m-%d %H:%M:%S'))

tmp_df = sub[["id","date_","set","tmprtr","hmdt","wnd_spd"]]



tmp_df["dt_of_ymd"] = tmp_df.date_.map(lambda x : str(x.year)+"_"+str(x.month)+"_"+str(x.day)) # 연/월/일 column

tmp_df['fill_emd_nm_si'] = sub['emd_nm2'] # emd_nm column '시'로 재정렬

tmp_df["dt_of_hour"] = tmp_df.date_.map(lambda x : x.hour)



tmp_df_rf = tmp_df.groupby(["dt_of_ymd","dt_of_hour","fill_emd_nm_si"])["tmprtr","hmdt","wnd_spd"].median().reset_index()



tmp_df["dt_of_md"] = tmp_df.date_.map(lambda x : str(x.month)+"_"+str(x.day)) # 발생 월/일 column

tmp_df["dt_of_ym"] = tmp_df.date_.map(lambda x : str(x.year)+"_"+str(x.month)) # 발생 연/월 column

tmp_df["dt_of_m"] = tmp_df.date_.map(lambda x : x.month) # 발생월 column
# 기후요소 이상치 칼럼 파생변수 추가 (기후 3요소 중 0값의 열을 포함하는 개수)



sub[["tmprtr","hmdt","wnd_spd"]] = sub2[["tmprtr","hmdt","wnd_spd"]]

sub['hmdt_0'] = sub['hmdt'].apply(lambda x : 1 if x == 0 else 0)

sub['tmprtr_0'] = sub['tmprtr'].apply(lambda x : 1 if x == 0 else 0)

sub['wnd_spd_0'] = sub['wnd_spd'].apply(lambda x : 1 if x == 0 else 0)

sub['cli_sum_0'] = sub['wnd_spd_0'] + sub['tmprtr_0'] + sub['hmdt_0']

main['cli_sum_0'] = sub['cli_sum_0']
sns.barplot(x = ratio('cli_sum_0',main,1)['label_name'], y = ratio('cli_sum_0',main,1)['fire_ratio'])

plt.ylim(0,0.3)

plt.show()

ratio('cli_sum_0',main,1)
# 온도 (tmprtr)



def fill_na_tmp(x): # 251개 -> 249개

    ymd = x.dt_of_ymd

    hour = x.dt_of_hour

    emd_nm = x.fill_emd_nm_si

    

    if str(x.tmprtr)== "nan":

        return tmp_df_rf[(tmp_df_rf.dt_of_ymd == ymd)&(tmp_df_rf.dt_of_hour == hour)&(tmp_df_rf.fill_emd_nm_si == emd_nm)].tmprtr.values[0]

    else:

        return x.tmprtr

    

def fill_na_tmp2(x): # 249개 -> 82개

    ymd = x.dt_of_ymd

    hour = x.dt_of_hour

    

    if str(x.tmprtr)== "nan":

        return tmp_df[(tmp_df.dt_of_ymd == ymd)&(tmp_df.dt_of_hour == hour)].tmprtr.median()

    else:

        return x.tmprtr



## 함수실행

tmp_df["tmprtr"] = tmp_df.apply(lambda x : fill_na_tmp(x), axis=1)

tmp_df["tmprtr"] = tmp_df.apply(lambda x : fill_na_tmp2(x), axis=1)

tmp_df['f_tmprtr'] = tmp_df.sort_values(['date_'])['tmprtr'].fillna(method='ffill') # 82개 -> 0개



# 습도 (hmdt)



def fill_na_hmdt(x): # 150개 -> 148개

    ymd = x.dt_of_ymd

    hour = x.dt_of_hour

    emd_nm = x.fill_emd_nm_si

    

    if str(x.hmdt)== "nan":

        return tmp_df_rf[(tmp_df_rf.dt_of_ymd == ymd)&(tmp_df_rf.dt_of_hour == hour)&(tmp_df_rf.fill_emd_nm_si == emd_nm)].hmdt.values[0]

    else:

        return x.hmdt

    

def fill_na_hmdt2(x): # 148개 -> 56개

    ymd = x.dt_of_ymd

    hour = x.dt_of_hour

    

    if str(x.hmdt)== "nan":

        return tmp_df[(tmp_df.dt_of_ymd == ymd)&(tmp_df.dt_of_hour == hour)].hmdt.median()

    else:

        return x.hmdt



    

## 함수실행

tmp_df["hmdt"] = tmp_df.apply(lambda x : fill_na_hmdt(x), axis=1)

tmp_df["hmdt"] = tmp_df.apply(lambda x : fill_na_hmdt2(x), axis=1)

tmp_df['f_hmdt'] = tmp_df.sort_values(['date_'])['hmdt'].fillna(method='ffill') # 56개 -> 0개



# 풍속 (wnd_spd) 처리



def fill_na_wnd_spd(x): # 3199개 -> 3197개

    ymd = x.dt_of_ymd

    hour = x.dt_of_hour

    emd_nm = x.fill_emd_nm_si

    

    if str(x.wnd_spd)== "nan":

        return tmp_df_rf[(tmp_df_rf.dt_of_ymd == ymd)&(tmp_df_rf.dt_of_hour == hour)&(tmp_df_rf.fill_emd_nm_si == emd_nm)].wnd_spd.values[0]

    else:

        return x.wnd_spd

    

def fill_na_wnd_spd2(x): # 3197 -> 912개

    ymd = x.dt_of_ymd

    hour = x.dt_of_hour

    

    if str(x.wnd_spd)== "nan":

        return tmp_df[(tmp_df.dt_of_ymd == ymd)&(tmp_df.dt_of_hour == hour)].wnd_spd.median()

    else:

        return x.wnd_spd



## 함수실행

tmp_df["wnd_spd"] = tmp_df.apply(lambda x : fill_na_wnd_spd(x), axis=1)

tmp_df["wnd_spd"] = tmp_df.apply(lambda x : fill_na_wnd_spd2(x), axis=1)

tmp_df['f_wnd_spd'] = tmp_df.sort_values(['date_'])['wnd_spd'].fillna(method='ffill') # 912개 -> 0개

sub = sub.merge(tmp_df[['id','f_tmprtr','f_hmdt','f_wnd_spd']], how = 'left', on = 'id') # 병합

main[['tmprtr','hmdt','wnd_spd']] = sub[['f_tmprtr','f_hmdt','f_wnd_spd']]
sns.barplot(x = percentile(main, 'hmdt',10).index, y = percentile(main, 'hmdt',10)['fr_rate'])

plt.ylim(0,0.3)

plt.show()

percentile(main, 'hmdt',10)
ratio('bldng_us_clssfctn',sub,1)
# 전기 파생변수 

# 당월 전기 사용량 grouping



ele_col = sub2.columns[sub2.columns.map(lambda x : x.startswith("ele"))]

col = ["id", "jmk" ,"dt_of_fr","fr_yn","lnd_ar","ttl_ar"]

for i in ele_col:

    col.append(i)



ele_df = sub2[col]

ele_df["dt_of_fr_fm"] = ele_df.dt_of_fr.map(lambda x : datetime.datetime.strptime(x, '%Y-%m-%d %H:%M:%S'))

ele_df["fr_yr_mn"] = ele_df.dt_of_fr_fm.map(lambda x : str(x.year))+ele_df.dt_of_fr_fm.map(lambda x : "0"+str(x.month) if len(str(x.month))==1 else str(x.month))



def find_col(x):

    id_ = x.id

    date_ =  x.fr_yr_mn

    col = x[x.index.map(lambda x : str(x).endswith(date_))].index[0]

    value = x[col]

    return value



def div_used_ele (x):

    if x == -1:

        return "Unknown"

    elif x == 0:

        return "93p"

    elif 0 < x <=1830:

        return "96p"

    elif 1830 < x <= 93100:

        return "99p"

    else:

        return "100p"



ele_df["ele_used"]= ele_df.apply(lambda x : find_col(x), axis=1)

ele_df.ele_used.fillna(-1, inplace=True)

ele_df = ele_df[["id","ele_used","fr_yn"]]



ele_df["grp_ele"] = ele_df.ele_used.map(lambda x: div_used_ele(x))

main = main.merge(ele_df[["id","grp_ele"]], how="left", on="id")

desc_df = ele_df.describe(percentiles=[i*0.001+0.98 for i in range(20)])
# 김해의 전기확률이 20~22%임을 감안하여 범주 생성



sns.barplot(x = ratio('grp_ele',main,1)['label_name'], y = ratio('grp_ele',main,1)['fire_ratio'])

plt.ylim(0,1)

plt.show()

ratio('grp_ele',main,1)
# 승인일자 cleaning / 파생변수



dt_of_a = sub2[["id","set","dt_of_athrztn"]].fillna("nodata")

dt_of_a["no_spc_col"] = dt_of_a.dt_of_athrztn.map(lambda x : str(x).replace(" ", "0"))



def chg_yr(x):

    if str(x).startswith("9") and len(str(x)) == 6:

        return "19"+ str(x)

    elif str(x).startswith("9") and len(str(x)) == 7:

        return "1"+ str(x)

    else:

        return str(x)



dt_of_a = sub2[["id","set","dt_of_athrztn"]].fillna("nodata")

dt_of_a["no_spc_col"] = dt_of_a.dt_of_athrztn.map(lambda x : str(x).replace(" ", "0"))



def chg_yr(x):

    if str(x).startswith("9") and len(str(x)) == 6:

        return "19"+ str(x)

    elif str(x).startswith("9") and len(str(x)) == 7:

        return "1"+ str(x)

    else:

        return str(x)



dt_of_a["base_col"] = dt_of_a.no_spc_col.map(lambda x: chg_yr(x))

dt_of_a["athz_year"] = dt_of_a.base_col.map(lambda x : x[:4] if x != "nodata" else x)



def year_mm(x):

    if len(x) == 4:

        return x+"00"

    elif x =="nodata":

        return x

    else:

        return x[:6]



dt_of_a["athz_year_mm"] = dt_of_a.base_col.map(lambda x : year_mm(x))

dt_of_a



def div_deca_yr(x):

    if x != "nodata":

        x = x[:3]+ x[3].replace(x[3],"0")

        return x

    else:

        return x



dt_of_a["athz_yr_deca"] = dt_of_a.athz_year.map(lambda x : div_deca_yr(x))

main = main.merge(dt_of_a[["id","athz_yr_deca"]],how="left", on="id")
# 이상치 cleaning



main['athz_yr_deca'] = main['athz_yr_deca'].apply(lambda x : '1990' if x == '9710' else x)

main['athz_yr_deca'] = main['athz_yr_deca'].apply(lambda x : '1970' if x == '9780' else x)

main['athz_yr_deca'] = main['athz_yr_deca'].apply(lambda x : '1980' if x == '9800' else x)

main['athz_yr_deca'] = main['athz_yr_deca'].apply(lambda x : '1980' if x == '9820' else x)

main['athz_yr_deca'] = main['athz_yr_deca'].apply(lambda x : '1990' if x == '9900' else x)

main['athz_yr_deca'] = main['athz_yr_deca'].apply(lambda x : '1990' if x == '9990' else x)
plt.figure(figsize=(15,10))

sns.barplot(x = ratio('athz_yr_deca',main,1)['label_name'], y = ratio('athz_yr_deca',main,1)['fire_ratio'])

plt.ylim(0,0.8)

plt.show()

ratio('athz_yr_deca',main,1)
plt.rc("font", family = "Pyunji R")



fig = plt.figure(figsize = (35, 100))

j = 0

for item in sub.select_dtypes(include='object').columns:

    if item != "dt_of_athrztn" and item !="dt_of_fr" and item !="emd_nm":

        if item != "fr_fghtng_fclt_spcl_css_5_yn" and item != 'fr_fghtng_fclt_spcl_css_6_yn' and item != 'fr_yn' and item !='slf_fr_brgd_yn':

            plt.subplot(8, 2, j+1)

            j += 1

            

            chart = sns.countplot(x=item,hue='fr_yn', 

                                  data=sub.select_dtypes(include='object'),

                                 palette='Set1')

            chart.set_xticklabels(

            chart.get_xticklabels(),

            rotation=45,

            fontsize=15)

            plt.legend(loc=1,fontsize='large'),

            chart.set_xlabel(xlabel = item, fontsize=25)

fig.tight_layout()

fig.subplots_adjust(top=0.95)

plt.show() 
plt.figure(figsize=(20,10))

plt.subplot(2,3,1)

sns.barplot(x = percentile(sub, 'ttl_ar',10).index, y = percentile(sub, 'ttl_ar',10)['fr_rate'])

plt.title('ttl_ar',size=20)

plt.subplot(2,3,2)

sns.barplot(x = percentile(sub, 'bldng_cnt',10).index, y = percentile(sub, 'bldng_cnt',10)['fr_rate'])

plt.title('bldng_cnt',size=20)

plt.subplot(2,3,3)

sns.barplot(x = percentile(sub, 'bldng_ar',10).index, y = percentile(sub, 'bldng_ar',10)['fr_rate'])

plt.title('bldng_ar',size=20)

plt.subplot(2,3,4)

sns.barplot(x = percentile(sub, 'fr_wthr_fclt_in_100m',10).index, y = percentile(sub, 'fr_wthr_fclt_in_100m',10)['fr_rate'])

plt.title('fr_wthr_fclt_in_100m',size=20)

plt.subplot(2,3,5)

sns.barplot(x = percentile(sub, 'lnd_ar',10).index, y = percentile(sub, 'lnd_ar',10)['fr_rate'])

plt.title('lnd_ar',size=20)

plt.subplot(2,3,6)

sns.barplot(x = percentile(sub, 'fr_sttn_dstnc',10).index, y = percentile(sub, 'fr_sttn_dstnc',10)['fr_rate'])

plt.title('fr_sttn_dstnc',size=20)

plt.ylim(0,1)

plt.show()
# bldng_cnt

main['bldng_cnt'] = sub['bldng_cnt']



# cctv_in_100m

main['cctv_in_100m'] = sub['cctv_in_100m']



# fr_wthr_fclt_in_100m

main['fr_wthr_fclt_in_100m'] = sub['fr_wthr_fclt_in_100m']



# ahsm_dstnc

main['ahsm_dstnc'] = sub['ahsm_dstnc']



# sft_emrgnc_bll_dstnc

main['sft_emrgnc_bll_dstnc'] = sub['sft_emrgnc_bll_dstnc'].apply(lambda x : np.log10(x))



# tbc_rtl_str_dstnc

main['tbc_rtl_str_dstnc'] = sub['tbc_rtl_str_dstnc']



# cctv_dstnc

main['cctv_dstnc'] = sub['cctv_dstnc']



# lw 관련 컬럼들은 axis=0 기준으로 가장 큰 결과값을 추출

lw = sub2[sub2.columns[sub2.columns.map(lambda x : x.startswith('lw'))]]

lw = lw.max(axis=1)

lw = lw.fillna(6)

lw = lw.apply(lambda x : round(x))

main['lw'] = lw



# trgt_crtr

main['trgt_crtr'] = sub['trgt_crtr']



# dngrs_thng_yn

main['dngrs_thng_yn'] = sub['dngrs_thng_yn']



# blk_dngrs_thng_mnfctr_yn

main['blk_dngrs_thng_mnfctr_yn'] = sub['blk_dngrs_thng_mnfctr_yn']



# lnd_ar

main['lnd_ar'] = sub['lnd_ar']



# rgnl_ar_nm

main['rgnl_ar_nm'] = sub['rgnl_ar_nm']



# rgnl_ar_nm2

main['rgnl_ar_nm2'] = sub['rgnl_ar_nm2']



# jmk

main['jmk'] = sub['jmk']



# bldng_us / bldng_clssfctn / bldng_archtctr

# 3변수 None 값들은 비슷하게 분포되어있으며, None의 화재확률은 0%가까이 수렴// 결측치 처리 안함



main['bldng_us'] = sub['bldng_us']

main['bldng_us_clssfctn'] = sub['bldng_us_clssfctn']

main['bldng_archtctr'] = sub['bldng_archtctr']



# rd_sd_nm

main['rd_sd_nm'] = sub['rd_sd_nm']



# ttl_ar

main['ttl_ar'] = sub['ttl_ar']



# ttl_grnd/dwn_flr

main['ttl_grnd_flr'] = sub['ttl_grnd_flr'].fillna(0)

main['ttl_dwn_flr'] = sub['ttl_dwn_flr'].fillna(0)



# mlt_us_yn

main['mlt_us_yn'] = sub['mlt_us_yn']



# bldng_ar

main['bldng_ar'] = sub['bldng_ar']

main['fr_yn'][main.bldng_ar==0] = 'N'



# lnd_us_sttn_nm

main['lnd_us_sttn_nm'] = sub['lnd_us_sttn_nm']



# bldng_cnt_in_50m

main['bldng_cnt_in_50m'] = sub['bldng_cnt_in_50m'].apply(lambda x : 10 if x > 10 else x)



#fr_wthr_fclt_dstnc

main['fr_wthr_fclt_dstnc'] = sub['fr_wthr_fclt_dstnc']



# no_tbc_zn_dstnc

main['no_tbc_zn_dstnc'] = sub['no_tbc_zn_dstnc']



# fr_sttn_dstnc

main['fr_sttn_dstnc'] = sub['fr_sttn_dstnc']



# prcpttn

main['prcpttn'] = sub['prcpttn'].fillna(-1)

main['prcpttn'] = main['prcpttn'].apply(lambda x : 10 if x > 10 else x)
## AWS: (현데이터 split후, 랜덤샘플링 검증을 통해 검증) 직관적 결측치 처리(0.85)보다 낮은 수치기록(0.55) -> 미사용



## Datawig Simple Imputer(AWS)



# from datawig.simple_imputer import SimpleImputer

# from datawig.column_encoders import *

# from datawig.mxnet_input_symbols import *



# # set up imputer model  "fr_wthr_fclt_in_100m"   "rgnl_ar_nm" "rgnl_ar_nm2" 

# data_encoder_cols = [CategoricalEncoder("fr_wthr_fclt_in_100m"), CategoricalEncoder("rgnl_ar_nm2"),  CategoricalEncoder("rgnl_ar_nm"), CategoricalEncoder('emd_nm'),  CategoricalEncoder("fr_mn_cnt"), CategoricalEncoder("bldng_us"), CategoricalEncoder('bldng_cnt'), CategoricalEncoder('bldng_cnt_in_50m'), CategoricalEncoder("rd_sd_nm"), CategoricalEncoder("lnd_us_sttn_nm"),]

# label_encoder_cols = [ CategoricalEncoder("bldng_us_clssfctn")]

# data_featurizer_cols = [EmbeddingFeaturizer("fr_wthr_fclt_in_100m"), EmbeddingFeaturizer("rgnl_ar_nm2"),  EmbeddingFeaturizer("rgnl_ar_nm"), EmbeddingFeaturizer("emd_nm"),EmbeddingFeaturizer("lnd_us_sttn_nm"), EmbeddingFeaturizer("bldng_us"), EmbeddingFeaturizer("fr_mn_cnt"), EmbeddingFeaturizer('bldng_cnt'), EmbeddingFeaturizer('bldng_cnt_in_50m'),  EmbeddingFeaturizer("rd_sd_nm")]



# imputer = Imputer(

#     data_featurizers=data_featurizer_cols,

#     label_encoders=label_encoder_cols,

#     data_encoders=data_encoder_cols,

#     output_path='imputer_model'

# )

# imputer.fit(patience=10, num_epochs=150, calibrate = True, train_df = origin2[origin2.bldng_us_clssfctn.notnull()][["fr_mn_cnt","rgnl_ar_nm","rgnl_ar_nm2","fr_wthr_fclt_in_100m","rd_sd_nm", "lnd_us_sttn_nm", "emd_nm", "bldng_cnt", "bldng_cnt_in_50m", "bldng_us", "bldng_us_clssfctn", "bldng_archtctr"]])
cols = main.select_dtypes('object').columns



main_ = main.copy()



for i in cols:

    main_[i] = pd.Categorical(main_[i]).codes

    main_[i] = main_[i].astype('int8')



main2 = main_.copy()



train_sub = main2[main2.set == 1]

test_sub = main2[main2.set == 2]

val_sub = main2[main2.set == 3]



train_sub['fr_yn'] = LabelEncoder().fit_transform(train_sub['fr_yn'])

test_sub['fr_yn'] = LabelEncoder().fit_transform(test_sub['fr_yn'])



data = pd.concat([train_sub,test_sub])



x_data_df = data.drop(columns=['id','set','fr_yn'])

y_data = data['fr_yn']

x_train_df = train_sub.drop(columns=['id','set','fr_yn'])

y_train = train_sub['fr_yn']

x_val_df = test_sub.drop(columns=['id','set','fr_yn'])

y_val = test_sub['fr_yn']
def rmse_expm1(true,pred):

    return f1_score(true,pred)



def evaluate(x_train_df, y_train,x_val_df,y_val):

    model = LGBMClassifier(objective='binary')

    model.fit(x_train_df, y_train, eval_set=[(x_val_df, y_val)],verbose=False)

    val_pred = model.predict(x_val_df)

    score = rmse_expm1(y_val,val_pred)

    return score



def rfe(data, method, ratio=1, min_feats=5):

    

    feats = data.columns.drop(['id','set','fr_yn']).tolist()

    

    archive = pd.DataFrame(columns=['model', 'n_feats', 'feats', 'score'])

    

    x_train = data[data.set ==1].drop('fr_yn',1)

    x_val = data[data.set ==2].drop('fr_yn',1)

    y_train = data[data.set ==1]['fr_yn']

    y_val = data[data.set ==2]['fr_yn']



    data_df = data.drop(columns=['id','set'])

    x_train_df = x_train.drop(columns=['id','set'])

    x_val_df = x_val.drop(columns=['id','set'])

    

    while True: 

        

        model = LGBMClassifier(objective='binary')

 

        model.fit(x_train_df[feats], y_train, eval_set=[(x_val_df[feats], y_val)],verbose=False)

        val_pred = model.predict(x_val_df[feats])

        score = rmse_expm1(y_val,val_pred)

        n_feats = len(feats)

        print(n_feats, score)

        archive = archive.append({'model': model, 'n_feats': n_feats, 'feats': feats, 'score': score}, ignore_index=True)

        

        if method == 'basic':

            feat_imp = pd.Series(model.feature_importances_, index=feats).sort_values(ascending=False)

        elif method == 'perm':

            perm = PermutationImportance(model, random_state=0).fit(x_val[feats], y_val)

            feat_imp = pd.Series(perm.feature_importances_, index=feats).sort_values(ascending=False)

        elif method == 'shap':

            explainer = shap.TreeExplainer(model)

            shap_values = explainer.shap_values(x_data_df[feats])

            feat_imp = pd.Series(np.abs(shap_values).mean(axis=0).mean(axis=0), index=feats).sort_values(ascending=False)

        next_n_feats = int(n_feats - ratio)

        

        if next_n_feats < min_feats:

            break

        else:

            feats = feat_imp.iloc[:next_n_feats].index.tolist()

            

    return archive



feats = [col for col in data.columns if col not in ['fr_yn','id','set']]

len(feats)
# 기존 f1_Score 확인



model = LGBMClassifier(objective='binary')

model.fit(x_train_df, y_train, eval_set=[(x_val_df, y_val)], verbose=False)

val_pred = model.predict(x_val_df)

score = rmse_expm1(y_val,val_pred)

score
# 1) Basic Feature Importance

# RFE



basic_archive = rfe(data,'basic')
# One-Shot



feat_imp = pd.Series(model.feature_importances_, index=feats).sort_values(ascending=False)

for i in range(5, 43,1):

    print(i, evaluate(x_train_df[feat_imp.iloc[:i].index], y_train,x_val_df[feat_imp.iloc[:i].index],y_val))
# 2) Permutation Importance

# RFE



perm_archive = rfe(data,'perm')
perm = PermutationImportance(model).fit(x_val_df, y_val)

perm_feat_imp = pd.Series(perm.feature_importances_, index=feats).sort_values(ascending=False)

eli5.show_weights(perm)
for i in range(5, 43,1):

    print(i, evaluate(x_train_df[perm_feat_imp.iloc[:i].index], y_train,x_val_df[perm_feat_imp.iloc[:i].index],y_val))
# 3) Shap

# RFE



shap_archive = rfe(data,'shap')
explainer = shap.TreeExplainer(model)

shap_values = explainer.shap_values(x_data_df)

shap_feat_imp = pd.Series(np.abs(shap_values).mean(axis=0).mean(axis=0), index=feats).sort_values(ascending=False)

shap.summary_plot(shap_values, x_data_df.columns)
# one-shot



for i in range(5, 43,1):

    print(i, evaluate(x_train_df[shap_feat_imp.iloc[:i].index], y_train,x_val_df[shap_feat_imp.iloc[:i].index],y_val))
# 3개의 평균을 이용하여 상위 20개 확인



feat_imp_archive = pd.DataFrame(index=feats, columns=['basic', 'perm', 'shap', 'mean'])

feat_imp_archive['basic'] = feat_imp.rank(ascending=False)

feat_imp_archive['perm'] = perm_feat_imp.rank(ascending=False)

feat_imp_archive['shap'] = shap_feat_imp.rank(ascending=False)

feat_imp_archive['mean'] = feat_imp_archive[['basic', 'perm', 'shap']].mean(axis=1)

feat_imp_archive = feat_imp_archive.sort_values('mean')

feat_imp_archive[feat_imp_archive['mean']>30].plot(kind='bar', figsize=(20, 10), title='feature importance rankings');
for i in range(5, 43):

    print(i, evaluate(x_train_df[feat_imp_archive.iloc[:i].index], y_train,x_val_df[feat_imp_archive.iloc[:i].index],y_val))
basic_cols = basic_archive[basic_archive.score == basic_archive.score.max()].feats.tolist()[0]

perm_cols = perm_archive[perm_archive.score == perm_archive.score.max()].feats.tolist()[0]

shap_cols = shap_archive[shap_archive.score == shap_archive.score.max()].feats.tolist()[0]
main_ = main.copy()



# category_cols

cols = ['rgnl_ar_nm','jmk','bldng_archtctr','bldng_us','rd_sd_nm','bldng_us_clssfctn',

       'hours_3','season','lnd_us_sttn_nm','rgnl_ar_nm2','fr_months','grp_ele'

        ,'athz_yr_deca']





# drop_cols

drop_cols = ['emd_nm2','ttl_dwn_flr','mlt_us_yn','ttl_grnd_flr','weekend',

            'prcpttn','fr_years','bldng_cnt',

            'fr_hours','cli_sum_0','cctv_in_100m','ahsm_dstnc','sft_emrgnc_bll_dstnc',

            'tbc_rtl_str_dstnc','cctv_dstnc','lw','trgt_crtr','fr_wthr_fclt_in_100m',

             'dngrs_thng_yn', 'blk_dngrs_thng_mnfctr_yn','lnd_ar']
for i in cols:

    main_[i] = pd.Categorical(main_[i]).codes

    main_[i] = main_[i].astype('category')



main2 = main_.drop(drop_cols,1)
# 2-1) LightGBM



train_sub = main2[main2.set == 1]

test_sub = main2[main2.set == 2]

val_sub = main2[main2.set == 3]



train_sub.drop(columns = ['id','set'], inplace = True)

test_sub.drop(columns = ['id','set'], inplace = True)

val_sub.drop(columns = ['id','set'], inplace = True)



train_X = train_sub.drop('fr_yn',1)

train_y = train_sub['fr_yn']

test_X = test_sub.drop('fr_yn',1)

test_y = test_sub['fr_yn']

val_X = val_sub.drop('fr_yn',1)
lgbm = LGBMClassifier(num_iterations=92, min_data_in_leaf=30,min_sum_hessian_in_leaf=0.47)



lgbm.fit(train_X,train_y)

pred = lgbm.predict(test_X)

pred_sub = lgbm.predict(test_X)

test_y = sub['fr_yn'][sub.set == 2]

test_y = np.array(test_y)



le = LabelEncoder()

test_y1 = le.fit_transform(test_y)

pred_y = le.fit_transform(pred)

print('accraucy: ',round(accuracy_score(test_y1,pred_y),5))

print('precision: ',round(precision_score(test_y1,pred_y),5))

print('recall: ',round(recall_score(test_y1,pred_y),5))

print('f1: ',round(f1_score(test_y1,pred_y),5))
# ### 2-1) XGBoost



# # One-Hot



# for i in cols:

#     main_[i] = pd.Categorical(main_[i]).codes

#     main_[i] = main_[i].astype('str')



# main2 = main_.drop(drop_cols,1)



# main2 = pd.concat([main2, pd.get_dummies(main2[cols],

#                                              prefix=cols)], axis=1, join='inner')



# train_sub = main2[main2.set == 1]

# test_sub = main2[main2.set == 2]

# val_sub = main2[main2.set == 3]



# train_sub.drop(columns = cols, inplace = True)

# test_sub.drop(columns = cols, inplace = True)

# val_sub.drop(columns = cols, inplace = True)



# train_sub.drop(columns = ['id','set'], inplace = True)

# test_sub.drop(columns = ['id','set'], inplace = True)

# val_sub.drop(columns = ['id','set'], inplace = True)



# train_X = train_sub.drop('fr_yn',1)

# train_y = train_sub['fr_yn']

# test_X = test_sub.drop('fr_yn',1)

# test_y = test_sub['fr_yn']

# val_X = val_sub.drop('fr_yn',1)



# xgbc = XGBClassifier(scale_pos_weight=1.1)

# xgbc.fit(train_X, train_y)





# print(round(xgbc.score(test_X,test_y),5))



# pred = xgbc.predict(test_X)

# pred



# from sklearn.preprocessing import LabelEncoder

# le = LabelEncoder()

# test_y1 = le.fit_transform(test_y)



# pred_y1 = le.fit_transform(pred)





# print(round(f1_score(test_y1,pred_y1),5))



# gb = GradientBoostingClassifier()

# gb.fit(train_X, train_y)





# print(round(gb.score(test_X,test_y),5))



# pred = gb.predict(test_X)



# from sklearn.preprocessing import LabelEncoder

# le = LabelEncoder()

# test_y1 = le.fit_transform(test_y)



# pred_y3 = le.fit_transform(pred)





# print(round(f1_score(test_y1,pred_y3),5))
feature_imp = pd.DataFrame(sorted(zip(lgbm.feature_importances_,train_X.columns)), columns=['Value','Feature'])



plt.figure(figsize=(15, 10))

sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value", ascending=False))

plt.title('LightGBM Features')

plt.tight_layout()

plt.show()
x_train=train_X

y_train = le.fit_transform(train_y)

x_test = test_X

y_test = le.fit_transform(test_y)



lgb_train = lgb.Dataset(x_train, y_train)

lgb_test = lgb.Dataset(x_test, y_test, reference=lgb_train)



params = {

    'num_leaves': 5,

    'metric': ('auc', 'logloss'),

    'verbose': 0

    

     }



evals_result = {} 



gbm = lgb.train(params,

                lgb_train,

                num_boost_round=100,

                valid_sets=[lgb_train, lgb_test],

                evals_result=evals_result,

                verbose_eval=10)



ax = lgb.plot_metric(evals_result, metric='auc')

lgb.plot_tree(gbm, tree_index=1, figsize=(20,8), show_info=['split_gain'],) # Draw Tree

plt.show()
## 6) Classification Report



#LGBM" 0.518 -> 0.543 (Leaderboard)

#XGB" 0.51519 -> 0.532 (Leaderboard)

#0.51447" 0.51447 -> 0.529 (Leaderboard)



print(classification_report(test_y1,pred_y))
lgbm_model = LGBMClassifier(num_iterations=92, min_data_in_leaf=30,min_sum_hessian_in_leaf=0.47)

lgbm_model.fit(train_X,train_y)

lgbm_pred_val = lgbm_model.predict(val_X)

val_df = pd.DataFrame(lgbm_pred_val)

val_df.to_csv("final_predict.csv", encoding="utf-8-sig", index=False)