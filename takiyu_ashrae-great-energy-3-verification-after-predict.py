# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

import os

import lightgbm as lgbm

from sklearn.preprocessing import LabelEncoder

import gc

import warnings



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.

#np.seterr(divide='ignore', invalid='ignore')

#numpy.seterr(all='raise')
def reduce_mem_usage(df):

    #code from

    #https://www.kaggle.com/rohanrao/ashrae-half-and-half

    """ iterate through all the columns of a dataframe and modify the data type

        to reduce memory usage.        

    """

    start_mem = df.memory_usage().sum() / 1024**2

    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))

    

    for col in df.columns:

        col_type = df[col].dtype

        

        if col_type != object:

            c_min = df[col].min()

            c_max = df[col].max()

            if str(col_type)[:3] == 'int':

                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:

                    df[col] = df[col].astype(np.int8)

                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:

                    df[col] = df[col].astype(np.int16)

                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:

                    df[col] = df[col].astype(np.int32)

                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:

                    df[col] = df[col].astype(np.int64)  

            else:

                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:

                    df[col] = df[col].astype(np.float16)

                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:

                    df[col] = df[col].astype(np.float32)

                else:

                    df[col] = df[col].astype(np.float64)

        else:

            df[col] = df[col].astype('category')



    end_mem = df.memory_usage().sum() / 1024**2

    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))

    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))

    

    return df
#displayの行数と列数を増やす

#display Increase the number of rows and columns in 

pd.set_option('display.max_columns', 100)

pd.set_option('display.max_rows', 200)

warnings.simplefilter('ignore')

#csvを読み込む

#read csv

building = pd.read_csv('../input/ashrae-energy-prediction/building_metadata.csv')

weather_train = pd.read_csv('../input/ashrae-energy-prediction/weather_train.csv')

weather_test = pd.read_csv('../input/ashrae-energy-prediction/weather_test.csv')

train = pd.read_csv('../input/ashrae-energy-prediction/train.csv') 

test = pd.read_csv('../input/ashrae-energy-prediction/test.csv')



#ここはテストデータから作成した予測データを読み込む

#Load forecast data

#submission_test = pd.read_csv('../input/ashrae-half-and-half/submission_13.csv')

submission_test = pd.read_csv('../input/ashrae-test-leak-validation-and-more/submission.csv')



le = LabelEncoder()

building.primary_use = le.fit_transform(building.primary_use)

def boundary_show_train_test(train,test):

    """

    Training data and test data (test data has already been added with the predicted objective variable)

    :param train:

    :param test:

    """    

    

    datas = train.tail(100) 

    datas = datas.append(test.head(100))

    

    datas = datas.reset_index()

    

    return datas



print("----building------------------------------------------")

display(building.head(5))

display(building.describe().T)

display(building.dtypes)

print(building.shape)



print("----weather_train------------------------------------------")

display(weather_train.head(5))

display(weather_train.describe().T)

display(weather_train.dtypes)

print(weather_train.shape)



print("----train------------------------------------------")

display(train.head(5))

display(train.describe().T)

display(train.dtypes)

print(train.shape)

#基礎的な統計量を出す

print("building" + "-" * 50)

display(building.describe().T)

print("weather_train" + "-" * 50)

display(weather_train.describe().T)

print("train" + "-" * 50)

display(train.describe().T)



#欠損を出す

print("train null" + "-" * 50)

display(train.isnull().sum())



print("weather_train null" + "-" * 50)

display(weather_train.isnull().sum())



print("building null" + "-" * 50)

display(building.isnull().sum())
####################################

#Data merge

####################################



#訓練データとビル情報

train = train.merge(building, on = 'building_id', how = 'left')



#訓練データと気象データ

train = train.merge(weather_train, on = ['site_id', 'timestamp'], how = 'left')



#テストデータとビル情報

test = test.merge(building, on = 'building_id', how = 'left')



#テストデータと気象データ

test = test.merge(weather_test, on = ['site_id', 'timestamp'], how = 'left')



#テストデータと予測データ

test = test.merge(submission_test, on = 'row_id', how = 'left')



#reduce_mem_usage

train = reduce_mem_usage(train)

test = reduce_mem_usage(test)
del weather_train, weather_test,building, submission_test
#時間によって変化があるため時間の処理が必要

train["timestamp"] = pd.to_datetime(train["timestamp"])

train["hour"] = np.uint8(train["timestamp"].dt.hour)

train["day"] = np.uint8(train["timestamp"].dt.day)

train["weekday_name"] = train["timestamp"].dt.weekday_name 

train["weekday"] = np.uint8(train["timestamp"].dt.weekday)

train["month"] = np.uint8(train["timestamp"].dt.month)



test["timestamp"] = pd.to_datetime(test["timestamp"])

test["hour"] = np.uint8(test["timestamp"].dt.hour)

test["day"] = np.uint8(test["timestamp"].dt.day)

test["weekday_name"] = test["timestamp"].dt.weekday_name 

test["weekday"] = np.uint8(test["timestamp"].dt.weekday)

test["month"] = np.uint8(test["timestamp"].dt.month)



#meter_readingをログに入れる

train['meter_reading_log'] =  np.log1p(train['meter_reading'])

train['square_feet_log'] =  np.log1p(train['square_feet'])



#ビルごと、meterごと、月ごとの集計を出す

data = train.groupby(['building_id','meter','month']).sum()

data.to_csv("building_merter_month_meter_reading_sum.csv")



#ビルごと、meterごと、月ごとの集計を出す

data = train.groupby(['building_id','meter','month']).mean()

data.to_csv("building_merter_month__meter_reading_mean.csv")





display(data)

#check_validation_train_test(train,test)
def building_plot(building_id,train, test,comment="" ):

    """

    building data plot

    :param building_id:

    :param train:

    :param test:

    :param comment:

    """



    plt.rcParams['figure.figsize'] = (19,11)

    plt.title("building_id_%d  %s" % (building_id,comment ))

    

    ###########################################

    #訓練とテストデータをbuilding_idでクエリする

    ###########################################

    

    #訓練データ側

    query_str = ('building_id == %s' % str(building_id) )

    temp_df_train = train.query(query_str)

    temp_df_train['meter_reading'] = np.log1p(temp_df_train['meter_reading'])

    temp_df_train = temp_df_train.reset_index()

    #日ごとのデータに変換する

    group = temp_df_train.groupby([temp_df_train['timestamp'].dt.year,temp_df_train['timestamp'].dt.month,temp_df_train['timestamp'].dt.day, 'meter' ]).groups

    #print(group)

    

    temp_df_train2 = pd.DataFrame(columns=["meter","timestamp","meter_reading"])

    

    cnt = 0

    for k ,v in group.items():

        meter = k[3]

        year =k[0]

        month =k[1]

        day =k[2]



        report_data = temp_df_train.iloc[ v,:  ]

        meter_reading_mean = report_data["meter_reading"].mean()

        time_day = pd.to_datetime(('%d-%d-%d') % (year, month,day))

        tmp_se = pd.Series([meter,

                            time_day,

                            meter_reading_mean,

                        ],index=temp_df_train2.columns, name=str(cnt)) 

        cnt +=1

        temp_df_train2 = temp_df_train2.append(tmp_se)

   



    #テストデータ側

    temp_df_test = test.query(query_str)

    temp_df_test['meter_reading'] = np.log1p(temp_df_test['meter_reading'])

    temp_df_test = temp_df_test.reset_index()

    group = temp_df_test.groupby([temp_df_test['timestamp'].dt.year,temp_df_test['timestamp'].dt.month,temp_df_test['timestamp'].dt.day, 'meter' ]).groups

    temp_df_train3 = pd.DataFrame(columns=["meter","timestamp","meter_reading"])

    

    cnt = 0

    for k ,v in group.items():

        meter = k[3]

        year =k[0]

        month =k[1]

        day =k[2]



        report_data = temp_df_test.iloc[ v,:  ]

        meter_reading_mean = report_data["meter_reading"].mean()

        time_day = pd.to_datetime(('%d-%d-%d') % (year, month,day))

        tmp_se = pd.Series([meter,

                            time_day,

                            meter_reading_mean,

                        ],index=temp_df_train2.columns, name=str(cnt)) 

        cnt +=1

        temp_df_train3 = temp_df_train3.append(tmp_se)





    

    

    

    #境界線部分の作成

    testdata = boundary_show_train_test(temp_df_train,temp_df_test)

    #display(testdata)

    testdata.to_csv(("building_id_%s.csv" % str(building_id)),encoding = 'utf-8-sig')    

    

    #グラフを書く(訓練側)

    alpha = 0.5

    ax = sns.lineplot(data = temp_df_train2.query("meter == 0"), x = 'timestamp', y = 'meter_reading', color = 'r',alpha=alpha,label = "merter0")

    ax = sns.lineplot(data = temp_df_train2.query("meter == 1"), x = 'timestamp', y = 'meter_reading', color = 'g',alpha=alpha,label = "merter1")

    ax = sns.lineplot(data = temp_df_train2.query("meter == 2"), x = 'timestamp', y = 'meter_reading', color = 'b',alpha=alpha,label = "merter2")

    ax = sns.lineplot(data = temp_df_train2.query("meter == 3"), x = 'timestamp', y = 'meter_reading', color = 'c',alpha=alpha,label = "merter3")

      

    

    #グラフを書く(テスト側)

    ax = sns.lineplot(data = temp_df_train3.query("meter == 0"), x = 'timestamp', y = 'meter_reading', color = 'r',alpha=alpha)

    ax = sns.lineplot(data = temp_df_train3.query("meter == 1"), x = 'timestamp', y = 'meter_reading', color = 'g',alpha=alpha)

    ax = sns.lineplot(data = temp_df_train3.query("meter == 2"), x = 'timestamp', y = 'meter_reading', color = 'b',alpha=alpha)

    ax = sns.lineplot(data = temp_df_train3.query("meter == 3"), x = 'timestamp', y = 'meter_reading', color = 'c',alpha=alpha)

    

    plt.ylabel('Log Meter Reading')



    #訓練データとテストデータの境界線を書く

    plt.axvline(x=pd.to_datetime("2017-01-01"), color='b')

    plt.text(pd.to_datetime("2017-01-01"), ax.get_ylim()[1], "Prediction Begin",

             horizontalalignment='center',

             verticalalignment='center',

             color='b',

             bbox=dict(facecolor='white', alpha=0.9))    

    

    plt.show()
def make_correlation_matrix_train(train_df):



    #カラムの選択

    train_df = train_df.reset_index()

    train_df = train_df.loc[:, ["meter_reading_log",'floor_count','year_built',"air_temperature","cloud_coverage","dew_temperature",'precip_depth_1_hr','wind_direction','square_feet','sea_level_pressure']]

    df_corr = train_df.corr()

    display(df_corr)

    sns.heatmap(df_corr, vmax=1, vmin=-1, center=0)

 
#訓練データで相関図を作る

#display(train.head(5))

make_correlation_matrix_train(train )
#building_id:29 meter:1 diff:6.173086  さらにmeter=0が壊滅的にやばい

#building_plot(29,train ,test,comment="building_id:29 meter:1 diff:6.173086 resample before")



#research

#3month_meter_reading 100 ander(site_id = 0 nasi(104nasi))

building = pd.read_csv('/kaggle/input/3month-meter-reading-zero/3month_meter_reading_zero.csv')



building = building["building_id"].unique()



for i in building:

    building_plot(i,train ,test,comment="3month-meter-reading 100 ander")









"""

#リサンプリングを試してみる

#building_id = 29 meter=0

bull29 = train[(train['building_id'] ==  29) &  (train['meter'] ==  0)]

display(bull29)

#serial_num = pd.RangeIndex(start=1, stop=len(bull29.index) + 1, step=1)

#bull29['No'] = serial_num



bull29.set_index("timestamp")

dateTimeIndex = pd.DatetimeIndex(bull29['timestamp'])

bull29.index = dateTimeIndex

bull29 = bull29.resample('H').mean()

display(bull29)

bull29 = bull29.reset_index()

#trainデータからリサンプリング前の情報を削除

train = train[ ~ ((train["building_id"] == 29)  & (train.meter == 0) ) ]

train.append(bull29)

train.reset_index()

"""
"""

display(train[ train['building_id'] == 29 & (train.meter == 0) ])

building_plot(29,train ,test,comment="building_id:29 meter:1 diff:6.173086 resample after")

"""
building_plot(1018,train ,test)

building_plot(1013,train ,test)

building_plot(740,train ,test)

building_plot(1022,train ,test)

building_plot(287,train ,test)

building_plot(279,train ,test)

building_plot(252,train ,test)



#ここには、訓練データとテストデータ(予測結果)において比較したいbuilding idを記載する

#Here, enter the building id you want to compare between training data and test data



#最も０メータが多いビル

building_plot(954,train ,test,comment="most 0 meter count")



#２番目に０メータが多いビル

building_plot(799,train ,test,comment="2nd 0 meter count")



#3番目に０メータが多いビル

building_plot(1232,train ,test,comment="3rd 0 meter count")



#4番目に０メータが多いビル

building_plot(1022,train ,test,comment="4th 0 meter count")



#5番目に０メータが多いビル

building_plot(1324,train ,test,comment="5th 0 meter count")





#分散値が最も低いmeter_reading「4.55E-12」

building_plot(740,train ,test,comment="most low dispersion")



#標準偏差が低いmeter_reading「0.000350764 meter=1」

building_plot(1018,train ,test,comment="meter_reading 0.000350764 meter=1")





#分散値が２番めに低いmeter_reading「0.007100055」

building_plot(636,train ,test,comment="2nd low dispersion")



#分散値が3番めに低いmeter_reading「0.014564584」

building_plot(637,train ,test,comment="3rd low dispersion")



#分散値が4番めに低いmeter_reading「0.017347898」

building_plot(846,train ,test,comment="4th low dispersion")



#分散の中間(ビルを1000個ならべたときの中間の位置)「650.9789」

building_plot(1082,train ,test,comment="middium 1 dispersion")



#分散の中間(ビルを1000個ならべたときの中間の位置)「654.3975」

building_plot(733,train ,test,comment="middium 2 dispersion")





#分散が最も多いmeter_reading「23370951000000」

building_plot(1099,train ,test,comment="most value dispersion")



#分散が２番めに多いmeter_reading「13617740000」

building_plot(778,train ,test,comment="2nd value dispersion")







#building_id:60 meter:1 diff:6.149655

building_plot(60,train ,test,comment="building_id:60 meter:1 diff:6.149655")



#building_id:803 meter:0 diff:7.762360

building_plot(803,train ,test,comment="building_id:803 meter:0 diff:7.762360")



#building_id:993 meter:0 diff:7.081440

building_plot(993,train ,test,comment="building_id:993 meter:0 diff:7.081440")



#building_id:993 meter:1 diff:7.570996

building_plot(993,train ,test,comment="building_id:993 meter:1 diff:7.570996")



#12/31日がエネルギー使用量０のビル

building_plot(28,train ,test,comment="12/31 meter_reading is 0")

building_plot(43,train ,test,comment="12/31 meter_reading is 0")

building_plot(103,train ,test,comment="12/31 meter_reading is 0")

building_plot(191,train ,test,comment="12/31 meter_reading is 0")

building_plot(263,train ,test,comment="12/31 meter_reading is 0")



#all diff to csv

def boundary_validation_train_test(train,test):

    """

    boundary_validation_train_test

    :param train:

    :param test:

    :return diff value

    """    



    #３日分取得 訓練データ

    datas = train.tail(72)

    if len(datas) < 1:

        return -1

    datas["meter_reading"] = np.log1p(datas["meter_reading"])

    

    train_mean = datas['meter_reading'].mean()

    

    

    #３日分取得 テストデータ

    datas = test.head(72)

    if len(datas) < 1:

        return -1

    datas["meter_reading"] = np.log1p(datas["meter_reading"])

    test_mean = datas['meter_reading'].mean()   

    

    

    

    #datas = datas.append(test.head(72))

    #datas = datas.reset_index()

    

    #if(len(datas) < 74 ):

    #    return -1

    

    #meter_reading diff check 

    #np.log1p(temp_df_train['meter_reading'])

    #display(datas)

    #v_train = np.log1p(datas.loc[0]["meter_reading"])

    #v_test = np.log1p(datas.loc[1]["meter_reading"])

    

    

    return abs(train_mean - test_mean)

def check_validation_train_test(train,test):

    #building id毎に訓練データと予測データの境界でmeter_readingが著しくずれているところを検出する



    #訓練データ毎の繰り返し

    building_ids = train['building_id'].unique()

    building_ids.sort()



    #boundary = 2.0 #しきい値の設定



    list_df = pd.DataFrame(

        columns=[

            'building_id',

            'meter',

            'diff',

            'count',

            'train_mean',

            'train_std',

            'train_min',

            'train_max'

            

            ])    

    

    error_building_ids = []

    i = 0

    for building_id in building_ids:

        #print(building_id)

        

        #build idで抽出

        query_str = ('building_id == %s' % str(building_id) )

        temp_df_build_train = train.query(query_str)

        temp_df_build_test = test.query(query_str)

        

        for meter in range(0,4):

            #print("building_id" + str(building_id) + "meter:" + str(meter))

            query_str = ('meter == %s' % str(meter) )

            temp_df_train = temp_df_build_train.query(query_str)

            temp_df_test = temp_df_build_test.query(query_str)

            diff = boundary_validation_train_test(temp_df_train,temp_df_test)

            

            if(diff == -1):

                continue



            temp_df_train["meter_reading"] = np.log1p(temp_df_train["meter_reading"])

            tmp_se = pd.Series([building_id,

                    meter,

                    diff,

                    temp_df_train.describe().at['count', 'meter_reading'],

                    temp_df_train.describe().at['mean', 'meter_reading'],

                    temp_df_train.describe().at['std', 'meter_reading'],

                    temp_df_train.describe().at['min', 'meter_reading'],

                    temp_df_train.describe().at['max', 'meter_reading']

                                

                    ],index=list_df.columns, name=str(i))

            

            list_df = list_df.append(tmp_se)

            #display(list_df)

            i += 1

                

            #if(diff > boundary):

                #訓練データとテストデータで差が開いていると判定

            #print("building_id:%d meter:%d diff:%f" %(building_id,meter ,diff))

            #error_building_ids.append(building_id)

            

            

            

    #boundary

    #list(set(error_building_ids))

    list_df.to_csv("check_validation_train_test.csv",encoding = 'utf-8-sig')    

    #このあとはbuild isの描画する処理
#building id毎に訓練データと予測データの境界でmeter_readingが著しくずれているところを検出する

check_validation_train_test(train,test)