# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
train_transaction = pd.read_csv("../input/ieee-fraud-detection/train_transaction.csv")

train_identity = pd.read_csv("../input/ieee-fraud-detection/train_identity.csv")
pd.set_option('display.max_columns', 500)

train_transaction.head(5)
# 時刻の変換

train_transaction['Transaction_dow'] = np.floor((train_transaction['TransactionDT'] / (3600 * 24) - 1) % 7)

train_transaction['Transaction_hour'] = np.floor(train_transaction['TransactionDT'] / 3600) % 24
plt.plot(train_transaction['Transaction_dow'])
plt.plot(train_transaction['Transaction_hour'][0:30000])
# train_identifyの中で欠損データが無いのを探す

na_columns = train_identity.isna().sum()

na_columns[na_columns==0]
# train_transactionの中で欠損データが無いのを探す

na_columns = train_transaction.isna().sum()

na_columns[na_columns==0]
print(na_columns[na_columns>0]/train_transaction.shape[0])
#seabornでグラフ描画　pyplotよりもお手軽

sns.distplot(train_identity['id_01'])
#列名を取り出す

transaction_columns = train_transaction.columns

#数値型の列名を取り出す

numericCols = train_transaction._get_numeric_data().columns 
#文字列のデータ列だけを取り出す

categoricalCols = list(set(transaction_columns) - set(numericCols))
#文字列データの空欄（np.nan）missingに置き換える

train_transaction[categoricalCols] = train_transaction[categoricalCols].replace({np.nan:'missing'})

#数値データの空欄を（np.nan）－１に置き換える

train_transaction[numericCols] = train_transaction[numericCols].replace({np.nan:-1})

train_transaction.head()
#詐欺データの割合をみる　Fraud:詐欺

sns.countplot(train_transaction['isFraud'])
#使われたカードの支払い方法がcredit(後から引き落とし)かdebit（すぐに引き落とし）

sns.countplot(train_transaction['card6'])
#カード支払い方法と詐欺件数の対応

cardTypes = {'credit','debit','debit or credit','charge card'}

for i,i_card in enumerate(cardTypes):

    #eval　文字列を実行文とあつかう

    cardData = eval('train_transaction.loc[train_transaction["card6"]=="'+i_card+'"]')

    plt.figure(i)

    sns.countplot(cardData['isFraud']).set_title(i_card)
#カード種別

sns.countplot(train_transaction['card4'])
#カードの種類と詐欺件数の関係

cardTypes = ['discover','mastercard','visa','american express','missing']

for i,i_card in enumerate(cardTypes):

    cardData = eval('train_transaction.loc[train_transaction["card4"]=="'+i_card+'"]')

    plt.figure(i)

    sns.countplot(cardData['isFraud']).set_title(i_card)
#詐欺だったときの払った平均金額と、詐欺じゃないときの払った平均金額

false_Fraud_Amt = np.mean(train_transaction.loc[train_transaction["isFraud"]==0]['TransactionAmt'])

true_Fraud_Amt = np.mean(train_transaction.loc[train_transaction["isFraud"]==1]['TransactionAmt'])

print(false_Fraud_Amt)

print(true_Fraud_Amt)

#詐欺だったときの払った最大金額と、詐欺じゃないときの払った最大金額

false_Fraud_Amt_max = np.max(train_transaction.loc[train_transaction["isFraud"]==0]['TransactionAmt'])

true_Fraud_Amt_max = np.max(train_transaction.loc[train_transaction["isFraud"]==1]['TransactionAmt'])

print(false_Fraud_Amt_max)

print(true_Fraud_Amt_max)
#詐欺だったときの払った最小金額と、詐欺じゃないときの払った最小金額

false_Fraud_Amt_min = np.min(train_transaction.loc[train_transaction["isFraud"]==0]['TransactionAmt'])

true_Fraud_Amt_min = np.min(train_transaction.loc[train_transaction["isFraud"]==1]['TransactionAmt'])

print(false_Fraud_Amt_min)

print(true_Fraud_Amt_min)
#カードの種類と金額の最大最小平均

maxCardData = {}

minCardData = {}

meanCardData = {}

for i,i_card in enumerate(cardTypes):

    cardData = eval('train_transaction.loc[train_transaction["card4"]=="'+i_card+'" ]')

    maxCardData[i_card] = np.max(cardData['TransactionAmt'])

    minCardData[i_card] = np.min(cardData['TransactionAmt'])

    meanCardData[i_card] = np.mean(cardData['TransactionAmt'])
print('The maximum transactions by card are:', maxCardData)

print('The minimum transactions by card are:', minCardData)

print('The average transactions by card are:', meanCardData)
#詐欺だったときの金額情報

fraudDataTransaction = train_transaction.loc[train_transaction["isFraud"]==1]

fraudDataTransaction.TransactionAmt.describe()
#詐欺じゃないときの金額情報

normalDataTransaction = train_transaction.loc[train_transaction["isFraud"]==0]

normalDataTransaction.TransactionAmt.describe()
#製品コード　uniqueで種類をみつける

prodTypes = train_transaction['ProductCD'].unique()

for i,i_prod in enumerate(prodTypes):

    productData = eval('train_transaction.loc[train_transaction["ProductCD"]=="'+i_prod+'"]')

    plt.figure(i)

    ax = sns.barplot(x="isFraud", y="isFraud", data=productData, estimator=lambda x: len(x) / len(productData) * 100)

    ax.set(ylabel="Percent")

    ax.set_title(i_prod)
del fraudDataTransaction,normalDataTransaction,productData,cardData
#identityデータの解析

identity_data_columns = train_identity.columns



#数値項目の抜き出し

numericCols = train_identity._get_numeric_data().columns

#文字列項目の抜き出し

categoricalCols = list(set(identity_data_columns) - set(numericCols))

print('The categorical columns in identity data are: ',categoricalCols)

train_identity[categoricalCols] = train_identity[categoricalCols].replace({ np.nan:'missing'})

train_identity[numericCols] = train_identity[numericCols].replace({ np.nan:-1})


#OSのタイプを変換

train_identity.loc[train_identity['id_30'].str.contains('Mac', na=False), 'id_30'] = 'mac'

train_identity.loc[train_identity['id_30'].str.contains('iOS', na=False), 'id_30'] = 'iOS'

train_identity.loc[train_identity['id_30'].str.contains('Android', na=False), 'id_30'] = 'android'

train_identity.loc[train_identity['id_30'].str.contains('Windows', na=False), 'id_30'] = 'Windows'

train_identity.loc[train_identity['id_30'].str.contains('Linux', na=False), 'id_30'] = 'Linux'
#デバイス名を変換

train_identity['device_name'] = train_identity['DeviceInfo'].str.split('/', expand=True)[0]



train_identity.loc[train_identity['device_name'].str.contains('SM', na=False), 'device_name'] = 'Samsung'

train_identity.loc[train_identity['device_name'].str.contains('SAMSUNG', na=False), 'device_name'] = 'Samsung'

train_identity.loc[train_identity['device_name'].str.contains('GT-', na=False), 'device_name'] = 'Samsung'

train_identity.loc[train_identity['device_name'].str.contains('Moto G', na=False), 'device_name'] = 'Motorola'

train_identity.loc[train_identity['device_name'].str.contains('Moto', na=False), 'device_name'] = 'Motorola'

train_identity.loc[train_identity['device_name'].str.contains('moto', na=False), 'device_name'] = 'Motorola'

train_identity.loc[train_identity['device_name'].str.contains('LG-', na=False), 'device_name'] = 'LG'

train_identity.loc[train_identity['device_name'].str.contains('rv:', na=False), 'device_name'] = 'RV'

train_identity.loc[train_identity['device_name'].str.contains('HUAWEI', na=False), 'device_name'] = 'Huawei'

train_identity.loc[train_identity['device_name'].str.contains('ALE-', na=False), 'device_name'] = 'Huawei'

train_identity.loc[train_identity['device_name'].str.contains('-L', na=False), 'device_name'] = 'Huawei'

train_identity.loc[train_identity['device_name'].str.contains('Blade', na=False), 'device_name'] = 'ZTE'

train_identity.loc[train_identity['device_name'].str.contains('BLADE', na=False), 'device_name'] = 'ZTE'

train_identity.loc[train_identity['device_name'].str.contains('Linux', na=False), 'device_name'] = 'Linux'

train_identity.loc[train_identity['device_name'].str.contains('XT', na=False), 'device_name'] = 'Sony'

train_identity.loc[train_identity['device_name'].str.contains('HTC', na=False), 'device_name'] = 'HTC'

train_identity.loc[train_identity['device_name'].str.contains('ASUS', na=False), 'device_name'] = 'Asus'



train_identity.loc[train_identity.device_name.isin(train_identity.device_name.value_counts()[train_identity.device_name.value_counts() < 200].index), 'device_name'] = "Others"
print(np.unique(train_identity['id_31']))

train_identity['id_31'] = train_identity['id_31'].str.replace('\d+', '')

print(np.unique(train_identity['id_31']))
#transaction と itentity をTransactionIDを軸に結合

raw_train_data = pd.merge(train_transaction, train_identity, on='TransactionID', how='left')

del train_identity,train_transaction
raw_train_data_columns = raw_train_data.columns

#数値項目を抜き出す

numericCols = raw_train_data._get_numeric_data().columns

#文字列項目を抜き出す

categoricalCols = list(set(raw_train_data_columns) - set(numericCols))
#欠損データの置き換え

raw_train_data[categoricalCols] = raw_train_data[categoricalCols].replace({ np.nan:'missing'})

raw_train_data[numericCols] = raw_train_data[numericCols].replace({ np.nan:-1})
# credit to @guiferviz for the memory reduction 

def memory_usage_mb(df, *args, **kwargs):

    """Dataframe memory usage in MB. """

    return df.memory_usage(*args, **kwargs).sum() / 1024**2



def reduce_memory_usage(df, deep=True, verbose=True):

    # All types that we want to change for "lighter" ones.

    # int8 and float16 are not include because we cannot reduce

    # those data types.

    # float32 is not include because float16 has too low precision.

    numeric2reduce = ["int16", "int32", "int64", "float64"]

    start_mem = 0

    if verbose:

        start_mem = memory_usage_mb(df, deep=deep)



    for col, col_type in df.dtypes.iteritems():

        best_type = None

        if col_type in numeric2reduce:

            downcast = "integer" if "int" in str(col_type) else "float"

            df[col] = pd.to_numeric(df[col], downcast=downcast)

            best_type = df[col].dtype.name

        # Log the conversion performed.

        if verbose and best_type is not None and best_type != str(col_type):

            print(f"Column '{col}' converted from {col_type} to {best_type}")

    

    if verbose:

        end_mem = memory_usage_mb(df, deep=deep)

        diff_mem = start_mem - end_mem

        percent_mem = 100 * diff_mem / start_mem

        print(f"Memory usage decreased from"

              f" {start_mem:.2f}MB to {end_mem:.2f}MB"

              f" ({diff_mem:.2f}MB, {percent_mem:.2f}% reduction)")

        

    return df
#変数を安いやつにしてメモリを節約してる

raw_train_data = reduce_memory_usage(raw_train_data, deep=True, verbose=True)
#デバイスタイプごとの詐欺率

deviceType = raw_train_data['DeviceType'].unique()

for i, i_device in enumerate(deviceType):

    deviceData = eval('raw_train_data.loc[raw_train_data["DeviceType"]=="'+i_device+'"]')

    plt.figure(i)

    sns.countplot(deviceData['isFraud']).set_title(i_device)

    
deviceType
variables = list(numericCols) #数値項目をlistに

variables.remove('isFraud')   #詐欺判定を除く

correlationMatrix = raw_train_data.loc[:,variables].corr().abs()
#ヒートマップを作る

plt.figure(figsize=(20,20))

heat = sns.heatmap(data=correlationMatrix)

plt.title('Heatmap of corr')

#数値項目の欠損データだった割合

na_vals = np.sum(raw_train_data.loc[:,variables]==-1)/ raw_train_data.shape[0]



#欠損値が少ないデータを集めてリストにする

goodNumericVars = []

for i_var in variables:

    if na_vals[i_var] < 0.85:

        goodNumericVars.append(i_var)

        

goodNumericVars.remove('TransactionDT') #経過時間

goodNumericVars.remove('TransactionID') #取引ID

corrThresh = 0.9 #相関しきい値？



#対角要素のない三角行列をつくる

upper = correlationMatrix.where(np.triu(np.ones(correlationMatrix.shape), k=1).astype(np.bool))



#相関の大きすぎる値は除く

to_drop = [column for column in upper.columns if any(upper[column] > corrThresh) ]
for i_var in to_drop:

    if i_var in goodNumericVars:

        goodNumericVars.remove(i_var)



#これで使えそうな数値項目が決まりました
del to_drop,corrThresh,upper,correlationMatrix,na_vals
variables = list(categoricalCols) #文字列項目



#欠損データの割合

na_vals = np.sum(raw_train_data.loc[:,variables] == 'missing') / raw_train_data.shape[0]



#文字列項目でつかえそうなやつ

goodCategoricalVars = []

for i_var in  variables:

    if na_vals[i_var] < 0.85:

        goodCategoricalVars.append(i_var)

featureToUse = goodNumericVars + goodCategoricalVars

train_data = raw_train_data.loc[:,featureToUse]

target_data = raw_train_data['isFraud']
featureToUse
#特徴量を新たに作る



#項目毎の支払い金額の平均

train_data['TransactionAmt_to_mean_card1'] = train_data['TransactionAmt'] / train_data.groupby(['card1'])['TransactionAmt'].transform('mean')

train_data['TransactionAmt_to_mean_card4'] = train_data['TransactionAmt'] / train_data.groupby(['card4'])['TransactionAmt'].transform('mean')

train_data['TransactionAmt_to_mean_card5'] = train_data['TransactionAmt'] / train_data.groupby(['card5'])['TransactionAmt'].transform('mean')

train_data['TransactionAmt_to_mean_ProductCD'] = train_data['TransactionAmt'] / train_data.groupby(['ProductCD'])['TransactionAmt'].transform('mean')

train_data['TransactionAmt_to_mean_addr1'] = train_data['TransactionAmt'] / train_data.groupby(['addr1'])['TransactionAmt'].transform('mean')

train_data['TransactionAmt_to_mean_id31'] = train_data['TransactionAmt'] / train_data.groupby(['id_31'])['TransactionAmt'].transform('mean')

train_data['TransactionAmt_to_mean_devicename'] = train_data['TransactionAmt'] / train_data.groupby(['device_name'])['TransactionAmt'].transform('mean')

train_data['TransactionAmt_to_mean_dow'] = train_data['TransactionAmt'] / train_data.groupby(['Transaction_dow'])['TransactionAmt'].transform('mean')

train_data['TransactionAmt_to_mean_hour'] = train_data['TransactionAmt'] / train_data.groupby(['Transaction_hour'])['TransactionAmt'].transform('mean')



#数値を文字列に変換して項目を作る

train_data['card1_card2'] = train_data['card1'].astype(str) + '_' + train_data['card2'].astype(str)

train_data['addr1_dist1'] = train_data['addr1'].astype(str) + '_' + train_data['dist1'].astype(str)

train_data['card1_addr1'] = train_data['card1'].astype(str) + '_' + train_data['addr1'].astype(str)

train_data['card1_addr2'] = train_data['card1'].astype(str) + '_' + train_data['addr2'].astype(str)

train_data['card2_addr1'] = train_data['card2'].astype(str) + '_' + train_data['addr1'].astype(str)

train_data['card2_addr2'] = train_data['card2'].astype(str) + '_' + train_data['addr2'].astype(str)

train_data['card4_addr1'] = train_data['card4'].astype(str) + '_' + train_data['addr1'].astype(str)

train_data['card4_addr2'] = train_data['card4'].astype(str) + '_' + train_data['addr2'].astype(str)

train_data['DeviceInfo_P_emaildomain'] = train_data['DeviceInfo'].astype(str) + '_' + train_data['P_emaildomain'].astype(str)

train_data['P_emaildomain_addr1'] = train_data['P_emaildomain'].astype(str) + '_' + train_data['addr1'].astype(str)

train_data['id01_addr1'] = train_data['id_01'].astype(str) + '_' + train_data['addr1'].astype(str)



# std(標準偏差)の値で、支払金額を割っている

#card1を軸にしたTransactionAmtの標準偏差で割る

train_data['TransactionAmt_to_std_card1'] = train_data['TransactionAmt'] / train_data.groupby(['card1'])['TransactionAmt'].transform('std')

train_data['TransactionAmt_to_std_card4'] = train_data['TransactionAmt'] / train_data.groupby(['card4'])['TransactionAmt'].transform('std')

train_data['TransactionAmt_to_std_card5'] = train_data['TransactionAmt'] / train_data.groupby(['card5'])['TransactionAmt'].transform('std')

train_data['TransactionAmt_to_std_ProductCD'] = train_data['TransactionAmt'] / train_data.groupby(['ProductCD'])['TransactionAmt'].transform('std')

train_data['TransactionAmt_to_std_addr1'] = train_data['TransactionAmt'] / train_data.groupby(['addr1'])['TransactionAmt'].transform('std')

train_data['TransactionAmt_to_std_id31'] = train_data['TransactionAmt'] / train_data.groupby(['id_31'])['TransactionAmt'].transform('std')

train_data['TransactionAmt_to_std_devicename'] = train_data['TransactionAmt'] / train_data.groupby(['device_name'])['TransactionAmt'].transform('std')

train_data['TransactionAmt_to_std_dow'] = train_data['TransactionAmt'] / train_data.groupby(['Transaction_dow'])['TransactionAmt'].transform('std')

train_data['TransactionAmt_to_std_hour'] = train_data['TransactionAmt'] / train_data.groupby(['Transaction_hour'])['TransactionAmt'].transform('std')



#TransactionAmtの小数点以下を１０００倍

train_data['TransactionAmt_decimal'] = ((train_data['TransactionAmt'] - train_data['TransactionAmt'].astype(int)) * 1000).astype(int)
#軸として支払金額の平均で割る

train_data['TransactionAmt_to_mean_C1'] = train_data['TransactionAmt'] / train_data.groupby(['C1'])['TransactionAmt'].transform('mean')

train_data['TransactionAmt_to_mean_C3'] = train_data['TransactionAmt'] / train_data.groupby(['C3'])['TransactionAmt'].transform('mean')

train_data['TransactionAmt_to_mean_C5'] = train_data['TransactionAmt'] / train_data.groupby(['C5'])['TransactionAmt'].transform('mean')

train_data['TransactionAmt_to_mean_C13'] = train_data['TransactionAmt'] / train_data.groupby(['C13'])['TransactionAmt'].transform('mean')

train_data['TransactionAmt_to_mean_D15'] = train_data['TransactionAmt'] / train_data.groupby(['D15'])['TransactionAmt'].transform('mean')

#軸として支払金額の標準偏差で割る

train_data['TransactionAmt_to_std_C1'] = train_data['TransactionAmt'] / train_data.groupby(['C1'])['TransactionAmt'].transform('std')

train_data['TransactionAmt_to_std_C3'] = train_data['TransactionAmt'] / train_data.groupby(['C3'])['TransactionAmt'].transform('std')

train_data['TransactionAmt_to_std_C5'] = train_data['TransactionAmt'] / train_data.groupby(['C5'])['TransactionAmt'].transform('std')

train_data['TransactionAmt_to_std_D15'] = train_data['TransactionAmt'] / train_data.groupby(['D15'])['TransactionAmt'].transform('std')
#支払金額をlogにする

train_data['TransactionAmt'] = np.log(train_data['TransactionAmt'])

#aqrt(詐欺なし/詐欺あり)

scale_pos_weight = np.sqrt(len(target_data.loc[target_data==0])/len(target_data.loc[target_data==1]))

del raw_train_data
test_identity_data = pd.read_csv("../input/ieee-fraud-detection/test_identity.csv")

test_transaction_data = pd.read_csv("../input/ieee-fraud-detection/test_transaction.csv")
#test_transaction_data の項目ごとの欠損データの数を調べてる

na_columns = test_transaction_data.isna().sum()

print(na_columns[na_columns==0])

print(na_columns[na_columns>0] / test_transaction_data.shape[0])
# https://www.kaggle.com/fchmiel/day-and-time-powerful-predictive-feature



#時刻に変換

test_transaction_data['Transaction_dow'] = np.floor((test_transaction_data['TransactionDT'] / (3600 * 24) - 1) % 7)

test_transaction_data['Transaction_hour'] = np.floor(test_transaction_data['TransactionDT'] / 3600) % 24



#取引データ

transaction_data_columns = test_transaction_data.columns



#数値項目をとりだす

numericCols = test_transaction_data._get_numeric_data().columns



#文字列項目を取り出す

categoricalCols = list(set(transaction_data_columns) - set(numericCols))



#欠損値を変換

test_transaction_data[categoricalCols] = test_transaction_data[categoricalCols].replace({ np.nan:'missing'})

test_transaction_data[numericCols] = test_transaction_data[numericCols].replace({ np.nan:-1})
identity_data_columns = test_identity_data.columns



#数値項目の抜き出し

numericCols = test_identity_data._get_numeric_data().columns



#文字列項目の抜き出し

categoricalCols = list(set(identity_data_columns) - set(numericCols))



#欠損データの置き換え

test_identity_data[categoricalCols] = test_identity_data[categoricalCols].replace({ np.nan:'missing'})

test_identity_data[numericCols] = test_identity_data[numericCols].replace({ np.nan:-1})



#ブラウザのバージョン名を消す

test_identity_data['id_31'] = test_identity_data['id_31'].str.replace('\d+', '')



#OS名の整頓

test_identity_data.loc[test_identity_data['id_30'].str.contains('Mac', na=False), 'id_30'] = 'mac'

test_identity_data.loc[test_identity_data['id_30'].str.contains('iOS', na=False), 'id_30'] = 'iOS'

test_identity_data.loc[test_identity_data['id_30'].str.contains('Android', na=False), 'id_30'] = 'android'

test_identity_data.loc[test_identity_data['id_30'].str.contains('Windows', na=False), 'id_30'] = 'Windows'

test_identity_data.loc[test_identity_data['id_30'].str.contains('Linux', na=False), 'id_30'] = 'Linux'



#デバイス名の整頓

test_identity_data['device_name'] = test_identity_data['DeviceInfo'].str.split('/', expand=True)[0]

test_identity_data.loc[test_identity_data['device_name'].str.contains('SM', na=False), 'device_name'] = 'Samsung'

test_identity_data.loc[test_identity_data['device_name'].str.contains('SAMSUNG', na=False), 'device_name'] = 'Samsung'

test_identity_data.loc[test_identity_data['device_name'].str.contains('GT-', na=False), 'device_name'] = 'Samsung'

test_identity_data.loc[test_identity_data['device_name'].str.contains('Moto G', na=False), 'device_name'] = 'Motorola'

test_identity_data.loc[test_identity_data['device_name'].str.contains('Moto', na=False), 'device_name'] = 'Motorola'

test_identity_data.loc[test_identity_data['device_name'].str.contains('moto', na=False), 'device_name'] = 'Motorola'

test_identity_data.loc[test_identity_data['device_name'].str.contains('LG-', na=False), 'device_name'] = 'LG'

test_identity_data.loc[test_identity_data['device_name'].str.contains('rv:', na=False), 'device_name'] = 'RV'

test_identity_data.loc[test_identity_data['device_name'].str.contains('HUAWEI', na=False), 'device_name'] = 'Huawei'

test_identity_data.loc[test_identity_data['device_name'].str.contains('ALE-', na=False), 'device_name'] = 'Huawei'

test_identity_data.loc[test_identity_data['device_name'].str.contains('-L', na=False), 'device_name'] = 'Huawei'

test_identity_data.loc[test_identity_data['device_name'].str.contains('Blade', na=False), 'device_name'] = 'ZTE'

test_identity_data.loc[test_identity_data['device_name'].str.contains('BLADE', na=False), 'device_name'] = 'ZTE'

test_identity_data.loc[test_identity_data['device_name'].str.contains('Linux', na=False), 'device_name'] = 'Linux'

test_identity_data.loc[test_identity_data['device_name'].str.contains('XT', na=False), 'device_name'] = 'Sony'

test_identity_data.loc[test_identity_data['device_name'].str.contains('HTC', na=False), 'device_name'] = 'HTC'

test_identity_data.loc[test_identity_data['device_name'].str.contains('ASUS', na=False), 'device_name'] = 'Asus'



#レアなデバイス名をothersにする

test_identity_data.loc[test_identity_data.device_name.isin(test_identity_data.device_name.value_counts()[test_identity_data.device_name.value_counts() < 200].index), 'device_name'] = "Others"
#TransactionIDを起点にマージする

raw_test_data = pd.merge(test_transaction_data, test_identity_data, on='TransactionID', how='left')



#取引ID

transactionID = raw_test_data.loc[:,'TransactionID']

del test_identity_data,test_transaction_data
raw_test_data_columns = raw_test_data.columns



#数値項目を取り出す

numericCols = raw_test_data._get_numeric_data().columns



#文字列項目を取り出す

categoricalCols = list(set(raw_test_data_columns) - set(numericCols))

print('The categorical columns in training data are: ',categoricalCols)



#欠損値を置き換え

raw_test_data[categoricalCols] = raw_test_data[categoricalCols].replace({ np.nan:'missing'})

raw_test_data[numericCols] = raw_test_data[numericCols].replace({ np.nan:-1})



#メモリ節約

raw_test_data = reduce_memory_usage(raw_test_data, deep=True, verbose=True)
test_data = raw_test_data.loc[:,featureToUse]
#https://www.kaggle.com/artgor/eda-and-models#Feature-engineering

test_data['TransactionAmt_to_mean_card1'] = test_data['TransactionAmt'] / test_data.groupby(['card1'])['TransactionAmt'].transform('mean')

test_data['TransactionAmt_to_mean_card4'] = test_data['TransactionAmt'] / test_data.groupby(['card4'])['TransactionAmt'].transform('mean')

test_data['TransactionAmt_to_mean_card5'] = test_data['TransactionAmt'] / test_data.groupby(['card5'])['TransactionAmt'].transform('mean')

test_data['TransactionAmt_to_mean_ProductCD'] = test_data['TransactionAmt'] / test_data.groupby(['ProductCD'])['TransactionAmt'].transform('mean')

test_data['TransactionAmt_to_mean_addr1'] = test_data['TransactionAmt'] / test_data.groupby(['addr1'])['TransactionAmt'].transform('mean')

test_data['TransactionAmt_to_mean_id31'] = test_data['TransactionAmt'] / test_data.groupby(['id_31'])['TransactionAmt'].transform('mean')

test_data['TransactionAmt_to_mean_devicename'] = test_data['TransactionAmt'] / test_data.groupby(['device_name'])['TransactionAmt'].transform('mean')

test_data['TransactionAmt_to_mean_dow'] = test_data['TransactionAmt'] / test_data.groupby(['Transaction_dow'])['TransactionAmt'].transform('mean')

test_data['TransactionAmt_to_mean_hour'] = test_data['TransactionAmt'] / test_data.groupby(['Transaction_hour'])['TransactionAmt'].transform('mean')

test_data['card1_card2'] = test_data['card1'].astype(str) + '_' + test_data['card2'].astype(str)

test_data['addr1_dist1'] = test_data['addr1'].astype(str) + '_' + test_data['dist1'].astype(str)

test_data['card1_addr1'] = test_data['card1'].astype(str) + '_' + test_data['addr1'].astype(str)

test_data['card1_addr2'] = test_data['card1'].astype(str) + '_' + test_data['addr2'].astype(str)

test_data['card2_addr1'] = test_data['card2'].astype(str) + '_' + test_data['addr1'].astype(str)

test_data['card2_addr2'] = test_data['card2'].astype(str) + '_' + test_data['addr2'].astype(str)

test_data['card4_addr1'] = test_data['card4'].astype(str) + '_' + test_data['addr1'].astype(str)

test_data['card4_addr2'] = test_data['card4'].astype(str) + '_' + test_data['addr2'].astype(str)

test_data['DeviceInfo_P_emaildomain'] = test_data['DeviceInfo'].astype(str) + '_' + test_data['P_emaildomain'].astype(str)

test_data['P_emaildomain_addr1'] = test_data['P_emaildomain'].astype(str) + '_' + test_data['addr1'].astype(str)

test_data['id01_addr1'] = test_data['id_01'].astype(str) + '_' + test_data['addr1'].astype(str)

test_data['TransactionAmt_to_std_card1'] = test_data['TransactionAmt'] / test_data.groupby(['card1'])['TransactionAmt'].transform('std')

test_data['TransactionAmt_to_std_card4'] = test_data['TransactionAmt'] / test_data.groupby(['card4'])['TransactionAmt'].transform('std')

test_data['TransactionAmt_to_std_card5'] = test_data['TransactionAmt'] / test_data.groupby(['card5'])['TransactionAmt'].transform('std')

test_data['TransactionAmt_to_std_ProductCD'] = test_data['TransactionAmt'] / test_data.groupby(['ProductCD'])['TransactionAmt'].transform('std')

test_data['TransactionAmt_to_std_addr1'] = test_data['TransactionAmt'] / test_data.groupby(['addr1'])['TransactionAmt'].transform('std')

test_data['TransactionAmt_to_std_id31'] = test_data['TransactionAmt'] / test_data.groupby(['id_31'])['TransactionAmt'].transform('std')

test_data['TransactionAmt_to_std_devicename'] = test_data['TransactionAmt'] / test_data.groupby(['device_name'])['TransactionAmt'].transform('std')

test_data['TransactionAmt_to_std_dow'] = test_data['TransactionAmt'] / test_data.groupby(['Transaction_dow'])['TransactionAmt'].transform('std')

test_data['TransactionAmt_to_std_hour'] = test_data['TransactionAmt'] / test_data.groupby(['Transaction_hour'])['TransactionAmt'].transform('std')

# New feature - decimal part of the transaction amount.

test_data['TransactionAmt_decimal'] = ((test_data['TransactionAmt'] - test_data['TransactionAmt'].astype(int)) * 1000).astype(int)
test_data['TransactionAmt_to_mean_C1'] = test_data['TransactionAmt'] / test_data.groupby(['C1'])['TransactionAmt'].transform('mean')

test_data['TransactionAmt_to_mean_C3'] = test_data['TransactionAmt'] / test_data.groupby(['C3'])['TransactionAmt'].transform('mean')

test_data['TransactionAmt_to_mean_C5'] = test_data['TransactionAmt'] / test_data.groupby(['C5'])['TransactionAmt'].transform('mean')

test_data['TransactionAmt_to_mean_C13'] = test_data['TransactionAmt'] / test_data.groupby(['C13'])['TransactionAmt'].transform('mean')

test_data['TransactionAmt_to_mean_D15'] = test_data['TransactionAmt'] / test_data.groupby(['D15'])['TransactionAmt'].transform('mean')

test_data['TransactionAmt_to_std_C1'] = test_data['TransactionAmt'] / test_data.groupby(['C1'])['TransactionAmt'].transform('std')

test_data['TransactionAmt_to_std_C3'] = test_data['TransactionAmt'] / test_data.groupby(['C3'])['TransactionAmt'].transform('std')

test_data['TransactionAmt_to_std_C5'] = test_data['TransactionAmt'] / test_data.groupby(['C5'])['TransactionAmt'].transform('std')

test_data['TransactionAmt_to_std_D15'] = test_data['TransactionAmt'] / test_data.groupby(['D15'])['TransactionAmt'].transform('std')



test_data['TransactionAmt'] = np.log(test_data['TransactionAmt'])
#使えるデータを追加

goodCategoricalVars.append('card1_addr1')

goodCategoricalVars.append('card1_addr2')

goodCategoricalVars.append('card2_addr1')

goodCategoricalVars.append('card2_addr2')

goodCategoricalVars.append('card4_addr1')

goodCategoricalVars.append('card4_addr2')

goodCategoricalVars.append('DeviceInfo_P_emaildomain')

goodCategoricalVars.append('P_emaildomain_addr1')

goodCategoricalVars.append('id01_addr1')

goodCategoricalVars.append('card1_card2')

goodCategoricalVars.append('device_name')

goodCategoricalVars.append('addr1_dist1')
from sklearn.preprocessing import LabelEncoder

#LabelEncoder 文字列を数値に変換する

#['amsterdam', 'paris', 'tokyo']

#le.transform(["tokyo", "tokyo", "paris"]) 

#array([2, 2, 1]...)



for i_cat in goodCategoricalVars:

    le = LabelEncoder()

    curData = pd.concat([train_data.loc[:,i_cat],test_data.loc[:,i_cat]],axis = 0)

    le.fit(curData)

    train_data.loc[:,i_cat] = le.transform(train_data.loc[:,i_cat])   

    test_data.loc[:,i_cat] = le.transform(test_data.loc[:,i_cat])
print('The shape of training data is:',np.shape(train_data))

del raw_test_data
from sklearn.model_selection import StratifiedKFold,GroupKFold

from sklearn.metrics import roc_curve, auc

from lightgbm import LGBMClassifier

import lightgbm as lgb



#group_kfold = GroupKFold(n_splits=5)

cv = StratifiedKFold(n_splits=5, random_state=123, shuffle=True)
def compute_roc_auc(clf,index):

    y_predict = clf.predict_proba(train_data.iloc[index])[:,1]

    fpr, tpr, thresholds = roc_curve(target_data.iloc[index], y_predict)

    auc_score = auc(fpr, tpr)

    return fpr, tpr, auc_score
#LBGMには多くのパラメータがあります

params = {'bagging_fraction': 0.7982116702024386, 

          'feature_fraction': 0.1785051643813966, 

          'max_depth': int(49.17611603427576), 

          'min_child_weight': 3.2852905549011155, 

          'min_data_in_leaf': int(31.03480802715621), 

          'n_estimators': int(1491.3676131788188), 

          'num_leaves': int(52.851307790411965), 

          'reg_alpha': 0.45963319421692145, 

          'reg_lambda': 0.6591286807489907, 

          'metric':'auc',

          'boosting_type': 'dart', 

          'colsample_bytree':.8, 

          'subsample':.9, 

          'min_split_gain':.01, 

          'max_bin':127, 

          'bagging_freq':5, 

          'learning_rate':0.1 , 

          'early_stopping_rounds':500 }
fprs_lgb, tprs_lgb, scores_lgb = [], [], []

feature_importances = pd.DataFrame()

feature_importances['feature'] = train_data.columns     

predictions = np.zeros(len(test_data))

for (train, test), i in zip(cv.split(train_data, target_data), range(5)):

    lgb_best = LGBMClassifier(boosting = params['boosting_type'],n_estimators =  params['n_estimators'],

                     learning_rate =  params['learning_rate'],num_leaves =  params['num_leaves'],

                     colsample_bytree = params['colsample_bytree'],subsample =  params['subsample'],

                     max_depth =  params['max_depth'],reg_alpha =  params['reg_alpha'],

                     reg_lambda =  params['reg_lambda'],min_split_gain =  params['min_split_gain'],

                     min_child_weight =  params['min_child_weight'],max_bin =  params['max_bin'],

                     bagging_freq =  params['bagging_freq'],feature_fraction =  params['feature_fraction'],

                     bagging_fraction =  params['bagging_fraction'],min_data_in_leaf = params['min_data_in_leaf'],

                             early_stopping_rounds = params['early_stopping_rounds'])

    lgb_best.fit(train_data.iloc[train,:], target_data.iloc[train],

                 eval_set = [(train_data.iloc[train,:], target_data.iloc[train]), 

                             (train_data.iloc[test,:], target_data.iloc[test])],eval_metric='auc',verbose = 200)

    feature_importances['fold_{}'.format(i + 1)] = lgb_best.feature_importances_

    _, _, auc_score_train = compute_roc_auc(lgb_best,train)

    fpr, tpr, auc_score = compute_roc_auc(lgb_best,test)

    scores_lgb.append((auc_score_train, auc_score))

    fprs_lgb.append(fpr)

    tprs_lgb.append(tpr) 

    predictions += lgb_best.predict_proba(test_data, num_iteration=lgb_best.best_iteration_)[:,1]/cv.n_splits
plt.figure()

lw = 2

colors = ['black','red','blue','darkorange','green']

for i in range(0,5):

    plt.plot(fprs_lgb[i], tprs_lgb[i], color=colors[i],

             lw=lw, label='ROC curve (area = %0.5f)' % scores_lgb[i][1])

    

plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.legend(loc="lower right")

plt.show()
print('Mean AUC:', np.mean(scores_lgb,axis = 1))
feature_importances['average'] = feature_importances[['fold_{}'.format(fold + 1) for fold in range(cv.n_splits)]].mean(axis=1)

plt.figure(figsize=(16, 16))

sns.barplot(data=feature_importances.sort_values(by='average', ascending=False).head(50), x='average', y='feature');

plt.title('50 TOP feature importance over {} folds average'.format(cv.n_splits));
data = {'TransactionID':transactionID,'isFraud':predictions}

submissionDF = pd.DataFrame(data)

submissionDF.to_csv('sample_submission2.csv',index=False)
predictions