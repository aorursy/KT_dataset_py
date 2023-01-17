# 函数注释:"""功能概述， 空格， 参数说明， 返回值说明"""

def get_fee(loan_bal, fee_rate):

    """Calculate fee amount by multiplying loan_bal and fee_rate 

    

    Args:

        loan_bal: numeric, value balance of the loan

        fee_rate: float, fee rate of the loan

        

    Returns:

        fee amount

    """

    return loan_bal * fee_rate
# numpy 矩阵操作

import numpy as np

import time



# 生成测试数据

data_len, data_wid = 100, 100

arr1 = np.random.rand(data_len, data_wid)

arr2 = np.random.rand(data_len, data_wid)

arr3 = np.random.rand(data_len, data_wid)

print("数据量：{}".format(data_len, data_wid))



# 测试1：循环运算

start_tm1 = time.time()

print("测试1进行中……")

for i in range(arr1.shape[0]):

    for j in range(arr1.shape[1]):

        arr3[i,j] = arr1[i,j] + arr2[i,j]

end_tm1 = time.time()



# 测试2：矩阵运算

start_tm2 = time.time()

print("测试2进行中……")

arr3 = arr1 + arr2

end_tm2 = time.time()



# 测试结果

test1_cost, test2_cost = end_tm1 - start_tm1, end_tm2 - start_tm2

print("测试1用时是测试2的{:.2f}倍".format(test1_cost/test2_cost))
# pandas 矩阵操作

import pandas as pd

from datetime import datetime



raw_data = pd.read_csv('../input/groceries-dataset/Groceries_dataset.csv')

print("数据量：{}".format(raw_data.shape))

print(raw_data.head())





# 定义函数

def date_to_days(date_str, date_format='%d-%m-%Y'):

    '''Convert date string to time type

    

    Args:

        date_str: string, date

        date_format: string, format of date(the first arguement)

    

    Returns:

        time type of the date_str

    '''

    

    date = datetime.strptime(date_str, date_format)

    days = (datetime.now() - date).days

    return days



# 测试1：循环运算

start_tm1 = time.time()

print("测试1进行中……")

test_df = pd.DataFrame()

for i in range(len(raw_data)):

    test_df.loc[i, 'days'] = date_to_days(raw_data.loc[i, 'Date'])



end_tm1 = time.time()



# 测试2：矩阵运算

start_tm2 = time.time()

print("测试2进行中……")

test_df = pd.DataFrame()

test_df['days']  = raw_data['Date'].apply(date_to_days)

test_df.head()

end_tm2 = time.time()





# 测试结果

test1_cost, test2_cost = end_tm1 - start_tm1, end_tm2 - start_tm2

print("测试1用时是测试2的{:.2f}倍".format(test1_cost/test2_cost))