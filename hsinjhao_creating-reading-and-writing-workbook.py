import pandas as pd
pd.set_option('max_rows', 5)
from learntools.advanced_pandas.creating_reading_writing import *
pd.set_option('max_rows', 5)
from learntools.advanced_pandas.creating_reading_writing import *
check_q1(pd.DataFrame())
# Your code here
# 导入库文件以及测验文件
import pandas as pd
pd.set_option('max_rows', 5)
from learntools.advanced_pandas.creating_reading_writing import *

# 创建DataFrame对象fruits
fruits = pd.DataFrame(data={'Apples':[30],'Bananas':[21]})
print(fruits)

# 查看fruits.shape
print('fruits.shape=',fruits.shape)

check_q1(fruits)
# Your code here

# 创建DataFrame对象fruits_2
fruits_2 = pd.DataFrame(data=[[35,21],[41,34]],index=['2017 Sales','2018 Sales'],columns=['Apples','Bananas'])
print(fruits_2)

# 查看DataFrame对象fruits_2的维度
print('fruits_2.shape=',fruits_2.shape)

check_q2(fruits_2)
# Your code here
# pandas.Series官方文档
# http://pandas.pydata.org/pandas-docs/stable/generated/pandas.Series.html?highlight=series

# 创建pandas.Series对象dinner
data_dinner = {'Flour':'4 cups','Milk':'1 cup','Eggs':'2 large','Spam':'1 can'}
dinner = pd.Series(data=data_dinner,index=data_dinner.keys(),dtype = object,name= 'Dinner')
# 查看pandas.Series结果
print('dinner=\n',dinner)

#查看dinner.shape
print('dinner.shape=',dinner.shape)

check_q3(dinner)
# Your code here 
# 定义数据文件路径
data_wine_path = '../input/wine-reviews/winemag-data_first150k.csv'

# 读入csv文件数据
wine_reviews = pd.read_csv(data_wine_path,index_col=0)
# 使用读入数据的第一列作为索引(index)值：index_col=0

# 查看数据维度和数据前三行
print('wine_reviews.shape=',wine_reviews.shape)
print('wine_reviews.head(3)=\n',wine_reviews.head(3))

check_q4(wine_reviews)
# 需要指定index_col之后才能将导入的数据变为pandas.DataFrame数据类型,否则check_q4校验错误
# Your code here

# 读入XLS文件
data_wic_path = '../input/publicassistance/xls_files_all/WICAgencies2014ytd.xls'
wic = pd.read_excel(data_wic_path,sheet_name='Pregnant Women Participating')
# XLS文件的默认索引即可以满足pandas.DataFrame中的index_col

# 查看数据维度和数据前三行
print('wic.shape=',wic.shape)
print('wic.head(3)=\n',wic.head(3))

check_q5(wic)
q6_df = pd.DataFrame({'Cows': [12, 20], 'Goats': [22, 19]}, index=['Year 1', 'Year 2'])
# Your code here
# 将pandas.DataFrame格式数据写入至CSV文件
q6_df.to_csv('cows_and_goats.csv')

check_q6()
# Your Code Here
# 导入库文件
import sqlite3 as sql3

# 读取SQL文件
data_sql = sql3.connect('../input/pitchfork-data/database.sqlite')
pitch = pd.read_sql_query('select * from artists',con=data_sql)

# 查看数据维度和数据前三行
print('pitch.shape=',pitch.shape)
print('pitch.head(3)=\n',pitch.head(3))

check_q7(pitch)