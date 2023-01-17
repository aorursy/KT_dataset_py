import pandas as pd
import seaborn as sns
from learntools.advanced_pandas.indexing_selecting_assigning import *

reviews = pd.read_csv("../input/wine-reviews/winemag-data-130k-v2.csv", index_col=0)
pd.set_option("display.max_rows", 5) #设置显示行数最多为5行
reviews.head()
check_q1(pd.DataFrame())
# Your code here

# 用索引实现取出description列
reviews_des_1 = reviews['description'] #方法1
reviews_des_2 = reviews.description #方法2
## 显示取出的列中前五个元素
print('reviews_des_1[0:5]=\n',reviews_des_1[0:5])
print('reviews_des_2[0:5]=\n',reviews_des_2[0:5])

# 用.iloc[]函数取出description列
reviews_des_3 = reviews.iloc[:,1]
## 显示取出的列中前五个元素
print('reviews_des_3.head(5)=\n',reviews_des_3.head(5))
print('reviews_des_3.shape=\n',reviews_des_3.shape)

#对结果进行检查
#check_q1(reviews_des_1)
#check_q1(reviews_des_2)
check_q1(reviews_des_3)
# 打印reviews.iloc[:,1]及其维度
print('reviews.iloc[:,1]=\n',reviews.iloc[:,1])
print('\n')
print('reviews.iloc[:,1].shape=',reviews.iloc[:,1].shape)

# 打印reviews.iloc[:,[1]]及其维度
print('reviews.iloc[:,[1]]=\n',reviews.iloc[:,[1]])
print('\n')
print('reviews.iloc[:,[1]].shape=',reviews.iloc[:,[1]].shape)
# Your code here

# 直接使用reviews取值
# 方法1：对对象元素取值
# print(reviews.description[0])
# check_q2(reviews.description[0])

# 方法二：通过.iloc函数取值
print(reviews.iloc[0,[1]])
check_q2(reviews.iloc[0,[1]])
# 两种方法的打印结果不一样(主要原因应该是不同的方法取出的数据类型不一样)，但是结果校验正确
# Your code here
print('reviews.iloc[0]=\n',reviews.iloc[0])
check_q3(reviews.iloc[0])
# Your code here
# 取出description列中的前10个值
reviews_des_first10 = reviews.description[0:10]
# 打印
print('reviews_des_first10=\n',reviews_des_first10)

check_q4(reviews_des_first10)
# Your code here
# 利用.iloc[]函数实现
data_somerows = reviews.iloc[[1,2,3,5,8]]
print('data_somerows=\n',data_somerows)

check_q5(data_somerows)
# Your code here
# 取出指定数据
data_rows_columns = reviews.loc[[0,1,10,100],['country','province','region_1','region_2']]
# 打印数据进行查看
print('data_rows_columns=\n',data_rows_columns) # 由打印出的数据可以看出，此种方法取出的数据对象的数据类型为DataFrame

check_q6(data_rows_columns)
# Your code here
# 取出指定数据
data_ex7 = reviews.loc[0:100,['country','variety']]
# 打印数据查看
print('data_ex7=\n',data_ex7) # 由打印出的数据可以看出，此种方法取出的数据对象的数据类型为DataFrame

check_q7(data_ex7)
# Your code here
# 从数据中选择出产自Italy的红酒数据
data_Italy_wines = reviews.loc[reviews.country == 'Italy'] # 通过reviews.country == 'Italy'条件进行判断 
# 打印数据进行查看
print('data_Italy_wines=\n',data_Italy_wines)

check_q8(data_Italy_wines)
# Your code here
# 选择region_2数据非空的数据
data_region_2_notNaN = reviews.loc[reviews.region_2.notnull()]
# 打印数据进行查看
print('data_region_2_notNaN=\n',data_region_2_notNaN)

check_q9(data_region_2_notNaN)
# Your code here
# 选择points列的数据
data_points = reviews.loc[:,'points']

# 打印数据进行查看
print('data_points=\n',data_points)

check_q10(data_points)
# Your code here
# 选择points列的前1000行数据
data_points_first1000 = reviews.loc[0:1000,'points']

# 打印数据进行查看
print('data_points_first1000 = \n',data_points_first1000)

check_q11(data_points_first1000)
# Your code here
# 选择points列的后1000行数据
data_points_last1000 = reviews.loc[-1000:,'points']

# 打印数据进行查看
print('data_points_last1000 = \n',data_points_last1000)

check_q11(data_points_last1000)
# Your code here
# 选择红酒产地为Italy的points列的数据
data_Italy = reviews.loc[reviews.country == 'Italy'] # 取出红酒产地为Italy的数据
data_Italy_points = data_Italy.loc[:,'points'] # 从data_Italy数据中取出points数据

# 打印数据进行查看
print('data_Italy_points =\n',data_Italy_points)

check_q13(data_Italy_points)
# Your code here
# 取出产地为France或者Italy的数据
data_FranceItaly = reviews.loc[(reviews.country.isin(['Italy', 'France'])) & (reviews.points >= 90)]
# 取出points数据
data_FranceItaly_points90 = data_FranceItaly.loc[:,'country']
# 打印数据进行查看
print('data_FranceItaly_points90=\n',data_FranceItaly_points90)

check_q14(data_FranceItaly_points90)