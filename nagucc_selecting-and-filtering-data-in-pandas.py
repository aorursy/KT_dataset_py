import pandas as pd
# 读取CSV文件

melbourne_file_path = '../input/melbourne-housing-market/MELBOURNE_HOUSE_PRICES_LESS.csv'

melbourne_data = pd.read_csv(melbourne_file_path) 



print(melbourne_data.columns)
# 选取其中一列

melbourne_price_data = melbourne_data.Price



# 读取数据集的最前面几行

print(melbourne_price_data.head())
# 选取多行

two_columns_of_data = melbourne_data[['Rooms', 'Price']]

two_columns_of_data.describe()
# 选取第0至第2行(不含)的数据

sub_dataset = melbourne_data[0:3]

print(sub_dataset)