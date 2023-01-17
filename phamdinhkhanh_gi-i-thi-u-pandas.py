# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
import pandas as pd
names = ['khanh', 'trang', 'nga', 'nhung', 'linh']
births = [9, 10, 7, 15, 21]

datalist = list(zip(names, births))

dataset = pd.DataFrame(datalist, columns = ['Name', 'Birth'])
dataset
dataset = pd.read_csv('https://raw.githubusercontent.com/phamdinhkhanh/LSTM/master/shampoo.csv', header = 0, index_col = 0, sep = '\t')
dataset.head()
dataset.dtypes
dataset = pd.read_csv('https://raw.githubusercontent.com/plotly/datasets/master/iris.csv', header = 0, sep = ',')
dataset.head()
# Kích thước dữ liệu
dataset.shape
# lấy dữ liệu các dòng từ 10:15 và cột 2:3.
dataset.iloc[10:16, 2:4]
# lấy dữ liệu từ toàn bộ các cột của dòng đầu tiên
dataset.iloc[1, :]
# lấy dữ liệu từ 3 dòng đầu tiên.
dataset.iloc[:3, :]
# lấy dữ liệu từ 3 dòng cuối cùng
dataset.iloc[-3:, :]
# lấy dữ liệu từ 3 cột đầu tiên
dataset.iloc[:, :3]
# lấy dữ liệu từ 3 cột cuối cùng
dataset.iloc[:, -3:]
# Lấy toàn bộ các cột của dataset.
cols = dataset.columns
cols
# Lấy các cột sô 2:3
take_cols = cols[2:4]
take_cols
# Lấy các cột dụa trên tên cột.
dataset[take_cols].head()
melt_data = dataset.melt(['Name'])
melt_data.head()
# Muốn thay đổi tên mặc định variable và value
dataset.melt(['Name'], var_name = ['measurement'], value_name = 'length').head()
dataset.describe()
# Lấy trung bình, min, max, độ lệch chuẩn, số quan sát, median
print('mean: -----------------\n', dataset.mean())
print('min: -----------------\n', dataset.min())
print('max: -----------------\n', dataset.max())
print('std: -----------------\n', dataset.std())
print('count: -----------------\n', dataset.count())
print('median: -----------------\n', dataset.median())
# Giả sử lấy 5 dòng dữ liệu đầu tiên từ dataset và thay thế cột thứ 2 bằng missing tại dòng 4 và 5
import numpy as np

datana = dataset.iloc[:5, :].copy()
datana.iloc[3:,1] = np.nan
datana
# drop các dòng missing
datana.dropna()
# Kiểm tra dữ liệu missing
datana.isna()
# Lấy trung bình cột 2
mean = datana.iloc[:, 1].mean()
# Thay thế các dữ liệu missing bằng trung bình 
datana.fillna(mean)
sum_data = dataset.groupby(['Name']).sum()
sum_data
# Không muốn biến Name thành index
dataset.groupby(['Name'], as_index = False).sum()
pd.pivot_table(dataset, 
               values = ['SepalLength', 'SepalWidth'], 
               index = ['Name'], 
               aggfunc = sum)
# Nếu SepalLength ta muốn tính sum và SepalWidth ta muốn tính mean
pd.pivot_table(dataset, 
               values = ['SepalLength', 'SepalWidth'], 
               index = ['Name'], 
               aggfunc = {'SepalLength': sum,
                         'SepalWidth': np.mean})
# Biểu đồ line
dataset.plot()
# Biểu đồ line. Muốn chia thành các chuỗi riêng lẻ.
dataset.plot(subplots = True, figsize = (6, 10))
# Biểu đồ bar
dataset.iloc[:5, :4].plot.bar()
# Hoặc
dataset.iloc[5, :4].plot(kind = 'bar')
# Bar xoay ngang
dataset.iloc[5, :4].plot(kind = 'barh')
# Biểu đồ piece
dataset.iloc[5, :4].plot(kind = 'pie', autopct = '%.2f')
# Biểu đồ boxplot
dataset.iloc[:, :4].plot(kind = 'box')
# Biểu đồ area
dataset.iloc[:, :4].plot(kind = 'area')
# Biểu đồ area không chồng.
dataset.iloc[:, :4].plot(kind = 'area', stacked = False)
# Biểu đồ scatter. Lưu ý cần 2 chiều x, y
dataset.iloc[:, :2].plot(kind = 'scatter', x = 'SepalLength', y = 'SepalWidth')
# Biểu đồ hexagonal. Cần trục x, y
dataset.iloc[:, :2].plot(kind = 'hexbin', x = 'SepalLength', y = 'SepalWidth')
# Biểu đồ density
dataset.iloc[:, :2].plot(kind = 'kde')
# Biểu đồ histogram
dataset.iloc[:, :2].plot(kind = 'hist', bins = 10)
