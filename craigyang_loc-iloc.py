import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
print(os.listdir("../input"))
pd.set_option('max_rows', 7)
ratings = pd.read_csv('../input/ramen-ratings.csv')
ratings
ratings.iloc[2,2]
#输入两个"点",返回一个"点"str
ratings.iloc[2:10,2]
#输入一"点"+一"线",返回一"线"Series
ratings.iloc[2,:]
#上面是y轴的"线",这里是x轴的"线",其中后半部的slice其实可省略
ratings.iloc[2:10,[2,4]]
#输入两"线",返回一"面"DataFrame