# Set up code checking
from learntools.core import binder
binder.bind(globals())
from learntools.machine_learning.ex2 import *
print("Setup Complete")
import pandas as pd

# Path of the file to read
iowa_file_path = '../input/home-data-for-ml-course/train.csv'

# Fill in the line below to read the file into a variable home_data
home_data = pd.read_csv(iowa_file_path)

# Call line below with no argument to check that you've loaded the data correctly
step_1.check()
# Lines below will give you a hint or solution code
#step_1.hint()
step_1.solution()
# Print summary statistics in next line
print(home_data.head(5))
# print(home_data.summary())
print(home_data.describe())
col = home_data.columns
# print(col)
# What is the average lot size (rounded to nearest integer)?
# round ： 四舍五入函数
des = home_data.describe()
# print(type(des))  # 查看类型
avg_lot_size = round(des['LotArea']['mean'])
# print(avg_lot_size)

# As of today, how old is the newest home (current year - the date in which it was built)
# 该如何处理时间之间的加减
# 查看最大最小年，和年的类型，发现是numpy float64
nb = des['YearBuilt']['max']
# nl = des['YearBuilt']['min']
# print(nb)
# print(nl)
# print(type(nb))
now = 2019
newest_home_age = now - nb  # 这里用min函数为什么出来这么多数值，不应该是一个数值吗？
# print(newest_home_age)
# Checks your answers
step_2.check()
step_2.hint()
step_2.solution()