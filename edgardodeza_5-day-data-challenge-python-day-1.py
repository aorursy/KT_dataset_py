import numpy as np      # linear algebra
import pandas as pd     # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt    # data visualization
# list files from input directory
from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))
df = pd.read_csv('../input/archive.csv')    # read csv file

df.head(20)    # Alternatively use print(df.head(20)), though 
               # on Kaggle you don't need to explicitly use print()
print(df.shape)    # print number of rows and columns

print("The number of rows is: ", df.shape[0])
print("The number of columns is: ", df.shape[1])
print(df.columns)
df.describe()    # print statistical information
df.info()