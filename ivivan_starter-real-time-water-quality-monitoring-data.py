from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt # plotting
import numpy as np # linear algebra
import os # accessing directory structure
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

nRowsRead = 1000 # specify 'None' if want to read whole file
# Johnstone_river_coquette_point_joined.csv may have more rows in reality, but we are only loading/previewing the first 1000 rows
df1 = pd.read_csv('/kaggle/input/Johnstone_river_coquette_point_joined.csv', delimiter=',', nrows = nRowsRead)
df1.dataframeName = 'Johnstone_river_coquette_point_joined.csv'
nRow, nCol = df1.shape
print(f'There are {nRow} rows and {nCol} columns')
df1.head(5)
nRowsRead = 1000 # specify 'None' if want to read whole file
# Johnstone_river_innisfail_joined.csv may have more rows in reality, but we are only loading/previewing the first 1000 rows
df2 = pd.read_csv('/kaggle/input/Johnstone_river_innisfail_joined.csv', delimiter=',', nrows = nRowsRead)
df2.dataframeName = 'Johnstone_river_innisfail_joined.csv'
nRow, nCol = df2.shape
print(f'There are {nRow} rows and {nCol} columns')
df2.head(5)
plotScatterMatrix(df2, 12, 10)
nRowsRead = 1000 # specify 'None' if want to read whole file
# Mulgrave_river_deeral_joined.csv may have more rows in reality, but we are only loading/previewing the first 1000 rows
df3 = pd.read_csv('/kaggle/input/Mulgrave_river_deeral_joined.csv', delimiter=',', nrows = nRowsRead)
df3.dataframeName = 'Mulgrave_river_deeral_joined.csv'
nRow, nCol = df3.shape
print(f'There are {nRow} rows and {nCol} columns')
df3.head(5)
import missingno as msno
import seaborn as sns; sns.set(style="whitegrid", font_scale=2)
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker



# missing data plot
# pick up one station and check the missing data status
# missing value happens at different timestamps for different water quality variables


file = '/kaggle/input/Tully_river_euramo_joined.csv'

df = pd.read_csv(file)
df.set_index('Timestamp', inplace=True)
df.drop(columns=['Dayofweek', 'Month'],inplace=True)
df = df.loc['2019-02-01T00:00:00':'2019-03-31T00:00:00']
df.replace(0, np.nan, inplace=True)


msno.matrix(df.set_index(pd.period_range(start='2019-02-01', periods=1393, freq='H')) , freq='10D', fontsize=20)
