%pylab inline
import pandas
import seaborn

data = pandas.read_csv('../input/player_punt_data_analysis.csv')

### import new data
data
nRowsRead = 1000 # specify 'None' if want to read whole file
df1 = pandas.read_csv('../input/player_punt_data_analysis.csv', delimiter=',', nrows = nRowsRead)
df1.dataframeName = 'player_punt_data_analysis.csv'
nRow, nCol = df1.shape
print(f'There are {nRow} rows and {nCol} columns')
df1.head(5)