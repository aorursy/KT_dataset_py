import pandas as pd
df = pd.read_csv('../input/sample-data/data.csv')
df.head(1)
# At first lets see the test file

sepdf = pd.read_csv('../input/textfile/separator.txt')

sepdf.head()
# when we use sep we get a clean organized DataFrame

sepdf = pd.read_csv('../input/textfile/separator.txt', sep='|')

sepdf.head()
# Columns or Features name as header 

headerdf = pd.read_csv('../input/sample-data/data.csv', header = 0)

headerdf.head(2)
# Make second line as a header 

headerdf = pd.read_csv('../input/sample-data/data.csv', header = 1)

headerdf.head(2)
headerdf = pd.read_csv('../input/sample-data/data.csv', header = None)

headerdf.head(2)
columns_name = ['Company','Rank','Revenue_change','Profits','Assets','Profit_Change','CEO','Industry' ,'Sector','OldRank','Country','Location','WebSite', 'GlobalList', 'Emp','StockHolder']

namesdf = pd.read_csv('../input/sample-data/data.csv', names=columns_name)

namesdf.head()
df = pd.read_csv('../input/sample-data/data.csv', index_col = 0)

df.head(2)
df1 = pd.read_csv('../input/sample-data/data.csv', index_col=[0,1])

df1.head(2)
df2 = pd.read_csv('../input/sample-data/data.csv', index_col=["company", "rank"])

df2.head(2)
df = pd.read_csv('../input/sample-data/data.csv', usecols=['rank', 'company','profits'])

df.head(2)
# Alternative way by columns serial 

df = pd.read_csv('../input/sample-data/data.csv', usecols=[0,1,2,3])

df.head(2)
# rank and profit columns are removed 

df = pd.read_csv('../input/sample-data/data.csv', usecols= lambda column : column not in ['rank', 'profits'])

df.columns
df = pd.read_csv('../input/squeez/squeeze.csv')

df.head()
type(df)
df = pd.read_csv('../input/squeez/squeeze.csv', squeeze=True)

type(df)
# No header name 

# By default it set 0 to N

df = pd.read_csv('../input/sample-data/data.csv', header=None)

df.head(2)
df = pd.read_csv('../input/sample-data/data.csv', header=None, prefix='Col')

df.head(2)
df = pd.read_csv('../input/doublecol/duplicated_columns.csv')

df.head(1)
print('Total columns= ',df.shape[1])
# Make duplicated columns name unique . add 1 at the end of the duplicated colums name

df = pd.read_csv('../input/doublecol/duplicated_columns.csv', mangle_dupe_cols=True)

df.head(1)
# Read all integer data 



df = pd.read_csv('../input/sample-data/data.csv')

df.head(2)
df.info()
# Here rank, revenues is int64 lets make it float64

df1 = pd.read_csv('../input/sample-data/data.csv', dtype={'revenues':"float64", "rank":"float64"})

df1.info()
# Replace '.' with ',' in a floating point  columns

func = lambda x : (x.replace('.', ','))

df = pd.read_csv('../input/sample-data/data.csv', converters = {'profits': func})
df.profits.head(3)
df = pd.read_csv('../input/sample-data/data.csv', encoding = 'ISO-8859-1', true_values = ['Yes'], false_values = ['No'])
df.col1.head(3)

# Yes and No values are changes with in True and False in Col1 columns
# lets skip 2nd rows( where company = 'State Grid')

df = pd.read_csv('../input/sample-data/data.csv', skiprows=2)

df.head()



# It removes all above 2 rows
# skip selected rows 

df = pd.read_csv('../input/sample-data/data.csv', skiprows=[3,4,5])

df.shape

# Three rows are removed from 500 roews
df = pd.read_csv('../input/sample-data/data.csv', skipfooter=2, engine='python')

df.shape
# read only first 10 rows

df = pd.read_csv('../input/sample-data/data.csv', nrows=10)

df.shape
df = pd.read_csv('../input/sample-data/data.csv')
df = pd.read_csv('../input/sample-data/new_data.csv')

df.head(3)

# If you directly onel the file using excel you will find col1 has some blank rows and and both col1, col2 has NaN values
# If your data automatically not converted to 'NaN' then you could use this 

# After encoding the values 

df = pd.read_csv('../input/sample-data/new_data.csv', encoding = 'ISO=8859-1')

df.head()
# If you like to keep default null valuese 

df = pd.read_csv('../input/sample-data/new_data.csv', encoding = 'ISO=8859-1', keep_default_na=False)

df.head(2)
df = pd.read_csv('../input/sample-data/new_data.csv')

df.head(2)
# In company columns first rows has "walmart". lets make it null values

df = pd.read_csv('../input/sample-data/new_data.csv', na_values = ['Walmart'])

df.head(2)



# Walmart replaced with NaN (Not a Number) Values 
df = pd.read_csv('../input/sample-data/new_data.csv', na_values = {'country', 'China'})

df.country.head()

# All china contry are removed to NaN
df = pd.read_csv('../input/sample-data/new_data.csv', verbose=True)