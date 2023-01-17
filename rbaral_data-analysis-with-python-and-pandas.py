import numpy as np
a = np.array([1,2,3, 0, 14, 5, 7, 6, 14])
print(a)
#sorting
a.sort()
print(a)
#reverse sorting
a[::-1].sort()
print(a)
#reverse sorting a list
a_list = list(a)
a_list.sort(reverse=True)
print(a_list)
#finding the maximum value
a_max = np.max(a)
print(a_max)
#finding min
a_min = np.min(a)
print(a_min)
#finding average
a_avg = np.average(a)
print(a_avg)
#find the index of the maximum element
max_index = np.where(a==a_max)[0]
print(max_index)
#find the size of the array
print(len(a))
b = np.ones((2,4), dtype=int) #specify datatype, default is float
print(b)
#find sum across rows
print(np.sum(b, axis = 1))
#find sum across cols
print(np.sum(b, axis = 0))
print(b)
print(np.sum(b))
print(np.average(b))
#lets change a value and see the difference
b[1][1] = 20
print(b)
b_max = np.max(b)
print(b_max)
print(np.min(b))
#find position of the max
print(np.where(b == b_max))
b[0][0] = 0
print(b)
b_non_zeros = np.nonzero(b)
print(b_non_zeros)
print(len(b_non_zeros[0]))
b[0][1] = -5
b[1][1] = -5
print(b)
b.sort(axis = 1)
print(b)
#sort in reverse
print(b[::-1])
import pandas as pd

#create our own dataframes
my_df = pd.DataFrame({'field1': ['K0', 'K1', 'K2', 'K3', 'K4', 'K5'],
                      'field2': ['A0', 'A1', 'A2', 'A3', 'A4', 'A5']})
print(my_df.shape)
print(my_df.columns.values)
df = pd.read_csv("../input/my_store_data.csv") #read data as dataframe
#find the number of rows and cols
print(df.shape)
#print header
print(df.columns.values)
#print the first n rows
print(df.head())
df_sel_fields = df[['Store', 'Weekly_Sales']]
print(df_sel_fields.head())
#get column by index
print(df.iloc[:, [1,3]][:10]) #get the first 10 values from the second and third column/fields
wk_sales_min = np.min(df['Weekly_Sales'])
print(wk_sales_min)
wk_sales_max = np.max(df['Weekly_Sales'])
print(wk_sales_max)
wk_sales_avg = np.average(df['Weekly_Sales'])
print(wk_sales_avg)
df_filled = df.fillna(0)
print(df_filled.head())
#now lets get the average
wk_sales_avg = np.average(df_filled['Weekly_Sales'])
print(wk_sales_avg)
#drop the fields with dropna
df_dropped = df.dropna()
print(df_dropped.head())
print(df_dropped.shape)
df.drop_duplicates(inplace=True)
print(df.shape)
df['Year'] = df['Date'].str.split('-').str[0]
print(df.columns.values)
print(df.head())
#make the year as int
df['Year'] = df.Year.astype(int)
def get_month(date_str):
    return date_str.split('-')[1]


df_filled['Month'] = df_filled['Date'].apply(get_month)
print(df_filled.head())
def get_month_str(date_str):
    month = int(date_str.split('-')[1])
    if month==1:
        return 'January'
    elif month==2:
        return 'February'
    else:
        return 'Other month'

df_filled['Month_str'] = df_filled['Date'].apply((lambda x: get_month_str(x)))
print(df_filled.head())
#lets replace every negative value in Weekly_Sales with the same positive value
def replace_negatives(wk_sales):
    return np.abs(wk_sales)

df_filled['Weekly_Sales_Pos1'] = df_filled['Weekly_Sales'].apply((lambda x:replace_negatives(x)))
print(df_filled.head())
#lets find the transactions that were done for the month of 02
df_month_02 = df_filled[df_filled['Month']=='02']
print(df_month_02.head())
print(df_month_02.shape)
#transactions of month 02 where the amount is positive
df_month_02_pos = df_filled[(df_filled['Month']=='02') & (df_filled['Weekly_Sales']>=0)]
print(df_month_02_pos.head())
print(df_month_02_pos.shape)
for index, row in df_filled[:10].iterrows():
    print(index, row['Weekly_Sales'], row['Month'])
#lets group the transactions by month
print(df_filled.columns.values)
df_month_mean = df_filled.groupby(['Month'])['Weekly_Sales'].mean()
print(df_month_mean)

#find total sales by month
#df_month_sum = df_filled.groupby(['Month'])['Weekly_Sales'].sum()
#print(df_month_sum)

#find max sales in each month
#df_month_max = df_filled.groupby(['Month'])['Weekly_Sales'].max()
#print(df_month_max)

#find min sales in each month
#df_month_min = df_filled.groupby(['Month'])['Weekly_Sales'].min()
#print(df_month_min)
print(df_filled.columns.values)
df_filled['Year'] = df_filled['Date'].str.split('-').str[0]
df_year_mean = df_filled.groupby(['Year'])['Weekly_Sales'].mean()
print(df_year_mean)
#verify the above by selection query
df_2010_sales = df_filled[df_filled['Year']=='2010']['Weekly_Sales']
print(np.mean(df_2010_sales))
#lets see the monthly transactions by grouping
df_month_mean = df_filled.groupby(['Month'])['Weekly_Sales'].mean()
print(df_month_mean)
#verify the above by selection query
df_02_sales = df_filled[df_filled['Month']=='02']['Weekly_Sales']
print(np.mean(df_02_sales))
#see the groups created by groupby expression
df_month_grp = df_filled.groupby(['Month'])
#iterate through the group
#for name,group in df_month_grp:
#    print(group)

#get the first n rows of each group
df_month_grp_heads = df_filled.groupby(['Month_str']).head(1)
print(df_month_grp_heads)
#print some fields from the group
#for name,group in df_month_grp:
#    print(group[['Weekly_Sales','Date']])
#get the first n rows of each group
df_month_grp_heads = df_filled.groupby(['Month_str']).head(1)
print(df_month_grp_heads[['Weekly_Sales','Month_str', 'Month', 'Store', 'Date']])
df_filled_1 = df_filled[df_filled['Month'].isin(['02','03','04'])]
df_filled_2 = df_filled[df_filled['Month'].isin(['05','06','07'])]
print(df_filled_1.shape)
print(df_filled_2.shape)
print(df_filled_1.columns.values)
print(df_filled_2.columns.values)
#concatenation - when the fields match
df_filled_concat = pd.concat([df_filled_1, df_filled_2])
print(df_filled_concat.shape)
#now we join the two dataframes
df_filled_1 = df_filled_1[['Store', 'Date', 'Weekly_Sales']][:100]
df_filled_2 = df_filled_2[['Store', 'Date', 'IsHoliday']][:100]
df_filled_join_inner = pd.merge(df_filled_1[:100], df_filled_2[:100], how='inner', on=['Store'])
print(df_filled_join_inner.shape)
print(df_filled_join_inner.head())
print(df_filled_join_inner[df_filled_join_inner['Weekly_Sales']>0].shape)
print(pd.value_counts(df_filled['Month']))
#check if this sums to the total entries in our data
print(pd.value_counts(df_filled['Month']).sum())
bins = [10000, 20000, 30000, 40000, 50000]
#bins = np.linspace(df.Weekly_Sales.min(), df.Weekly_Sales.max(), 3) #set bins a/c to data
df_filled['bin'] = np.digitize(df_filled.Weekly_Sales.values, bins=bins)
print(df_filled[['Weekly_Sales', 'bin']].head())
df_filled['Weekly_Sales_Norm'] = (df_filled['Weekly_Sales'] - df['Weekly_Sales'].min()) / (df['Weekly_Sales'].max() - df['Weekly_Sales'].min())
print(df_filled.head())
#make the range between 0 and 1
df_filled['Weekly_Sales_Norm'] = df_filled['Weekly_Sales_Norm'].clip(lower=0.0, upper=1)
print(df_filled.head())
#we can user .corr method from Pandas to do the correlation analysis between fields
print(df_filled['Weekly_Sales'].corr(df_filled['IsHoliday']))
print(df_filled['Weekly_Sales'].corr(df_filled['Month'].astype(float)))
#lets verify this
df_filled_mnth_grp = df_filled.groupby(['Month'])
for name,group in df_filled_mnth_grp:
    print(name, group['Weekly_Sales'].sum())
print(df_filled['Weekly_Sales'].corr(df_filled['Year'].astype(int)))
df_filled['Weekly_Sales_zscore'] = (df_filled.Weekly_Sales - df_filled.Weekly_Sales.mean())/df_filled.Weekly_Sales.std(ddof=0)
print(df_filled.head())
df_filled_agg = df_filled.agg({'Weekly_Sales' : ['sum', 'min', 'max'], 'Year' : ['min', 'max']})
print(df_filled_agg)
df_filled_sample = df_filled.sample(frac=0.1, replace=True) #sample 10% data randomly and with replacement
print(df_filled_sample.shape)
print(df_filled_sample.sort_values(by=['Store', 'Date', 'Weekly_Sales']).head())
%matplotlib inline
import matplotlib.pyplot as plt
import pandas as pd
df_filled.Month = df_filled.Month.astype(int)
df_filled_month_sales = df_filled[['Weekly_Sales', 'Month']][:100]
df_filled_month_sales.groupby(['Month']).sum().plot(kind='bar', stacked=True)
#lets verify the sale as shown in the graph
month_4_sale = df_filled_month_sales[df_filled_month_sales['Month']==4]['Weekly_Sales'].sum()
print(month_4_sale)
month_1_sale = df_filled_month_sales[df_filled_month_sales['Month']==1]['Weekly_Sales'].sum()
print(month_1_sale)