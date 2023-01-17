#load required modules
import pandas as pd
import numpy as np

#loading data file from colab synced path
data = pd.read_excel('../input/hid2.xlsx')
#little idea about data and columns
data.head()
#convert data into dataframe
df = pd.DataFrame(data)
df.columns
#find column index from dataframe which is datetime64 object and if found we put a try/catch to cast that column to datetime
for col in df.columns:
    if df[col].dtype == 'datetime64':
        try:
            df[col] = pd.to_datetime(df[col])
        except ValueError:
            pass

#preparing list/object of datatype of every column
type_list = df.dtypes
type_list
#we find out that which column is to be taken as out target column
i = 0
dates_cols = []
for type_name in type_list:
    # '<M8[ns]' is the datetime64 object
    if type_name=='<M8[ns]':
        dates_cols.append(i)
    i = i+1

#printing column indexes that are of type datetime found by previous loop
print(dates_cols)
#finding size of list for loop control
dates_list_size = len(dates_cols)
i=1
reducing_range = []
# to subtract columns in general order if dates are available
for date_col in dates_cols:
    if i < dates_list_size:
        # we will prepare a reduced list from original list which will help us subtracting remaining date columns from current iterative column which loop refers to
        reducing_range = dates_cols[i:]
        for j in reducing_range:
            #we will prepare dynamic but easily readable column title to easily identify column
            col_name = str(df.columns[date_col]) + ' - ' + str(df.columns[j])
            # we will actually subtract date from a date here and also we will divide that result by a numpy timedelta of 1 day to get an integer value
            df[col_name] = (df[df.columns[date_col]] - df[df.columns[j]])/np.timedelta64(1,'D')
    i+=1

df
# we will reverse sort them columns array
dates_cols.sort(reverse = True)
i=1
# to subtract columns in reverse order if dates are getting bigger towards left to right ->
for date_col in dates_cols:
    if i < dates_list_size:
        reducing_range = dates_cols[i:]
        for j in reducing_range:
            col_name = str(df.columns[date_col]) + ' - ' + str(df.columns[j])
            df[col_name] = (df[df.columns[date_col]] - df[df.columns[j]])/np.timedelta64(1,'D')
    i+=1

df
# to find Pearson corelation
df.corr(method='pearson')
# to get full corelation table's abs() values in a variable for preprocessing
corelation_set = df.corr(method='pearson').abs()
# I will extract upper triangle of corelation set/matrix with help of numpy
upper_triangle = corelation_set.where(np.triu(np.ones(corelation_set.shape), k=1).astype(np.bool))
upper_triangle
# I will find index of feature columns with correlation greater than 0.85 from our 
to_drop = [column for column in upper_triangle.columns if any(upper_triangle[column] > 0.85)]
to_drop
# I will drop whole list of columns from dataframe
df.drop(to_drop, axis=1, inplace=True)
#will see how remaining dataframe looks like
df