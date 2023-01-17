import pandas as pd
# creating a series from an array
a = pd.Series([1, 2, 3], index=['a', 'b', 'c'])
print(f'The pandas series with index a,b,c is \n{a}')
print(f'\nJust for the peace of mind, type of the variable storing the series is:\n{type(a)}')
# creating a series from a dictionary
a = pd.Series({'a': 1, 'b': 2, 'c':3})
print(f'The pandas series with index a,b,c is \n{a}')
print(f'\nJust for the peace of mind, type of the variable storing the series is:\n{type(a)}')
# accessing the values in a series via index
print(f'Value of 1st index = {a["a"]}')
print(f'Value of 2nd index = {a["b"]}')
print(f'Value of 3rd index = {a["c"]}')
# creating a dataframe
dictionary = {
    'column1': [1,2,3],
    'column2':[4,5,6],
    'column3':[7,8,9],
    'column4':[10,11,12]
}
our_df = pd.DataFrame(dictionary, index=["index1", "index2", "index3"])
print(f'The data frame we created :\n{our_df}')
#playing with the above dataframe
print(f'The first column: \n{our_df.column1}')
print(f'\nThe second column: \n{our_df.column2}')
print(f'\nThe third column: \n{our_df.column3}')
# find the details or information about the dataframe in hand
print(f'The number of rows and columns (shape of the df) in this dataframe = {our_df.shape}')
print(f'The number of rows in this dataframe = {our_df.shape[0]}')
print(f'The number of columns in this dataframe = {our_df.shape[1]}')
print(f'The number of elements in this dataframe = {our_df.size}')
# importing a csv file from the mentioned source in form of dataframe
df = pd.read_csv("https://api.covid19india.org/csv/latest/state_wise.csv")
df
# see the index colun above? Here is a way to avoid that!
df= pd.read_csv("https://api.covid19india.org/csv/latest/state_wise.csv", index_col = 0)
df
# find the details or information about the dataframe in hand
print(f'The number of rows and columns (shape of the df) in this dataframe = {df.shape}')
print(f'The number of rows in this dataframe = {df.shape[0]}')
print(f'The number of columns in this dataframe = {df.shape[1]}')
print(f'The number of elements in this dataframe = {df.size}')
# first five rows of the dataframe
df.head() # by default it prints first five rows
# we can state the number of rows we want in the paranthesis like - head.df(2)
#last five rows of the dataframe
df.tail() # by default it returns the last 5 rows
# we can state the number of rows we want in the paranthesis like - tail.df(2)
# information for column names and data types of the dataframe at hand
df.info()
#get the column names
df.columns
# get all data from a column
df['Confirmed']
# the above output was in form of a Series
# get all data from a column - another example
df[['Confirmed','Active']]
# the above output was in form of a dataframe
#loc = label based index, i.e, we can access the column and its whole row using the text present in it
print(f'This will give output in form of a Series \n\n{df.loc["Delhi"]}')
print(f'\nDatatype of above output \n{type(df.loc["Delhi"])}') #just for the peace of mind :p
#loc = label based index, i.e, we can access the column and its whole row using the text present in it
print(f'This will give output in form of a DataFrame \n\n{df.loc[["Delhi"]]}')
print(f'\nDatatype of above output \n{type(df.loc[["Delhi"]])}') #just for the peace of mind :p
#iloc = integer based index, i.e, we can access the column and its whole row using it's number or position
print(f'This will give output in form of a Series \n\n{df.iloc[3]}')
print(f'\nDatatype of above output \n{type(df.iloc[3])}') #just for the peace of mind :p
#iloc = integer based index, i.e, we can access the column and its whole row using it's number or position
print(f'This will give output in form of a Dataframe \n\n{df.iloc[[3]]}')
print(f'\nDatatype of above output \n{type(df.iloc[[3]])}') #just for the peace of mind :p
print(f'Maximum : \n\n{df.max()} \n\n')
print(f'Minimum : \n\n{df.min()}\n\n')
print("Since the datraframe in hand has several types of values, the max and min values are not necessarily from one single row.\nWe need to be careful about this!")
print(f'Mean : \n\n{df.mean()} \n\n')
print(f'Median : \n\n{df.median()} \n\n')
print(f'Standard Deviation : \n\n{df.std()} \n\n')
print(f'Variance : \n\n{df.var()} \n\n')
# everything in a nutshell - .describe()
df.describe()
# finding the statistics using groupby() in dataframe 
print(f'Mean for all columns w.r.t to the "Acive" column \n\n')
df.groupby("Active").mean()