# Importing necessary libraries for all problems:



import numpy as np

import pandas as pd
df = pd.DataFrame({'X':[78,85,96,80,86], 'Y':[84,94,89,83,86],'Z':[86,97,96,72,83]})

df
exam_data  = {'name': ['Anastasia', 'Dima', 'Katherine', 'James', 'Emily', 'Michael', 'Matthew', 'Laura', 'Kevin', 'Jonas'],

        'score': [12.5, 9, 16.5, np.nan, 9, 20, 14.5, np.nan, 8, 19],

        'attempts': [1, 3, 2, 3, 2, 3, 1, 1, 2, 1],

        'qualify': ['yes', 'no', 'yes', 'no', 'no', 'yes', 'yes', 'no', 'no', 'yes']}

labels = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j']



df = pd.DataFrame(exam_data, index = labels)



print(df)
exam_data  = {'name': ['Anastasia', 'Dima', 'Katherine', 'James', 'Emily', 'Michael', 'Matthew', 'Laura', 'Kevin', 'Jonas'],

        'score': [12.5, 9, 16.5, np.nan, 9, 20, 14.5, np.nan, 8, 19],

        'attempts': [1, 3, 2, 3, 2, 3, 1, 1, 2, 1],

        'qualify': ['yes', 'no', 'yes', 'no', 'no', 'yes', 'yes', 'no', 'no', 'yes']}

labels = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j']



df = pd.DataFrame(exam_data, index = labels)



df.info()
# Solution: 1



df.head(3)
# Solution: 2



df.iloc[:3, :]
df[['name', 'score']]
# Solution 1:



df[['name', 'score']].iloc[[1, 3, 5, 6], :]
# Solution 2:



df.iloc[[1, 3, 5, 6], [0, 1]]
df[df['attempts'] > 2][['name', 'score']]
print('Number of rows of DataFrame:', df.shape[0])

print('Number of columns of DataFrame:', df.shape[1])
print(df[df['score'].isnull()])
print(df[(df['score'] >= 15) & (df['score'] <= 20)])
print(df[(df['attempts'] < 3) & (df['score'] > 15)])
# Solution 1: 



import warnings

warnings.filterwarnings('ignore')





exam_data  = {'name': ['Anastasia', 'Dima', 'Katherine', 'James', 'Emily', 'Michael', 'Matthew', 'Laura', 'Kevin', 'Jonas'],

        'score': [12.5, 9, 16.5, np.nan, 9, 20, 14.5, np.nan, 8, 19],

        'attempts': [1, 3, 2, 3, 2, 3, 1, 1, 2, 1],

        'qualify': ['yes', 'no', 'yes', 'no', 'no', 'yes', 'yes', 'no', 'no', 'yes']}

labels = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j']



df = pd.DataFrame(exam_data, index = labels)



df['score'].loc['d'] = 11.5

df['score']                  
# Solution 2 (Preferable):



exam_data  = {'name': ['Anastasia', 'Dima', 'Katherine', 'James', 'Emily', 'Michael', 'Matthew', 'Laura', 'Kevin', 'Jonas'],

        'score': [12.5, 9, 16.5, np.nan, 9, 20, 14.5, np.nan, 8, 19],

        'attempts': [1, 3, 2, 3, 2, 3, 1, 1, 2, 1],

        'qualify': ['yes', 'no', 'yes', 'no', 'no', 'yes', 'yes', 'no', 'no', 'yes']}

labels = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j']



df = pd.DataFrame(exam_data, index = labels)



df.loc['d', 'score'] = 11.5

df['score']
print('Sum of the examination attempts by the students:', df['attempts'].sum())
exam_data  = {'name': ['Anastasia', 'Dima', 'Katherine', 'James', 'Emily', 'Michael', 'Matthew', 'Laura', 'Kevin', 'Jonas'],

        'score': [12.5, 9, 16.5, np.nan, 9, 20, 14.5, np.nan, 8, 19],

        'attempts': [1, 3, 2, 3, 2, 3, 1, 1, 2, 1],

        'qualify': ['yes', 'no', 'yes', 'no', 'no', 'yes', 'yes', 'no', 'no', 'yes']}

labels = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j']



df = pd.DataFrame(exam_data , index=labels)

print("\nMean score for each different student in data frame:", df['score'].mean())
df.loc['k'] = ['Suresh', 15.5, 'yes', 1]     # This line is similar to adding new data in SQL. 

print('Dataframe after adding new rows: ')

print(df)



print('\n')

# df = df.drop('k')      # We need to assign df = df.drop('k'), to perform inplace drop of row. Only df.drop('k') won't work.

df.drop('k', inplace = True)

print('Dataframe after removing the added rows:')

print(df)
df.sort_values(by = ['name', 'score'], ascending = [False, True])
df['qualify'] = df['qualify'].map({'yes': 'True', 'no': 'False'})

df
df['name'].replace('James', 'Suresh', inplace = True)

df    # We can see the changed name in index row 'd'.
exam_data  = {'name': ['Anastasia', 'Dima', 'Katherine', 'James', 'Emily', 'Michael', 'Matthew', 'Laura', 'Kevin', 'Jonas'],

        'score': [12.5, 9, 16.5, np.nan, 9, 20, 14.5, np.nan, 8, 19],

        'attempts': [1, 3, 2, 3, 2, 3, 1, 1, 2, 1],

        'qualify': ['yes', 'no', 'yes', 'no', 'no', 'yes', 'yes', 'no', 'no', 'yes']}

labels = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j']

df = pd.DataFrame(exam_data , index=labels)



df.drop('attempts', axis = 1, inplace = True)

df
exam_data = {'name': ['Anastasia', 'Dima', 'Katherine', 'James', 'Emily', 'Michael', 'Matthew', 'Laura', 'Kevin', 'Jonas'],

'score': [12.5, 9, 16.5, np.nan, 9, 20, 14.5, np.nan, 8, 19],

'attempts': [1, 3, 2, 3, 2, 3, 1, 1, 2, 1],

'qualify': ['yes', 'no', 'yes', 'no', 'no', 'yes', 'yes', 'no', 'no', 'yes']}

labels = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j']

df = pd.DataFrame(data = exam_data, index = labels)



df['color'] = ['Red','Blue','Orange','Red','White','White','Blue','Green','Green','Red']

print('DataFrame after adding Color column:')

df
exam_data = [{'name':'Anastasia', 'score':12.5}, {'name':'Dima','score':9}, {'name':'Katherine','score':16.5}]

df = pd.DataFrame(data = exam_data)



for index, row in df.iterrows():

    print(row['name'], row['score'])
exam_data = {'name': ['Anastasia', 'Dima', 'Katherine', 'James', 'Emily', 'Michael', 'Matthew', 'Laura', 'Kevin', 'Jonas'],

'score': [12.5, 9, 16.5, np.nan, 9, 20, 14.5, np.nan, 8, 19],

'attempts': [1, 3, 2, 3, 2, 3, 1, 1, 2, 1],

'qualify': ['yes', 'no', 'yes', 'no', 'no', 'yes', 'yes', 'no', 'no', 'yes']}

labels = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j']



df = pd.DataFrame(exam_data, labels)

print(df.columns.values)
data = {'col1': [1, 2, 3], 'col2': [4, 5, 6], 'col3': [7, 8, 9]}

df = pd.DataFrame(data)



df.rename(columns = {'col1': 'Column1', 'col2': 'Column2', 'col3': 'Column3'})
data = {'col1': [1, 4, 3, 4, 5], 'col2': [4, 5, 6, 7, 8], 'col3': [7, 8, 9, 0, 1]}

df = pd.DataFrame(data)



print(df[df['col1'] == 4])
data = {'col1': [1, 4, 3, 4, 5], 'col2': [4, 5, 6, 7, 8], 'col3': [7, 8, 9, 0, 1]}

df = pd.DataFrame(data)



print('Original DataFrame:\n')

print(df)



print('\nDataFrame afer altering columns:')

df = df[['col3', 'col2', 'col1']]

print(df)
data = {'col1': [1, 4, 3, 4, 5], 'col2': [4, 5, 6, 7, 8], 'col3': [7, 8, 9, 0, 1]}

df = pd.DataFrame(data)



df2 = {'col1': 11, 'col2': 12, 'col3':13}

df = df.append(df2, ignore_index = True)

print(df)
data = {'col1': [1, 4, 3, 4, 5], 'col2': [4, 5, 6, 7, 8], 'col3': [7, 8, 9, 0, 1]}

df = pd.DataFrame(data)



df.to_csv('new_file.csv', sep = '\t', index = False)



new_df = pd.read_csv('new_file.csv')

print(new_df)
df = pd.DataFrame({'name': ['Anastasia', 'Dima', 'Katherine', 'James', 'Emily', 'Michael', 'Matthew', 'Laura', 'Kevin', 'Jonas'],

                   'city': ['California', 'Los Angeles', 'California', 'California', 'California', 'Los Angeles', 'Los Angeles', 'Georgia', 'Georgia', 'Los Angeles']

                  })



no_of_people_by_city = df.groupby(by = ['city']).count()

print(no_of_people_by_city)
data = {'col1': [1, 4, 3, 4, 5], 'col2': [4, 5, 6, 7, 8], 'col3': [7, 8, 9, 0, 1]}

df = pd.DataFrame(data)



print(df[df['col2'] != 5])    #df['col2'] != 5 = False only for index=1. So it will show all other rows values.
data = {'col1': [1, 4, 3, 4, 5], 'col2': [4, 5, 6, 7, 8], 'col3': [7, 8, 9, 0, 1]}

df = pd.DataFrame(data)

pd.set_option('display.max_rows', 500)

pd.set_option('display.max_columns', 500)

pd.set_option('display.width', 1000)

print(df)



# NO changes observed by changing options value. 

# Learned this by taking help from official solution. 
print(df.iloc[[2]])    # sepcifying index 2 into [] make the result dataframe.



print('\n')



print(df.iloc[2])      # # sepcifying index 2 without [] make the result series.
# Solution 1:



exam_data = {'name': ['Anastasia', 'Dima', 'Katherine', 'James', 'Emily', 'Michael', 'Matthew', 'Laura', 'Kevin', 'Jonas'],

        'score': [12.5, 9, 16.5, np.nan, 9, 20, 14.5, np.nan, 8, 19],

        'attempts': [1, 3, 2, 3, 2, 3, 1, 1, 2, 1],

        'qualify': ['yes', 'no', 'yes', 'no', 'no', 'yes', 'yes', 'no', 'no', 'yes']}

df = pd.DataFrame(exam_data)



print(df)   # Original Dataframe



print('\n')

print(df.replace(np.nan, 0.0) )  # After replacing NaN with 0.0     
# Solution 2:



exam_data = {'name': ['Anastasia', 'Dima', 'Katherine', 'James', 'Emily', 'Michael', 'Matthew', 'Laura', 'Kevin', 'Jonas'],

        'score': [12.5, 9, 16.5, np.nan, 9, 20, 14.5, np.nan, 8, 19],

        'attempts': [1, 3, 2, 3, 2, 3, 1, 1, 2, 1],

        'qualify': ['yes', 'no', 'yes', 'no', 'no', 'yes', 'yes', 'no', 'no', 'yes']}

df = pd.DataFrame(exam_data)



print(df.fillna(0))
exam_data = {'name': ['Anastasia', 'Dima', 'Katherine', 'James', 'Emily', 'Michael', 'Matthew', 'Laura', 'Kevin', 'Jonas'],

        'score': [12.5, 9, 16.5, np.nan, 9, 20, 14.5, np.nan, 8, 19],

        'attempts': [1, 3, 2, 3, 2, 3, 1, 1, 2, 1],

        'qualify': ['yes', 'no', 'yes', 'no', 'no', 'yes', 'yes', 'no', 'no', 'yes']}

df = pd.DataFrame(exam_data)





print('\nSetting index as a column of dataframe:\n')

print(df.reset_index(level = 0))     

# level = 0, Only remove the first index levels from the index. Removes all levels by default. This parameter is useful for multilevel index.





print('\n\nHiding index from DataFrame: ')

print(df.to_string(index= False))
exam_data = {'name': ['Anastasia', 'Dima', 'Katherine', 'James', 'Emily', 'Michael', 'Matthew', 'Laura', 'Kevin', 'Jonas'],

        'score': [12.5, 9, 16.5, np.nan, 9, 20, 14.5, np.nan, 8, 19],

        'attempts': [1, 3, 2, 3, 2, 3, 1, 1, 2, 1],

        'qualify': ['yes', 'no', 'yes', 'no', 'no', 'yes', 'yes', 'no', 'no', 'yes']}

df = pd.DataFrame(exam_data)



# df.set_value(index = 8, col = 'score', value = 10.2)  # set_value is deprecated and will be removed in a future release.



# Setting 10.2 for index = 8 and column = 'score':

df.at[8, 'score'] = 10.2



print('Dataframe after setting value for index = 8 and column = score:\n')

print(df)



# Note: 

# DataFrame.at : Access a single value for a row/column label pair.

# DataFrame.loc : Access a group of rows and columns by label(s).

# DataFrame.iloc : Access a group of rows and columns by integer position(s).
# Number of null values on one or more columns:



df.isnull().values.sum()
data = {'col1': [1, 4, 3, 4, 5], 'col2': [4, 5, 6, 7, 8], 'col3': [7, 8, 9, 0, 1]}

df = pd.DataFrame(data)



print('DataFrame after removing 2nd & 4th rows:')

df.drop(index = [2, 4], inplace = True)

print(df)
exam_data = {'name': ['Anastasia', 'Dima', 'Katherine', 'James', 'Emily', 'Michael', 'Matthew', 'Laura', 'Kevin', 'Jonas'],

        'score': [12.5, 9, 16.5, np.nan, 9, 20, 14.5, np.nan, 8, 19],

        'attempts': [1, 3, 2, 3, 2, 3, 1, 1, 2, 1],

        'qualify': ['yes', 'no', 'yes', 'no', 'no', 'yes', 'yes', 'no', 'no', 'yes']}

df = pd.DataFrame(exam_data)



df.drop(index = [0, 1], inplace = True)

print('Dataframe after droping first two rows:\n')

print(df)



print('\n\nDataframe after reset index:')

df.reset_index()
# Learned new technique form this problem:

# ----------------------------------------



# randn() returns a sample (or samples) from the "standard normal" distribution.

df = pd.DataFrame(np.random.randn(10, 2))     # randn(rows, columns)

print('Original Dataframe:')

print(df)



part_70 = df.sample(frac = 0.7, random_state = 10)    # Return a random sample of items from an axis of object.

print('\n\n70% of the dataframe:')

print(part_70)



print('\n\n30% of the dataframe:')           # Extract rest 30% of the dataframe.

part_30 = df.drop(part_70.index)          

print(part_30)
ds1 = pd.Series(['100', '200', 'python', '300.12', '400'])

ds2 = pd.Series(['10', '20', 'php', '30.12', '40'])



df = pd.concat([ds1, ds2], axis = 1)   # For column wise concatenate we used axis = 1

print('Combines result of two series into a Dataframe: ')

print(df)
exam_data = {'name': ['Anastasia', 'Dima', 'Katherine', 'James', 'Emily', 'Michael', 'Matthew', 'Laura', 'Kevin', 'Jonas'],

        'score': [12.5, 9, 16.5, np.nan, 9, 20, 14.5, np.nan, 8, 19],

        'attempts': [1, 3, 2, 3, 2, 3, 1, 1, 2, 1],

        'qualify': ['yes', 'no', 'yes', 'no', 'no', 'yes', 'yes', 'no', 'no', 'yes']}

df = pd.DataFrame(exam_data)



suffled_df = df.sample(frac = 1)         # I think, suffling rows using frac can be useful form cross-validation. 

print('Dataframe after suffled rows: \n')

print(suffled_df)
date_data = ['3/11/2000', '3/12/2000', '3/13/2000']

ds = pd.Series(date_data)



print('Original String type column:')

print(ds)



date_s = pd.to_datetime(ds)

print('\nColumn converted into Datetime:')

print(date_s)
data = {'col1': [1, 4, 3, 4, 5], 'col2': [4, 5, 6, 7, 8], 'col3': [7, 8, 9, 0, 1]}

df = pd.DataFrame(data)



rename_col_2 = df.rename(columns = {'col2': 'column2'})

print('Dataframe after renaming second column: \n')

print(rename_col_2)
df_column_to_list = df['col2'].to_list()

print('Converted dataframe column into list: ', df_column_to_list)
# Learned this technique new, it's a handy way to create dataframe from numpy array manually:



dtype = [('column1', 'int64'), ('column2', 'float64'), ('column3', 'float64')]

values = np.zeros(15, dtype = dtype)      # np.zeros(shape, dtype), dtype is by default float unless we specify. 

index = ['Index'+str(i) for i in range(1, len(values)+1)]



df = pd.DataFrame(data = values, index = index)

df
data = {'col1': [1, 4, 3, 4, 7], 'col2': [4, 5, 6, 9, 5], 'col3': [7, 8, 12, 1, 11]}

df = pd.DataFrame(data)



result = df.idxmax().to_list()



# I know the below lines of codes may be tough for someone, but I think it is the robust solution of this problem. 

for i in range(1, len(df.columns) + 1):

    print('For {column} row {index} has maximum value.'.format(column = 'col'+str(i), index = result[i-1]))
data = {'col1': [1, 2, 3, 4, 7], 'col2': [4, 5, 6, 9, 5], 'col3': [7, 8, 12, 1, 11]}

df = pd.DataFrame(data)



# col_check = input('Enter a column name to check:\n')      # Uncomment this line if you want to take user input.

col_check = 'col3'



if col_check in df.columns:

    print('{} is present in dataframe'.format(col_check))

else:

    print('\n\n{} is not present in dataframe'.format(col_check))
# Uncomment the below line if you want to take user input.

# index = int(input('Enter a row number to print it\'s row values: '))



index = 3    # For index value > 4 it will raise index out of bound error. 



print('\nValues of row {}'.format(index + 1))

print(df.iloc[index])
exam_data = {'name': ['Anastasia', 'Dima', 'Katherine', 'James', 'Emily', 'Michael', 'Matthew', 'Laura', 'Kevin', 'Jonas'],

        'score': [12.5, 9, 16.5, np.nan, 9, 20, 14.5, np.nan, 8, 19],

        'attempts': [1, 3, 2, 3, 2, 3, 1, 1, 2, 1],

        'qualify': ['yes', 'no', 'yes', 'no', 'no', 'yes', 'yes', 'no', 'no', 'yes']}

df = pd.DataFrame(exam_data)



print(df.dtypes)
df1 = pd.DataFrame()

df2 = pd.DataFrame({'col1': range(3), 'col2': range(3)})



df1 = df1.append(df2)

print('Dataframe afer appending data to original empty dataframe:\n')

print(df1)
exam_data = {'name': ['Anastasia', 'Dima', 'Katherine', 'James', 'Emily', 'Michael', 'Matthew', 'Laura', 'Kevin', 'Jonas'],

        'score': [12.5, 9, 16.5, np.nan, 9, 20, 14.5, np.nan, 8, 19],

        'attempts': [1, 3, 2, 3, 2, 3, 1, 1, 2, 1],

        'qualify': ['yes', 'no', 'yes', 'no', 'no', 'yes', 'yes', 'no', 'no', 'yes']}

df = pd.DataFrame(exam_data)



print(df.sort_values(by = ['attempts', 'name']))
# We will change the Data type of 'score' column from float to int:

exam_data = {'name': ['Anastasia', 'Dima', 'Katherine', 'James', 'Emily', 'Michael', 'Matthew', 'Laura', 'Kevin', 'Jonas'],

        'score': [12.5, 9.1, 16.5, 12.77, 9.21, 20.22, 14.5, 11.34, 8.8, 19.13],

        'attempts': [1, 3, 2, 3, 2, 3, 1, 1, 2, 1],

        'qualify': ['yes', 'no', 'yes', 'no', 'no', 'yes', 'yes', 'no', 'no', 'yes']}

df = pd.DataFrame(exam_data)



print('Datatype before:')

print(df.dtypes)



df['score'] = df['score'].astype('int64')    # Selected column can't have null values. If so they will raise an error. 

print('\nDatatypes after:')

print(df.dtypes)
df = pd.DataFrame([1000.0, 2000.0, 3000.0, -4000.0, np.inf, -np.inf])

print('Original Dataframe:')

print(df)



new_df = df.replace(to_replace = [np.inf, -np.inf], value = np.nan)

print('\n\nNew dataframe:')

print(new_df)
df = pd.DataFrame({'col2': [4, 5, 6, 9, 5], 'col3': [7, 8, 12, 1, 11]})

print('Dataframe before insertion of new column:')

print(df)



df.insert(loc = 0, column = 'col1', value = [1, 2, 3, 4, 7])

print('\nInserting `col1` at the beginning of original dataframe:')

print(df)
df = pd.DataFrame(data = [[2, 4], [1, 3]], columns = ['col1', 'col2'])

print(df)
df = pd.DataFrame(

    {

        'col1':['C1','C1','C2','C2','C2','C3','C2'], 

        'col2':[1, 2, 3, 3, 4, 6, 5]

    })



print('Grouped by `col1` and showing `col2`\'s values as list:')

df.groupby(by = 'col1')['col2'].apply(list)
data = {'col1': [1, 2, 3, 4, 7], 'col2': [4, 5, 6, 9, 5], 'col3': [7, 8, 12, 1, 11]}

df = pd.DataFrame(data)



print('Columns index for `col2` is: ')

df.columns.get_loc(key = 'col2')
# Solution 1:



df.columns.value_counts().sum()
# Solution 2: (Preferable)



len(df.columns)
# Solution 1:



print('All the columns except `col2`:\n')

print(df.drop(columns = ['col2']))
# Solution 2:



print('All the columns except `col2`:\n')

print(df.iloc[:, df.columns != 'col2'])
# Solution 1:



print('First 3 records of the DataFrame:\n')

print(df.iloc[:3, :])
# Solution 2:



print('First 3 records of the DataFrame:\n')

print(df.head(3))
df = pd.DataFrame({'col1': [1, 2, 3, 4, 7, 11], 'col2': [4, 5, 6, 9, 5, 0], 'col3': [7, 5, 8, 12, 1,11]})



print('Topmost 3 records by `col1`:')

df2 = df.nlargest(n = 3, columns = 'col1', keep = 'all')

print(df2)



print('\nTopmost 3 records by `col2`:')

df2 = df.nlargest(n = 3, columns = 'col2', keep = 'all')    # setting keep = 'all' will give us two rows for duplicate occurance of value 5.

print(df2)     # for setting keep = 'all' we are getting 4 topmost rows instead of 3. That means it allows duplicates.



print('\nTopmost 3 records by `col3`:')

df2 = df.nlargest(n = 3, columns = 'col3', keep = 'all')

print(df2)
# Solution 2:



df = pd.DataFrame({'col1': [1, 2, 3, 4, 7, 11], 'col2': [4, 5, 6, 9, 5, 0], 'col3': [7, 5, 8, 12, 1,11]})

print('Top 3 records by `col1`:')

print(df.sort_values(by = 'col1', ascending = False).head(3))



print('\nTop 3 records by `col2`:')

print(df.sort_values(by = 'col2', ascending = False).head(3))    # We should get another row in the output as col2 has two rows having value 5.



print('\nTop 3 records by `col3`:')

print(df.sort_values(by = 'col3', ascending = False).head(3))
df = pd.DataFrame({'col1': [1, 2, 3, 4, 7, 11], 'col2': [4, 5, 6, 9, 5, 0], 'col3': [7, 5, 8, 12, 1,11]})



print('Dataframe after first three rows: \n')

print(df.iloc[3:, :])
df = pd.DataFrame({'col1': [1, 2, 3, 4, 7, 11], 'col2': [4, 5, 6, 9, 5, 0], 'col3': [7, 5, 8, 12, 1,11]})



print('Dataframe without last three rows: \n')

print(df.iloc[:-3, :])