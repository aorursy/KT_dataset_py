# Allow several prints in one cell

from IPython.core.interactiveshell import InteractiveShell

InteractiveShell.ast_node_interactivity = "all"



import numpy as np

import pandas as pd
# input

df = pd.read_csv("../input/cars93/Cars93.csv")
# print here your solution

# unhide the next cell and check your result

# input

df = pd.read_csv("../input/cars93/Cars93.csv")



# Solution 1

print("We have a total of {} nulls".format(df.isnull().sum().sum()))



d = {'Rear.seat.room': np.nanmean, 'Luggage.room': np.nanmedian}

df[['Rear.seat.room', 'Luggage.room']] = df[['Rear.seat.room', 'Luggage.room']].apply(lambda x, d: x.fillna(d[x.name](x)), args=(d, ))



print("We have a total of {} nulls".format(df.isnull().sum().sum()))



df["Rear.seat.room"].sum()

df["Luggage.room"].sum()





# Solution 2

# impor the df

df = pd.read_csv("../input/cars93/Cars93.csv")



# check nulls

print("We have a total of {} nulls".format(df.isnull().sum().sum()))



# define a custom function

def num_inputer(x, strategy):

    if strategy.lower() == "mean":

        x = x.fillna(value = np.nanmean(x))

    if strategy.lower() == "median":

        x = x.fillna(value = np.nanmedian(x))

    return x



# apply the custon function and using args whe can pass the strategy we want

df['Rear.seat.room'] = df[['Rear.seat.room']].apply(num_inputer, args = ["mean"])

df['Luggage.room'] = df[['Luggage.room']].apply(num_inputer, args = ["median"])



# check for nulls

print("We have a total of {} nulls".format(df.isnull().sum().sum()))



df["Rear.seat.room"].sum()

df["Luggage.room"].sum()
 # input

df = pd.DataFrame(np.arange(20).reshape(-1, 5), columns=list('abcde'))

# print here your solution

# unhide the next cell and check your result

# Solution

# using to_frame()

type(df["a"].to_frame())

# using pandas DataFrame

type(pd.DataFrame(df["a"]))



# Other solutions

# Solution

type(df[['a']])

type(df.loc[:, ['a']])

type(df.iloc[:, [0]])



# This returns a series

# Alternately the following returns a Series

type(df.a)

type(df['a'])

type(df.loc[:, 'a'])

type(df.iloc[:, 1])
# input

df = pd.DataFrame(np.arange(20).reshape(-1, 5), columns=list('abcde'))
# print here your solution

# unhide the next cell and check your result

# Solution to question 1

# we pass a list with the custom names BUT THIS DOESN'T change in place

df = pd.DataFrame(np.arange(20).reshape(-1, 5), columns=list('abcde'))

df[["c", "b", "a", "d", "e"]]

df



# if we reasing that this will work

df = df[["c", "b", "a", "d", "e"]]

df



# Solution to question 2

def change_cols(df, col1, col2):

    df_columns = df.columns.to_list()

    index1 = df_columns.index(col1)

    index2 = df_columns.index(col2)

    # swaping values

    df_columns[index1], df_columns[index2] = col1, col2

    

    return df[df_columns]





df = change_cols(df, "b", "e")

df

    



# Solution to question 3

df = pd.DataFrame(np.arange(20).reshape(-1, 5), columns=list('abcde'))

col_list = list(df.columns)

col_list_reversed = col_list[::-1]

col_list

col_list_reversed

# using the trick from solution 1

df = df[col_list_reversed]

df





print("Solution from the website")

print("-------------------------")

# Others solution from the website



# Input

df = pd.DataFrame(np.arange(20).reshape(-1, 5), columns=list('abcde'))



# Solution Q1

df[list('cbade')]



# Solution Q2 - No hard coding

def switch_columns(df, col1=None, col2=None):

    colnames = df.columns.tolist()

    i1, i2 = colnames.index(col1), colnames.index(col2)

    colnames[i2], colnames[i1] = colnames[i1], colnames[i2]

    return df[colnames]



df1 = switch_columns(df, 'a', 'c')



# Solution Q3

df[sorted(df.columns)]

# or

df.sort_index(axis=1, ascending=False, inplace=True)
# input

df = pd.read_csv("../input/cars93/Cars93.csv")

# print here your solution

# unhide the next cell and check your result

# we use set_option to set the maximun rows and columns to display

pd.set_option("display.max_columns",10)

pd.set_option("display.max_rows",10)

df
# input

df = pd.DataFrame(np.random.random(5)**10, columns=['random'])



'''

Desired Output

#>    random

#> 0  0.0035

#> 1  0.0000

#> 2  0.0747

#> 3  0.0000

'''
# print here your solution

# unhide the next cell and check your result

print("Initial DF")

df

print("Using solution 1")

# Solution 1

df.round(4)

df

pd.reset_option('^display.', silent=True)



print("Using solution 2")

# Solution 2

df.apply(lambda x: '%.4f' %x, axis=1).to_frame()

df

pd.reset_option('^display.', silent=True)



print("Using solution 3")

# Solution 3

pd.set_option('display.float_format', lambda x: '%.4f'%x)

df

pd.reset_option('^display.', silent=True)

df
# input

df = pd.DataFrame(np.random.random(4), columns=['random'])

df
# print here your solution

# unhide the next cell and check your result

# Solution 1

# Using style.format we can pass a dictionary to each column and display as we want

out = df.style.format({

    'random': '{0:.2%}'.format,

})

out



# This applies to all the df

pd.options.display.float_format = '{:,.2f}%'.format

# to get the % multiply by 100

df*100

pd.reset_option('^display.', silent=True)
# input

df = pd.read_csv("../input/cars93/Cars93.csv")

df
# print here your solution

# unhide the next cell and check your result

# First let's import only the columns we need

df = pd.read_csv("../input/cars93/Cars93.csv", usecols=["Manufacturer", "Model", "Type"])



# Solution 1

# Using normal python slicing

df[::20]



df = pd.read_csv("../input/cars93/Cars93.csv", usecols=["Manufacturer", "Model", "Type"])



# Solution 2

# Using iloc

df.iloc[::20, :][['Manufacturer', 'Model', 'Type']]

# input

df = pd.read_csv("../input/cars93/Cars93.csv")

df

# print here your solution

# unhide the next cell and check your result

# Solution

df = pd.read_csv("../input/cars93/Cars93.csv", usecols=["Manufacturer", "Model", "Type", "Min.Price", "Max.Price"])



# let's check if we have null

df.isnull().sum().sum()

df.fillna("missing")

# create new index

df["new_index"] = df["Manufacturer"] + df["Model"] + df["Type"]

# set new index

df.set_index("new_index", inplace = True)

df
# input

df = pd.DataFrame(np.random.randint(1, 30, 30).reshape(10,-1), columns=list('abc'))

df
# print here your solution

# unhide the next cell and check your result

# Solution 1



# argsort give the index of the smallest to largest number in an array

# arg_sort[0] is the index of the smallest number in df["a"]

arg_sort = df["a"].argsort()



#arg_sort.to_frame()

#arg_sort[0]



# now let's sort by arg_sort

#df

df = df.iloc[arg_sort]

df["arg_sort"] = arg_sort

df

n_largest = 5

print("The {} largest values in our DF is at row/index {} and the value is {}".format(n_largest, (df[df["arg_sort"] == (n_largest-1)].index[0]), df[df["arg_sort"] == (n_largest-1)]["a"].iloc[0]))



# Shorter solution

n = 5

# select column, argsort, inders (largest to smallest) and select the n largest

df['a'].argsort()[::-1][n]
# input

ser = pd.Series(np.random.randint(1, 100, 15))

# print here your solution

# unhide the next cell and check your result

# Solution using argsort and boolean filtering of pandas series

# I understood that I wanted the second largest of all values that is greter than the mean

# so I sorted

#ser

sorted_ser = ser[ser.argsort()[::-1]]

#sorted_ser

sorted_ser[sorted_ser > sorted_ser.mean()].index[1]



# If you understood that the 2 value you encounter that is bigger than the mean

# This is the correct solution

print('ser: ', ser.tolist(), 'mean: ', round(ser.mean()))

np.argwhere(ser > ser.mean())[1]



# Another solution

ser[ser > ser.mean()].index[1]
# input

df = pd.DataFrame(np.random.randint(10, 40, 60).reshape(-1, 4))

df1 = df.copy(deep = True)
# print here your solution

# unhide the next cell and check your result

# Solution 1

df["sum"] = df.sum(axis = 1)

df



print("The index of the rows that are greater than 100 are {}".format((df[df["sum"] > 100].index).to_list()[-2:]))



# Solution 2 using numpy

rowsums = df1.apply(np.sum, axis=1)



# last two rows with row sum greater than 100

last_two_rows = df1.iloc[np.where(rowsums > 100)[0][-2:], :]

last_two_rows
# input

ser = pd.Series(np.logspace(-2, 2, 30))

ser1 = ser.copy(deep = True)

ser2 = ser.copy(deep = True)
# print here your solution

# unhide the next cell and check your result

# Solution 1

# get the quantiles values

quantiles = np.quantile(ser, [0.05, 0.95])

ser



# filter ser using numpy to know where the values are below or greater than 5% or 95% and replace the values

ser.iloc[np.where(ser < quantiles[0])] = quantiles[0]

ser.iloc[np.where(ser > quantiles[1])] = quantiles[1]

    

# or we can just do

ser1[ser1 < quantiles[0]] = quantiles[0]

ser1[ser1 > quantiles[1]] = quantiles[1]



ser1



# Solution from the webpage

def cap_outliers(ser, low_perc, high_perc):

    low, high = ser.quantile([low_perc, high_perc])

    print(low_perc, '%ile: ', low, '|', high_perc, '%ile: ', high)

    ser[ser < low] = low

    ser[ser > high] = high

    return(ser)



capped_ser = cap_outliers(ser2, .05, .95)

ser2

capped_ser
# input

df = pd.DataFrame(np.random.randint(-20, 50, 100).reshape(10,-1))

# print here your solution

# unhide the next cell and check your result

# This solution sorts the values.

# Not want we want

# my_array = np.array(df.values.reshape(-1, 1))

# my_array = my_array[my_array > 0]

# my_array.shape[0]

# lar_square = int(np.floor(my_array.shape[0]**0.5))

# arg_sort = np.argsort(my_array)[::-1]

# my_array[arg_sort][0:lar_square**2].reshape(lar_square, lar_square)





# Correct solution

my_array = np.array(df.values.reshape(-1, 1)) # convert to numpy

my_array = my_array[my_array > 0] # filter only positive values

lar_square = int(np.floor(my_array.shape[0]**0.5)) # find the largest square

arg_sort = np.argsort(my_array)[::-1][0:lar_square**2] # eliminate the smallest values that will prevent from converting to a square

my_array = np.take(my_array, sorted(arg_sort)).reshape(lar_square, lar_square) # filter the array and reshape back

my_array





# Solution from the webpage

# Step 1: remove negative values from arr

arr = df[df > 0].values.flatten()

arr_qualified = arr[~np.isnan(arr)]



# Step 2: find side-length of largest possible square

n = int(np.floor(arr_qualified.shape[0]**.5))



# Step 3: Take top n^2 items without changing positions

top_indexes = np.argsort(arr_qualified)[::-1]

output = np.take(arr_qualified, sorted(top_indexes[:n**2])).reshape(n, -1)

print(output)

# input

df = pd.DataFrame(np.arange(25).reshape(5, -1))

df
# print here your solution

# unhide the next cell and check your result

# THIS SWAPS the columns

print("Original DataFrame")

df

temp_col = df[1].copy(deep = True)

df[1], df[2] = df[2], temp_col

print("Swapped Columns DataFrame")

df



# # THIS SWAPS the rows

print("Original DataFrame")

df

temp_row = df.iloc[1].copy(deep = True)

df.iloc[1], df.iloc[2] = df.iloc[2], temp_row

print("Swapped Rows DataFrame")

df



# Solution from the webpage

def swap_rows(df, i1, i2):

    a, b = df.iloc[i1, :].copy(), df.iloc[i2, :].copy()

    df.iloc[i1, :], df.iloc[i2, :] = b, a

    return df



print(swap_rows(df, 1, 2))
# input

df = pd.DataFrame(np.arange(25).reshape(5, -1))
# print here your solution

# unhide the next cell and check your result

# Solution 1

df

df.iloc[df.index.to_list()[::-1]]



# Solutions from the webpage

# Solution 2

df.iloc[::-1, :]



# Solution 3

print(df.loc[df.index[::-1], :])
# input

df = pd.DataFrame(np.arange(25).reshape(5,-1), columns=list('abcde'))



'''

Desired Output



   0  5  10  15  20   b   c   d   e

0  1  0   0   0   0   1   2   3   4

1  0  1   0   0   0   6   7   8   9

2  0  0   1   0   0  11  12  13  14

3  0  0   0   1   0  16  17  18  19

4  0  0   0   0   1  21  22  23  24

'''
# print here your solution

# unhide the next cell and check your result

# Using pd.get_dummies

dummies = pd.get_dummies(df["a"])

df = pd.concat([dummies, df], axis = 1)

df



# Solution from the webpage

# in one line

df_onehot = pd.concat([pd.get_dummies(df['a']), df[list('bcde')]], axis=1)

df_onehot
# input

df = pd.DataFrame(np.random.randint(1,100, 40).reshape(10, -1))

# print here your solution

# unhide the next cell and check your result

# Solution 1

def get_col(df):

    columns = list(df.columns)

    df["col_index_with_max"] = ""

    for i in range(len(df)):

        row_values = list(df.iloc[i, :-1].values)

        max_value = np.max(row_values)

        col_index = row_values.index(max_value)

        df["col_index_with_max"].iloc[i] = col_index



get_col(df)



df

print("The col with maximum amont of maximun per row if {} with a total of {} maximus".format(df.groupby("col_index_with_max").size()[::-1].index[0], \

                                                                                              df.groupby("col_index_with_max").size()[::-1].values[0]))



# Solution 2

# Another much more elegant solution from the webpage

print('Column with highest row maxes: ', df.apply(np.argmax, axis=1).value_counts().index[0])
# input

df = pd.DataFrame(np.random.randint(1,100, 40).reshape(10, -1), columns=list('pqrs'), index=list('abcdefghij'))



'''

Desired Output



df

#    p   q   r   s nearest_row   dist

# a  57  77  13  62           i  116.0

# b  68   5  92  24           a  114.0

# c  74  40  18  37           i   91.0

# d  80  17  39  60           i   89.0

# e  93  48  85  33           i   92.0

# f  69  55   8  11           g  100.0

# g  39  23  88  53           f  100.0

# h  63  28  25  61           i   88.0

# i  18   4  73   7           a  116.0

# j  79  12  45  34           a   81.0



'''
# print here your solution

# unhide the next cell and check your result

#######################################################################################################################################

# Solution 1

# input

df = pd.DataFrame(np.random.randint(1,100, 40).reshape(10, -1), columns=list('pqrs'), index=list('abcdefghij'))



# place holders

corr_list = []

index_list = []



# temporary var

max_corr = 0

current_index = ""



# nested loop to calculate

for i in range(len(df)):

    for j in range(len(df)):

        if i == j:

            pass

        else:

            # distance

            curr_corr = sum((df.iloc[i] - df.iloc[j])**2)**.5

            # correlation

            #curr_corr = df.iloc[i].corr(df.iloc[j])

            if curr_corr >= max_corr:

                max_corr = curr_corr

                current_index = list(df.index)[j]

                

    corr_list.append(max_corr)

    index_list.append(current_index)

    

    max_corr = 0

    current_index = ""

    

df["nearest_row"] = index_list

df["dist"] = corr_list

df

df.drop(["nearest_row", "dist"], axis = 1, inplace = True)



#######################################################################################################################################



# Solution from the webpage

# init outputs

nearest_rows = []

nearest_distance = []



# iterate rows.

for i, row in df.iterrows():

    curr = row

    rest = df.drop(i)

    e_dists = {}  # init dict to store euclidean dists for current row.

    # iterate rest of rows for current row

    for j, contestant in rest.iterrows():

        # compute euclidean dist and update e_dists

        e_dists.update({j: round(np.linalg.norm(curr.values - contestant.values))})

    # update nearest row to current row and the distance value

    nearest_rows.append(max(e_dists, key=e_dists.get))

    nearest_distance.append(max(e_dists.values()))



df['nearest_row'] = nearest_rows

df['dist'] = nearest_distance

df
# input

df = pd.DataFrame(np.random.randint(1,100, 80).reshape(8, -1), columns=list('pqrstuvwxy'), index=list('abcdefgh'))

# print here your solution

# unhide the next cell and check your result



# calculate the correlation, returns a matrix 

df_corr = np.abs(df.corr())

# sorted -2 because it goes from min to max

# max = 1 because it's correlation againts each other

# so we pick -2

max_corr = df_corr.apply(lambda x: sorted(x)[-2], axis = 0)

max_corr
# input

df = pd.DataFrame(np.random.randint(1,100, 80).reshape(8, -1))

df1 = df.copy(deep = True)

df2 = df.copy(deep = True)
# print here your solution

# unhide the next cell and check your result

# Solution 1

df["min_by_max"] = (df.apply(min, axis = 1)/df.apply(max, axis = 1))

df



# Other solution from the webpage

# Solution 2

min_by_max = df1.apply(lambda x: np.min(x)/np.max(x), axis=1)

min_by_max

# Solution 3

min_by_max = np.min(df2, axis=1)/np.max(df2, axis=1)

min_by_max
# input

df = pd.DataFrame(np.random.randint(1,100, 80).reshape(8, -1))

# print here your solution

# unhide the next cell and check your result

# Using lambda and numpy partition

df["penultimate"] = df.apply(lambda x: np.partition(x, -2)[-2], axis = 1)

df

df.drop("penultimate", inplace = True, axis = 1)



# Using lambda and python lists

df["penultimate"] = df.apply(lambda x: sorted(list(x))[-2], axis = 1)

df

df.drop("penultimate", inplace = True, axis = 1)



# Solution from the webpage

out = df.apply(lambda x: x.sort_values().unique()[-2], axis=1)

df['penultimate'] = out

df
# input

df = pd.DataFrame(np.random.randint(1,100, 80).reshape(8, -1))

df1 = df.copy(deep = True)
# print here your solution

# unhide the next cell and check your result

# First normalization: mean and std    

df = df.apply(lambda x: ((x-np.mean(x))/np.std(x)), axis = 0)

df



# min max

df1 = df1.apply(lambda x: ((x.max() - x)/(x.max() - x.min())).round(2))

df1

# input

df = pd.DataFrame(np.random.randint(1,100, 80).reshape(8, -1))

# print here your solution

# unhide the next cell and check your result

df["corr"] = 0

for i in range(len(df)-1):

    

    values1 = df.iloc[i, :-1].astype('float64')

    values2 = df.iloc[i+1, :-1].astype('float64')

    corr = values1.corr(values2)

    df["corr"].iloc[i] = corr

df

df.drop("corr", inplace = True, axis = 1)



# Solution from the webpage

# using list comprehension

[df.iloc[i].corr(df.iloc[i+1]).round(2) for i in range(df.shape[0])[:-1]]

# input

df = pd.DataFrame(np.random.randint(1,100, 100).reshape(10, -1))

df1 = df.copy(deep = True)



'''

Desired Output (might change because of randomness)



#     0   1   2   3   4   5   6   7   8   9

# 0   0  46  26  44  11  62  18  70  68   0

# 1  87   0  52  50  81  43  83  39   0  59

# 2  47  76   0  77  73   2   2   0  14  26

# 3  64  18  74   0  16  37   0   8  66  39

# 4  10  18  39  98   0   0  32   6   3  29

# 5  29  91  27  86   0   0  28  31  97  10

# 6  37  71  70   0   4  72   0  89  12  97

# 7  65  22   0  75  17  10  43   0  12  77

# 8  47   0  96  55  17  83  61  85   0  86

# 9   0  80  28  45  77  12  67  80   7   0

'''
# print here your solution

# unhide the next cell and check your result

# input

df = pd.DataFrame(np.random.randint(1,100, 100).reshape(10, -1))

df1 = df.copy(deep = True)



# Using nested loops

print("Original DF")

df

for i in range(len(df)):

    for j in range(len(df)):

        if i == j:

            df.iloc[i ,j] = 0

            # Inverse the matrix so that we can replace the other diagonal

            df[::-1].iloc[i, j] = 0



print("DF from the solution 1")

df



# Solution from the webpage

# Solution

for i in range(df1.shape[0]):

    df1.iat[i, i] = 0

    df1.iat[df1.shape[0]-i-1, i] = 0

    

print("DF from the solution 2")

df1
# input

df = pd.DataFrame({'col1': ['apple', 'banana', 'orange'] * 3,

                   'col2': np.random.rand(9),

                   'col3': np.random.randint(0, 15, 9)})



df_grouped = df.groupby(['col1'])
# print here your solution

# unhide the next cell and check your result

# Solution 1

pd.DataFrame(df_grouped)

df_grouped.groups["apple"]

df_grouped.get_group("apple")



# Solution 2

for i, dff in df_grouped:

    if i == 'apple':

        print(dff)
# input

df = pd.DataFrame({'fruit': ['apple', 'banana', 'orange'] * 3,

                   'rating': np.random.rand(9),

                   'price': np.random.randint(0, 15, 9)})
# print here your solution

# unhide the next cell and check your result

# Solution 1

grouped_by = df["rating"].groupby(df["fruit"])

grouped_by.get_group("banana")

list(grouped_by.get_group("banana"))[1]



# Solution from the webpage

df_grpd = df['rating'].groupby(df.fruit)

df_grpd.get_group('banana')

df_grpd.get_group('banana').sort_values().iloc[-2]
# input

df = pd.DataFrame({'fruit': ['apple', 'banana', 'orange'] * 3,

                   'rating': np.random.rand(9),

                   'price': np.random.randint(0, 15, 9)})

df
# print here your solution

# unhide the next cell and check your result

# Using pandas pivot table

df_grouped = pd.pivot_table(df[["fruit", "price"]], index = ["fruit"], aggfunc = np.mean ).reset_index()

df_grouped



# using groupby

out = df.groupby('fruit', as_index=False)['price'].mean()

out
# input

df1 = pd.DataFrame({'fruit': ['apple', 'banana', 'orange'] * 3,

                    'weight': ['high', 'medium', 'low'] * 3,

                    'price': np.random.randint(0, 15, 9)})



df2 = pd.DataFrame({'pazham': ['apple', 'orange', 'pine'] * 2,

                    'kilo': ['high', 'low'] * 3,

                    'price': np.random.randint(0, 15, 6)})

df1

df2
# print here your solution

# unhide the next cell and check your result

# Solution 1

# using pandas merge

merge_df = pd.merge(df1, df2, left_on=["fruit", "weight"], right_on=["pazham", "kilo"])

merge_df





# Solution from the webpage

pd.merge(df1, df2, how='inner', left_on=['fruit', 'weight'], right_on=['pazham', 'kilo'], suffixes=['_left', '_right'])
# input

df1 = pd.DataFrame({'fruit': ['apple', 'banana', 'orange'] * 3,

                    'weight': ['high', 'medium', 'low'] * 3,

                    'price': np.random.randint(0, 10, 9)})



df2 = pd.DataFrame({'pazham': ['apple', 'orange', 'pine'] * 2,

                    'kilo': ['high', 'low'] * 3,

                    'price': np.random.randint(0, 10, 6)})



df1

df2
# print here your solution

# unhide the next cell and check your result

# We might use pandas merge

#df1.merge(df2, how = "inner", left_on = ["fruit", "weight", "price"], right_on = ["pazham", "kilo", "price"])



df1["concat"] = df1["fruit"].astype(str) + df1["weight"].astype(str) + df1["price"].astype(str)

#df1



df2["concat"] = df2["pazham"].astype(str) + df2["kilo"].astype(str) + df2["price"].astype(str)

#df2



df1 = df1[~df1["concat"].isin(df2["concat"])]

df1.drop("concat", inplace = True, axis = 1)

df1



# Solution from the webpage, IMHO it's incorrect

#df1[~df1.isin(df2).all(1)]
# input

df = pd.DataFrame({'fruit1': np.random.choice(['apple', 'orange', 'banana'], 10),

                    'fruit2': np.random.choice(['apple', 'orange', 'banana'], 10)})

df

# print here your solution

# unhide the next cell and check your result

# Solution

np.where(df.fruit1 == df.fruit2)
# input

df = pd.DataFrame(np.random.randint(1, 100, 20).reshape(-1, 4), columns = list('abcd'))

df



'''

Desired Output



    a   b   c   d  a_lag1  b_lead1

0  66  34  76  47     NaN     86.0

1  20  86  10  81    66.0     73.0

2  75  73  51  28    20.0      1.0

3   1   1   9  83    75.0     47.0

4  30  47  67   4     1.0      NaN

'''
# print here your solution

# unhide the next cell and check your result

df["lag1"] = df["a"].shift(1)

df["lead1"] = df["b"].shift(-1)

df
# input

df = pd.DataFrame(np.random.randint(1, 10, 20).reshape(-1, 4), columns = list('abcd'))

# print here your solution

# unhide the next cell and check your result

# input

df = pd.DataFrame(["STD, City    State",

"33, Kolkata    West Bengal",

"44, Chennai    Tamil Nadu",

"40, Hyderabad    Telengana",

"80, Bangalore    Karnataka"], columns=['row'])



df



'''

Desired Output



0 STD        City        State

1  33     Kolkata  West Bengal

2  44     Chennai   Tamil Nadu

3  40   Hyderabad    Telengana

4  80   Bangalore    Karnataka

'''
# print here your solution

# unhide the next cell and check your result

# we do " ".join(x.split()) to replace multiple spaces to 1 space

# we do split(None, 2, ) to split a string on the second space ()this way we have West Bengal together

df["re"] = df["row"].apply(lambda x: " ".join(x.split()).split(None, 2, ))



new_header = df["re"][0]

values = df["re"][1:]



# our values is a series of lists, we have to do some list comprehension no extract the values

d = {new_header[0]:[int(values.iloc[i][0].replace(",", "")) for i in range(len(values))], \

     new_header[1]:[values.iloc[i][1].replace(",", "") for i in range(len(values))], \

     new_header[2]:[values.iloc[i][2].replace(",", "") for i in range(len(values))]}



# create a pandas DF from a dict

new_df = pd.DataFrame(d)

new_df