import numpy as np
# So easy to create a numpy array:

myarray = np.array([0, 1, 2, 3])

print(myarray)
# Wanna sum the values, 0 + 1 + 2 + 3 = 6? 

# go:

print(myarray.sum())
# Want to find the maximum or the minimum, nothing has changed ..

print(f'Max: {myarray.max()}, Min: {myarray.min()}')
# I know, you want to know the average values

# Very difficult..

print (myarray.mean ())
# Let's create an array like the previous one, [0, 1, 2, 3]

myarray2 = np.arange (4)

print (myarray2)
# No,

# I want to create an array, with 4 positions but only with the value 1

print (np.ones (4))
# Or with a value of 0

print (np.zeros (4))
# Or with random (random) values

print (np.random.random (4))
# but Italo, the demand requires random numbers, however, integers

print (np.random.randint (4))
# ooops .. it only returned a random number from 0 to 3 ..

# but our demand requires it to be an array with 4 positions and

# with integer numbers. Complicated? I don't think so…

# With numpy, we will use the randint function (start, end, qty)

print (np.random.randint (0, 10, 4))
# And just to finish these cool commands, remember the arange

print (np.arange (4))
# So, another widely used numpy command is linspace.

print (np.linspace (0, 3, 4))
# It is very similar to the linspace function, it returns spaced values

# at a break, but evenly. So why not use the

# arange? Well, linspace is used a lot when we are plotting

# data, both when viewing the values ​​or when we are assembling

# the axes of this plot

print (np.linspace (0, 1, 4))
# let's import the pandas library into python

import pandas as pd
# and create the Week Series, I make it simple like this:

week = pd.Series (['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday'])

print (week)
# or I could create like this

data = ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday']

week = pd.Series (data)

print (week)
# some ways to print df data or attributes

# if you want to print only the values

print (week.values)



# if you want to print only the indexes

print (week.index)



# if you want… aff, it's already repetitive… but if you want to print Monday,

print (week.iloc [1])



# and remember, the Series is an array, so I could print like that too

print (week [1])
# Now we will create a DataFrame

df = pd.DataFrame ()

df ['column 1'] = [1, 2, 3, 4]

df ['column 2'] = ['a', 'b', 'c', 'd']



# let's print our dataframe

print(df)
# see, we have indexes ranging from 0 to 3, but we can also name the indexes. Like? So for example:

df.index = ['line0', 'line1', 'line2', 'line3']

print(df)
# a very common way would be to create this dataframe using a 'dict' dictionary

data = {

'column 1': [1, 2, 3, 4],

'column 2': ['a', 'b', 'c', 'd']

}

index = ['row0', 'row1', 'row2', 'row3']

df = pd.DataFrame (data = data, index = index)



print(df)
# importing data from a csv file

dfCSV = pd.read_csv('/kaggle/input/minidatasetsexemplos/meuArquivo.csv')

dfCSV
# importing data from a json file

dfJSON = pd.read_json('/kaggle/input/minidatasetsexemplos/meuArquivo.json')

dfJSON
# receiving data from an excel file

dfXLSX = pd.read_excel('/kaggle/input/minidatasetsexemplos/meuArquivo.xlsx')

dfXLSX
# Let's go back to our simple df

print(df, '\n\n')

# and explore our data a little more

# if I want to print column 1,

df['column 1']
# if I want to print column 2,

print(df['column 2'])
# if i want row 1

print (df.loc['row1'])
# but you guys remember that before renaming our indexes,

# Did they have your number string?

# So, in this case, just use iloc

print(df.iloc[1])
# e to print only the value of [column 1] x [row1] .. then

print (df['column 1']['row1'])
# let's import my dataset meuArquivo.csv

df = pd.read_csv('/kaggle/input/minidatasetsexemplos/meuArquivo.csv')
# or I could make a copy of the dfCSV that we already imported

df = dfCSV.copy()
# let's view the first 5 records of our dataset

df.head()
# or the last 5 records

df.tail()
# Italo, can you increase it to 10 records? Yes, just add the number you want to view, on the head or tail

df.head(10)
# And if you want to print or randomly pick some lines

# of your dataset, the sample function will help you

# and make it a lot easier ...

# So if I want 8 random lines from the df dataset, that's enough.



df.sample(8)

# run more than once so you can see that the values really change
# we also have the info function that gives us basic information about the structure of the dataset and its data

df.info()
# another widely used function is the shape, which returns only the number of columns and rows in the dataset

print(df.shape)
# Continuing the analysis, a widely used function is the isnull function

df.isnull()
df.isnull().sum()
# What if there are null values? Well, here comes the analysis.

# just to do a test, let's create a copy of the df

test = df.copy ()



# let's show the shape

test.shape
# now, I will use a feature to add data to the dataframe

# and in the test dataset, I'm going to add all the data from the df dataset again,

# ie I will duplicate values

test = test.append(df)



test.shape
# now yes, another rich function: remove duplicate data

test = test.drop_duplicates ()



test.shape
# so we can do the same thing for null data

# let's review this information

test.isnull().sum ()
# see, there is no column with a null value ...

# so let's generate a value, just for analysis

test.head()
# we will display the first five products on the test dataframe

test['PRODUTO'].head()
# I will change the value of the first product to null (None)

test['PRODUTO'][0] = None

test['PRODUTO'].head ()
# ulalaaa, our first product is worthless, so let's analyze the info from our test dataset

test.isnull().sum()
# and now we see that the PRODUCT variable (or column) has a null value

# let's print the shape of this dataset

print(test.shape)
test.dropna()

print(test.shape) 
test = test.dropna()
# let's see the summary of null records

print(test.isnull().sum ())

# and see the shape

print(test.shape)
# another tip, imagining that I want to have a variable, all the products that were sold



products = test['PRODUTO']

print(products)
products = test['PRODUTO'].unique()

print(products)
# just for curiosity: what is the type of the products variable?

#?

#?

type(products)

# numpy array…. what a wonderful World..
# of entire dataset

test.describe()
# or one only numeric feature

print(test['QTDE'].describe())
# or categorical

test['LOJA'].describe()
# summing up values

print(test['QTDE'].sum())
# count the number of records

print(test['QTDE'].count())
# return the minimum or maximum value

print(test['QTDE'].min())

print(test['QTDE'].max())
# want the average

print(test.mean())
# or the median

print(test.median())
# want to know which are the 10 best selling products,

# we will use the value_counts function combined with the head function.

# That's it buddy ..

print(test['PRODUTO'].value_counts().head(10))
# if I want to make a slice, that is, a slicing, in the dataset

# I simply assign the desired columns (variables)



sales_shop = test [['LOJA', 'VENDEDOR']]
# let's see the result of this

print (sales_shop)
# but I want only the stores called GFFF

condition = (sales_shop['LOJA'] == 'GFFF')

print(condition)

# That is, it returned the variable (column) SHOP, and where the value is equal to GFFF,

# it returns True, if not false
# To display only GFFF stores, we will use this technique

print (sales_shop[condition])
# we created the second condition

condition2 = (sales_shop ['VENDEDOR'] == 'DEEDE')
# and now we will display the data with both conditions

# I want the condition AND condition2

print(sales_shop [condition & condition2])
# I just want to print a dataframe with all ´A´ products

test.query ('PRODUTO == "A"')
# But only products sold in the GGFF store

test.query ('PRODUTO == "A" and LOJA == "GGFG"')
# example, print all sales for product category BB

test.query ('`CATEGORIA PRODUTO` == "BB"')
# For example, I want to know by product category,

# how many products and how much in value

# If I only put the 3 variables involved in this proposal

# see the result:

test[['CATEGORIA PRODUTO', 'QTDE', 'VALOR']].sum()
# This is where groupby fits like a glove

# first, I grouped by the variable to be grouped, PRODUCT CATEGORY

# second, I put my target variables, that is, that I will have the result

# third, the operation

test.groupby('CATEGORIA PRODUTO')[['QTDE', 'VALOR']].sum()
# but from this list, I just want the top 5 and the 3 worst

# how Pandas is the buddy

print ('Top 5')

test.groupby('CATEGORIA PRODUTO')[['QTDE', 'VALOR']].sum().head()
print('3 Piores')

test.groupby('CATEGORIA PRODUTO')[['QTDE', 'VALOR']].sum().tail(3)
test.corr()
# First,

# let's remember our variables in the test dataset

test.columns
# or

test.info()
# Objective: Display the total sales value month by month

# how I want the total month to month, so I will have to group

totalSales = test.groupby('MES') ['TOTAL'].sum ()

print (totalSales)
# NOTICE, it is not importing a variable to receive the value

# we could plot directly, as in the line commented below

# test.groupby('MES')['TOTAL'].sum()

# do the test..



# But I will display the graph through the variable totalSales.

totalSales.plot()
# but what if we want horizontal bars

totalSales.plot(kind = 'barh', title = 'Bar Graph')
# or vertical bars only

totalSales.plot(kind = 'bar', title = 'Bar Chart')
# or in pie

totalSales.plot (kind = 'pie', title = 'Pie Chart')
# or boxplot

totalSales.plot (kind = 'box', title = 'Boxplot Chart')
# and lastly, let's make a chart that shows month to month

# how TOTAL sales behaved, and with possible sales without discount

totalXbruto = test.groupby('MES')[['QTDE','VALOR','DESCONTO','TOTAL']].sum ()

print (totalXbruto)
# to understand, the total variable, brings the result

# qty * amount - discount applied

# QTDE * VALOR - DESCONTO variables

# then to have the gross, we could use only the QTDE * VALOR

# or add the total discount

# let's go for the easiest, I will create a column in our dataset and do the calculation

totalXbruto['GROSS VALUE'] = (totalXbruto['QTDE'] * totalXbruto['VALOR'])

print(totalXbruto)
# done, our data is ready, now we just have to show

totalXbruto.plot (kind = 'line', title = 'Line Graph', y = ['TOTAL', 'GROSS VALUE'])