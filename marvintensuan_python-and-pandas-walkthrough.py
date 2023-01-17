# Numbers with no decimal digits are int
type(123456789)
# Numbers with decimal digits are float
type(42.0)
# Complex numbers are numbers followed by j
type(42j)
# Characters, regardless of length are called str
type('Marvin')
# There are also iterable data types such as lists, tuples. There's also dict.

# A list is enclosed in brackets.
# It may contain other data types, including a list in itself!
type([1, 2, 3, 'Marvin', 23.4, 45j])
# 2D list, or lists within a list

type(
[[ 1,  2,     3],
 [86, 89,   100],
 ['c', 'd', 'a']])
# A tuple is enclosed in parentheses.
# A one-element tuple should have a trailing comma

type((12, 13, 14, 'Marvin', 16))
# If you remove the comma, it will return "int" because Python will fail to know
# that you were creating a tuple
type((12,))
# A dict is enclosed by curly brackets
# Dict data type should always a "key-value" pairs

type({"name" : "Marvin", "age" : 18 })
# We can store all these data types in a variable.

x = 42
shopping_list = ["1 kg beef", "1 tetra pack of Joy", "toothpaste"]

type(x)
type(shopping_list)
1+1
2020-1985
23*23
49/7    # Notice how both numerator and denominator are both int but outputs a float
23//7   # returns int type
2**3    # Exponent
23%7
"Marvin" + " Pogi" # Concatenate String
[1, 2, 3] + ["Marvin", "Hey Jude"] # Combining two lists
(1,2,3) + (4,5,6) #Combining tuples
1 == 1
1 is 1
42 == "42"
42 == 42.0
"beans" in ["jack", "beans", "stalk"]
"beans" in ["jack", "and", "the", "beanstalk"]
"Marvin" in "Marvin is genius."
sports = ["Basketball", "Football", "League of Legends", "Tennis", "Golf"]

sports[0]
sports[-1]
sports[1:3]
sports[-3:-1]
x = {"name": "Marvin", "age" : 18}

x["name"]
"name" in {"name": "Marvin", "age" : 18}
shopping_list = ["ground beef", "spaghetti sauce", "tomato paste"]

for i in shopping_list:
    print(i)
x = 42

while x <=50:
    print(x)
    x += 2
for _ in range(5):
    print("\U0001F44F Marvin \U0001F44F is \U0001F44F awesome \U0001F44F")
ppg = ["Blossom", "Bubbles", "Buttercup", "Prof. Utonium", "Mojo Jojo"]

for character in ppg:
    if character[0] == 'B':
        print(character + " is a PowerPuff Girl.")
    elif character == "Mojo Jojo":
        print(character + " is a villain.")
    else:
        print(character + " is a supporting character.")
[print(character + " is a PowerPuff") for character in ppg if character[0] == 'B']
import math

#pi #NameError: name 'pi' is not defined.
math.pi
from math import pi

pi
import math as m

m.pi
from IPython.display import Image
Image('../input/20200919/sample_dataframe.jpg')
Image("../input/20200919/sample_series.jpg")
Image("../input/20200919/pandas_etl.jpg")
import pandas as pd

#Creating a DataFrame from a Python list

shopping_list = ["ground beef", "spaghetti sauce", "tomato paste"]

df1 = pd.DataFrame(shopping_list)

df1

# Here we can see the values that were contained in the list.
# We can also see the rows and columns indices (starts at 0).
prices = [ 24.5, 10, 53]

df2 = pd.DataFrame([shopping_list, prices])

df2.head() # The .head() method shows the first 5 rows of the DataFrame
df3 = pd.DataFrame({
                    'A': shopping_list,
                    'B': prices
                    })

df3.tail() # The .tail() method shows the last 5 rows of the DataFrame
df3['A'] #Get Column 'A' from df3
df3.loc[0] #Get Row '1' from df3
df3.index #no parenthesis
df3.columns
#Defining own columns and rows

df4 = pd.DataFrame({
                    'Items': shopping_list,
                    'Price': prices
                    },

                    index = [1, 2, 3]) #Removing zero-index because I am more comfy w/ Excel :)

df4.head()
df4['Items']
#Create sample timestamps

dates = pd.date_range('20200901', periods=10)

dates
#Create sample stock prices

p = pd.Series([21.0, 23.0, 24.0, 23.5, 23.3,
               22.0, 22.5, 21.3, 22.2, 23.0 ], index=dates)

p
#Create name of the stock

name = "MRVN"
#Create DataFrame

stock_price = pd.DataFrame({'Stock Name': pd.Series(name, index=dates),
                            'Stock Price': p
                            }, index = dates)

stock_price
stock_price[1:3] #Use default indexes to select specific layers
stock_price.iloc[2:6, 0:1]
stock_price.loc[dates[2:6]] #also works if indexes are str
                            #the idea is to select using labels
stock_price.loc[:, ['Stock Price']] #get only the 'Stock Price' column
                                    #note that ":" was used to denote ALL
stock_price[stock_price['Stock Price']>=22.5]
assets = pd.DataFrame({'A': ['Grab', 'Uber', 'Angkas', 'Joyride', 'Owto'],
                      'B': [200000, 20000, 150000, 85000, 60000]})

assets
operating = pd.DataFrame({'A': ['Grab', 'Angkas', 'Taxicabs'],
                          'B': ['Yes', 'Yes', 'Yes']})

operating
pd.merge(assets, operating, on='A') #inner join
pd.merge(assets, operating, on='A', how='left')
pd.merge(assets, operating, on='A', how='right')
new_company = pd.DataFrame({'A': ["Marvin's Company", "Bong Go's Company"],
                            'B': [500000, 123000]})

new_company
pd.concat([assets, new_company])
assets.append(new_company, ignore_index=True) # ignore_index also works for concat
transactions = pd.DataFrame({'A': ['Pen', 'Paper', 'Pen', 'Paper',
                                   'Pen', 'Paper', 'Paper', 'Paper'],
                             'B': ['Janice', 'Chandler', 'Janice', 'Chandler',
                                   'Janice', 'Chandler', 'Janice', 'Chandler'],
                             'C':[100, 200, -150, 300,
                                  400, -300, 150, 500]
                            })

transactions
transactions.groupby('A').sum()
transactions.groupby(['A', 'B']).sum()
pd.pivot_table(transactions, values='C', index=['A', 'B'], aggfunc={'C': sum})
transactions.stack()
x = transactions.stack()

x.unstack()
stock_price.plot()
dataset = pd.read_csv('../input/world-bank-population-data-set/world_pop_dataset.csv')

dataset.head()
#My answer
dataset[dataset['Country Name']=='Philippines']['2002']
dataset[dataset['Country Name']=='Japan'][['2017', '2018', '2019']]
japan_population = dataset[dataset['Country Name']=='Japan'][['Country Name', '2017', '2018', '2019']]
japan_population.to_excel("This is our output file.xlsx", index=False)
dataset.describe()
len(dataset) #not including the headers
# Total Population of 2019 column
# Note that this is not the actual world population as there are aggregate amounts
# found in some rows (e.g. Europe and Central Asia on row 64 and World on row 257)


dataset['2019'].sum()
#Least population on 2019
dataset[dataset['2019'] == dataset['2019'].min()]
#Most population on 2019
dataset[dataset['2019'] == dataset['2019'].max()][['Country Name', 'Country Code', '2019']]
Image("../input/20200919/ms_store.png")
Image("../input/20200919/pip_freeze.png")
