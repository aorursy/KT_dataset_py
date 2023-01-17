import numpy as np
import pandas as pd    
my_series = pd.Series( data = [2,3,5,4],             # Data
                       index= ['a', 'b', 'c', 'd'])  # Indexes

my_series
my_dict = {"x": 2, "a": 5, "b": 4, "c": 8}

my_series2 = pd.Series(my_dict)

my_series2 
my_series["a"]
my_series[0]
my_series[1:3]
my_series + my_series
my_series + my_series2
np.mean(my_series)        # numpy array functions generally work on series
# Create a dictionary with some different data types as values

my_dict = {"name" : ["Joe","Bob","Frans"],
           "age" : np.array([10,15,20]),
           "weight" : (75,123,239),
           "height" : pd.Series([4.5, 5, 6.1], 
                                index=["Joe","Bob","Frans"]),
           "siblings" : 1,
           "gender" : "M"}

df = pd.DataFrame(my_dict)   # Convert the dict to DataFrame

df                           # Show the DataFrame
my_dict2 = {"name" : ["Joe","Bob","Frans"],
           "age" : np.array([10,15,20]),
           "weight" : (75,123,239),
           "height" :[4.5, 5, 6.1],
           "siblings" : 1,
           "gender" : "M"}

df2 = pd.DataFrame(my_dict2)   # Convert the dict to DataFrame

df2                            # Show the DataFrame
df2 = pd.DataFrame(my_dict2,
                   index = my_dict["name"] )

df2
# Get a column by name

df2["weight"]
df2.weight
# Delete a column

del df2['name']
# Add a new column

df2["IQ"] = [130, 105, 115]

df2
df2["Married"] = False

df2

df2["College"] = pd.Series(["Harvard"],
                           index=["Frans"])

df2
df2.loc["Joe"]          # Select row "Joe"
df2.loc["Joe","IQ"]     # Select row "Joe" and column "IQ"
df2.loc["Joe":"Bob" , "IQ":"College"]   # Slice by label
df2.iloc[0]          # Get row 0
df2.iloc[0, 5]       # Get row 0, column 5
df2.iloc[0:2, 5:8]   # Slice by numeric row and column index
boolean_index = [False, True, True]  

df2[boolean_index] 
# Create a boolean sequence with a logical comparison
boolean_index = df2["age"] > 12

# Use the index to get the rows where age > 12
df2[boolean_index]
df2[ df2["age"] > 12 ]
titanic_train = pd.read_csv("../input/train.csv")

type(titanic_train)
titanic_train.shape      # Check dimensions
titanic_train.head(6)    # Check the first 6 rows
titanic_train.tail(6)   # Check the last 6 rows
titanic_train.index = titanic_train["Name"]  # Set index to name
del titanic_train["Name"]                    # Delete name column

print(titanic_train.index[0:10])             # Print new indexes
titanic_train.columns
titanic_train.describe()    # Summarize the first 6 columns
np.mean(titanic_train,
        axis=0)          # Get the mean of each numeric column
titanic_train.info()