import pandas as pd

import numpy as np
data = pd.read_csv('/kaggle/input/mercedes-benz-greener-manufacturing/train.csv.zip')

data.head()
data_limited_columns = pd.read_csv('/kaggle/input/mercedes-benz-greener-manufacturing/train.csv.zip', usecols = ['X1', 'X2', 'X3', 'X4', 'X5', 'X6'])

data_limited_columns.head()
# Lets have a look at how many labels each variable has.



for col in data_limited_columns.columns:

    print(col, ": ", len(data_limited_columns[col].unique()), ' labels.')
data_limited_columns.shape
# Lets examine how many columns will it create if we use One Hot Encoding on these variables.



pd.get_dummies(data_limited_columns).shape

pd.get_dummies(data_limited_columns, drop_first = True).shape

# Lets find the top 10 most frequent categories for the variable X2.



# data_limited_columns.value_counts().sort_values(ascending = False).head(20)



data_limited_columns.X2.value_counts().sort_values(ascending = False).head(20)

# Lets make a list with the most frequent categories of the variable 



top_10 = [x for x in data_limited_columns.X2.value_counts().sort_values(ascending = False).head(10).index]



top_10
# Now we make the 10 Binary Variables



for label in top_10:

    data_limited_columns[label] = np.where(data_limited_columns['X2'] == label, 1, 0)

    # Here we are having a where condition for X2 to match with the label, if matches then 1 else 0.

        
data_limited_columns.shape
data_limited_columns.head()
data_limited_columns[['X2']+top_10].head(10)
top_10
# Lets drop the columns we just created

# data_limited_columns.drop(['as', 'ae', 'ai', 'm', 'ak', 'r', 'n', 's', 'f', 'e'], axis=1, inplace=True)

data_limited_columns.drop(columns = top_10, axis = 1, inplace=True)
data_limited_columns.head()
def OHE_Top_X(df, variable, top_x_labels):

    for label in top_x_labels:

        df[variable+'_'+label] = np.where(df[variable] == label, 1, 0)

        
# Read the data again

# data_limited_columns = pd.read_csv('/kaggle/input/mercedes-benz-greener-manufacturing/train.csv.zip', usecols = ['X1', 'X2', 'X3', 'X4', 'X5', 'X6'])

# data_limited_columns.head()
# Encode X2 into the 10 most frequent categories.

OHE_Top_X(data_limited_columns, 'X2', top_10)

data_limited_columns.head()
data_limited_columns.drop('X2', axis = 1, inplace=True)

data_limited_columns.head()
# Find the 10 Most frequent categories for X1



top_10_X1 = [x for x in data_limited_columns.X1.value_counts().sort_values(ascending = False).head(10).index]



OHE_Top_X(data_limited_columns, 'X1', top_10_X1)

# data_limited_columns.head()



data_limited_columns.drop('X1', axis = 1, inplace=True)

data_limited_columns.head()
# data = { 

#     'A':['A1', 'A2', 'A3', 'A4', 'A5'],  

#     'B':['B1', 'B2', 'B3', 'B4', 'B5'],  

#     'C':['C1', 'C2', 'C3', 'C4', 'C5'],  

#     'D':['D1', 'D2', 'D3', 'D4', 'D5'],  

#     'E':['E1', 'E2', 'E3', 'E4', 'E5'] } 

  

# # Convert the dictionary into DataFrame  

# df = pd.DataFrame(data) 

  

# print(df.shape)



# # Remove two columns name is 'C' and 'D' 

# df.drop(['C', 'D'], axis = 1, inplace=True) 



# print(df.shape)