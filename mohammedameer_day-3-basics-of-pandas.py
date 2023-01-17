def res(a,b):
    print(a+b)
    
res(2,3)

# res is a function
a =1
# a is a variable
import pandas as pd
# Whwnever we import a library in python, it is treated as an object
# Libraries are of 2 types : In built library (random,os,time), User defined library (pandas,numpy, matplotlib)
# 2 main objects
# 1. Dataframe - 2 Dimension ( Rows * column)
# 2. Series - 1 Dimension ( Column)
list1 = [1,2,3,4]
pd.DataFrame(list1)
list1 = [['Ameer','Pranil','Shivraj','Erica'],[23,17,18,19]]
pd.DataFrame(list1)
list1 = [['Ameer',23],['Pranil',17],['Shivraj',18],['Erica',19]]
pd.DataFrame(list1)
# Datasets can be of various types (formats)
# 1. csv - comma separated values - pd.read_csv()
# 2. tsv - tab separated values - pd.read_table()
# 3. txt - Text Files - 
# 4. json - Javascript Object Notation - 

# NOTE : Read those datasets and convert them into DataFrame
df1 = pd.read_csv('../input/solar-power-generation-data/Plant_1_Generation_Data.csv')
df2 = pd.read_csv('../input/solar-power-generation-data/Plant_1_Weather_Sensor_Data.csv')
df1
# Scalars (0 Dimension), Vector (1 Dimension), Matrix (2 Dimension)
df.ndim  # Attritube of an object df - N Dimension
df.shape # Another attribute
df.size # Another attribute representing the total values
68778* 7
df['DC_POWER']
type(df['DC_POWER'])
type(df)
df.index
df.columns
df.notnull().count()
df.dtypes
# In Pandas, String datatype is represnted as Object
df.info()  # Summary of the dataset
df['PLANT_ID'].unique() # Unique values of Plant_ID
df['SOURCE_KEY'].unique() # Displaying unique number of Inverters
df['SOURCE_KEY'].nunique() # Total number of unqiue inverters
df.nunique() # Gives total number of unique values in every column
df['SOURCE_KEY'].value_counts()
# Basic Statistics
# Mean, Median, Mode, Range
df['TOTAL_YIELD'].mean()
df['TOTAL_YIELD'].max()
df['TOTAL_YIELD'].min()
df.describe()
6.512003e+06
# Percentile :
df1['SOURCE_KEY']==df1['SOURCE_KEY'][0] 

# Taking one specific Inverter 
# Mentions TRUE when you that particular inverter is mentioned, else it is FALSE
one_inv = df1[df1['SOURCE_KEY']==df1['SOURCE_KEY'][0]]
one_inv
# Mention the complete dataset considering one inverter
one_inv.sum()['DAILY_YIELD']

# Total daily yield from one particular Inverter
abc = df1[df1['SOURCE_KEY']==df1['SOURCE_KEY'][0]][['AC_POWER','DC_POWER']]

# Taking 3154 values of AC and DC Power based on one specific INverter

import matplotlib.pyplot as plt
plt.scatter(abc['AC_POWER'],abc['DC_POWER'])
plt.xlabel('AC_POWER')
plt.ylabel('DC_POWER')
plt.title('AC vs DC')

# This is called Postive Correlation because there is a linear relationship in the increasing order between ac and dc power
abc['AC_POWER'].corr(abc['DC_POWER'])  # Finding correlation between 2 columns
# If greater than 0, positive correlation
# If lesser than 0, negative correlation

df1.corr()
df1[df1['AC_POWER']==df1['AC_POWER']]
df1[df1['AC_POWER']==df1['AC_POWER'].max()]
df1[df1['AC_POWER']==df1['AC_POWER'].max()].style.highlight_max(color='red')
