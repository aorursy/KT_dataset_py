import numpy as np 
Data = [[1,2,3],[4,5,6],[7,8,9]]
Array = np.array(Data)

Array
np.array(Data)[0]
np.array(Data)[2,2]
np.arange(0,100) #this function stores 0 to 100 values as arrays
np.arange(0,100,5)
np.zeros(100) #Create array of 100's of 0
np.ones(100) #Create an array of 100 1's
np.zeros((25,4)) #the first number tells the number of columns. the second number indicates the number of lines
np.eye(10)
np.linspace(0,100,5) # Divide between 0 and 100 into 5 equal parts and show
np.random.randint(0,100) #Generate a random number between 0 and 100 . Also the two are equal np.random.randint (0,100) = np.random.randint (100)
np.random.randint(0,100,5) #Generate 5 random numbers between 0 and 100
np.random.rand(5)
np.random.randn(5)#Generate 5 values ​​between -1 and 1 with 'gauss distribution'
Data1 = np.arange(225)
Data1.reshape(15,15)#RESHAPE method allows us to give the data the shape we want and converts it into a matrix
Data1.cumsum() #Sum up the numbers cumulatively

Data1.min() # min value
Data1.max()# max value
Data1.sum() # sum up all the data
Data1.argmax() #this function will give the index of the largest number
Data1.argmin() #this function will give the index of the smallest number
Data2 = np.array([[1,2],[3,4]])

np.linalg.det(Data2)
stddev = np.std(Data1)# this function finds Standard deviation.

stddev
variance = np.var(Data1) #this function finds Standard variance

variance
Data1 = np.arange(0,100)
Data1[:6]#Show from 0 to 6th index is also equivalent to this code = Data1[0,6]
Data1[::3]#Skip 3 jump from start to finish
Data1[:1] = 10 #Save the numbers from the beginning to the 2nd index as 10
Data1 > 40 
NewArray = np.array([5,10,15,20,25])

NewArray1 = np.array([1,2,3,4,5])
NewArray + NewArray1
NewArray - NewArray1
NewArray / NewArray1
NewArray * NewArray1
NewArray ** NewArray1
np.sqrt(NewArray)
a = [3,6,9]

b = np.array(a)

b.T
b = np.array([a])

b.T
import pandas as pd
Label_list = ['I','am','Learning','Data','Science']

Data_List = [1,2,3,4,5]

Pd_Series1 = pd.Series(Data_List,Label_list)

Pd_Series1
data = np.array(['a','b','c','d'])

Series = pd.Series(data,[100,101,102,103])

Series
DataDict = {'Michael_s exam result': 35, 'Olivia_s exam result': 85}

A = pd.Series(DataDict)

DataDict2 = {'Michael_s exam result': 44}

B = pd.Series(DataDict2)

DataDict3 = {'Darth_Vader_s exam result' :99}

C = pd.Series(DataDict3)
A
B
C
A + B
DataDict4 = C.append(A)

DataDict4
A['Michael_s exam result']
C['Darth_Vader_s exam result']
from numpy.random import randn

df = pd.DataFrame(data = randn(5,5), index = ['A','B','C','D','E'], columns = ['Columns1','Columns2','Columns3','Columns4','Columns5'])

df
df[['Columns1','Columns5']]
df['Columns6'] = pd.Series(randn(5),['A','B','C','D','E'])

df
df['Columns7'] = (df['Columns6'] + df['Columns4'] - df['Columns1'] ) / df['Columns2'] * df['Columns3']

df
df.drop('Columns2', axis = 1, inplace = True)

df
df.index.names
df.columns.names
df.loc['C']
df.loc['A']
df.iloc[0]
df.loc['A','Columns5']
df.loc[['A','Columns5'] and ['B','Columns4']]
df.loc[['A','D'],['Columns4','Columns2']]
df
df > 0.2
booleanDF = df > 0

booleanDF
df[booleanDF]
df[df > 0.5]
df['Columns1'] > 0
df[df['Columns3']> 0]
df[df['Columns4']> 0]
df[df['Columns5']> 0]
df[df['Columns1']> 0]
df[(df['Columns1']> 0) & (df['Columns3']> 0)]
df[(df['Columns1']> 0) & (df['Columns3']> 0)]
df
df['Columns6'] = ['NewValue1','NewValue2','NewValue3','NewValue4','NewValue5']

df
df.set_index('Columns6' , inplace = True)

df
df.index.names
df.columns.names
OuterIndex = ['Group1','Group1','Group1','Group2','Group2','Group2','Group3','Group3','Group3']

InnerIndex = ['Index1','Index2','Index3','Index1','Index2','Index3','Index1','Index2','Index3']

list(zip(OuterIndex,InnerIndex))
hierarchy = list(zip(OuterIndex,InnerIndex))

hierarchy = pd.MultiIndex.from_tuples(hierarchy)

hierarchy
df = pd.DataFrame(randn(9,3),hierarchy,columns = ['Column1','Colum2','Column3'])

df
df['Column1']
df.loc['Group1']
df.loc[['Group1','Group2']]
df.loc['Group1']
df.loc[['Group1','Group2']]
df.loc['Group1'].loc['Index1']
df.loc['Group1'].loc['Index1']['Column1']
df.index.names = ['Groups','Indexes']

df
df.xs('Group1') # df.xs('Group1') = df.loc['Group1']
df.xs('Group1').xs('Index1')
df.xs('Group1').xs('Index1').xs('Column1')
df.xs('Index1', level = 'Indexes')
df.xs('Index1', level = 'Indexes')['Column1']
arr = np.array([[10,20,np.nan],[3,np.nan,np.nan],[13,np.nan,4]])

arr
df = pd.DataFrame(arr, index = ['Index1','Index2','Index3'],columns = ['Column1','Column2','Column3'] )

df
df.dropna()
df.dropna(axis = 1)
df.dropna(thresh = 2)
df.fillna(value = 0)
df.sum()
df.sum().sum()
df.fillna(value = (df.sum().sum())/ 5)
df.size
df.isnull()
df.isnull().sum()
df.isnull().sum().sum()
df.size - df.isnull().sum()
data = {'Job': ['Data Mining','CEO','Lawyer','Lawyer','Data Mining','CEO'],'Labouring': ['Immanuel','Jeff','Olivia','Maria','Walker','Obi-Wan'], 'Salary': [4500,30000,6000,5250,5000,35000]}

data
df = pd.DataFrame(data)

df
SalaryGroupBy = df.groupby('Salary')

SalaryGroupBy
SalaryGroupBy.sum()
SalaryGroupBy.min()
SalaryGroupBy.max()
df.groupby('Salary').sum()
df.groupby('Job').sum().loc['CEO'] #Total of salaries of CEOs.
df.groupby('Job').count()
df.groupby('Job').min()
df.groupby('Job').min()['Salary']
df.groupby('Job').min()['Salary']['Lawyer']
df.groupby('Job').mean()['Salary']['CEO']
data = {'A': ['A1','A2','A3','A4'],'B': ['B1','B2','B3','B4'],'C': ['C1','C2','C3','C4']}

data1 = {'A': ['A5','A6','A7','A8'],'B': ['B5','B6','B7','B8'],'C': ['C5','C6','C7','C8']}

df1 = pd.DataFrame(data, index = [1,2,3,4])

df2 = pd.DataFrame(data1, index = [5,6,7,8])

df1
df2
pd.concat([df1,df2])
pd.concat([df1,df2], axis = 1)
data1 = {

        'id': ['1', '2', '3', '4', '5'],

        'Feature1': ['A', 'C', 'E', 'G', 'I'],

        'Feature2': ['B', 'D', 'F', 'H', 'J']}
df1 = pd.DataFrame(data1, columns = ['id', 'Feature1', 'Feature2'])

df1
data2 = {

        'id': ['1', '2', '6', '7', '8'],

        'Feature1': ['K', 'M', 'O', 'Q', 'S'],

        'Feature2': ['L', 'N', 'P', 'R', 'T']}
data = {'A': ['A1','A2','A3','A4'],'B': ['B1','B2','B3','B4'],'C': ['C1','C2','C3','C4']}

data1 = {'A': ['A5','A6','A7','A8'],'B': ['B5','B6','B7','B8'],'C': ['C5','C6','C7','C8']}

df1 = pd.DataFrame(data, index = [1,2,3,4])

df2 = pd.DataFrame(data1, index = [5,6,7,8])

df1
df2 = pd.DataFrame(data2, columns = ['id', 'Feature1', 'Feature2'])

df2
df1.join(df2)
df2.join(df1)
df1.join(df2, how = 'right')
df1.join(df2, how ='outer')
df1.join(df2, how = 'inner')
df1.join(df2, sort = 'True')
df1.join(df2, sort = 'False')
frames = [df1,df2]

df_keys = pd.concat(frames, keys=['x', 'y'])

df_keys
dataset1 = {'A':['A1','A2','A3'], 'B': ['B1','B2','B3'], 'Key': ['K1','K2','K3']}

dataset2 = {'X':['X1','X2','X3','X4'], 'Y': ['Y1','Y2','Y3','Y4'], 'Key': ['K1','K2','K3','K4']}

df1 = pd.DataFrame(dataset1,index = [1,2,3])

df2 = pd.DataFrame(dataset2,index = [1,2,3,4])

df1
df2
pd.merge(df1,df2, on = 'Key')
pd.merge(df1,df2, on = 'Key', how = 'left')
pd.merge(df1,df2, on = 'Key', how = 'right')
pd.merge(df1,df2, on = 'Key', how = 'outer')
pd.merge(df1,df2, on = 'Key', how = 'inner')
pd.merge(df1,df2, on = 'Key', how = 'right',right_index=True)
pd.merge(df1,df2, left_index=True, right_index=True, how='outer')
pd.merge(df1,df2, left_index=True, right_index=True, how='inner')
data = {'Column1': [1,2,3,4,5,6], 'Column2': [1000,1000,2000,3000,3000,1000],'Column3': ['Mace Windu','Darth Vader','Palpatine','Kylo Ren','Rey','Obi-Wan']}

df = pd.DataFrame(data)

df
df.head()
df.head(n = 2) # df.head(2) = df.head(n = 2)
df.describe()
df.describe().T
df.info
df['Column2'].unique()
df['Column2'].nunique()
df['Column2'].value_counts()
df[df['Column1'] >= 2]
df[(df['Column1'] >= 1) & (df['Column2'] == 1000) ]
def square(x): 

    return x ** 2

square(2)
df['Column2'].apply(square)
lambda x : x **2
df['Column2'].apply(lambda x : x **2 )
# df['Column2'] = df['Column2'].apply(square) for code's update
df['Column3']
df['Column3'].apply(len) 

# Get the length of the string
df.drop('Column3', axis = 1 )
df.index
df.index.names
df
df.sort_values(by=['Column1', 'Column3'])# More than one value can be 'Sort'.
df.sort_values('Column2', ascending = True) #from small to large
df.sort_values('Column2', ascending = False) # from large to small
df.sort_values('Column1', kind = 'heapsort')
df.sort_values('Column1', kind = 'mergesort')
df.sort_values('Column1', kind = 'quicksort')
df.sort_values('Column1', na_position = 'first')
df.sort_values('Column1', na_position = 'last')
df = pd.DataFrame({'Month': ['January','February','March','January','February','March','January','February','March'],'State':['New York','New York','New York','Texas','Texas','Texas','Washington','Washington','Washington'],

'moisture': [20,25,65,34,56,85,21,56,79]})

df
df.corr()
df.pivot_table(index = 'Month', columns = 'State', values = 'moisture')
df.pivot_table(index = 'State', columns = 'Month', values = 'moisture')
print('Have a Nice Day')