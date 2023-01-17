#Eye

import numpy as np

import pandas as pd

np.eye(2, dtype = int).reshape(4, 1).T 
#Vectorize

import numpy as np

a = np.arange(10)

f = np.vectorize(lambda x: x+1)

f(a)
#Full

np.full((2,2), 1) == np.ones((2,2))
#Percentile

import numpy as np

a = np.array( [7, 10, 2, 4, 13, 16])

med = np.percentile(a, 50)

print(med)
#Pandas Series

import pandas as pd

S1 = pd.Series([0, 2, 4, 6, 8, 10, 12])

S2 = pd.Series([0, 3, 6, 9, 12, 15])

print(S1)

print(S2)
#Pandas Series

import pandas as pd

S1 = pd.Series([0, 2, 4, 6, 8, 10, 12])

S2 = pd.Series([0, 3, 6, 9, 12, 15])

S3=S1[S1.isin(S2)]

print(S3)
#Combining Series

#Which of the following can combine two series s1 and s2 to form a dataframe df?

df = S1.append(S1)

print(df)
#Combining Series

import pandas as pd

df = pd.concat([S1, S2], 1)

print(df)

#The following command will combine the two series s1 and s2 to make a dataframe df.
#Combining Series

df = pd.DataFrame({'col1': S1, 'col2': S2})

print(df)

#Feedback :

#The following command will combine series s1 and s2 to make a dataframe with the columns col1 and col2.
#Combining Series

#Which of the following can combine two series s1 and s2 to form a dataframe df?

df = pd.Series({'col1': S1, 'col2': S2})

print(df)
#Pandas Series

S3=S1[~S1.isin(S2)]

print(S3)

#S1[S1.isin(S2)] returns the elements that are present both in S1 and S2. If you precede this command with a 

#‘not’, i.e. ‘~’ sign, it returns the elements in S1 that aren’t present in S2.
#Pandas Series

S4=S1[S1.isin(~S2)]

print(S4)
#Pandas Series

S5=S2[S1.isin(~S1)]

print(S5)
#Pandas Series

import pandas as pd

S1 = pd.Series([0, 2, 4, 6, 8, 10, 12])

S2 = pd.Series([0, 3, 6, 9, 12, 15])

S3=S1[S1.isin(S2)].index == S2[S2.isin(S1)].index

print(S3)

#S1[S1.isin(S2)].index returns the indices in S1 at which the element in S1 is also present in S2. Hence, the elements 0, 6, 

#and 12 are equal in both the series. And these elements in S1 are present at indices 0, 3, and 6.

#Similarly, S2[S2.isin(S1)].index returns the indices in S2 at which the element in S2 is also present in S1. Hence, the 

#elements 0, 6, and 12 are equal in both the series. And these elements in S2 are present at indices 0, 2, and 4.

#So when an ‘==’ operator is used between them, it is comparing two lists which are [0, 3, 6] and [0, 2, 4] respectively. 

#And these lists are checked for equality element-wise. Hence, the final output obtained becomes:

#[True, False, False]
#Pandas Series Creation

import pandas as pd

import numpy as np

res=pd.Series(np.array(range(0,9))**2, index = range(0,9))

print(res)
#Pandas Series Creation

import pandas as pd

import numpy as np

res=pd.Series(np.array(range(1,9))**2, index = range(1,9))

print(res)
#Pandas Series Creation

import pandas as pd

import numpy as np

res=pd.Series(np.array(range(0,10))**2, index = range(0,10))

print(res)
#Pandas Series Creation

import pandas as pd

import numpy as np

res=pd.Series(np.array(range(1,10))**2, index = range(1,10))

print(res)

#The np.array(range(1,10))**2 command returns the squares of numbers from 1 to 9, 

#and index = range(1, 10) sets the index explicitly from 1 to 9 (by default the indexing would have been 0 to 8). 
import pandas as pd

import numpy as np



epl1 = pd.DataFrame( {     

"Teams" : ["Chelsea", "Tottenham", "ManCity"] ,           

"Played" : [38, 38, 38] ,

"Won" : [30, 26, 23]

                    } ) 

epl1

epl1.dtypes
epl2 = pd.DataFrame( {     

"Teams" : ["Liverpool", "Arsenal", "ManUtd"] ,           

"Played" : [38, 38, 38] ,

"Won" : [22, 23, 18]

                    } ) 

epl2


epl_addn = pd.DataFrame( {               

"Draws" : [3, 8, 9, 10, 6, 15] ,

"Points" : [93, 86, 78, 76, 75, 69]

                    } ) 

epl_addn
pd.concat([epl1.append(epl2, ignore_index = False), epl_addn], axis = 0)
pd.concat([epl1.append(epl2, ignore_index = True), epl_addn], axis = 0)

#/home/nbuser/anaconda2_501/lib/python2.7/site-packages/ipykernel/__main__.py:1: FutureWarning: 

#Sorting because non-concatenation axis is not aligned. A future version

#of pandas will change to not sort by default.

#To accept the future behavior, pass 'sort=False'.

#To retain the current behavior and silence the warning, pass 'sort=True'.

#  if __name__ == '__main__':
#pd.concat([epl1.append(epl2, ignore_index = False), epl_addn], axis = 1)

#InvalidIndexError: Reindexing only valid with uniquely valued Index objects
pd.concat([epl1.append(epl2, ignore_index = True), epl_addn], axis = 1)

#You first need to combine the rows of the first two dataframes. For that, you can simply use epl1.append(epl2, 

#ignore_index = True). Then, you need to combine columns of the dataframe obtained from the operation performed 

#and the third dataframe, i.e. ‘epl_addn’. This can be done using pd.concat with the axis argument set to 1. 

#Hence, the final command which achieves this is:

#pd.concat([epl1.append(epl2, ignore_index = True), epl_addn], axis = 1)
final = pd.DataFrame( {    

"League" : ["EPL", "La Liga","La Liga", "EPL","EPL","La Liga","EPL","EPL","La Liga","EPL","La Liga","La Liga"] , 

"Teams" : ["Chelsea","Real Madrid","Barcelona","Tottenham","ManCity","Atletico Madrid","Liverpool","Arsenal","Sevilla","ManUtd","Villarreal","RS"] ,           

"Played" : [38, 38, 38,38, 38, 38,38, 38, 38,38, 38, 38] ,

"Won" : [30, 29, 28, 26, 23, 23, 22, 23, 21, 18, 19, 19],

"Draws" : [3, 6, 6, 8, 9, 9, 10, 6, 9, 15, 10, 7] ,

"Points" : [93, 93, 90, 86, 78, 78, 76, 75, 72, 69, 67, 64]   

                    } ) 

final
final.pivot_table(values = 'League', index = 'Points', aggfunc = 'sum')
final.pivot_table(values = 'Points', index = 'League', aggfunc = 'sum')

#Correct! When you use the ‘pivot_table’ to group your dataframe, you need to set the index to the column for which

#you want to group (in this case, ‘League’), ‘values’ to the column on which you need to perform any function 

#(in this case, ‘Points’) and ‘aggfunc’ to the function which you need to 

#perform (in this case, ‘sum’).
final.groupby('League')['Points'].sum()

final.groupby('League')['Points'].sum()

#When you use ‘groupby’ to group your dataframe, you need to provide the column by which you want to group the 

#dataframe as an argument to groupby (in this case, ‘League’). When you perform this groupby, you get a

#dataframe and thus, can access any column in the dataframe by using square brackets as usual. And then, 

#you perform the sum() operation on this dataframe to find the sum of the ‘Points’ 

#for different leagues.
final.groupby('Points')['League'].sum()
final.loc[(final.Points > 75) & (final.Won > 23)]
final.loc[(final.Points >= 75) & (final.Won > 23)]
final.loc[(final.Points > 75) & (final.Won >= 23)]

#You need to perform conditional indexing as given above to retain the rows satisfying the given conditions.

#Also, you need to give an ‘&’ between the given conditions in order to specify that both the 

#conditions should be met simultaneously.
final.loc[(final.Points >= 75) & (final.Won >= 23)]
final.loc[(final.League == 'EPL')].sort_values(by = 'Points', ascending = True)[:4]
final.loc[(final.League == 'EPL')].sort_values(by = 'Points', ascending = True)[:3]
final.loc[(final.League == 'EPL')].sort_values(by = 'Points', ascending = False)[:4]

#Feedback :

#To retain the teams belonging to the league ‘EPL’, you need to perform conditional indexing as given below:

#final.loc[(final.League == ‘EPL’)]

#Now, this dataframe should be sorted in descending order of points in order to obtain the top four teams.

#This can be done using:

#sort_values(by = ‘Points’, ascending = False)

#And finally, to obtain the top four teams, you need to specify the index as [:4]. Hence, the final command is:

#final.loc[(final.League == 'EPL')].sort_values(by = 'Points', ascending = False)[:4]
final.loc[(final.League == 'EPL')].sort_values(by = 'Points', ascending = False)[:3]
#Drop Duplicate

import pandas as pd

import numpy as np

df = pd.DataFrame( {     

"Employee" : ["Susan", "Bart", "Emily", "Charles", "David", "Charles", "Julia", "Bart","Bart"] ,           

"City" : ["London", "London", "Philadelphia", "London", "London", "Philadelphia", "London", "Philadelphia","Philadelphia"] ,

"Age" : [20, 40, 18, 24, 37, 40, 44, 20, 20 ],

"Hours" : [24, 40, 50, 36, 54, 44, 41, 35, 35]} ) 

df
#Drop Duplicate

df1=df.drop_duplicates(subset = None, keep = 'first', inplace = True)

print(df1)

df.drop_duplicates(subset = None, keep = 'first', inplace = True)

#Let's see what the different attributes of the drop_duplicates command mean:

#subset: Whether you want to drop duplicate values based on certain columns only. If 'subset' is set to None,

#it would identify those rows as a duplicate for which the values in every column is the same.

#keep: Specifies if you want to retain certain values. If it's 'first', it will retain the first

#value of that duplicate in the dataframe. If it's set to 'last', it will retain the last value.

#If it's False, it will drop all the rows.

#inplace: Specifies whether you want to execute this command in place.
df2=df.drop_duplicates(subset = None, keep = 'last', inplace = True)

print(df2)

#Let's see what the different attributes of the drop_duplicates command mean:

#subset: Whether you want to drop duplicate values based on certain columns only. 

#If 'subset' is set to None, it would identify those rows as a duplicate for which the values in every

#column is the same keep: Specifies if you want to retain certain values. If it's 'first', it will retain 

#the first value of that duplicate in the dataframe. If it's set to 'last', it will retain the last value.

#If it's False, it will drop all the rows. inplace: Specifies whether you want to execute this command in place.
df3=df.drop_duplicates(subset = None, keep = False, inplace = True)

print(df3)
df2=df.drop_duplicates(subset = None, keep = False, inplace = False)

print(df2)
import pandas as pd

import numpy as np

df = pd.DataFrame( {     

"Employee" : ["Susan", "Kevin", "Charles", "David", "Ben"] ,           

"Salary" : [60000, 35000, 31000, 10000, 20000] ,

"Year" : [2019, 2019, 2019, 2019, 2019],

"Age" : [32, 35, 27, 24, 29],

"Bld Grp" : ["A", "B+", "O+", "O-", "B+"]

} ) 

df
df = df.rename(columns = {'Bld Grp':'Blood Group'})

print(df)

#The df.rename() function can be used for this purpose.
#df = df.rename(cols = {'Bld Grp':'Blood Group'})

#print(df)

#TypeError: rename() got an unexpected keyword argument "cols"
df = pd.DataFrame( {     

"Employee" : ["Susan", "Kevin", "Charles", "David", "Ben"] ,           

"Salary" : [60000, 35000, 31000, 10000, 20000] ,

"Year" : [2019, 2019, 2019, 2019, 2019],

"Age" : [32, 35, 27, 24, 29],

"Bld Grp" : ["A", "B+", "O+", "O-", "B+"]

} ) 

df

df.columns.values[4] = 'Blood Group'

df

#df.column.values[4] can be used to access the 5th column, i.e. the column indexed 4 of the dataframe, 

#and then you can change the name to whatever you like.
#df.columns.values[5] = 'Blood Group'

#df

#index 5 is out of bounds for axis 0 with size 5
import pandas as pd

import numpy as np

students = [ ('jack', np.NaN, 'Sydeny' , 'Australia') ,

                 ('Riti', np.NaN, 'Delhi' , 'India' ) ,

                 ('Vikas', 31, np.NaN , 'India' ) ,

                 ('Neelu', 32, 'Bangalore' , 'India' ) ,

                 ('John', 16, 'New York' , 'US') ,

                 ('John' , 11, np.NaN, np.NaN ) ,

                (np.NaN , np.NaN, np.NaN, np.NaN ) 

                 ]

print(students)
#Create a DataFrame object

import pandas as pd

df = pd.DataFrame(students, columns = ['Name' , 'Age', 'City' , 'Country'])

df
df_isnull=df.isnull()

print(df_isnull)
#df_isnull=df.isnull(axis = 0).sum()

#print(df_isnull)

#TypeError: isnull() got an unexpected keyword argument 'axis'
#df_isnull=df.isnull(axis = 1).sum()

#print(df_isnull)

#TypeError: isnull() got an unexpected keyword argument 'axis'
df_isnull=df.isnull().sum(axis = 1)

print(df_isnull)



#The df.isnull() function gives the null values in the whole dataframe. And the sum() function counts the total number.

#By default, it counts the missing values along the columns, i.e. axis = 0. If you want the row-wise NA count,

#just specify the axis as 1.
import pandas as pd

import numpy as np

movies = pd.DataFrame( {     

"Movie Name" : ["Tamil Movie", "English Movie", "Telugu Movie", "Hindu Movie", "Bengali Movie"] ,           

"Total Profit" : [60000, 35000, 31000,1000, 20000] ,

"Year" : [2019, 2010, 2012, 2020, 2019],

"Movie Market" : [32, 35, 27, 24, 29],

"Ratings" : ["A", "B+", "O+", "O-", "B+"]

} ) 

movies
dfs=movies.Ratings

print(dfs)
dfs=movies[['Ratings']]

print(dfs)

#If you specify the column name inside a square bracket in any form of indexing you apply,

#that column will be selected as a dataframe. 

#If the square bracket is not specified, then it gets selected as a series.
dfs=movies.loc[:, ['Ratings']]

print(dfs)

#If you specify the column name inside a square bracket in any form of indexing you apply,

#that column will be selected as a dataframe. 

#If the square bracket is not specified, then it gets selected as a series.
dfs=movies.iloc[:, [2]]

print(dfs)

#If you specify the column name inside a square bracket in any form of indexing you apply, 

#that column will be selected as a dataframe. If the square bracket 

#is not specified, then it gets selected as a series.