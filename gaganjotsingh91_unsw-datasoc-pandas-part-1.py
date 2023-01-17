# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os



# Any results you write to the current directory are saved as output.
'''

Setup file for Intro to Pandas.

Creates a series and prints it.

@Author - Gaganjot Singh on behalf on DataSoc

@Date - March 17, 2019

'''





'''

Creates a series with default index and prints it

'''

def series_1():

    s1_data = [2098.71, 5356.83, 8987.65, 5634.12]

    s1 = pd.Series(data=s1_data)

    print(s1)

    print()



'''

creates a series with labelled index and prints it

'''

def series_2():

    s2_data = [2098.71, 5356.83, 8987.65, 5634.12]

    s2_idx = [2006, 2007, 2008, 2009]

    s2 = pd.Series(data=s2_data , index= s2_idx , name='Sales Data')

    print(s2)

    print()



###Code to demonstrate Series

#series_1()

#series_2()
'''

Setup file for Intro to Pandas.

Creates a Dataframe and prints it.

@Author - Gaganjot Singh on behalf on DataSoc

@Date - March 17, 2019

'''



'''

creates a DataFrame from a dictionary and prints it.

'''

def df_1():

    df1_data = {'Randwick Store': [2563.89, 2098.18, 3013.45, 3856.99] ,

                 'Maroubra Store': [1887.35, 2098.65, 2562.77, 3298.99]}

    df1 = pd.DataFrame(data = df1_data)

    print(df1.head())

    print()



'''

creates a DataFrame from a dictionary using labelled index for rows and prints it

'''

def df_2():

    df2_data = {'Randwick Store': [2563.89, 2098.18, 3013.45, 3856.99],

                 'Maroubra Store': [1887.35, 2098.65, 2562.77, 3298.99]}

    df2_idx = [2006, 2007, 2008, 2009]

    df2 = pd.DataFrame(data=df2_data, index=df2_idx)

    print(df2.head())

    print()

###Code to demonstrate DataFrame

#df_1()

#df_2()
'''

Code to create a series from a list

'''

series_example = pd.Series(['Gagan','Sharon','Chris','Saksham','Dean'])

series_example
'''

Code to create a series from a list with index

'''

series_example = pd.Series(['Gagan','Sharon','Chris','Saksham','Dean'], 

                           index =['PG_Education','PG_VP','President','IT','Arc'])

###series_example
'''

uncomment the following lines to see how the labels can help you

'''



###series_example[0]

###series_example['President']
'''

Code to create a series from a list with index and a name

'''

series_example = pd.Series(['Gagan','Sharon','Chris','Saksham','Dean'], 

                           index =['PG_Education','PG_VP','President','IT','Arc'], name='DataSocTeam')

#series_example
'''

Another way of creating a pandas series from a list

'''

datasoc_team = ['Gagan','Sharon','Chris','Saksham','Dean']

datasoc_series = pd.Series(datasoc_team)
###datasoc_series
dataframe_example_1= pd.DataFrame( {'Gagan' :['PG_Education','MIT'] , 

                                    'Sharon':['PG_VP','MIT'],

                                    'Chris' :['President','UG Comp Sci and Commerce'],

                                    'Saksham':['IT','DS UG'],

                                    'Dean' :['Arc','DS UG'] })
dataframe_example_1
dataframe_example_1= pd.DataFrame( {'Gagan' :['PG_Education','MIT'] , 

                                    'Sharon':['PG_VP','MIT'],

                                    'Chris' :['President','UG Comp Sci and Commerce'],

                                    'Saksham':['IT','DS UG'],

                                    'Dean' :['Arc','DS UG'] }, index=['Post','Degree'])

dataframe_example_1

##interesting observation - The data makes more sense with Post and Degree being the columns, and all other being rows

##this can be done by simply below

dataframe_example_1.T

### below method works as well 

###dataframe_example_1.transpose()
datasoc_dict={'Gagan' :['PG_Education','MIT'] , 

                                    'Sharon':['PG_VP','MIT'],

                                    'Chris' :['President','UG Comp Sci and Commerce'],

                                    'Saksham':['IT','DS UG'],

                                    'Dean' :['Arc','DS UG'] }

dataframe_example_2 = pd.DataFrame(data=datasoc_dict , index = ['Post','Degree'])

##dataframe_example_2
array_2d=[['Nikita', 'Ellen'],['VP_Internal','Secretary'],['UG DS','UG DS'] ]

df_3 = pd.DataFrame(array_2d)

df_3
df_4 = pd.DataFrame( data =array_2d , index =['Name','Post','Degree'] )

df_4
'''

Code to read data from csv file

'''

df = pd.read_csv("../input/fifa19/data.csv")

#df.head()
'''

Code to read data from csv file

'''

df = pd.read_csv("../input/fifa19/data.csv" , index_col=0)

#df.head()
datasoc_dict={'Gagan' :['PG_Education','MIT'] , 

                                    'Sharon':['PG_VP','MIT'],

                                    'Chris' :['President','UG Comp Sci and Commerce'],

                                    'Saksham':['IT','DS UG'],

                                    'Dean' :['Arc','DS UG'] }

dataframe_example_2 = pd.DataFrame(data=datasoc_dict , index = ['Post','Degree'])

dataframe_example_2.to_csv("datasoc.csv")
df = pd.read_csv("datasoc.csv" , index_col=0)

#df.head()
#your code for question 1 here
#your code for question 2 here