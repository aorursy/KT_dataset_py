import pandas as pd

import numpy as np  # numpy is not necessary for pandas, but we will use some np code in this example

# in general it's good practice to import all pacakages at the beginning
# first let's look at series - think of this as a single column of a spreadsheet

# each entry in a series corresponds to an individual row in the spreadsheet

# we can create a series by converting a list, or numpy array



mylist = [5.4,6.1,1.7,99.8]

myarray = np.array(mylist)

myseries1 = pd.Series(data=mylist)

print(myseries1)

myseries2 = pd.Series(data=myarray)

print(myseries2)
# we access individual entries the same way as with lists and arrays

print(myseries1[2])
# we can add labels to the entries of a series



mylabels = ['first','second','third','fourth']

myseries3 = pd.Series(data=mylist,index=mylabels)

print(myseries3)
# we need not be explicit about the entries of pd.Series

myseries4 = pd.Series(mylist,mylabels)

print(myseries4)
# we can also access entries using the index labels

print(myseries4['second'])
# we can do math on series 

myseries5 = pd.Series([5.5,1.1,8.8,1.6],['first','third','fourth','fifth'])

print(myseries5)

print('')

print(myseries5+myseries4)
# we can combine series to create a dataframe using the concat function

df1 = pd.concat([myseries4,myseries5],axis=1,sort=False)

df1
# we can create a new dataframe 

df2 = pd.DataFrame(np.random.randn(5,5))

df2
# lets give labels to rows and columns

df3 = pd.DataFrame(np.random.randn(5,5),index=['first row','second row','third row','fourth row','fifth row'],

                   columns=['first col','second col','third col','fourth col','fifth col'])

df3
# we can access individual series in a data frame

print(df3['second col'])

print('')

df3[['third col','first col']]
# we can access rows of a dataframe

df3.loc['fourth row']
df3.iloc[2]
df3.loc[['fourth row','first row'],['second col','third col']]
# we can use logical indexing for dataframes just like for numpy arrays

df3>0
print(df3[df3>0])
# we can add columns to a dataframe

df3['sixth col'] = np.random.randn(5,1)

df3
# we can remove columns or rows from a dataframe

df3.drop('first col',axis=1,inplace=True)
df3
#df4 = df3.drop('first col',axis=1)

#df4
df5 = df3.drop('second row',axis=0)

df5
# we can remove a dataframe's index labels

df5.reset_index()
df5
df5.reset_index(inplace=True)

df5
# we can assign new names to the index

df5['new name'] = ['This','is','the','row']

df5

df5.set_index('new name',inplace=True)

df5




df7 = pd.DataFrame({"customer":['101','102','103','104'], 

                    'category': ['cat2','cat2','cat1','cat3'],

                    'important': ['yes','no','yes','yes'],

                    'sales': [123,52,214,663]},index=[0,1,2,3])



df8 = pd.DataFrame({"customer":['101','103','104','105'], 

                    'color': ['yellow','green','green','blue'],

                    'distance': [12,9,44,21],

                    'sales': [123,214,663,331]},index=[4,5,6,7])
pd.concat([df7,df8],axis=0,sort=False)
pd.concat([df7,df8],axis=0,sort=True)
pd.concat([df7,df8],axis=1,sort=False)
pd.merge(df7,df8,how='outer',on='customer') # outer merge is union of on
pd.merge(df7,df8,how='inner',on='customer') # inner merge is intersection of on
pd.merge(df7,df8,how='right',on='customer') # left merge is just first on, but all columns ... right is second
df9 = pd.DataFrame({'Q1': [101,102,103],

                    'Q2': [201,202,203]},

                   index=['I0','I1','I2'])



df10 = pd.DataFrame({'Q3': [301,302,303],

                    'Q4': [401,402,403]},

                   index=['I0','I2','I3'])
# join behaves just like merge, 

# except instead of using the values of one of the columns 

# to combine data frames, it uses the index labels

df9.join(df10,how='right') # outer, inner, left, and right work the same as merge
# let's now go over a few more basic functialities of pandas



df8['color'].unique()
df8['color'].value_counts()
df9.mean()
df8.columns
df8
new_df = df8[(df8['customer']!='105') & (df8['color']!='green')]

new_df
print(df8['sales'].sum())

print(df8['distance'].min())

def profit(s):

    return s*0.5 # 50% markup...
df8['sales'].apply(profit)
df8['color'].apply(len)
df11 = df8[['distance','sales']]

df11.applymap(profit)
def col_sum(co):

    return sum(co)

df11.apply(col_sum)
#df11.applymap(col_sum)
del df8['color']

df8
df8.index
df8.sort_values(by='distance',inplace=True)

df8
df8
# if some series has multiple of the same value then we can group all the unique entries together

mydict = {'customer': ['Customer 1','Customer 1','Customer2','Customer2','Customer3','Customer3'], 

          'product1': [1.1,2.1,3.8,4.2,5.5,6.9],

          'product2': [8.2,9.1,11.1,5.2,44.66,983]}

df6 = pd.DataFrame(mydict,index=['Purchase 1','Purchase 2','Purchase 3','Purchase 4','Purchase 5','Purchase 6'])

df6
grouped_data = df6.groupby('customer')

print(grouped_data)
grouped_data.std()
df8
# similar to numpy arrays, we can also save and load dataframes to csv files, and also Excel files



df8.to_csv('df8.csv',index=True)
new_df8 = pd.read_csv('df8.csv',index_col=0)

new_df8
df8.to_excel('df8.xlsx',index=False,sheet_name='first sheet')

newer_df8 = pd.read_excel('df8.xlsx',sheet_name='first sheet',index_col=1)

newer_df8

