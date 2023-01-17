#Import Packages



import numpy as np

import pandas as pd

import os

import matplotlib.pyplot as plt





from IPython.core.interactiveshell import InteractiveShell

InteractiveShell.ast_node_interactivity='all'
# Input data files are available in the "../input/" directory.

# List all files under the input directory



input_path = '/kaggle/input/80-cereals'



for dirpath, dirname, filenames in os.walk(input_path):

    for name in filenames:

        print (os.path.join(dirpath , name))



#read file

fname = 'cereal.csv'

cereal_df = pd.read_csv(os.path.join(input_path , fname))
#Property of Dataframe - returns a tuple representing the dimensionality of the DataFrame.

cereal_df.shape
#A dataframe function to print a concise summary of a DataFrame.

cereal_df.info()

#pandas.DataFrame.head(n=5) - Return the first `n` rows.

cereal_df.head()



#tail return last 'n' rows

cereal_df.tail()
# It returns object of  Type: Index, Length:16(in our case of cereal dataset)

# Immutable ndarray implementing an ordered, sliceable set. The basic object storing axis labels for all pandas objects



cereal_df.columns



cereal_df.columns[0]

cereal_df.columns[1]

cereal_df.columns[2]
#The individual Series that make up the columns of the DataFrame can be accessed

#via dictionary-style indexing of the column name



# Let us find out - How many 'type' of cereals 

cereal_df['type'].unique() #Returns ndarray (Numpy n-dimensional array)



cereal_df['type'].unique().size # size of ndarray
# There are 2 unique types - 'C' and 'H'

# From data description it means, Hot(H) and Cold(C) 
cereal_df[cereal_df['type'] == 'C'].count

cereal_df[cereal_df['type'] == 'C'].count()[0]

cereal_df[cereal_df.type == 'C'].count()[0]

cereal_df.count()[0]
# direct masking operations are interpreted row-wise 

# We use mask    cereal_df['type'] == 'C'



#Let us find out - How many cereals of each type 



cereal_df['name'][cereal_df['type'] == 'C'].count() 

cereal_df['name'][cereal_df['type'] == 'H'].count() 

#So, out of 77, 74 are of type Cold and only 3 are of type Hot.
# Let us peek into few 'Hot' cereals



cereal_df[['name','mfr','type','calories','protein','fat', 'sodium', 'fiber',

       'carbo', 'sugars', 'potass', 'vitamins', 'shelf', 'weight', 'cups',

       'rating']][cereal_df['type'] == 'H'].head()
# 3 Hot cereals are made by different manufacturers (Nabisco, American Home, Quaker).

# No Hot cereal is placed on Top shelf (3)

# We note -1.0 for carbo and -1 for sugar, potasss. This looks like NA value because 0 also present in sugar column.
# Let us peek into few 'Cold' cereals



cereal_df[['name','mfr','type','calories','protein','fat', 'sodium', 'fiber',

       'carbo', 'sugars', 'potass', 'vitamins', 'shelf', 'weight', 'cups',

       'rating']][cereal_df['type'] == 'C'].head()



cereal_df[['name','mfr','type','calories','protein','fat', 'sodium', 'fiber',

       'carbo', 'sugars', 'potass', 'vitamins', 'shelf', 'weight', 'cups',

       'rating']][cereal_df['type'] == 'C'].tail()
# Let us find out - How many 'mfr' of cereals 

cereal_df['mfr'].unique()



cereal_df['mfr'].unique().size
# There are 7 unique manufacturers - 'N', 'Q', 'K', 'R', 'G', 'P', 'A'

# From data description their names are given in next cell. 
#Let us find out - How many cereals of each Manufacturer 

cereal_df['name'][cereal_df['mfr'] == 'N'].count() # N = Nabisco

cereal_df['name'][cereal_df['mfr'] == 'Q'].count() # Q = Quaker Oats

cereal_df['name'][cereal_df['mfr'] == 'K'].count() # K = Kelloggs

cereal_df['name'][cereal_df['mfr'] == 'R'].count() # R = Ralston Purina

cereal_df['name'][cereal_df['mfr'] == 'G'].count() # G = General Mills

cereal_df['name'][cereal_df['mfr'] == 'P'].count() # P = Post

cereal_df['name'][cereal_df['mfr'] == 'A'].count() # A = American Home Food Products
# So out of 77 cereals, 23 are made by 'Kellogs' and 22 are made by 'General Mills'.(Higher values 45 if summed) 

# Then 9 from 'Post', 8 from 'Quaker Oats' and  8 from 'Ralston Purina'.(Next hiher values 25 if summed) (25+45=70)

# Then 6 from 'Nabisco' and 1 from 'Amreican Home food Products' (Lower Values 7 if summed)(70+7=77)
# Let us find out - How many 'shelf' cereals are placed on



cereal_df['shelf'].unique()



cereal_df['shelf'].unique().size
# There are 3 unique shelf numbers - 3, 1, 2

# From data description it means display shelf (1, 2, or 3, counting from the floor). 
#Let us find out - How many cereals of each shelf 



cereal_df['name'][cereal_df['shelf'] == 1].count() 

cereal_df['name'][cereal_df['shelf'] == 2].count() 

cereal_df['name'][cereal_df['shelf'] == 3].count() 
# So out of 77, there are 20 cereals placed on shelf 1

# there are 20 cereals placed on shelf 2, and

# there are 36 cereals placed on shelf 3
#Let us peek into some rows for shelf = 1



cereal_df[['name','mfr','type','calories','protein','fat', 'sodium', 'fiber',

       'carbo', 'sugars', 'potass', 'vitamins', 'shelf', 'weight', 'cups',

       'rating']][cereal_df['shelf'] == 1].head()
#Let us peek into some rows for shelf = 2



cereal_df[['name','mfr','type','calories','protein','fat', 'sodium', 'fiber',

       'carbo', 'sugars', 'potass', 'vitamins', 'shelf', 'weight', 'cups',

       'rating']][cereal_df['shelf'] == 2].head()
#Let us peek into some rows for shelf = 3



cereal_df[['name','mfr','type','calories','protein','fat', 'sodium', 'fiber',

       'carbo', 'sugars', 'potass', 'vitamins', 'shelf', 'weight', 'cups',

       'rating']][cereal_df['shelf'] == 3].head()
#cereal_df['name'].unique()



cereal_df['name'].unique().size
#There are 77 unique names. This column could be used as index for our data frame.

#Right now index is created by pandas. (Rangeindex from 0 to 76 - cereal_df.info() command )

df = pd.read_csv(os.path.join(input_path , fname),index_col='name')

df.shape

df.info()

df.head()
# Now it shows 15 columns

# We will stick with pandas provided Range index only.
cereal_df.describe()
# describe command is useful for numerical data.

# We will use it when analysing Quantitative data.

# Right now we are doing Qualitative data only.
#A Pandas Series is a one-dimensional array of indexed data.

data = pd.Series([1, 2, 3, 4, 4, 3, 4])

print("1. Series")

data



print("2. value_counts() on series")

data.value_counts() #how many times each value occurs



print("3. Access Series with index, values")

data.index

data.values



print("Access Series like a dictionary")

data.keys()

list(data.items())
type_counts = cereal_df['type'].value_counts()

type_counts # Series (index, value)



type_counts.index.values # array of index of Series

type_counts.values # array of value of Series

mfr_counts = cereal_df['mfr'].value_counts()

mfr_counts

mfr_counts.index.values # array of index of Series

mfr_counts.values # array of value of Series



shelf_counts = cereal_df['shelf'].value_counts()

shelf_counts

shelf_counts.index.values # array of index of Series

shelf_counts.values # array of value of Series
plt.style.use('seaborn-whitegrid')





# Get the figure and the axes (or subplots)



fig, (ax0, ax1, ax2) = plt.subplots(nrows=1, ncols=3, figsize=(15, 4))





ax0.bar(type_counts.index.values, type_counts.values, width=0.5, align='center')

ax0.set(title = 'type_counts', xlabel='type' , ylabel = 'Frequency')



ax1.bar(mfr_counts.index.values, mfr_counts.values, width=0.5, align='center')

ax1.set(title = 'mfr_counts', xlabel='mfr' , ylabel = 'Frequency')



ax2.bar(shelf_counts.index.values, shelf_counts.values, width=0.5, align='center')

ax2.set(title = 'shelf_counts', xlabel='shelf' , ylabel = 'Frequency')



# Title the figure

fig.suptitle('Frequency Distribution', fontsize=14, fontweight='bold');
num_rows = cereal_df['name'].count()



rel_freq_type = type_counts.values/num_rows

print(rel_freq_type)



rel_freq_mfr = mfr_counts.values/num_rows

print(rel_freq_mfr)



rel_freq_shelf = shelf_counts.values/num_rows

print(rel_freq_shelf)





plt.style.use('seaborn-whitegrid')



# Get the figure and the axes (or subplots)



fig, (ax0, ax1, ax2) = plt.subplots(nrows=1, ncols=3, figsize=(15, 4))





ax0.bar(type_counts.index.values, rel_freq_type, width=0.5, align='center')

ax0.set(title = 'type_counts', xlabel='type' , ylabel = 'Proportion')



ax1.bar(mfr_counts.index.values, rel_freq_mfr, width=0.5, align='center')

ax1.set(title = 'mfr_counts', xlabel='mfr' , ylabel = 'Proportion')



ax2.bar(shelf_counts.index.values, rel_freq_shelf, width=0.5, align='center')

ax2.set(title = 'shelf_counts', xlabel='shelf' , ylabel = 'Proportion')



# Title the figure

fig.suptitle('Relative Frequency - Proportion Distribution', fontsize=14, fontweight='bold');
num_rows = cereal_df['name'].count()



percent_freq_type = (type_counts.values/num_rows)*100

print(percent_freq_type)



percent_freq_mfr = (mfr_counts.values/num_rows)*100

print(percent_freq_mfr)



percent_freq_shelf = (shelf_counts.values/num_rows)*100

print(percent_freq_shelf)



plt.style.use('seaborn-whitegrid')



# Get the figure and the axes (or subplots)



fig, (ax0, ax1, ax2) = plt.subplots(nrows=1, ncols=3, figsize=(15, 4))





ax0.bar(type_counts.index.values, (type_counts.values/num_rows) *100, width=0.5, align='center')

ax0.set(title = 'type_counts', xlabel='type' , ylabel = 'Percentage')



ax1.bar(mfr_counts.index.values, (mfr_counts.values/num_rows) * 100, width=0.5, align='center')

ax1.set(title = 'mfr_counts', xlabel='mfr' , ylabel = 'Percentage')



ax2.bar(shelf_counts.index.values, (shelf_counts.values/num_rows) *100, width=0.5, align='center')

ax2.set(title = 'shelf_counts', xlabel='shelf' , ylabel = 'Percentage')



# Title the figure

fig.suptitle('Relative Frequency - Percentage Distribution', fontsize=14, fontweight='bold');
#So we do groupby on mfr and shelf



group_mfr_shelf = cereal_df.groupby(['mfr','shelf'])['name']  



group_mfr_shelf.count()
cereal_df.groupby('mfr')

cereal_df.groupby('mfr')['name']
cereal_df.groupby('mfr')['name'].count()
for (mfr, group) in cereal_df.groupby('mfr'):

    print("mfr={0:5} shape={1}".format(mfr, group.shape ))
print("group of mfr='P' contains these cereals")

cereal_df['name'][cereal_df['mfr'] == 'P']



print("group of mfr='Q' contains these cereals")

cereal_df['name'][cereal_df['mfr'] == 'Q']
# Useful for Categorical data

# Others will be useful for numeric data

print('count')

cereal_df.groupby('mfr').count()

print('first')

cereal_df.groupby('mfr').first()

print('last')

cereal_df.groupby('mfr').last()



print('first and last - as per alphabetical sorting of names for a group')

cereal_df.groupby('mfr').describe()
cereal_df.groupby('mfr').describe().unstack()
cereal_df.groupby('mfr').sum()

cereal_df.groupby('mfr').mean()

cereal_df.groupby('mfr').median()
cereal_df.groupby('type')['name']

cereal_df.groupby('type')['name'].count()

cereal_df.groupby('mfr')['name'].count()

cereal_df.groupby('shelf')['name'].count()
cat_df = pd.DataFrame({'name': cereal_df['name'], 

                        'mfr': cereal_df['mfr'], 

                       'shelf': cereal_df['shelf']}, 

                       columns = ['name','mfr', 'shelf'])





print('cat_df.shape--------->')

cat_df.shape

print('cat_df.info()--------->')

cat_df.info()

print('cat_df.describe()--------->')

cat_df.describe()

print('cat_df.head()--------->')

cat_df.head()

num_df = pd.DataFrame({'name':cereal_df['name'],

                    'calories':cereal_df['calories'],# grams of calories per serving 

                    'protein': cereal_df['protein'],# grams

                    'fat':cereal_df['fat'],# grams

                    'sugars': cereal_df['sugars'],# grams

                    'sodium':cereal_df['sodium']/1000,# miligrams to grams

                    'potass': cereal_df['potass']/1000, # miligrams to grams

                       'rating': cereal_df['rating']

                      },

                      columns = ['name','calories', 'protein','fat','sugars','sodium','potass'])



num_df.head()
num_df.mean()

num_df.mean()[0]
num_df.mean(axis='columns').head()
col_df = pd.DataFrame({'name':'COL_TOTAL',

                    'calories':num_df.mean()[0],# grams of calories per serving 

                    'protein': num_df.mean()[1],# grams

                    'fat':num_df.mean()[2],# grams

                    'sugars': num_df.mean()[3],# grams

                    'sodium':num_df.mean()[4],# miligrams to grams

                    'potass': num_df.mean()[5] # miligrams to grams

                      },

                     index=['1'])

col_df



row_df = pd.DataFrame({'name':num_df.name,

                        'ROW_TOTAL':num_df.mean(axis='columns')},

                      index=num_df.index)

row_df.tail()



num_df.shape[0]

row_df1 = pd.DataFrame({'name':'COL_TOTAL',

                        'ROW_TOTAL':np.sum(row_df['ROW_TOTAL'])},

                      index=[num_df.shape[0]])



row_df1
num_df1 = num_df.append(col_df,ignore_index=True)#ignore_index creted new index



num_df1.tail()

#num_df.append??



row_df2 = row_df.append(row_df1)

row_df2.tail()
cat_df.groupby('mfr').aggregate(['min', np.median, 'max']) # median makes sense for shelf
cat_df.groupby('mfr').aggregate({'shelf': 'min','name': 'max'})

cat_df.groupby('mfr').aggregate({'shelf': 'min','name': 'min'})

cat_df.groupby('mfr').aggregate({'shelf': 'max','name': 'max'})

cat_df.groupby('mfr').aggregate({'shelf': 'max','name': 'min'})
# Kellogs and General Mills are higher count

cereal_df['name'][cereal_df['mfr'] == 'A'].count()

cereal_df['name'][cereal_df['mfr'] == 'N'].count()
print(' groupby(\'mfr\',\'shelf\') - Which manufacturers cereal on which shelf? ')



mfr_shelf_df1 = cat_df.groupby(['mfr','shelf'])['name'].aggregate('count')

mfr_shelf_df1

mfr_shelf_df1.index



mfr_shelf_df2 = cat_df.groupby(['mfr','shelf'])['name'].aggregate('count').unstack()

mfr_shelf_df2
mfr_shelf_df3 = mfr_shelf_df2.fillna(0)

mfr_shelf_df3

# Get the figure and the axes (or subplots)

fig, ax = plt.subplots(figsize=(10, 5))



# Using dataframe's plot

mfr_shelf_df3.plot(kind='bar', ax=ax, fontsize=10, colormap=plt.cm.coolwarm_r, grid=False)



ax.set(xlabel='Manufacturer' , ylabel = 'Frequency',title='Side-by-side barplot')



fig.suptitle('Number of cereals on each shelf (by manufacturer type)', fontsize=14, fontweight='bold');
# Get the figure and the axes (or subplots)

fig, ax = plt.subplots(figsize=(10, 5))



# Using dataframe's plot

mfr_shelf_df3.plot(kind='bar', ax=ax, stacked='True',colormap=plt.cm.coolwarm_r, grid=False)#plt.cm.<TAB> to choose colors



ax.set(xlabel='Manufacturer' , ylabel = 'Frequency',title='Stacked barplot')



fig.suptitle('Number of cereals on each shelf (by manufacturer type)', fontsize=14, fontweight='bold');
print('cat_df.head() ---->')

cat_df.head()

cat_df.index

cat_df.columns
arr = mfr_shelf_df3.values

arr

arr.shape[1]
# Create a new row at bottom to hold COLUMN TOTALS

new_row = np.zeros(shape=(1,arr.shape[1]))

new_row
# Vertically stack new row of zeroes at bottom

arr1 = np.vstack([arr, new_row])

arr1
# Create a new column at right to hold ROW TOTALS

new_col = np.zeros(shape=(arr1.shape[0],1))

new_col
arr1.shape

new_col.shape
# Horizontally stack new column of zeroes at right

arr2 = np.hstack([arr1, new_col])

arr2

arr2.shape
arr2.sum(axis=0) # column totals
arr3 = arr2.copy()

arr3

arr3.shape[0]
arr3[arr3.shape[0]-1] # last row
# Fill last row with sum of all values in each column

arr3[arr3.shape[0]-1]=arr2.sum(axis=0) # column total

arr3
arr3[:, 3] # all rows and last columns
# Fill last column with sum of all values in each row

arr3[:, arr3.shape[1]-1] = arr3.sum(axis=1) # row total

arr3
arr4 = arr3.copy()

arr4

total_rows = arr4.shape[0]

total_rows



total_cols = arr4.shape[1]

total_cols
last_row = arr4[total_rows-1] # last row contains column totals

last_row



shelf_freq_df=pd.DataFrame(last_row, index=['1','2','3','row_total'],columns=['col_total'])

shelf_freq_df

last_row.size

total = last_row[last_row.size-1]



last_row_prop = [  round( item/total, ndigits=4)   for item in last_row]

last_row_prop



shelf_prop_df=pd.DataFrame(last_row_prop, index=['1','2','3','row_total'],columns=['col_total'])

shelf_prop_df
last_row_percent = [  round( item/total*100, ndigits=2)   for item in last_row]

last_row_percent



shelf_percent_df=pd.DataFrame(last_row_percent, index=['1','2','3','row_total'],columns=['col_total'])

shelf_percent_df
num_rows = cereal_df['name'].count()



percent_freq_type = (type_counts.values/num_rows)*100

print(percent_freq_type)



percent_freq_mfr = (mfr_counts.values/num_rows)*100

print(percent_freq_mfr)



percent_freq_shelf = (shelf_counts.values/num_rows)*100

print(percent_freq_shelf)



plt.style.use('seaborn-whitegrid')



# Get the figure and the axes (or subplots)



fig, (ax0, ax1, ax2) = plt.subplots(nrows=1, ncols=3, figsize=(15, 4))





ax0.bar(type_counts.index.values, (type_counts.values/num_rows) *100, width=0.5, align='center')

ax0.set(title = 'type_counts', xlabel='type' , ylabel = 'Percentage')



ax1.bar(mfr_counts.index.values, (mfr_counts.values/num_rows) * 100, width=0.5, align='center')

ax1.set(title = 'mfr_counts', xlabel='mfr' , ylabel = 'Percentage')



ax2.bar(shelf_counts.index.values, (shelf_counts.values/num_rows) *100, width=0.5, align='center')

ax2.set(title = 'shelf_counts', xlabel='shelf' , ylabel = 'Percentage')



# Title the figure

fig.suptitle('Relative Frequency - Percentage Distribution (One variable)', fontsize=14, fontweight='bold');
last_col = arr4[:,total_cols-1] # last col contains row totals

last_col



mfr_freq_df=pd.DataFrame(last_col, index=['A','G','K','N','P','Q','R','col_total'],columns=['row_total'])

mfr_freq_df
last_col.size

total = last_col[last_col.size-1]



last_col_prop = [  round( item/total, ndigits=4)   for item in last_col]

last_col_prop



mfr_prop_df=pd.DataFrame(last_col_prop, index=['A','G','K','N','P','Q','R','col_total'],columns=['row_total'])

mfr_prop_df
last_col_percent = [  round( item/total*100, ndigits=4)   for item in last_col]

last_col_percent



mfr_percent_df=pd.DataFrame(last_col_percent, index=['A','G','K','N','P','Q','R','col_total'],columns=['row_total'])

mfr_percent_df
mfr_percent_df.plot(kind='bar',colormap=plt.cm.coolwarm_r)
arr4
#column proportion array

arr5 = arr4[:,:]/ arr4[arr4.shape[0]-1]

arr5



arr5_minus_last_rc = arr5[ 0 : arr5.shape[0]-1 , 0 : arr5.shape[1]-1]

arr5_minus_last_rc



col_prop_table = pd.DataFrame(arr5, columns=['1','2','3','row_total'],index=['A', 'G', 'K', 'N', 'P', 'Q', 'R','col_total'])

col_prop_table



col_prop_df = pd.DataFrame(arr5_minus_last_rc, columns=['1','2','3'],index=['A', 'G', 'K', 'N', 'P', 'Q', 'R'])

col_prop_df

#column percent array

arr6 = arr5.copy()

arr6 = (arr6[:,:]/ arr6[arr6.shape[0]-1])*100

arr6



arr6_minus_last_rc = arr6[ 0 : arr6.shape[0]-1 , 0 : arr6.shape[1]-1]

arr6_minus_last_rc



col_per_df = pd.DataFrame(arr6_minus_last_rc, columns=['1','2','3'],index=['A', 'G', 'K', 'N', 'P', 'Q', 'R'])

col_per_df
# Get the figure and the axes (or subplots)

fig, ax = plt.subplots(figsize=(10, 5))



# Using dataframe's plot

col_per_df.plot(kind='bar', ax=ax, stacked='True',colormap=plt.cm.coolwarm_r, grid=False)#plt.cm.<TAB> to choose colors



ax.set(xlabel='Manufacturer' , ylabel = 'Percent',title='Stacked barplot')



fig.suptitle('Percentage of cereals on each shelf (by manufacturer type)', fontsize=14, fontweight='bold');
# Get the figure and the axes (or subplots)

fig, ax = plt.subplots(figsize=(10, 5))



# Using dataframe's plot

col_per_df.plot(kind='bar', ax=ax, colormap=plt.cm.coolwarm_r, grid=False)#plt.cm.<TAB> to choose colors



ax.set(xlabel='Manufacturer' , ylabel = 'Percent',title='Stacked barplot')



fig.suptitle('Percentage of cereals on each shelf (by manufacturer type)', fontsize=14, fontweight='bold');
### Other way (Divide each row value by row total)
arr4
arr4

arr4_minus_rc = arr4[ :arr4.shape[0]-1 , :arr4.shape[1]-1]

arr4_minus_rc
# Three cols are for shelf 1,2,3

# 7 Rows are for mfr types (A,G,K,N,P,Q,R)



index = [(1,'A'), (1,'G'),(1,'K'),(1,'N'),(1,'P'),(1,'Q'),(1,'R'),

         (2,'A'), (2,'G'),(2,'K'),(2,'N'),(2,'P'),(2,'Q'),(2,'R'),

         (3,'A'), (3,'G'),(3,'K'),(3,'N'),(3,'P'),(3,'Q'),(3,'R'),

        ]

#index



cereal_count = [0,6,4,3,2,1,4,

                1,7,7,2,1,3,0,

                0,9,12,1,6,4,4]

cereal_count



cereal_count_ser = pd.Series(cereal_count, index=index)

print('cereal_count_ser->\n',cereal_count_ser)



id = [i     for i in cereal_count_ser.index     if i[1] == 'G']

id

cereal_count_ser[id]



id = [i     for i in cereal_count_ser.index     if i[0] == 1]

id

cereal_count_ser[id]



cereal_count_ser[(2,'A'):(3, 'A')]
m_index = pd.MultiIndex.from_tuples(index)

m_index



cereal_count_ser = cereal_count_ser.reindex(m_index)

cereal_count_ser



cereal_count_df = cereal_count_ser.unstack()

cereal_count_df
cereal_count_df.plot(kind='bar',colormap = plt.cm.Set3_r,edgecolor='brown')



cereal_count_df.plot(kind='bar',stacked=True, colormap = plt.cm.Set3_r,edgecolor='brown')

arr7 = col_prop_df.values

arr7

proportion_df = pd.DataFrame({'proportion': [ round(i,ndigits=4)  for i in arr7.flatten().flatten() ]})

proportion_df

proportion_df.unstack()