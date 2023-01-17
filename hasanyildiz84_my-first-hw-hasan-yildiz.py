# My First HW, Hasan YILDIZ



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns  # visualization tool

import os

print (os.listdir("../input"))
data = pd.read_csv('../input/2017.csv')
data.info()
data.corr()
#correlation map

f,ax = plt.subplots(figsize=(20, 20))

sns.heatmap(data.corr(), annot=True, linewidths=1, fmt= '.1f',ax=ax)

plt.show()
data.columns
data.info()
data.head()
# Line Plot

# color = color, label = label, linewidth = width of line, alpha = opacity, grid = grid, linestyle = sytle of line

data.Family.plot(kind = 'line', color = 'b',label = 'Family',linewidth=2,alpha = 1,grid = True,linestyle = ':')

data.Freedom.plot(color = 'r',label = 'Freedom',linewidth=1, alpha = 0.5,grid = True,linestyle = '-.')

plt.legend(loc='upper right')

plt.xlabel('x axis')

plt.ylabel('y axis')

plt.title('Line Plot')

plt.show()
# Scatter Plot 

# x = attack, y = defense

data.plot(kind='scatter', x='Family', y='Freedom',alpha = 0.5,color = 'r')

plt.xlabel('Family') 

plt.ylabel('Freedom')

plt.title('Family Freedom Scatter Plot')

plt.show()
# Histogram

# bins = number of bar in figure

data.Family.plot(kind = 'hist',bins = 50,figsize = (18,18))

data.Freedom.plot(kind = 'hist',bins = 50,figsize = (9,9))

plt.show()
series = data['Family']

print(type(series))

data_frame = data[['Freedom']]

print(type(data_frame))
# Filtering Pandas data frame



filterX = data['Freedom']>0.6

data[filterX]
# Filtering pandas with logical_and



data[np.logical_and(data['Family']>0.7, data['Freedom']>0.6 )]
# My first homework is done

# Let's continue Second & Third HW's.

data.head(9)



# This means, show that first nine data.
data.tail()



# This means, show that end of 5 data.
data.shape



# Shape of data.
data.info()
print(data['Family'].value_counts(dropna =False))
data.describe()



#get some numeric details; median etc.
data.boxplot(column='Family',by = 'Freedom')
# tidy data studying



data_new = data.head()

data_new.info()
data_new.corr()
data_new.describe()
data_new
melted = pd.melt(frame=data_new,id_vars = 'Country', value_vars= ['Family','Freedom'])

melted
#Let's turn back non-melted version of data :)



melted.pivot(index = 'Country', columns = 'variable',values='value')
data1 = data.head()

data2= data.tail()

conc_data_row = pd.concat([data1,data2],axis =0,ignore_index =True)

conc_data_row
data1 = data['Family'].head()

data2= data['Freedom'].head()

conc_data_col = pd.concat([data1,data2],axis =1)

conc_data_col
data1 = data['Family'].head()

data2= data['Freedom'].head()

conc_data_col = pd.concat([data1,data2],axis =0)

conc_data_col
data.dtypes
data_new.dtypes
data['Freedom'] = data['Freedom'].astype('category')

data['Family'] = data['Family'].astype('float')

data.dtypes
data.info()
data["Generosity"].value_counts(dropna =False)
data1=data

data1["Generosity"].dropna(inplace = True)
assert  data['Generosity'].notnull().all()
data.head()
# Let's get the sixth part of HW



# Plotting all data 

data1 = data.loc[:,["Generosity","Family"]]

data1.plot()

plt.show()
data1.plot(subplots = True)

plt.show()
data1.plot(kind = "scatter",x="Generosity",y = "Family")

plt.show()
data.plot(kind = "hist",y = "Family",bins = 500,range= (0,2),normed = True)
fig, axes = plt.subplots(nrows=2,ncols=1)

data1.plot(kind = "hist",y = "Generosity",bins = 500,range= (0,1),normed = True,ax = axes[0])

data1.plot(kind = "hist",y = "Family",bins = 500,range= (0,1),normed = True,ax = axes[1],cumulative = True)

plt.savefig('graph.png')

plt
data.describe()
data
#MANIPULATING DATA FRAMES WITH PANDAS

# Let's use MANIPULATING DATA FRAMES WITH PANDAS Details :) 

data.Family[9]
data[["Family","Generosity"]]
test1_filter = data.Family > 1

test2_filter = data.Generosity > 0.5



data[test1_filter & test2_filter]
# wouww :)



 #   def div(n):

 #       return n/2

 #   data.Family.apply(div)



# Other Ways ;



data.Family.apply(lambda n : n/2)
# Let's going to last term of this section :)

#INDEX OBJECTS AND LABELED DATA 



print(data.index.name)

data.index.name = "name of index"

data.head()
data1 = data.set_index(["Family","Generosity"]) 

data1.head(9)
dic = {"treatment":["A","A","B","B"],"gender":["F","M","F","M"],"response":[10,45,5,9],"age":[15,4,72,65]}

df = pd.DataFrame(dic)

df
# at the end of this homework 

# have a nice day :) 