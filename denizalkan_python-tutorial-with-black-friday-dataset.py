
import numpy as np # import Numpy
import pandas as pd # import Pandas
import matplotlib.pyplot as plt # import Matplotlib
import seaborn as sns # import Seaborn

# Input data files are available in the "../input/" directory.

# Add Data
data = pd.read_csv('../input/blackfriday/BlackFriday.csv')

# Info About Data (Row Count, Column Count & Datatype etc.)
data.info()

# Correlation Between Columns
data.corr()

# To see the correlation visually we will use Seaborn & Matplotlib Librarys
f,ax = plt.subplots(figsize=(15, 15))
sns.heatmap(data.corr(), annot=True, linewidths=.1, fmt= '.1f',ax=ax)
plt.show()
# Using Line Plot
data.Product_Category_1.plot(kind = 'line', color = 'g',label = 'Product_Category_1',linewidth=1,alpha = 0.5,grid = True,linestyle = ':')
data.Product_Category_2.plot(color = 'r',label = 'Product_Category_2',linewidth=1, alpha = 0.5,grid = True,linestyle = '-.')
plt.legend(loc='upper right')      
plt.xlabel('x axis')               
plt.ylabel('y axis')
plt.title('Line Plot')            
plt.show()


# Using Scatter Plot 
data.plot(kind='scatter', x='Product_Category_1', y='Product_Category_2',alpha = 0.5,color = 'red')
plt.xlabel('Product_Category_1')             
plt.ylabel('Product_Category_2')
plt.title('Products Scatter Plot')
plt.show()


# Using Histogram
data.Purchase.plot(kind='hist', color = 'green')
plt.show()
# Creating Dictionary
dic = {'TURKEY' : 'Galatasaray','GERMANY' : 'Bayern Munich', 'ENGLAND' : 'M.United'}
# Dictionary Keys
print(dic.keys())
# Dictionary Values
print(dic.values())

# UPDATE Value
dic['TURKEY'] = 'Fenerbahçe'
print(dic.values())

# INSERT New Value
dic['SPAIN'] = 'R. Madrid'
print(dic.values())

# DELETE Value with Key
del dic['TURKEY'] 
print(dic.values())

# IS Include? (Return = True or False)
print('FRANCE' in dic)  # False
print('TURKEY' in dic)  # True

# Remove All Values in Dictionary
dic.clear()
print(dic.values())

# DELETE Dictionary
del dic

# Data Frame & Series
DataAgeSeries = data['Age']
print(type(DataAgeSeries)) # Output : Series
DataAgeDF = data[['Age']]  
print(type(DataAgeDF))     # Output : DataFrame

# Logic Operators
print(1 > 0) # True
print(1!=0)  # True
print(True and False) # False
print(True or False)  #True


# Filtering Data  
x = data['Purchase'] > 20000
data[x]

# AND 
data[np.logical_and(data['Gender']=='F', data['Marital_Status']==1 )] # 55223 rows
# OR
data[np.logical_or(data['Marital_Status']==0, data['Age']=='25-36' )] # 317817 rows 
data[(data['Marital_Status']==0) & (data['Age']=='26-35')] # 317817 rows


# WHILE
i = 0
while i != 5 : 
    print('i is : ',i)
    i +=1 
print(i,' is equal to 5')
print('------------------')

# FOR LOOP
lis = [1,2,3,4,5]
for i in lis:
    print('i is: ',i)
print('------------------')


# ENUMERATE : index and value of list
for index, value in enumerate(lis):
    print(index,". index is : ",value)
print('------------------')

# Using FOR LOOP in DICTIONARY
dictionary = {'TURKEY':'Galatasaray','ENGLAND':'M.United'}
for key,value in dictionary.items():
    print(key," : ",value)
print('------------------')

# Iterrows Function
for index,value in data[['Age']][0:10].iterrows():
    print(index," : ",value)
