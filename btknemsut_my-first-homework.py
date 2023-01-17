# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns  # visualization tool
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
data=pd.read_csv('../input/loan.csv')
data.info()
data.corr()
f,ax = plt.subplots(figsize=(25, 25))
sns.heatmap(data.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)
plt.show()
data.head(6)
data.columns
data.installment.plot(kind = 'line', color ='blue', label = 'installment' ,linewidth = 1 , alpha = 0.6 , grid = True , linestyle = ':' )
data.total_acc.plot(color = 'red', label= 'total_acc', linewidth = 1 , alpha = 0.6 , grid =True , linestyle ='-.')
plt.legend(loc= 'lower left')
plt.xlabel('installment')
plt.ylabel('total_acc')
plt.title('Line Plot')
plt.show()
data.plot( kind= 'scatter' , x='loan_amnt', y='funded_amnt_inv', alpha = 0.6 , color='green')
plt.xlabel('loan_amnt')
plt.ylabel('funded_amnt_inv')
plt.title('Loan_Amnt Funded_amnt_inv Scatter Plot')
plt.show()
data.loan_amnt.plot( kind= 'hist', bins=45 , figsize = (15,15) )
plt.xlabel('loan_amnt')

plt.show()
data.loan_amnt.plot(kind = 'hist',bins = 45)
plt.clf()
dictionary = {'turkey' : 'istanbul','italy' : 'roma'}
print(dictionary.keys())
print(dictionary.values())
dictionary['turkey'] = "ankara"
print(dictionary)
dictionary['germany'] = "hamburg"
print(dictionary)
del dictionary['germany']
print(dictionary)
print('italy' in dictionary)
dictionary.clear()                 
print(dictionary)


print(dictionary)       
data=pd.read_csv('../input/loan.csv')
series = data['loan_amnt']      
print(type(series))
data_frame = data[['loan_amnt']]  
print(type(data_frame))
print(8.3 > 8.2)
print(8.3!=8.2)

print(True and False)
print(True or False)
x = data['loan_amnt']>34500   
data[x]
data[np.logical_and(data['loan_amnt']>34500, data['last_pymnt_amnt']>36200 )]
data[(data['loan_amnt']>34500) & (data['last_pymnt_amnt']>36200)]
i = 0
while i != 20 :
    print('i is: ',i)
    i +=2 
print(i,' is equal to 10')
lis = [1,2,3,4,5,6,7,8,9]
for i in lis:
    print('i is: ',i)
print('')


for index, value in enumerate(lis):
    print(index," : ",value)
print('')   

dictionary = {'turkey':'ankara','italy':'roma'}
for key,value in dictionary.items():
    print(key," : ",value)
print('')


for index,value in data[['loan_amnt']][0:3].iterrows():
    print(index," : ",value)
