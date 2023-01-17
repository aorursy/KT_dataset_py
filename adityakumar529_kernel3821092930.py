import pandas as pd
#read the csv file
data = pd.read_csv('http://www.phdstipends.com/csv')
#get the details of the data available
data.head()

#get the type of data
data.dtypes
#Rename the overall pay(optional)
data = data.rename(columns={'Overall Pay': 'Overall_pay'})

data.head()
#delete the dollar symbol 
data['Overall_pay'] = data['Overall_pay'].str.replace('$', '')
data.head()
#delete the dollar symbol: optinal
data['12 M Gross Pay'] = data['12 M Gross Pay'].str.replace('$', '')
#delete the dollar symbol: optional
data['9 M Gross Pay'] = data['9 M Gross Pay'].str.replace('$', '')
#delete the dollar symbol :optional
data['3 M Gross Pay'] = data['3 M Gross Pay'].str.replace('$', '')
data.dtypes
data.head()

#delete the , symbol else it wont be converted to float
data['Overall_pay'] = data['Overall_pay'].str.replace(',', '')
data.head()
#delete the , symbol else it wont be converted to float
data['Overall_pay'] = data['Overall_pay'].astype(float)
data['Overall_pay'].max()
#delete the , symbol else it wont be converted to float
data['Fees'] = data['Fees'].str.replace('$', '')
data['Fees'] = data['Fees'].str.replace(',', '')
data.head()


data.head(100)

#University with highest overall pay 
data[data['Overall_pay'] == data['Overall_pay'].max()]['University']
#Depatment with highest overall pay 
data[data['Overall_pay'] == data['Overall_pay'].max()]['Department']
# detelete the row with value nan in department and university coloumn
data = data[data['Department'].notnull()]
data.head()
#University with highest pay
high10_University = data.nlargest(10, "Overall_pay") 
print(high10_University.University)
#University with highest pay
high10_Department = data.nlargest(10, "Overall_pay").Department
print(high10_Department)

