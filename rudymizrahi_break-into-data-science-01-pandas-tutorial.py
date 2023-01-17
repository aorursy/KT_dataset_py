import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))
import numpy as np 
import pandas as pd
data = pd.read_csv('../input/acs2015_census_tract_data.csv')
data.info()
data.head(10)
data.tail(10)
data.shape
data.describe()
data['State'].nunique()
data['State'].unique()
pd.isnull(data).tail(10)
data = data.dropna()
data.shape
data.head()
data = data.drop(['CensusTract', 'County'],axis=1) #data = data.drop('CensusTract',axis=1)  if dropping only one column
data.head()
data['M/W_Ratio'] = data['Men'] / data['Women']
data.head()
data = data.drop('M/W_Ratio',axis=1)
data.head()
data.columns
percentages = ['Hispanic', 'White', 'Black', 'Native', 'Asian', 'Pacific','Poverty', 'ChildPoverty', 'Professional', 'Service', 'Office','Construction','Production','Drive','Carpool','Transit','Walk','OtherTransp','WorkAtHome', 'PrivateWork', 'PublicWork', 'SelfEmployed', 'FamilyWork', 'Unemployment']
for i in percentages:
    data[i] = round(data['TotalPop'] * data[i] / 100)   
#We won't be doing further data cleaning on our data in this tutorial.
data.head(20)
# you can see that we can make multiple calculations and update our table with the power of loops
data = data.groupby('State', as_index=False).sum()
data.head(10)
pop = data['TotalPop'].sum()
print('Total Population is: ', pop)
data['TotalPop'].max()
data = data.sort_values('TotalPop')
data.head()
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns

fig, ax = plt.subplots(figsize=(14,4))
fig = sns.barplot(x=data['State'], y=data['TotalPop'], data=data)
fig.axis(ymin=0, ymax=40000000)
plt.xticks(rotation=90)
sns.distplot(data['TotalPop'])
data_hisp = data[data['Hispanic'] >= 1000000] 
data_hisp
data.columns
data[(data['Women']/data['Men'] > 1) & (data['Native'] > data['Asian']) | (data['TotalPop']>35000000)]
data['State'].head() 
data[['State', 'TotalPop']].head() 
data['State'].iloc[0] 
data['State'].iloc[51]
data['State'].loc[0] 
data.iloc[0,:] 
data.iloc[:,1].head() # Second column
data.iloc[0,0]# First element of first column, and so on...
df = pd.DataFrame(np.random.rand(15,5)) 
df
my_list = [1,2,3,4,5,6,7,8,9,10]
pd.Series(my_list) 
df.index = pd.date_range('2018/1/1',periods=df.shape[0]) 
df 
df2 = pd.DataFrame(np.random.rand(2,5)) 
df2
dfnew = df.append(df2)
dfnew
df3 = pd.DataFrame(np.random.rand(16,3))
df3.index = pd.date_range('2018/1/1',periods=df3.shape[0]) #This will add a date index
df3

dfnew2=pd.concat([dfnew, df3],axis=1)
dfnew2