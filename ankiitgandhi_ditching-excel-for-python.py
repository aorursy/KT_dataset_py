import pandas as pd

import numpy as np
wkbks = glob(os.path.join(os.pardir, 'input', 'xlsx_files_all', 'Ir*.xls'))

sorted(wkbks)
filename = 'Iris.xlsx'
df = pd.read_excel(filename)
print(df)
df1 = pd.read_excel(filename,sheet_name='Sheet2')
print(df1)
df = pd.read_excel(filename,sheet_name='Sheet1', index_col=0)
print(df)
df = pd.read_excel(filename, sheet_name='Sheet1', header=None, skiprows=1, index_col=0)
print(df)
df = pd.read_excel(filename, sheet_name='Sheet1', header=None, skiprows=1, usecols='B,D')
print(df)
df = pd.read_excel(filename)

#Importing the file again to the dataframe in the original shape to use it for further analysis
df.head(10)
df.tail()
df['SepalLength'].head()
df.columns
df.info()
df.shape[0]
print('Total rows in Dataframe is: ',  df.shape[0])

print('Total columns in Dataframe is: ',  df.shape[0])
df.dtypes
df['Name'].head()
df.iloc[:,[4]].head()
df.loc[:,['Name']].head()
df[['Name', 'PetalLength']].head()
#Pass a variable as a list

SpecificColumnList = ['Name', 'PetalLength']

df[SpecificColumnList].head()
df.loc[20:30] 
df.loc[20:30, ['Name']]
df[df['Name'] == 'Iris-versicolor'].head()
df[df['Name'].isin(['Iris-versicolor', 'Iris-virginica'])]
Filter_Value = ['Iris-versicolor', 'Iris-virginica']
df[df['Name'].isin(Filter_Value)]
df[~df['Name'].isin(Filter_Value)]
width = [2]

Flower_Name = ['Iris-setosa']

df[~df['Name'].isin(Flower_Name) & df['PetalWidth'].isin(width)]
df[df['SepalLength'] == 5.1].head()
df[df['SepalLength'] > 5.1].head()
df[df['Name'].map(lambda x: x.endswith('sa'))]
df[df['Name'].map(lambda x: x.endswith('sa')) & (df['SepalLength'] > 5.1)]
df[df['Name'].str.contains('set')]
df['SepalLength'].unique()
df.drop_duplicates(subset=['Name'])
df.drop_duplicates(subset=['Name']).iloc[:,[3,4]]
df.sort_values(by = ['SepalLength'])
df.sort_values(by = ['SepalLength'], ascending = False)
df.describe()
df.describe(include = ['object'])
df.describe(include = 'all')
pd.value_counts(df['Name'])
df.count(axis=0)
df.sum(axis = 0) # 0 for column wise total
df.sum(axis =1) # row wise sum
df['Total'] = df.sum(axis =1)
df.head()
df['Total_loc']=df.loc[:,['SepalLength', 'SepalWidth']].sum(axis=1)
df.head()
df['Total_DFSum']= df['SepalLength'] + df['SepalWidth']
df.head()
df.drop(['Total_DFSum'], axis = 1)
Sum_Total = df[['SepalLength', 'SepalWidth', 'Total']].sum()
Sum_Total
T_Sum = pd.DataFrame(data=Sum_Total).T
T_Sum
T_Sum = T_Sum.reindex(columns=df.columns)
T_Sum
Row_Total = df.append(T_Sum,ignore_index=True)
Row_Total
df[df['Name'] == 'Iris-versicolor'].sum()
df[df['Name'].map(lambda x: x.endswith('sa')) & (df['SepalLength'] > 5.1)].sum()
df[df['Name'] == 'Iris-versicolor'].mean()
df[df['Name'].map(lambda x: x.endswith('sa')) & (df['SepalLength'] > 5.1)].mean()
df[df['Name'] == 'Iris-versicolor'].max()
df[df['Name'] == 'Iris-versicolor'].min()
df[['Name','SepalLength']].groupby('Name').sum()
GroupBy = df.groupby('Name').sum()
Group_By.append(pd.DataFrame(df[['SepalLength','SepalWidth','PetalLength','PetalWidth']].sum()).T)
pd.pivot_table(df, index= 'Name')#Same as Groupby
pd.pivot_table(df, values='SepalWidth', index= 'SepalLength',columns='Name', aggfunc = np.sum)
pd.pivot_table(df, values='SepalWidth', index= 'SepalLength',columns='Name', aggfunc = np.sum, fill_value=0)
pd.pivot_table(df, values=['SepalWidth', 'PetalWidth'], index= 'SepalLength',columns='Name', aggfunc = np.sum, fill_value=0)
pd.pivot_table(df, values=['SepalWidth', 'PetalWidth'], index= ['SepalLength', 'PetalLength'],columns='Name', aggfunc = np.sum, fill_value=0)
pd.pivot_table(df, values=['SepalWidth', 'PetalWidth'], index= 'SepalLength',columns='Name', 

               aggfunc = {'SepalWidth': np.sum, 'PetalWidth': np.mean}, fill_value=0)
pd.pivot_table(df, values=['SepalWidth', 'PetalWidth'], index= 'SepalLength',columns='Name', 

               aggfunc = {'SepalWidth': np.sum, 'PetalWidth': np.mean}, fill_value=0, margins=True)
df1 = pd.read_excel(filename)
lookup = df.merge(df,on='Name')
lookup