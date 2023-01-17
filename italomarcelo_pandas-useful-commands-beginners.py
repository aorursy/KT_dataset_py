import pandas as pd

import numpy as np
# Series

mySerie1 = pd.Series([1,2,3,4,5,])

mySerie2 = pd.Series(list('abcde'))
mySerie1
mySerie1.index, mySerie1.values
mySerie2
mySerie2.index, mySerie2.values
# DataFrames

names = ['Joao', 'Pedro', 'Carla']

ages = [20, 22, 15]

sex = ['Male', None, 'Female']

myDf = pd.DataFrame(data={'Names': names, 'Ages': ages, 'Sex': sex})



# or 



myDf2 = pd.DataFrame(columns=['Names', 'Ages'])

myDf2['Names'] = names

myDf2['Ages'] = ages
myDf
myDf2
path = 'https://raw.githubusercontent.com/italomarcelogit/python.free/master/fbprophet/pdv.csv'

df = pd.read_csv(path)

df.head(3)
desconto = pd.read_csv(path, usecols=['desconto'])

desconto.head()
pd.read_csv(path, nrows=3, usecols=['loja', 'cidade'])
# Excel



xls = 'https://github.com/italomarcelogit/python.free/blob/master/dataset-sales-random/cities-BR.xlsx?raw=true'



df = pd.read_excel(xls)
df.head(3)
pd.read_excel(xls, nrows=3)
df.info()
df.describe()
df.describe(include=['object'])
df.shape
df.index
df.columns
df.select_dtypes(include=['float64', 'int64']).head(2)
df.select_dtypes(include=[np.number]).head(2)
df.select_dtypes(include=['object']).head(2)
df1 = pd.DataFrame({'a':np.zeros(5), 'b':np.arange(1,6), 'c': np.array(5)})

df1
df2 = df1

df2
df2['a'] = np.array(3)

df2['a'].values, df1['a'].values
df1 = pd.DataFrame({'a':np.zeros(5), 'b':np.arange(1,6), 'c': np.array(5)})

df2 = df1.copy()

df2['a'] = np.array(3)

df2['a'].values, df1['a'].values
df2
# add column d

df2['d'] = np.random.randint(100,150.5)

df2
# add row with index 5

df2 = df2.append({'a': 301, 'b': 304, 'c': 303, 'd': 304}, ignore_index=True)

df2
# MODIFY COLUMN

df2['d'] = np.array(333)

df2
# MODIFY ROW

df2.iloc[5] = [444, 444,444,444]

df2
# DELETE COLUMN d

df2 = df2.drop('d', axis=1)

df2
# DELETE ROW 5, by index



df2 = df2.drop([5])

df2
df2
df2.sum()
df2['a'].sum()
df2['b'].min()
df2['b'].max()
df2.mean()
df2['b'].median()
df3 = pd.DataFrame([{'name': 'Joao', 'status':1}, {'name':'pedro', 'status': 0}])

df3
status = {1:'passed', 0:'dont passed'}

df3['textStatus'] = df3['status'].map(status)

df3
df3 = pd.DataFrame([

                    {'name': 'Joao', 'h':1.72, 'w': 85}, 

                    {'name':'pedro', 'h':1.70, 'w': 65}

                    ])

df3
def imcBMI(h, w):

  return w/h**2



df3['imc'] = df3.apply(lambda x: imcBMI(x['h'], x['w']), axis=1)

df3
x = np.random.randint(1, 4, 50)

x
pd.value_counts(x)
df4 = pd.DataFrame({'x': x})

df4['x'].value_counts()
df4['x'].describe()
names = ['A', 'B', 'C']

ages = [20, np.nan, 15]

sex = [np.nan, np.nan, 'Female']

df5 = pd.DataFrame({'name': names, 'age': ages, 'sex':sex})

df5
df5.isnull()
df5.isnull().sum()
df5.info()
df5
# AGE ==> assuming the rule is: for null values, enter the MAX value



df5.age = df5.age.fillna(df5.age.max())

df5
# SEX ==> assuming the rule is: for null values, enter the UNKNOW value



df5.sex = df5.sex.fillna('Unknow')

df5
# filtering data with sex == Unknow

df5[df5.sex == 'Unknow']
# or

df5.query('sex == "Unknow"')
df5.iloc[1]
# or 

df5.iloc[1].values.tolist()
# or return only age value from row 1

df5.age.iloc[1]
# update sex columns, 'Unknow' to 'Not Informed' values

df5
df5.sex = df5.sex.apply(lambda x: 'Not Informed' if str(x) == "Unknow" else x)

df5
df
def eda(dfA, all=False, desc='Exploratory Data Analysis'):

    print(desc)

    print(f'\nShape:\n{dfA.shape}')

    print(f'\nDTypes - Numerics')

    print(dfA.select_dtypes(include=np.number).columns.tolist())

    print(f'\nDTypes - Categoricals')

    print(dfA.select_dtypes(include='object').columns.tolist())

    print(f'\nIs Null: {dfA.isnull().sum().sum()}')

    print(f'{dfA.isnull().mean().sort_values(ascending=False)}')

    dup = dfA.duplicated()

    print(f'\nDuplicated: \n{dfA[dup].shape}\n')

    try:

        print(dfA[dfA.duplicated(keep=False)].sample(4))

    except:

        pass

    if all:  # here you put yours prefered analysis that detail more your dataset

        

        print(f'\nDTypes - Numerics')

        print(dfA.describe(include=[np.number]))

        print(f'\nDTypes - Categoricals')

        print(dfA.describe(include=['object']))



# function Fill NaN values

def cleanNaN(dfA):

  for col in dfA:

    if type(dfA[col]) == 'object':

        dfA[col] = dfA[col].fillna('unknow')

    else:

        dfA[col] = dfA[col].fillna(0)

  return dfA
eda(df)
eda(df, all=True)