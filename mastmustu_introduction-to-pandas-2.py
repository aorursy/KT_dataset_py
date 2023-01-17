# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_csv('/kaggle/input/iris-flower-dataset/IRIS.csv')

df.head()
df[ df['species'] =='Iris-setosa']  # select all columns where flower is setosa
df[ df['species'] =='Iris-setosa'][['sepal_length' ,'sepal_width']]  # select few  columns where flower is setosa
df.loc[0]  # by row index 
df.loc[5]  # locate sixth row
df['petal_width']
df.loc[0,'sepal_width']  # 0 --> first row  and get the value at column --> sepal_width
df.loc[0:5,'sepal_length']   # multiple rows and 1 columns
df.loc[0:5,['sepal_length' ,'petal_width']] 



# multiple rows and multiple columns



#NaN nan
df.iloc[0:5 , 1:3]  # get row # 0 , 1, 2, 3, 4 and columns # 0 ,1, 2
df.groupby('species').sum()
df.groupby('species').count()
df.groupby('species')['sepal_length', 'sepal_width'].mean()
num_list = range(150)



print(num_list)

df['flower_no'] = num_list # adding a new columns in df



df.head(20)
df2 = pd.DataFrame()

df2['flower_no'] = df['flower_no']

df2['flower_weight'] = np.random.random(150)

df2.head(20)
#Lets join the 2 data frame



df3 = pd.merge(df, df2 , on ='flower_no', how='inner')
df3.head(20)
df['sepal_width'].mean()
df['sepal_width'].median()
df['sepal_width'].mode()
df['sepal_width'].std()
df['sepal_width'].var()
df.cov()
df.corr()
dataVal = [(10,20,30,40,50,60,70,80,90,100),



           (10,10,40,40,50,60,70,80,80,80),



           (10,10,10,10,20,30,50,50,60,80)]



dataFrame = pd.DataFrame(data=dataVal);

dataFrame = dataFrame.T

skewValue = dataFrame.skew()



 



print("DataFrame:")



print(dataFrame)



 



print("Skew:")



print(skewValue)
dataFrame.hist()


dataMatrix = [(65,75,74,73,74,75,76,77,78,79,80,95,76,62,90),



              (20 ,30,50,70,101,120,130,140,157,160,191,192,200,210,300)];



       



dataFrame = pd.DataFrame(data=dataMatrix);



dataFrame = dataFrame.T



kurt = dataFrame.kurt();



print("Data:");



print(dataFrame);



print("Kurtosis:");



print(kurt);



 



dataFrame.hist()
df.kurt()
df3.head()
df3.drop('flower_weight' , axis = 1) # axis = 0 --> row wise , # axis = 1 --> column wise

df3.head()
df4 = df3.drop('flower_weight' , axis = 1) # axis = 0 --> row wise , # axis = 1 --> column wise

df4.head()



# This is called soft drop 

# i.e. it will not be dropped from the Original DF , but it drop and return a new DF
# Hard Drop 



df4.drop('flower_no', axis = 1, inplace=True)

df4.head()
df4.tail()
small_df = pd.DataFrame([5.9,3.0,5.1,1.8,'Iris-virginica'])

print(small_df)
small_df= small_df.T # Transpose

print(small_df)
small_df.columns = df4.columns

print(small_df)
df5=df4.append(small_df)

df5.tail()
# Indexing is not right



df5=df4.append(small_df , ignore_index=True)

df5.tail()
df6 = df5.drop_duplicates()

df6.tail()
df_double = df4.append(df4 , ignore_index = True)

df_double.shape
df_double
df_double.drop_duplicates(inplace=True)
df_double.shape
titanic = pd.read_csv('/kaggle/input/titanic/train.csv')

titanic.head()
titanic.info() 
titanic.isna().sum()
# Treatment of Age 



titanic['Age'] = titanic['Age'].fillna(30) # by a value



titanic['Age'] = titanic['Age'].fillna(np.mean(titanic['Age'])) # by mean Age 

titanic['Age'] = titanic['Age'].fillna(np.median(titanic['Age'])) # by median Age 
# Treatment of Embarked



titanic['Embarked'].unique()
titanic['Embarked'].value_counts()
titanic['Embarked'] = titanic['Embarked'].fillna(titanic['Embarked'].mode().values[0])

# mode returns a series 
titanic['Embarked'].value_counts()

# Treatment of Cabin 



titanic['Cabin'].unique()
# So Many Missing Values 



titanic = titanic.drop('Cabin' , axis = 1)

titanic.isna().sum()
# Filtering the data with multiple condition



# Select records who survived and have age greater than 20



titanic.loc[(titanic['Survived'] ==1) & (titanic['Age'] > 20)]


