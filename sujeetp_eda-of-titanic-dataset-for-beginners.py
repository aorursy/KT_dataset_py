import numpy as np
import pandas as pd
import os
## load the data in the dataframe using pandas

import os
print(os.listdir("../input"))
train_df=pd.read_csv("../input/train.csv",index_col='PassengerId')
test_df=pd.read_csv("../input/test.csv",index_col='PassengerId')
type(train_df)
## use .infor() to get information about the dataframe
train_df.info()
test_df.info()
##Information of the dataset
#1) Survived: Number of passengers survived(1=yes,0=No)
#2) PClass is the passenger class (1=1st class,2=2nd class,...)
#3) Name
#4) Sex
#5) Age
#6) Sibsp stnads for siblings and spouse
#7) Parch stands for parents/children
#8) Ticket
#9) Fare
#10) Cabin
#11) Embarked is the boarding point for the passengers
test_df['Survived']=-888 #Add default value to survived column
df=pd.concat((train_df,test_df),axis=0)
df.info()

#use head() to get top 5 records
df.head()
df.tail()

#column selection with dot
df.Name
#select data using label based using loc[rows_range,col_range]
df.loc[5:10,'Age':'Survived']
#position based indexing using iloc()
df.iloc[5:10, 0:5]
#filter row based on the condition
male_passengers=df.loc[df.Sex=='male',:]
print("Number of male passengers:{0}".format(len(male_passengers)))
male_passengers_first_class=df.loc[((df.Sex=='male')& (df.Pclass==1)),:]
print("Number of male passengers with first class:{0}".format(len(male_passengers_first_class)))
df.describe()
#Box plot
%matplotlib inline
df.Fare.plot(kind='box')
df.describe(include='all')
df.Sex.value_counts(normalize=True)
df.Sex.value_counts().plot(kind='bar')
df.Pclass.value_counts().plot(kind='bar')
df.Pclass.value_counts().plot(kind='bar',rot=0,title="class of passengers count",color='red');

