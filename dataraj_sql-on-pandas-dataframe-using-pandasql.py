## Importing pandasql 

import pandasql as psql
##  Birth Data Set 

birth = psql.load_births()

birth.head()
##  Meat Data Set 

meat = psql.load_meat()

meat.head()
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
toydf = pd.read_csv('/kaggle/input/toy-dataset/toy_dataset.csv',header=0)
toydf.head()
toydf.City.unique()
toydf.Age.describe()
toydf.Income.describe()
toydf.Illness.unique()
sdf =  psql.sqldf("select Age, Income from toydf")

sdf.head()
sdfc =  psql.sqldf("select Age, Income, City from toydf")

sdfc.head()
sdfc1 =  psql.sqldf("select Age, Income, City from toydf limit 5")

sdfc1.head()
dallasDf = psql.sqldf("select * from toydf where City ='Dallas'")
dallasDf.head()
dallasDf1 = psql.sqldf("select * from toydf where City ='Dallas' limit 5")
dallasDf1
dallasDf1 = psql.sqldf("select * from toydf where City ='Dallas' and Age > 30 limit 5")
dallasDf1
cdf = psql.sqldf("select count(*), City from toydf group by City")
cdf.head()
## This query will return average age City wise

cdf1 = psql.sqldf("select avg(Age), City from toydf group by City")

cdf1.head()
## This query will return average age and Income grouped on City and Gender

cdf2 = psql.sqldf("select avg(Age),avg(Income), City, Gender from toydf group by City, Gender")

cdf2.head()
importDf = pd.read_csv("/kaggle/input/india-trade-data/2018-2010_import.csv",header=0)

exportDf = pd.read_csv("/kaggle/input/india-trade-data/2018-2010_export.csv",header=0)
importDf.head()
exportDf.head()
importDf.HSCode.describe()
exportDf1 = psql.sqldf("select * from exportDf where HSCode <30")

importDf1 = psql.sqldf("select * from importDf where HSCode <30")
joinedData = psql.sqldf("select i.HSCode, i.value as importVal,e.value as exportVal, i.year from  importDf1 i inner join exportDf1 e on i.HSCode = e.HSCode; ")
joinedData.head(500)