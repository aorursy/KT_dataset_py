#import libraries

import pandas as pd

import pandasql 

import matplotlib.pyplot as plt



from plotly.offline import iplot, init_notebook_mode

import cufflinks as cf

cf.go_offline()

print( cf.__version__)
read_csv = pd.read_csv("../input/province-wise.csv")
read_csv.head()
read_csv = read_csv.rename(columns={'Literacy Rate':'Literacy_Rate'})

read_csv = read_csv.rename(columns={'Population Aged 5 years & above':'Population_Aged_5_years_above'})

read_csv = read_csv.rename(columns={'Population who are Can read & write':'Population_who_are_Can_read_write'})
read_csv.head()
sql1 = "select Province,Population_Aged_5_years_above,Population_who_are_Can_read_write,Literacy_Rate as Total from read_csv where(Sex=='Total' and Province != 'Nepal')"

sql2 = "select Literacy_Rate as Male from read_csv where(Sex=='Male' and Province != 'Nepal')"

sql3 = "select Literacy_Rate as Female from read_csv where(Sex=='Female' and Province != 'Nepal')"
df1 = pandasql.sqldf(sql1)

df2 = pandasql.sqldf(sql2)

df3 = pandasql.sqldf(sql3)
df1 = df1.join(df2)

df1 = df1.join(df3)
df1
df1.iplot(x='Province',y=['Female','Male'], kind='bar', title='Literacy Rate in Nepal(2011)')
df1.iplot(x='Province',y=['Total'], title='Literacy Rate in Nepal(2011)', kind='bar')
df1.iplot(x='Province',y=['Population_who_are_Can_read_write'],kind='bar',title='Literacy Rate in Nepal(2011)')