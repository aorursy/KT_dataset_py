import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import plotly
plotly.offline.init_notebook_mode(connected=True)
import plotly.graph_objs as go

df=pd.read_csv('../input/BlackFriday.csv')
df.head()
df.info()
df.describe()
(df.isnull().sum()/len(df))
for x in df.columns:
    print('Unique value for',x, df[x].unique())
df.fillna(0,inplace=True)
for x in df.columns:
    print('Unique value for',x,'are :', df[x].nunique())
data = [go.Histogram(
        x = df.Purchase,
        xbins = {'start': 1, 'size': 100, 'end' :25000}
)]

plotly.offline.iplot(data, filename='Purchase distribution')
colonne=['Gender','Age','Occupation','City_Category','Stay_In_Current_City_Years','Marital_Status','Product_Category_1','Product_Category_2','Product_Category_3']
plt.figure(figsize=(40,25))
for x,i in zip(colonne,range(9)):
    plt.subplot(331+i)
    plt.title(x,fontsize=30)
    bar = df[x].value_counts().sort_values(ascending=True)
    plt.bar(bar.index,bar.values,color='royalblue')
    plt.tick_params(labelsize=25)
New_column=['Gender','Age','Occupation','City_Category','Stay_In_Current_City_Years','Marital_Status','Product_Category_1','Product_Category_2','Product_Category_3','Purchase']
print('Average item Cost by:')
plt.figure(figsize=(40,25))
plt.figure(figsize=(40,25))
for x,i in zip(colonne,range(9)):
    plt.subplot(331+i)    
    plt.title(x,fontsize=30)
    pie=df[New_column].groupby(x).mean()
    plt.bar(pie.index,pie['Purchase'],color='royalblue')
    plt.tick_params(labelsize=25)
