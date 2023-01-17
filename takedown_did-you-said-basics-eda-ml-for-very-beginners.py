import pandas as pd # for data preprocessing in the workspace

import numpy as np #calculus and linear algebra

# plotters

import matplotlib.pyplot as plt 

import seaborn as sns



import plotly.plotly as py  # plotly

from plotly.offline import init_notebook_mode, iplot

init_notebook_mode(connected=True)

import plotly.graph_objs as go
Data = pd.read_csv('../input/vgsales.csv')

Data.head(20)# first 20 records 
Data.columns
plt.figure(figsize=(15,8))

sns.set()

plt.grid(True)

sort_plat = Data['Platform'].value_counts().sort_values(ascending=False)

sort_plat.head()

sns.barplot(y=sort_plat.index,x=sort_plat.values,orient='h')

plt.xlabel('Values count')

plt.ylabel('Games Platform')

plt.title('Grouped Platforms count')


df_gl=Data.loc[:99,:] # data.iloc[:100,:] -- data.head(100)



import plotly.graph_objs as go



trace1=go.Scatter(

                x=df_gl.Rank,

                y=df_gl.NA_Sales,

                mode="lines+markers",

                name="North America Sales",

                text=df_gl.Name)

trace2=go.Scatter(

                x=df_gl.Rank,

                y=df_gl.EU_Sales,

                mode="lines",

                name="Europe Sales",

                text=df_gl.Name)



edit_df=[trace1,trace2]

layout=dict(title="World rank of the top 100 video games, EU Sales and NA Sales .",

            xaxis=dict(title="World Rank",tickwidth=5,ticklen=8,zeroline=False))

fig=dict(data=edit_df,layout=layout)

iplot(fig)
Data.describe()


sns.heatmap(Data.corr(),cbar=True,annot=True)
max_Sales = Data[Data['Global_Sales']==max(Data['Global_Sales'])]

max_Sales
sns.set()

sns.regplot(Data['Global_Sales'],Data['NA_Sales'])

plt.xlabel('Global Sales')

plt.ylabel('North America Sales')

plt.title('Global Sales - NA Sales ')
sns.regplot(Data['Global_Sales'],Data['EU_Sales'])

plt.xlabel('Global Sales')

plt.ylabel('Europe Sales')

plt.title('Global Sales - EU Sales ')
sns.regplot(Data['Global_Sales'],Data['JP_Sales'])

plt.xlabel('Global Sales')

plt.ylabel('Japan Sales')

plt.title('Global Sales - JP Sales ')
sns.regplot(Data['Global_Sales'],Data['Other_Sales'])

plt.xlabel('Other countries Sales')

plt.ylabel('North America Sales')

plt.title('Global Sales - Others Sales ')
plt.figure(figsize=(15,8))

cop = Data.copy()

cop.sort_values('Global_Sales',ascending=False)

print(cop.shape)

cop1 = cop.head(1000).copy()

sns.barplot(y=cop1['Publisher'],x=cop1['Global_Sales'],orient='h')

#Some label encoding since we have some categorical DATA

obj_cols = [col for col in cop.columns if cop[col].dtype=='object']

print('Columns that will be encoded are ='+str(obj_cols))
#Quick peak into NA columns



fig = plt.figure(figsize=(15, 8))

cop.isna().sum().sort_values(ascending=True).plot(kind='barh', fontsize=20)

cop.drop('Year',axis=1)
print(cop.shape)
# Get number of unique entries in each column with categorical data

object_nunique = list(map(lambda col: cop[col].nunique(), obj_cols))

d = dict(zip(obj_cols, object_nunique))



# Print number of unique entries by column, in ascending order

sorted(d.items(), key=lambda x: x[1])
from sklearn.model_selection import train_test_split #Best approach to test the model

from sklearn.metrics import mean_absolute_error # mean absolute error , error = predictions - validation_y then abs for pos value

from sklearn.tree import DecisionTreeRegressor #model

features = ['NA_Sales','EU_Sales','JP_Sales','Other_Sales']#our features

X = cop[features]

y = cop.Global_Sales #target

train_X , val_X , train_y , val_y = train_test_split(X,y,test_size=0.25,random_state=1)

model = DecisionTreeRegressor(random_state=1)

model.fit(train_X,train_y)
predictions =model.predict(val_X)

mae = mean_absolute_error(predictions, val_y)

print('Mean absolute error '+str(mae))


df = pd.DataFrame({'Actual': val_y, 'Predicted': predictions})

df



df1 = df.head(80)

df1.plot(kind='bar',figsize=(15,8))

plt.show()
val_X['Global_Sales']=predictions

print(len(df.index))

val_X['Rank'] = df.index

val_X[['Rank','Global_Sales']].to_csv('sub_for_nothing.csv',index=False)

df.to_csv('predvsval_y.csv',index=False)